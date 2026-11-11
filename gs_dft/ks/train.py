"""The splat trainer: one verb for every Coulomb strategy.

Variational splat KS-DFT is direct minimization of ``ks(model, state)`` over
{splats, C} with plain Adam (no weight decay — this is a basis optimization,
not neural-net training; decay would bias the energy). The strategy (exact vs
DF, dense vs screened, single- vs multi-device) lives on the
:class:`~gs_dft.ks.energy.SplatKS` builder, so one loop serves them all:

    ks     = SplatKS(mol, PBE(), grid=native_grid(mol),
                     coulomb=df(), screen=pairlist(eps=1e-7))
    result = train(ks, model, steps=6000)                    # TrainResult
    E      = evaluate(ks, result.model, result.state)        # exact-Coulomb energy

- ``train`` owns the run mechanics: the occupation-frozen partition, the
  refresh cadence (pair list + frozen product aux rebuilt as the splats move,
  at a fixed pad so the jitted step compiles once — required on multi-GPU),
  checkpoint/resume (:func:`ckpt`), the collapse guard (:func:`collapse`),
  and the multi-node grid/param preparation when ``ks.mesh`` spans processes.
- The optimizer is any ``optax.GradientTransformation``; the house schedule
  (warmup → cosine, global-norm clip) ships as :func:`splat_adam`.
- ``evaluate`` is the exact-Coulomb energy of a converged model at O(M²)
  memory: streaming Coulomb for E_J (RI-K exchange for hybrids — the exact
  M⁴ exchange tensor does not fit at the M where splats win), streamed V_ne
  over the full pair list, chunked grid density.
- ``flops`` is the compile-time XLA FLOP count of one training step.
"""

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import NamedSharding, PartitionSpec as P
import optax

from dftax.energy.grid import xc_energy
from dftax.energy.xc import PBE, PBE0, B3LYP
from dftax.grid import Points, points, becke_grid as _becke_grid
from dftax.integrals.nuclear_repulsion import nuclear_repulsion

from gs_dft.ks.energy import (SplatKS, SplatModel, SplatState, refresh,
                                       _check_electrons)
from gs_dft.ks.orthonormalize import lowdin_orthonormalize, orthonormalize
from gs_dft.ks.terms import ProductDFCoulomb, ScreenedProductDFCoulomb
from gs_dft.ks.diagnostics import CollapseGuard
from gs_dft.integrals import dense as _full
from gs_dft.coulomb import stream as _cs
from gs_dft.coulomb import ri as _cdf
from gs_dft.screening import grid as _gscr
from gs_dft import screening as _scr
from gs_dft.screening import cells as _cells
from gs_dft.ks import diagnostics as _mtr
from gs_dft.ks import resolution as _res
from gs_dft.ks import shard as _shd

__all__ = ["train", "evaluate", "flops", "TrainResult",
           "splat_adam", "ckpt", "collapse", "monitor", "native_grid"]


# ---------------------------------------------------------------------------
# Run-mechanics values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ckpt:
    """Preemption-safe checkpointing (see :func:`ckpt`)."""

    base: str
    every: int = 100
    resume: bool = False


NELEC_TOL_PER_ATOM = 1e-4
"""Grid electron error per atom, |Σwρ − N_e|/N_atoms, past which a run stops once it has been inside
it. Clean states sit near 1e-5 per atom; the error is extensive, so a fixed count would not scale."""


@dataclass(frozen=True)
class Collapse:
    """Collapse-guard spec (see :func:`collapse`)."""

    nelec_tol_per_atom: float = NELEC_TOL_PER_ATOM


def ckpt(base: str, *, every: int = 100, resume: bool = False) -> Ckpt:
    """Checkpoint the trainable pytree + optimizer state (Adam moments and the
    schedule step, so the LR resumes correctly) every ``every`` steps to
    ``base``.eqx / ``base``.step. Writes are atomic (tmp + os.replace);
    ``resume=True`` restarts from an existing checkpoint (the pair list and
    the frozen aux are derived from the loaded splats, not checkpointed)."""
    return Ckpt(base=str(base), every=int(every), resume=bool(resume))


@dataclass(frozen=True)
class Monitor:
    """Observability spec for :func:`train` (see :func:`monitor`)."""

    every: int = 100
    cond_every: int = 50
    verbose: bool = True


def monitor(every: int = 100, *, cond_every: int = 50, verbose: bool = True) -> Monitor:
    """What to observe during training, and how often.

    Every ``every`` steps (final step always included) the trainer records the
    per-term energies + gradient norm into the :class:`TrainResult` history
    and merges :func:`~gs_dft.ks.diagnostics.lindep_metrics` (worst
    dense MO-Gram eigenvalue ends, hybrid RI-K; the costly cond(V) at the sparser
    ``cond_every``). ``verbose`` prints the one-line summary on rank 0.
    ``train(monitor=None)`` is silent with empty history arrays.
    """
    return Monitor(every=int(every), cond_every=int(cond_every), verbose=bool(verbose))


def collapse(*, nelec_tol_per_atom: float = NELEC_TOL_PER_ATOM) -> Collapse:
    """Stop the run on a non-finite E or |g| (checked every step), or on a grid electron error per
    atom past ``nelec_tol_per_atom`` once the run has been inside it (checked at the monitor and
    refresh points; a young run's grid error is large and is not a collapse). The guard keeps no
    state of the run: the periodic checkpoint is the one to inspect after a stop.
    :func:`train` runs with ``collapse()`` unless told otherwise."""
    return Collapse(nelec_tol_per_atom=float(nelec_tol_per_atom))


def splat_adam(lr: float = 1e-2, steps: int = 1000, *,
               warmup: int | None = None, clip: float = 1.0
               ) -> optax.GradientTransformation:
    """The house optimizer: global-norm clip → Adam on a linear-warmup +
    cosine-decay schedule (warmup = min(steps//10, 200) unless given).
    Plain Adam, no weight decay (variational basis optimization)."""
    w = min(steps // 10, 200) if warmup is None else int(warmup)
    sched = optax.join_schedules(
        [optax.linear_schedule(lr * 0.01, lr, w),
         optax.cosine_decay_schedule(lr, max(1, steps - w), alpha=0.01)], [w])
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched))


# level → (radial shells, Lebedev order). `becke_grid` already defaults to `prune='nwchem'`, so
# pruning is banked, not an available speedup. Lebedev orders are the vendored set.
_GRID_LEVELS = {0: (20, 50), 1: (35, 110), 2: (50, 194), 3: (75, 302),
                4: (90, 434), 5: (110, 590)}


def native_grid(system, level: int = 3, *, chunk: int | None = None) -> Points:
    """A native dftax Becke grid as a ``points`` spec — the splat XC quadrature.
    Accepts a dftax ``Molecule`` (anything exposing ``symbols`` + ``atom_coords``).

    ``chunk`` streams the grid density in point-chunks (O(chunk·M) memory) — pass e.g.
    4096 for screened/large runs; ``None`` keeps the dense eval. (Culling small-|w| points was
    tried and dropped: the small weights sit on the innermost shells, where ρ is largest, and a
    1e-12 relative cull moved E_xc by 2e-5 Ha on water.)"""
    if int(level) not in _GRID_LEVELS:
        raise ValueError(f"native_grid: no level {level}; have {sorted(_GRID_LEVELS)} "
                         f"(5 is the finest, so a level-5 run cannot be audited)")
    nr, lb = _GRID_LEVELS[int(level)]
    if not hasattr(system, "symbols"):
        raise TypeError("native_grid: need a dftax Molecule (with element symbols)")
    syms = list(system.symbols)
    gp, gw = _becke_grid(syms, jnp.asarray(system.atom_coords()), n_radial=nr, lebedev=lb)
    return points(gp, gw, chunk=chunk)


# ---------------------------------------------------------------------------
# Checkpoint primitives (atomic; shared by train + the CLIs)
# ---------------------------------------------------------------------------

def _save_ckpt(base, trainable, opt_state, step):
    tmp = base + ".eqx.tmp"
    eqx.tree_serialise_leaves(tmp, (trainable, opt_state))
    os.replace(tmp, base + ".eqx")
    tmp_s = base + ".step.tmp"
    with open(tmp_s, "w") as f:
        f.write(str(int(step)))
    os.replace(tmp_s, base + ".step")


def _load_ckpt(base, trainable, opt_state):
    """Return (trainable, opt_state, step) if a checkpoint exists, else None."""
    if not os.path.exists(base + ".eqx") or not os.path.exists(base + ".step"):
        return None
    trainable, opt_state = eqx.tree_deserialise_leaves(
        base + ".eqx", (trainable, opt_state))
    with open(base + ".step") as f:
        step = int(f.read().strip())
    return trainable, opt_state, step


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Outcome of a :func:`train` run.

    ``step``/``energy``/``e_hartree``/``grad_norm`` are the monitor-cadence
    history as arrays (empty when ``monitor=None``). ``nelec_step``/``nelec_err``
    are the grid electron error per atom at every monitor and refresh point and
    the last step. ``state`` is the last refresh state — pass it straight to
    :func:`evaluate` / :func:`~gs_dft.ks.forces.splat_forces`. ``n_cap_*`` count the
    splats pressed against the grid's sharpness cap, the width floor and the
    anisotropy limit after the last step (:mod:`gs_dft.ks.resolution`).
    """

    model: SplatModel
    state: SplatState
    n_steps: int
    collapsed: bool
    step: np.ndarray
    energy: np.ndarray
    e_hartree: np.ndarray
    grad_norm: np.ndarray
    stop_reason: str | None = None          # None ran to the end; "nonfinite" | "nelec" stopped
    nelec_step: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    nelec_err: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (Σwρ − N_e)/N_atoms
    n_cap_sharp: int = 0
    n_cap_wide: int = 0
    n_cap_aniso: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_df(ks: SplatKS) -> bool:
    return isinstance(ks.coulomb, (ProductDFCoulomb, ScreenedProductDFCoulomb))


def _prepare_mesh(ks, model):
    """Multi-device run prep for the run inputs only: the grid/nuclear arrays
    were laid out for the mesh at build time (see SplatKS.__init__); what
    remains run-scoped is promoting the model to replicated globals on
    multi-node.
    Returns (ks, model, G, multinode, rank0), with ``G = ks.mesh.size`` — the
    mesh on ``ks``, not the host's device count (sub-meshes are representable).
    """
    if ks.mesh is None:
        return ks, model, 1, False, True
    multinode = jax.process_count() > 1
    G = int(ks.mesh.size)
    if multinode:
        model = _shd.tree_to_global(model, ks.mesh)
    else:
        # Every input committed as the compiled step reads it, so no call transfers or reshards.
        model = _replicated(model, ks.mesh)
        ks = _replicated(ks, ks.mesh)
        split = NamedSharding(ks.mesh, P("g"))
        ks = eqx.tree_at(lambda k: (k.grid_points, k.grid_weights), ks,
                         (jax.device_put(ks.grid_points, split), jax.device_put(ks.grid_weights, split)))
    return ks, model, G, multinode, jax.process_index() == 0


def _placer(ks, multinode):
    """Puts a fresh pytree (an optimizer state, a loaded checkpoint) where the step expects it."""
    if ks.mesh is None:
        return lambda t: t
    if multinode:
        return lambda t: jax.tree.map(
            lambda x: _shd.to_global(x, ks.mesh) if isinstance(x, jax.Array) and x.is_fully_addressable
            else x, t)
    return lambda t: _replicated(t, ks.mesh)


def _pair_fill(state, pad_to):
    """``(n_live, fill)`` — significant pairs now, against the pad frozen at step 0.

    The pad is a trajectory-length assumption, so its fill is worth logging: pair count grows
    over training (measured 1.10x on water to 1.85x on the alanine dipeptide over 12k steps), and
    a truncation only announces itself after pairs have been dropped. Free — ``valid`` already
    marks the real entries.
    """
    if state.pairs is None or not pad_to:
        return None, None
    n = int(jnp.sum(state.pairs[2] != 0))          # bool mask or unique-pair weights
    return n, n / float(pad_to)


def _replicated(tree, mesh):
    """Put every array leaf of `tree` on `mesh`, replicated."""
    spec = NamedSharding(mesh, P())
    arr, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.tree.map(lambda x: jax.device_put(x, spec), arr), static)


def _placed(state, mesh):
    """A refresh state committed as the step reads it: the pair list, the cell lists (leading device
    axis) and the metric factor's rows split over "g", so each device holds its own share and no
    more; the aux replicated."""
    split = NamedSharding(mesh, P("g"))
    put = lambda xs: None if xs is None else tuple(jax.device_put(x, split) for x in xs)
    return SplatState(
        pairs=put(state.pairs),
        aux=None if state.aux is None else _replicated(state.aux, mesh),
        aux_K=None if state.aux_K is None else _replicated(state.aux_K, mesh),
        cholV=None if state.cholV is None else jax.device_put(state.cholV, split),
        cells=None if state.cells is None else tuple(put(g) for g in state.cells))


def _on_one_device(tree):
    """Pin every array leaf of `tree` to this process's first LOCAL device.

    `jax.devices()` is the GLOBAL list under multi-controller, so device 0 belongs to one process
    only and every other process would be pinning across the network.
    """
    dev = jax.local_devices()[0]
    arr, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.tree.map(lambda x: jax.device_put(x, dev), arr), static)


def _local_copy(tree):
    """This process's copy of a replicated global pytree."""
    return jax.tree.map(lambda x: x.addressable_data(0) if isinstance(x, jax.Array) else x, tree)


def _refresh_state(ks, basis, pad_to, multinode, like=None):
    """One refresh: pairs (fixed pad; an overflow raises) + frozen product aux + cell lists at
    ``like``'s widths (an overflow raises); on a mesh the pair list, the cell lists and the metric
    factor's rows are split over "g" (``_placed``), multi-node as global arrays.

    On a mesh the rebuild runs on ONE device and the products are then placed. Running it with
    mesh-resident inputs executed the same work redundantly on
    every device with each dispatch gated by the slowest: measured 4.2x on 4 GPUs (PJM49,
    0.18 -> 0.77 s) and reproduced at 4.6x on a 4-device CPU mesh. Pinning the inputs also
    fixes the products' sharding, which is what stops the jitted step recompiling: uncommitted
    at step 0 and mesh-resident from step 1, the state arrived with two different shardings and
    compile-once silently became compile-twice.
    """
    # Single-process mesh only: rebuild on one device and broadcast. Multi-controller takes the
    # `to_global` path below instead -- these are different promotions and an early return here
    # would make that branch unreachable.
    if ks.mesh is not None and not multinode:
        return _placed(refresh(ks, _on_one_device(basis), pad_to=pad_to, like=like), ks.mesh)
    # Multi-node, each process rebuilds from its own copy of the replicated basis, then promotes.
    state = refresh(ks, _local_copy(basis) if multinode else basis, pad_to=pad_to, like=like)
    if multinode:
        split = lambda x: _shd.to_global(x, ks.mesh, sharded=True)      # as _placed lays them out
        state = SplatState(
            pairs=None if state.pairs is None else tuple(split(x) for x in state.pairs),
            aux=None if state.aux is None else _shd.tree_to_global(state.aux, ks.mesh),
            aux_K=None if state.aux_K is None else _shd.tree_to_global(state.aux_K, ks.mesh),
            cholV=state.cholV,                  # already global: ri.metric_factor builds it on the mesh
            cells=None if state.cells is None else tuple(
                tuple(split(x) for x in g) for g in state.cells),
        )
    return state


def _partition(model):
    """Trainable = everything except the occupations (data, not parameters)."""
    filt = eqx.tree_at(lambda m: m.occupations, jax.tree.map(lambda _: True, model),
                       replace=False)
    return eqx.partition(model, filt)


def _pad_of(ks, basis, G):
    """The pair pad frozen for the run: ``ks.screen.pad`` × the significant count now, a multiple of
    the mesh size. ``(pad, n0)``; ``(None, None)`` when ``ks`` does not screen."""
    if ks.screen is None:
        return None, None
    n0 = _scr.count_pairs(basis, eps=ks.screen.eps, unique=ks.screen.unique)
    return -(-int(ks.screen.pad * n0) // G) * G, n0


def _capped(m_tr, res, atoms):
    return eqx.tree_at(lambda m: m.basis, m_tr, _res.capped(m_tr.basis, res, atoms))


def zero_sharded(inner: optax.GradientTransformation, mesh) -> optax.GradientTransformation:
    """``inner`` with its state split over ``mesh``'s axis "g" (ZeRO-1): every leaf is raveled and padded
    to whole device shares, so a device keeps 1/G of the moments and updates only its share; the
    updates come back whole, in the leaves' shapes. For elementwise transformations (Adam, clipping by
    the global norm, whose norm the compiler sums over the devices): the same arithmetic per element,
    and the pad's zeros stay zero."""
    G = int(mesh.size)
    split, whole = NamedSharding(mesh, P("g")), NamedSharding(mesh, P())

    def flat(t):
        return jax.tree.map(lambda x: jax.lax.with_sharding_constraint(
            jnp.pad(jnp.ravel(x), (0, (-x.size) % G)), split), t)

    def init(params):
        return inner.init(flat(params))

    def update(updates, state, params=None):
        u, state = inner.update(flat(updates), state, None if params is None else flat(params))
        u = jax.tree.map(lambda v, x: jax.lax.with_sharding_constraint(v[:x.size].reshape(x.shape), whole),
                         u, updates)
        return u, state

    return optax.GradientTransformation(init, update)


def _opt_initializer(opt, mesh, m_tr):
    """``opt.init`` compiled once, with the state where the step expects it: the moments split over
    "g" (:func:`zero_sharded`) and every scalar replicated."""
    split, whole = NamedSharding(mesh, P("g")), NamedSharding(mesh, P())
    shardings = jax.tree.map(lambda a: split if a.ndim else whole, jax.eval_shape(opt.init, m_tr))
    return jax.jit(opt.init, out_shardings=shardings)


def _step_fn(opt, m_st, remat):
    """One optimizer step, the splats then projected onto the grid's resolution cap and the width
    limits (the optimizer state untouched). ``ks``, the state and the resolution ``res`` are
    arguments (``fixed``): a multi-host grid cannot be captured by a jit, and a captured state would be
    compiled in as constants."""
    def step_fn(fixed, m_tr, opt_state):
        ks_, state, res = fixed
        def E(tr):
            return ks_(eqx.combine(tr, m_st), state)
        E_ = jax.checkpoint(E) if remat else E
        (e, eaux), g = jax.value_and_grad(E_, has_aux=True)(m_tr)
        gnorm = optax.global_norm(g)
        u, opt_state = opt.update(g, opt_state, m_tr)
        m_tr = _capped(eqx.apply_updates(m_tr, u), res, ks_.atom_coords)
        # Not a no-op on a mesh. Replicated params are recomputed per device through regeigh, an
        # eigendecomposition of a near-degenerate overlap, so a last-bit difference between copies
        # amplifies into divergent updates. Must stay unconditional.
        if ks_.mesh is not None:
            m_tr = _shd.sync_replicas(m_tr, ks_.mesh)
        return m_tr, opt_state, e, eaux, gnorm, _res.cap_counts(m_tr.basis, res, ks_.atom_coords)
    return eqx.filter_jit(step_fn)


def _compiled_step(step_fn, *args):
    """``step_fn`` compiled ahead of time for ``args``. A call with another shape, dtype or sharding
    raises instead of recompiling (a mid-run compile deadlocks a mesh)."""
    compiled = step_fn.lower(*args).compile()
    if os.environ.get("DFTAX_AOT_ANALYZE"):
        # compile-only probe: names the temps behind a step OOM without allocating them
        mem = compiled.compiled.memory_analysis()
        print(f"AOT temp={mem.temp_size_in_bytes/2**30:.2f}GiB "
              f"args={mem.argument_size_in_bytes/2**30:.2f}GiB "
              f"out={mem.output_size_in_bytes/2**30:.2f}GiB", flush=True)
        hlo = os.environ["DFTAX_AOT_ANALYZE"]
        if hlo != "1":   # a path => dump the HLO so the big temps can be named by shape
            txt = compiled.compiled.as_text()
            with open(hlo, "w") as fh:
                fh.write(txt)
            print(f"AOT hlo -> {hlo} ({len(txt)/2**20:.0f} MiB)", flush=True)
        raise SystemExit(0)
    return compiled


# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------

def train(ks: SplatKS, model: SplatModel, *,
          steps: int = 1000,
          optimizer: optax.GradientTransformation | None = None,
          refresh_every: int = 50,
          reset_opt_on_refresh: bool = False,
          monitor: "Monitor | None" = Monitor(),
          remat: bool = False,
          checkpoint: Ckpt | None = None,
          guard: Collapse | None = Collapse(),
          snapshot_every: int = 0, snapshot_cb: "Callable | None" = None,
          step_cb: "Callable | None" = None) -> TrainResult:
    """Variationally minimize ``ks`` over {splats, C} (occupations frozen).

    The step is compiled once, ahead of time, before step 0: a shape, dtype or sharding change
    raises instead of recompiling.

    Args:
        ks: the built energy functional — its ``coulomb``/``screen``/``mesh``
            values select the execution strategy; ``train`` adds no strategy
            of its own.
        steps: optimizer step budget.
        optimizer: any ``optax.GradientTransformation``; default
            ``splat_adam(1e-2, steps)``.
        reset_opt_on_refresh: drop the optimizer state at each refresh, so Adam restarts on the
            rebuilt functional. Costs the warmed-up per-coordinate scaling.
        refresh_every: host-rebuild cadence for the pair list / frozen aux as
            the splats move (0 = never). The pair pad is fixed for the whole
            run at ``ks.screen.pad ×`` the initial significant count (rounded
            to the device count); a refresh that overflows it raises
            :class:`~gs_dft.screening.PairPadOverflow`.
        monitor: a :func:`monitor` spec — what to observe and how often
            (``None`` = silent, empty history arrays).
        remat: ``jax.checkpoint`` the loss (lower peak memory, one extra
            forward per step) — only when a run would otherwise OOM.
        checkpoint: a :func:`ckpt` value (atomic save/resume).
        guard: a :func:`collapse` value (on by default; ``None`` turns it off).
        snapshot_every: cadence of ``snapshot_cb`` (0 = never).
        snapshot_cb: ``snapshot_cb(step, model, aux, E)`` at that cadence
            (frame 0 included on fresh runs).
        step_cb: ``step_cb(metrics_dict)`` every step (forces a metrics build).

    Returns:
        :class:`TrainResult` — final model + last refresh state + history.
    """
    _check_electrons(ks, model)
    ks, model, G, multinode, rank0 = _prepare_mesh(ks, model)
    place = _placer(ks, multinode)
    n_e = float(jnp.sum(model.occupations))
    n_atoms = int(ks.atom_coords.shape[0])

    # --- the grid's resolution cap and the width limits, on the splats from the start ---
    say = rank0 and monitor is not None and monitor.verbose
    res = place(_res.grid_resolution(ks, verbose=say))
    cap = eqx.filter_jit(_capped)
    model = cap(model, res, ks.atom_coords)
    caps = np.asarray(_res.cap_counts(model.basis, res, ks.atom_coords)).tolist()
    if say:
        print(f"  cap at start: {caps[0]} sharp / {caps[1]} wide / {caps[2]} anisotropic splats"
              + ("" if res is not None else " (no shells in the grid: width limits only)"), flush=True)

    m_tr, m_st = _partition(model)
    opt = splat_adam(1e-2, steps) if optimizer is None else optimizer
    # On one process's mesh the optimizer state is split over the devices (each keeps 1/G of the
    # moments); multi-node keeps it replicated, since rank 0 alone writes the checkpoint.
    if ks.mesh is not None and not multinode:
        opt = zero_sharded(opt, ks.mesh)
        fresh_opt = _opt_initializer(opt, ks.mesh, m_tr)
        put_opt = lambda t, like: jax.device_put(t, jax.tree.map(lambda x: x.sharding, like))
    else:
        fresh_opt = lambda t: place(opt.init(t))
        put_opt = lambda t, like: place(t)
    opt_state = fresh_opt(m_tr)

    # --- resume: before the lists are sized, so a restart after an overflow sizes them anew ---
    start = 0
    if checkpoint is not None and checkpoint.resume:
        loaded = _load_ckpt(checkpoint.base, m_tr, opt_state)
        if loaded is not None:
            m_tr, loaded_opt, start = loaded
            m_tr, opt_state = place(cap(m_tr, res, ks.atom_coords)), put_opt(loaded_opt, opt_state)
            if rank0:
                print(f"  resumed from {checkpoint.base} at step {start}/{steps}", flush=True)
    basis0 = eqx.combine(m_tr, m_st).basis

    # --- refresh products: fixed pair pad (compile-once), frozen product aux ---
    pad_to, n0 = _pad_of(ks, _local_copy(basis0), G)             # counted on one device

    def rebuild(basis, at, like=None):
        # `like`: the cell lists keep the first build's widths (compile once).
        try:
            return _refresh_state(ks, basis, pad_to, multinode, like=like)
        except _scr.PairPadOverflow as exc:
            raise _scr.PairPadOverflow(
                f"step {at}: {exc} (pad = {ks.screen.pad:g} x the {n0} pairs of step {start})") from exc
        except _cells.CellOverflow as exc:
            raise _cells.CellOverflow(f"step {at}: {exc}") from exc

    state = rebuild(basis0, start)
    _n_pairs, _pair_fill_frac = (_pair_fill(state, pad_to) if not multinode else (None, None))

    step = _compiled_step(_step_fn(opt, m_st, remat), (ks, state, res), m_tr, opt_state)

    if snapshot_every and snapshot_cb is not None and start == 0:   # frame 0
        snapshot_cb(0, eqx.combine(m_tr, m_st), state.aux, float("nan"))

    cguard = (CollapseGuard(nelec_ref=n_e, nelec_tol=guard.nelec_tol_per_atom * n_atoms)
              if guard is not None else None)

    hist_step, hist_e, hist_ej, hist_g = [], [], [], []
    nel_step, nel_err = [], []
    stop_reason = None
    i = start
    for i in range(start, steps):
        refreshed = bool(refresh_every and i and i % refresh_every == 0)
        if refreshed:                                            # splats moved
            try:
                state = rebuild(eqx.combine(m_tr, m_st).basis, i, like=state)
            except (_scr.PairPadOverflow, _cells.CellOverflow):
                # The point is valid, only its lists outgrew the run's: save it, so a restart
                # resumes here with lists sized to this cloud.
                if rank0 and checkpoint is not None and checkpoint.every:
                    _save_ckpt(checkpoint.base, m_tr, opt_state, i)
                raise
            if not multinode:
                _n_pairs, _pair_fill_frac = _pair_fill(state, pad_to)
            if reset_opt_on_refresh:
                # A refresh changes the energy functional discontinuously, so Adam's moments hold
                # curvature from the old one. Dropping them costs the warmed-up per-coordinate
                # scaling, so expect a transient after every refresh.
                opt_state = fresh_opt(eqx.filter(m_tr, eqx.is_inexact_array))
        m_tr, opt_state, E, eaux, gnorm, n_cap = step((ks, state, res), m_tr, opt_state)

        do_monitor = bool(monitor is not None and monitor.every
                          and (i % monitor.every == 0 or i == steps - 1))
        # The grid electron error per atom, at the monitor and refresh points (and the ends).
        nel = (float(eaux.nelec) if (do_monitor or refreshed or i in (start, steps - 1))
               else None)
        if nel is not None:
            nel_step.append(i)
            nel_err.append((nel - n_e) / n_atoms)
        if cguard is not None and cguard.check(i, E, gnorm, nelec=nel):
            stop_reason = cguard.reason
            if rank0:
                why = ("" if stop_reason != "nelec" else
                       f" grid electron error {nel_err[-1]:+.2e}/atom past "
                       f"{guard.nelec_tol_per_atom:g}/atom")
                print(f"  step {i:4d}  *** STOP ({stop_reason}) *** E={float(E):.4e} "
                      f"|g|={float(gnorm):.2e}{why}", flush=True)
            break
        if rank0 and checkpoint is not None and checkpoint.every and (
                (i + 1) % checkpoint.every == 0 or i == steps - 1):
            _save_ckpt(checkpoint.base, m_tr, opt_state, i + 1)
        if snapshot_every and snapshot_cb is not None and (
                (i + 1) % snapshot_every == 0 or i == steps - 1):
            snapshot_cb(i + 1, eqx.combine(m_tr, m_st), state.aux, float(E))

        if step_cb is None and not do_monitor:
            continue
        m_ = {"step": i, "E_total": float(E), "grad_norm": float(gnorm),
              "E_kinetic": float(eaux.kinetic), "E_hartree": float(eaux.hartree),
              "E_xc": float(eaux.xc), "E_external": float(eaux.external),
              "nelec": float(eaux.nelec)}
        m_["nelec_err_per_atom"] = (m_["nelec"] - n_e) / n_atoms
        caps = np.asarray(n_cap).tolist()
        m_["n_cap_sharp"], m_["n_cap_wide"], m_["n_cap_aniso"] = caps
        if _n_pairs is not None:
            m_["n_pairs"] = _n_pairs                  # significant pairs at the last refresh
            m_["pair_fill"] = _pair_fill_frac         # ... as a fraction of the frozen pad
        if do_monitor and not multinode:
            m_.update(_mtr.lindep_metrics(
                ks, eqx.combine(m_tr, m_st), state,
                include_cond=bool(monitor.cond_every)
                and (i % monitor.cond_every == 0 or i == steps - 1)))
        if step_cb is not None:
            step_cb(m_)
        if do_monitor:
            hist_step.append(i); hist_e.append(m_["E_total"])
            hist_ej.append(m_["E_hartree"]); hist_g.append(m_["grad_norm"])
            if monitor.verbose and rank0:
                lo, cv = m_.get("dense_min_eig"), m_.get("cond_V")
                # Both ends: the Löwdin floor is relative, so the ratio is what crosses it.
                extra = (f"  E_J={m_['E_hartree']:.4f}"
                         f"  λmin={lo:.2e}"
                         f"  λmax={m_.get('dense_max_eig', float('nan')):.2e}"
                         f"  λmin/λmax={m_.get('dense_eig_ratio', float('nan')):.2e}"
                         if lo is not None else "")
                extra += f"  cond(V)={cv:.1e}" if cv is not None else ""
                print(f"  step {i:4d}  E={float(E):.6f}  |g|={float(gnorm):.2e}"
                      f"  dN/atom={m_['nelec_err_per_atom']:+.1e}"
                      f"  cap s/w/a={caps[0]}/{caps[1]}/{caps[2]}{extra}", flush=True)

    if steps > start:
        caps = np.asarray(n_cap).tolist()
    return TrainResult(
        model=eqx.combine(m_tr, m_st), state=state,
        n_steps=(i + 1 if steps > start else start), collapsed=stop_reason is not None,
        step=np.asarray(hist_step, dtype=int), energy=np.asarray(hist_e),
        e_hartree=np.asarray(hist_ej), grad_norm=np.asarray(hist_g),
        stop_reason=stop_reason, nelec_step=np.asarray(nel_step, dtype=int),
        nelec_err=np.asarray(nel_err),
        n_cap_sharp=caps[0], n_cap_wide=caps[1], n_cap_aniso=caps[2],
    )


# Overlap cutoff of the pair list the reported energy is evaluated on (E_J and V_ne):
# ≤ 1 µHa from the unscreened list (experiments/kernels/check_evaluate.py).
_EVAL_EPS = 1e-12


@eqx.filter_jit
def _evaluate_energy(ks, model, pairs, aux_K):
    """The reported energy, compiled as one function; every array is an argument (a closure
    would compile it in as a constant)."""
    coords, charges = ks.atom_coords, ks.atom_charges
    occ = model.occupations
    pi, pj, w = pairs
    # S and T are never formed — see `SplatKS._finish`; the contractions carry (M, n_occ).
    C = orthonormalize(model.C, model.C.T @ _full.overlap_times(model.basis, model.C))
    E_kin = occ @ jnp.sum(C * _full.kinetic_times(model.basis, C), axis=0)
    E_ext = _scr.screened_external_energy(model.basis, C, occ, pi, pj, coords, charges, w)
    # Second implementation of the energy tail: every term added to `SplatKS._finish` must be
    # added here too, or the audit silently disagrees with the objective it audits.
    kind = ks.xc.xc_type
    if kind == "MGGA":
        rho, grad, tau = _gscr.chunked_density_grad_tau(model.basis, C, occ, ks.grid_points)
    else:
        rho, grad = _gscr.chunked_density_and_grad(model.basis, C, occ, ks.grid_points)
        tau = None
        if kind != "GGA":
            grad = None
    E_J = _cs.screened_coulomb_energy(model.basis, C, occ, pi, pj, w)    # exact ERI, unique pairs
    a_x = ks.coulomb.hf_coeff
    E_K = (_cdf.df_exchange_energy(model.basis, C, aux_K)
           if a_x > 0.0 else 0.0)                            # RI-K (exact-K infeasible)
    a_lr = getattr(ks.coulomb, "hf_coeff_lr", 0.0)
    E_K_lr = (_cdf.df_exchange_energy(model.basis, C, aux_K,
                                      omega=getattr(ks.coulomb, "omega", 0.0))
              if a_lr != 0.0 else 0.0)
    rho = jnp.maximum(rho, 1e-30)
    E_xc = xc_energy(ks.xc, rho, ks.grid_weights, grad_rho=grad, tau=tau)
    nlc_b = float(getattr(ks.xc, "nlc_b", 0.0) or 0.0)
    if nlc_b:
        from gs_dft.ks.nlc import vv10_energy
        E_xc = E_xc + vv10_energy(rho, jnp.sum(grad ** 2, axis=-1), ks.grid_points,
                                  ks.grid_weights, b=nlc_b,
                                  c=float(getattr(ks.xc, "nlc_c", 0.0) or 0.0))
    E_nn = nuclear_repulsion(coords, charges)
    return E_kin + E_ext + E_J + a_x * E_K + a_lr * E_K_lr + E_xc + E_nn


def evaluate(ks: SplatKS, model: SplatModel, state: SplatState | None = None) -> float:
    """Exact-Coulomb energy of a converged model, streamed at O(M²) memory.

    E_J is the exact ERI over the unique pairs with |Ŝ_ij| > 1e-12 (fused pair-pair kernel), V_ne
    streams over the same list; hybrids use RI-K exchange with the converged K-aux (the exact M⁴
    exchange tensor does not fit at the sizes where splats win, and RI-K is the method's own
    ~1 mHa exchange). The grid density streams in chunks, so no dense (M, M, 3, 3) or (G, M)
    buffer ever forms. One compiled function.

    Args:
        ks: the built energy functional; its grid is the quadrature used. To
            audit on a finer grid, build a second ``SplatKS`` with that grid
            (construction is cheap) and call ``evaluate`` on it.
        model: the converged model.
        state: the training run's ``res.state`` — required for hybrids, which
            need ``state.aux_K``.

    Returns:
        The total energy, as a float.
    """
    _check_electrons(ks, model)
    a_x = ks.coulomb.hf_coeff
    if a_x != 0.0 and (state is None or state.aux_K is None):
        raise ValueError("evaluate: a hybrid ks needs state.aux_K (the converged K-aux) — "
                         "pass the training run's `res.state`, or refresh(ks, model.basis).")
    pairs = _scr.neighbor_pairs(model.basis, eps=_EVAL_EPS, unique=True)
    aux_K = None if state is None else state.aux_K
    return float(_evaluate_energy(ks, model, pairs, aux_K))


def flops(ks: SplatKS, model: SplatModel, *,
          optimizer: optax.GradientTransformation | None = None,
          steps: int = 1000) -> float:
    """Compile-time XLA FLOP estimate of ONE training step (no execution).

    Lowers the same step as :func:`train` (with a fresh refresh state) and reads the lowered
    module's cost analysis. ``steps`` only shapes the default optimizer's schedule.
    """
    G = 1 if ks.mesh is None else int(ks.mesh.size)
    pad_to, _ = _pad_of(ks, model.basis, G)
    state = refresh(ks, model.basis, pad_to=pad_to)
    m_tr, m_st = _partition(model)
    opt = splat_adam(1e-2, steps) if optimizer is None else optimizer
    lowered = _step_fn(opt, m_st, False).lower((ks, state, _res.grid_resolution(ks)), m_tr,
                                               opt.init(m_tr))
    low = getattr(lowered, "lowered", lowered)        # unwrap the equinox Lowered
    ca = low.cost_analysis()
    if ca is None:                                    # newer jax: analysis needs the compile
        ca = low.compile().cost_analysis()
    if isinstance(ca, (list, tuple)):                 # some versions: one dict per device
        ca = ca[0]
    return float(ca["flops"])


# ---------------------------------------------------------------------------
# CLI: DF training vs an exact-Coulomb run from the same init
# ---------------------------------------------------------------------------

_XC = {"pbe": PBE, "pbe0": PBE0, "b3lyp": B3LYP}


def main():
    import argparse
    import time
    jax.config.update("jax_enable_x64", True)
    # Persistent XLA compilation cache: DRAC/SLURM requeues re-run identical shapes,
    # so cache hits erase the ~30 s-per-function recompiles. Env var wins if set.
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update("jax_compilation_cache_dir",
                          os.path.expanduser("~/.cache/gs_dft/xla"))
    import jax.random as jr
    from gs_dft.benchmark import build_reference
    from gs_dft import init_model
    from gs_dft.ks.terms import df as df_spec, pairlist, hf_coeff

    ap = argparse.ArgumentParser(
        description="Frozen-product-aux DF training, vs an exact-Coulomb run from the same init")
    ap.add_argument("--molecule", default="co2")
    ap.add_argument("--xc", default="pbe", choices=sorted(_XC))
    ap.add_argument("--M", type=int, default=84)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--aux_scales", default="1.0",
                    help="comma tempered ladder of exponent multipliers for the product aux")
    ap.add_argument("--skip_exact", action="store_true")
    ap.add_argument("--exact_ref", type=float, default=None)
    ap.add_argument("--monitor_every", type=int, default=100)
    ap.add_argument("--screened", action="store_true",
                    help="screened pairs + chunked XC; the pair list refreshes periodically")
    ap.add_argument("--refresh", type=int, default=50)
    ap.add_argument("--pair_eps", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid_check", type=int, default=0)
    args = ap.parse_args()

    mol, e_ref, nao, _ = build_reference(args.molecule, "cc-pvdz", args.xc, 3)
    xc_obj = _XC[args.xc]()
    a_x = hf_coeff(xc_obj)
    print(f"{args.molecule}/{args.xc} (a_x={a_x})  E_ref={e_ref:.6f}  nao={nao}  "
          f"M={args.M}  steps={args.steps}\n", flush=True)
    scales = tuple(float(s) for s in args.aux_scales.split(","))
    grid = native_grid(mol, 3, chunk=4096 if args.screened else None)
    ks = SplatKS(mol, xc_obj, grid=grid,
                 coulomb=df_spec(scales=scales),
                 screen=pairlist(eps=args.pair_eps) if args.screened else None)

    print(f"=== DF (frozen product aux) training [{args.xc}] ===")
    t0 = time.time()
    model = init_model(mol, args.M, jr.PRNGKey(args.seed))
    res = train(ks, model, steps=args.steps, refresh_every=args.refresh,
                monitor=monitor(args.monitor_every))
    e_df_true = evaluate(ks, res.model, res.state)
    t_df = time.time() - t0
    grid_note = ""
    if args.grid_check:
        # honest-grid audit: every term is analytic except the E_xc quadrature; a floating
        # basis can in principle exploit the training grid, so re-evaluate on a finer one
        ks_fine = SplatKS(mol, xc_obj,
                          grid=native_grid(mol, args.grid_check),
                          coulomb=df_spec(scales=scales))
        e_ck = evaluate(ks_fine, res.model, res.state)
        grid_note = (f"  E_grid{args.grid_check}={e_ck:.6f}  "
                     f"grid_bias_mha={(e_df_true - e_ck) * 1e3:+.2f}")

    if args.skip_exact or args.xc == "b3lyp":
        if args.xc == "b3lyp" and not args.skip_exact:
            print("  (exact B3LYP run skipped — the dense-ERI exchange is pbe/pbe0-sized only)")
        e_ex, t_ex = args.exact_ref, 0.0
    else:
        print(f"\n=== exact-Coulomb training [{args.xc}] (same init/seed/steps) ===")
        t0 = time.time()
        ks_ex = SplatKS(mol, xc_obj,
                        grid=native_grid(mol, 3))
        res_ex = train(ks_ex, init_model(mol, args.M, jr.PRNGKey(args.seed)),
                       steps=args.steps, monitor=None)
        e_ex = float(ks_ex(res_ex.model)[0])
        t_ex = time.time() - t0

    CHEM_ACC_MHA = 1.594
    gap_mha = (e_df_true - e_ref) * 1e3
    n_aux = int(res.state.aux.n_basis)
    print(f"\n=== RESULT (M={args.M}, {args.steps} steps) ===")
    if e_ex is not None:
        print(f"  exact-Coulomb run:           E={e_ex:.6f}  "
              f"gap_vs_ref={(e_ex - e_ref) * 1e3:+.2f} mHa  ({t_ex:.0f}s)")
        print(f"  DF-vs-exact converged energy diff: {(e_df_true - e_ex) * 1e3:+.3f} mHa  "
              f"(target |.|<1)")
    try:
        peak_gpu_mb = jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        peak_gpu_mb = -1.0
    # benchmark-format line (parsed by the exp7/8/9 launchers + READMEs)
    print(f"RESULT {args.molecule}/{args.xc}/splat DF: E_splat={e_df_true:.6f}  "
          f"E_ref={e_ref:.6f}  gap={gap_mha:+.3f} mHa  beats_ref={e_df_true <= e_ref}  "
          f"chem_acc={abs(gap_mha) <= CHEM_ACC_MHA}  M={args.M}  N_aux={n_aux}  "
          f"steps={args.steps}  seed={args.seed}  time={t_df:.0f}s  "
          f"peak_gpu_mb={peak_gpu_mb:.0f}{grid_note}", flush=True)


if __name__ == "__main__":
    main()
