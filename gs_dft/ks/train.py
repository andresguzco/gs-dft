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
from dataclasses import dataclass
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
from gs_dft.ks import diagnostics as _mtr
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


@dataclass(frozen=True)
class Collapse:
    """Collapse-guard spec (see :func:`collapse`)."""

    base: str
    nelec_rtol: float = 1e-3


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


def collapse(base: str, *, nelec_rtol: float = 1e-3) -> Collapse:
    """On collapse — a non-finite E/|g|, or an electron count that regresses by more than
    ``nelec_rtol`` (relative) after having been right — dump the last-good ("before") and
    first-broken ("after") bundles to ``base``_{before,after} and halt the run."""
    return Collapse(base=str(base), nelec_rtol=float(nelec_rtol))


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
    history as arrays (empty when ``monitor=None``). ``state`` is the last
    refresh state — pass it straight to :func:`evaluate` /
    :func:`~gs_dft.ks.forces.splat_forces`.
    """

    model: SplatModel
    state: SplatState
    n_steps: int
    collapsed: bool
    step: np.ndarray
    energy: np.ndarray
    e_hartree: np.ndarray
    grad_norm: np.ndarray


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
        # Commit the model to the mesh before the first step. Left uncommitted it lives on device 0
        # until step_fn's first output, so step_fn saw two input shardings and compiled twice.
        model = _replicated(model, ks.mesh)
    return ks, model, G, multinode, jax.process_index() == 0


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


def _on_one_device(tree):
    """Pin every array leaf of `tree` to this process's first LOCAL device.

    `jax.devices()` is the GLOBAL list under multi-controller, so device 0 belongs to one process
    only and every other process would be pinning across the network.
    """
    dev = jax.local_devices()[0]
    arr, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.tree.map(lambda x: jax.device_put(x, dev), arr), static)


def _refresh_state(ks, basis, pad_to, multinode):
    """One refresh: pairs (fixed pad, overflow-capped) + frozen product aux;
    multi-node, the products are promoted to global arrays (the pair list is
    replicated — the shard_map reshards it to "g").

    On a mesh the rebuild runs on ONE device and the products are broadcast. Its outputs are
    replicated, so running it with mesh-resident inputs executed the same work redundantly on
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
        return _replicated(refresh(ks, _on_one_device(basis), pad_to=pad_to), ks.mesh)
    state = refresh(ks, basis, pad_to=pad_to)
    if multinode:
        state = SplatState(
            pairs=None if state.pairs is None else tuple(
                _shd.to_global(x, ks.mesh) for x in state.pairs),
            aux=None if state.aux is None else _shd.tree_to_global(state.aux, ks.mesh),
            aux_K=None if state.aux_K is None else _shd.tree_to_global(state.aux_K, ks.mesh),
            cholV=None if state.cholV is None else _shd.to_global(state.cholV, ks.mesh),
        )
    return state


def _partition(model):
    """Trainable = everything except the occupations (data, not parameters)."""
    filt = eqx.tree_at(lambda m: m.occupations, jax.tree.map(lambda _: True, model),
                       replace=False)
    return eqx.partition(model, filt)


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
          guard: Collapse | None = None,
          snapshot_every: int = 0, snapshot_cb: "Callable | None" = None,
          step_cb: "Callable | None" = None) -> TrainResult:
    """Variationally minimize ``ks`` over {splats, C} (occupations frozen).

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
            to the device count); overflow keeps the largest-overlap pairs, so
            the jitted step compiles once (the constant-shape contract; see
            ``gs_dft.screening``).
        monitor: a :func:`monitor` spec — what to observe and how often
            (``None`` = silent, empty history arrays).
        remat: ``jax.checkpoint`` the loss (lower peak memory, one extra
            forward per step) — only when a run would otherwise OOM.
        checkpoint: a :func:`ckpt` value (atomic save/resume).
        guard: a :func:`collapse` value (dump + halt on blow-up).
        snapshot_every: cadence of ``snapshot_cb`` (0 = never).
        snapshot_cb: ``snapshot_cb(step, model, aux, E)`` at that cadence
            (frame 0 included on fresh runs).
        step_cb: ``step_cb(metrics_dict)`` every step (forces a metrics build).

    Returns:
        :class:`TrainResult` — final model + last refresh state + history.
    """
    _check_electrons(ks, model)
    ks, model, G, multinode, rank0 = _prepare_mesh(ks, model)

    # --- refresh products: fixed pair pad (compile-once), frozen product aux ---
    pad_to = None
    if ks.screen is not None:
        n0 = int(_scr.neighbor_pairs(model.basis, eps=ks.screen.eps,
                                     unique=ks.screen.unique)[0].shape[0])
        pad_to = -(-int(ks.screen.pad * n0) // G) * G
    state = _refresh_state(ks, model.basis, pad_to, multinode)
    _n_pairs, _pair_fill_frac = (_pair_fill(state, pad_to) if not multinode else (None, None))

    m_tr, m_st = _partition(model)
    opt = splat_adam(1e-2, steps) if optimizer is None else optimizer
    opt_state = opt.init(m_tr)

    @eqx.filter_jit
    def step_fn(ks_, m_tr, opt_state, state):    # ks_/state as ARGS, not closed over: the
        def E(tr):                                # sharded global grid inside ks_ can't be
            m = eqx.combine(tr, m_st)             # captured by a jit in multi-host
            return ks_(m, state)
        E_ = jax.checkpoint(lambda tr: E(tr)) if remat else E
        (e, eaux), g = jax.value_and_grad(lambda tr: E_(tr), has_aux=True)(m_tr)
        gnorm = optax.global_norm(g)
        u, opt_state = opt.update(g, opt_state, m_tr)
        m_tr = eqx.apply_updates(m_tr, u)
        # Not a no-op on a mesh. Replicated params are recomputed per device through regeigh, an
        # eigendecomposition of a near-degenerate overlap, so a last-bit difference between copies
        # amplifies into divergent updates. Must stay unconditional.
        if ks_.mesh is not None:
            m_tr = _shd.sync_replicas(m_tr, ks_.mesh)
        return m_tr, opt_state, e, eaux, gnorm

    if os.environ.get("DFTAX_AOT_ANALYZE"):
        # compile-only probe: names the temps behind a jit_step_fn OOM without allocating them
        compiled = step_fn.lower(ks, m_tr, opt_state, state).compile().compiled
        mem = compiled.memory_analysis()
        print(f"AOT temp={mem.temp_size_in_bytes/2**30:.2f}GiB "
              f"args={mem.argument_size_in_bytes/2**30:.2f}GiB "
              f"out={mem.output_size_in_bytes/2**30:.2f}GiB", flush=True)
        hlo = os.environ["DFTAX_AOT_ANALYZE"]
        if hlo != "1":   # a path => dump the HLO so the big temps can be named by shape
            pathlib_txt = compiled.as_text()
            with open(hlo, "w") as fh:
                fh.write(pathlib_txt)
            print(f"AOT hlo -> {hlo} ({len(pathlib_txt)/2**20:.0f} MiB)", flush=True)
        raise SystemExit(0)

    # --- resume ---
    start = 0
    if checkpoint is not None and checkpoint.resume:
        loaded = _load_ckpt(checkpoint.base, m_tr, opt_state)
        if loaded is not None:
            m_tr, opt_state, start = loaded
            state = _refresh_state(ks, eqx.combine(m_tr, m_st).basis, pad_to, multinode)
            if rank0:
                print(f"  resumed from {checkpoint.base} at step {start}/{steps}", flush=True)

    if snapshot_every and snapshot_cb is not None and start == 0:   # frame 0
        snapshot_cb(0, eqx.combine(m_tr, m_st), state.aux, float("nan"))

    # `nelec_ref` arms the electron-count guard: a regression after the run has integrated to
    # Σocc is corruption, not physics — see CollapseGuard._nelec_bad for the rule and why
    # finiteness checks alone miss it.
    cguard = (CollapseGuard(guard.base,
                            lambda path, bundle, step: (_save_ckpt(path, bundle[0], bundle[1], step)
                                                        if rank0 else None),
                            nelec_ref=float(jnp.sum(model.occupations)), nelec_rtol=guard.nelec_rtol)
              if guard is not None else None)

    hist_step, hist_e, hist_ej, hist_g = [], [], [], []
    collapsed = False
    i = start
    for i in range(start, steps):
        if refresh_every and i and i % refresh_every == 0:      # splats moved
            # DFTAX_REFRESH_PROF=1 attributes the refresh step's cost across the rebuild, the
            # pair-fill host sync, and the step that follows. Off, this is one extra `if`.
            _prof = os.environ.get("DFTAX_REFRESH_PROF")
            if _prof:
                import time as _time
                _t0 = _time.time()
                state = _refresh_state(ks, eqx.combine(m_tr, m_st).basis, pad_to, multinode)
                jax.block_until_ready(jax.tree.leaves(eqx.filter(state, eqx.is_array)))
                _t1 = _time.time()
                _n_pairs, _pair_fill_frac = ((None, None) if multinode
                                             else _pair_fill(state, pad_to))
                _t2 = _time.time()
                print(f"# REFRESH_PROF step={i} rebuild={_t1 - _t0:.3f}s "
                      f"pair_fill={_t2 - _t1:.3f}s", flush=True)
            else:
                state = _refresh_state(ks, eqx.combine(m_tr, m_st).basis, pad_to, multinode)
                if not multinode:
                    _n_pairs, _pair_fill_frac = _pair_fill(state, pad_to)
            if reset_opt_on_refresh:
                # A refresh changes the energy functional discontinuously, so Adam's moments hold
                # curvature from the old one. Dropping them costs the warmed-up per-coordinate
                # scaling, so expect a transient after every refresh.
                opt_state = opt.init(eqx.filter(m_tr, eqx.is_inexact_array))
        cur_bundle = (m_tr, opt_state)             # pre-update refs — guard's "after" on collapse
        m_tr, opt_state, E, eaux, gnorm = step_fn(ks, m_tr, opt_state, state)
        if cguard is not None and cguard.check(i, cur_bundle, E, gnorm,
                                               nelec=float(eaux.nelec)):
            if rank0:
                why = ("" if cguard.reason != "nelec" else
                       f" nelec={float(eaux.nelec):.4f} vs {cguard.nelec_ref:.4f}")
                print(f"  step {i:4d}  *** COLLAPSE ({cguard.reason}) *** E={float(E):.4e} "
                      f"|g|={float(gnorm):.2e}{why}"
                      f" — saved {guard.base}_{{before,after}}; halting", flush=True)
            collapsed = True
            break
        if rank0 and checkpoint is not None and checkpoint.every and (
                (i + 1) % checkpoint.every == 0 or i == steps - 1):
            _save_ckpt(checkpoint.base, m_tr, opt_state, i + 1)
        if snapshot_every and snapshot_cb is not None and (
                (i + 1) % snapshot_every == 0 or i == steps - 1):
            snapshot_cb(i + 1, eqx.combine(m_tr, m_st), state.aux, float(E))

        do_monitor = (monitor is not None and monitor.every
                      and (i % monitor.every == 0 or i == steps - 1))
        if step_cb is None and not do_monitor:
            continue
        m_ = {"step": i, "E_total": float(E), "grad_norm": float(gnorm),
              "E_kinetic": float(eaux.kinetic), "E_hartree": float(eaux.hartree),
              "E_xc": float(eaux.xc), "E_external": float(eaux.external),
              "nelec": float(eaux.nelec)}
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
                print(f"  step {i:4d}  E={float(E):.6f}  |g|={float(gnorm):.2e}{extra}",
                      flush=True)

    return TrainResult(
        model=eqx.combine(m_tr, m_st), state=state,
        n_steps=(i + 1 if steps > start else start), collapsed=collapsed,
        step=np.asarray(hist_step, dtype=int), energy=np.asarray(hist_e),
        e_hartree=np.asarray(hist_ej), grad_norm=np.asarray(hist_g),
    )


def evaluate(ks: SplatKS, model: SplatModel, state: SplatState | None = None) -> float:
    """Exact-Coulomb energy of a converged model, streamed at O(M²) memory.

    E_J is the exact streaming Coulomb; hybrids use RI-K exchange with the
    converged K-aux (the exact M⁴ exchange tensor does not fit at the sizes
    where splats win, and RI-K is the method's own ~1 mHa exchange). The
    one-electron energies stream over the full pair list and the grid density
    streams in chunks, so no dense (M, M, 3, 3) or (G, M) buffer ever forms.

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
    coords, charges = ks.atom_coords, ks.atom_charges
    occ = model.occupations
    pi, pj, v = _scr.neighbor_pairs(model.basis, eps=-1.0)   # full list — exact, streamed V_ne
    # S and T are never formed — see `SplatKS._finish`; the contractions carry (M, n_occ). P is
    # still built here because the exact streamed E_J below takes the density matrix itself.
    C = orthonormalize(model.C, model.C.T @ _full.overlap_times(model.basis, model.C))
    P = C @ jnp.diag(occ) @ C.T
    E_kin = occ @ jnp.sum(C * _full.kinetic_times(model.basis, C), axis=0)
    E_ext = _scr.screened_external_energy(                   # V_ne streamed, full pair list
        model.basis, C, occ, pi, pj, coords, charges, v)
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
    E_J = _cs.full_coulomb_energy(model.basis, P)            # exact streaming, O(M²)
    E_K = (_cdf.df_exchange_energy(model.basis, C, state.aux_K)
           if a_x > 0.0 else 0.0)                            # RI-K (exact-K infeasible)
    a_lr = getattr(ks.coulomb, "hf_coeff_lr", 0.0)
    E_K_lr = (_cdf.df_exchange_energy(model.basis, C, state.aux_K,
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
    return float(E_kin + E_ext + E_J + a_x * E_K + a_lr * E_K_lr + E_xc + E_nn)


def flops(ks: SplatKS, model: SplatModel, *,
          optimizer: optax.GradientTransformation | None = None,
          steps: int = 1000) -> float:
    """Compile-time XLA FLOP estimate of ONE training step (no execution).

    Builds the same jitted step as :func:`train` (with a fresh refresh state)
    and reads the lowered module's cost analysis. ``steps`` only shapes the
    default optimizer's schedule.
    """
    pad_to = None
    if ks.screen is not None:
        G = 1 if ks.mesh is None else int(ks.mesh.size)      # mesh needs a G-multiple pad
        n0 = int(_scr.neighbor_pairs(model.basis, eps=ks.screen.eps,
                                     unique=ks.screen.unique)[0].shape[0])
        pad_to = -(-int(ks.screen.pad * n0) // G) * G
    state = refresh(ks, model.basis, pad_to=pad_to)
    m_tr, m_st = _partition(model)
    opt = splat_adam(1e-2, steps) if optimizer is None else optimizer
    opt_state = opt.init(m_tr)

    @eqx.filter_jit
    def step_fn(ks_, m_tr, opt_state, state):
        def E(tr):
            return ks_(eqx.combine(tr, m_st), state)
        (e, eaux), g = jax.value_and_grad(E, has_aux=True)(m_tr)
        u, opt_state = opt.update(g, opt_state, m_tr)
        return eqx.apply_updates(m_tr, u), opt_state, e

    lowered = step_fn.lower(ks, m_tr, opt_state, state)
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
