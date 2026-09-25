"""Run one rung of the machinery ablation, adding one engineering piece at a time.

Supplies Figure 2(b), reach under density fitting, screening and chunking, Figure 2(c), the
sharded run, and the regularized-eigensolve table of Appendix E. Every rung is an entry of
:data:`RUNGS`::

    uv run python -m experiments.exp1_ablation_ladder.machinery experiment=exp1_machinery system=ethanol rung=M2

A disabled feature is switched off by rebinding the function the energy calls, so every arm runs
the real training path. Every run prints one ``RESULT kind=splat`` row, including runs that fail:
an out-of-memory ceiling is a data point of Figure 2(b).
"""
import os
import pathlib
import time

import hydra
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from omegaconf import DictConfig

from gs_dft import init_model, nao as nao_of
from gs_dft.basis.chem import chem
from gs_dft.integrals import dense as _full
from gs_dft.ks import diagnostics as _mtr
from gs_dft.ks import energy as _energy
from gs_dft.ks.energy import SplatKS, SplatModel
from gs_dft.ks.orthonormalize import reg_inv_sqrt
from gs_dft.ks.terms import df, exact, pairlist
from gs_dft.ks.train import ckpt, evaluate, monitor, native_grid, splat_adam, train
from experiments.common import builders, resources, results, systems, tracking
from experiments.exp1_ablation_ladder.ladder import build_basis


def _free_mb():
    """Free memory on THIS process's first device, in MB, or -1 if it cannot be read.

    Local, not ``jax.devices()[0]``: under multi-controller that is the global list, so device 0
    belongs to one process and every other would report a peer's card.

    Context for a failure row: an allocation failure on a shared node may be the neighbours rather
    than the method. Never raises -- it runs on the crash-reporting path.
    """
    try:
        import jax
        st = jax.local_devices()[0].memory_stats() or {}
        lim, used = st.get("bytes_limit", 0), st.get("bytes_in_use", 0)
        return int((lim - used) / 1e6) if lim else -1
    except Exception:
        return -1


def _peak_mb():
    """Peak device memory in MB, or -1 if it cannot be read.

    The high-water mark, which is what decides whether a rung fits on the card -- this is panel 2's
    reach axis. Never raises, like :func:`_free_mb`.
    """
    try:
        import jax
        st = jax.local_devices()[0].memory_stats() or {}
        return int(st.get("peak_bytes_in_use", 0) / 1e6) or -1
    except Exception:
        return -1


#: rung -> (density fitting, eigenvalue floor, screening, grid chunk, sharded, sync_replicas).
#: Ablating ``sync_replicas`` is only meaningful on a sharded run; :func:`run` raises otherwise.
#: The P* rungs are the figure's arms; the M* rungs below decompose the same features one by one
#: and are reachable but unplotted.
RUNGS = {
    "P2_dense": (False, True,  False, None, False, True),
    "P2_df":    (True,  True,  False, None, False, True),
    "P2_fast":  (True,  True,  True,  4096, False, True),
    "P3_nofloor": (True, False, True, 4096, False, True),
    "P3_floor":   (True, True,  True, 4096, False, True),
    "P3_multi":   (True, True,  True, 4096, True,  True),
    "P3_nosync":  (True, True,  True, 4096, True,  False),
    "P3_multi_nofloor": (True, False, True, 4096, True,  True),
    "P3_multi_floor":   (True, True,  True, 4096, True,  True),
}
_WHY = {
    "P2_dense":   "exact Coulomb, dense grid — the reach ceiling this panel exists to exceed",
    "P2_df":      "+ frozen pair-product DF",
    "P2_fast":    "+ screening + chunked grid (the production fast path)",
    "P3_nofloor": "fast path WITHOUT the regularized eigensolve — expected to fail",
    "P3_floor":   "+ regularized eigensolve",
    "P3_multi":   "+ multi-GPU sharding & sync_replicas",
    "P3_nosync":  "sharded WITHOUT sync_replicas — expected to give a WRONG ANSWER, not to crash",
    "P3_multi_nofloor": "sharded, floor INERT — the budget the collapse needs, made affordable",
    "P3_multi_floor":   "sharded + regularized eigensolve — the same budget, floored",
}

RUNGS.update({
    "M0": (False, False, False, None, False, True),
    "M1": (True,  False, False, None, False, True),
    "M2": (True,  True,  False, None, False, True),
    "M3": (True,  True,  True,  None, False, True),
    "M4": (True,  True,  True,  4096, False, True),
    "M5": (True,  True,  True,  4096, True,  True),
})
_WHY.update({"M0": "exact Coulomb, no floor", "M1": "+ DF", "M2": "+ floor", "M3": "+ screening",
             "M4": "+ chunked grid", "M5": "+ sharding"})

# Above either bound the exact `evaluate()` is skipped and E is the DF training objective, marked
# by `E_exact=False` on the row. Both are needed: the exact streamed Coulomb scales with M, not with
# atoms, and M=560 already exhausts an 80 GB card.
EXACT_EVAL_MAX_ATOMS = int(os.environ.get("EXACT_EVAL_MAX_ATOMS", "50"))
EXACT_EVAL_MAX_M = int(os.environ.get("EXACT_EVAL_MAX_M", "500"))

_CKPT_DIR = pathlib.Path(__file__).resolve().parent / "results" / "ckpt"
_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def ckpt_base(rung, system, M, seed, steps, grid_level) -> str:
    """The checkpoint path for one run -- the single definition, shared by the runner and the tests.

    The name carries every variable that changes the checkpoint's contents. ``grid_level`` is
    included even though it changes no array shape: without it a grid-3 checkpoint deserializes
    cleanly into a grid-2 run and the job continues under a quadrature it never used. Set
    ``CKPT_SUFFIX`` to keep replicate runs of one configuration from resuming each other.
    """
    suffix = os.environ.get("CKPT_SUFFIX", "")
    return str(_CKPT_DIR / f"{rung}_{system}_M{int(M)}_s{int(seed)}"
               f"_n{int(steps)}_g{int(grid_level)}{('_' + suffix) if suffix else ''}")


def _ckpt_step(base) -> int:
    """Steps already completed at ``base``, or 0 if there is no checkpoint.

    The ms/step denominator must be the steps THIS process ran, not the configured budget, or a
    resumed run reports a per-step cost far below the truth.
    """
    try:
        return int(pathlib.Path(f"{base}.step").read_text().strip())
    except Exception:
        return 0


def _set_floor(enabled: bool):
    """Rebind the orthonormalization the energy uses. Returns the original, for restoration.

    The unfloored arm uses plain ``eigh`` with no clamp, not ``reg_inv_sqrt`` with a tiny
    ``floor_rel``: that clamp is ``maximum(w, floor_rel * w[-1])``, which rescues a NEGATIVE
    eigenvalue however small the floor, so a small floor still removes the failure being ablated.
    With no clamp the arm dies the moment the occupied Gram goes indefinite, which is the point.
    """
    original = _energy.lowdin_orthonormalize
    if enabled:
        return original

    def unfloored(C, S):
        M_mo = C.T @ (S @ C)
        w, U = jnp.linalg.eigh(M_mo)
        T = (U * (1.0 / jnp.sqrt(w))) @ U.T        # no clamp: 1/sqrt(negative) -> NaN, deliberately
        return C @ T

    _energy.lowdin_orthonormalize = unfloored
    return original


def _set_sync(enabled: bool):
    """Rebind ``shard.sync_replicas``. Returns the original, for restoration.

    ``ks/train.py`` reaches the function through its module, so replacing the attribute leaves the
    step function, mesh and collectives untouched -- only the all-reduce that re-synchronizes the
    replicas is gone. The expected failure is a WRONG ANSWER, not a crash: judge this arm on the
    energy it reports against the ndev=1 control, never on whether it completed.
    """
    from gs_dft.ks import shard as _shd
    original = _shd.sync_replicas
    if not enabled:
        _shd.sync_replicas = lambda tree, mesh: tree
    return original


def _restore_sync(original):
    from gs_dft.ks import shard as _shd
    _shd.sync_replicas = original


def electron_count(model):
    """Tr(P·S), the state's normalization. NOT an N-representability test.

    Because ``lowdin_orthonormalize`` constructs C by orthonormalizing, this equals sum(occupations)
    identically unless ``reg_inv_sqrt`` clamps -- so it reports whether the floor engaged, and it is
    ``||C^T S C - I||`` under another name. It detects an orthonormality collapse and nothing else;
    auxiliary staleness conserves it exactly. Use ``E_train - E_exact`` for that failure instead.
    """
    S = _full.overlap_matrix(model.basis)
    C = _energy.lowdin_orthonormalize(model.C, S)
    P = (C * model.occupations) @ C.T
    return float(jnp.sum(P * S))


def dense_gram_ends(model):
    """``(lambda_min, lambda_max, min_gap)`` of C^T S C against the DENSE overlap.

    Dense, not screened: a Gram assembled from a stale pair list is not the Gram of anything, and
    the dense matrix is the one ``lowdin_orthonormalize`` actually inverts.
    """
    S = _full.overlap_matrix(model.basis)
    M_mo = model.C.T @ (S @ model.C)
    w = jnp.linalg.eigvalsh(M_mo)
    # The minimum eigenvalue GAP, not lambda_min, is what `reg_inv_sqrt`'s backward rule reacts to:
    # its divided difference diverges when two eigenvalues collide, degenerate or not small.
    gaps = jnp.diff(w)                                          # w ascends out of eigvalsh
    return float(w[0]), float(w[-1]), float(jnp.min(gaps)) if gaps.size else float("nan")


def kinetic_energy(model, state=None):
    """Kinetic energy at the current state.

    A negative value is impossible for a valid normalized state and signals an indefinite occupied
    Gram -- the failure the eigenvalue floor prevents.
    """
    S = _full.overlap_matrix(model.basis)
    C = _energy.lowdin_orthonormalize(model.C, S)
    P = (C * model.occupations) @ C.T
    return float(jnp.sum(P * _full.kinetic_matrix(model.basis)))


def run(cfg: DictConfig) -> None:
    """Run one rung and emit its ``RESULT`` row.

    Split from the ``@hydra.main`` entry point so tests can call it with a composed config: the
    decorator resolves ``config_path`` relative to this file and fails when invoked in-process.
    """
    rung = str(cfg.rung)
    if rung not in RUNGS and rung.upper() in RUNGS:
        rung = rung.upper()
    use_df, floor, screen, chunk, sharded, sync = RUNGS[rung]
    if not sync and not sharded:
        raise ValueError(f"rung {rung}: sync_replicas can only be ablated on a sharded run")
    tracking.init(cfg, __name__, name=f"{rung}_{cfg.system}_s{cfg.seed}")

    mol = systems.molecule(cfg)
    M = int(cfg.m) if cfg.m is not None else int(cfg.m_mult * nao_of(mol))
    print(f"# exp1 {rung}: {_WHY[rung]}", flush=True)
    print(f"#   system={cfg.system} M={M} df={use_df} floor={floor} screen={screen} sync={sync} "
          f"chunk={chunk} sharded={sharded} steps={cfg.steps} seed={cfg.seed}", flush=True)

    original = _set_floor(floor)
    original_sync = _set_sync(sync)
    try:
        model = init_model(mol, M, jr.PRNGKey(int(cfg.seed)),
                           init=chem() if str(cfg.init) == "chem" else None)
        mesh = None
        if sharded:
            from gs_dft.ks.shard import make_mesh
            mesh = make_mesh()
        ks = SplatKS(mol, builders.xc_of(cfg.xc),
                     grid=native_grid(mol, int(cfg.grid_level), chunk=chunk),
                     coulomb=df(lam=float(cfg.df_lam),
                                scales=builders.aux_scales(cfg.aux_mult)) if use_df else exact(),
                     screen=pairlist(eps=float(cfg.get("screen_eps", 1e-7)), pad=float(cfg.screen_pad)) if screen else None,
                     mesh=mesh)

        n_atoms = len(mol.symbols)
        exact_eval = n_atoms <= EXACT_EVAL_MAX_ATOMS and M <= EXACT_EVAL_MAX_M
        if not exact_eval:
            why = (f"{n_atoms} atoms > {EXACT_EVAL_MAX_ATOMS}" if n_atoms > EXACT_EVAL_MAX_ATOMS
                   else f"M={M} > {EXACT_EVAL_MAX_M}")
            print(f"#   exact evaluate() SKIPPED ({why}); E is the DF objective. "
                  f"Accuracy comes from the GTO baselines.", flush=True)

        ck = ckpt(ckpt_base(rung, cfg.system, M, cfg.seed, cfg.steps, cfg.grid_level),
                  every=min(max(int(cfg.steps) // 20, 25), 50), resume=True)
        resumed_from = _ckpt_step(ck.base)
        steps_run = max(int(cfg.steps) - resumed_from, 1)
        if resumed_from:
            print(f"#   resuming at step {resumed_from}/{cfg.steps} — ms_step is over the "
                  f"{steps_run} steps THIS process runs, not the full budget", flush=True)
        _wall_file = ck.base + ".wall"
        try:
            with open(_wall_file) as _fh:
                _wall_prev = float(_fh.read().strip() or 0.0)
        except (OSError, ValueError):
            _wall_prev = 0.0
        if resumed_from and _wall_prev:
            print(f"#   {_wall_prev / 60:.1f} min already spent before this allocation", flush=True)
        t0 = time.time() - _wall_prev
        trace_n = max(int(cfg.steps) // 200, 1)

        n_target = 2.0 * (int(mol.nelectron) // 2)

        # A one-cell list so `_trace` can read what the step callback wrote without `nonlocal`.
        _LAST_GNORM = [float("nan")]

        def _trace(step, model_, _aux, E):
            try:
                nel = electron_count(model_)
            except Exception:
                nel = float("nan")
            try:
                lo, hi, gp = dense_gram_ends(model_)
            except Exception:
                lo = hi = gp = float("nan")
            try:
                from gs_dft.screening import neighbor_pairs as _np
                _p = _np(model_.basis, eps=float(cfg.get("screen_eps", 1e-7)))
                npair = int(_p[2].sum()) if len(_p) > 2 and _p[2] is not None else int(_p[0].size)
            except Exception:
                npair = -1
            _g = _LAST_GNORM[0]
            print(f"TRACE rung={rung} step={int(step)} t={time.time() - t0:.3f} "
                  f"E={float(E):.8f} gnorm={_g:.6e} nelec={nel:.6f} "
                  f"dnelec={nel - n_target:+.3e} "
                  f"gmin={lo:+.6e} gratio={lo / hi if hi else float('nan'):+.3e} "
                  f"mingap={gp / hi if hi else float('nan'):.3e} "
                  f"npair={npair}", flush=True)
            try:
                with open(_wall_file, "w") as _fh:
                    _fh.write(f"{time.time() - t0:.3f}")
            except OSError:
                pass

        try:
            res = train(ks, model, steps=int(cfg.steps), refresh_every=int(cfg.refresh),
                        monitor=monitor(trace_n), checkpoint=ck,
                        optimizer=splat_adam(steps=int(cfg.steps)),
                        snapshot_every=trace_n, snapshot_cb=_trace,
                        step_cb=lambda m_: _LAST_GNORM.__setitem__(0, float(m_["grad_norm"])))
            # Read the peak BEFORE evaluate(): it is a process high-water mark, and the exact
            # Coulomb below would replace the training footprint panel 2 plots.
            peak_train = _peak_mb()
            E = (float(evaluate(ks, res.model, res.state)) if exact_eval
                 else float(ks(res.model, res.state)[0]))
            e_kin = kinetic_energy(res.model)
            _nelec = electron_count(res.model)
            gnorm = float(res.grad_norm[-1]) if len(res.grad_norm) else float("nan")
            lam = _mtr.lindep_metrics(ks, res.model, res.state)
            failed = ""
        except Exception as exc:
            msg = str(exc).replace("\n", " ")[:160]
            failed = ("oom" if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc)
                      else type(exc).__name__)
            E = e_kin = gnorm = _nelec = float("nan")
            res, lam = None, {}
            peak_train = _peak_mb()
            print(f"# {rung} FAILED ({failed}): {msg}", flush=True)
        wall = time.time() - t0
    finally:
        _energy.lowdin_orthonormalize = original
        _restore_sync(original_sync)

    n_occ = int(mol.nelectron // 2)
    par = resources.splat_params(M, n_occ)
    row = dict(rung=rung, M=M, n_occ=n_occ, params=par["total"], params_basis=par["basis"],
               df=bool(use_df), floor=bool(floor), screen=bool(screen), sync=bool(sync),
               init=str(cfg.init),
               chunk=(chunk or 0), sharded=bool(sharded), steps=int(cfg.steps),
               seed=int(cfg.seed), xc=cfg.xc, E=E, E_exact=bool(exact_eval),
               E_kin=e_kin, grad_norm=gnorm,
               collapsed=bool(res.collapsed) if res is not None else False,
               failed=(failed or "none"), finite=bool(np.isfinite(E)),
               kinetic_ok=bool(e_kin >= 0.0),
               nelec=_nelec, nelec_err=(_nelec - 2.0 * n_occ),
               nelec_ok=bool(abs(_nelec - 2.0 * n_occ) < 1e-3 * 2.0 * n_occ),
               lam_min=lam.get("dense_min_eig", float("nan")),
               lam_max=lam.get("dense_max_eig", float("nan")),
               lam_ratio=lam.get("dense_eig_ratio", float("nan")),
               peak_mb=_peak_mb(), peak_train_mb=peak_train, gpu_free_mb=_free_mb(),
               resumed_from=int(resumed_from),
               wall_s=round(wall, 1), ms_step=round(1e3 * wall / steps_run, 2))
    print(results.result("splat", cfg.system, **row), flush=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
