"""One point of the splat M ladder: (system, M, seed, T, init), Adam on the paper's schedule.

    python -m experiments.rebuttal.ladder --system porphine --M 812 --seed 0 --T 12000 --init chem
    python -m experiments.rebuttal.ladder --system porphine --M 812 --T 12000 --two_t   # 2T protocol

- ``--init random``: the atom-centered random splats and random C of Figure 5. ``--init chem``:
  the chemistry-informed splats and MinAO C (Appendix E), the C cached next to the checkpoint.
- Adam on warmup -> cosine (``splat_adam``) over T steps. With ``--two_t`` the schedule runs over
  2T steps and the exact energy is also taken at step T (RESULT_PARTIAL step=T).
- The exact ``evaluate()`` runs at the end (and at T) only, never on a cadence.
- Checkpoints every ``--ckpt_every`` steps; rerunning the same command resumes, the T point
  included (``<base>.T.json`` marks it done).
- RESULT carries the run's diagnostics: the grid electron error per atom and the per-splat grid
  norms (computed here), plus any field ``train`` reports (``TrainResult.diagnostics`` or named
  attributes, read defensively).
"""
import argparse
import inspect
import json
import math
import os
import time

import numpy as np


def _args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--init", choices=("random", "chem"), default="random")
    ap.add_argument("--two_t", action="store_true", help="schedule over 2T, exact energy at T too")
    ap.add_argument("--xc", default="pbe")
    ap.add_argument("--basis", default="cc-pvdz", help="reference basis for n_ao in RESULT")
    ap.add_argument("--grid_level", type=int, default=3)
    ap.add_argument("--refresh", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--ckpt_dir", default=None, help="default $DFTAX_CKPT_DIR, else results/ckpt")
    ap.add_argument("--monitor", type=int, default=500)
    ap.add_argument("--G", type=int, default=None, help="mesh size (when the builder takes one)")
    ap.add_argument("--grid_check", type=int, default=0, help="re-evaluate on this grid level")
    return ap.parse_args()


def _tag(a):
    return (f"rb_{a.system}_M{a.M}_s{a.seed}_T{a.T}_{a.init}" + ("_2T" if a.two_t else ""))


def _scalar(v):
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        x = np.asarray(v)
        if x.size == 1:
            return x.item()
    except Exception:
        pass
    return None


# Names train() may report (W2's guards); read when present, never required.
_TRAIN_FIELDS = ("stop_reason", "reason", "nelec_err_per_atom", "nelec_err_atom_max",
                 "grid_nelec_err", "splat_norm_max", "splat_norm_n_over", "n_compiles",
                 "n_cap_sharp", "n_cap_wide", "n_cap_aniso")


def train_fields(res):
    out = {}
    d = getattr(res, "diagnostics", None)
    items = list(d.items()) if isinstance(d, dict) else []
    items += [(k, getattr(res, k, None)) for k in _TRAIN_FIELDS]
    for k, v in items:
        v = _scalar(v)
        if v is not None and not (isinstance(v, str) and not v.strip()):
            out[f"train_{k}"] = str(v).replace(" ", "_") if isinstance(v, str) else v
    return out


def grid_diagnostics(ks, model, chunk=4096, occupied=1e-6):
    """Grid electron count error per atom, and the per-splat grid norms (int g^2 on the grid), over
    all splats and over the occupied ones, (C C^T)_mm > ``occupied``."""
    import jax.numpy as jnp
    from gs_dft.integrals import dense
    from gs_dft.ks.orthonormalize import orthonormalize
    from gs_dft.screening import grid as gscr

    occ = model.occupations
    C = orthonormalize(model.C, model.C.T @ dense.overlap_times(model.basis, model.C))
    gp, gw = ks.grid_points, ks.grid_weights
    rho, _ = gscr.chunked_density_and_grad(model.basis, C, occ, gp)
    n_grid = float(jnp.sum(gw * rho))
    n_atoms = int(ks.atom_coords.shape[0])
    norms = jnp.zeros(model.C.shape[0])
    for s in range(0, int(gp.shape[0]), chunk):
        B = dense.eval_basis_on_grid(model.basis, gp[s:s + chunk])
        norms = norms + jnp.sum(gw[s:s + chunk, None] * B * B, axis=0)
    norms = np.asarray(norms)
    occ_norms = norms[np.asarray(jnp.sum(C * C, axis=1)) > occupied]
    out = {"nelec_grid": n_grid,
           "nelec_err_per_atom": abs(n_grid - float(jnp.sum(occ))) / n_atoms,
           "n_occupied_splats": int(occ_norms.size)}
    for tag, x in (("", norms), ("_occ", occ_norms)):
        x = x if x.size else np.full(1, np.nan)
        out.update({f"splat_norm{tag}_max": float(x.max()), f"splat_norm{tag}_min": float(x.min()),
                    f"splat_norm{tag}_dev_max": float(np.abs(x - 1.0).max()),
                    f"splat_norm{tag}_n_gt_1p1": int((x > 1.1).sum())})
    return out


def _init(a, mol, M, ckdir, ks):
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    from gs_dft import init_model, chem
    from gs_dft.ks.train import native_grid

    key = jr.PRNGKey(a.seed)
    if a.init == "random":
        return init_model(mol, M, key)
    model = init_model(mol, M, key, init=chem())
    from gs_dft.ks import resolution as _res
    model = eqx.tree_at(lambda m: m.basis, model,           # MinAO C for the capped splats train runs
                        _res.capped(model.basis, _res.grid_resolution(ks), ks.atom_coords))
    from experiments.exp8_data_free_init.minao_init import minao_C0
    path = os.path.join(ckdir, f"initC_{a.system}_M{M}_s{a.seed}_minao_capped.npy")
    shape = tuple(int(x) for x in model.C.shape)
    C = np.load(path) if os.path.exists(path) else None
    if C is None or C.shape != shape:
        t0 = time.time()
        grid = native_grid(mol, a.grid_level)
        C = np.asarray(minao_C0(model.basis, list(mol.symbols), np.asarray(mol.coords), shape[1],
                                np.asarray(grid.coords), np.asarray(grid.weights)), dtype=np.float64)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.save(fh, C)
        os.replace(tmp, path)
        print(f"# init chem: MinAO C built in {time.time() - t0:.1f}s, cached {path}", flush=True)
    return eqx.tree_at(lambda m: m.C, model, jnp.asarray(C))


def _saved_step(base):
    try:
        with open(base + ".step") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return 0


def main():
    a = _args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import gs_dft
    from experiments.common import systems, builders, results, probes, paths
    from gs_dft import nao as nao_of
    from gs_dft.ks.train import (train, evaluate, splat_adam, monitor, collapse,
                                 ckpt as ckpt_spec)

    t_start = time.time()
    mol = systems.molecule(system=a.system, basis=a.basis)
    nao = int(nao_of(mol))
    M, T = int(a.M), int(a.T)
    total = 2 * T if a.two_t else T
    xc = builders.xc_of(a.xc)
    kw = {}
    if a.G is not None:
        if "G" not in inspect.signature(builders.splat_ks).parameters:
            raise SystemExit("--G needs builders.splat_ks(G=...)")
        kw["G"] = int(a.G)
    ks = builders.splat_ks(mol, xc, grid_level=a.grid_level, screened=True, **kw)
    G = 1 if ks.mesh is None else int(ks.mesh.size)
    ckdir = a.ckpt_dir or paths.artifact_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          "results", "ckpt"))
    os.makedirs(ckdir, exist_ok=True)
    tag = _tag(a)
    base = os.path.join(ckdir, tag)
    print(f"# gs_dft from {gs_dft.__file__}  devices={jax.devices()}", flush=True)
    print(f"# {tag}: {len(mol.symbols)} atoms  nao({a.basis})={nao}  M={M} ({M / nao:.2f}x)  "
          f"T={T} schedule={total}  init={a.init}  xc={a.xc}  grid={a.grid_level} "
          f"({int(ks.grid_points.shape[0])} pts)  refresh={a.refresh}  G={G}  ckpt={base}",
          flush=True)

    model0 = _init(a, mol, M, ckdir, ks)
    opt = splat_adam(a.lr, total)
    common = dict(M=M, seed=a.seed, T=T, init=a.init, schedule=total, xc=a.xc, nao=nao,
                  n_atoms=len(mol.symbols), grid_level=a.grid_level, G=G)

    def run(steps):
        return train(ks, model0, steps=steps, optimizer=opt, refresh_every=a.refresh,
                     monitor=monitor(a.monitor, cond_every=0),
                     checkpoint=ckpt_spec(base, every=a.ckpt_every, resume=True),
                     guard=collapse())

    def report(res, step, head):
        """Exact energy + diagnostics as one line; None after a collapse."""
        t0 = time.time()
        row = dict(common, steps_run=res.n_steps, collapsed=bool(res.collapsed),
                   **train_fields(res))
        if res.collapsed:
            row.update(E=float("nan"))
        else:
            row["E"] = float(evaluate(ks, res.model, res.state))
            row.update(grid_diagnostics(ks, res.model))
            if a.grid_check:
                row.update(builders.grid_fields(mol, xc, res.model, a.grid_check, row["E"],
                                                state=res.state))
        row.update(eval_s=round(time.time() - t0, 1), wall_s=round(time.time() - t_start),
                   peak_gpu_mb=round(probes.peak_gpu_mb()))
        if head == "partial":
            return results.partial("splat", a.system, step=step, **row)
        return results.result("splat", a.system, steps=total, **row)

    if a.two_t:
        marker = base + ".T.json"
        if os.path.exists(marker):
            with open(marker) as fh:
                print(f"# T point already evaluated: {json.load(fh)}", flush=True)
        elif _saved_step(base) > T:
            print(f"# T point lost: the checkpoint is past T ({_saved_step(base)} > {T})",
                  flush=True)
        else:
            res = run(T)
            line = report(res, T, "partial")
            print(line, flush=True)
            if res.collapsed:
                print(results.result("splat", a.system, steps=total, **dict(
                    common, E=float("nan"), collapsed=True, steps_run=res.n_steps)), flush=True)
                return
            with open(marker, "w") as fh:
                json.dump(results.parse(line), fh)
    res = run(total)
    print(report(res, total, "result"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from gs_dft.screening import PairPadOverflow
        from gs_dft.screening.cells import CellOverflow
        if not isinstance(exc, (PairPadOverflow, CellOverflow)):
            raise
        print(f"RESTART ({type(exc).__name__}): {exc}", flush=True)   # the point is saved
        raise SystemExit(75)
