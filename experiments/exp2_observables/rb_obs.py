"""Figure 3 for the rebuttal systems: polish a splat ladder checkpoint so its Hellmann-Feynman forces
are converged, then measure it against the Gaussian ladder.

``polish``: the 12k-step checkpoint of ``experiments/rebuttal/ladder.py`` (``rb_<system>_M<M>_s0_T12000_chem``
under ``$RB_CKPT``) continues through ``polish_forces.LEGS`` (2000 steps each at 1e-3, 3e-4, 1e-4) on
the ladder's own functional. Each leg checkpoints every 250 steps and resumes, so a leg survives the
end of a job. ``polish_forces``'s SCF jump of the coefficients and the 1000 steps after it are left
out: the dense Fock build does not fit past M ~ 1000, and on the paper's ladders the two together
moved max|dF| by under 5%.

``obs``: the polished cloud against every Gaussian rung, on the level-5 reporting grid of
``gto_ref.py``: the exact-Coulomb energy, the density (streamed over the grid), the Hellmann-Feynman
forces (on the screened pair list, which holds at any M), the multipoles, the density L1 against the
largest rung and max|dF| against the largest rung with forces. Archives ``splatref_<system>_M<M>.npz``
for ``selfref.py``. ``--raw`` measures the unpolished ladder checkpoint instead; it archives only a rung
with no polished model (the four largest rungs of the A–T pair, porphine, retinal and cholesterol),
so a diagnostic never replaces a polished archive.

    python -m experiments.exp2_observables.rb_obs polish sf6 816
    python -m experiments.exp2_observables.rb_obs obs sf6 816 [--raw]
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ORDER = ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z")
GRID = 3                    # the ladder's training grid
OBS_LEVEL = 5               # gto_ref's reporting grid (conf/experiment/exp2_gto_ref.yaml)
FORCE_EPS = 1e-12           # pair cutoff of the force's V_ne, as evaluate()'s
DENSE_CHECK = 6e6           # M^2 x atoms up to which the dense force is also taken, as a check


def ladder_base(system, M):
    # Where experiments/rebuttal/ladder.py wrote the ladder: $RB_CKPT, else the ladder's own default.
    d = os.environ.get("RB_CKPT") or os.environ.get("DFTAX_CKPT_DIR") or os.path.join(
        HERE, "..", "rebuttal", "results", "ckpt")
    return os.path.join(d, f"rb_{system}_M{M}_s0_T12000_chem")


def _mol(system):
    from experiments.common import systems
    return systems.molecule(system=system, basis="cc-pvdz")


def _template(mol, M):
    import jax.random as jr
    from gs_dft import init_model
    return init_model(mol, M, jr.PRNGKey(0))


def load_ladder(mol, system, M):
    """The model of a ladder checkpoint. It holds ``(trainable, opt_state)``; only the trainable part,
    which comes first in the file, is read, so the optimizer state's layout does not matter."""
    import equinox as eqx
    from gs_dft.ks.train import _partition
    base = ladder_base(system, M)
    with open(base + ".step") as fh:
        step = int(fh.read().strip())
    m_tr, m_st = _partition(_template(mol, M))
    shapes = [x.shape for x in __import__("jax").tree.leaves(m_tr)]
    with open(base + ".eqx", "rb") as fh:
        m_tr = eqx.tree_deserialise_leaves(fh, m_tr)
    got = [x.shape for x in __import__("jax").tree.leaves(m_tr)]
    if got != shapes:
        raise ValueError(f"{base}.eqx: leaf shapes {got} != {shapes}")
    return eqx.combine(m_tr, m_st), step


def save_model(path, model):
    import equinox as eqx
    tmp = path + f".tmp{os.getpid()}"
    eqx.tree_serialise_leaves(tmp, model)
    os.replace(tmp, path)


def load_model(path, mol, M):
    import equinox as eqx
    return eqx.tree_deserialise_leaves(path, _template(mol, M))


def polished_path(system, M):
    return os.path.join(RES, f"ckpt_{system}_M{M}_polished_rb.eqx")


def refs(system):
    out = {}
    for b in ORDER:
        p = os.path.join(RES, f"ref_{system}_{b}.npz")
        if os.path.exists(p):
            out[b] = np.load(p, allow_pickle=True)
    return out


def force_ref(system):
    """``(basis, F)`` of the largest rung with a gradient (``selfref.ref_grad``), else ``(None, None)``."""
    from experiments.exp2_observables.selfref import ref_grad
    for b in reversed(ORDER):
        g = ref_grad(system, b)
        if g is not None:
            return b, -g
    return None, None


def hf_forces(mol, model):
    """F = -dE/dR at fixed splats on the screened pair list (V_ne and E_nn only)."""
    from dftax.energy.xc import PBE
    from gs_dft.ks.energy import SplatKS, refresh
    from gs_dft.ks.terms import pairlist
    from gs_dft.ks.train import native_grid
    from gs_dft.ks.forces import splat_forces
    ks = SplatKS(mol, PBE(), grid=native_grid(mol, 0), screen=pairlist(eps=FORCE_EPS, pad=1.0))
    return np.asarray(splat_forces(ks, model, refresh(ks, model.basis)))


def polish(system, M):
    from dftax.energy.xc import PBE
    from experiments.common import builders, results
    from experiments.exp2_observables.polish_forces import LEGS
    from gs_dft.ks.train import train, splat_adam, monitor, ckpt
    t0 = time.time()
    mol = _mol(system)
    d = os.path.join(RES, "polish")
    os.makedirs(d, exist_ok=True)
    tag = f"{system}_M{M}"
    fb, F_ref = force_ref(system)

    def report(stage, model, **kv):
        F = hf_forces(mol, model)
        dF = float(np.abs(F - F_ref).max()) if F_ref is not None else float("nan")
        print(results.result("forces", system, M=M, dF_vs=fb or "none", stage=stage, dF_max=dF,
                             maxF=float(np.abs(F).max()), net_F=float(np.abs(F.sum(0)).max()),
                             wall_s=round(time.time() - t0), **kv), flush=True)

    # A finished leg is `<tag>_leg<k>_done.eqx`; `<tag>_leg<k>.eqx` is the trainer's checkpoint of a leg
    # cut by the walltime, resumed below. A full model under the old name is a finished leg: adopt it.
    for k in range(len(LEGS)):
        old_name = f"{d}/{tag}_leg{k}.eqx"
        if os.path.exists(old_name) and not os.path.exists(f"{d}/{tag}_leg{k}_done.eqx"):
            try:
                load_model(old_name, mol, M)
                os.replace(old_name, f"{d}/{tag}_leg{k}_done.eqx")
            except (RuntimeError, ValueError):
                pass                                    # a training checkpoint: the leg resumes from it
    done = [k for k in range(len(LEGS)) if os.path.exists(f"{d}/{tag}_leg{k}_done.eqx")]
    if done:
        k0 = max(done) + 1
        model = load_model(f"{d}/{tag}_leg{k0 - 1}_done.eqx", mol, M)
        print(f"# {tag}: legs 0..{k0 - 1} done, resuming at leg {k0}", flush=True)
    else:
        k0 = 0
        model, step = load_ladder(mol, system, M)
        print(f"# {tag}: ladder checkpoint at step {step}", flush=True)
        report("ckpt", model, ckpt_step=step)
    ks = builders.splat_ks(mol, PBE(), grid_level=GRID, screened=True)    # the ladder's functional
    for k in range(k0, len(LEGS)):
        steps, lr = LEGS[k]
        res = train(ks, model, steps=steps, optimizer=splat_adam(lr, steps),
                    monitor=monitor(500, cond_every=0),
                    checkpoint=ckpt(f"{d}/{tag}_leg{k}", every=250, resume=True))
        if res.stop_reason is not None:
            raise SystemExit(f"leg {k} stopped: {res.stop_reason}")
        model = res.model
        save_model(f"{d}/{tag}_leg{k}_done.eqx", model)
        gn = float(res.grad_norm[-1]) if len(res.grad_norm) else float("nan")
        done_steps = sum(s for s, _ in LEGS[:k + 1])
        report(f"+{done_steps}@lr{lr:g}", model, grad_norm=gn,
               E_train=float(res.energy[-1]) if len(res.energy) else float("nan"))
    save_model(polished_path(system, M), model)
    report("final", model, path=polished_path(system, M))


def obs(system, M, raw=False):
    import jax.numpy as jnp
    from dftax.energy.xc import PBE
    from dftax.grid import becke_grid
    from experiments.common import results, probes
    from experiments.exp2_observables.gto_ref import multipoles, AU2DEBYE
    from gs_dft.integrals import dense
    from gs_dft.ks.energy import SplatKS
    from gs_dft.ks.forces import splat_forces
    from gs_dft.ks.orthonormalize import orthonormalize
    from gs_dft.ks.terms import df
    from gs_dft.ks.train import evaluate, _GRID_LEVELS
    from gs_dft.screening import grid as gscr
    t0 = time.time()
    mol = _mol(system)
    model = load_ladder(mol, system, M)[0] if raw else load_model(polished_path(system, M), mol, M)
    coords = np.asarray(mol.atom_coords())
    charges = np.asarray(mol.atom_charges(), float)
    nr, lb = _GRID_LEVELS[OBS_LEVEL]
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(coords), n_radial=nr, lebedev=lb)
    w = np.asarray(gw)

    ks5 = SplatKS(mol, PBE(), grid=(gp, gw), coulomb=df())
    E = float(evaluate(ks5, model))
    occ = model.occupations
    C = orthonormalize(model.C, model.C.T @ dense.overlap_times(model.basis, model.C))
    rho = np.asarray(gscr.chunked_density(model.basis, C, occ, gp))[: w.shape[0]]
    rho_nuc = np.asarray(gscr.chunked_density(model.basis, C, occ, jnp.asarray(coords)))[: len(coords)]
    nelec = float(np.dot(w, rho))
    mu, theta = multipoles(np.asarray(gp), w, rho, coords, charges)
    F = hf_forces(mol, model)
    extra = {}
    if M * M * len(coords) <= DENSE_CHECK:
        F_dense = np.asarray(splat_forces(ks5, model))
        extra["dF_dense_vs_screened"] = float(np.abs(F_dense - F).max())
    print(f"# {system} M={M}: E={E:.8f} N_e(grid)={nelec:.6f} (expect {float(occ.sum()):.1f}) "
          f"[{time.time() - t0:.0f}s]", flush=True)

    if not raw or not os.path.exists(polished_path(system, M)):
        sref = os.path.join(RES, f"splatref_{system}_M{M}.npz")
        tmp = sref + f".tmp{os.getpid()}.npz"
        np.savez(tmp, rho=rho, grad=-F, w=w, E=E, M=int(M), n_grid=int(rho.shape[0]))
        os.replace(tmp, sref)

    rs = refs(system)
    if not rs and not raw:
        raise SystemExit(f"no ref_{system}_*.npz in {RES}")
    best = ([b for b in ORDER if b in rs] or ["none"])[-1]
    l1 = {b: float(np.dot(w, np.abs(rho - np.asarray(rs[b]["rho"])))) for b in rs
          if np.asarray(rs[b]["rho"]).shape == rho.shape}
    fb, F_ref = force_ref(system)
    dF = float(np.abs(F - F_ref).max()) if F_ref is not None else float("nan")
    print("  density L1 " + "  ".join(f"{b[-3:]}:{v:.5f}" for b, v in l1.items())
          + f"   max|F - F_{fb}| = {dF:.6f} Ha/Bohr", flush=True)
    for a in range(F.shape[0]):
        r = F_ref[a] if F_ref is not None else np.full(3, np.nan)
        print(f"  F[{a}] splat=({F[a][0]:+.5f},{F[a][1]:+.5f},{F[a][2]:+.5f})  "
              f"{fb}=({r[0]:+.5f},{r[1]:+.5f},{r[2]:+.5f})")
    print(results.result("obs", system, M=M, vs=best, ckpt="raw" if raw else "polished", E=E,
                         mu_D=float(np.linalg.norm(mu) * AU2DEBYE), theta_zz=float(theta[2, 2]),
                         nelec_grid=nelec, n_grid=int(rho.shape[0]),
                         rho_nuc=",".join(f"{v:.4f}" for v in rho_nuc),
                         dF_max=dF, dF_vs=fb or "none", maxF=float(np.abs(F).max()),
                         net_F=float(np.abs(F.sum(0)).max()), L1_rho=l1.get(best, float("nan")),
                         **{f"L1_{b[-3:]}": v for b, v in l1.items()}, **extra,
                         peak_gpu_mb=round(probes.peak_gpu_mb()), wall_s=round(time.time() - t0)),
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("polish", "obs"))
    ap.add_argument("system")
    ap.add_argument("M", type=int)
    ap.add_argument("--raw", action="store_true", help="obs: the ladder checkpoint, unpolished")
    a = ap.parse_args()
    if a.what == "polish":
        polish(a.system, a.M)
    else:
        obs(a.system, a.M, raw=a.raw)


if __name__ == "__main__":
    main()
