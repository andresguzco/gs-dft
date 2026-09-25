"""Check the floating-basis force law of Appendix A against the optimization residual.

The splat force F = -dE/dR is the partial derivative at fixed splat parameters theta. At the
minimum dE/dtheta = 0 and F is the Born-Oppenheimer force; away from it the error is bounded by
||dE/dtheta|| ||dtheta*/dR||. Three checks, none needing a reference:

  (a) autodiff F equals a fourth-order finite difference of E at fixed theta.
  (b) the net force |sum_A F_A|, zero for the exact force, falls with ||dE/dtheta|| along training.
  (c) one force component against a finite difference of the re-optimized energy E*(R).

    uv run python -m experiments.exp2_observables.forces_residual [water|ethanol]
"""
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from gs_dft.ks.train import native_grid

from gs_dft.benchmark import build_reference
from gs_dft import init_model
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.forces import splat_forces
from dftax.energy.xc import PBE

VARIANT = "full"          # exact full covariance — the production backend
MCFG = {"water": ("h2o", 48, 2000), "ethanol": ("ethanol", 72, 2500)}
LOG_EVERY = 100


def make_grid(mol, level=3):
    _g = native_grid(mol, level)
    return _g.coords, _g.weights


def _trainable(model):
    filt = jax.tree.map(lambda _: True, model)
    return eqx.tree_at(lambda m: m.occupations, filt, replace=False)


def residual_rms(ef, model):
    """RMS of the variational gradient dE/dth (basis + C, occupations frozen) = ||r||."""
    filt = _trainable(model)
    tr, st = eqx.partition(model, filt)
    g = eqx.filter_grad(lambda t: ef(eqx.combine(t, st))[0])(tr)
    v = jnp.concatenate([l.ravel() for l in jax.tree.leaves(g) if eqx.is_array(l)])
    return float(jnp.sqrt(jnp.mean(v ** 2)))


def fd_force_component(ef, model, a, i, eps=1e-3):
    """4th-order central FD of E at FIXED splats, atom a axis i (Ha/Bohr)."""
    def E_at(shift):
        c = ef.atom_coords.at[a, i].add(shift)
        return float(eqx.tree_at(lambda e: e.atom_coords, ef, c)(model)[0])
    return -(-E_at(2 * eps) + 8 * E_at(eps) - 8 * E_at(-eps) + E_at(-2 * eps)) / (12 * eps)


def train(ef, model, steps, lr, warm, sweep=None):
    """Plain-Adam exact-Coulomb training; if sweep is a list, append (step, E, ||r||, |sumF|) rows."""
    filt = _trainable(model)
    tr, st = eqx.partition(model, filt)
    sched = optax.join_schedules(
        [optax.linear_schedule(1e-4, lr, max(warm, 1)),
         optax.cosine_decay_schedule(lr, max(steps - warm, 1), alpha=0.01)], [max(warm, 1)])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    ostate = opt.init(tr)

    @eqx.filter_jit
    def step(tr, ostate):
        def loss(t):
            return ef(eqx.combine(t, st))[0]
        e, g = eqx.filter_value_and_grad(loss)(tr)
        upd, ostate = opt.update(g, ostate, tr)
        return eqx.apply_updates(tr, upd), ostate, e

    e = jnp.nan
    for i in range(steps):
        tr, ostate, e = step(tr, ostate)
        if sweep is not None and (i % LOG_EVERY == 0 or i == steps - 1):
            m = eqx.combine(tr, st)
            r = residual_rms(ef, m)
            sF = float(jnp.linalg.norm(jnp.sum(splat_forces(ef, m), axis=0)))
            sweep.append((i, float(e), r, sF))
            print(f"    step {i:4d}  E={float(e):.6f}  ||r||={r:.3e}  |sumF|={sF:.3e}", flush=True)
    return eqx.combine(tr, st), float(e)


def main(system="water"):
    name, M, steps = MCFG[system]
    mol, e_ref, nao, _ = build_reference(name, "cc-pvdz", "pbe", 3)
    coords = jnp.array(mol.atom_coords()); charges = jnp.array(mol.atom_charges(), float)
    gp, gw = make_grid(mol, 3)
    ef = SplatKS((coords, charges), PBE(), grid=(gp, gw))
    print(f"{system}  nao={nao}  M={M}  variant={VARIANT}  E_ref(cc-pVDZ)={e_ref:.6f}\n", flush=True)

    t0 = time.time()
    model = init_model(mol, M, jr.PRNGKey(0))
    sweep = []
    print(f"=== converge + log residual sweep ({steps} steps) ===", flush=True)
    model, E = train(ef, model, steps, lr=1e-2, warm=min(steps // 10, 300), sweep=sweep)
    print(f"  converged E={E:.6f}  gap_vs_ref={(E - e_ref)*1e3:+.1f} mHa  ({time.time()-t0:.0f}s)\n", flush=True)

    # (a) identity: autodiff vs 4th-order FD at fixed th
    F = splat_forces(ef, model)
    F_fd = jnp.array([[fd_force_component(ef, model, a, i) for i in range(3)]
                      for a in range(coords.shape[0])])
    id_err = float(jnp.max(jnp.abs(F - F_fd)))
    print(f"(a) max|F_autodiff - F_FD| at fixed th = {id_err:.2e}  (identity; expect ~1e-7)", flush=True)

    # (b) residual sweep summary: regress log|sumF| on log||r|| (expect slope ~1)
    rs = np.array([(r, s) for _, _, r, s in sweep if r > 0 and s > 0])
    slope, b = np.polyfit(np.log10(rs[:, 0]), np.log10(rs[:, 1]), 1)
    corr = float(np.corrcoef(np.log10(rs[:, 0]), np.log10(rs[:, 1]))[0, 1])
    print(f"(b) residual sweep: |sumF| ~ ||r||^{slope:.2f}  (Pearson log-log r={corr:.3f}; "
          f"||r||: {rs[:,0].max():.1e} -> {rs[:,0].min():.1e}, |sumF|: {rs[:,1].max():.1e} -> {rs[:,1].min():.1e})",
          flush=True)

    # (c) Born-Oppenheimer cross-check on one heavy-atom component
    a_bo, i_bo, d = 0, 2, 0.01
    Estar = {}
    for sgn in (+1, -1):
        c_disp = coords.at[a_bo, i_bo].add(sgn * d)
        ef_d = eqx.tree_at(lambda e: e.atom_coords, ef, c_disp)
        m_warm, _ = train(ef_d, model, 1000, lr=2e-3, warm=0)
        Estar[sgn] = float(ef_d(m_warm)[0])
    F_bo = -(Estar[+1] - Estar[-1]) / (2 * d)
    print(f"(c) BO cross-check atom{a_bo} axis{i_bo}: F_HF={float(F[a_bo,i_bo]):+.5f}  "
          f"F_BO={F_bo:+.5f}  gap={abs(float(F[a_bo,i_bo])-F_bo)*1e3:.2f} mHa/Bohr", flush=True)

    print(f"\nRESULT system={system} id_err={id_err:.2e} sweep_slope={slope:.3f} sweep_corr={corr:.3f} "
          f"bo_gap_mha={abs(float(F[a_bo,i_bo])-F_bo)*1e3:.3f} final_r={rs[-1,0]:.3e} final_sumF={rs[-1,1]:.3e}",
          flush=True)
    np.save(f"experiments/exp2_observables/forces_sweep_{system}.npy", rs)
    print(f"wrote forces_sweep_{system}.npy  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "water")
