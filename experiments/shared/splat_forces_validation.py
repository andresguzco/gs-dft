"""Validate splat nuclear forces F = -dE/dR on a converged water model.

The splats do not move with the nuclei, so the force is the Hellmann-Feynman force, exact at the
variational minimum. The script trains a water model to convergence and checks:

  (a) autodiff against a fourth-order finite difference of E at fixed splats;
  (b) translational invariance, sum_A F_A = 0, which holds only at convergence;
  (c) the force against a finite difference of the re-optimized energy E*(R).

    uv run python -m experiments.shared.splat_forces_validation
"""
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
jax.config.update("jax_enable_x64", True)
from gs_dft.ks.train import native_grid

from gs_dft.benchmark import build_reference
from gs_dft import init_model
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.forces import splat_forces
from dftax.energy.xc import PBE

MOL = "h2o"
VARIANT = "splat"
STEPS = 2000
LR = 1e-2


def make_grid(mol, level=3):
    _g = native_grid(mol, level)
    return _g.coords, _g.weights


def train(model, coords, charges, gp, gw, steps, warm=None, lr=LR, label=""):
    """Plain-Adam variational training (exact full Coulomb) — returns (converged model, E)."""
    ef = SplatKS((coords, charges), PBE(), grid=(gp, gw))
    filt = jax.tree.map(lambda _: True, model)
    filt = eqx.tree_at(lambda m: m.occupations, filt, replace=False)
    tr, st = eqx.partition(model, filt)
    warm = warm if warm is not None else min(steps // 10, 200)
    sched = optax.join_schedules(
        [optax.linear_schedule(1e-4, lr, warm),
         optax.cosine_decay_schedule(lr, max(steps - warm, 1), alpha=0.01)], [warm])
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
        if label and (i % 400 == 0 or i == steps - 1):
            print(f"    [{label}] step {i:4d}  E={float(e):.6f}", flush=True)
    return eqx.combine(tr, st), float(e)


def fd_force_component(ef, model, a, i, eps=1e-3):
    """4th-order central FD of E at FIXED splats, atom a axis i."""
    def E_at(shift):
        c = ef.atom_coords.at[a, i].add(shift)
        return float(eqx.tree_at(lambda e: e.atom_coords, ef, c)(model)[0])
    return -(-E_at(2 * eps) + 8 * E_at(eps) - 8 * E_at(-eps) + E_at(-2 * eps)) / (12 * eps)


def main():
    mol, e_ref, nao, _ = build_reference(MOL, "cc-pvdz", "pbe", 3)
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
    gp, gw = make_grid(mol, 3)
    M = nao
    print(f"{MOL}  nao={nao}  variant={VARIANT}  M={M}  E_ref(cc-pVDZ)={e_ref:.6f}\n", flush=True)

    print(f"=== train converged {VARIANT}-covariance model ({STEPS} steps) ===", flush=True)
    t0 = time.time()
    model0 = init_model(mol, M, jr.PRNGKey(0))
    model, E = train(model0, coords, charges, gp, gw, STEPS, label="train")
    print(f"  converged E={E:.6f}  gap_vs_ref={(E - e_ref) * 1e3:+.1f} mHa  ({time.time()-t0:.0f}s)\n", flush=True)

    ef = SplatKS((coords, charges), PBE(), grid=(gp, gw))  # E_nn always analytic
    F = splat_forces(ef, model)
    print("Hellmann–Feynman forces F = −∂E/∂R  (Ha/Bohr):", flush=True)
    for a in range(coords.shape[0]):
        print(f"  atom {a} ({mol.symbols[a]}):  {F[a, 0]:+.5f}  {F[a, 1]:+.5f}  {F[a, 2]:+.5f}", flush=True)
    print(f"  |F| RMS={float(jnp.sqrt(jnp.mean(F**2))):.5f}  max={float(jnp.max(jnp.abs(F))):.5f}\n", flush=True)

    # --- (a) autodiff vs FD at fixed θ ---
    print("(a) autodiff vs 4th-order FD at fixed splats:", flush=True)
    F_fd = jnp.array([[fd_force_component(ef, model, a, i)
                       for i in range(3)] for a in range(coords.shape[0])])
    print(f"    max |F_autodiff − F_fd| = {float(jnp.max(jnp.abs(F - F_fd))):.2e}  (expect ~1e-7)\n", flush=True)

    # --- (b) translational invariance ---
    netF = jnp.sum(F, axis=0)
    print("(b) translational invariance  Σ_A F_A (≈0 at convergence):", flush=True)
    print(f"    Σ F = [{netF[0]:+.2e}, {netF[1]:+.2e}, {netF[2]:+.2e}]  |Σ F|={float(jnp.linalg.norm(netF)):.2e}\n", flush=True)

    # --- (c) Born–Oppenheimer test: HF force vs FD of the RE-OPTIMIZED energy ---
    print("(c) Born–Oppenheimer test — HF force vs central FD of re-optimized E*(R):", flush=True)
    a_bo, i_bo, d = 0, 2, 0.01            # oxygen, z-axis, 0.01 Bohr displacement
    Estar = {}
    for sgn in (+1, -1):
        c_disp = coords.at[a_bo, i_bo].add(sgn * d)
        m_warm = init_model(mol, M, jr.PRNGKey(0))
        m_warm = eqx.tree_at(lambda m: (m.basis, m.C), m_warm, (model.basis, model.C))  # warm-start θ*
        _, Estar[sgn] = train(m_warm, c_disp, charges, gp, gw, 800, warm=0, lr=2e-3,
                              label=f"reopt{sgn:+d}")
    F_bo = -(Estar[+1] - Estar[-1]) / (2 * d)
    print(f"    HF force  F[O,z] = {float(F[a_bo, i_bo]):+.5f}", flush=True)
    print(f"    BO force  F[O,z] = {F_bo:+.5f}   (from re-optimized E*±)", flush=True)
    print(f"    non-stationarity gap = {abs(float(F[a_bo, i_bo]) - F_bo) * 1e3:.2f} mHa/Bohr\n", flush=True)

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
