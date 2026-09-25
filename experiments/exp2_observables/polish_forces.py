"""Polish a Figure 3 checkpoint so its Hellmann-Feynman forces are converged.

F = -dE/dR at fixed splats is exact only where dE/dtheta = 0. This continues training through
low-learning-rate legs (1e-3, 3e-4, 1e-4), recording the theta-gradient norm and the force error
per leg, then applies an SCF polish of the coefficients and writes ``ckpt_<system>_polished.eqx``.

    uv run python -m experiments.exp2_observables.polish_forces <water|ethanol>
"""
import os
import sys
import time

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from experiments.common import systems, builders, results, probes, tracking
from experiments.exp2_observables.train_ckpt import RES, fresh_init, ckpt_path, GRID, AUX_MULT
from gs_dft.ks.energy import SplatKS, SplatState
from gs_dft.ks.terms import df
from gs_dft.integrals import dense as _full
from gs_dft.ks.train import train, evaluate, splat_adam
from gs_dft.ks.forces import splat_forces
from gs_dft.ks.orthonormalize import lowdin_orthonormalize
from experiments.exp2_observables.scf import scf_solve
from dftax.energy.xc import PBE
from dftax.grid import points

LEGS = [(2000, 1e-3), (2000, 3e-4), (2000, 1e-4)]
GRID_CHECK = 5


def diagnostics(model, aux, ef, F_ref):
    """(max|F − F_5Z|, θ-gradient RMS, C-gradient RMS) at the current parameters."""
    state = SplatState(aux=aux)
    F = np.asarray(splat_forces(ef, model, state))
    dF = float(np.abs(F - F_ref).max())
    g = eqx.filter_grad(lambda m: ef(m, state)[0])(model)
    g_theta = jnp.concatenate([g.basis.quat.ravel(), g.basis.log_scale.ravel(),
                               g.basis.centers.ravel()])
    return dF, float(jnp.sqrt(jnp.mean(g_theta ** 2))), float(jnp.sqrt(jnp.mean(g.C ** 2))), F


def force_reference(system):
    """The largest Gaussian rung that actually HAS a force, as (name, F_ref)."""
    for b in ("cc-pv5z", "cc-pvqz", "cc-pvtz", "cc-pvdz"):
        p = f"{RES}/ref_{system}_{b}.npz"
        if os.path.exists(p):
            g = np.load(p)["grad"]
            if np.isfinite(g).all():
                return b, -g
        ext = f"{RES}/refgrad_{system}_{b}.npz"
        if os.path.exists(ext):
            z = np.load(ext)
            g = np.asarray(z["grad"])
            if np.isfinite(g).all():
                print(f"force reference from {str(z['src'])} ({b}): dftax has none at this cardinal",
                      flush=True)
                return b, -g
    raise FileNotFoundError(
        f"no Gaussian reference with a finite force for {system}: the polish measures "
        f"max|F_splat - F_ref| and cannot run without one. Neither ref_*.npz nor refgrad_*.npz "
        f"carries a finite gradient at any cardinal -- run pyscf_forces.py for this system.")


def main(system, M=None):
    tag = f"{system}_M{M}" if M is not None else system
    tracking.init({}, __name__, name=f"forces_{tag}")      # not Hydra-configured; dict => defaults
    mol = systems.molecule(system=system, basis="cc-pvdz")
    model_t, aux_t = fresh_init(system, mol, M)
    model, aux = eqx.tree_deserialise_leaves(ckpt_path(system, M), (model_t, aux_t))
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), float)
    occ = model.occupations
    ks_train = builders.splat_ks(mol, PBE(), grid_level=GRID, aux_mult=AUX_MULT,
                                 screened=builders.auto_screen(int(model.basis.n_basis)))
    gp, gw = ks_train.grid_points, ks_train.grid_weights
    ef = SplatKS((coords, charges), PBE(), grid=points(gp, gw, chunk=4096), coulomb=df())
    fref_basis, F_ref = force_reference(system)
    print(f"force reference: {fref_basis}", flush=True)

    dF, gt, gc, _ = diagnostics(model, aux, ef, F_ref)
    print(results.result("forces", system, M=M, dF_vs=fref_basis, stage="ckpt", dF_max=dF,
                        gradtheta_rms=gt, gradC_rms=gc), flush=True)

    t0 = time.time()
    done = 0
    for steps, lr in LEGS:
        res = train(ks_train, model, steps=steps, optimizer=splat_adam(lr, steps),
                    monitor=None)
        model, aux = res.model, res.state.aux
        done += steps
        dF, gt, gc, _ = diagnostics(model, aux, ef, F_ref)
        print(results.result("forces", system, M=M, dF_vs=fref_basis, stage=f"+{done}@lr{lr:g}", dF_max=dF,
                            gradtheta_rms=gt, gradC_rms=gc, wall_s=round(time.time() - t0)),
              flush=True)

    # SCF polish of C at the frozen cloud, kept only if it lowers the DF objective
    S = _full.one_electron_integrals(model.basis, coords, charges)[0]
    res = scf_solve(model.basis, aux, coords, charges, gp, gw, occ, PBE(),
                    C0=lowdin_orthonormalize(model.C, S), max_iter=50)
    n_occ = occ.shape[0]
    D = jnp.diag(1.0 + jnp.linspace(0.0, 1e-3, n_occ))
    cand = eqx.tree_at(lambda m: m.C, model, res.C @ D)
    if float(ef(cand, SplatState(aux=aux))[0]) < float(ef(model, SplatState(aux=aux))[0]):
        model = cand
    # the SCF C-jump re-breaks θ-stationarity (∂E/∂θ ≠ 0 at the new C → force noise returns);
    # one short low-lr θ re-equilibration restores it — energy AND force polished together
    res = train(ks_train, model, steps=1000, optimizer=splat_adam(1e-4, 1000),
                monitor=None)
    model, aux = res.model, res.state.aux

    dF, gt, gc, F = diagnostics(model, aux, ef, F_ref)
    E = float(evaluate(ef, model))
    _gf = builders.grid_fields(mol, PBE(), model, GRID_CHECK, E)
    path = ckpt_path(system, M, polished=True)
    probes.atomic_save(path, (model, aux))
    print(results.result("forces", system, M=M, dF_vs=fref_basis, stage="final", E=E, dF_max=dF, **_gf,
                        gradtheta_rms=gt, gradC_rms=gc, wall_s=round(time.time() - t0), path=path),
          flush=True)
    for a in range(F.shape[0]):
        print(f"  F[{a}] splat=({F[a][0]:+.5f},{F[a][1]:+.5f},{F[a][2]:+.5f})  "
              f"5Z=({F_ref[a][0]:+.5f},{F_ref[a][1]:+.5f},{F_ref[a][2]:+.5f})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
