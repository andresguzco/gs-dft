"""Gaussian PBE reference observables for Figure 3 at one (system, basis), with DF-RKS on the same
functional, grid and auxiliary basis as the splat side.

Writes ``results/ref_<system>_<basis>.npz``: the energy, orbital energies, nuclear gradient, dipole,
traceless quadrupole (origin at the centre of nuclear charge), kinetic energy, the density at the
nuclei, and the density on the shared level-3 Becke grid, whose points are the same for every rung
and for the splat side.

    uv run python -m experiments.exp2_observables.gto_ref experiment=exp2_gto_ref system=water basis=cc-pv5z
"""
import os
import time

import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import numpy as np

from experiments.common import builders, systems, results, tracking, probes, griddensity
from dftax.ks import KS, scf, forces, df
from dftax.ks.energy import ao_on_grid
from dftax.basis.loader import build_basis_data
from dftax.grid import becke, becke_grid
from dftax.integrals.overlap import kinetic_matrix
from dftax.integrals.multipole import dipole_matrices
from gs_dft import nao
from gs_dft.ks.train import _GRID_LEVELS, native_grid

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
AU2DEBYE = 2.541746473


def multipoles(coords_g, weights, rho, atom_coords, charges):
    """Dipole vector + traceless quadrupole from a gridded density. Origin = center of nuclear
    charge (must match the splat side; quadrupole is origin-dependent when the dipole != 0)."""
    origin = (charges[:, None] * atom_coords).sum(0) / charges.sum()
    rn = atom_coords - origin
    rg = coords_g - origin
    mu = (charges[:, None] * rn).sum(0) - ((weights * rho)[:, None] * rg).sum(0)
    def quad(r, q):                                   # q: (+Z per nucleus) or (-w*rho per point)
        rr = np.einsum("i,ia,ib->ab", q, r, r)
        return 0.5 * (3.0 * rr - np.trace(rr) * np.eye(3))
    theta = quad(rn, charges) + quad(rg, -(weights * rho))
    return mu, theta


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"gto_{cfg.system}_{cfg.basis}")
    system, basis = cfg.system, cfg.basis
    mol = systems.molecule(cfg)
    coulomb = df("def2-universal-jkfit")
    # From the config, never an import: pinned to PBE, `xc=r2scan` composed fine and produced a
    # PBE reference for an r2SCAN cloud. `builders.xc_of` is the single reader.
    xc = builders.xc_of(cfg.xc)
    if builders.has_nlc(xc):
        builders.use_streamed_vv10()   # else the SCF's own ∂E/∂ρ asks for 684 GiB
    # A concrete chunked points object, not a bare Becke spec: the spec makes KS materialize a
    # >140 GB intermediate on ethanol and up. VV10's pair quadrature is nonlocal across chunks, so a
    # "-V" functional must take the materialized grid -- which is why `xc` is read first.
    grid = native_grid(mol, cfg.grid_level,
                       chunk=None if builders.has_nlc(xc) else 4096)
    ks = KS(mol, xc, grid=grid, coulomb=coulomb)

    t0 = time.perf_counter()
    res = scf(ks, max_iter=100)
    if not res.converged:
        res = scf(ks, max_iter=200, level_shift=0.2)
    e = float(res.e_tot)
    n_occ = int(mol.nelectron // 2)
    Ptot = jnp.sum(res.P, axis=0)                     # spin-summed density matrix (nao, nao)

    coords = np.asarray(mol.atom_coords())
    charges = np.asarray(mol.atom_charges(), dtype=float)
    basis_data = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)

    # Density on the basis-independent REPORTING grid, which must NOT be the training grid: a
    # floating basis can exploit the quadrature it trained on. `observables.py` evaluates the splat
    # density on this same grid, so change both together or the L1 compares different points.
    nr, lb = _GRID_LEVELS[int(cfg.obs_grid_level)]
    gc, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=nr, lebedev=lb)
    # `gto_density`, not `ao_on_grid`: the latter builds a (ngrid, nao, 3) Jacobian before the
    # caller drops it -- 68 GiB in one request on ethanol/cc-pV5Z, for a quantity nothing reads.
    rho = griddensity.gto_density(basis_data, Ptot, gc)
    rho_nuc = griddensity.gto_density(basis_data, Ptot, jnp.asarray(coords))
    gc_np, gw_np = np.asarray(gc), np.asarray(gw)

    mu, theta = multipoles(gc_np, gw_np, rho, coords, charges)
    # analytic dipole cross-check: same converged density, operator expectation (not gridded)
    origin = (charges[:, None] * coords).sum(0) / charges.sum()
    r_mats = dipole_matrices(basis_data, origin=tuple(origin))          # (3, nao, nao)
    mu_elec = -np.einsum("mn,anm->a", np.asarray(Ptot), np.asarray(r_mats))
    mu_analytic = (charges[:, None] * (coords - origin)).sum(0) + mu_elec
    T = float(jnp.sum(Ptot * kinetic_matrix(basis_data)))               # <T> = Tr(P·T)
    # Nuclear forces on the default materialized DF backend; the exact four-center force tensor does
    # not fit in memory. Above `force_max_nao` the force is skipped, because the three-center
    # gradient can exhaust host memory and kill the process before the energy is written.
    force_max_nao = int(cfg.get("force_max_nao", 200))
    grad = None
    if int(nao(mol)) > force_max_nao:
        print(f"  forces SKIPPED: nao={int(nao(mol))} > force_max_nao={force_max_nao}; the "
              f"materialized int3c gradient is SIGKILLed at this size and would take the "
              f"energy and dipole with it", flush=True)
    else:
        try:
            grid_f = becke(*_GRID_LEVELS.get(cfg.grid_level, _GRID_LEVELS[3]),
                           **({} if builders.has_nlc(xc) else {'chunk': 2048}))
            # The SCF's OWN `coulomb`, not a fresh `df()`: a force must differentiate the energy
            # that produced the density, and a bare df() stops matching the moment the auxbasis
            # above changes. Geometry gradients need the materialized backend.
            grad = -np.asarray(forces(mol, xc, res, grid=grid_f, coulomb=coulomb))   # dE/dR = -force
        except Exception as ex:                            # RESOURCE_EXHAUSTED if the int3c grad won't fit
            grad = None
            print(f"  forces UNAVAILABLE ({type(ex).__name__}): materialized DF int3c grad too large "
                  f"for this (system,basis) — skipped", flush=True)
    mo_energy = np.asarray(res.mo_energy).reshape(-1)
    wall = time.perf_counter() - t0

    os.makedirs(RES, exist_ok=True)
    np.savez(f"{RES}/ref_{system}_{basis}.npz",
             E=e, nao=nao(mol), n_occ=n_occ, mo_energy=mo_energy,
             grad=(grad if grad is not None else np.full((len(coords), 3), np.nan)),
             mu=mu, mu_analytic=mu_analytic, theta=theta, T=T, rho_nuc=rho_nuc,
             rho=rho, n_grid=gc_np.shape[0])
    homo, lumo = mo_energy[n_occ - 1], mo_energy[n_occ]
    # The reference values also go in the RESULT row, except `rho` (about 1e5 points), which only the
    # npz holds.
    print(results.result("gto", system, basis=basis, nao=int(nao(mol)), E=e, xc=str(cfg.xc),
                        conv=bool(res.converged), mu_D=float(np.linalg.norm(mu) * AU2DEBYE),
                        mu_grid_vs_analytic=float(np.linalg.norm(mu - mu_analytic)),
                        homo=float(homo), lumo=float(lumo), gap=float(lumo - homo), T=T,
                        n_occ=int(n_occ), n_grid=int(gc_np.shape[0]),
                        occ_mo=",".join(f"{v:.8f}" for v in mo_energy[:n_occ]),
                        rho_nuc=",".join(f"{v:.6f}" for v in np.asarray(rho_nuc).reshape(-1)),
                        maxF=(float(np.abs(grad).max()) if grad is not None else float("nan")),
                        # Same probe/units/field as the splat call sites, so peak memory is a
                        # usable comparison axis rather than one representation and a gap.
                        peak_gpu_mb=round(probes.peak_gpu_mb()),
                        wall_s=round(wall)), flush=True)


if __name__ == "__main__":
    main()
