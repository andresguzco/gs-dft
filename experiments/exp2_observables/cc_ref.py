"""Coupled-cluster reference densities for water and ethanol.

The Figure 3 ladders are PBE, so the distance from a PBE number to a CCSD(T) number is functional
error, not representation error. The CC density shows what each observable converges toward.
The density is evaluated on the same grid as ``gto_ref.py``, and the run refuses to write a
reference if its PBE density disagrees with the committed ``ref_<system>_<basis>.npz`` by more
than ``RHO_L1_TOL``. ``corr=mp2`` writes ``mp2_<system>_<basis>.npz`` for systems where CCSD(T) is
too expensive.

    uv run --extra external python -m experiments.exp2_observables.cc_ref experiment=exp2_gto_ref system=water basis=cc-pvtz
"""
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from experiments.common import results, systems, tracking

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
AU2DEBYE = 2.541746473

#: Density-L1 between PySCF's PBE and dftax's PBE on the same grid, above which the two are not
#: solving the same problem. The QZ-5Z noise floor in `observables.py` is ~1e-2 e; this is tighter.
RHO_L1_TOL = 5.0e-3

#: Beyond this many basis functions, CCSD(T) is not attempted — see the module docstring.
MAX_NAO_CCSD_T = 260


def _multipoles(coords_g, weights, rho, atom_coords, charges):
    """Dipole and traceless quadrupole from a gridded density. Same convention as `gto_ref`."""
    origin = (charges[:, None] * atom_coords).sum(0) / charges.sum()
    d = coords_g - origin
    mu = (charges[:, None] * (atom_coords - origin)).sum(0) - np.einsum("g,g,ga->a", weights, rho, d)
    r2 = np.einsum("ga,ga->g", d, d)
    q_e = -np.einsum("g,g,ga,gb->ab", weights, rho, d, d)
    q_n = np.einsum("i,ia,ib->ab", charges, atom_coords - origin, atom_coords - origin)
    q = q_e + q_n
    tr = np.trace(q)
    return mu, 1.5 * q - 0.5 * tr * np.eye(3)


def _mp2(cfg, m, hf, ao, gc, gw, coords, charges, ref, l1_pbe, nao_, t0, numint):
    """The MP2 reference, written as `mp2_<sys>_<basis>.npz`."""
    from pyscf import mp as pmp
    # PySCF's MP2 gradient sizes its integral blocks from mol.max_memory (default 4 GB); on the
    # dipeptide that block size collapses to 0 ("cannot reshape array of size 0"). Give it the node.
    big = max(int(m.max_memory), int(cfg.get("max_memory_mb", 160_000)))
    m.max_memory = hf.max_memory = big          # the MP2 and its gradient copy max_memory at construction
    mp2 = pmp.MP2(hf)
    mp2.max_memory = big
    mp2.run()
    if not getattr(mp2, "converged", True):
        raise SystemExit("cc_ref REFUSES: MP2 did not converge.")
    dm = mp2.make_rdm1(ao_repr=True)
    rho = numint.eval_rho(m, ao, dm, xctype="LDA")
    nelec = float(np.dot(gw, rho))
    mu, theta = _multipoles(gc, gw, rho, coords, charges)
    try:
        g = mp2.nuc_grad_method()
        g.max_memory = big
        grad = np.asarray(g.kernel())
    except Exception as exc:                              # gradients are optional for a density ref
        import traceback
        print(f"  MP2 gradient unavailable ({type(exc).__name__}: {exc}); density only\n"
              + "".join(traceback.format_exc().splitlines(True)[-6:]), flush=True)
        grad = np.full((len(charges), 3), np.nan)
    l1_vs_pbe = float(np.dot(gw, np.abs(rho - np.asarray(ref["rho"]))))
    out = os.path.join(RES, f"mp2_{cfg.system}_{cfg.basis}.npz")
    np.savez(out, rho=rho, grad=grad, mu=mu, theta=theta, E_hf=hf.e_tot, E_mp2=mp2.e_tot,
             L1_rho_vs_pbe=l1_vs_pbe, gate_L1_pbe=l1_pbe, nao=nao_,
             n_occ=int(m.nelectron // 2), n_grid=gc.shape[0], src="pyscf-mp2", nelec=nelec)
    print(results.result("mp2", cfg.system, basis=cfg.basis, nao=nao_,
                         E_hf=float(hf.e_tot), E_mp2=float(mp2.e_tot),
                         mu_D=float(np.linalg.norm(mu) * AU2DEBYE),
                         dF_max=float(np.max(np.abs(grad))) if np.isfinite(grad).all() else float("nan"),
                         L1_rho_vs_pbe=l1_vs_pbe, nelec=nelec,
                         gate_L1_pbe=l1_pbe, wall_s=round(time.time() - t0)), flush=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from pyscf import gto as pgto, scf as pscf, dft as pdft, cc as pcc
    from pyscf.dft import numint
    import jax.numpy as jnp
    from dftax.grid import becke_grid
    from gs_dft.ks.train import _GRID_LEVELS

    tracking.init(cfg, __name__, name=f"cc_{cfg.system}_{cfg.basis}")
    mol_dftax = systems.molecule(cfg)
    coords = np.asarray(mol_dftax.atom_coords())          # Bohr
    charges = np.asarray(mol_dftax.atom_charges(), dtype=float)
    symbols = list(mol_dftax.symbols)

    m = pgto.M(atom=[(s, tuple(c)) for s, c in zip(symbols, coords)],
               basis=cfg.basis, unit="Bohr", charge=int(systems.charge(cfg.system)), verbose=0)
    nao_ = int(m.nao)

    # the SAME grid gto_ref reports on, so every L1 is grid-exact
    nr, lb = _GRID_LEVELS[int(cfg.obs_grid_level)]
    gc, gw = becke_grid(symbols, jnp.asarray(coords), n_radial=nr, lebedev=lb)
    gc, gw = np.asarray(gc, dtype=float), np.asarray(gw, dtype=float)
    ao = numint.eval_ao(m, gc, deriv=0)

    # ── GATE: same functional, same grid, two programs ──────────────────────────────────────────
    ref_path = os.path.join(RES, f"ref_{cfg.system}_{cfg.basis}.npz")
    if not os.path.exists(ref_path):
        raise SystemExit(f"cc_ref: no {ref_path}; run gto_ref for this rung first — the CC density "
                         f"is only trustworthy once the plumbing is checked against it.")
    ref = np.load(ref_path, allow_pickle=True)
    ks = pdft.RKS(m, xc="pbe").density_fit()
    ks.grids.level = int(cfg.grid_level)
    e_pbe = ks.kernel()
    rho_pbe = numint.eval_rho(m, ao, ks.make_rdm1(), xctype="LDA")
    l1_pbe = float(np.dot(gw, np.abs(rho_pbe - np.asarray(ref["rho"]))))
    print(f"  gate: PySCF PBE vs dftax PBE on the same grid — density L1 = {l1_pbe:.3e} e "
          f"(tol {RHO_L1_TOL:.0e}), dE = {(e_pbe - float(ref['E'])) * 1e3:+.3f} mHa", flush=True)
    if not np.isfinite(l1_pbe) or l1_pbe > RHO_L1_TOL:
        raise SystemExit(f"cc_ref REFUSES: PySCF and dftax disagree on the PBE density by "
                         f"L1 = {l1_pbe:.3e} e > {RHO_L1_TOL:.0e}. The grid, the AO evaluation or "
                         f"the geometry differs; a CC number computed here would inherit it.")

    # ── the reference itself ────────────────────────────────────────────────────────────────────
    t0 = time.time()
    hf = pscf.RHF(m).run()
    corr = str(cfg.get("corr", "ccsd_t")).lower()
    if corr not in ("ccsd_t", "mp2"):
        raise SystemExit(f"cc_ref: corr must be 'ccsd_t' or 'mp2', got {corr!r}")
    if corr == "mp2":
        return _mp2(cfg, m, hf, ao, gc, gw, coords, charges, ref, l1_pbe, nao_, t0, numint)
    cc = pcc.CCSD(hf).run()
    e_t = cc.ccsd_t() if nao_ <= MAX_NAO_CCSD_T else float("nan")
    dm_cc = cc.make_rdm1(ao_repr=True)                    # un-relaxed (Λ-CCSD) 1-RDM, AO basis
    if not getattr(cc, "converged_lambda", False):
        raise SystemExit("cc_ref REFUSES: the lambda equations did not converge, so the density and "
                         "every property derived from it are not the CC property they claim to be.")
    rho_cc = numint.eval_rho(m, ao, dm_cc, xctype="LDA")
    nelec_cc = float(np.dot(gw, rho_cc))

    mu, theta = _multipoles(gc, gw, rho_cc, coords, charges)
    grad = np.asarray(cc.nuc_grad_method().kernel())      # dE/dR for CCSD

    l1_vs_pbe = float(np.dot(gw, np.abs(rho_cc - np.asarray(ref["rho"]))))
    out = os.path.join(RES, f"cc_{cfg.system}_{cfg.basis}.npz")
    np.savez(out, rho=rho_cc, grad=grad, mu=mu, theta=theta, E_hf=hf.e_tot, E_ccsd=cc.e_tot,
             L1_rho_vs_pbe=l1_vs_pbe, gate_L1_pbe=l1_pbe,
             E_ccsd_t=cc.e_tot + (e_t if e_t == e_t else 0.0), nao=nao_, n_occ=int(m.nelectron // 2),
             n_grid=gc.shape[0], src="pyscf-ccsd", nelec=nelec_cc)
    print(results.result("cc", cfg.system, basis=cfg.basis, nao=nao_,
                         E_hf=float(hf.e_tot), E_ccsd=float(cc.e_tot),
                         E_ccsd_t=float(cc.e_tot + e_t) if e_t == e_t else float("nan"),
                         mu_D=float(np.linalg.norm(mu) * AU2DEBYE),
                         dF_max=float(np.max(np.abs(grad))),
                         L1_rho_vs_pbe=l1_vs_pbe, nelec=nelec_cc,
                         gate_L1_pbe=l1_pbe, wall_s=round(time.time() - t0)), flush=True)


if __name__ == "__main__":
    main()
