"""Gaussian force references from PySCF for alanine dipeptide.

Differentiating dftax's DF-RKS builds a three-center gradient tensor that does not fit for alanine
dipeptide, so its reference gradients come from PySCF's ``nuc_grad_method``. The gradient is written
only if the PySCF energy matches dftax's to ``e_tol``; with ``density_fit()`` and the same grid
level they agree to 0.04 mHa at cc-pVDZ. ``observables.py`` records the source as ``dF_src``.

    uv run --extra external python -m experiments.exp2_observables.pyscf_forces experiment=exp2_gto_ref system=alanine_dipeptide basis=cc-pvdz
"""
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from experiments.common import results, systems, tracking

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
E_TOL_HA = 5.0e-4          # 0.5 mHa: comfortably above the ~0.1 mHa RI-JK error, far below any claim


def _require_pyscf():
    try:
        from pyscf import dft, gto  # noqa: F401
    except ImportError as exc:                                    # pragma: no cover
        raise SystemExit("pyscf missing — run with `uv run --extra external`") from exc


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _require_pyscf()
    from pyscf import dft, gto as pyscf_gto

    system, basis = str(cfg.system), str(cfg.basis)
    tracking.init(cfg, __name__, name=f"pyscfF_{system}_{basis}")
    mol = systems.molecule(cfg)
    m = pyscf_gto.M(atom=[(s, tuple(c)) for s, c in zip(mol.symbols, mol.atom_coords())],
                    basis=basis, unit="Bohr", charge=systems.charge(system), spin=0)
    m.build()

    mf = dft.RKS(m, xc=str(cfg.xc)).density_fit()      # RI-JK, matching the dftax reference
    mf.grids.level = int(cfg.grid_level)               # must match or the comparison drifts
    t0 = time.perf_counter()
    e = float(mf.kernel())
    if not mf.converged:
        raise SystemExit(f"pyscf SCF did not converge for {system}/{basis}; no reference written")

    # --- the gate: same Hamiltonian, or no reference -----------------------------------------
    ref_npz = os.path.join(RES, f"ref_{system}_{basis}.npz")
    e_dftax = float(np.load(ref_npz)["E"]) if os.path.exists(ref_npz) else float("nan")
    if e_dftax == e_dftax and abs(e - e_dftax) > E_TOL_HA:
        raise SystemExit(
            f"REFUSING to write a force reference: pyscf E={e:.8f} vs dftax E={e_dftax:.8f} "
            f"differ by {1e3 * abs(e - e_dftax):.3f} mHa > {1e3 * E_TOL_HA:.1f} mHa. The two are "
            f"not evaluating the same Hamiltonian, so the gradient would not be a reference for "
            f"our force — check xc, grid_level and density_fit before trusting either.")

    grad = np.asarray(mf.nuc_grad_method().kernel())   # dE/dR, the same sign convention as gto_ref
    wall = time.perf_counter() - t0
    out = os.path.join(RES, f"refgrad_{system}_{basis}.npz")
    np.savez(out, grad=grad, E=e, E_dftax=e_dftax, nao=int(m.nao_nr()), src="pyscf",
             xc=str(cfg.xc), grid_level=int(cfg.grid_level))
    print(results.result("gto_force", system, basis=basis, nao=int(m.nao_nr()), src="pyscf",
                         xc=str(cfg.xc), E=e, dE_vs_dftax_mha=1e3 * (e - e_dftax),
                         max_grad=float(np.abs(grad).max()), wall_s=round(wall, 1), path=out),
          flush=True)


if __name__ == "__main__":
    main()
