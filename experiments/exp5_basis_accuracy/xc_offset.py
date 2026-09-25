"""Offset between the dftax and libxc implementations of PBE, on a converged reference density.

The splat and Gaussian energies share the dftax functional, so this bounds how much of a comparison
with an external code is the XC implementation rather than the basis.

    uv run python -m experiments.exp5_basis_accuracy.xc_offset [basis]
"""
import sys
from gs_dft.benchmark import build_reference, xc_consistency

BASIS = sys.argv[1] if len(sys.argv) > 1 else "cc-pvtz"


def main():
    print(f"XC offset δ_xc = E_xc(engine) − E_xc(libxc)  [on {BASIS} density, grid 3]\n")
    for s in ["h2o", "ethanol"]:
        mol, e_ref, nao, mf = build_reference(s, BASIS, "pbe", 3)
        exc_e, exc_p = xc_consistency(mol, mf, 3)
        print(f"  {s:8s}: δ_xc = {(exc_e - exc_p) * 1e3:+.4f} mHa   "
              f"(E_xc engine {exc_e:.6f} vs libxc {exc_p:.6f})", flush=True)
    print("\nSubtract δ_xc from the splat ΔE-from-CBS for an XC-matched comparison.")


if __name__ == "__main__":
    main()
