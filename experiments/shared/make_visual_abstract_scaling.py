"""Rebuild ``data/figures/fig1_scaling.csv`` for Figure 1 from the Figure 5 data.

    uv run python -m experiments.shared.make_visual_abstract_scaling
"""
import csv
import os

import numpy as np

from experiments.common import ladder, resources, systems

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "data", "figures", "fig5_isoparams.csv")
SYSTEM, NOCC, BUDGET = "alanine_dipeptide", 39, 12000


def _basis_data(nao):
    from dftax.basis.loader import build_basis_data
    from gs_dft import nao as nao_of
    for b in ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"):
        mol = systems.molecule(system=SYSTEM, basis=b)
        if int(nao_of(mol)) == int(nao):
            return build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    raise KeyError(f"no cardinal with nao={nao}")


def main():
    data = ladder.load(RESULTS, budget=BUDGET)
    e = data[SYSTEM]
    gl = ladder.gto_ladder(e)
    gx = np.array([resources.gto_params(_basis_data(B), B, NOCC)["total"] for B, _ in gl], float)
    gy = np.array([g for _, g in gl], float)                    # already (E - gto CBS)*1e3

    raw = ladder.splat_ladder(e, require_uniform_seeds=True)
    Ms = sorted(raw)
    sx = np.array([resources.splat_params(M, NOCC)["total"] for M in Ms], float)
    px = {M: x for M, x in zip(Ms, sx)}
    fit = ladder.cbs_splat(e, xkey=px.get)                      # the splats' OWN limit
    shift = (e["cbs"] - fit["value"]) * 1e3
    sy = np.array([raw[M][1] + shift for M in Ms], float)       # best seed, vs the splat limit

    if (sy <= 0).any() or (gy <= 0).any():
        raise SystemExit(f"non-positive residual: gy={gy} sy={sy}")
    a_g = float(np.polyfit(np.log10(gx), np.log10(gy), 1)[0])
    a_s = float(np.polyfit(np.log10(sx), np.log10(sy), 1)[0])

    out = os.path.join(HERE, "..", "..", "data", "figures", "fig1_scaling.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["ladder", "params", "error_mha"])
        w.writerows([["gto", repr(float(x)), repr(float(y))] for x, y in zip(gx, gy)])
        w.writerows([["splat", repr(float(x)), repr(float(y))] for x, y in zip(sx, sy)])
    print(f"wrote {out}")
    print(f"  splat E_inf {fit['value']:.6f}  p={fit['p']:.3f}  shift {shift:+.3f} mHa")
    print(f"  gy {np.round(gy, 3)}   a_g={a_g:.3f}")
    print(f"  sy {np.round(sy, 3)}   a_s={a_s:.3f}")


if __name__ == "__main__":
    main()
