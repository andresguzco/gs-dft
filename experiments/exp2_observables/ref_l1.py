"""Density error of the Gaussian ladder against its own largest rung, for Figure 3's density panel.

``gto_ref.py`` stores each rung's density on the shared grid, so

    L1(X) = sum_g w_g |rho_X(g) - rho_top(g)|

is the same quantity, on the same points and against the same kind of reference, as the splat rows.

    uv run python -m experiments.exp2_observables.ref_l1 experiment=exp2_observables system=water
"""
import os

import hydra
import numpy as np
from dftax.grid import becke_grid

from experiments.common import results, systems
from experiments.exp2_observables.observables import _GRID_LEVELS, ORDER, RES


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    system = str(cfg.system)
    mol = systems.molecule(cfg)
    nr, lb = _GRID_LEVELS[int(cfg.obs_grid_level)]
    _, gw = becke_grid(list(mol.symbols), np.asarray(mol.atom_coords()), n_radial=nr, lebedev=lb)
    w = np.asarray(gw)

    refs = {}
    for b in ORDER:
        path = f"{RES}/ref_{system}_{b}.npz"
        if os.path.exists(path):
            refs[b] = np.load(path)
    if not refs:
        raise SystemExit(f"no ref_{system}_*.npz under {RES} — run gto_ref.py first")

    top = [b for b in ORDER if b in refs][-1]
    rho_top = np.asarray(refs[top]["rho"])
    print(f"exp2 ref-L1 {system}: {len(refs)} rungs, reference = {top}, "
          f"{w.shape[0]} grid points (level {int(cfg.obs_grid_level)})", flush=True)

    for b in ORDER:
        if b not in refs:
            continue
        l1 = float(np.dot(w, np.abs(np.asarray(refs[b]["rho"]) - rho_top)))
        print(results.result("gtol1", system, basis=b, nao=int(refs[b]["nao"]),
                             n_occ=int(refs[b]["n_occ"]), vs=top, L1_rho=l1), flush=True)


if __name__ == "__main__":
    main()
