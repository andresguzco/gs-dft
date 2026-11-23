"""The shared reporting grid of Figure 3, as an npz for engines without JAX (``g4p_ref.py``).

The same Becke grid ``gto_ref.py`` and the splat side evaluate densities on (``obs_grid_level`` 5),
with the geometry in Bohr, the symbols, nuclear charges, net charge and electron count, so the other
engine builds the identical molecule and reports on the identical points.

    python -m experiments.exp2_observables.obs_grid cubane
"""
import argparse
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OBS_LEVEL = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("system")
    ap.add_argument("--level", type=int, default=OBS_LEVEL)
    a = ap.parse_args()
    import jax.numpy as jnp
    from dftax.grid import becke_grid
    from experiments.common import systems
    from gs_dft.ks.train import _GRID_LEVELS

    mol = systems.molecule(system=a.system, basis="cc-pvdz")
    nr, lb = _GRID_LEVELS[a.level]
    gc, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=nr, lebedev=lb)
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"obsgrid_{a.system}.npz")
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, coords=np.asarray(gc), weights=np.asarray(gw),
             atom_coords=np.asarray(mol.atom_coords()), symbols=np.array(list(mol.symbols)),
             charges=np.asarray(mol.atom_charges(), dtype=float),
             charge=int(systems.charge(a.system)), nelectron=int(mol.nelectron), level=a.level)
    os.replace(tmp, path)
    print(f"wrote {path}: {gc.shape[0]} points (level {a.level}), {len(mol.symbols)} atoms, "
          f"{int(mol.nelectron)} electrons", flush=True)


if __name__ == "__main__":
    main()
