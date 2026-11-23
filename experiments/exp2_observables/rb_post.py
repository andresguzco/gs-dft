"""Small Figure 3 products from the reference archives, for the committed CSVs.

``grads <system>``: the Gaussian rungs' nuclear gradients as ``grad_<system>.csv`` (the columns of
``data/figures/fig3_gto_gradients.csv``), the top row's force baseline.

``ccfloor <system> [--dir D]``: the coupled-cluster floor of the density and force columns, the PBE
density and forces against CCSD's at the same basis (``cc_<system>_<basis>.npz`` and
``ref_<system>_<basis>.npz`` of ``cc_ref.py`` and ``gto_ref.py``, on the same grid): one
``RESULT kind=cc_floor`` with ``L1_rho`` and ``dF_max`` (max|F_CCSD - F_PBE|, Ha/Bohr).

    python -m experiments.exp2_observables.rb_post grads cubane
    python -m experiments.exp2_observables.rb_post ccfloor water
"""
import argparse
import csv
import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z")


def grads(system):
    from experiments.exp2_observables.selfref import ref_grad
    rows = []
    for b in ORDER:
        g = ref_grad(system, b)
        if g is None:
            continue
        rows += [dict(system=system, basis=b, atom=a, dE_dx=repr(float(g[a, 0])),
                      dE_dy=repr(float(g[a, 1])), dE_dz=repr(float(g[a, 2])))
                 for a in range(g.shape[0])]
    out = os.path.join(RES, f"grad_{system}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["system", "basis", "atom", "dE_dx", "dE_dy", "dE_dz"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows", flush=True)


def ccfloor(system, d, basis):
    from experiments.common import results
    cc = np.load(os.path.join(d, f"cc_{system}_{basis}.npz"), allow_pickle=True)
    ref = np.load(os.path.join(d, f"ref_{system}_{basis}.npz"), allow_pickle=True)
    if np.asarray(cc["rho"]).shape != np.asarray(ref["rho"]).shape:
        raise SystemExit("the CC and PBE densities are not on the same grid")
    from dftax.grid import becke_grid
    import jax.numpy as jnp
    from experiments.common import systems
    from gs_dft.ks.train import _GRID_LEVELS
    mol = systems.molecule(system=system, basis=basis)
    nr, lb = _GRID_LEVELS[5]
    _, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=nr, lebedev=lb)
    w = np.asarray(gw)
    l1 = float(np.dot(w, np.abs(np.asarray(cc["rho"]) - np.asarray(ref["rho"]))))
    gc, gp = np.asarray(cc["grad"]), np.asarray(ref["grad"])
    dF = float(np.abs(gc - gp).max()) if np.isfinite(gc).all() and np.isfinite(gp).all() else float("nan")
    print(results.result("cc_floor", system, basis=basis, wf="ccsd", L1_rho=l1,
                         L1_rho_vs_pbe_ccref=float(cc["L1_rho_vs_pbe"]), dF_max=dF,
                         nelec=int(mol.nelectron), n_grid=int(w.shape[0])), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("grads", "ccfloor"))
    ap.add_argument("system")
    ap.add_argument("--dir", default=RES)
    ap.add_argument("--basis", default="cc-pvqz")
    a = ap.parse_args()
    if a.what == "grads":
        grads(a.system)
    else:
        ccfloor(a.system, a.dir, a.basis)


if __name__ == "__main__":
    main()
