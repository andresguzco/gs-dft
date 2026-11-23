"""GPU4PySCF PBE references for Figure 3 where dftax cannot hold the basis or its forces.

Writes the same ``ref_<system>_<basis>.npz`` as ``gto_ref.py`` (energy, orbital energies, nuclear
gradient, dipole, traceless quadrupole, kinetic energy, density at the nuclei and on the shared
reporting grid), so ``observables``, ``ref_l1``, ``selfref`` and ``rb_obs`` read either. Runs in the
GPU4PySCF environment (no JAX): the grid, geometry and charge come from ``obsgrid_<system>.npz``,
written by ``obs_grid.py`` from dftax's Becke grid, so the points are the splat side's.

The SCF is the rebuttal's (``experiments/rebuttal/gpu4pyscf_df.py``): DF first, then direct SCF from
the DF density, conv_tol 1e-9, PySCF grid level 3. The reference is the direct solution (the DF one if
direct does not converge). ``E`` is shifted onto the dftax scale by the system's dftax - GPU4PySCF
offset (``data/rebuttal/offsets.csv``), as for Figure 5; ``E_g4p`` keeps the raw value.

A basis with h functions (cc-pV5Z) is past GPU4PySCF's angular-momentum cap (g, ``GPU_LMAX``), so it
runs on the CPU with PySCF, density fitting only (the direct SCF does not fit there). It serves as the
density and force reference; its energy has no offset onto the dftax scale, so ``E`` is NaN (the raw
value is ``E_pyscf``) and the energy CBS keeps the smaller rungs.

    $ENV/bin/python experiments/exp2_observables/g4p_ref.py --system cubane --basis cc-pvtz
"""
import argparse
import csv
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OFFSETS = os.path.join(HERE, "..", "..", "data", "rebuttal", "offsets.csv")
AU2DEBYE = 2.541746473
GPU_LMAX = 4                # gpu4pyscf.scf.jk.LMAX (1.9.2): g functions
CARD = {"cc-pvdz": 2, "cc-pvtz": 3, "cc-pvqz": 4, "cc-pv5z": 5}


def offset(system):
    """``(Ha, basis)``: dftax - GPU4PySCF at the largest basis where both converged, from
    ``data/rebuttal/offsets.csv``; ``(0.0, "none")`` without an overlap."""
    with open(OFFSETS) as fh:
        rows = [r for r in csv.DictReader(fh) if r["system"] == system]
    if not rows:
        return 0.0, "none"
    r = max(rows, key=lambda r: CARD.get(r["basis"], 0))
    return float(r["offset_mha"]) * 1e-3, r["basis"]


def multipoles(coords_g, weights, rho, atom_coords, charges):
    """Dipole and traceless quadrupole from the gridded density; ``gto_ref.multipoles`` without JAX."""
    origin = (charges[:, None] * atom_coords).sum(0) / charges.sum()
    rn, rg = atom_coords - origin, coords_g - origin
    mu = (charges[:, None] * rn).sum(0) - ((weights * rho)[:, None] * rg).sum(0)

    def quad(r, q):
        rr = np.einsum("i,ia,ib->ab", q, r, r)
        return 0.5 * (3.0 * rr - np.trace(rr) * np.eye(3))
    return mu, quad(rn, charges) + quad(rg, -(weights * rho))


def density(mol, dm, points, gpu=True, chunk=16384):
    """rho on ``points``: PySCF's AO values (CPU), the contraction on the GPU (or the CPU)."""
    xp = __import__("cupy") if gpu else np
    dm_g = xp.asarray(dm)
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        ao = xp.asarray(mol.eval_gto("GTOval_sph", points[s:s + chunk]))
        out[s:s + chunk] = _np(xp.einsum("gi,gi->g", ao @ dm_g, ao))
    return out


def _np(x):
    return np.asarray(x.get() if hasattr(x, "get") else x)


def scf(mol, a, dm0=None, df=True, gpu=True):
    if gpu:
        from gpu4pyscf.dft import rks
    else:
        from pyscf.dft import rks
    mf = rks.RKS(mol, xc=a.xc)
    if df:
        mf = mf.density_fit(auxbasis=a.auxbasis)
    mf.grids.level = a.grid_level
    mf.conv_tol = a.conv_tol
    mf.max_cycle = a.max_cycle
    mf.kernel(dm0=dm0)
    return mf


def _fmt(v):
    """``experiments.common.results._fmt``, which this environment cannot import."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v != 0.0 and abs(v) < 1e-6:
            return f"{v:.2e}"
        return f"{v:.8f}" if abs(v) < 1e6 else f"{v:.8g}"
    return str(v)


def result(kind, system, **kv):
    return " ".join(["RESULT", f"kind={kind}", f"system={system}"]
                    + [f"{k}={_fmt(v)}" for k, v in kv.items()])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--out", default=RES)
    ap.add_argument("--grid", default=None, help="default <RES>/obsgrid_<system>.npz")
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--auxbasis", default=None, help="DF aux basis; default PySCF's choice")
    ap.add_argument("--grid_level", type=int, default=3)
    ap.add_argument("--conv_tol", type=float, default=1e-9)
    ap.add_argument("--max_cycle", type=int, default=200)
    ap.add_argument("--host_mem_mb", type=int, default=None,
                    help="PySCF's max_memory; default 85%% of the job's memory, else 100000")
    ap.add_argument("--compare", default=None, help="another ref npz on the same grid: print the "
                    "density L1, dE and max|dgrad| against it (the plumbing check)")
    ap.add_argument("--name", default="ref", help="writes <name>_<system>_<basis>.npz and a RESULT of "
                    "kind <name> (gto for ref); refgrad = the forces where dftax has none")
    a = ap.parse_args()

    import pyscf
    t0 = time.perf_counter()
    if a.host_mem_mb is None:
        a.host_mem_mb = int(0.85 * int(os.environ["SLURM_MEM_PER_NODE"])) \
            if os.environ.get("SLURM_MEM_PER_NODE") else 100000
    g = np.load(a.grid or os.path.join(RES, f"obsgrid_{a.system}.npz"))
    gc, gw = np.asarray(g["coords"]), np.asarray(g["weights"])
    coords, charges = np.asarray(g["atom_coords"]), np.asarray(g["charges"], float)
    symbols = [str(s) for s in g["symbols"]]
    mol = pyscf.M(atom=[(s, tuple(c)) for s, c in zip(symbols, coords)], basis=a.basis,
                  unit="Bohr", charge=int(g["charge"]), spin=0, verbose=4,
                  max_memory=a.host_mem_mb)
    if int(mol.nelectron) != int(g["nelectron"]):
        raise SystemExit(f"electron count {mol.nelectron} != dftax's {int(g['nelectron'])}")
    gpu = max(mol.bas_angular(i) for i in range(mol.nbas)) <= GPU_LMAX
    if gpu:
        import gpu4pyscf
        engine = f"gpu4pyscf {gpu4pyscf.__version__}"
    else:
        engine = f"CPU, {pyscf.lib.num_threads()} threads, max_memory {a.host_mem_mb} MB"
    print(f"# {a.system}/{a.basis}: nao={mol.nao} n_grid={len(gw)} {engine} pyscf {pyscf.__version__}",
          flush=True)

    mf_df = scf(mol, a, gpu=gpu)
    e_df, conv_df = float(mf_df.e_tot), bool(mf_df.converged)
    print(f"# DF: E={e_df:.10f} converged={conv_df} [{time.perf_counter() - t0:.0f}s]", flush=True)
    if not gpu:
        if not conv_df:
            raise SystemExit("the DF SCF did not converge")
        mf, src = mf_df, "pyscf_df"
    else:
        import cupy
        mf, src = scf(mol, a, dm0=mf_df.make_rdm1(), df=False), "g4p_direct"
        print(f"# direct: E={mf.e_tot:.10f} converged={mf.converged} "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)
        if mf.converged:
            del mf_df                               # its 3-index tensor is not needed again
            cupy.get_default_memory_pool().free_all_blocks()
        elif conv_df:
            mf, src = mf_df, "g4p_df"
        else:
            raise SystemExit("neither the DF nor the direct SCF converged")
    g_mf = mf.nuc_grad_method()
    if not gpu:
        # PySCF's DF gradient sizes its 3-index blocks from the free memory; past 2^31 entries (retinal
        # at cc-pV5Z with ~780 GB free) libcint's indices overflow and it segfaults. 32 GB free keeps
        # every block under 2^31 entries.
        g_mf.max_memory = pyscf.lib.current_memory()[0] + 32000
    grad = _np(g_mf.kernel())                                   # dE/dR, (n_atoms, 3)
    print(f"# gradient [{time.perf_counter() - t0:.0f}s]", flush=True)

    dm = _np(mf.make_rdm1())
    rho = density(mol, dm, gc, gpu)
    rho_nuc = density(mol, dm, coords, gpu)
    nelec = float(np.dot(gw, rho))
    mu, theta = multipoles(gc, gw, rho, coords, charges)
    origin = (charges[:, None] * coords).sum(0) / charges.sum()
    with mol.with_common_orig(origin):
        r_mats = mol.intor("int1e_r", comp=3)
    mu_analytic = (charges[:, None] * (coords - origin)).sum(0) - np.einsum("xij,ji->x", r_mats, dm)
    T = float(np.einsum("ij,ji->", mol.intor("int1e_kin"), dm))
    mo_energy = _np(mf.mo_energy).reshape(-1)
    n_occ = int(mol.nelectron // 2)
    e_raw = float(mf.e_tot)
    off, off_b = offset(a.system) if gpu else (float("nan"), "none")
    e = e_raw + off                                 # NaN on the CPU: no offset onto the dftax scale
    wall = time.perf_counter() - t0
    print(f"# grid electrons {nelec:.6f} (expect {mol.nelectron}); dipole grid vs analytic "
          f"{np.linalg.norm(mu - mu_analytic):.2e} au", flush=True)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.name}_{a.system}_{a.basis}.npz")
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, E=e, **{"E_g4p" if gpu else "E_pyscf": e_raw}, offset=off, nao=int(mol.nao), n_occ=n_occ,
             mo_energy=mo_energy, grad=grad, mu=mu, mu_analytic=mu_analytic, theta=theta, T=T,
             rho_nuc=rho_nuc, rho=rho, n_grid=int(gc.shape[0]), src=src)
    os.replace(tmp, path)
    if a.compare:
        r = np.load(a.compare)
        print(f"# vs {a.compare}: density L1 {np.dot(gw, np.abs(rho - r['rho'])):.3e} e, "
              f"dE(raw) {(e_raw - float(r['E'])) * 1e3:+.4f} mHa, max|dgrad| "
              f"{np.abs(grad - r['grad']).max():.2e} Ha/Bohr", flush=True)
    homo, lumo = mo_energy[n_occ - 1], mo_energy[n_occ]
    print(result("gto" if a.name == "ref" else a.name, a.system, basis=a.basis, nao=int(mol.nao), E=e, xc=a.xc.lower(),
                 conv=bool(mf.converged), src=src, **{"E_g4p" if gpu else "E_pyscf": e_raw},
                 offset_mha=off * 1e3,
                 offset_basis=off_b, E_df=e_df,
                 mu_D=float(np.linalg.norm(mu) * AU2DEBYE),
                 mu_grid_vs_analytic=float(np.linalg.norm(mu - mu_analytic)),
                 homo=float(homo), lumo=float(lumo), gap=float(lumo - homo), T=T, n_occ=n_occ,
                 n_grid=int(gc.shape[0]), nelec_grid=nelec, maxF=float(np.abs(grad).max()),
                 wall_s=round(wall)), flush=True)


if __name__ == "__main__":
    main()
