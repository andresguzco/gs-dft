"""Peak GPU memory of dftax's materialized DF-RKS (the GTO references), per system and basis.

    python -m experiments.rebuttal.gto_memory [systems ...] [--bases cc-pvdz,cc-pvtz,...] [--gpus 4]

What dftax 0.8.0 holds (ks/energy.py, ks/terms.py, integrals/eri3c_bucketed.py):

- The 3-center tensor (mn|P), materialized FULL-SQUARE in the spherical spans: (nao, nao, naux)
  float64. df(screen=...) skips building screened pairs, but they stay allocated (as zeros).
- Its build, class by class (la, lb, lc): one block (ntrip, nca, ncb, ncc) in cartesian
  components with its gathered inputs, three einsums to spherical, and a scatter (plus a
  transposed copy for the bra swap). Peak = tensor + the largest class's chain.
- The metric (naux, naux) and its pseudo-inverse (an eigh), in the same jit as the tensor.
- J (PBE): gamma = (mn|P) P and J = (mn|P) V^-1 gamma, nothing above O(nao^2). K (hybrids only):
  a (nao, nocc, naux) intermediate through the occupied orbitals.
- The SCF: O(nao^2) matrices (S, H, F, P and a DIIS space of 8) and the XC grid streamed in
  chunks of 4096 points (AO values and gradients, cartesian, forward and backward).

df(chunk="auto") materializes only when nao^2 naux <= max(2^28, bytes_limit / 32) elements (a
quarter of the pool). Above it dftax falls back to slab-streamed RI-J, which rebuilds the
3-center integrals every SCF iteration; gto_ladder forces the materialized path instead when this
estimate fits. On a mesh (KS(mesh=mesh())) the tensor is built and held in aux slabs, 1/G each.

Class blocks are bounded by merging every contraction length of an (la, lb, lc) class, so the
build term is an upper bound. ``calibrate`` rows (measured peaks) live in data/gto_memory.json.
"""
import argparse
import json
import os
from collections import defaultdict

GIB = 2.0 ** 30
POOL_GIB = 0.92 * 79.2            # XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 of an A100/H100-80GB
FIT_MARGIN = 0.9                  # "fits" below 90% of the pool, "borderline" up to 100%
XC_CHUNK = 4096
DIIS = 8
PTS_PER_ATOM = 12000              # level-3 Becke grid, NWChem pruning (order of magnitude)
FLOOR_GIB = 0.5                   # measured: formic acid DZ peaks at 0.38 GB with a ~0 GiB tensor
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _shell_list(bd):
    from dftax.energy.gto import shell_records
    return [(l, npr) for (l, _row, _nc, npr) in shell_records(bd.angular, bd.exponents)]


def _ncart(l):
    return (l + 1) * (l + 2) // 2


def estimate(symbols, coords, basis, *, aux="def2-universal-jkfit", hybrid=False, nocc=0, gpus=1):
    """Bytes per component and the verdicts, for one molecule and orbital basis."""
    from dftax.basis.loader import build_basis_data
    bd = build_basis_data(symbols, coords, basis, spherical=True)
    ad = build_basis_data(symbols, coords, aux, spherical=True)
    nao, naux = int(bd.cart2sph.shape[1]), int(ad.cart2sph.shape[1])
    nao_cart = int(bd.centers.shape[0])
    bra, auxs = _shell_list(bd), _shell_list(ad)

    # ordered bra pairs (ia <= ib) per (la, lb), aux shells per lc, widest contraction per l
    pairs, cum = defaultdict(int), defaultdict(int)
    for lb, _ in bra:
        cum[lb] += 1
        for la, c in list(cum.items()):
            pairs[(la, lb)] += c
    naux_l = defaultdict(int)
    for lc, _ in auxs:
        naux_l[lc] += 1
    np_b, np_a = defaultdict(int), defaultdict(int)
    for l, n in bra:
        np_b[l] = max(np_b[l], n)
    for l, n in auxs:
        np_a[l] = max(np_a[l], n)

    worst = 0
    for (la, lb), npair in pairs.items():
        for lc, nc in naux_l.items():
            t = npair * nc
            ca, cb, cc = _ncart(la), _ncart(lb), _ncart(lc)
            sa, sb, sc = 2 * la + 1, 2 * lb + 1, 2 * lc + 1
            inputs = 9 + np_b[la] + np_b[lb] + np_a[lc] + ca * np_b[la] + cb * np_b[lb] + cc * np_a[lc]
            chain = max(ca * cb * cc + sa * cb * cc, sa * cb * cc + sa * sb * cc,
                        sa * sb * cc + sa * sb * sc, 2 * sa * sb * sc)
            worst = max(worst, t * (inputs + chain))
    b = {
        "int3c": 8.0 * nao * nao * naux,
        "build_class": 8.0 * worst,
        "metric": 8.0 * 3 * naux * naux,
        "scf": 8.0 * (2 * DIIS + 8) * nao * nao,
        "xc_chunk": 8.0 * 3 * 4 * XC_CHUNK * nao_cart,
        "grid": 8.0 * 4 * PTS_PER_ATOM * len(symbols),
        "k": 8.0 * 2 * nao * nocc * naux if hybrid else 0.0,
    }
    g = max(1, int(gpus))
    build = b["int3c"] / g + b["build_class"] / g + b["metric"] + b["grid"] / g
    run = b["int3c"] / g + b["metric"] / 3 + b["scf"] + b["xc_chunk"] + b["grid"] / g + b["k"] / g
    peak = max(build, run) + FLOOR_GIB * GIB
    pool = POOL_GIB * GIB
    auto_mat = g > 1 or nao * nao * naux <= max(2 ** 28, int(0.25 * pool) // 8)
    verdict = ("fits" if peak <= FIT_MARGIN * pool else "borderline" if peak <= pool else "no")
    return {"basis": basis, "gpus": g, "nao": nao, "naux": naux, "nao_cart": nao_cart,
            "int3c_gib": b["int3c"] / GIB, "build_class_gib": b["build_class"] / GIB,
            "peak_gib": peak / GIB, "verdict": verdict,
            "auto_path": "materialized" if auto_mat else "streamed"}


def table(names, bases, gpus_list=(1, 4)):
    from experiments.common.systems import molecule
    rows = []
    for name in names:
        for basis in bases:
            mol = molecule(system=name, basis=basis)
            sy = list(mol.symbols)
            for g in gpus_list:
                r = estimate(sy, mol.atom_coords(), basis, gpus=g,
                             nocc=int(mol.nelectron) // 2)
                r.update(system=name, n_atoms=len(sy))
                rows.append(r)
                print(f"{name:13s} {basis:8s} G={g} nao {r['nao']:6d} naux {r['naux']:6d}  "
                      f"int3c {r['int3c_gib']:9.1f} GiB  class {r['build_class_gib']:7.1f}  "
                      f"peak {r['peak_gib']:9.1f} GiB  {r['verdict']:10s} auto={r['auto_path']}",
                      flush=True)
    return rows


def main():
    from experiments.rebuttal.systems import RUNGS, BASES
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--gpus", default="1,4")
    ap.add_argument("--out", default=os.path.join(DATA, "gto_memory.json"))
    a = ap.parse_args()
    names = a.systems or [n for pair in RUNGS.values() for n in pair]
    rows = table(names, a.bases.split(","), tuple(int(x) for x in a.gpus.split(",")))
    old = {}
    if os.path.exists(a.out):
        with open(a.out) as fh:
            old = json.load(fh)
    old["estimate"] = rows
    old.setdefault("calibrate", [])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(old, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
