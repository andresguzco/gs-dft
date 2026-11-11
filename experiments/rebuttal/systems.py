"""The rebuttal's systems: candidates per size rung, the source of each geometry, and a check.

    python -m experiments.rebuttal.systems [--out experiments/rebuttal/data/systems.json]

The check reads every geometry, asserts its atom count and formula, rejects any bond shorter than
0.9 Å between two heavy atoms (and any contact under 0.7 Å), and counts n_ao for cc-pVDZ…cc-pV5Z
and the def2-universal-jkfit auxiliary size. It imports dftax, so it runs on a compute node
(JAX_PLATFORMS=cpu is enough). The JSON it writes feeds the run-table generator and the GTO memory
estimate, which then run anywhere.

Every geometry is a bare xyz in ``experiments/common/geometry/``; ``experiments.common.systems``
resolves it by name, neutral and closed shell.
"""
import argparse
import collections
import json
import os

BASES = ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z")
AUX = "def2-universal-jkfit"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "systems.json")

# Size rung -> the two candidates (one is used per rung).
RUNGS = {
    "7": ("sf6",),
    "16": ("cubane",),
    "30": ("adenine_thymine",),
    "38": ("porphine",),
    "49": ("retinal",),
    "74": ("cholesterol",),
    "100-200": ("chignolin", "cyclosporin"),
}

# Geometries not fetched by experiments.common.pubchem (no PubChem 3-D conformer past 50 heavy
# atoms); built by experiments/rebuttal/build_geometry.py.
BUILT = {
    "sf6": {"formula": "F6S", "n_atoms": 7,
            "source": "octahedral, r(S-F) = 1.561 A (gas-phase electron diffraction)"},
    "adenine_thymine": {"formula": "C10H11N7O2", "n_atoms": 30,
                        "source": "S22 complex 7, adenine-thymine Watson-Crick (Jurecka et al. 2006), "
                                  "from Psi4's S22 database"},
    "chignolin": {"formula": "C62H75N11O20", "n_atoms": 168,
                  "source": "PDB 5AWL (CLN025, X-ray 1.11 A) + H, neutral; RDKit MMFF94 on H only"},
    "cyclosporin": {"formula": "C62H111N11O12", "n_atoms": 196,
                    "source": "PubChem CID 5284373 SMILES; RDKit ETKDGv3 + MMFF94, lowest of 30"},
}


def spec(name):
    """``{"formula", "n_atoms", "source"}`` for a registered system."""
    if name in BUILT:
        return BUILT[name]
    from experiments.common import pubchem
    s = pubchem.SYSTEMS[name]
    return {"formula": s["formula"], "n_atoms": s["n_atoms"],
            "source": f"PubChem CID {s['cid']} 3-D conformer"}


_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16}


def _formula(syms):
    c = collections.Counter(syms)
    tok = lambda e: e if c[e] == 1 else f"{e}{c[e]}"
    rest = sorted(k for k in c if k not in ("C", "H"))
    return "".join(tok(e) for e in ("C", "H") if c.get(e)) + "".join(tok(e) for e in rest)


def check_geometry(name):
    """Atom count, formula and short contacts. Raises on any failure."""
    import numpy as np
    from experiments.common.systems import geometry
    lines = [ln.split() for ln in geometry(name).splitlines() if ln.split()]
    syms = [p[0] for p in lines]
    xyz = np.array([[float(v) for v in p[1:4]] for p in lines])
    want = spec(name)
    if len(syms) != want["n_atoms"] or _formula(syms) != want["formula"]:
        raise ValueError(f"{name}: got {_formula(syms)} ({len(syms)} atoms), "
                         f"want {want['formula']} ({want['n_atoms']})")
    if not np.any(np.abs(xyz[:, 2]) > 1e-6):
        raise ValueError(f"{name}: planar (z = 0), a 2-D depiction")
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) + np.eye(len(syms)) * 1e9
    heavy = np.array([s != "H" for s in syms])
    hh = d[np.ix_(heavy, heavy)].min() if heavy.sum() > 1 else float("inf")
    if hh < 0.9 or d.min() < 0.7:
        raise ValueError(f"{name}: short contact (heavy-heavy {hh:.3f} A, any {d.min():.3f} A)")
    return {"n_atoms": len(syms), "formula": _formula(syms), "n_heavy": int(heavy.sum()),
            "min_heavy_heavy_A": round(float(hh), 4), "min_any_A": round(float(d.min()), 4),
            "nelec": int(sum(_Z[s] for s in syms))}


def counts(name):
    """n_ao per basis (spherical) and the def2-universal-jkfit size, both spans."""
    from dftax.basis.loader import build_basis_data
    from experiments.common.systems import molecule
    from gs_dft import nao
    out = {"nao": {b: int(nao(molecule(system=name, basis=b))) for b in BASES}}
    mol = molecule(system=name, basis=BASES[0])
    aux_s = build_basis_data(list(mol.symbols), mol.atom_coords(), AUX, spherical=True)
    aux_c = build_basis_data(list(mol.symbols), mol.atom_coords(), AUX)
    out["naux_sph"] = int(aux_s.cart2sph.shape[1])
    out["naux_cart"] = int(aux_c.centers.shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA)
    ap.add_argument("systems", nargs="*")
    a = ap.parse_args()
    names = a.systems or [n for pair in RUNGS.values() for n in pair]
    table = {}
    for name in names:
        row = {"rung": next(r for r, pair in RUNGS.items() if name in pair), **spec(name),
               **check_geometry(name), **counts(name)}
        table[name] = row
        n = row["nao"]
        print(f"{name:13s} {row['n_atoms']:4d} {row['formula']:15s} "
              f"DZ {n['cc-pvdz']:5d} TZ {n['cc-pvtz']:5d} QZ {n['cc-pvqz']:5d} "
              f"5Z {n['cc-pv5z']:6d} naux {row['naux_sph']:6d}  "
              f"minHH {row['min_heavy_heavy_A']:.3f}  {row['source']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(table, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
