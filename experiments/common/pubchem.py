"""PubChem 3D conformers: fetch a small-molecule geometry and check its composition.

The fetched geometry is committed under ``geometry/`` with its PubChem CID, so runs need no network::

    uv run python -m experiments.common.pubchem penicillin
"""
from __future__ import annotations

import collections
import datetime
import os
import urllib.request

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
CITATION = ("PubChem, National Library of Medicine (NCBI). https://pubchem.ncbi.nlm.nih.gov/ — "
            "Kim et al., Nucleic Acids Res. 2023, 51, D1373. PubChem data are public domain.")
_GEOM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry")

SYSTEMS: dict[str, dict] = {
    # benzylpenicillin, the neutral acid. 41 atoms, closed shell. The ONLY second-row element (S)
    # anywhere in the exp2 accuracy ladder — that is deliberate, and it is the point of the system.
    "penicillin": {"cid": 5904, "formula": "C16H18N2O4S", "n_atoms": 41, "charge": 0, "spin": 0,
                   "note": "penicillin G / benzylpenicillin, neutral acid form"},
}


def _formula(comp: collections.Counter) -> str:
    """Hill ordering: C, H, then the rest alphabetically. Counts of 1 carry no digit."""
    def tok(e):
        return e if comp[e] == 1 else f"{e}{comp[e]}"
    rest = sorted(k for k in comp if k not in ("C", "H"))
    return "".join(tok(e) for e in ["C", "H"] if comp.get(e)) + "".join(tok(e) for e in rest)


def _parse_sdf(text: str) -> tuple[list[str], list[tuple[float, float, float]]]:
    """(symbols, coords_angstrom) from a V2000 SDF. RAISES on a planar (2-D) record.

    The counts line is line 4; atom blocks follow, ``x y z`` in fixed columns 0:10/10:20/20:30 and
    the element symbol at 31:34.
    """
    lines = text.splitlines()
    n = int(lines[3][0:3])
    syms, xyz = [], []
    for i in range(n):
        f = lines[4 + i]
        xyz.append((float(f[0:10]), float(f[10:20]), float(f[20:30])))
        syms.append(f[31:34].strip())
    if not any(abs(z) > 1e-6 for _, _, z in xyz):
        raise ValueError(
            "PubChem returned a PLANAR record — every z is 0, so this is the 2-D depiction, not a "
            "conformer. It has the right atom count and the right formula and would produce a "
            "meaningless energy. Request record_type=3d (and note that not every CID has one).")
    return syms, xyz


def fetch(name: str, *, out_dir: str | None = None, timeout: int = 60) -> tuple[str, dict]:
    """Download the 3-D conformer for a registered ``name``, assert it, write the .xyz.

    Returns ``(xyz_path, provenance)``. NETWORK REQUIRED — dev machines only; see the module note.
    """
    if name not in SYSTEMS:
        raise KeyError(f"{name!r} is not registered; add it to pubchem.SYSTEMS with its CID and "
                       f"expected formula (known: {', '.join(sorted(SYSTEMS))})")
    spec = SYSTEMS[name]
    url = f"{BASE}/{spec['cid']}/SDF?record_type=3d"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")

    syms, xyz = _parse_sdf(text)
    comp = collections.Counter(syms)
    got = _formula(comp)
    if got != spec["formula"] or len(syms) != spec["n_atoms"]:
        raise ValueError(
            f"PubChem CID {spec['cid']}: expected {spec['formula']} ({spec['n_atoms']} atoms), got "
            f"{got} ({len(syms)} atoms). Either the record was revised upstream or the CID is wrong. "
            f"Do NOT quote an energy against this geometry until resolved.")

    root = out_dir or _GEOM_DIR
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{name}.xyz")
    with open(path, "w") as fh:                      # bare element/x/y/z — the format systems.py reads
        for s, (x, y, z) in zip(syms, xyz):
            fh.write(f"{s:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
    return path, {"name": name, "cid": spec["cid"], "formula": got, "n_atoms": len(syms),
                  "charge": spec["charge"], "spin": spec["spin"],
                  "retrieved": datetime.date.today().isoformat(),
                  "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{spec['cid']}"}


def verify(name: str, path: str | None = None) -> dict:
    """Offline re-check of a COMMITTED .xyz against the registry. No network.

    This is what keeps the committed payload honest: the file is in git, so nothing would otherwise
    notice if it were edited, truncated, or swapped for a different conformer.
    """
    spec = SYSTEMS[name]
    path = path or os.path.join(_GEOM_DIR, f"{name}.xyz")
    syms, zs = [], []
    for line in open(path):
        if line.split():
            p = line.split()
            syms.append(p[0]); zs.append(float(p[3]))
    comp = collections.Counter(syms)
    got = _formula(comp)
    if got != spec["formula"] or len(syms) != spec["n_atoms"]:
        raise ValueError(f"{path}: expected {spec['formula']} ({spec['n_atoms']} atoms), "
                         f"got {got} ({len(syms)} atoms)")
    if not any(abs(z) > 1e-6 for z in zs):
        raise ValueError(f"{path}: PLANAR — every z is 0, so this is a 2-D depiction, not a conformer")
    return {"name": name, "formula": got, "n_atoms": len(syms), "path": path}


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1:] or sorted(SYSTEMS)):
        p, prov = fetch(nm)
        print(f"wrote {p}")
        for k, v in prov.items():
            print(f"  {k}: {v}")
        print(f"  citation: {CITATION}")
