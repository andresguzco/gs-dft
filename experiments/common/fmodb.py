"""RIKEN FMODB: fetch a protein structure and its published FMO energies, with provenance.

FMODB (https://drugdesign.riken.jp/FMODB/) publishes FMO calculations on biomacromolecules. The
structures of Table 3 come from it, with their composition and charge checked on download, and its
FMO2-HF energies are the reference column of that table. The structures are CC BY-SA 4.0 and are
fetched rather than committed::

    uv run python -m experiments.common.fmodb --materialize        # all five proteins

    from experiments.common import fmodb
    pdb_path, meta = fmodb.fetch("25J8R")                          # cached after the first call
"""
from __future__ import annotations

import collections
import datetime
import os
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, asdict

BASE = "https://drugdesign.riken.jp/FMODB"
CITATION = ("FMODB, FMO Drug Design Consortium (FMODD), https://drugdesign.riken.jp/FMODB/ — "
            "CC BY-SA 4.0. Takaya et al., J. Chem. Inf. Model. 2021, 61, 777.")
_CACHE = os.environ.get(
    "FMODB_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fmodb_cache"))


@dataclass(frozen=True)
class FmodbEntry:
    """Everything needed to defend a number that came from this structure."""
    fmodb_id: str
    read_geom: str            # the file the calculation ACTUALLY consumed (.ajf ReadGeom)
    method: str               # .ajf Method
    basis: str                # as the log reports it
    nao: int | None           # NUMBER OF BASIS FUNCTIONS, at `basis`
    charge: int | None        # CHARGE OF MOLECULE — from the LOG; nothing else publishes it
    nelec: int | None         # ALL ELECTRON — the authoritative count, cross-checked against Z-sum
    n_atoms: int
    formula: str
    composition: dict
    e_hf: float | None        # FMO2-HF total
    e_mp2: float | None       # FMO2-MP2 total (plain, NOT partially renormalised)
    retrieved: str
    source_url: str
    licence: str = "CC BY-SA 4.0"

    def as_dict(self):
        return asdict(self)


SYSTEMS: dict[str, dict] = {
    # The five proteins of Table 3. `charge` is FMODB's `CHARGE OF MOLECULE`; every entry is
    # closed-shell (sum(Z) - charge equals the published electron count and is even).
    "PJM49": {"formula": "C98H150N27O29", "charge": 1,
              "note": "Trp-cage (PDB 1L2Y), 304 atoms, nelec=1158, nao(cc-pVTZ)=6720"},
    "9LQN2": {"formula": "C151H224N44O40S6", "charge": 2,
              "note": "defensin HNP-3, 465 atoms, nelec=1852, nao(cc-pVTZ)=10390"},
    "LG5K9": {"formula": "C306H463N84O88S6", "charge": 1,
              "note": "insulin, 947 atoms, nelec=3686, nao(cc-pVTZ)=21026"},
    "38KNL": {"formula": "C571H867N151O162S4", "charge": 0,
              "note": "calponin-homology domain of human beta-spectrin (PDB 1BKR chain A), "
                      "1755 atoms, nelec=6710, nao(cc-pVTZ)=38794"},
    "25J8R": {"formula": "C878H1375N240O237S12", "charge": 5,
              "note": "3IFU chain A, 2742 atoms, nelec=10406, nao(cc-pVTZ)=60308"},
}


def _formula(comp: dict) -> str:
    """Hill-ish ordering: C, H, then the rest alphabetically."""
    parts = []
    for e in ("C", "H"):
        if comp.get(e):
            parts.append(f"{e}{comp[e]}")
    for e in sorted(k for k in comp if k not in ("C", "H")):
        parts.append(f"{e}{comp[e]}")
    return "".join(parts)


def parse_pdb_composition(text: str) -> tuple[int, dict]:
    """(n_atoms, {element: count}) from ATOM/HETATM records."""
    comp: collections.Counter = collections.Counter()
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            el = line[76:78].strip()
            if not el:                                    # pre-v2.3 files omit the element column
                el = re.sub(r"[^A-Za-z]", "", line[12:16].strip())[:1]
            comp[el.capitalize()] += 1
    return sum(comp.values()), dict(comp)


def _parse_ajf(text: str) -> dict:
    def g(key):
        m = re.search(rf"^\s*{key}\s*=\s*'([^']*)'", text, re.M | re.I)
        return m.group(1) if m else None
    return {"read_geom": g("ReadGeom"), "method": g("Method"), "basis": g("BasisSet")}


def _parse_log(text: str) -> dict:
    """Basis, AO count and the two PLAIN FMO totals."""
    out: dict = {"basis": None, "nao": None, "e_hf": None, "e_mp2": None,
                 "charge": None, "nelec": None}
    m = re.search(r"^\s*BASIS SET\s*=\s*(\S+)", text, re.M)
    if m:
        out["basis"] = m.group(1)
    m = re.search(r"NUMBER OF BASIS FUNCTIONS\s*=\s*(\d+)", text)
    if m:
        out["nao"] = int(m.group(1))
    m = re.search(r"CHARGE OF MOLECULE\s*=\s*(-?\d+)", text)
    if m:
        out["charge"] = int(m.group(1))
    m = re.search(r"ALL ELECTRON\s*=\s*(\d+)", text)
    if m:
        out["nelec"] = int(m.group(1))

    i = text.find("## FMO TOTAL ENERGY")
    if i < 0:
        return out
    block = text[i:]
    cut = block.find("Partially renormalized")
    if cut > 0:
        block = block[:cut]
    tot = [float(x) for x in re.findall(r"Total\s+energy\s*=\s*(-?\d+\.\d+)", block)]
    if tot:
        out["e_hf"] = tot[0]                              # the FMO2-HF block comes first
    if len(tot) > 1:
        out["e_mp2"] = tot[1]                             # then plain FMO2-MP2
    return out


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)                                 # atomic: a half-file is never cached


def fetch(fmodb_id: str, *, cache_dir: str | None = None,
          expect_formula: str | None = "auto") -> tuple[str, FmodbEntry]:
    """Download (once) and parse an FMODB entry. Returns ``(pdb_path, FmodbEntry)``.

    ``expect_formula='auto'`` uses ``SYSTEMS[fmodb_id]['formula']`` when present; pass an explicit
    string to assert one, or ``None`` to skip. A mismatch RAISES — that assertion is the whole point
    of this module.
    """
    fmodb_id = fmodb_id.strip().upper()
    root = cache_dir or _CACHE
    zpath = os.path.join(root, fmodb_id, f"{fmodb_id}.zip")
    url = f"{BASE}/data_download/{fmodb_id}/compressed/{fmodb_id}.zip"
    if not os.path.exists(zpath):
        _download(url, zpath)

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        ajf = next((n for n in names if n.endswith(".ajf")), None)
        log = next((n for n in names if n.endswith(".log") and "IFIE" not in n
                    and "PIEDA" not in n and "charge" not in n), None)
        meta = _parse_ajf(z.read(ajf).decode("utf-8", "replace")) if ajf else {}
        lg = _parse_log(z.read(log).decode("utf-8", "replace")) if log else {}
        # the structure the calculation actually consumed, not merely "a pdb in the zip"
        want = meta.get("read_geom")
        pdb_name = want if want in names else next(n for n in names if n.endswith(".pdb"))
        pdb_text = z.read(pdb_name).decode("utf-8", "replace")
        pdb_path = os.path.join(root, fmodb_id, os.path.basename(pdb_name))
        if not os.path.exists(pdb_path):
            with open(pdb_path, "w") as f:
                f.write(pdb_text)

    n_atoms, comp = parse_pdb_composition(pdb_text)
    _Z = {"H":1,"C":6,"N":7,"O":8,"S":16,"P":15,"F":9,"Cl":17,"Se":34,"Zn":30,"Fe":26,"Mg":12,
          "Ca":20,"Na":11,"K":19,"Mn":25,"Cu":29}
    z_sum = sum(_Z.get(e, 0) * n for e, n in comp.items())
    q, ne = lg.get("charge"), lg.get("nelec")
    if q is not None and ne is not None and all(e in _Z for e in comp):
        if z_sum - q != ne:
            raise ValueError(
                f"FMODB {fmodb_id}: composition/charge INCONSISTENT — sum(Z)={z_sum}, "
                f"charge={q:+d} implies {z_sum - q} electrons but the log says {ne}. "
                f"Do not run this structure until resolved.")
        if ne % 2:
            raise ValueError(f"FMODB {fmodb_id}: {ne} electrons is ODD — not closed-shell, and RKS "
                             f"is all we do. Needs an open-shell path or a different entry.")
    entry = FmodbEntry(
        fmodb_id=fmodb_id, read_geom=want or pdb_name,
        method=meta.get("method") or "?", basis=lg.get("basis") or meta.get("basis") or "?",
        nao=lg.get("nao"), charge=q, nelec=ne,
        n_atoms=n_atoms, formula=_formula(comp), composition=comp,
        e_hf=lg.get("e_hf"), e_mp2=lg.get("e_mp2"),
        retrieved=datetime.date.today().isoformat(),
        source_url=f"{BASE}/detail.php?FMODBID={fmodb_id}")

    exp = (SYSTEMS.get(fmodb_id, {}).get("formula") if expect_formula == "auto"
           else expect_formula)
    if exp and entry.formula != exp:
        raise ValueError(
            f"FMODB {fmodb_id}: composition CHANGED — expected {exp}, got {entry.formula}. "
            f"Either the entry was revised upstream or the wrong structure was read "
            f"(ReadGeom={entry.read_geom}). Do NOT quote an energy against this until resolved.")
    return pdb_path, entry


def to_xyz(pdb_path: str, xyz_path: str) -> int:
    """PDB → plain element/x/y/z (Å), the format `experiments/common/systems.py` peptides use."""
    lines = []
    for line in open(pdb_path):
        if line.startswith(("ATOM  ", "HETATM")):
            el = line[76:78].strip() or re.sub(r"[^A-Za-z]", "", line[12:16].strip())[:1]
            lines.append(f"{el.capitalize():2s} {float(line[30:38]):12.6f} "
                         f"{float(line[38:46]):12.6f} {float(line[46:54]):12.6f}")
    with open(xyz_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


_GEOM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry")


def materialize(fmodb_id: str, dest_dir: str | None = None) -> str:
    """Write ``geometry/<ID>.xyz`` from the verified structure and return its path.

    `fetch` asserts the composition against :data:`SYSTEMS`, so a materialized file is the molecule
    the registry describes. The structures are CC BY-SA 4.0 and are not part of this repository."""
    pdb_path, _entry = fetch(fmodb_id)
    dest = dest_dir or _GEOM_DIR
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, f"{fmodb_id.strip().upper()}.xyz")
    to_xyz(pdb_path, out)
    return out


def onboard(fmodb_id: str, *, basis: str = "cc-pvtz") -> dict:
    """Fetch an ID, verify it, and report everything needed to add it to :data:`SYSTEMS`.

    Discovery is the one thing this module cannot do: FMODB's search renders client-side and cannot
    be scripted. The workflow is: a human finds the id in a browser, this turns it into a registry
    entry.
    """
    path, e = fetch(fmodb_id, expect_formula=None)
    out = {"fmodb_id": e.fmodb_id, "formula": e.formula, "n_atoms": e.n_atoms,
           "charge": e.charge, "nelec": e.nelec,
           "fmodb_basis": e.basis, "fmodb_nao": e.nao, "method": e.method,
           "e_hf": e.e_hf, "e_mp2": e.e_mp2, "pdb": path}
    try:                       # our own count in OUR basis — what actually sizes a run
        import numpy as np
        from dftax.system import Molecule
        from gs_dft import nao as nao_of
        syms, xyz = [], []
        for line in open(path):
            if line.startswith(("ATOM  ", "HETATM")):
                import re as _re
                el = line[76:78].strip() or _re.sub(r"[^A-Za-z]", "", line[12:16].strip())[:1]
                syms.append(el.capitalize())
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        mol = Molecule(symbols=syms, coords_bohr=np.array(xyz) / 0.529177210903,
                       basis=basis, charge=int(e.charge or 0), spherical=True)
        out[f"nao_{basis}"] = int(nao_of(mol))
    except Exception as ex:
        out[f"nao_{basis}"] = f"ERR {type(ex).__name__}: {ex}"
    return out


if __name__ == "__main__":
    import sys
    argv = [a for a in sys.argv[1:] if a != "--materialize"]
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m experiments.common.fmodb <FMODBID> [<FMODBID> ...]\n"
                         "       python -m experiments.common.fmodb --materialize [<FMODBID> ...]\n"
                         "Find ids in a browser at https://drugdesign.riken.jp/FMODB/ — the search\n"
                         "renders client-side and cannot be driven from a script (see onboard()).")
    if "--materialize" in sys.argv[1:]:
        for _id in (argv or sorted(SYSTEMS)):
            print(f"{_id}: {materialize(_id)}")
        raise SystemExit(0)
    for _id in argv:
        try:
            info = onboard(_id)
        except Exception as exc:
            print(f"{_id}: FAILED {type(exc).__name__}: {exc}"); continue
        print(f"\n=== {info['fmodb_id']} ===")
        for k in ("formula", "n_atoms", "charge", "nelec", "nao_cc-pvtz", "fmodb_basis",
                  "fmodb_nao", "method", "e_hf", "e_mp2"):
            print(f"  {k:14} {info.get(k)}")
        print("  paste into fmodb.SYSTEMS:")
        print(f'    "{info["fmodb_id"]}": {{"formula": "{info["formula"]}", '
              f'"charge": {info["charge"]}, "note": "{info["n_atoms"]} atoms, '
              f'nelec={info["nelec"]}, nao(cc-pVTZ)={info.get("nao_cc-pvtz")}"}},')
