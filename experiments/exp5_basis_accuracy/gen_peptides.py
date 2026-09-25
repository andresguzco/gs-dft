"""Generate the poly-alanine geometries (Ala_n) in ``peptides/`` with RDKit (ETKDG embedding and MMFF
relaxation), so runs need no RDKit.

    uv run --with rdkit python -m experiments.exp5_basis_accuracy.gen_peptides
"""
import os
from rdkit import Chem
from rdkit.Chem import AllChem

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peptides")
os.makedirs(OUT, exist_ok=True)
SIZES = [2, 4, 8, 16]


def main():
    for n in SIZES:
        m = Chem.MolFromSequence("A" * n)
        if m is None:
            print(f"ala_{n}: MolFromSequence failed"); continue
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = 0
        if AllChem.EmbedMolecule(m, p) != 0:
            AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=0)
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        conf = m.GetConformer()
        lines = [f"{a.GetSymbol()} {conf.GetAtomPosition(a.GetIdx()).x:.6f} "
                 f"{conf.GetAtomPosition(a.GetIdx()).y:.6f} {conf.GetAtomPosition(a.GetIdx()).z:.6f}"
                 for a in m.GetAtoms()]
        path = os.path.join(OUT, f"ala_{n}.xyz")
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        print(f"ala_{n}: {m.GetNumAtoms()} atoms -> {path}", flush=True)


if __name__ == "__main__":
    main()
