"""Geometries PubChem has no 3-D conformer for (over 50 heavy atoms), built once with RDKit.

    uv run --no-project --with rdkit python experiments/rebuttal/build_geometry.py \
        --pdb 5AWL.pdb --out experiments/common/geometry

chignolin    CLN025 (YYDPETGTWY), the X-ray structure PDB 5AWL (1.11 Å; waters dropped). Hydrogens
             added for the neutral form: NH2 and COOH termini, Asp and Glu protonated. MMFF94
             relaxes the hydrogens only; the heavy atoms stay at the crystal positions.
cyclosporin  cyclosporin A from the PubChem CID 5284373 isomeric SMILES. ETKDGv3 with macrocycle
             torsions, MMFF94-minimized conformers, the lowest-energy one kept.

RDKit and the standard library only (no JAX), so it runs in a throwaway environment. Writes bare
``element x y z`` lines, the format ``experiments.common.systems`` reads.
"""
import argparse
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

CLN025 = "YYDPETGTWY"
CYCLOSPORIN_SMILES = (     # PubChem CID 5284373, isomeric SMILES (retrieved 2026-11-06)
    "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N("
    "[C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)"
    "CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C")


def write_xyz(mol, path, conf_id=-1):
    conf = mol.GetConformer(conf_id)
    with open(path, "w") as fh:
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            fh.write(f"{a.GetSymbol():2s} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}\n")


def chignolin(pdb_path):
    """CLN025 heavy atoms from the crystal, bond orders from the sequence, hydrogens by MMFF94."""
    block = "".join(ln for ln in open(pdb_path) if ln.startswith("ATOM")) + "END\n"
    xtal = Chem.MolFromPDBBlock(block, removeHs=True)
    template = Chem.MolFromSequence(CLN025)              # neutral termini and side chains
    mol = AllChem.AssignBondOrdersFromTemplate(template, xtal)
    Chem.AssignStereochemistryFrom3D(mol)
    molH = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(molH)
    ff = AllChem.MMFFGetMoleculeForceField(molH, props)
    for a in molH.GetAtoms():
        if a.GetAtomicNum() > 1:
            ff.AddFixedPoint(a.GetIdx())
    rc = ff.Minimize(maxIts=20000)
    print(f"chignolin: {CalcMolFormula(molH)} {molH.GetNumAtoms()} atoms, MMFF H-only "
          f"rc={rc} E={ff.CalcEnergy():.2f} kcal/mol", flush=True)
    return molH, -1


def cyclosporin(n_conf, seed, threads):
    """Lowest-MMFF94 of ``n_conf`` ETKDGv3 macrocycle conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles(CYCLOSPORIN_SMILES))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.useMacrocycleTorsions = True
    ps.numThreads = threads
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=ps))
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=20000)
    ok = [(e, cid) for cid, (notconv, e) in zip(cids, res) if notconv == 0]
    if not ok:
        raise RuntimeError("cyclosporin: no MMFF minimization converged")
    e, best = min(ok)
    print(f"cyclosporin: {CalcMolFormula(mol)} {mol.GetNumAtoms()} atoms, {len(cids)} conformers "
          f"({len(ok)} converged), best conf {best} E={e:.2f} kcal/mol "
          f"(spread {max(x for x, _ in ok) - e:.1f})", flush=True)
    return mol, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="5AWL.pdb (https://files.rcsb.org/download/5AWL.pdb)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_conf", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mol, cid = chignolin(a.pdb)
    write_xyz(mol, os.path.join(a.out, "chignolin.xyz"), cid)
    mol, cid = cyclosporin(a.n_conf, a.seed, a.threads)
    write_xyz(mol, os.path.join(a.out, "cyclosporin.xyz"), cid)


if __name__ == "__main__":
    main()
