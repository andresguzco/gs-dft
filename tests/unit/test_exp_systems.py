"""The system registry builds a Molecule for every experiment's systems: peptides from xyz, charged
anions, diatomics at a given R, and the water alias.
"""
import pytest

from experiments.common import systems


def test_small_molecule_and_water_alias():
    mol = systems.molecule(system="water", basis="sto-3g")
    assert mol.nelectron == 10 and systems.charge("water") == 0
    # water is an alias for the h2o preset — same geometry
    assert systems.geometry("water") == systems.geometry("h2o")


def test_anions_carry_charge():
    assert systems.charge("f_anion") == -1 and systems.charge("oh_anion") == -1
    f = systems.molecule(system="f_anion", basis="sto-3g")
    assert f.nelectron == 10          # F (9) + 1
    oh = systems.molecule(system="oh_anion", basis="sto-3g")
    assert oh.nelectron == 10         # O (8) + H (1) + 1


@pytest.mark.parametrize("name", ["lih", "lif"])
def test_diatomic_r_scan(name):
    grid = systems.R_GRID[name]
    assert grid == sorted(grid) and grid[0] > 0
    a1 = systems.geometry(name, r=1.5)
    a2 = systems.geometry(name, r=3.0)
    assert "3.000000" in a2 and a1 != a2                    # R lands in the atom string
    mol = systems.molecule(system=name, basis="sto-3g", r=grid[0])
    assert mol.nelectron > 0
    with pytest.raises(ValueError):                         # R is mandatory for a diatomic
        systems.geometry(name)


def test_peptide_from_xyz():
    # alanine_dipeptide ships as an xyz in the peptides store (searched, dir-move-robust)
    mol = systems.molecule(system="alanine_dipeptide", basis="sto-3g")
    assert len(mol.symbols) == 22


def test_unknown_system_raises():
    with pytest.raises((AttributeError, ValueError, KeyError)):
        systems.geometry("no_such_system_xyz")


# --- FMODB proteins ---------------------------------------------------------------------------
# FMODB's published `ALL ELECTRON` and `CHARGE OF MOLECULE`. The electron count is sum(Z) - charge,
# so it checks the geometry and the charge together. The structures are fetched, not committed, so
# the geometry checks skip until `python -m experiments.common.fmodb --materialize` has run.
FMODB_NELEC = {"PJM49": 1158, "9LQN2": 1852, "LG5K9": 3686, "25J8R": 10406,
               "38KNL": 6710}
FMODB_CHARGE = {"PJM49": 1, "9LQN2": 2, "LG5K9": 1, "25J8R": 5, "38KNL": 0}


@pytest.mark.parametrize("fid", sorted(FMODB_NELEC))
def test_fmodb_protein_matches_published_electron_count(fid):
    assert systems.charge(fid) == FMODB_CHARGE[fid]
    _need_geometry(fid)
    mol = systems.molecule(system=fid, basis="sto-3g")
    assert mol.nelectron == FMODB_NELEC[fid], (
        f"{fid}: nelec {mol.nelectron} != published {FMODB_NELEC[fid]} — geometry or charge is wrong")
    assert mol.nelectron % 2 == 0, f"{fid} is open-shell; RKS cannot run it"


def _need_geometry(fid):
    if systems._xyz_path(fid) is None:
        pytest.skip(f"{fid}.xyz not fetched")


def test_fmodb_aliases_resolve_to_the_same_molecule():
    for alias, fid in systems._FMODB_ALIASES.items():
        assert systems.charge(alias) == systems.charge(fid), alias
        _need_geometry(fid)
        assert systems.geometry(alias) == systems.geometry(fid), alias


def test_fmodb_insulin_is_not_the_insulin_xyz():
    """`insulin` (784 atoms) and `fmodb_insulin` (947 atoms) are different molecules."""
    _need_geometry("LG5K9")
    assert len(systems.molecule(system="insulin", basis="sto-3g").symbols) == 784
    assert len(systems.molecule(system="fmodb_insulin", basis="sto-3g").symbols) == 947
    assert systems.charge("insulin") == 0 and systems.charge("fmodb_insulin") == 1
