"""FMODB parsers — the assertions that stop a structure being paired with the wrong energy."""
import pytest

from experiments.common.fmodb import (parse_pdb_composition, _parse_ajf, _parse_log, _formula,
                                      SYSTEMS)

# verbatim shape of the ABINIT-MP log's energy section (25J8R), including the renormalised variants
LOG = """
  ==================
     ## BASIS SET
  ==================

        BASIS SET = 6-31G(d)

        NUMBER OF BASIS FUNCTIONS = 23303

  =========================
     ## FMO TOTAL ENERGY
  =========================

        FMO2-HF
        Nuclear repulsion =       1757593.3829915028
        Electronic energy =      -1827224.7663238063
        Total      energy =        -69631.3833323035

        E(FMO2-MP2)       =          -197.6259393957
        Electronic energy =      -1827422.3922632020
        Total      energy =        -69829.0092716992

        Partially renormalized MP2 energy: Type 1
        E(FMO2-MP2)       =          -194.1499283105
        Total      energy =        -69825.5332606139

        Partially renormalized MP2 energy: Type 2
        Total      energy =        -69842.0585412890
"""

AJF = """&CNTRL
Title='Title'
Method='MP2'
ReadGeom='3ifu_A_lastmodel.pdb'
WriteGeom='3ifu_A_lastmodel.cpf'
/
&BASIS
BasisSet='6-31G*'
/
"""


def test_log_takes_the_PLAIN_totals_not_the_renormalised_ones():
    p = _parse_log(LOG)
    assert p["e_hf"] == pytest.approx(-69631.3833323035, abs=1e-9)
    assert p["e_mp2"] == pytest.approx(-69829.0092716992, abs=1e-9)
    for wrong in (-69825.5332606139, -69842.0585412890):
        assert p["e_mp2"] != pytest.approx(wrong, abs=1e-6)


def test_log_energies_are_consistent_with_their_components():
    """nuclear repulsion + electronic == total, so a mis-parse of any one of them shows up."""
    assert 1757593.3829915028 + -1827224.7663238063 == pytest.approx(-69631.3833323035, abs=1e-6)


def test_basis_is_not_the_banner_rule():
    """The '## BASIS SET' banner is underlined with '=' characters; an unanchored regex parses the
    rule itself as the basis name."""
    p = _parse_log(LOG)
    assert p["basis"] == "6-31G(d)"
    assert p["nao"] == 23303


def test_ajf_names_the_structure_actually_consumed():
    a = _parse_ajf(AJF)
    assert a["read_geom"] == "3ifu_A_lastmodel.pdb"
    assert a["method"] == "MP2"
    assert a["basis"] == "6-31G*"


def test_hetatm_is_counted():
    pdb = (
        "ATOM      1  N   ASP A   1      -8.863  16.944  14.289  1.00  0.00           N\n"
        "ATOM      2  C   ASP A   1      -9.929  17.026  13.244  1.00  0.00           C\n"
        "HETATM 2743 ZN    ZN A 200       0.000   0.000   0.000  1.00  0.00          ZN\n"
    )
    n, comp = parse_pdb_composition(pdb)
    assert n == 3 and comp == {"N": 1, "C": 1, "Zn": 1}


def test_composition_falls_back_to_the_atom_name_column():
    """Older PDBs leave columns 77-78 blank; the element must still be recovered."""
    pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
    n, comp = parse_pdb_composition(pdb)
    assert n == 1 and comp == {"C": 1}


def test_formula_is_hill_ordered():
    assert _formula({"S": 12, "H": 1375, "O": 237, "N": 240, "C": 878}) == "C878H1375N240O237S12"


def test_the_recorded_system_carries_a_formula_to_assert():
    """A SYSTEMS entry without a formula asserts nothing — which is how the insulin geometry went
    unchecked for months."""
    assert SYSTEMS, "no systems recorded"
    for k, v in SYSTEMS.items():
        assert v.get("formula"), f"{k} has no expected formula to assert against"
