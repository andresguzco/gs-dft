"""Experimental reference values carry provenance, and the two ways of misusing them raise.

Comparing a zero-point-averaged measurement against a clamped-nuclei calculation, or a measurement
of a different conformer or molecule, produces a plausible number rather than an error, so the
registry refuses both.
"""
import pytest

from experiments.exp2_observables.reference_values import (DIPOLES, check_comparable)


def test_every_value_carries_a_source():
    for system, vals in DIPOLES.items():
        assert vals, system
        for v in vals:
            assert v.source and len(v.source) > 20, (system, v)
            assert v.unit == "D" and v.kind in ("mu_0", "mu_e"), (system, v)


def test_exactly_one_preferred_comparator_per_system():
    for system, vals in DIPOLES.items():
        assert sum(v.preferred for v in vals) == 1, system


def test_water_prefers_the_measurement_plus_a_separately_cited_correction():
    """The column headed "experiment" has to print an experimental number.

    An earlier version preferred Clough's mu_e = 1.8473 D, which is a THEORY-CORRECTED value whose
    correction is contradicted: it puts mu_e 7.3 mD below mu_0, while Shostak's own vibrationless
    constant and Lodi's CCSD(T)/CBS mu_e both put it at or above. So the preferred entry is the
    measurement itself, and the bridge to clamped nuclei is a separate, separately cited field.
    """
    v = check_comparable("water")
    assert v.kind == "mu_0" and v.value == pytest.approx(1.85498)
    assert v.uncertainty == pytest.approx(9e-5), "the most precise measurement, not the famous one"
    assert v.clamped_nuclei_offset_mD == pytest.approx(4.1) and v.offset_source
    assert v.clamped_nuclei_target == pytest.approx(1.85908, abs=5e-6)

    superseded = [x for x in DIPOLES["water"] if x.kind == "mu_e"]
    assert superseded and not any(x.preferred for x in superseded), (
        "Clough's mu_e is recorded for the record, never selected")
    assert "contradicted" in superseded[0].note, "the reason it is not used must travel with it"


def test_conformer_selection_picks_the_rotamer_that_was_computed():
    """Selection is BY CONFORMER first, then `preferred`.

    An earlier version took the single preferred entry and then rejected any conformer that did not
    match it, so recording a second rotamer turned a valid comparison into a refusal. Our geometry
    is anti; anti must resolve, and gauche must still resolve to its own measurement.
    """
    anti = check_comparable("ethanol", conformer="anti")
    assert anti.value == pytest.approx(1.441) and anti.conformer == "anti"
    assert check_comparable("ethanol", conformer="gauche").value == pytest.approx(1.679)


def test_unrecorded_rotamer_refuses_rather_than_substituting_a_neighbour():
    with pytest.raises(ValueError, match="rotamer"):
        check_comparable("ethanol", conformer="eclipsed")


def test_conformer_must_be_stated_when_it_matters():
    with pytest.raises(ValueError, match="conformer-specific"):
        check_comparable("ethanol")


def test_unknown_system_refuses_rather_than_guesses():
    with pytest.raises(KeyError, match="no experimental dipole"):
        check_comparable("penicillin")
