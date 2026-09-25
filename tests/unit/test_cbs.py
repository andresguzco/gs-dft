"""CBS extrapolation reports what it actually did, and refuses to invent a limit."""
import math

import pytest

from experiments.common import cbs


def test_decaying_energy_triple_is_extrapolated():
    # a clean geometric approach: steps -0.040, -0.020 -> r = 0.5, limit = last - 0.020
    out = cbs.extrapolate({"cc-pvdz": -76.240, "cc-pvtz": -76.280, "cc-pvqz": -76.300})
    assert out["kind"] == "cbs"
    assert out["r"] == pytest.approx(0.5)
    assert out["value"] == pytest.approx(-76.320)
    assert out["spread"] == pytest.approx(0.020)


def test_the_real_water_dipole_ladder_is_refused():
    """The measured ladder is non-monotone, so it must NOT be extrapolated."""
    out = cbs.extrapolate({"cc-pvdz": 2.2353, "cc-pvtz": 2.2738, "cc-pvqz": 2.2614})
    assert out["kind"] == "largest"
    assert out["value"] == pytest.approx(2.2614), "falls back to the largest cardinal, unmodified"
    assert out["r"] is None


def test_growing_steps_are_refused():
    """Differences that grow are diverging, not converging on a limit."""
    assert cbs.extrapolate({"cc-pvdz": 1.0, "cc-pvtz": 1.1, "cc-pvqz": 1.4})["kind"] == "largest"


def test_fewer_than_three_cardinals_returns_nan_not_a_substitute():
    """The alanine-dipeptide failure: returning the largest basis as 'the limit' turned every
    error-from-CBS into an error-from-DZ with nothing in the figure saying so."""
    out = cbs.extrapolate({"cc-pvdz": -1.0, "cc-pvtz": -1.1})
    assert out["kind"] == "none" and math.isnan(out["value"])
    assert math.isnan(cbs.error_vs_limit(-1.05, out)), "an unusable limit must poison the error too"


def test_unknown_bases_are_ignored_not_misordered():
    out = cbs.extrapolate({"cc-pvdz": -76.24, "cc-pvtz": -76.28, "cc-pvqz": -76.30,
                           "def2-svp": -76.10})
    assert out["cardinals"] == [2, 3, 4] and out["kind"] == "cbs"


def test_largest_three_cardinals_win_when_more_exist():
    out = cbs.extrapolate({"cc-pvdz": 0.0, "cc-pvtz": -76.240, "cc-pvqz": -76.280,
                           "cc-pv5z": -76.300})
    assert out["cardinals"] == [3, 4, 5]


def test_summarize_names_the_fallbacks():
    limits = {"E": cbs.extrapolate({"cc-pvdz": -76.24, "cc-pvtz": -76.28, "cc-pvqz": -76.30}),
              "mu": cbs.extrapolate({"cc-pvdz": 2.2353, "cc-pvtz": 2.2738, "cc-pvqz": 2.2614}),
              "F": cbs.extrapolate({"cc-pvdz": 0.1, "cc-pvtz": 0.09})}
    s = cbs.summarize(limits)
    assert "extrapolated: E" in s and "mu" in s and "NOT extrapolated" in s and "UNAVAILABLE" in s
