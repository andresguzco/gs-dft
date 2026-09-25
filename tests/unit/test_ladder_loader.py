"""The cardinal-ladder loader resolves a rung that appears in several logs deterministically.

The log order is explicit, the newest log wins, and an ambiguous rung is reported.
"""
import os
import time

import pytest

from experiments.common import ladder

OLD = "RESULT system=ethanol basis=cc-pvqz nao=345 E=-154.9134853735 converged=True\n"
NEW = "RESULT kind=gto system=ethanol basis=cc-pvqz mode=df nao=345 E=-154.9135935900 converged=True\n"
# a third cardinal each side, so fit_cbs has something to work with
CTX = ("RESULT kind=gto system=ethanol basis=cc-pvtz nao=174 E=-154.89869357 converged=True\n"
       "RESULT kind=gto system=ethanol basis=cc-pvdz nao=72 E=-154.83811121 converged=True\n")


def _write(d, name, text, mtime):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(text)
    os.utime(p, (mtime, mtime))
    return p


@pytest.mark.parametrize("old_name,new_name", [("a_old.log", "z_new.log"),   # alphabetical == age
                                               ("z_old.log", "a_new.log")])  # alphabetical REVERSED
def test_newest_row_wins_regardless_of_filename(tmp_path, old_name, new_name):
    d = str(tmp_path)
    t = time.time()
    _write(d, old_name, OLD + CTX, t - 10_000)
    _write(d, new_name, NEW, t)
    got = ladder.load(d)["ethanol"]["gto"]["cc-pvqz"][1]
    assert got == pytest.approx(-154.91359359, abs=1e-9), (
        "the older row won — rung resolution is following filename order, not recency")


def test_prerebuild_rung_is_reported_only_when_it_actually_wins(tmp_path):
    d = str(tmp_path)
    t = time.time()
    _write(d, "june.log", OLD + CTX, t - 10_000)
    data = ladder.load(d)
    assert "cc-pvqz" in data["ethanol"]["prerebuild"]
    assert ladder.era_warnings(data), "a CBS built on an engine-0.4 rung must be reported"

    _write(d, "august.log", NEW, t)               # a fresh row supersedes it
    data = ladder.load(d)
    assert "cc-pvqz" not in data["ethanol"]["prerebuild"], (
        "a stale row that a fresh one replaced is not contamination")
    assert not ladder.era_warnings(data)


# ---------------------------------------------------------------------------------------------
# An inverted ladder means two different calculations were merged under one key.

# D/T/Q rungs at one water geometry and a 5Z rung at another. Every row is well-formed and carries
# `kind=`, so only the monotonicity check can see it.
MIXED_GEOM = (
    "RESULT kind=gto system=h2o basis=cc-pvdz mode=df xc=pbe nao=24 E=-76.33347551 converged=True\n"
    "RESULT kind=gto system=h2o basis=cc-pvtz mode=df xc=pbe nao=58 E=-76.37289638 converged=True\n"
    "RESULT kind=gto system=h2o basis=cc-pvqz mode=df xc=pbe nao=115 E=-76.38312412 converged=True\n"
    "RESULT kind=gto system=h2o basis=cc-pv5z mode=df xc=pbe nao=201 E=-76.33429907 converged=True\n"
)
CLEAN_GEOM = MIXED_GEOM.replace("E=-76.33429907", "E=-76.38730662")   # the re-run 5Z


def test_inverted_ladder_is_refused(tmp_path):
    """`load` RAISES on the merge, and names the rung — it is not recoverable downstream."""
    d = str(tmp_path)
    _write(d, "ladder.log", MIXED_GEOM, time.time())
    with pytest.raises(ValueError, match=r"cc-pv5z is 48\.\d+ mHa ABOVE cc-pvqz"):
        ladder.load(d)
    # and the CBS it would otherwise have produced is the stale rung itself, 56 mHa off
    bad = ladder.load(d, strict=False)["water"]
    assert bad["cbs"] == pytest.approx(-76.33429907, abs=1e-6)


def test_clean_ladder_passes_and_fits(tmp_path):
    d = str(tmp_path)
    _write(d, "ladder.log", CLEAN_GEOM, time.time())
    data = ladder.load(d)                       # strict, must not raise
    assert not ladder.variational_violations(data)
    assert data["water"]["cbs"] == pytest.approx(-76.39020035, abs=1e-6)


def test_two_basis_families_are_not_compared(tmp_path):
    """An anion's aug-cc-pVDZ legitimately sits BELOW plain cc-pVTZ.

    Sorting the ladder by `nao` interleaves the two families and invents an inversion; exp4's f⁻
    and OH⁻ both do this, and a detector that fired on them would be turned off within a week.
    """
    d = str(tmp_path)
    _write(d, "anion.log",
           "RESULT kind=gto system=oh_anion basis=aug-cc-pvdz nao=41 E=-75.79 converged=True\n"
           "RESULT kind=gto system=oh_anion basis=cc-pvtz nao=58 E=-75.758 converged=True\n"
           "RESULT kind=gto system=oh_anion basis=aug-cc-pvtz nao=92 E=-75.80 converged=True\n",
           time.time())
    assert not ladder.variational_violations(ladder.load(d, strict=False))
