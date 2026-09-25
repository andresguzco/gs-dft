"""The electron-count guard, on recorded ``nelec`` traces of a 45-residue alanine chain (M = 4299).

The single-device trace must run clean to the end, and the two-device trace, whose electron count
drifts, must trip the guard at step 100.
"""
from gs_dft.ks.diagnostics import CollapseGuard

NELEC_REF = 1720.0

# (step, nelec) — verbatim from ala_45_nd1_ab.jsonl / ala_45_nd2_ab.jsonl
ND1 = [(0, 1742.714), (25, 1935.038), (50, 1739.988), (75, 1720.331),
       (100, 1720.015), (125, 1720.010), (150, 1720.030)]
ND2 = [(0, 1742.714), (25, 1935.038), (50, 1739.988), (75, 1721.058),
       (100, 1741.756), (125, 1609.240), (150, 1550.713)]


def _run(trace, tmp_path):
    saved = []
    g = CollapseGuard(str(tmp_path / "g"), lambda p, s, st: saved.append((p, st)),
                      nelec_ref=NELEC_REF)
    for step, nelec in trace:
        if g.check(step, ("state", step), -10000.0, 1.0, nelec=nelec):
            return step, g
    return None, g


def test_healthy_ndev1_never_trips(tmp_path):
    """The ndev=1 arm converges onto Sum(occ) and must not be halted — including at step 25, where
    the residual is +215 (12%) and a plain threshold would false-positive."""
    step, g = _run(ND1, tmp_path)
    assert step is None, f"guard fired on a healthy run at step {step}"
    assert g.nelec_earned, "the healthy run should have earned the high-water mark"


def test_ndev2_corruption_trips_at_step_100(tmp_path):
    """The real failure: nelec jumps 1721.058 -> 1741.756 at step 100 while E_kinetic stays
    continuous. The run this came from finished with collapsed=False."""
    step, g = _run(ND2, tmp_path)
    assert step == 100, f"expected the guard to fire at step 100, got {step}"
    assert g.reason == "nelec"


def test_no_ref_is_inert(tmp_path):
    """Without nelec_ref the guard keeps its old non-finite-only behaviour (back-compat)."""
    step, _ = _run(ND2, tmp_path)  # sanity: fires with a ref
    assert step == 100
    saved = []
    g = CollapseGuard(str(tmp_path / "h"), lambda p, s, st: saved.append(p))
    for st, ne in ND2:
        assert not g.check(st, ("s", st), -10000.0, 1.0, nelec=ne)


def test_early_excursion_before_earning_is_ignored(tmp_path):
    """A young run whose residual is large and getting larger is NOT a collapse — the mark must be
    earned first, or every run would be halted in its first tens of steps."""
    saved = []
    g = CollapseGuard(str(tmp_path / "e"), lambda p, s, st: saved.append(p), nelec_ref=NELEC_REF)
    for st, ne in [(0, 1742.7), (5, 1800.0), (10, 1935.0), (15, 2100.0)]:
        assert not g.check(st, ("s", st), -1.0, 1.0, nelec=ne)
    assert not g.nelec_earned
