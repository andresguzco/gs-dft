"""The collapse guard (``gs_dft.ks.diagnostics.CollapseGuard``).

It trips only on a non-finite energy or gradient (or a regressed electron count:
tests/test_nelec_guard.py), never on a finite jump, and latches once triggered.
"""
from gs_dft.ks.diagnostics import CollapseGuard


def test_non_finite_is_always_collapse():
    g = CollapseGuard()
    assert g.is_collapse(0, float("nan"))
    assert g.is_collapse(0, float("inf"))
    assert g.is_collapse(100, float("-inf"))


def test_non_finite_gradient_is_collapse():
    g = CollapseGuard()
    assert g.is_collapse(0, -100.0, gnorm=float("nan"))    # finite E but a non-finite gradient trips it


def test_finite_energy_is_never_a_collapse():
    g = CollapseGuard()
    assert not g.check(0, -100.0)
    # any finite energy rides through — the benign pair/aux-refresh discontinuity and the
    # self-resolving splat-collision spike must NOT halt a run that would recover
    assert not g.check(5, 200.0)                  # +300 Ha
    assert not g.check(6, 1e9)                    # a huge finite jump is still not a collapse


def test_non_finite_fires_and_latches():
    g = CollapseGuard()
    assert not g.check(0, -100.0)
    assert g.check(6, float("nan"))
    assert g.reason == "nonfinite"
    assert g.check(7, -100.0)                     # once triggered, every later check is True
