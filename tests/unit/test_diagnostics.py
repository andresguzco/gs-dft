"""The collapse guard (``gs_dft.ks.diagnostics.CollapseGuard``).

It trips only on a non-finite energy or gradient, never on a finite jump, dumps the state before
and after, and latches once triggered.
"""
import json

from gs_dft.ks.diagnostics import CollapseGuard


def _recorder():
    """A stand-in ``save_fn(path_prefix, state, step)`` that records its calls."""
    calls = []

    def save_fn(path_prefix, state, step):
        calls.append((path_prefix, state, step))

    return calls, save_fn


def test_non_finite_is_always_collapse():
    _, save_fn = _recorder()
    g = CollapseGuard(None, save_fn)
    # NaN / ±inf collapse regardless of step or whether a sane step was recorded
    assert g.is_collapse(0, float("nan"))
    assert g.is_collapse(0, float("inf"))
    assert g.is_collapse(100, float("-inf"))


def test_non_finite_gradient_is_collapse():
    _, save_fn = _recorder()
    g = CollapseGuard(None, save_fn)
    assert g.is_collapse(0, -100.0, gnorm=float("nan"))    # finite E but a non-finite gradient trips it


def test_finite_energy_is_never_a_collapse():
    calls, save_fn = _recorder()
    g = CollapseGuard(None, save_fn)
    g.check(0, "s0", -100.0)                        # baseline
    # any finite energy rides through — the benign pair/aux-refresh discontinuity and the
    # self-resolving splat-collision spike must NOT halt a run that would recover
    assert not g.check(5, "s1", 200.0)             # +300 Ha
    assert not g.check(6, "s2", 1e9)               # a huge finite jump is still not a collapse
    assert g.last_good[0] == 6                      # ...each becomes the new last-good
    assert calls == []                             # nothing serialized


def test_non_finite_fires_saves_and_latches(tmp_path):
    calls, save_fn = _recorder()
    base = str(tmp_path / "run")
    g = CollapseGuard(base, save_fn)
    g.check(0, "good", -100.0)                      # sane baseline
    assert g.check(6, "broken", float("nan"))      # E → NaN → collapse

    # before = last-good bundle, after = the broken bundle
    saved = {c[0]: (c[1], c[2]) for c in calls}
    assert saved[f"{base}_before"] == ("good", 0)
    assert saved[f"{base}_after"] == ("broken", 6)

    # metadata JSON records the transition
    with open(f"{base}_collapse.json") as fh:
        meta = json.load(fh)
    assert meta["collapse_step"] == 6
    assert meta["before_step"] == 0
    assert meta["E_before"] == -100.0

    # once triggered every later check short-circuits to True without re-saving
    assert g.check(7, "later", -100.0)
    assert len(calls) == 2                          # still just the one before/after pair
