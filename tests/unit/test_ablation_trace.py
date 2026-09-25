"""The Figure 2(c) trajectory parser reads what the runner writes: the TRACE line and the log name.

The fixture is built with the producer's own ``_trace_line``, and the log name is checked against
the expansion in ``run_panels.sh``.
"""
import os
import pathlib
import re

import numpy as np
import pytest

from experiments.exp1_ablation_ladder import plot_ablation as P

SYS = "ala_45"          # panel 3's system; the log name carries it and the readers require it


def _as_csv(d):
    """The logs of ``d`` as the data CSV the plotter reads, converted the way data/ was."""
    from experiments.common.results import write_csv
    out = os.path.join(str(d), "data.csv")
    write_csv(sorted(str(q) for q in pathlib.Path(d).glob("*.log")), out)
    return out


def _write(tmp_path, tag, body, system=SYS):
    d = tmp_path / "results"
    d.mkdir(exist_ok=True)
    (d / P.p3_log_name(tag, system)).write_text(body, encoding="utf-8")
    return str(d)



def _trace_line(rung="P3_floor", step=250, t=61.5, E=-7434.411806):
    """Exactly the f-string in machinery.run's `_trace` callback."""
    return f"TRACE rung={rung} step={int(step)} t={t:.3f} E={float(E):.8f}\n"


def test_trace_time_parses_the_runners_own_format(tmp_path):
    body = "".join(_trace_line(step=s, t=1.5 * s, E=-7434.0 - 0.001 * s) for s in (50, 100, 150))
    d = _write(tmp_path, "P3_floor", body)
    P.DATA = _as_csv(d)
    t, E = P.trace_time("P3_floor", SYS)
    assert len(t) == 3, "the plotter must see every streamed point"
    assert t[0] == pytest.approx(75.0 / 60.0), "t is reported in MINUTES for the panel's x axis"
    assert E[-1] == pytest.approx(-7434.15)


def test_a_diverged_arm_keeps_its_nan_rather_than_vanishing(tmp_path):
    """The unfloored arm's death IS the panel's claim, so a non-finite energy must survive parsing:
    the plotter marks the last finite point, which it cannot do if the NaN row is dropped."""
    body = _trace_line(rung="P3_nofloor", step=50, t=10.0, E=-7434.0) + \
           "TRACE rung=P3_nofloor step=100 t=20.000 E=nan\n"
    d = _write(tmp_path, "P3_nofloor", body)
    P.DATA = _as_csv(d)
    t, E = P.trace_time("P3_nofloor", SYS)
    assert len(t) == 2 and np.isfinite(E[0]) and not np.isfinite(E[1])


def test_another_systems_log_is_not_absorbed(tmp_path):
    """A log belonging to ANOTHER system must be invisible, not merely deprioritized — otherwise the
    runner's skip-check reads a previous system's logs as this run's and the panel draws two
    molecules as one experiment.
    """
    d = tmp_path / "results"
    d.mkdir()
    (d / P.p3_log_name("P3_floor", "PJM49")).write_text(
        _trace_line(rung="P3_floor", step=180, t=6480.0, E=-7434.4), encoding="utf-8")
    P.DATA = _as_csv(d)
    t, _E = P.trace_time("P3_floor", "ala_45")
    assert len(t) == 0, "another system's log must not be read as this system's trajectory"
    # ...and the one that IS this system's is still found, so the guard is not just refusing work.
    (d / P.p3_log_name("P3_floor", "ala_45")).write_text(
        _trace_line(rung="P3_floor", step=50, t=600.0, E=-11015.1), encoding="utf-8")
    P.DATA = _as_csv(d)
    t, E = P.trace_time("P3_floor", "ala_45")
    assert len(t) == 1 and E[0] == pytest.approx(-11015.1)


def test_another_arms_lines_are_not_absorbed(tmp_path):
    """Arms are matched on wall-clock in one allocation, so two arms' lines can share a file. The
    rung guard is what keeps arm 3's trajectory out of arm 2's curve."""
    body = _trace_line(rung="P3_floor", step=50, t=10.0) + \
           _trace_line(rung="P3_multi", step=50, t=3.0)
    d = _write(tmp_path, "P3_floor", body)
    P.DATA = _as_csv(d)
    t, _E = P.trace_time("P3_floor", SYS)
    assert len(t) == 1, "only the arm being asked for"



def test_gto_minimize_parses_the_engines_own_verbose_line():
    import inspect

    from dftax.ks import minimize

    from experiments.exp1_ablation_ladder.gto_minimize import _MIN_LINE

    src = inspect.getsource(minimize)
    assert 'print(f"  min {step:4d}: E={float(e):.10f}  |g|={gnorm:.2e}")' in src, \
        "dftax.ks.minimize's verbose line changed; gto_minimize._MIN_LINE no longer matches it"
    # ...and the regex actually accepts a line built by that format string.
    step, e, g = 1234, -76.4123456789, 3.21e-4
    assert _MIN_LINE.match(f"  min {step:4d}: E={float(e):.10f}  |g|={g:.2e}")


def test_gto_minimize_emits_our_trace_grammar(capsys):
    """The converter's OUTPUT must be the same line every other arm emits, or the plotter needs a
    second parser and the panel grows a way to disagree with itself."""
    import sys

    from experiments.exp1_ablation_ladder.gto_minimize import _Retrace

    r = _Retrace(sys.stdout, "p1_gto", t0=0.0)
    r.write("  min   12: E=-76.4123456789  |g|=3.21e-04\n")
    r.write("some other engine chatter\n")
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("TRACE")]
    assert len(line) == 1, out
    assert P._TRACE.match(line[0]), f"gto_minimize emitted a line plot_ablation cannot read: {line}"
    assert "some other engine chatter" in out, "non-matching output must pass through, not vanish"



def test_the_log_name_matches_the_runner_that_writes_it():
    """The producer is a shell expansion in `run_panels.sh`; the consumer is `p3_log_name`. Nothing
    else connects them, and when they drifted apart the reader's tests went red while the panel
    stayed fine — so assert against the shell source itself rather than a copy of it."""
    sh = (pathlib.Path(__file__).resolve().parents[2]
          / "experiments/exp1_ablation_ladder/run_panels.sh").read_text()
    m = re.search(r'run "(p3_\$\{arm\}[^"]*)"', sh)
    assert m, "run_panels.sh no longer builds a p3_ arm log name the way this test expects"
    shell = (m.group(1)
             .replace("${arm}", "P3_floor")
             .replace("${P3_SYS}", SYS)
             .replace("${P3_SEED:-0}", "0")
             .replace("${P3_REP:+_r$P3_REP}", "_rconv")) + ".log"
    assert P.p3_log_name("P3_floor", SYS, rep="conv") == shell, (
        f"reader builds {P.p3_log_name('P3_floor', SYS, rep='conv')}, runner writes {shell}")


def test_no_replicate_tag_drops_the_segment_entirely():
    """`P3_REP` empty means an unreplicated run, whose files carry no `_r` segment at all — the
    shell's `${P3_REP:+_r$P3_REP}` expands to nothing. A reader that appended a bare `_r` would
    look for a file no run ever writes."""
    assert P.p3_log_name("P3_floor", SYS, rep="") == f"p3_P3_floor_{SYS}_s0.log"
