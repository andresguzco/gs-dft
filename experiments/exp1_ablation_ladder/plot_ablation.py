"""Figure 2: one row, three panels.

    (a) parametrization             energy against a converged Gaussian basis
    (b) density fitting and reach   peak memory against free parameters
    (c) sharding                    energy against wall-clock time

    TUEPLOTS=1 USETEX=1 uv run python -m experiments.exp1_ablation_ladder.plot_ablation

Reads ``data/figures/fig2_ablation.csv`` and writes ``ablation.{png,pdf}`` next to this file. A run that ran
out of memory or diverged is drawn as a hatched mark labelled OOM or NaN.
"""
import glob
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.common import cbs as _cbs, ladder as _ladder
from experiments.common.results import csv_sources, iter_results, read_records
from experiments.common.style import (apply as apply_style, PAPER_FIG_HEIGHT_IN,
                                       save_at_height)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "figures", "fig2_ablation.csv")

FIG_HEIGHT_IN = PAPER_FIG_HEIGHT_IN

# Colour is identity, never rank: purple is ours, grey the baseline. Failure is carried by hatch
# and an explicit label, so nothing depends on colour alone.
OURS = "#5b2d91"
BASE = "#9a9a9a"
FAIL = "#c23b22"
INK = "#333333"


def rows(pattern, want):
    """RESULT rows for the arms named in ``want``, keyed by rung, newest wins."""
    out = {}
    for d in iter_results(DATA, sources=[pattern]):
        if d.get("rung") in want:
            out[d["rung"]] = d
    return out


EXP2_RESULTS = os.path.join(HERE, "..", "..", "data", "figures", "fig3_observables.csv")


def gto_reference(system="water"):
    """The CBS limit for panel 1's zero, from the committed Gaussian ladder."""
    rows = {}
    for d in iter_results(EXP2_RESULTS, kind="gto"):
        if d.get("system") == system and d.get("basis") and d.get("E"):
            rows[d["basis"]] = float(d["E"])
    if not rows:
        return None
    lim = _cbs.extrapolate(rows)
    return float(lim["value"]) if lim["kind"] != "none" else None


def _key(ax, ncol=1, loc="upper right"):
    """Draw the panel's key floated inside its axes.

    Inside rather than above: a key above the axes costs the whole row a band of figure height.
    Panels 1 and 3 both have an empty top-right corner by construction. Single column by default --
    three entries laid out horizontally reach back over panel 1's opening transient.
    """
    tight = ncol >= 3
    ax.legend(fontsize=4.1 if tight else 5.0, frameon=False, ncol=ncol, loc=loc,
              handlelength=0.8 if tight else 1.4, columnspacing=0.45 if tight else 1.0,
              handletextpad=0.25 if tight else 0.4, borderaxespad=0.15, labelspacing=0.2)


def _label_end(ax, x, y, text, colour, dy=0):
    """Write a curve's name at its own end, instead of in a legend box."""
    if not len(x):
        return
    ax.annotate(text, (x[-1], y[-1]), textcoords="offset points", xytext=(3, dy),
                fontsize=5.0, color=colour, va="center", ha="left",
                annotation_clip=False, zorder=6)


def _blank(ax, msg):
    """Show a message on an empty panel and hide its axes: a 0-to-1 axis on absent data reads as
    real scale."""
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=6, color=INK,
            transform=ax.transAxes)
    ax.set_axis_off()


def _fail_bar(ax, x, label, top):
    """Draw a failed arm hatched to the panel ceiling, outlined and labelled, rather than omitting
    it."""
    ax.bar(x, top, width=0.36, color="none", edgecolor=FAIL, hatch="///", linewidth=1.0, zorder=3)
    ax.text(x, top * 0.97, label, ha="center", va="top", fontsize=6, color=FAIL, zorder=4)


def _steps(tag, rung=None):
    """(step, E) from a run's streamed TRACE lines."""
    sources = sorted(csv_sources(DATA, f"{tag}.log"))
    if not sources:
        return np.empty(0), np.empty(0)
    rec = []
    for d in read_records(DATA, record="TRACE", sources=sources[-1:]):
        if rung is None or d["rung"] == rung:
            e = d["E"]
            rec.append((int(d["step"]), float("nan") if e == "nan" else float(e)))
    if not rec:
        return np.empty(0), np.empty(0)
    seen = {}
    for st, e in rec:
        seen[st] = e
    rec = [(st, seen[st]) for st in sorted(seen)]
    a = np.asarray(rec, dtype=float)
    return a[:, 0], a[:, 1]


def _decade_xaxis(ax, x0=None):
    """Log x with WHOLE-DECADE ticks, and limits that end on decades.

    `x0` pins the left edge instead of taking it from the data, for a panel that should start at a
    stated decade rather than wherever its first sample lands."""
    ax.set_xscale("log")
    xs = [v for ln in ax.get_lines() for v in np.asarray(ln.get_xdata(), dtype=float)]
    xs = [v for v in xs if np.isfinite(v) and v > 0]
    if xs:
        lo = 10.0 ** np.floor(np.log10(min(xs))) if x0 is None else float(x0)
        ax.set_xlim(lo, 10.0 ** np.ceil(np.log10(max(xs))))
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.xaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def panel_parametrization(ax, ref=None):
    """Panel 1: three optimization trajectories to convergence, on one step axis.

    The Gaussian arm is direct minimization rather than SCF so that the x axis means one thing --
    an SCF has fixed-point iterations, not gradient steps -- leaving the representation as the only
    difference between the curves. y is the offset from the converged Gaussian energy on a symlog
    axis: the arms span +12,060 to -10.8 mHa, and our curve legitimately crosses zero, which a log
    axis would drop.
    """
    arms = [("p1_old", "old", "FEGO", BASE, ":"),
            ("p1_gto", "p1_gto", "cc-pVTZ", BASE, "--"),
            ("p1_new", "new", "Ours", OURS, "-")]
    ref = gto_reference("water") if ref is None else ref
    drawn = 0
    for tag, rung, label, colour, ls in arms:
        st, E = _steps(tag, rung)
        if not len(st):
            continue
        good = np.isfinite(E)
        drawn += 1
        yy = 1e3 * (E[good] - ref)
        ax.plot(st[good], yy, ls=ls, color=colour, lw=1.3, zorder=3, label=label)
    if not drawn:
        _blank(ax, "parametrization:\nrun PHASE=p1")
        return
    ax.set_yscale("log")
    # Linear optimizer-step axis: the panel reads as progress through the run, not as decades.
    xs = [v for ln in ax.get_lines() for v in np.asarray(ln.get_xdata(), dtype=float)]
    xs = [v for v in xs if np.isfinite(v)]
    if xs:
        ax.set_xlim(0.0, max(xs))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(r"$\Delta_{\mathrm{CBS}}$ (mHa)")
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 30.0)
    _key(ax, ncol=3)


def _cost_panel(ax, r, order, labels, title):
    """Draw one arm: per-step cost where it survived, a hatched ceiling where it did not."""
    have = [k for k in order if k in r]
    if not have:
        _blank(ax, f"{title}:\nno rows yet")
        return
    ok = [float(r[k]["ms_step"]) for k in have
          if r[k].get("failed", "none") == "none" and r[k].get("kinetic_ok") != "False"]
    top = max(ok) * 1.6 if ok else 1.0
    for x, k in enumerate(have):
        d = r[k]
        failed = d.get("failed", "none") != "none"
        collapsed = d.get("kinetic_ok") == "False" or d.get("finite") == "False"
        if failed or collapsed:
            _fail_bar(ax, x, "OOM" if d.get("failed") == "oom" else "collapse", top)
        else:
            ax.bar(x, float(d["ms_step"]), color=OURS if x else BASE, width=0.36, zorder=3)
            ax.text(x, float(d["ms_step"]), f"\n{float(d['ms_step']):.0f}", ha="center",
                    va="bottom", fontsize=6, color=INK)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([labels[k] for k in have], fontsize=6)
    ax.set_ylim(0, top)
    ax.set_ylabel("ms / step")


FLOOR_REL = 1e-4
P3_SYS = os.environ.get("P3_SYS", "ala_45")


def p3_log_name(arm, system, seed=0, rep=None):
    """An arm log's filename — ONE definition, mirroring the producer in ``run_panels.sh``::

        p3_${arm}_${P3_SYS}_s${P3_SEED:-0}${P3_REP:+_r$P3_REP}

    The producer is a shell expansion in another file and nothing but this function connects them;
    when the two drift the trace comes back empty and the panel renders blank. ``P3_REP`` is
    resolved in the body because it is declared further down this module, so a signature default
    would raise ``NameError`` at import.
    """
    rep = P3_REP if rep is None else rep
    return f"p3_{arm}_{system}_s{seed}" + (f"_r{rep}" if rep else "") + ".log"


P2_THIN = {"P2_fast": (4, 5)}


def _mem_series(tag):
    """(params, peak_mb, failed) for every M of one arm, ascending in params."""
    pts = []
    for d in iter_results(DATA, sources=["p2_*.log"]):
        if d.get("rung") != tag:
            continue
        pk = d.get("peak_train_mb", d.get("peak_mb"))
        if d.get("params") is None or pk is None:
            continue
        done = _f(d, "resumed_from")
        want = _f(d, "steps")
        if done == done and want == want and done >= want:
            print(f"    SKIP {d.get('rung')} M={d.get('M')}: resumed_from={int(done)} >= "
                  f"steps={int(want)} — trained nothing, its peak is a no-op")
            continue
        pts.append((float(d["params"]), float(pk), d.get("failed", "none")))
    latest = {}
    for p, mb, failed in pts:
        latest[p] = (p, mb, failed)
    pts = sorted(latest.values())
    drop = P2_THIN.get(tag, ())
    if drop:
        pts = [p for i, p in enumerate(pts) if i not in drop]
    return pts


def _f(d, k):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _gto_series(arm):
    """(params, peak_mb, failed) for one Gaussian arm, ascending in params."""
    pts = []
    for d in iter_results(DATA, sources=["p2_gto_*.log"]):
        if d.get("kind") != "gto_mem" or d.get("arm") != arm or d.get("params") is None:
            continue
        pk = d.get("peak_train_mb")
        if pk is None:
            continue
        pts.append((float(d["params"]), float(pk), d.get("failed", "none")))
    pts.sort()
    return pts


def _card_name(vram_mb):
    """The capacity of the GPU the panel-2 points were measured on, in GB."""
    return f"{vram_mb / 1024:.0f} GB"


def panel_reach(ax, vram_mb=81920.0):
    """Peak memory vs free parameters: where each arm meets the card."""
    arms = [("P2_dense", "Exact", OURS, ":", _mem_series),
            ("P2_df", "+ DF", "#1f78b4", "--", _mem_series),
            ("P2_fast", "+ Screen", "#0f9b74", "-", _mem_series),
            ("gto_exact", "GTO exact", "#b0b0b0", ":", _gto_series),
            ("gto_df", "GTO + DF", "#7d7d7d", "--", _gto_series),
            ("gto_fast", "GTO + screen", "#4a4a4a", "-", _gto_series)]
    drawn = 0
    fails = []
    for tag, label, colour, ls, getter in arms:
        pts = getter(tag)
        ok = [(x, y) for x, y, f in pts if f == "none"]
        bad = [(x, y) for x, y, f in pts if f != "none"]
        mk = "s" if getter is _gto_series else "o"
        if ok:
            drawn += 1
            ax.plot([p[0] for p in ok], [p[1] for p in ok], ls=ls, color=colour, lw=1.3,
                    marker=mk, ms=2.8, mfc="white", mew=0.9, zorder=4, label=label)
        elif pts:
            ax.plot([], [], ls=ls, color=colour, lw=1.3, marker=mk, ms=2.8, mfc="white",
                    mew=0.9, label=label)
        if not bad:
            continue
        x_f = bad[0][0]
        if ok:
            ax.plot([ok[-1][0], x_f], [ok[-1][1], vram_mb], ls=ls, color=colour, lw=0.9,
                    alpha=0.6, zorder=3)
        ax.plot([x_f], [vram_mb], marker=mk, color=colour, ms=3.6, mfc=colour, mew=0.9,
                zorder=6, ls="none")
        fails.append((x_f, label, colour))
    if not drawn:
        _blank(ax, "reach:\nrun PHASE=p2")
        return
    ax.axhline(vram_mb, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.annotate(_card_name(vram_mb), (0.99, vram_mb), xycoords=("axes fraction", "data"),
                ha="right", va="top", textcoords="offset points", xytext=(0, -2),
                fontsize=5.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Free parameters")
    ax.set_ylabel("Memory (MB)")
    lo = min(y for _t, _l, _c, _ls, g in arms for _x, y, f in g(_t) if f == "none")
    ax.set_ylim(bottom=lo / 40.0, top=vram_mb * 1.6)
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0 / 1.5, x1 * 3.0)
    ax.legend(fontsize=4.6, frameon=False, loc="lower right", ncol=2, handlelength=1.1,
              borderaxespad=0.25, labelspacing=0.18, columnspacing=0.7, handletextpad=0.35)


_TRACE = re.compile(r"^TRACE rung=(\S+) step=(\d+) t=([\d.]+) E=(-?[\d.eE+]+|nan)")


def trace_time(rung, system=None, rep=None):
    """``(t_seconds, E)`` from an arm's streamed TRACE lines, or empty arrays.

    Takes the RUNG, which is what both the filename and the row are keyed by. Read from the log
    rather than the RESULT row: the row is one endpoint, this panel is the path, and the log is
    what survives a walltime kill.
    """
    system = P3_FINAL_SYS if system is None else system
    rep = P3_REP if rep is None else rep
    sources = csv_sources(DATA, p3_log_name(rung, system, rep=rep))
    if not sources:
        return np.empty(0), np.empty(0)
    rec = []
    for d in read_records(DATA, record="TRACE", sources=sources[-1:]):
        if d["rung"] == rung:
            e = d["E"]
            rec.append((float(d["t"]), float("nan") if e == "nan" else float(e)))
    if not rec:
        return np.empty(0), np.empty(0)
    a = np.asarray(rec, dtype=float)
    return a[:, 0] / 60.0, a[:, 1]


P3B_SYS = os.environ.get("P3B_SYS", P3_SYS)


def _stability(ax, arms, system, title, ylabel=True):
    """Energy vs wall-clock for a set of arms on one system. Shared by both rows of panel 3."""
    drawn, finals = 0, []
    for tag, label, colour, ls in arms:
        t, E = trace_time(tag, system)
        if not len(t):
            continue
        drawn += 1
        good = np.isfinite(E)
        ax.plot(t[good], E[good], ls=ls, color=colour, lw=1.4, zorder=3, label=label)
        finals.append(float(E[good][-1]))
        if (~good).any() and good.any():
            i = max(int(np.argmax(~good)) - 1, 0)
            ax.plot([t[i]], [E[i]], "x", color=FAIL, ms=7, mew=1.6, zorder=6)
    if not drawn:
        _blank(ax, title.replace(" (", "\n("))
        return
    if finals:
        lo, hi = min(finals), max(finals)
        pad = max((hi - lo) * 3.0, abs(lo) * 1e-4, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Wall-clock (min)")
    if ylabel:
        ax.set_ylabel("$E$ (Ha)")
    ax.legend(fontsize=5.2, frameon=False, loc="upper right", handlelength=1.5,
              borderaxespad=0.2, labelspacing=0.25)


P3_FINAL_SYS = os.environ.get("P3_FINAL_SYS", "ala_45")
P3_REP = os.environ.get("P3_REP_TAG", "conv")
GAP_REL = 1e-4


def _grad(tag):
    """(step, |g|, min-eigenvalue-gap) from a run's monitor and TRACE lines."""
    sources = sorted(csv_sources(DATA, f"{tag}.log"))
    if not sources:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    g, gap = [], []
    for d in read_records(DATA, sources=sources[-1:]):
        if d["_record"] == "STEP" and "E" in d and "grad_norm" in d:
            g.append((int(d["step"]), float(d["grad_norm"])))
        elif d["_record"] != "STEP" and "step" in d and "mingap" in d:
            gap.append((int(d["step"]), float(re.match(r"[0-9.eE+-]+", d["mingap"]).group(0))))
    ga = np.asarray(g, dtype=float) if g else np.empty((0, 2))
    pa = np.asarray(gap, dtype=float) if gap else np.empty((0, 2))
    return (ga[:, 0] if ga.size else np.empty(0), ga[:, 1] if ga.size else np.empty(0),
            pa[:, 0] if pa.size else np.empty(0), pa[:, 1] if pa.size else np.empty(0))


def panel_stability(ax):
    """Panel 3: the scale-out claim -- the same trajectory, more of it per hour.

    Two arms, ``P3_floor`` and ``P3_multi``, which differ in ``sharded`` and nothing else. x is
    wall-clock because sharding buys steps per second rather than steps; on a step axis the two
    curves coincide, and that coincidence is the control -- at matched step they agree to ~1 mHa.

    The arms ran concurrently on one node and shared host memory and PCIe, so their throughput
    ratio is not a clean scaling measurement.
    """
    arms = [("P3_floor", "1 GPU", BASE, "--"),
            ("P3_multi", "2 GPUs, sharded", OURS, "-")]
    series = {r: trace_time(r) for r, _l, _c, _ls in arms}
    _all = [float(v) for _r, (_t, _E) in series.items() for v in _E if v == v]
    if not _all:
        _blank(ax, f"scale-out:\nno p3_*_{P3_FINAL_SYS}_s0_r{P3_REP} logs")
        return
    e_ref = min(_all)
    drawn, tail_vals = 0, []
    for rung, label, colour, ls in arms:
        t, E = series[rung]
        if not len(t):
            continue
        drawn += 1
        good = np.isfinite(E)
        y = E[good] - e_ref
        keep = y > 0.0
        # SECONDS. The run spans 179-1195 min, 0.82 of a decade, and the unit decides how much of
        # a decade-aligned axis the data fills: in seconds it is 10.7k-71.7k, filling 82% of the
        # single decade 1e4-1e5, where minutes or hours straddle two decades and fill 41%.
        ax.plot(t[good][keep] * 60.0, y[keep], ls=ls, color=colour, lw=1.3, zorder=3, label=label)
        tail_vals.extend(float(v) for v in y[keep])
        if (~good).any() and good.any():
            i = max(int(np.argmax(~good)) - 1, 0)
            ax.plot([t[i] * 60.0], [E[i] - e_ref], "x", color=FAIL, ms=5, mew=1.3, zorder=6)
    if not drawn:
        _blank(ax, f"scale-out:\nno p3_*_{P3_FINAL_SYS}_s0_r{P3_REP} logs")
        return
    ax.set_yscale("log")
    if tail_vals:
        ax.set_ylim(min(tail_vals) / 2.0, max(tail_vals) * 1.6)
    _decade_xaxis(ax)
    ax.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Wall-clock (s)")
    ax.set_ylabel(r"$E - E_{\mathrm{best}}$ (Ha)")
    _key(ax, loc="lower left")


def _has_data(panel):
    scratch = plt.figure()
    ax = scratch.add_subplot(111)
    try:
        panel(ax)
        return bool(ax.axison)
    finally:
        plt.close(scratch)


def _equal_gaps(fig, axes):
    """Re-space the panels so every y-label sits the same distance from the panel to its left.

    `tight_layout` spaces by tick-label width, so a panel with wider labels ends up closer to its
    neighbour. Panel widths stay equal and the outer edges stay where tight_layout put them.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pos = [ax.get_position() for ax in axes]
    left = [inv.transform((ax.yaxis.get_tightbbox(r).x0, 0))[0] for ax in axes]
    deco = [p.x0 - l for p, l in zip(pos, left)]
    gap = float(np.mean([left[i] - pos[i - 1].x1 for i in range(1, len(axes))]))
    x0, x1 = pos[0].x0, pos[-1].x1
    w = (x1 - x0 - sum(gap + d for d in deco[1:])) / len(axes)
    x = x0
    for i, (ax, p) in enumerate(zip(axes, pos)):
        if i:
            x += gap + deco[i]
        ax.set_position([x, p.y0, w, p.height])
        x += w


def main():
    os.environ.setdefault("TUEPLOTS", "1")
    ready = [p for p in (panel_parametrization, panel_reach, panel_stability) if _has_data(p)]
    n = len(ready)
    if n < 3:
        print(f"  {3 - n} panel(s) have no data yet — rendering {n} of 3")
    height_in = float(os.environ.get("HEIGHT_IN", str(FIG_HEIGHT_IN)))
    usetex = apply_style(nrows=1, ncols=n, height_in=height_in)
    fig, axes = plt.subplots(1, n, squeeze=False)
    axes = axes[0]
    for ax, panel in zip(axes, ready):
        panel(ax)
    # Bold panel letters (a)/(b)/(c) lead each x-axis label, where they cover no data.
    for _i, ax in enumerate(axes):
        if ax.axison:
            _l = "abcdefgh"[_i]
            _pre = rf"\textbf{{{_l})}} " if usetex else rf"$\mathbf{{{_l}}}$) "
            ax.set_xlabel(_pre + ax.get_xlabel())
    for ax in axes:
        if ax.axison:
            ax.grid(axis="x" if panel_parametrization in ready and ax is axes[0] else "both", color="#e8e8e8", lw=0.6, zorder=0)
            ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes:
        if ax.axison:
            ax.yaxis.labelpad = 1.5
    fig.tight_layout(pad=0.4, w_pad=0.0)
    _equal_gaps(fig, [ax for ax in axes if ax.axison])
    out = os.path.join(HERE, "ablation")
    save_at_height(fig, out + ".png", height_in=height_in, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=height_in)
    print(f"wrote {out}.{{png,pdf}}  (usetex={usetex})")


if __name__ == "__main__":
    main()
