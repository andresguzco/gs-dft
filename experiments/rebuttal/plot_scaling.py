"""Experiment 2 figure: strong-scaling speedup t(1)/t(G) per system, with the ideal line.

    python -m experiments.rebuttal.plot_scaling <scaling.csv> [out]      # writes out.pdf and out.png

The step time per (system, G) is the median over repeats of each run's median step; the error bar
spans the repeats' min..max speedup. Printed per point: the step, speedup and parallel efficiency
t(1)/(G t(G)); the refresh (median over repeats) and the efficiency with the refresh amortized at the
ladder runs' cadence (every 50 steps); and the largest per-device peak.
"""
import csv
import statistics
import sys
from collections import defaultdict

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")      # categorical slots 1-4, fixed order


CADENCE = 50          # the ladder runs' refresh cadence
FIG_H = 3.2           # inches: a tall half-width panel, so the curves separate from the ideal line
REL_W = 0.5           # of the text width; the style and save_at_height size by the full width
BAR_H = 2.6           # inches: the speedup bars
SHADES = {1: ("#5b2d91",), 2: ("#b39ddb", "#5b2d91"), 3: ("#c9b8e8", "#8a63c4", "#4a2477")}   # light to dark with G
NAMES = {"retinal": "Retinal", "chignolin": "Chignolin", "cyclosporin": "Cyclosporin A",
         "fmodb_insulin": "Insulin", "fmodb_spectrin_ch": r"$\beta$-spectrin CH"}


def load(path):
    t = defaultdict(lambda: defaultdict(list))
    ref = defaultdict(lambda: defaultdict(list))
    peak = defaultdict(lambda: defaultdict(list))
    natoms = {}
    for r in csv.DictReader(open(path)):
        if r["step_med_s"]:
            s, g = r["system"], int(r["G"])
            t[s][g].append(float(r["step_med_s"]))
            if r.get("refresh_med_s"):
                ref[s][g].append(float(r["refresh_med_s"]))
            dev = r.get("peak_gpu_mb_dev") or r.get("peak_gpu_mb") or ""
            if dev:
                peak[s][g].append(max(float(x) for x in str(dev).split("/")))
            natoms[s] = int(float(r["n_atoms"]))
    return t, natoms, ref, peak


def main(path, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.common.style import apply, save_at_height
    apply(height_in=FIG_H / REL_W, nrows=1, ncols=1, rel_width=REL_W)
    t, natoms, ref, peak = load(path)
    systems = sorted(t, key=lambda s: natoms[s])
    fig, ax = plt.subplots()
    gmax = max(g for s in systems for g in t[s])
    ax.plot([1, gmax], [1, gmax], color="0.55", lw=1.0, ls="--", label="ideal")
    for c, s in zip(COLORS, systems):
        if 1 not in t[s]:
            continue
        base = statistics.median(t[s][1])
        gs = sorted(t[s])
        sp = [base / statistics.median(t[s][g]) for g in gs]
        lo = [sp[i] - base / max(t[s][g]) for i, g in enumerate(gs)]
        hi = [base / min(t[s][g]) - sp[i] for i, g in enumerate(gs)]
        ax.errorbar(gs, sp, yerr=[lo, hi], color=c, lw=1.0, marker="o", ms=3, capsize=1.5,
                    label=f"{NAMES.get(s, s)} ({natoms[s]} atoms)")
        amort = {g: statistics.median(t[s][g]) + (statistics.median(ref[s][g]) / CADENCE
                                                   if ref[s][g] else 0.0) for g in gs}
        for g, v in zip(gs, sp):
            r = f"{statistics.median(ref[s][g]):.2f}s" if ref[s][g] else "-"
            pk = f"{max(peak[s][g]):.0f} MB" if peak[s][g] else "-"
            print(f"{s:13s} G={g}  t={statistics.median(t[s][g]):.3f}s  speedup {v:.2f}  "
                  f"efficiency {v / g:.2f}  refresh {r}  efficiency at refresh/{CADENCE} "
                  f"{amort[1] / (g * amort[g]):.2f}  peak/device {pk}")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Speedup")
    ax.set_xticks(sorted({g for s in systems for g in t[s]}))
    ax.set_yticks(range(1, gmax + 1))
    ax.set_ylim(0.5, gmax + 0.4)
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.4, labelspacing=0.3)
    fig.tight_layout()
    out = out or path.rsplit(".", 1)[0]
    save_at_height(fig, out + ".png", height_in=FIG_H / REL_W, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=FIG_H / REL_W)
    print(f"wrote {out}")


def speedup_bars(path, out=None):
    """Speedup t(1)/t(G) as grouped bars: one group per system, one bar per G > 1 (one hue, light to dark
    with G), the error bar over the repeats' min..max, each bar's ideal (G) dashed across it.
    Writes out.pdf and out.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.common.style import apply, save_at_height
    apply(height_in=BAR_H / REL_W, nrows=1, ncols=1, rel_width=REL_W)
    t, natoms, _ref, _peak = load(path)
    systems = sorted((s for s in t if 1 in t[s]), key=lambda s: natoms[s])
    gs = sorted({g for s in systems for g in t[s] if g > 1})
    shades = dict(zip(gs, SHADES[len(gs)]))
    fig, ax = plt.subplots()
    width = 0.8 / len(gs)
    for k, g in enumerate(gs):
        xs, v, lo, hi = [], [], [], []
        for i, s in enumerate(systems):
            if g not in t[s]:
                continue
            base = statistics.median(t[s][1])
            sp = base / statistics.median(t[s][g])
            xs.append(i + (k - (len(gs) - 1) / 2) * width)
            v.append(sp)
            lo.append(sp - base / max(t[s][g]))
            hi.append(base / min(t[s][g]) - sp)
        ax.bar(xs, v, width * 0.92, color=shades[g], label=f"{g} GPUs", zorder=3)
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor="0.25", elinewidth=0.6, capsize=1.5, zorder=4)
        for x, y in zip(xs, v):
            ax.hlines(g, x - width * 0.46, x + width * 0.46, colors="0.35", lw=0.8, ls="--", zorder=5)
            ax.text(x, max(y, g) + 0.08, f"{y:.2f}", ha="center", va="bottom", fontsize=5.5, color="0.15")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([f"{NAMES.get(s, s)}\n({natoms[s]} atoms)" for s in systems])
    ax.set_ylabel("Speedup")
    ax.set_ylim(0, max(gs) + 0.6)
    ax.set_yticks(range(0, max(gs) + 1))
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(gs),
              borderaxespad=0.2)                  # above the axes: the bars fill the plot area
    fig.tight_layout()
    out = out or path.rsplit(".", 1)[0] + "_bars"
    save_at_height(fig, out + ".png", height_in=BAR_H / REL_W, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=BAR_H / REL_W)
    print(f"wrote {out}")


def memory_bars(path, out=None):
    """Memory reduction peak(1)/peak(G) as grouped bars, in the style of :func:`speedup_bars`: one group
    per system, one bar per G > 1 (one hue, light to dark with G), each bar's ideal (G) dashed across it.
    A peak is the largest device's, the one that decides whether a run fits. Reads exp2_memory.csv
    (data/rebuttal/); writes out.pdf and out.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.common.style import apply, save_at_height
    apply(height_in=BAR_H / REL_W, nrows=1, ncols=1, rel_width=REL_W)
    rows = list(csv.DictReader(open(path)))
    natoms = {r["system"]: int(r["n_atoms"]) for r in rows}
    peak = {(r["system"], int(r["G"])): max(float(x) for x in r["peak_gpu_mb_dev"].split("/"))
            for r in rows if r["peak_gpu_mb_dev"]}
    systems = sorted((s for s, g in peak if g == 1), key=natoms.get)
    gs = sorted({g for s, g in peak if g > 1 and s in systems})
    shades = dict(zip(gs, SHADES[len(gs)]))
    fig, ax = plt.subplots()
    width = 0.8 / len(gs)
    for k, g in enumerate(gs):
        xs, v = [], []
        for i, s in enumerate(systems):
            if (s, g) not in peak:
                continue
            xs.append(i + (k - (len(gs) - 1) / 2) * width)
            v.append(peak[s, 1] / peak[s, g])
            print(f"{s:18s} G={g}  peak/GPU {peak[s, g]:.0f} MB (1 GPU: {peak[s, 1]:.0f})  "
                  f"reduction {v[-1]:.2f}  efficiency {v[-1] / g:.2f}")
        ax.bar(xs, v, width * 0.92, color=shades[g], label=f"{g} GPUs", zorder=3)
        for x, y in zip(xs, v):
            ax.hlines(g, x - width * 0.46, x + width * 0.46, colors="0.35", lw=0.8, ls="--", zorder=5)
            ax.text(x, max(y, g) + 0.08, f"{y:.2f}", ha="center", va="bottom", fontsize=5.5, color="0.15")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([f"{NAMES.get(s, s)}\n({natoms[s]} atoms)" for s in systems])
    ax.set_ylabel("Memory reduction")
    ax.set_ylim(0, max(gs) + 0.6)
    ax.set_yticks(range(0, max(gs) + 1))
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(gs),
              borderaxespad=0.2)
    fig.tight_layout()
    out = out or path.rsplit(".", 1)[0]
    save_at_height(fig, out + ".png", height_in=BAR_H / REL_W, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=BAR_H / REL_W)
    print(f"wrote {out}")

PE_H = 1.5            # inches, full text width: the time and memory panels side by side (Figure 7's height)
PE_SHADES = dict(zip((2, 3, 4), SHADES[3]))   # one shade per G in both panels


def _efficiency_bars(ax, systems, natoms, val, err=None):
    """Grouped bars on ``ax``: one group per system, one bar per G (light to dark), each bar's ideal (G)
    dashed across it and its value above; ``val[(system, G)]``, ``err[(system, G)] = (lo, hi)``."""
    gs = sorted({g for _, g in val})
    width = 0.8 / len(gs)
    for k, g in enumerate(gs):
        xs, v, lo, hi = [], [], [], []
        for i, s in enumerate(systems):
            if (s, g) in val:
                xs.append(i + (k - (len(gs) - 1) / 2) * width)
                v.append(val[s, g])
                lo.append(err[s, g][0] if err else 0.0)
                hi.append(err[s, g][1] if err else 0.0)
        ax.bar(xs, v, width * 0.92, color=PE_SHADES[g], zorder=3)
        if err:
            ax.errorbar(xs, v, yerr=[lo, hi], fmt="none", ecolor="0.25", elinewidth=0.6, capsize=1.2, zorder=4)
        for x, y in zip(xs, v):
            ax.hlines(g, x - width * 0.46, x + width * 0.46, colors="0.35", lw=0.8, ls="--", zorder=5)
            ax.text(x, max(y, g) + 0.08, f"{y:.2f}", ha="center", va="bottom", fontsize=4.8, color="0.15")
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([f"{NAMES.get(s, s)}\n({natoms[s]} atoms)" for s in systems], fontsize=5.5)
    ax.set_ylim(0, 4.7)
    ax.set_yticks(range(0, 5))
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def parallel_efficiency(scaling_csv, memory_csv, out):
    """The parallel efficiency, one full-width figure: a) the speedup t(1)/t(G) of the step (median over
    repeats, error bars over their min..max), b) the memory reduction, the busiest GPU's peak at one GPU
    over its peak at G. Writes out.pdf and out.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.common.style import apply, save_at_height
    usetex = apply(height_in=PE_H, nrows=1, ncols=2, rel_width=1.0)
    t, natoms, _ref, _peak = load(scaling_csv)
    tsys = sorted((s for s in t if 1 in t[s]), key=lambda s: natoms[s])
    speed, err = {}, {}
    for s in tsys:
        base = statistics.median(t[s][1])
        for g in t[s]:
            if g > 1:
                speed[s, g] = base / statistics.median(t[s][g])
                err[s, g] = (speed[s, g] - base / max(t[s][g]), base / min(t[s][g]) - speed[s, g])
    rows = list(csv.DictReader(open(memory_csv)))
    matoms = {r["system"]: int(r["n_atoms"]) for r in rows}
    peak = {(r["system"], int(r["G"])): max(float(x) for x in r["peak_gpu_mb_dev"].split("/"))
            for r in rows if r["peak_gpu_mb_dev"]}
    msys = sorted((s for s, g in peak if g == 1), key=matoms.get)
    red = {(s, g): peak[s, 1] / peak[s, g] for s, g in peak if g > 1 and s in msys}
    for (s, g), v in sorted(speed.items()):
        print(f"  time   {s:18s} G={g}  speedup {v:.2f}  efficiency {v / g:.2f}")
    for (s, g), v in sorted(red.items()):
        print(f"  memory {s:18s} G={g}  reduction {v:.2f}  efficiency {v / g:.2f}")
    fig, axes = plt.subplots(1, 2)
    _efficiency_bars(axes[0], tsys, natoms, speed, err)
    _efficiency_bars(axes[1], msys, matoms, red)
    for k, (ax, title, ylab) in enumerate(zip(axes, ("Time", "Memory"), ("Speedup", "Reduction"))):
        tag = rf"\textbf{{{'ab'[k]})}}" if usetex else rf"$\mathbf{{{'ab'[k]}}}$)"
        ax.set_title(f"{tag} {title}", fontsize=7, pad=2.0)
        ax.set_ylabel(ylab)
    gs = sorted({g for _, g in speed} | {g for _, g in red})
    handles = [plt.Rectangle((0, 0), 1, 1, color=PE_SHADES[g], label=f"{g} GPUs") for g in gs]
    fig.legend(handles=handles, loc="upper center", ncol=len(gs), frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(top=0.80)
    save_at_height(fig, out + ".png", height_in=PE_H, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=PE_H)
    print(f"wrote {out}")

if __name__ == "__main__":
    main(*sys.argv[1:])
