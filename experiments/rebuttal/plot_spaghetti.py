"""Scaling across system size: energy error against the number of basis functions, one line per system,
for the cc-pVXZ ladder (left) and the splat ladder (right), in the paper's style.

    TUEPLOTS=1 python -m experiments.rebuttal.plot_spaghetti [data/rebuttal/accuracy.csv] [out]

Each ladder is measured from its own limit, as in Figure 5: the Gaussian ladder from its CBS limit, the
splat ladder from its fitted power-law limit, and from the Gaussian CBS where the splat ladder has too
few rungs to fit one (printed). Each system's exponent is the slope of the log-log line through its points. One colour per system by its size (viridis, as in the scaling-law papers), the same in both panels;
the legend gives the size and, in parentheses, the exponent.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import ladder
from experiments.common.style import apply as apply_style, save_at_height

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "rebuttal", "accuracy.csv")
OUT = os.path.join(HERE, "figures", "spaghetti")
FIG_H = 1.39         # inches, full text width: two panels with a legend row (7.5% under Figure 7's height)
BUDGET = 12000
SYSTEMS = ("sf6", "cubane", "adenine_thymine", "retinal", "cholesterol", "chignolin", "cyclosporin")   # no porphine
RAMP = plt.get_cmap("viridis")   # the scaling-law papers' map: small systems dark, large ones bright


def _exponent(xs, ys):
    """Minus the slope of log10(y) against log10(x), or None below two points."""
    if len(xs) < 2:
        return None
    return -float(np.polyfit(np.log10(xs), np.log10(ys), 1)[0])


def ladders(path, names, atoms):
    """``{system: (gto [(n, gap)], splat [(M, gap)], splat_ref)}``, gaps in mHa, positive points only."""
    data = ladder.load(path, budget=BUDGET)
    out = {}
    for s in names:
        e = data.get(s)
        if e is None or e["cbs"] != e["cbs"]:
            print(f"  {s}: no Gaussian CBS, left out")
            continue
        g = ladder.gto_ladder(e)
        raw = ladder.splat_ladder(e, require_uniform_seeds=True)
        Ms = sorted(raw)
        fit = ladder.cbs_splat(e, xkey=lambda M: M)
        own = fit["value"] == fit["value"]
        shift = (e["cbs"] - fit["value"]) * 1e3 if own else 0.0
        sp = [(M, raw[M][0] + shift) for M in Ms]
        dropped = [M for M, y in sp if y <= 0] + [n for n, y in g if y <= 0]
        if dropped:
            print(f"  {s}: non-positive gaps not drawn at {dropped}")
        out[s] = ([(n, y) for n, y in g if y > 0], [(M, y) for M, y in sp if y > 0],
                  "own limit" if own else "Gaussian CBS")
        print(f"  {s:16s} ({atoms[s]} atoms): splats from the {out[s][2]}")
    return out


def sizes():
    """``{system: n_atoms}`` from experiments/rebuttal/data/systems.json."""
    import json
    with open(os.path.join(HERE, "data", "systems.json")) as fh:
        return {s: v["n_atoms"] for s, v in json.load(fh).items()}


def colours(names):
    """One viridis colour per system, in the order given (by size), short of the pale end."""
    return {s: RAMP(0.88 * i / max(1, len(names) - 1)) for i, s in enumerate(names)}


def main(path=DATA, out=OUT, names=None, titles=None, atoms=None):
    atoms = atoms or sizes()
    names = sorted(names or SYSTEMS, key=atoms.get)
    lad = ladders(path, names, atoms)
    names = [s for s in names if s in lad]
    usetex = apply_style(nrows=1, ncols=2, rel_width=1.0, height_in=FIG_H)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    colour = colours(names)
    for ax, side, marker, title in ((axes[0], 0, "o", "cc-pVXZ"), (axes[1], 1, "s", "Splats")):
        for s in names:
            pts = lad[s][side]
            if not pts:
                continue
            xs, ys = zip(*pts)
            a = _exponent(xs, ys)
            print(f"  {title:8s} {s:16s} alpha = {a:.2f}" if a is not None else f"  {title:8s} {s}: one point")
            ax.plot(xs, ys, "-", marker=marker, ms=3, lw=1.0, color=colour[s])
            if a is not None:                    # the exponents, one row in the empty band at the bottom
                k = names.index(s)
                ax.text(0.10 + 0.125 * k, 0.04, f"${a:.2f}$", color=colour[s],
                        transform=ax.transAxes, fontsize=4.8, ha="left", va="bottom")
        ax.text(0.02, 0.04, r"$\alpha$:", transform=ax.transAxes, fontsize=4.8, ha="left", va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=7, pad=2.0)
        ax.set_xlabel("Number of basis functions")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"$\Delta$ (mHa)")
    lo = min(y for s in names for side in (0, 1) for _, y in lad[s][side])
    axes[0].set_ylim(bottom=lo / 30)             # an empty band under the lines for the exponents
    handles = [plt.Line2D([], [], color=colour[s], lw=1.0, marker="s", ms=2.5, label=f"{atoms[s]} atoms")
               for s in names]
    fig.legend(handles=handles, loc="upper center", ncol=len(names), frameon=False, fontsize=5.2,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.9)
    fig.tight_layout(w_pad=0.6)
    fig.subplots_adjust(top=0.80)
    save_at_height(fig, out + ".png", height_in=FIG_H, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=FIG_H)
    print(f"wrote {out}")
    return usetex


if __name__ == "__main__":
    main(*sys.argv[1:])
