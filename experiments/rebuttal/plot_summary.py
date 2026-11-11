"""Summary of the accuracy scaling, from the ladders of the spaghetti figure (each from its own limit).

    TUEPLOTS=1 USETEX=1 python -m experiments.rebuttal.plot_summary [data/rebuttal/accuracy.csv] [out_dir]

Each ladder is fitted with the power law Delta = A N**(-alpha) (the log-log line through its points, as
in the spaghetti figure's legend). One full-width figure, three panels against the system size:

- a) the crossover per atom, N*/n, where N* = (A_s / A_g)**(1 / (alpha_s - alpha_g)) is the number of
  basis functions at which the splat line crosses the Gaussian one; the dashed line is the mean over
  the C/H/N/O systems;
- b) the number of basis functions that reaches Delta = 1 mHa, N = A**(1/alpha), for both methods, with
  a power law in the size per method; a value outside a ladder's measured range is an extrapolation
  (open marker);
- c) the error per atom at each system's cc-pVTZ size: the Gaussian TZ point, and the splat power law at
  M = n_ao(TZ) (open marker where that is an extrapolation).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.style import apply as apply_style, save_at_height, PAPER_FIG_HEIGHT_IN
from experiments.rebuttal import plot_spaghetti as sp

OUT_DIR = os.path.join(sp.HERE, "figures")
GTO_C, SPL_C = "#9a9a9a", "#5b2d91"     # the paper's method colours
TARGET_MHA = 1.0
FIG_H = 1.2                            # inches, one full-width row with a legend above
ORGANIC = {"H", "C", "N", "O"}


def fit(pts):
    """``(A, alpha, n_lo, n_hi)`` of Delta = A N**(-alpha) through ``pts`` [(N, Delta)], or None."""
    if len(pts) < 2:
        return None
    x, y = (np.log10([p[i] for p in pts]) for i in (0, 1))
    b, a = np.polyfit(x, y, 1)
    return 10.0 ** a, -b, min(p[0] for p in pts), max(p[0] for p in pts)


def _size_law(xs, ys):
    """``(c, beta)`` of y = c x**beta, or None."""
    if len(xs) < 2:
        return None
    b, a = np.polyfit(np.log10(xs), np.log10(ys), 1)
    return 10.0 ** a, b


def _elements(system):
    """The element symbols of a system's geometry file."""
    path = os.path.join(os.path.dirname(os.path.dirname(sp.HERE)), "experiments", "common", "geometry",
                        f"{system}.xyz")
    with open(path) as fh:
        return {ln.split()[0] for ln in fh if ln.split()}


def crossover(ax, fits, atoms, names, colour):
    """Panel a: the crossover per atom against the system size."""
    per = {}
    for s in names:
        g, p = fits[s]
        if g is None or p is None or p[1] <= g[1]:
            print(f"  crossover {s}: no crossing (fits {g}, {p})")
            continue
        n_star = (p[0] / g[0]) ** (1.0 / (p[1] - g[1]))
        per[s] = n_star / atoms[s]
        print(f"  crossover {s:16s} N* = {n_star:8.0f}  ({per[s]:.1f} per atom)")
        ax.plot(atoms[s], per[s], "o", ms=3.5, color=colour[s], zorder=3)
    org = [v for s, v in per.items() if _elements(s) <= ORGANIC]
    if org:
        m, sd = float(np.mean(org)), float(np.std(org))
        print(f"  crossover per atom, C/H/N/O systems: {m:.1f} +- {sd:.1f} ({len(org)} systems)")
        ax.axhline(m, color="0.4", lw=0.9, ls="--", zorder=2)
        ax.text(0.97, 0.04, rf"C/H/N/O mean ${m:.1f} \pm {sd:.1f}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color="0.35")
    ax.set_yscale("linear")
    ax.set_ylim(0, max(per.values()) * 1.15 if per else 1)
    ax.set_ylabel(r"$N^*$ per atom")


def basis_for_target(ax, fits, atoms, names, target=TARGET_MHA):
    """Panel b: the basis functions that reach Delta = ``target`` mHa, against the system size."""
    for side, colour, marker, label, ty in ((0, GTO_C, "o", "cc-pVXZ", 0.88), (1, SPL_C, "s", "Splats", 0.06)):
        xs, ys = [], []
        for s in names:
            f = fits[s][side]
            if f is None:
                continue
            A, alpha, lo, hi = f
            n = (A / target) ** (1.0 / alpha)
            inside = lo <= n <= hi
            print(f"  1 mHa {label:8s} {s:16s} N = {n:9.0f}  ({'interpolated' if inside else 'extrapolated'})")
            ax.plot(atoms[s], n, marker, ms=3.5, color=colour, zorder=3,
                    markerfacecolor=colour if inside else "white", markeredgewidth=0.9)
            xs.append(atoms[s])
            ys.append(n)
        law = _size_law(xs, ys)
        if law:
            xf = np.geomspace(min(xs) / 1.3, max(xs) * 1.3, 50)
            ax.plot(xf, law[0] * xf ** law[1], "-", color=colour, lw=1.0, zorder=2)
            ax.text(0.97 if side else 0.04, ty, rf"$N \propto n^{{{law[1]:.2f}}}$", transform=ax.transAxes,
                    ha="right" if side else "left", va="bottom" if side else "top", fontsize=6, color=colour)
    ax.set_ylabel(rf"$N$ for $\Delta = {target:g}$ mHa")


def error_per_atom(ax, fits, lad, atoms, names, basis="cc-pvtz"):
    """Panel c: the error per atom at each system's ``basis`` size, both methods."""
    with open(os.path.join(sp.HERE, "data", "systems.json")) as fh:
        nao = {s: v["nao"][basis] for s, v in json.load(fh).items()}
    gx, gy, sx, sy = [], [], [], []
    for s in names:
        g = dict(lad[s][0])
        if nao[s] in g:
            gx.append(atoms[s])
            gy.append(g[nao[s]] / atoms[s])
        f = fits[s][1]
        if f is not None:
            A, alpha, lo, hi = f
            y = A * nao[s] ** (-alpha) / atoms[s]
            ax.plot(atoms[s], y, "s", ms=3.5, color=SPL_C, zorder=3,
                    markerfacecolor=SPL_C if lo <= nao[s] <= hi else "white", markeredgewidth=0.9)
            sx.append(atoms[s])
            sy.append(y)
        print(f"  {basis} size {s:16s} n_ao {nao[s]:6d}  cc-pVXZ {g.get(nao[s], float('nan')) / atoms[s]:.4f}  "
              f"splats {sy[-1] if f is not None else float('nan'):.4f} mHa/atom")
    ax.plot(gx, gy, "o-", ms=3.5, lw=1.0, color=GTO_C, zorder=3)
    ax.plot(sx, sy, "-", lw=1.0, color=SPL_C, zorder=2)
    ax.set_ylabel(r"$\Delta$ per atom (mHa)")


def main(path=sp.DATA, out_dir=OUT_DIR, names=None):
    atoms = sp.sizes()
    names = sorted(names or sp.SYSTEMS, key=atoms.get)
    lad = sp.ladders(path, names, atoms)
    names = [s for s in names if s in lad]
    fits = {s: (fit(lad[s][0]), fit(lad[s][1])) for s in names}
    usetex = apply_style(nrows=1, ncols=3, rel_width=1.0, height_in=FIG_H)
    fig, axes = plt.subplots(1, 3)
    crossover(axes[0], fits, atoms, names, sp.colours(names))
    basis_for_target(axes[1], fits, atoms, names)
    error_per_atom(axes[2], fits, lad, atoms, names)
    for k, (ax, title) in enumerate(zip(axes, ("Crossover per atom", "Basis size at 1 mHa",
                                               "Error per atom at TZ size"))):
        tag = rf"\textbf{{{'abc'[k]})}}" if usetex else rf"$\mathbf{{{'abc'[k]}}}$)"
        ax.set_title(f"{tag} {title}", fontsize=7, pad=2.0)
        ax.set_xscale("log")
        if k:
            ax.set_yscale("log")
        ax.set_xlabel("System size (atoms)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], color=GTO_C, marker="o", ms=3.5, lw=1.0, label="cc-pVXZ"),
               plt.Line2D([], [], color=SPL_C, marker="s", ms=3.5, lw=1.0, label="Splats")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.6)
    fig.tight_layout(w_pad=0.8)
    fig.subplots_adjust(top=0.74)
    out = os.path.join(out_dir, "accuracy_summary")
    save_at_height(fig, out + ".png", height_in=FIG_H, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=FIG_H)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
