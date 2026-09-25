"""Figure 1: fixed Gaussian functions against the splat cloud, and the scaling result.

Left: water's cc-pVDZ primitives as 1-sigma rings at the nuclei. Middle: the trained splat cloud of
the same molecule as 1-sigma ellipses. Right: accuracy per parameter. Reads
``data/figures/fig1_*.csv`` and writes
``visual_abstract.{pdf,png}`` next to this file::

    uv run python experiments/shared/visual_abstract.py
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch
from matplotlib.colors import to_rgb

# ---------------------------------------------------------------- palette (validated reference)
INK = "#0b0b0b"; INK2 = "#52514e"; SURFACE = "#ffffff"
VIOLET = "#4a3aa7"          # splats
GTO_GRAY = "#9a9a9a"        # the fixed-basis side, matching the scaling panel's gray curve
O_RED = "#d0342c"; H_GRAY = "#9297a0"; BOND = "#b9b8b3"

from tueplots import bundles
plt.rcParams.update(bundles.iclr2024(usetex=True, family="serif"))
plt.rcParams.update({
    "text.latex.preamble": plt.rcParams["text.latex.preamble"] + r"\usepackage{amsmath}",
    "axes.linewidth": 0.0, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.constrained_layout.use": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FS_TITLE = 8; FS_MATH = 6.8; FS_ROLE = 6.2                      # true print sizes (tueplots)

def _csv(name):
    """A ``data/figures/fig1_<name>.csv`` table as a dict of column arrays (strings)."""
    with open(f"data/figures/fig1_{name}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


_at, _sp = _csv("atoms"), _csv("splats")
coords = np.stack([_at[a].astype(float) for a in "xyz"], axis=1)
charges = _at["charge"].astype(float)
centers = np.stack([_sp[a].astype(float) for a in "xyz"], axis=1)
A = np.stack([_sp[f"A_{a}{b}"].astype(float) for a in "xyz" for b in "xyz"], axis=1).reshape(-1, 3, 3)
norm = _sp["norm"].astype(float)
_g = _csv("gto_primitives")                                                # cc-pVDZ primitives, same atoms
GTO = {"atom": _g["atom"].astype(int), "l": _g["l"].astype(int), "alpha": _g["alpha"].astype(float)}
_s = _csv("scaling")                                                       # ala-dipeptide accuracy/params
SC = {f"{k[0]}{c}": _s[col][_s["ladder"] == k].astype(float)
      for k in ("gto", "splat") for c, col in (("x", "params"), ("y", "error_mha"))}

# ------------------------------------------------------- molecular plane, upright pose
iO = int(np.argmax(charges)); O = coords[iO]
hs = [i for i in range(len(charges)) if i != iO]
b1 = coords[hs[0]] - O; b2 = coords[hs[1]] - O
v = b1 / np.linalg.norm(b1) + b2 / np.linalg.norm(b2); v /= np.linalg.norm(v)
n = np.cross(b1, b2); n /= np.linalg.norm(n)
u = np.cross(v, n); u /= np.linalg.norm(u)
B = np.stack([u, v], axis=1)
at2 = (coords - O) @ B
c2 = (centers - O) @ B
dperp = (centers - O) @ n

span = 3.15; ar = 0.62                                  # half-width, height/width of the window
cx, cy = 0.0, at2[:, 1].mean() + 0.15
XL, XH, YL, YH = cx - span, cx + span, cy - span * ar, cy + span * ar

# ------------------------------------------------------- figure scaffold
FW, FH = 5.5, 1.92
fig = plt.figure(figsize=(FW, FH))
axw, axh = 0.203, 0.62
AXW = [axw, axw, 0.456]
yc = 0.50
y0 = yc - axh / 2
xs = [0.022, 0.265, 0.522]                              # tight gaps; the divider marks the result
axes = [fig.add_axes([x, y0, AXW[i], axh]) for i, x in enumerate(xs)]
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor("none")
for ax in axes[:2]:
    ax.set_xlim(XL, XH); ax.set_ylim(YL, YH); ax.set_aspect("equal")

# drawn-content extent of the aspect-locked panels, in figure coords
content_h = (axw * FW * ar) / FH
c_top = yc + content_h / 2
c_bot = yc - content_h / 2
TITLE_Ym = c_top + 0.032                               # math line, close to the content
TITLE_Yn = c_top + 0.112                               # name line, on top
ROLE_Y = c_bot - 0.13

# tinted station cards (MOLLEO): content + title live on a light rounded card
CARD = ["#f5f5f3", "#f0eef8", "#fdf1ec"]               # neutral / violet / neutral tints
CARD_PAD = 0.014
CARD_B = c_bot - 0.075                                 # deep bottom strip: air for the badges
CARD_T = TITLE_Yn + 0.080
for i, col in enumerate(CARD):
    fig.patches.append(FancyBboxPatch(
        (xs[i] - CARD_PAD, CARD_B), AXW[i] + 2 * CARD_PAD, CARD_T - CARD_B,
        boxstyle="round,pad=0,rounding_size=0.016", transform=fig.transFigure,
        facecolor=col, edgecolor="none", zorder=-2, mutation_aspect=FW / FH))

def title(i, name, math=None, math_color=INK):
    fig.text(xs[i] + AXW[i] / 2, TITLE_Yn, name, fontsize=FS_TITLE, color=INK,
             ha="center", va="baseline")
    if math:
        fig.text(xs[i] + AXW[i] / 2, TITLE_Ym, math, fontsize=FS_MATH, color=math_color,
                 ha="center", va="baseline")

def role(i, txt, color=INK2):
    fig.text(xs[i] + AXW[i] / 2, ROLE_Y, r"\textsc{%s}" % txt, fontsize=FS_ROLE,
             color=color, ha="center", va="baseline", zorder=6)

def badge(i, txt, color, y=None):
    fig.text(xs[i] + AXW[i] / 2, CARD_B + 0.022 if y is None else y,
             r"\textsc{%s}" % txt, fontsize=5.2,
             color=color, ha="center", va="baseline", zorder=6)

# ------------------------------------------------------- rendered ball-and-stick molecule
def shaded_sphere(ax, xy, r, base, z=10, alpha=1.0, halo=False):
    base = np.array(to_rgb(base))
    if halo:
        ax.add_patch(Circle(xy, r * 1.12, facecolor=SURFACE, edgecolor="none",
                            zorder=z - 1, alpha=0.9 * alpha))
    K = 22
    ax.add_patch(Circle(xy, r, facecolor=0.55 * base, edgecolor="none", zorder=z, alpha=alpha))
    for i in range(K):
        t = i / (K - 1)
        rr = r * (1.0 - 0.92 * t)
        off = np.array([-0.32, 0.34]) * r * t
        col = base + (np.array([1.0, 1.0, 1.0]) - base) * (0.12 + 0.75 * t ** 1.6)
        ax.add_patch(Circle((xy[0] + off[0], xy[1] + off[1]), rr,
                            facecolor=np.clip(col, 0, 1), edgecolor="none",
                            zorder=z + 1 + i, alpha=alpha))

def draw_molecule(ax, scale=1.0, alpha=1.0, z=10, halo=False):
    effects = [pe.withStroke(linewidth=4.6 * scale, foreground=SURFACE)] if halo else None
    for h in range(len(at2)):
        if h == iO:
            continue
        ax.plot([at2[iO, 0], at2[h, 0]], [at2[iO, 1], at2[h, 1]], color=BOND,
                linewidth=3.2 * scale, solid_capstyle="round", zorder=z - 3, alpha=alpha,
                path_effects=effects)
        ax.plot([at2[iO, 0], at2[h, 0]], [at2[iO, 1], at2[h, 1]], color="#d8d7d2",
                linewidth=1.7 * scale, solid_capstyle="round", zorder=z - 2, alpha=alpha)
    for (x, y), Z in zip(at2, charges):
        r = (0.42 if Z > 2 else 0.27) * scale
        shaded_sphere(ax, (x, y), r, O_RED if Z > 2 else H_GRAY, z=z, alpha=alpha, halo=halo)

# card 1: the fixed GTO basis — every cc-pVDZ primitive as a 1-sigma ring at its nucleus.
# No elliptical window on either panel: everything shows, clipped at the panel rectangle.
ax = axes[0]
sig = 1.0 / np.sqrt(2.0 * GTO["alpha"])
for m in np.argsort(-sig):
    xy = at2[int(GTO["atom"][m])]
    ax.add_patch(Circle(xy, sig[m], facecolor=GTO_GRAY, alpha=0.05, edgecolor="none",
                        zorder=20))
    ax.add_patch(Circle(xy, sig[m], facecolor="none", edgecolor=GTO_GRAY, linewidth=0.4,
                        alpha=0.55, zorder=21))
draw_molecule(ax, scale=0.80, alpha=0.95, z=40, halo=True)
title(0, r"Fixed Gaussian Basis", r"$\phi_\mu(\mathbf{r} - \mathbf{R}_a)$", GTO_GRAY)
role(0, "gaussian-type orbitals")
badge(0, "pinned to atoms, isotropic", INK2)

# card 2: the splat cloud, same molecule, trained ---------------------------------------
ax = axes[1]
Sig = np.linalg.inv(A)
depth = np.exp(-0.5 * (dperp / 0.9) ** 2)
eigs = [np.linalg.eigh(B.T @ Sig[m] @ B) for m in range(len(centers))]
area = np.array([np.sqrt(w[0] * w[1]) for w, _ in eigs])
compact = np.clip((1.0 / area) / np.percentile(1.0 / area, 75), 0, 1)
for m in np.argsort(-area):
    w, V = eigs[m]
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    a_m = float(depth[m])
    ax.add_patch(Ellipse(c2[m], 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang,
                         facecolor=VIOLET, alpha=(0.04 + 0.11 * float(compact[m])) * a_m,
                         edgecolor="none", zorder=20))
    ax.add_patch(Ellipse(c2[m], 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]), angle=ang,
                         facecolor="none", edgecolor=VIOLET, linewidth=0.35,
                         alpha=0.30 * a_m, zorder=21))
draw_molecule(ax, scale=0.80, alpha=0.95, z=40, halo=True)
title(1, r"Splat Cloud", r"$\Theta = \{(\mathbf{m}_\mu, \mathbf{q}_\mu, \boldsymbol{\ell}_\mu)\}$", VIOLET)
role(1, "gs-dft (ours)", color=VIOLET)
badge(1, "floating, anisotropic, learned", VIOLET)

# the comparison marker, in the gap ----------------------------------------------------
# the divider, then card 3: the payoff — accuracy per parameter (real record) -----------
xdiv = (xs[1] + AXW[1] + CARD_PAD + xs[2] - CARD_PAD) / 2
fig.add_artist(plt.Line2D([xdiv, xdiv], [CARD_B + 0.02, CARD_T - 0.02],
                          transform=fig.transFigure, color="#c9c8c4", lw=0.7,
                          ls=(0, (1, 2.2)), solid_capstyle="round"))

_m = _csv("memory")
MEM = {f"{k}_{c}": _m[col][_m["series"] == k].astype(float)
       for k in ("dense", "fast") for c, col in (("x", "params"), ("y", "peak_mb"))}
MEM["dense_oom"] = _m["params"][_m["series"] == "dense_oom"].astype(float)
MEM["vram_mb"] = float(_m["peak_mb"][_m["series"] == "card"][0])
XL0 = xs[2] + 0.030
GUT = 0.052
CW = (AXW[2] - 0.030 - 0.010 - GUT) / 2
PB, PT = c_bot + 0.010, TITLE_Yn - 0.075                # plots rise toward the card title
subA = fig.add_axes([XL0, PB, CW, PT - PB])             # accuracy, left column
subB = fig.add_axes([XL0 + CW + GUT, PB, CW, PT - PB])  # memory, right column
for s in (subA, subB):
    s.set_facecolor("none")
    s.set_xscale("log"); s.set_yscale("log")
    s.set_xticks([]); s.set_yticks([])
    s.minorticks_off()
    for side in ("top", "right"):
        s.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        s.spines[side].set_visible(True)
        s.spines[side].set_linewidth(0.6); s.spines[side].set_color(INK2)
fig.text(XL0 + CW / 2, PT + 0.012, r"\textsc{accuracy}", fontsize=5.6, color=INK,
         ha="center", va="baseline", zorder=6)
fig.text(xs[2] + 0.011, (PB + PT) / 2, r"Energy error (mHa)", fontsize=5.0,
         color=INK2, ha="center", va="center", rotation=90, zorder=6)
fig.text(XL0 + CW + GUT + CW / 2, PT + 0.012, r"\textsc{memory}", fontsize=5.6,
         color=INK, ha="center", va="baseline", zorder=6)
fig.text(XL0 + CW + GUT - 0.016, (PB + PT) / 2, r"Peak memory (MB)", fontsize=5.0,
         color=INK2, ha="center", va="center", rotation=90, zorder=6)

# accuracy per parameter (record: exp5 ladder, best seed, CBS floor)
gpx, gpy, spx, spy = SC["gx"], SC["gy"], SC["sx"], SC["sy"]
# No hollow markers any more: each ladder is measured against ITS OWN limit, as in Fig 5, so
# nothing lands below the reference and every rung enters the fit.
for px, py, col in [(gpx, gpy, GTO_GRAY), (spx, spy, VIOLET)]:
    a, b = np.polyfit(np.log10(px), np.log10(py), 1)
    xf = np.geomspace(px.min() / 1.5, px.max() * 1.5, 40)
    subA.plot(xf, 10.0 ** (a * np.log10(xf) + b), ls=":", color=col, lw=0.8, zorder=2)
subA.plot(gpx, gpy, "o-", color=GTO_GRAY, ms=2.2, lw=0.9, zorder=4)
subA.plot(spx, spy, "-", color=VIOLET, lw=0.9, zorder=5)
subA.plot(spx, spy, "s", color=VIOLET, ms=2.2, zorder=6)
subA.set_xlim(gpx.min() / 1.8, max(gpx.max(), spx.max()) * 1.8)
yv = np.concatenate([gpy, spy])
subA.set_ylim(yv.min() / 7.0, yv.max() * 1.8)
subA.text(0.45, 0.70, r"\textsc{splats}", transform=subA.transAxes, ha="left",
          va="center", fontsize=5.0, color=VIOLET, zorder=7,
          bbox=dict(facecolor="#fdf1ec", edgecolor="none", pad=1.2))
subA.text(0.10, 0.32, r"\textsc{cc-pV$X$Z}", transform=subA.transAxes, ha="left",
          va="center", fontsize=5.0, color=GTO_GRAY, zorder=7,
          bbox=dict(facecolor="#fdf1ec", edgecolor="none", pad=1.2))
# 78 = 2 x 39 occupied orbitals: BOTH plots are the alanine-dipeptide record
subA.text(0.97, 0.95, r"\textsc{78 electrons}", transform=subA.transAxes,
          ha="right", va="top", fontsize=5.0, color=INK2, zorder=7,
          bbox=dict(facecolor="#fdf1ec", edgecolor="none", pad=1.2))

# peak training memory (record: exp1 panel 2, exact vs DF+screen, same molecule)
vram = float(MEM["vram_mb"])
subB.plot(MEM["dense_x"], MEM["dense_y"], "o:", color=INK2, ms=2.2, lw=0.9, zorder=4)
subB.plot(MEM["fast_x"], MEM["fast_y"], "s-", color=VIOLET, ms=2.2, lw=0.9, zorder=5)
x_oom = float(MEM["dense_oom"][0])
subB.plot([MEM["dense_x"][-1], x_oom], [MEM["dense_y"][-1], vram], ls=":", color=INK2,
          lw=0.8, alpha=0.55, zorder=3)
subB.plot([x_oom], [vram], "x", color="#c0392b", ms=5.5, mew=1.3, zorder=6)
subB.axhline(vram, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=2)
subB.set_xlim(min(MEM["dense_x"].min(), MEM["fast_x"].min()) / 1.7,
              MEM["fast_x"].max() * 1.7)
subB.set_ylim(MEM["fast_y"].min() / 3.5, vram * 6.5)
subB.annotate(r"80 GB", (0.02, vram), xycoords=("axes fraction", "data"),
              textcoords="offset points", xytext=(0, 2), ha="left", va="bottom",
              fontsize=4.8, color=INK)
subB.text(0.24, 0.40, r"$\mathcal{O}(B^3)$", transform=subB.transAxes, ha="center",
          va="center", fontsize=5.4, color=INK2)
subB.text(0.78, 0.28, r"$\mathcal{O}(M^2)$", transform=subB.transAxes, ha="center",
          va="top", fontsize=5.4, color=VIOLET)

fig.text(XL0 + CW / 2, CARD_B + 0.022, r"Free parameters", fontsize=5.0,
         color=INK2, ha="center", va="baseline", zorder=6)
fig.text(XL0 + CW + GUT + CW / 2, CARD_B + 0.022, r"Free parameters", fontsize=5.0,
         color=INK2, ha="center", va="baseline", zorder=6)
title(2, r"Scaling")
role(2, "result")

fig.savefig("experiments/shared/visual_abstract.pdf")
fig.savefig("experiments/shared/visual_abstract.png", dpi=330)
print("wrote experiments/shared/visual_abstract.pdf + png")
