"""Figure 5: energy error against total free parameters, per system, with the fitted scaling laws.

    TUEPLOTS=1 USETEX=1 uv run python -m experiments.shared.plot_isoparams

Reads ``data/figures/fig5_isoparams.csv``.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.common import ladder, resources, systems
from experiments.common.style import (apply as apply_style, PAPER_FIG_HEIGHT_IN,
                                       save_at_height)

FIG_H = PAPER_FIG_HEIGHT_IN * 0.92   # ONE row, with the band above the panels closed
_USETEX = apply_style(nrows=1, ncols=3, rel_width=1.0, height_in=FIG_H)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "data", "figures", "fig5_isoparams.csv")
NOCC = {"water": 5, "ethanol": 13, "alanine_dipeptide": 39}
TITLE = {"water": r"$\mathrm{H_2O}$", "ethanol": r"$\mathrm{C_2H_5OH}$",
         "alanine_dipeptide": "Ala dipeptide"}

FLOOR = 0.04      # mHa; the smallest error the FIT will take (a power law needs y > 0)
LINTHRESH = 1.0   # mHa; symlog linear zone, so signed sub-mHa errors have somewhere to sit

#: Rungs dropped from the splat curve, with the reason.
EXCLUDE: dict[tuple[str, int], str] = {}

#: Every rung trains for the same number of steps, or the curve would mix parameters with optimizer
#: effort. The 48,000-step functional sweep in the same directory is excluded by this budget.
BUDGET = 12000


def _symlog_label(v):
    """Tick label for the symlog error axis: powers of ten, signed."""
    import math
    e = int(round(math.log10(abs(v))))
    sign = "-" if v < 0 else ""
    return rf"${sign}10^{{{e}}}$"


def _fit_band(ax, xs, ys, color, zorder, trans, per_dec, ext=2.5, alt=None, k=1.0,
              ylim=None):
    """Dotted straight-line fit PLUS its prediction band, fitted in the AXIS'S OWN y coordinate.

    `per_dec` is how many transform units one decade of |y| spans, so dividing the fitted slope by
    it turns the slope back into an exponent. On a log axis it is 1; on symlog it is `linthresh`.

    The band is the forecast interval of the regression: `polyfit(..., cov=True)` gives the
    covariance of (slope, intercept), and the variance of the prediction at x is v(x)ᵀ C v(x) with
    v = [log x, 1]. It is narrow through the data and FLARES where the line extrapolates, which is
    where the scaling claim lives. `alt` is the same ladder against the alternative limit; given
    it, the band is the ENVELOPE of both fits, so it carries the reference's own uncertainty too."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xf = np.geomspace(min(xs) / ext, max(xs) * ext, 300)
    lf = np.log10(xf)

    def _one(xx, yy):
        xx, yy = np.asarray(xx, float), np.asarray(yy, float)
        ty = np.asarray(trans.transform(yy), float).ravel()
        m = np.isfinite(ty) & np.isfinite(xx) & (xx > 0)
        if m.sum() < 3:
            return None
        c, cov = np.polyfit(np.log10(xx[m]), ty[m], 1, cov=True)
        var = cov[0, 0] * lf ** 2 + 2.0 * cov[0, 1] * lf + cov[1, 1]
        return (c[0] * lf + c[1], np.sqrt(np.maximum(var, 0.0)),
                c[0] / per_dec, float(np.sqrt(max(cov[0, 0], 0.0))) / abs(per_dec))

    base = _one(xs, ys)
    if base is None:
        return None
    mu, sd, a, sa = base
    lo, hi = mu - k * sd, mu + k * sd
    if alt is not None:
        other = _one(*alt)
        if other is not None:
            mu2, sd2, _a2, _s2 = other
            lo, hi = np.minimum(lo, mu2 - k * sd2), np.maximum(hi, mu2 + k * sd2)
    if ylim is not None:
        t0, t1 = sorted(np.asarray(trans.transform(np.asarray(ylim, float))).ravel())
        lo, hi = np.clip(lo, t0, t1), np.clip(hi, t0, t1)
    inv = trans.inverted().transform
    ax.fill_between(xf, np.asarray(inv(lo)).ravel(), np.asarray(inv(hi)).ravel(),
                    color=color, alpha=0.22, lw=0, zorder=zorder - 1)
    ax.plot(xf, np.asarray(inv(mu)).ravel(), ls=":", color=color, lw=1.0, zorder=zorder)
    return a, sa


def exp_legend(ax, a_g, a_s, loc, text=True):
    """second legend: each fitted exponent. Each argument is the ``(alpha, sigma)`` pair
    `_fit_band` returns, or None; only alpha is shown -- the band already carries the spread."""
    pg, ps = ("cc-pVXZ: ", "Splats: ") if text else ("", "")
    h = []
    if a_g is not None:
        h.append(Line2D([0], [0], color="#9a9a9a", ls=":", lw=1.0,
                        label=fr"{pg}$\alpha_{{\mathrm{{GTO}}}}={-a_g[0]:.2f}$"))
    if a_s is not None:
        h.append(Line2D([0], [0], color="#5b2d91", ls=":", lw=1.0,
                        label=fr"{ps}$\alpha_{{\mathrm{{GS\text{{-}}DFT}}}}={-a_s[0]:.2f}$"))
    if not h:
        return None
    leg = ax.legend(handles=h, fontsize=5.2, frameon=True, facecolor="white", edgecolor="none",
                    framealpha=0.78, loc=loc, handlelength=0, handletextpad=0, labelspacing=0.3,
                    borderpad=0.2)
    for txt, hd in zip(leg.get_texts(), h):
        txt.set_color(hd.get_color())
    return leg


_NAO2BASIS = {}
_BD = {}


def _basis_data(system, nao):
    from dftax.basis.loader import build_basis_data
    if system not in _NAO2BASIS:
        m = {}
        for b in ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"):
            from gs_dft import nao as nao_of
            m[int(nao_of(systems.molecule(system=system, basis=b)))] = b
        _NAO2BASIS[system] = m
    basis = _NAO2BASIS[system][int(nao)]
    if (system, basis) not in _BD:
        mol = systems.molecule(system=system, basis=basis)
        _BD[(system, basis)] = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis,
                                                spherical=True)
    return _BD[(system, basis)]


def main(logdir=RESULTS):
    data = ladder.load(logdir, budget=BUDGET)
    present = ladder.systems_with_cbs(data)
    missing = [s for s in ("water", "ethanol", "alanine_dipeptide") if s not in present]
    if missing:
        print(f"  no CBS, panel omitted: {', '.join(missing)}")
    fig, axes = plt.subplots(1, len(present), squeeze=False)
    axes = axes[0]
    for ci, sysn in enumerate(present):
        ax = axes[ci]
        no = NOCC[sysn]
        gl = ladder.gto_ladder(data[sysn])
        gx = [resources.gto_params(_basis_data(sysn, B), B, no)["total"] for B, _ in gl]
        gy = [g for _, g in gl]                      # already (E - gto CBS)*1e3
        raw = ladder.splat_ladder(data[sysn], require_uniform_seeds=True)
        Ms = sorted(raw)
        sx = [resources.splat_params(M, no)["total"] for M in Ms]
        signed = [raw[M][1] for M in Ms]
        for M in Ms:
            if (sysn, M) in EXCLUDE:
                print(f"    {sysn}: M={M} excluded — {EXCLUDE[(sysn, M)]}")

        _px = {M: x for M, x in zip(Ms, sx)}
        fit_s = ladder.cbs_splat(data[sysn], xkey=_px.get)
        c_g, c_s = data[sysn]["cbs"], fit_s["value"]
        if c_s != c_s:
            print(f"    {sysn}: no splat limit; its curve keeps the Gaussian reference")
        off_s = (c_g - c_s) * 1e3 if c_s == c_s else 0.0
        sy = [y + off_s for y in signed]
        print(f"    {sysn:18s} gto CBS {c_g:.6f}   splat E_inf "
              + (f"{c_s:.6f}  p={fit_s['p']:.3f} R2={fit_s['r2']:.5f}  shift {off_s:+.3f} mHa"
                 if c_s == c_s else "UNAVAILABLE"))

        ax.set_xscale("log")
        # Both ladders now sit strictly above their own limit, so a plain log axis serves. The
        # guard is here because a symlog fallback is the only thing that can draw a non-positive
        # point at all, and silently dropping one would overstate the convergence.
        allv = [v for v in gy + sy if v == v]
        lt = (min([v for v in allv if v > 0]) / 5.0) if any(v > 0 for v in allv) else 1.0
        if min(allv) <= 0.0:
            print(f"    {sysn}: {sum(1 for v in allv if v <= 0)} point(s) not above their own "
                  f"limit; axis falls back to symlog")
            ax.set_yscale("symlog", linthresh=lt, linscale=0.35)
            ax.set_ylim(min(min(allv) * 1.6, -2 * lt), max(allv) * 1.6)
            ax.axhline(0.0, color="#c9c9c9", lw=0.6, zorder=1)
            ax.yaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(
                base=10.0, linthresh=lt))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _p, _lt=lt: "" if abs(v) < _lt * 0.999 else _symlog_label(v)))
        else:
            ax.set_yscale("log")
            ax.set_ylim(min(allv) / 2.0, max(allv) * 1.6)
            ax.yaxis.set_major_locator(
                matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=12))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        tr = ax.yaxis.get_transform()
        _pr = (np.array([lt * 10.0, lt * 100.0]) if min(allv) <= 0.0 else np.array([1.0, 10.0]))
        per_dec = float(np.diff(np.asarray(tr.transform(_pr)).ravel())[0])
        ylim = ax.get_ylim()

        a_g = (_fit_band(ax, gx, gy, "#9a9a9a", 2, tr, per_dec, ylim=ylim)
               if len(gx) >= 3 else None)
        fx = [x for M, x in zip(Ms, sx) if (sysn, M) not in EXCLUDE]
        fy = [y for M, y in zip(Ms, sy) if (sysn, M) not in EXCLUDE]
        a_s = (_fit_band(ax, fx, fy, "#5b2d91", 3, tr, per_dec, ylim=ylim)
               if len(fx) >= 3 else None)
        ax.plot(gx, gy, "o-", color="#9a9a9a", ms=3, lw=1.0, label="cc-pVXZ", zorder=4)
        ax.plot(sx, sy, "-", color="#5b2d91", lw=1.0, label="Splats", zorder=5)
        ax.plot(sx, sy, "s", color="#5b2d91", ms=3, zorder=6)
        # Bottom-left: every curve descends left-to-right, so the upper right is where
        # the fit lines and the converged end of both ladders actually are.
        leg_exp = exp_legend(ax, a_g, a_s, "lower left", text=False)
        if leg_exp is not None:
            leg_exp.set_zorder(8)
            ax.add_artist(leg_exp)
        ax.xaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
        ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _l = "abcdefgh"[ci]
        tag = rf"\textbf{{{_l})}}" if _USETEX else rf"$\mathbf{{{_l}}}$)"
        ax.set_xlabel("Free parameters")
        ax.set_title(f"{tag} {TITLE[sysn]}", fontsize=7, pad=2.0)   # default pad opens a gap
        ax.tick_params(axis="x", which="minor", labelsize=4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"$\Delta$ (mHa)")
    h_main = [Line2D([0], [0], color="#9a9a9a", marker="o", ls="-", ms=3, lw=1.0, label="cc-pVXZ"),
              Line2D([0], [0], color="#5b2d91", marker="s", ls="-", ms=3, lw=1.0, label="Splats")]
    fig.legend(handles=h_main, fontsize=6, frameon=False, ncol=2, loc="lower center",
               # 1.0, not 1.02: anchoring above the canvas floats the key clear of the
               # figure and the tight bbox then keeps the empty band between them.
               bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0, handlelength=1.6,
               columnspacing=1.6)
    fig.tight_layout(h_pad=0.2)
    fig.subplots_adjust(top=0.90)
    out = os.path.join(HERE, "isoparams")
    save_at_height(fig, out + ".png", height_in=FIG_H, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=FIG_H)
    print("wrote", out)


if __name__ == "__main__":
    main()
