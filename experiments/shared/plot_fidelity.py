"""Figure 4: the fluoride anion and the LiF dissociation curve.

    left    bare F-               energy against free parameters: splats, cc-pVXZ and aug-cc-pVXZ
    right   LiF at matched count  energy against bond length, M = nao(cc-pVTZ) = 60

Parameters are counted on both sides with ``common.resources``: exponents and contraction
coefficients for the Gaussian basis, 9 per function for the splat chart, and the coefficient block
for both::

    TUEPLOTS=1 USETEX=1 uv run python -m experiments.shared.plot_fidelity

Reads ``data/figures/fig4_anions.csv`` and ``data/figures/fig4_dissociation.csv``.
"""
import collections
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.common import resources, systems
from experiments.common.results import iter_results
from experiments.common import cbs as _cbs
from experiments.common.style import (apply as apply_style, PAPER_FIG_HEIGHT_IN,
                                       save_at_height)

REL_W = 1.0
FIG_H = PAPER_FIG_HEIGHT_IN * 1.00

HERE = os.path.dirname(os.path.abspath(__file__))
EXP3 = os.path.join(HERE, "..", "..", "data", "figures", "fig4_dissociation.csv")
EXP4 = os.path.join(HERE, "..", "..", "data", "figures", "fig4_anions.csv")

OURS = "#5b2d91"        # the cloud, in every panel
BASE = "#9a9a9a"        # the Gaussian reference
INK = "#333333"
FAIL = "#c23b22"      # the unaugmented control, where the fixed basis fails

# The augmented ladder, in ascending function count. These are the sets the cloud is asked to match
# and it is given none of their diffuse primitives.
AUG = ["aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "aug-cc-pv5z"]
PLAIN = ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"]

CARDINAL_M = {"f_anion": [14, 30, 55, 91], "oh_anion": [19, 44, 85, 146]}

# Occupied orbitals per system: both bare anions carry 10 electrons. This sets the coefficient block
# on BOTH sides (M x n_occ and nao x n_occ), so a wrong value moves every point on the params axis.
NOCC = {"f_anion": 5, "oh_anion": 5}

SPREAD_TOL = 10.0       # mHa; a rung whose seeds disagree by more than this is an optimization
                        # failure, not a measurement, and is drawn hollow rather than quoted


def _energy(row):
    """The quotable energy of a splat row: the level-5 re-evaluation when it exists, else the raw E."""
    e5 = _v(row, "E_grid5", r"-?[\d.]+")
    return float(e5) if e5 is not None else float(_v(row, "E", r"-?[\d.]+"))


def _converged(splat, spread):
    """The rungs whose seeds agree, in ascending M."""
    return [m for m in sorted(splat) if spread.get(m, 0.0) <= SPREAD_TOL]


def _v(row, key, pat=r"\S+", default=None):
    """The leading match of ``pat`` in ``row[key]``, else ``default``."""
    m = re.match(pat, row.get(key, ""))
    return m.group(0) if m else default


def _fields(row):
    """A result row without its bookkeeping columns, for the accessors in `common.resources`."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


_BD_CACHE = {}


def _basis_data(system, basis):
    """Cached `BasisData` for (system, basis) -- needed to price the TABULATED half of a Gaussian
    calculation's parameters. Cached because building it is not free and the ladders reuse it."""
    key = (system, basis)
    if key not in _BD_CACHE:
        from dftax.basis.loader import build_basis_data
        mol = systems.molecule(system=system, basis=basis)
        _BD_CACHE[key] = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    return _BD_CACHE[key]


def _rows(path):
    yield from iter_results(path)


def anion(system="f_anion"):
    """The three ladders in the (RESOURCE, energy) plane rather than the (functions, energy) one.

    Returns ``(splat, plain, aug, budgets)`` where each ladder is a list of dicts carrying
    ``params`` / ``peak_mb`` / ``E`` (plus ``M`` or ``nao``, and ``spread`` for the cloud)."""
    seeds, seed_mem, gto = collections.defaultdict(list), {}, {}
    for ln in _rows(EXP4):
        if _v(ln, "system") != system:
            continue
        if "basis" in ln:
            b = _v(ln, "basis")
            gto[b] = {"nao": int(_v(ln, "nao", r"\d+", 0)),
                      "E": float(_v(ln, "E", r"-?[\d.]+")),
                      "peak_mb": resources.peak_mb_from_row(_fields(ln))}
            continue
        M = _v(ln, "M", r"\d+")
        if not M:
            continue
        key = (int(M), int(_v(ln, "steps", r"\d+", 0)))
        seeds[key].append(_energy(ln))
        mb = resources.peak_mb_from_row(_fields(ln))
        if mb is not None:
            seed_mem[key] = max(seed_mem.get(key, 0.0), mb)
    budgets = {}
    for M, b in seeds:
        budgets[M] = max(budgets.get(M, 0), b)
    n_occ = NOCC[system]
    splat = []
    for M in sorted(budgets):
        vals = seeds[(M, budgets[M])]
        splat.append({"M": M, "E": min(vals), "spread": (max(vals) - min(vals)) * 1e3,
                      "params": resources.splat_params(M, n_occ)["total"],
                      "peak_mb": seed_mem.get((M, budgets[M]))})

    def _gto_ladder(names):
        out = []
        for b in names:
            if b not in gto:
                continue
            bd = _basis_data(system, b)
            out.append({"basis": b, "nao": gto[b]["nao"], "E": gto[b]["E"],
                        "peak_mb": gto[b]["peak_mb"],
                        "params": resources.gto_params(bd, gto[b]["nao"], n_occ)["total"]})
        return sorted(out, key=lambda r: r["params"])
    return splat, _gto_ladder(PLAIN), _gto_ladder(AUG), budgets


def dissociation(mol="lif", M=60, steps=12000, cmp="aug-cc-pvtz", ref="aug-cc-pvqz",
                 plain="cc-pvtz"):
    """Raw $E(R)$ for the cloud and for the augmented reference.

    Plotted as absolute energies, the conventional dissociation curve: the shape of the curve is
    itself part of the claim, and subtracting a reference hides it. The margin is a few mHa on a
    well ~110 mHa deep, so the gap is shaded rather than left to the reader's eye.
    `ref` is still read, but only to test the baselines for variational consistency below.
    """
    sp, gt, rf, pl, lad = {}, {}, {}, {}, collections.defaultdict(dict)
    for ln in _rows(EXP3):
        m = (_v(ln, "mol") or _v(ln, "system") or "").split("_")[0]
        R = _v(ln, "R", r"[\d.]+")
        if m != mol or not R:
            continue
        R = float(R)
        if "basis" in ln:
            b, e = _v(ln, "basis"), float(_v(ln, "E", r"-?[\d.]+"))
            lad[R][b] = e
            if b == cmp:
                gt[R] = e
            elif b == ref:
                rf[R] = e
            elif b == plain:
                pl[R] = e
            continue
        if int(_v(ln, "M", r"\d+", 0) or 0) != M or int(_v(ln, "steps", r"\d+", 0)) != steps:
            continue
        E = _energy(ln)                    # the level-5 re-evaluation; see the note on _energy
        if sp.get(R) is None or E < sp[R]:
            sp[R] = E
    def _monotone(g):
        seq = [g[b] for b in AUG if b in g]
        return len(seq) >= 2 and all(a >= b for a, b in zip(seq, seq[1:]))

    have = set(sp) & set(gt) & set(rf) & set(pl)
    Rs = sorted(r for r in have if _monotone(lad[r]))
    dropped = sorted(r for r in have if not _monotone(lad[r]))
    if dropped:
        print(f"  excluded {mol} geometries with a non-variational GTO ladder: {dropped}")
    return Rs, [sp[r] for r in Rs], [gt[r] for r in Rs], [pl[r] for r in Rs]


def _anion_panel(ax, splat, plain, aug, xkey, usetex, e_ref):
    """Draw the three ladders against ``xkey`` ("params" or "peak_mb")."""
    def _y(rows):
        return [(r["E"] - e_ref) * 1e3 for r in rows]

    missing = 0
    for rows, style, colour, lab in (
            (plain, "^:", FAIL, "cc-pV$X$Z (no diffuse)"),
            (aug, "o--", BASE, "aug-cc-pV$X$Z")):
        xs = [r[xkey] for r in rows if r.get(xkey) is not None]
        ys = _y([r for r in rows if r.get(xkey) is not None])
        missing += sum(1 for r in rows if r.get(xkey) is None)
        if xs:
            ax.plot(xs, ys, style, color=colour, ms=3.5, lw=1.2, zorder=3, label=lab)

    ok = [r for r in splat if r["spread"] <= SPREAD_TOL and r.get(xkey) is not None]
    missing += sum(1 for r in splat if r["spread"] <= SPREAD_TOL and r.get(xkey) is None)
    ok.sort(key=lambda r: r[xkey])
    if ok:
        xs = [r[xkey] for r in ok]
        ax.errorbar(xs, _y(ok), yerr=[r["spread"] for r in ok],
                    fmt="none", ecolor=OURS, elinewidth=1.0, capsize=2.0, zorder=5)
        ax.plot(xs, _y(ok), "s-", color=OURS, ms=3.5, lw=1.4, zorder=4,
                label="Splats (no diffuse)")
    return missing


def main():
    usetex = apply_style(nrows=1, ncols=2, rel_width=REL_W, height_in=FIG_H)
    splat, plain, aug, budgets = anion()
    fig, (axL, axR) = plt.subplots(1, 2)
    _eng = fig.get_layout_engine()
    if _eng is not None:
        # h_pad near zero: the legend sits just above the canvas, and the layout engine's
        # default vertical padding opened a band of empty page between it and the axes.
        _eng.set(w_pad=0.04, wspace=0.10, h_pad=0.01)

    _lim = _cbs.extrapolate({r["basis"].replace("aug-", ""): r["E"] for r in aug})
    print(f"  anion CBS: {_lim['kind']} {_lim['value']:.6f} from cardinals {_lim['cardinals']}")
    dropped = _anion_panel(axL, splat, plain, aug, "params", usetex, _lim["value"])
    axL.set_xlabel("Free parameters")
    # The caption defines the symbol; spelling it out here only costs panel width.
    axL.set_ylabel(r"$\Delta_{\mathrm{CBS}}$ (mHa)")
    xs_all = [r["params"] for r in plain + aug] + \
             [r["params"] for r in splat if r["spread"] <= SPREAD_TOL]
    ys_all = [(r["E"] - _lim["value"]) * 1e3 for r in plain + aug] + \
             [(r["E"] - _lim["value"]) * 1e3 for r in splat if r["spread"] <= SPREAD_TOL]
    xspan = max(xs_all) - min(xs_all)
    axL.set_xlim(min(xs_all) - 0.08 * xspan, max(xs_all) + 0.06 * xspan)
    _lt = 1.0
    axL.set_yscale("symlog", linthresh=_lt, linscale=0.35)
    axL.set_ylim(min(min(ys_all) * 1.6, -2 * _lt), max(ys_all) * 1.8)
    axL.axhline(0.0, color="#c9c9c9", lw=0.6, zorder=1)
    axL.yaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(base=10.0, linthresh=_lt))
    axL.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _p: "" if abs(v) < _lt * 0.999 else
        rf"${'-' if v < 0 else ''}10^{{{int(round(np.log10(abs(v))))}}}$"))

    # ---- right: dissociation, in (bond length, energy) ----------------------------------------
    Rs, Es, Eg, Ep = dissociation()
    axR.fill_between(Rs, Es, Eg, color=OURS, alpha=0.12, zorder=2)   # the margin, made visible
    axR.plot(Rs, Ep, "^:", color=FAIL, ms=3.0, lw=1.1, zorder=3,
             label="cc-pVTZ (60 fn)")
    axR.plot(Rs, Eg, "o--", color=BASE, ms=3.0, lw=1.2, zorder=4,
             label="aug-cc-pVTZ (92 fn)")
    axR.plot(Rs, Es, "s-", color=OURS, ms=3.0, lw=1.4, zorder=5,
             label="Splats (60 fn)")
    axR.set_xlabel(r"$R$ (\AA)" if usetex else "R (A)")
    axR.set_ylabel("$E$ (Ha)")
    # No in-panel headroom reserved any more: the legend moved out of the axes, so the well can
    # use the full height.
    rlo, rhi = min(min(Es), min(Eg), min(Ep)), max(max(Es), max(Eg), max(Ep))
    axR.set_ylim(rlo - 0.08 * (rhi - rlo), rhi + 0.06 * (rhi - rlo))

    _h = [Line2D([0], [0], color=FAIL, marker="^", ls=":", ms=3.0, lw=1.1,
                 label=r"cc-pV$X$Z"),
          Line2D([0], [0], color=BASE, marker="o", ls="--", ms=3.0, lw=1.2,
                 label=r"aug-cc-pV$X$Z"),
          Line2D([0], [0], color=OURS, marker="s", ls="-", ms=3.0, lw=1.4, label="Splats")]
    fig.legend(handles=_h, fontsize=5.5, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0, handlelength=1.5,
               columnspacing=0.9, handletextpad=0.35)

    # Axis text a touch under the bundle default; the panels carry a lot of tick labels.
    for _ax in (axL, axR):
        _ax.xaxis.label.set_size(7.0)
        _ax.yaxis.label.set_size(7.0)
        _ax.tick_params(labelsize=6.0, which="both")

    # Bold panel letters leading the x-axis label: no subplot titles by design (see above), and
    # the label row is the one place a letter costs neither data nor height.
    for _i, _ax in enumerate((axL, axR)):
        _l = "ab"[_i]
        _pre = rf"\textbf{{{_l})}} " if usetex else rf"$\mathbf{{{_l}}}$) "
        _ax.set_xlabel(_pre + _ax.get_xlabel())

    for ax in (axL, axR):
        ax.grid(axis="y", color="#e8e8e8", lw=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "fidelity")
    save_at_height(fig, out + ".png", height_in=FIG_H, dpi=200)
    save_at_height(fig, out + ".pdf", height_in=FIG_H)
    print(f"wrote {out}.{{png,pdf}}  (usetex={usetex})")
    ok = sorted((r for r in splat if r["spread"] <= SPREAD_TOL), key=lambda r: r["params"])
    bad = [r for r in splat if r["spread"] > SPREAD_TOL]
    print(f"  anion   : {len(ok)} splat points drawn, {len(plain)} plain + {len(aug)} augmented "
          f"rungs; {len(bad)} rung(s) excluded on seed spread; {dropped} point(s) lacked an x value")
    for r in bad:
        print(f"    EXCLUDED M={r['M']:>4}  spread {r['spread']:7.2f} mHa (> {SPREAD_TOL})")
    for r in ok:
        beat = [g for g in aug if g["E"] > r["E"] and g["params"] >= r["params"]]
        note = (f"beats {beat[0]['basis']:>12} ({beat[0]['params']:>5} p) by "
                f"{(beat[0]['E'] - r['E']) * 1e3:6.1f} mHa at "
                f"{100 * r['params'] / beat[0]['params']:5.1f}% of its parameters"
                if beat else "beats no augmented rung at or above its budget")
        print(f"    M={r['M']:>4}  {r['params']:>5} params  E={r['E']:.6f}  {note}")
    print(f"  lif     : {len(Rs)} geometries")
    for R, es, eg, ep in zip(Rs, Es, Eg, Ep):
        print(f"    R={R:>4.1f}  vs aug-TZ {(es - eg) * 1e3:+7.1f}   "
              f"vs plain TZ {(es - ep) * 1e3:+7.1f} mHa")


if __name__ == "__main__":
    main()
