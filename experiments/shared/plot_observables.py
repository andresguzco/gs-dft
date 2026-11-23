"""Figure 3: energy, density and force errors against resource, for three systems on shared axes.

The top row measures against the Gaussian ladder's limit, the bottom row against the splat ladder's
own limit, extrapolated with the same estimator from the splat rungs::

    TUEPLOTS=1 USETEX=1 uv run python -m experiments.shared.plot_observables

Reads ``data/figures/fig3_observables.csv`` and ``fig3_gto_gradients.csv`` and writes
``observables.{png,pdf}``. The bottom row comes from the ``kind=obs_selfref`` rows written by
``selfref.py``. ``main(floor_path=CC_FLOOR)`` adds a dashed floor to the density and force columns,
PBE against CCSD at cc-pVQZ (``fig3_cc_floor.csv``, ``rb_post.py ccfloor``): below it, the
representation error is smaller than the functional's. The paper's figure does not draw it.

The same figure for the rebuttal systems (``--rebuttal``) reads ``data/rebuttal/fig3_observables.csv``
and ``fig3_gto_gradients.csv`` and measures the splat ladders against their power-law limit (the
rebuttal's estimator; the rungs are not matched to the Gaussian counts). It writes one figure,
``observables_sizes``, of four rows: the three smallest systems (CBS, ISC), then the three
largest, with one viridis colour per system by atom count::

    TUEPLOTS=1 USETEX=1 uv run python -m experiments.shared.plot_observables --rebuttal
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.common.results import STALE_RUNGS, XC, iter_results, read_records
from experiments.common import cbs, resources, systems
from experiments.common.style import (apply as apply_style, PAPER_FIG_HEIGHT_IN,
                                       save_at_height)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
EXP2 = os.path.join(DATA, "figures", "fig3_observables.csv")
CC_FLOOR = os.path.join(DATA, "figures", "fig3_cc_floor.csv")
RB_OBS = os.path.join(DATA, "rebuttal", "fig3_observables.csv")
RB_GRADIENTS = os.path.join(DATA, "rebuttal", "fig3_gto_gradients.csv")
RB_SYSTEMS = ("sf6", "cubane", "adenine_thymine", "porphine", "retinal", "cholesterol")
RB_OUT = os.path.join(HERE, "..", "rebuttal", "figures", "observables_sizes")

OURS, BASE, GRIDC, FLOOR = "#5b2d91", "#8c8c8c", "#e9e9e9", "#2a7f62"
NOCC = {"water": 5, "ethanol": 13, "alanine_dipeptide": 39}
MARK = {"water": "o", "ethanol": "^", "alanine_dipeptide": "s", "sf6": "o", "cubane": "^",
        "adenine_thymine": "s", "porphine": "D", "retinal": "v", "cholesterol": "p"}
LABEL = {"water": r"\ce{H2O}", "ethanol": r"\ce{C2H5OH}",
         "alanine_dipeptide": "Ala dipeptide", "sf6": r"\ce{SF6}", "cubane": "Cubane",
         "adenine_thymine": "A--T pair", "porphine": "Porphine", "retinal": "Retinal",
         "cholesterol": "Cholesterol"}
PLAIN_LABEL = {"water": r"H$_2$O", "ethanol": r"C$_2$H$_5$OH",
               "alanine_dipeptide": "Ala dipeptide", "sf6": r"SF$_6$", "cubane": "Cubane",
               "adenine_thymine": "A–T pair", "porphine": "Porphine", "retinal": "Retinal",
               "cholesterol": "Cholesterol"}

XKEY, XLAB = "params", "Free parameters"

# One column per observable, drawn once per row. The FIELD changes with the row -- top row against
# the Gaussian reference, bottom row against the splat one -- while the quantity and the axis do
# not, so a column is comparable top to bottom as well as left to right.
COLS = [(("dE_gto", "dE_splat"), r"$\Delta$ (mHa/atom)", "symlog"),
        (("tv_e", "tv_e_self"), r"TV / electron", "log"),
        (("dF_max", "dF_self"), r"$\Delta F$ (meV/$\mathrm{\AA}$)", "log")]
ROW_TAG = ["CBS", "ISC"]
LINTHRESH = 0.1   # mHa/atom; symlog linear zone on the energy panels


FIG_H = PAPER_FIG_HEIGHT_IN * 1.65   # two rows (Figure 3)
SIZES_H = PAPER_FIG_HEIGHT_IN * 1.45  # one two-row block of observables_sizes
STACK_H = 0.85                        # of the block height, per further two-row block (the legend is shared)
HA_BOHR_TO_MEV_A = 51422.08     # (27.211386 eV/Ha) / (0.529177 A/Bohr), x1000


_NATOMS = {}


def _natoms(sysn):
    if sysn not in _NATOMS:
        _NATOMS[sysn] = len(systems.molecule(system=sysn, basis="cc-pvdz").symbols)
    return _NATOMS[sysn]


def size_colors(names, hi=0.8):
    """One viridis colour per system by atom count (the yellow end left out: it fades on white)."""
    order = sorted(names, key=_natoms)
    cm = matplotlib.colormaps["viridis"]
    return {s: cm(hi * i / max(len(order) - 1, 1)) for i, s in enumerate(order)}


def _nocc(sysn):
    if sysn not in NOCC:
        NOCC[sysn] = int(systems.molecule(system=sysn, basis="cc-pvdz").nelectron) // 2
    return NOCC[sysn]


def _unit(field, sysn):
    """Multiplier carrying a stored value into the panel's plotted unit."""
    if field in ("dE_gto", "dE_splat"):
        # PER ATOM so the three systems, which differ 7x in atom count, share one axis.
        return 1.0 / _natoms(sysn)
    if field in ("dF_max", "dF_self", "maxF"):
        return HA_BOHR_TO_MEV_A
    return 1.0


def _fields(row):
    return dict(m.split("=", 1) for m in row.split() if "=" in m)


def _f(d, k):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


#: Later wins when the same rung reports twice. An unlabelled row predates the field.
_CKPT_RANK = {"raw": 0, None: 1, "polished": 2}

_BD = {}


def _basis_data(system, basis):
    if (system, basis) not in _BD:
        from dftax.basis.loader import build_basis_data
        mol = systems.molecule(system=system, basis=basis)
        _BD[(system, basis)] = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis,
                                                spherical=True)
    return _BD[(system, basis)]


def load(system, csv=EXP2, grads=None):
    """(gto rows by basis, splat rows by M), each decorated with its resource coordinates."""
    gto, splat, ckpt_peak, obs_rows = {}, {}, {}, {}
    if not os.path.exists(csv):
        return gto, splat
    for row in iter_results(csv):
        d = {k: v for k, v in row.items() if not k.startswith("_")}
        if d.get("system") != system:
            continue
        # One functional per panel. Every curve here is PBE, so a functional-sweep row would be
        # a different calculation landing on the same x coordinate. A row with no `xc` key
        # predates the field and is PBE.
        if str(d.get("xc", XC)).lower() != XC:
            continue
        if d.get("kind") == "gto":
            d["params"] = resources.gto_params(_basis_data(system, d["basis"]),
                                               int(d["nao"]), _nocc(system))["total"]
            d["peak_mb"] = resources.peak_mb_from_row(d)
            gto.setdefault(d["basis"], {}).update(d)
        elif d.get("kind") == "gtol1":
            gto.setdefault(d["basis"], {})["L1_rho"] = d["L1_rho"]
        elif d.get("kind") == "obs":
            if (system, int(d["M"])) in STALE_RUNGS:
                continue
            d["params"] = resources.splat_params(int(d["M"]), _nocc(system))["total"]
            d["peak_mb"] = resources.peak_mb_from_row(d)
            obs_rows.setdefault(int(d["M"]), []).append(d)
        elif d.get("kind") == "gto_selfref":
            # The second row's Gaussian curve: same reference rung, same two quantities.
            gto.setdefault(d["basis"], {}).update(
                {k: d[k] for k in ("L1_rho_self", "dF_self", "vs_M", "frac_cbs") if k in d})
        elif d.get("kind") == "obs_selfref":
            # The bottom row: this rung against the LARGEST splat rung. Merged onto the same
            # splat row so both rows share one x coordinate per rung.
            if (system, int(d["M"])) in STALE_RUNGS:
                continue
            splat.setdefault(int(d["M"]), {}).update(
                {k: d[k] for k in ("L1_rho_self", "dF_self", "vs_M", "frac_cbs") if k in d})
        elif d.get("kind") == "ckpt" and d.get("M") is not None:
            ckpt_peak[int(d["M"])] = resources.peak_mb_from_row(d)
    for M, rows in obs_rows.items():
        cur = splat.setdefault(M, {})
        for d in sorted(rows, key=lambda r: _CKPT_RANK.get(r.get("ckpt"), 1)):
            for k, v in d.items():
                if k in cur and _f(cur, k) == _f(cur, k) and _f(d, k) != _f(d, k):
                    continue
                cur[k] = v
    for M, d in splat.items():
        if d.get("peak_mb") is None and ckpt_peak.get(M) is not None:
            d["peak_mb"] = ckpt_peak[M]
    for b, v in gto_dF(system, grads or GTO_GRADIENTS).items():
        if b in gto:
            gto[b]["dF_max"] = v
    nelec = 2 * _nocc(system)
    for d in list(gto.values()) + list(splat.values()):
        for src, dst in (("L1_rho", "tv_e"), ("L1_rho_self", "tv_e_self")):
            l1 = _f(d, src)
            d[dst] = (l1 / (2.0 * nelec)) if (l1 == l1 and l1 > 0.0) else float("nan")
    return gto, splat


def _series(rows, field):
    """(xs, ys) for one ladder in one panel, dropping points with no x or no value."""
    xs, ys = [], []
    rows = [r for r in rows if r.get(XKEY) is not None]
    for r in sorted(rows, key=lambda r: float(r[XKEY])):
        x, v = r.get(XKEY), _f(r, field)
        if x is None or v != v:
            continue
        xs.append(x)
        ys.append(abs(v) if field in ("maxF", "dF_max", "dF_self") else v)
    return xs, ys


GTO_GRADIENTS = os.path.join(HERE, "..", "..", "data", "figures", "fig3_gto_gradients.csv")
_ORDER = ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"]


def gto_dF(system, path=GTO_GRADIENTS):
    """``{basis: max|F_basis - F_ref|}`` for the Gaussian ladder -- the top row's force baseline.

    The reference is the largest rung with a finite gradient. The gradients are the ``grad`` arrays
    of the ``ref_*.npz`` files written by `exp2_observables/gto_ref.py` and `cc_ref.py`."""
    rows = {}
    if not os.path.exists(path):
        return {}
    for r in read_records(path):
        if r["system"] == system:
            rows.setdefault(r["basis"], []).append([float(r[f"dE_d{a}"]) for a in "xyz"])
    grads = {b: np.asarray(g) for b, g in rows.items()}
    if len(grads) < 2:
        return {}
    ref_b = [b for b in _ORDER if b in grads][-1]
    return {b: float(np.max(np.abs(g - grads[ref_b])))
            for b, g in grads.items() if b != ref_b and g.shape == grads[ref_b].shape}


def splat_limit(gto, splat):
    """The splat ladder's own CBS, from the SAME estimator the Gaussian ladder uses.

    `cbs.extrapolate` is keyed by basis, so each splat rung is first matched to the cardinal whose
    function count it was chosen to match (the matched-count rule: M = nao(cardinal)). Feeding
    those energies through the identical three-point geometric fit gives a limit that is the splat
    analogue of the Gaussian CBS, not a different estimator applied to different data.
    """
    naos = {b: _f(g, "nao") for b, g in gto.items() if _f(g, "nao") == _f(g, "nao")}
    if not naos:
        return cbs.extrapolate({}), {}
    byb = {}
    for M, s in splat.items():
        e = _f(s, "E")
        if e != e:
            continue
        b = min(naos, key=lambda k: abs(naos[k] - M))
        if abs(naos[b] - M) > 0.15 * max(M, 1):
            continue                     # no cardinal this rung was matched to
        byb[b] = e
    return cbs.extrapolate(byb), byb


def splat_limit_powerlaw(splat):
    """The rebuttal's splat limit: ``E(M) = E_inf + A M^-p`` over all the rungs (at least four)."""
    Ms = sorted(M for M, s in splat.items() if _f(s, "E") == _f(s, "E"))
    lim = cbs.fit_powerlaw(Ms, [_f(splat[M], "E") for M in Ms])
    return dict(lim, kind=("cbs" if lim["kind"] == "fit" else "none")), {M: M for M in Ms}


def floors(path=CC_FLOOR):
    """``{system: (tv_e, dF)}`` of PBE against CCSD, in the columns' units (TV per electron, Ha/Bohr)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for r in iter_results(path, kind="cc_floor"):
        nelec = float(r["nelec"])
        out[r["system"]] = (float(r["L1_rho"]) / (2.0 * nelec), abs(float(r["dF_max"])))
    return out


def _fmt_log(ax):
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _decorate(ax, xlab, usetex, pad=None):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis="both", color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlab, labelpad=pad)
    ax.tick_params(labelsize=6, which="both")
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(6.5)


def main(names=("water", "ethanol", "alanine_dipeptide"), csv=EXP2, grads=GTO_GRADIENTS,
         out=None, estimator="matched", floor_path=None, legend_ncol=None, colors=None,
         groups=None, block_h=FIG_H):
    # colors: {system: colour} draws each system in its own colour (splats solid, Gaussians dashed);
    # None keeps the method colours of the paper's Figure 3.
    # groups: tuples of systems, each drawn as its own two-row block (CBS, ISC) under the same
    # columns, stacked top to bottom; None is one block of `names`.
    groups = [tuple(names)] if groups is None else [tuple(g) for g in groups]
    nr = 2 * len(groups)
    fig_h = block_h * (1.0 + STACK_H * (len(groups) - 1))
    usetex = apply_style(nrows=nr, ncols=len(COLS), rel_width=1.0, height_in=fig_h)
    lab = LABEL if usetex else PLAIN_LABEL
    fig, axes = plt.subplots(nr, len(COLS), squeeze=False)
    # The style bundle installs a layout engine, which ignores subplots_adjust and warns.
    _eng = fig.get_layout_engine()
    if _eng is not None:
        _eng.set(h_pad=0.02, w_pad=0.02, hspace=0.04, wspace=0.06)
    # Each block shares x between its two rows; the density and force columns share y throughout.
    for c in range(len(COLS)):
        for r in range(1, nr):
            if c > 0:
                axes[r][c].sharey(axes[0][c])
            if r % 2:
                axes[r][c].sharex(axes[r - 1][c])
        for r in range(0, nr, 2):
            axes[r][c].tick_params(labelbottom=False)

    present, notes, snotes, raw_rungs = [], {}, {}, []
    for system, r0 in [(s, 2 * b) for b, g in enumerate(groups) for s in g]:
        gto, splat = load(system, csv, grads)
        if not splat:
            continue
        present.append(system)
        notes[system] = cbs.extrapolate({b: _f(g, "E") for b, g in gto.items()})
        snotes[system], matched = (splat_limit_powerlaw(splat) if estimator == "powerlaw"
                                   else splat_limit(gto, splat))
        raw_rungs += [f"{system} M={M}" for M, d in sorted(splat.items())
                      if d.get("ckpt") == "raw"]
        # The residual each row plots. Both ladders get both references: the bottom row is the
        # same figure with the yardstick swapped, not a different set of curves.
        for d in list(gto.values()) + list(splat.values()):
            for key, lim in (("dE_gto", notes[system]), ("dE_splat", snotes[system])):
                e = _f(d, "E")
                d[key] = ((e - lim["value"]) * 1e3
                          if lim["kind"] != "none" and e == e else float("nan"))
        print(f"    {system:18s} matched {sorted(matched)} -> splat CBS "
              f"{snotes[system]['kind']}"
              + (f" {snotes[system]['value']:.6f}" if snotes[system]["kind"] != "none" else "")
              + f"   gto CBS {notes[system]['value']:.6f}")
        for rows, gauss, ls, lw in ((list(gto.values()), True, "--", 1.0),
                                    (list(splat.values()), False, "-", 1.3)):
            colour = colors[system] if colors else (BASE if gauss else OURS)
            alpha = 0.55 if colors and gauss else 1.0
            for r in range(2):
                for j, (fields, _ylab, _sc) in enumerate(COLS):
                    xs, ys = _series(rows, fields[r])
                    if not xs:
                        continue
                    u = _unit(fields[r], system)
                    axes[r0 + r][j].plot(xs, [y * u for y in ys], ls=ls, color=colour, lw=lw, alpha=alpha,
                                    zorder=3, marker=MARK[system], ms=2.8, mfc="white", mew=0.9)

    # The functional's own error: PBE against CCSD at cc-pVQZ, in the density and force columns.
    fl = {s: v for s, v in floors(floor_path).items() if s in present}
    block = {s: 2 * b for b, g in enumerate(groups) for s in g}
    edge = matplotlib.transforms.blended_transform_factory
    for s, (tv, dF) in fl.items():
        for r in (block[s], block[s] + 1):
            for j, v in ((1, tv), (2, dF * HA_BOHR_TO_MEV_A)):
                ax = axes[r][j]
                ax.axhline(v, color=FLOOR, ls=(0, (4, 2)), lw=0.8, zorder=2)
                ax.plot([0.96], [v], transform=edge(ax.transAxes, ax.transData), ls="none",
                        color=FLOOR, marker=MARK[s], ms=2.8, mfc="white", mew=0.9, zorder=4,
                        clip_on=False)
        print(f"    {s:18s} CCSD floor: TV/e {tv:.3e}, dF {dF * HA_BOHR_TO_MEV_A:.1f} meV/A")

    for r in range(nr):
        for j, (_fields, ylab, scale) in enumerate(COLS):
            ax = axes[r][j]
            _decorate(ax, XLAB if r == nr - 1 else "", usetex, pad=1.0 if r % 2 == 0 else None)
            _v = [y for ln in ax.get_lines()
                  for y in np.asarray(ln.get_ydata(), dtype=float) if np.isfinite(y)]
            _p = [v for v in _v if v > 0]
            if scale == "symlog" and _v and min(_v) <= 0.0:
                lt = min(_p) / 5.0 if _p else LINTHRESH
                ax.set_yscale("symlog", linthresh=lt, linscale=0.35)
                ax.set_ylim(min(min(_v) * 1.6, -2 * lt), max(_v) * 1.6)
                ax.axhline(0.0, color="#c9c9c9", lw=0.6, zorder=1)
                ax.yaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(
                    base=10.0, linthresh=lt))
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                    lambda v, _p2, _lt=lt: "" if abs(v) < _lt * 0.999 else
                    rf"${'-' if v < 0 else ''}10^{{{int(round(np.log10(abs(v))))}}}$"))
            else:
                ax.set_yscale("log")
                _fmt_log(ax)
                if scale == "symlog" and _p:
                    ax.set_ylim(min(_p) / 1.6, max(_p) * 1.6)
            if r == 0:
                ax.set_title(ylab, fontsize=6.5, pad=3.0)
            if j == len(COLS) - 1:
                ax.tick_params(axis="y", pad=7.5)
                # Which reference this ROW uses, on the free right-hand edge: a title would have to
                # repeat it in every panel and the panels are too narrow to carry it.
                ax.text(1.04, 0.5, ROW_TAG[r % 2], transform=ax.transAxes, rotation=270,
                        va="center", ha="left", fontsize=6, color="#555555")
    for c, (_f2, _yl, scale) in enumerate(COLS):
        if c == 0:
            continue                    # set per panel above; the rows do not share a scale
        _v = [y for r in range(nr) for ln in axes[r][c].get_lines()
              for y in np.asarray(ln.get_ydata(), dtype=float)
              if np.isfinite(y) and (scale == "symlog" or y > 0)]
        if not _v:
            continue
        if scale == "symlog":
            axes[0][c].set_ylim(min(min(_v) * 1.6, -2 * LINTHRESH), max(_v) * 1.6)
        else:
            axes[0][c].set_ylim(min(_v) / 1.2, max(_v) * 1.6)

    ink = "#444444"
    handles = [Line2D([], [], color=ink if colors else OURS, lw=1.3, label="Splats"),
               Line2D([], [], color=ink if colors else BASE, lw=1.0, ls="--", label="cc-pV$X$Z")]
    if fl:
        handles.append(Line2D([], [], color=FLOOR, lw=0.8, ls=(0, (4, 2)), label="PBE vs CCSD"))
    handles += [Line2D([], [], color=colors[s] if colors else ink, marker=MARK[s], ls="none",
                       ms=2.8, mfc="white", mew=0.9, label=lab[s]) for s in present]
    fig.legend(handles=handles, fontsize=6, frameon=False, ncol=legend_ncol or len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0, handlelength=1.1,
               columnspacing=0.7, handletextpad=0.3)

    out = out or os.path.join(HERE, "observables")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_at_height(fig, out + ".png", height_in=fig_h, dpi=240)
    save_at_height(fig, out + ".pdf", height_in=fig_h)
    w, h = fig.get_size_inches()
    print(f"wrote {out}.{{png,pdf}}  (usetex={usetex}, {w:.2f}x{h:.2f} in)")
    for s in present:
        _g, sp = load(s, csv, grads)
        n = {k: sum(1 for d in sp.values() if _f(d, k) == _f(d, k))
             for k in ("E", "tv_e", "tv_e_self", "dF_self")}
        print(f"    {s:18s} splat pts: {n['E']} E, {n['tv_e']} tv(gto), "
              f"{n['tv_e_self']} tv(self), {n['dF_self']} dF(self)")
    for s_ in present:
        _g, sp = load(s_, csv, grads)
        fc = {round(float(d["frac_cbs"]), 4) for d in list(sp.values()) + list(_g.values())
              if d.get("frac_cbs") is not None}
        if fc:
            print(f"    {s_:18s} splat-limit field frac_cbs={sorted(fc)} "
                  f"(rest of the field kept the largest rung)")
    if raw_rungs:
        print("  UNPOLISHED rungs plotted beside polished ones: " + ", ".join(raw_rungs))


if __name__ == "__main__":
    import sys
    if "--rebuttal" in sys.argv:
        main(groups=(RB_SYSTEMS[:3], RB_SYSTEMS[3:]), csv=RB_OBS, grads=RB_GRADIENTS, out=RB_OUT,
             estimator="powerlaw", floor_path=None, colors=size_colors(RB_SYSTEMS), block_h=SIZES_H)
    else:
        main()
