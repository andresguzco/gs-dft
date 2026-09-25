"""Figure 3: energy, density and force errors against resource, for three systems on shared axes.

The top row measures against the Gaussian ladder's limit, the bottom row against the splat ladder's
own limit, extrapolated with the same estimator from the splat rungs::

    TUEPLOTS=1 USETEX=1 uv run python -m experiments.shared.plot_observables

Reads ``data/figures/fig3_observables.csv`` and ``fig3_gto_gradients.csv`` and writes
``observables.{png,pdf}``. The bottom row comes from the ``kind=obs_selfref`` rows written by
``selfref.py``.
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
EXP2 = os.path.join(HERE, "..", "..", "data", "figures", "fig3_observables.csv")

OURS, BASE, GRIDC = "#5b2d91", "#8c8c8c", "#e9e9e9"
NOCC = {"water": 5, "ethanol": 13, "alanine_dipeptide": 39}
MARK = {"water": "o", "ethanol": "^", "alanine_dipeptide": "s"}
LABEL = {"water": r"\ce{H2O}", "ethanol": r"\ce{C2H5OH}",
         "alanine_dipeptide": "Ala dipeptide"}
PLAIN_LABEL = {"water": r"H$_2$O", "ethanol": r"C$_2$H$_5$OH",
               "alanine_dipeptide": "Ala dipeptide"}

XKEY, XLAB = "params", "Free parameters"

# One column per observable, drawn once per row. The FIELD changes with the row -- top row against
# the Gaussian reference, bottom row against the splat one -- while the quantity and the axis do
# not, so a column is comparable top to bottom as well as left to right.
COLS = [(("dE_gto", "dE_splat"), r"$\Delta$ (mHa/atom)", "symlog"),
        (("tv_e", "tv_e_self"), r"TV / electron", "log"),
        (("dF_max", "dF_self"), r"$\Delta F$ (meV/$\mathrm{\AA}$)", "log")]
ROW_TAG = ["CBS", "ISC"]
LINTHRESH = 0.1   # mHa/atom; symlog linear zone on the energy panels


FIG_H = PAPER_FIG_HEIGHT_IN * 1.45   # two rows
HA_BOHR_TO_MEV_A = 51422.08     # (27.211386 eV/Ha) / (0.529177 A/Bohr), x1000


_NATOMS = {}


def _natoms(sysn):
    if sysn not in _NATOMS:
        _NATOMS[sysn] = len(systems.molecule(system=sysn, basis="cc-pvdz").symbols)
    return _NATOMS[sysn]


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


def load(system):
    """(gto rows by basis, splat rows by M), each decorated with its resource coordinates."""
    gto, splat, ckpt_peak, obs_rows = {}, {}, {}, {}
    if not os.path.exists(EXP2):
        return gto, splat
    for row in iter_results(EXP2):
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
                                               int(d["nao"]), NOCC[system])["total"]
            d["peak_mb"] = resources.peak_mb_from_row(d)
            gto.setdefault(d["basis"], {}).update(d)
        elif d.get("kind") == "gtol1":
            gto.setdefault(d["basis"], {})["L1_rho"] = d["L1_rho"]
        elif d.get("kind") == "obs":
            if (system, int(d["M"])) in STALE_RUNGS:
                continue
            d["params"] = resources.splat_params(int(d["M"]), NOCC[system])["total"]
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
    for b, v in gto_dF(system).items():
        if b in gto:
            gto[b]["dF_max"] = v
    nelec = 2 * NOCC[system]
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


def gto_dF(system):
    """``{basis: max|F_basis - F_ref|}`` for the Gaussian ladder -- the top row's force baseline.

    The reference is the largest rung with a finite gradient. The gradients are the ``grad`` arrays
    of the ``ref_*.npz`` files written by `exp2_observables/gto_ref.py` and `cc_ref.py`."""
    rows = {}
    for r in read_records(GTO_GRADIENTS):
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


def main():
    usetex = apply_style(nrows=2, ncols=len(COLS), rel_width=1.0, height_in=FIG_H)
    lab = LABEL if usetex else PLAIN_LABEL
    fig, axes = plt.subplots(2, len(COLS), squeeze=False)
    # The style bundle installs a layout engine, which ignores subplots_adjust and warns.
    _eng = fig.get_layout_engine()
    if _eng is not None:
        _eng.set(h_pad=0.02, w_pad=0.02, hspace=0.04, wspace=0.06)
    for c in range(len(COLS)):
        if c > 0:
            axes[1][c].sharey(axes[0][c])
        axes[1][c].sharex(axes[0][c])
        axes[0][c].tick_params(labelbottom=False)

    present, notes, snotes, raw_rungs = [], {}, {}, []
    for system in ("water", "ethanol", "alanine_dipeptide"):
        gto, splat = load(system)
        if not splat:
            continue
        present.append(system)
        notes[system] = cbs.extrapolate({b: _f(g, "E") for b, g in gto.items()})
        snotes[system], matched = splat_limit(gto, splat)
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
        for rows, colour, ls, lw in ((list(gto.values()), BASE, "--", 1.0),
                                     (list(splat.values()), OURS, "-", 1.3)):
            for r in range(2):
                for j, (fields, _ylab, _sc) in enumerate(COLS):
                    xs, ys = _series(rows, fields[r])
                    if not xs:
                        continue
                    u = _unit(fields[r], system)
                    axes[r][j].plot(xs, [y * u for y in ys], ls=ls, color=colour, lw=lw,
                                    zorder=3, marker=MARK[system], ms=2.8, mfc="white", mew=0.9)

    for r in range(2):
        for j, (_fields, ylab, scale) in enumerate(COLS):
            ax = axes[r][j]
            _decorate(ax, XLAB if r == 1 else "", usetex, pad=1.0 if r == 0 else None)
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
                ax.text(1.04, 0.5, ROW_TAG[r], transform=ax.transAxes, rotation=270,
                        va="center", ha="left", fontsize=6, color="#555555")
    for c, (_f2, _yl, scale) in enumerate(COLS):
        if c == 0:
            continue                    # set per panel above; the rows do not share a scale
        _v = [y for r in range(2) for ln in axes[r][c].get_lines()
              for y in np.asarray(ln.get_ydata(), dtype=float)
              if np.isfinite(y) and (scale == "symlog" or y > 0)]
        if not _v:
            continue
        if scale == "symlog":
            axes[0][c].set_ylim(min(min(_v) * 1.6, -2 * LINTHRESH), max(_v) * 1.6)
        else:
            axes[0][c].set_ylim(min(_v) / 1.2, max(_v) * 1.6)

    handles = [Line2D([], [], color=OURS, lw=1.3, label="Splats"),
               Line2D([], [], color=BASE, lw=1.0, ls="--", label="cc-pV$X$Z")]
    handles += [Line2D([], [], color="#444444", marker=MARK[s], ls="none", ms=2.8,
                       mfc="white", mew=0.9, label=lab[s]) for s in present]
    fig.legend(handles=handles, fontsize=6, frameon=False, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0, handlelength=1.1,
               columnspacing=0.7, handletextpad=0.3)

    out = os.path.join(HERE, "observables")
    save_at_height(fig, out + ".png", height_in=FIG_H, dpi=240)
    save_at_height(fig, out + ".pdf", height_in=FIG_H)
    w, h = fig.get_size_inches()
    print(f"wrote {out}.{{png,pdf}}  (usetex={usetex}, {w:.2f}x{h:.2f} in)")
    for s in present:
        _g, sp = load(s)
        n = {k: sum(1 for d in sp.values() if _f(d, k) == _f(d, k))
             for k in ("E", "tv_e", "tv_e_self", "dF_self")}
        print(f"    {s:18s} splat pts: {n['E']} E, {n['tv_e']} tv(gto), "
              f"{n['tv_e_self']} tv(self), {n['dF_self']} dF(self)")
    for s_ in present:
        _g, sp = load(s_)
        fc = {round(float(d["frac_cbs"]), 4) for d in list(sp.values()) + list(_g.values())
              if d.get("frac_cbs") is not None}
        if fc:
            print(f"    {s_:18s} splat-limit field frac_cbs={sorted(fc)} "
                  f"(rest of the field kept the largest rung)")
    if raw_rungs:
        print("  UNPOLISHED rungs plotted beside polished ones: " + ", ".join(raw_rungs))


if __name__ == "__main__":
    main()
