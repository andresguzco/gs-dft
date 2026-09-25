"""Appendix F figure: the total-energy dissociation ladders E(R) of LiH and LiF.

Plain cc-pVDZ/TZ/QZ and aug-cc-pVXZ against splat clouds sized to the plain function counts,
M = nao(cc-pVDZ/TZ/QZ). Restricted PBE dissociates LiX on the ionic Li+X- diabat. The Gaussian points
are single-point RKS from an atomic guess; the splat points are warm-started along R.

    MOL=both TUEPLOTS=1 USETEX=1 uv run python -m experiments.exp3_lih_dissociation.plot_energy_curve

Reads ``data/figures/fig4_dissociation.csv``, or the logs of a directory passed as the first argument.
"""
import glob
import os
import sys

from experiments.common.results import iter_results

PARTNER = {"lih": "H", "lif": "F"}
PRETTY = {"lih": "LiH", "lif": "LiF"}
PRETTY_B = {"cc-pvdz": "cc-pVDZ", "cc-pvtz": "cc-pVTZ", "cc-pvqz": "cc-pVQZ",
            "aug-cc-pvdz": "aug-cc-pVDZ", "aug-cc-pvtz": "aug-cc-pVTZ", "aug-cc-pvqz": "aug-cc-pVQZ"}
TUEPLOTS = bool(os.environ.get("TUEPLOTS"))
USETEX = bool(os.environ.get("USETEX"))


FIGURE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "figures",
                           "fig4_dissociation.csv")


def parse(src, mol):
    gto, gnao, splat = {}, {}, {}
    logs = sorted(glob.glob(src + "/*.log")) if os.path.isdir(src) else [src]
    for d in iter_results(logs, kind="gto"):
        if d.get("system", d.get("mol")) != mol or d.get("converged") == "False":
            continue
        gto.setdefault(d["basis"], {})[float(d["R"])] = float(d["E"])
        gnao[d["basis"]] = int(d["nao"])
    for d in [*iter_results(logs, kind="splat"), *iter_results(logs, kind="splat_warm")]:
        if d.get("system", d.get("mol")) != mol or "R" not in d:
            continue
        m, r, e = int(d["M"]), float(d["R"]), float(d["E"])
        c = splat.setdefault(m, {})
        c[r] = min(c.get(r, e), e)                       # best-of-runs per (M, R), cold + phase=warm
    return gto, gnao, splat


def _draw(ax, plt, mol, gto, gnao, splat, label_M=True):
    # label_M=True (single molecule) tags every entry with its function count (M=...); for the
    # two-panel figure the counts differ per molecule, so we label by rung and give M in the caption.
    shade = {"dz": 0.55, "tz": 0.74, "qz": 0.93}          # rung → position within each colormap
    ms = 3 if TUEPLOTS else 4
    for fam, cmap, ls, mk in [("", plt.cm.Blues, "--", "s"), ("aug-", plt.cm.Greens, "-.", "v")]:
        for rg in ("dz", "tz", "qz"):
            b = f"{fam}cc-pv{rg}"
            if b not in gto:
                continue
            rs = sorted(gto[b])
            lab = f"{PRETTY_B[b]} (M={gnao[b]})" if label_M else PRETTY_B[b]
            ax.plot(rs, [gto[b][r] for r in rs], ls=ls, marker=mk, ms=ms, lw=1.1,
                    color=cmap(shade[rg]), label=lab)
    for rg in ("dz", "tz", "qz"):                         # 3 splats at the plain cc-pVXZ counts
        M = gnao.get(f"cc-pv{rg}")
        if M is None or M not in splat:
            continue
        rs = sorted(splat[M])
        lab = f"splat (M={M})" if label_M else f"splat·{rg.upper()}"
        ax.plot(rs, [splat[M][r] for r in rs], ls="-", marker="D", ms=ms + 1, lw=1.5,
                color=plt.cm.Reds(shade[rg]), label=lab)
    ax.set_xlabel(f"R(Li-{PARTNER[mol]}) (\\AA)" if USETEX else f"R(Li-{PARTNER[mol]}) (Å)")
    ax.set_ylabel("E (Ha)")
    if not TUEPLOTS:
        ax.set_title(f"{PRETTY[mol]} dissociation E(R), restricted PBE (ionic Li⁺{PARTNER[mol]}⁻)")
    ax.grid(alpha=0.3)


def main(logdir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    logdir = logdir or FIGURE_DATA
    mol_env = os.environ.get("MOL", "lih")
    mols = ["lih", "lif"] if mol_env == "both" else [mol_env]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if TUEPLOTS:
        from tueplots import bundles, figsizes
        _bundle = getattr(bundles, "iclr2024", getattr(bundles, "iclr2023", None))
        _fsizes = getattr(figsizes, "iclr2024", getattr(figsizes, "iclr2023", None))
        plt.rcParams.update(_bundle(usetex=USETEX, family="serif"))
        from experiments.common.style import apply as _apply_style
        _apply_style()          # serif fallback only; figsize is set below
        rel = float(os.environ.get("REL_WIDTH", "1.0" if len(mols) == 2 else "0.6"))
        ratio = 0.5 if len(mols) == 2 else 1.05
        plt.rcParams.update(_fsizes(rel_width=rel, height_to_width_ratio=ratio))
        fig, axes = plt.subplots(1, len(mols), squeeze=False)
    else:
        fig, axes = plt.subplots(1, len(mols), figsize=(7.6 * len(mols), 5.6), squeeze=False)
    axes = axes[0]

    for ax, mol in zip(axes, mols):
        gto, gnao, splat = parse(logdir, mol)
        _draw(ax, plt, mol, gto, gnao, splat, label_M=(len(mols) == 1))
    # shared legend BELOW the panels (9 entries swamp a half-width axes if placed inside)
    handles, labels = axes[0].get_legend_handles_labels()
    if TUEPLOTS:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
                   ncol=3 if len(mols) == 1 else 5, fontsize=5, frameon=False,
                   handlelength=1.3, columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)
        fig.tight_layout(rect=(0, 0.0, 1, 0.98))
    else:
        axes[-1].legend(fontsize=7.5, ncol=3, loc="upper left")
        fig.tight_layout()

    stem = "dissociation_ladder_both" if len(mols) == 2 else f"dissociation_{mols[0]}_energy"
    out = os.path.join(here, stem + ".png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    if TUEPLOTS:
        fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out}" + (" (+ .pdf)" if TUEPLOTS else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
