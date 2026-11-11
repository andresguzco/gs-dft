"""The rebuttal's figures and table, in the paper's style, from ``data/rebuttal/``.

    TUEPLOTS=1 USETEX=1 python -m experiments.rebuttal.figures

- Experiments 1 and 3: the accuracy scaling of Figure 5 (experiments.shared.plot_isoparams), the splat
  ladder against the cc-pVXZ ladder, both from their own complete-basis limit, with the fitted exponents.
- The paper's Figure 5 extended to the rebuttal's systems (``isoparams``): its three systems and the six new
  small ones on a 3 x 3 grid, from ``data/figures/fig5_isoparams.csv`` merged with ``data/rebuttal/accuracy.csv``;
  its Figure 6 is experiment 3's (``accuracy_exp3``).
- Experiment 2: the strong-scaling speedup and the per-GPU memory reduction (experiments.rebuttal.plot_scaling).
- Experiment 4: GPU4PySCF density fitting against direct SCF on cyclosporin at cc-pVTZ, as a table.

Everything goes to ``experiments/rebuttal/figures/``.
"""
import csv
import json
import os

from experiments.shared import plot_isoparams
from experiments.rebuttal import plot_scaling, plot_spaghetti, plot_summary

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "rebuttal")
OUT = os.path.join(HERE, "figures")

EXP1 = ("sf6", "cubane", "adenine_thymine", "porphine", "retinal", "cholesterol")
EXP3 = ("chignolin", "cyclosporin")
# The paper's Figure 5 with the rebuttal's small systems, by size, on a 3 x 3 grid; the two
# intermediate systems (EXP3) are the paper's Figure 6.
FIG5 = ("water", "sf6", "ethanol", "cubane", "alanine_dipeptide", "adenine_thymine", "porphine",
        "retinal", "cholesterol")
PAPER_FIG5 = os.path.join(ROOT, "data", "figures", "fig5_isoparams.csv")
PAPER_ATOMS = {"water": 3, "ethanol": 9, "alanine_dipeptide": 22}
NAMES = {"water": "Water", "ethanol": "Ethanol", "alanine_dipeptide": "Alanine dipeptide",
         "sf6": r"$\mathrm{SF_6}$", "cubane": "Cubane", "adenine_thymine": "A–T pair",
         "porphine": "Porphine", "retinal": "Retinal", "cholesterol": "Cholesterol", "chignolin": "Chignolin",
         "cyclosporin": "Cyclosporin A"}


def titles():
    """Each panel's title: the name and its atom count, as `Name (X atoms)` (experiments/rebuttal/data/systems.json)."""
    with open(os.path.join(HERE, "data", "systems.json")) as fh:
        sizes = {s: v["n_atoms"] for s, v in json.load(fh).items()} | PAPER_ATOMS
    return {s: f"{n} ({sizes[s]} atoms)" for s, n in NAMES.items()}


def merged_csv(paths, out):
    """Every row of the CSVs ``paths``, in order, under the union of their columns, written to ``out``."""
    rows, cols = [], []
    for p in paths:
        with open(p, newline="") as fh:
            r = csv.DictReader(fh)
            cols += [c for c in r.fieldnames if c not in cols]
            rows += list(r)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out


def exp4_table(path, out):
    """Markdown table of the cyclosporin cc-pVTZ pair; returns it."""
    rows = {r["mode"]: r for r in csv.DictReader(open(path))}
    head = "| GPU4PySCF, cyclosporin cc-pVTZ | E (Ha) | wall (s) | peak GPU (MB) | peak host (MB) |"
    lines = [head, "|---|---|---|---|---|"]
    for mode, label in (("df", "density fitting"), ("direct", "direct SCF")):
        r = rows.get(mode)
        if r:
            lines.append(f"| {label} | {float(r['E']):.6f} | {r['wall_s']} | {r['peak_gpu_mb']} | "
                         f"{r['peak_host_mb']} |")
    if {"df", "direct"} <= rows.keys():
        d = (float(rows["df"]["E"]) - float(rows["direct"]["E"])) * 1e3
        lines.append(f"\nE_DF - E_direct = {d:+.3f} mHa")
    text = "\n".join(lines) + "\n"
    with open(out, "w") as fh:
        fh.write(text)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    acc = os.path.join(DATA, "accuracy.csv")
    t = titles()
    plot_isoparams.main(acc, names=EXP1, titles=t, out=os.path.join(OUT, "accuracy_exp1"), ncols=3)
    plot_isoparams.main(acc, names=EXP3, titles=t, out=os.path.join(OUT, "accuracy_exp3"),
                        height_in=1.5, xlim=(1e5, 1e7))     # x labels at whole decades
    fig5 = merged_csv([PAPER_FIG5, acc], os.path.join(OUT, "fig5_isoparams_extended.csv"))
    plot_isoparams.main(fig5, names=FIG5, titles=t, out=os.path.join(OUT, "isoparams"), ncols=3)
    plot_spaghetti.main(acc, os.path.join(OUT, "spaghetti"), titles=NAMES)
    plot_summary.main(acc, OUT)
    sc = os.path.join(DATA, "exp2_scaling.csv")
    if os.path.exists(sc):
        plot_scaling.main(sc, os.path.join(OUT, "scaling_exp2"))
        plot_scaling.speedup_bars(sc, os.path.join(OUT, "speedup_exp2"))
    mem = os.path.join(DATA, "exp2_memory.csv")
    if os.path.exists(mem):
        plot_scaling.memory_bars(mem, os.path.join(OUT, "memory_exp2"))
    if os.path.exists(sc) and os.path.exists(mem):
        plot_scaling.parallel_efficiency(sc, mem, os.path.join(OUT, "parallel_efficiency"))
    e4 = os.path.join(DATA, "exp4.csv")
    if os.path.exists(e4):
        print(exp4_table(e4, os.path.join(OUT, "exp4_table.md")))


if __name__ == "__main__":
    main()
