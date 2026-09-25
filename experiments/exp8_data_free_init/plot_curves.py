"""Convergence curves and the initialization table of Appendix E from ``results/ladder_*.log``.

Prints, per scheme, the final energy over seeds and its spread, the steps to reach within
``THRESH_MHA`` of the common plateau, and the final cond(V); writes
``results/curves_<system>_<chart>.png``::

    uv run python -m experiments.exp8_data_free_init.plot_curves [THRESH_MHA=1.0]
"""
import os
import sys
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common.results import iter_results

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(HERE, "results")
THRESH = float(sys.argv[1]) * 1e-3 if len(sys.argv) > 1 else 1e-3   # Ha
SCHEME_ORDER = ["S0", "S1", "S2", "S3", "S4"]
SCHEME_LABEL = {"S0": "S0 random", "S1": "S1 +A1 exps", "S2": "S2 +B bonds",
                "S3": "S3 +C aniso", "S4": "S4 +D alloc"}


def parse():
    # curves[(system,chart,scheme)][seed] = {step: E};  finals[(system,chart,scheme)][seed]=(E,condV)
    curves = defaultdict(lambda: defaultdict(dict))
    finals = defaultdict(lambda: defaultdict(lambda: (None, None)))
    logs = glob.glob(os.path.join(RESDIR, "ladder_*.log"))
    for d in iter_results(logs, tag="RESULT_PARTIAL", kind="splat"):
        key = (d["system"], d["chart"], d["scheme"])
        curves[key][int(d["seed"])][int(d["step"])] = float(d["E"])
    for d in iter_results(logs, kind="splat"):           # final RESULT lines
        key = (d["system"], d["chart"], d["scheme"])
        finals[key][int(d["seed"])] = (float(d["E"]), float(d.get("cond_V", -1)))
    return curves, finals


def seed_mean_curve(per_seed):
    """{seed:{step:E}} → (steps, mean_E, std_E) over the steps present in ALL seeds."""
    common = set.intersection(*[set(c) for c in per_seed.values()]) if per_seed else set()
    steps = sorted(common)
    if not steps:
        return np.array([]), np.array([]), np.array([])
    M = np.array([[per_seed[s][t] for s in per_seed] for t in steps])   # (n_step, n_seed)
    return np.array(steps), M.mean(1), M.std(1)


def main():
    curves, finals = parse()
    if not curves:
        print(f"no ladder_*.log found in {RESDIR}"); return
    systems = sorted({k[0] for k in curves})
    charts = sorted({k[1] for k in curves})

    print(f"\nsteps to within {THRESH*1e3:.1f} mHa of the common plateau (seed-mean):\n")
    hdr = f"{'system':8s} {'chart':9s} " + " ".join(f"{s:>10s}" for s in SCHEME_ORDER)
    print(hdr); print("-" * len(hdr))

    for system in systems:
        for chart in charts:
            present = [s for s in SCHEME_ORDER if (system, chart, s) in curves]
            if not present:
                continue
            # common plateau = lowest seed-mean final energy across schemes for this system+chart
            plateau = min(np.mean([finals[(system, chart, s)][sd][0]
                                   for sd in finals[(system, chart, s)]
                                   if finals[(system, chart, s)][sd][0] is not None])
                          for s in present)
            target = plateau + THRESH

            fig, ax = plt.subplots(figsize=(6, 4))
            row = {}
            for s in present:
                steps, mean, std = seed_mean_curve(curves[(system, chart, s)])
                if steps.size == 0:
                    row[s] = "—"; continue
                ax.plot(steps, mean, label=SCHEME_LABEL[s], marker=".", ms=3)
                ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
                hit = np.where(mean <= target)[0]
                row[s] = str(int(steps[hit[0]])) if hit.size else f">{int(steps[-1])}"
            ax.axhline(target, ls="--", c="k", lw=0.8, label=f"plateau+{THRESH*1e3:.0f}mHa")
            ax.set_xlabel("step"); ax.set_ylabel("E (Ha)")
            ax.set_title(f"{system} / {chart}  (plateau={plateau:.5f} Ha)")
            ax.set_ylim(plateau - 2e-3, plateau + 30e-3)
            ax.legend(fontsize=7); fig.tight_layout()
            png = os.path.join(RESDIR, f"curves_{system}_{chart}.png")
            fig.savefig(png, dpi=130); plt.close(fig)
            print(f"{system:8s} {chart:9s} " + " ".join(f"{row.get(s,'—'):>10s}" for s in SCHEME_ORDER))

    # plateau + cond(V) check
    print("\nfinal energy (seed-mean, Ha) / cond(V):\n")
    for system in systems:
        for chart in charts:
            for s in SCHEME_ORDER:
                k = (system, chart, s)
                if k not in finals:
                    continue
                Es = [finals[k][sd][0] for sd in finals[k] if finals[k][sd][0] is not None]
                Cs = [finals[k][sd][1] for sd in finals[k] if finals[k][sd][1] not in (None, -1)]
                if Es:
                    cnote = f"cond(V)={np.mean(Cs):.1e}" if Cs else ""
                    print(f"  {system:8s} {chart:9s} {s}: E={np.mean(Es):.6f}  {cnote}")
    print(f"\nplots → {RESDIR}/curves_*.png")


if __name__ == "__main__":
    main()
