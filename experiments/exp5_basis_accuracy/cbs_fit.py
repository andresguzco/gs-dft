"""CBS extrapolation and accuracy and cost tables for the Gaussian baselines.

Fits E(X) = E_CBS + A r^X per system over the three largest cardinals (DZ=2 to 5Z=5), using the
closed form for equally spaced cardinals: r = (E_{X+2} - E_{X+1}) / (E_{X+1} - E_X) and
E_CBS = E_X - (E_{X+1} - E_X) / (r - 1). Reports the energy error from the CBS limit and the wall
time and peak memory per basis.

    uv run python -m experiments.exp5_basis_accuracy.cbs_fit [results_dir]
"""
import sys
import glob
import json
import math

from experiments.common.results import iter_results

CARD = {"cc-pvdz": 2, "cc-pvtz": 3, "cc-pvqz": 4, "cc-pv5z": 5, "cc-pv6z": 6}
ORDER = ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z", "cc-pv6z"]


def parse(logdir):
    """GTO baseline rows."""
    rows = {}
    for d in iter_results(glob.glob(logdir + "/*.log")):
        if d.get("kind", "gto") != "gto":
            continue
        if {"system", "basis", "nao", "E", "wall_s", "peak_mb", "converged"} <= d.keys():
            rows.setdefault(d["system"], {})[d["basis"]] = {
                "nao": int(d["nao"]), "E": float(d["E"]),
                "wall_s": float(d["wall_s"]), "peak_mb": float(d["peak_mb"]),
                "converged": d["converged"] == "True"}
    return rows


def fit_cbs(bases):
    """E_CBS from the largest 3 available cardinals (closed-form 3-point exponential)."""
    items = sorted(((CARD[b], v["E"]) for b, v in bases.items() if b in CARD))
    if len(items) < 3:
        return math.nan, None, [x for x, _ in items]
    (x0, e0), (x1, e1), (x2, e2) = items[-3:]          # consecutive cardinals, spacing 1
    d01, d12 = e1 - e0, e2 - e1
    if abs(d01) < 1e-12 or d12 / d01 <= 0 or d12 / d01 >= 1:
        return e2, None, [x0, x1, x2]                  # non-convergent triple → use largest as proxy
    r = d12 / d01                                      # = e^{−α}
    e_cbs = e0 - d01 / (r - 1.0)
    return e_cbs, {"r": r, "alpha": -math.log(r)}, [x0, x1, x2]


def main(logdir="experiments/exp5_basis_accuracy/results"):
    rows = parse(logdir)
    if not rows:
        print(f"no RESULT lines in {logdir}/*.log yet"); return
    out = {}
    for sys_, bases in sorted(rows.items()):
        e_cbs, fit, used = fit_cbs(bases)
        fit_str = f", α={fit['alpha']:.2f}" if fit else ", proxy"
        print(f"\n=== {sys_} ===  E_CBS = {e_cbs:.6f} Ha  (exp fit on cardinals {used}{fit_str})")
        print(f"  {'basis':9s} {'nao':>5s} {'E (Ha)':>15s} {'ΔE-CBS(mHa)':>12s} "
              f"{'wall_s':>9s} {'peak_GB':>8s} {'conv':>5s}")
        for b in ORDER:
            if b in bases:
                v = bases[b]
                dE = (v["E"] - e_cbs) * 1e3
                print(f"  {b:9s} {v['nao']:5d} {v['E']:15.6f} {dE:12.3f} "
                      f"{v['wall_s']:9.1f} {v['peak_mb']/1024:8.2f} {str(v['converged']):>5s}")
        out[sys_] = {"E_CBS": e_cbs, "fit": fit, "used_cardinals": used, "bases": bases}
    with open(logdir + "/cbs_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {logdir}/cbs_summary.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
