"""Measure both ladders against the splat ladder's own extrapolated limit, for Figure 3's bottom row.

The limit is ``cbs.extrapolate_field`` over the matched splat rungs, the same three-point estimator
as the energy, applied at every grid point and force component. ``results.STALE_RUNGS`` are dropped
first. Writes one ``RESULT kind=obs_selfref`` per splat rung and one ``RESULT kind=gto_selfref`` per
Gaussian rung, each with ``L1_rho_self`` and ``dF_self``.

    uv run python -m experiments.exp2_observables.selfref [system ...]
"""
import glob
import os
import sys

import numpy as np

from experiments.common import cbs, results

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ("cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z")


def rungs(system):
    """``{M: path}`` of this system's archived splat rungs."""
    out = {}
    for p in glob.glob(os.path.join(RES, f"splatref_{system}_M*.npz")):
        try:
            out[int(os.path.basename(p).rsplit("_M", 1)[1][:-4])] = p
        except ValueError:
            continue
    return out


def _gto_fields(system, basis):
    """``(rho, grad)`` for one Gaussian rung, or ``(None, None)``.

    Same `refgrad_*` fallback as `plot_observables.gto_dF`: `ref_*` carries no usable gradient for
    the dipeptide at any cardinal, and PySCF supplies the substitute.
    """
    ref = os.path.join(RES, f"ref_{system}_{basis}.npz")
    if not os.path.exists(ref):
        return None, None
    d = np.load(ref, allow_pickle=True)
    rho = np.asarray(d["rho"]) if "rho" in d.files else None
    g = np.asarray(d["grad"]) if "grad" in d.files else None
    if g is None or not np.isfinite(g).all():
        ext = os.path.join(RES, f"refgrad_{system}_{basis}.npz")
        g = None
        if os.path.exists(ext):
            gg = np.asarray(np.load(ext, allow_pickle=True)["grad"])
            g = gg if np.isfinite(gg).all() else None
    return rho, g


def _nao(system, basis):
    """Function count for one Gaussian rung, from its archive."""
    p = os.path.join(RES, f"ref_{system}_{basis}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return int(d["nao"]) if "nao" in d.files else None


def _matched(system, Ms):
    """``{basis: M}`` under the matched-count rule, M = nao(cardinal), within 15%."""
    out = {}
    for basis in ORDER:
        n = _nao(system, basis)
        if n is None:
            continue
        M = min(Ms, key=lambda m: abs(m - n))
        if abs(M - n) <= 0.15 * max(M, 1):
            out[basis] = M
    return out


def run(system):
    found = rungs(system)
    found = {M: p for M, p in found.items() if (system, M) not in results.STALE_RUNGS}
    if len(found) < 3:
        print(f"# {system}: {len(found)} archived rung(s) — need >= 3, skipping", flush=True)
        return
    Ms = sorted(found)
    arch = {M: np.load(found[M]) for M in Ms}
    match = _matched(system, Ms)
    if len(match) < 3:
        print(f"# {system}: only {len(match)} count-matched rung(s), skipping", flush=True)
        return
    shp = np.asarray(arch[Ms[-1]]["rho"]).shape
    rho_by = {b: np.asarray(arch[M]["rho"]) for b, M in match.items()
              if np.asarray(arch[M]["rho"]).shape == shp}
    g_by = {b: np.asarray(arch[M]["grad"]) for b, M in match.items()
            if np.isfinite(np.asarray(arch[M]["grad"])).all()}
    lim_rho = cbs.extrapolate_field(rho_by)
    lim_g = cbs.extrapolate_field(g_by)
    if lim_rho["kind"] == "none":
        print(f"# {system}: density limit unavailable, skipping", flush=True)
        return
    w = np.asarray(arch[Ms[-1]]["w"])
    rho_r, g_r = lim_rho["value"], lim_g["value"]
    print(f"# {system}: reference is the extrapolated splat limit over "
          f"{sorted(match.items(), key=lambda kv: ORDER.index(kv[0]))}; "
          f"rho frac_cbs={lim_rho['frac_cbs']:.4f}, force frac_cbs={lim_g['frac_cbs']:.4f}",
          flush=True)

    def _emit(kind, rho, g, **tags):
        if rho is None or rho.shape != shp:
            return
        dF = (float(np.abs(g - g_r).max())
              if g is not None and g_r is not None and g.shape == g_r.shape
              and np.isfinite(g).all() else float("nan"))
        print(results.result(kind, system, vs="splatcbs",
                             L1_rho_self=float(np.dot(w, np.abs(rho - rho_r))),
                             dF_self=dF, frac_cbs=lim_rho["frac_cbs"], **tags), flush=True)

    for M in Ms:
        d = arch[M]
        _emit("obs_selfref", np.asarray(d["rho"]), np.asarray(d["grad"]),
              M=int(M), E=float(d["E"]))
    for basis in ORDER:
        rho, g = _gto_fields(system, basis)
        _emit("gto_selfref", rho, g, basis=basis)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["water", "ethanol", "alanine_dipeptide"]):
        run(s)
