"""Measure both ladders against the splat ladder's own extrapolated limit, for Figure 3's bottom row.

The limit is ``cbs.extrapolate_field`` over the matched splat rungs, the same three-point estimator
as the energy, applied at every grid point and force component. ``results.STALE_RUNGS`` are dropped
first. Writes one ``RESULT kind=obs_selfref`` per splat rung and one ``RESULT kind=gto_selfref`` per
Gaussian rung, each with ``L1_rho_self`` and ``dF_self``.

``--estimator powerlaw`` is for ladders whose rungs are not matched to the Gaussian counts (the
rebuttal's doubling ladders): the rebuttal's splat limit, ``E(M) = E_inf + A M^-p`` over all rungs
(``cbs.fit_powerlaw``, at least four), carried to the fields as ``f(M) = f_inf + a M^-p`` per grid
point and force component, with one exponent per field fitted by least squares over all rungs.

    uv run python -m experiments.exp2_observables.selfref [system ...]
    uv run python -m experiments.exp2_observables.selfref --estimator powerlaw sf6 cubane
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


def ref_grad(system, basis):
    """One Gaussian rung's nuclear gradient dE/dR, or None.

    ``refgrad_*`` wins over ``ref_*``'s own when present: dftax's gradient does not fit for the
    dipeptide at any cardinal (PySCF supplies it) nor for SF6 past cc-pVDZ (GPU4PySCF supplies all
    three cardinals, so one code covers the whole force ladder).
    """
    for name in ("refgrad", "ref"):
        p = os.path.join(RES, f"{name}_{system}_{basis}.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            if "grad" in d.files and np.isfinite(np.asarray(d["grad"])).all():
                return np.asarray(d["grad"])
    return None


def _gto_fields(system, basis):
    """``(rho, grad)`` for one Gaussian rung, or ``(None, None)``."""
    ref = os.path.join(RES, f"ref_{system}_{basis}.npz")
    if not os.path.exists(ref):
        return None, None
    d = np.load(ref, allow_pickle=True)
    rho = np.asarray(d["rho"]) if "rho" in d.files else None
    return rho, ref_grad(system, basis)


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


def powerlaw_field(Ms, fields, weights=None, ps=np.linspace(0.2, 8.0, 391)):
    """``f(M) = f_inf + a M^-p`` at every element of ``fields`` (one array per rung of ``Ms``), with
    one exponent ``p`` for the whole field: for each candidate ``p`` the elementwise fit is linear, so
    ``p`` is the grid value with the least weighted squared residual over all rungs and elements.
    Returns ``{"value", "p", "r2", "kind"}``; ``kind`` is ``"fit"``, or ``"none"`` under 4 rungs."""
    if len(Ms) < 4:
        return {"value": None, "p": float("nan"), "r2": float("nan"), "kind": "none"}
    shape = np.asarray(fields[0]).shape
    Y = np.stack([np.asarray(f, dtype=float).reshape(-1) for f in fields])      # (rungs, n)
    w = np.ones(Y.shape[1]) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    x = np.asarray(Ms, dtype=float) / float(min(Ms))
    best = None
    for p in ps:
        X = np.stack([np.ones_like(x), x ** -p], axis=1)                         # (rungs, 2)
        coef = np.linalg.solve(X.T @ X, X.T @ Y)                                 # (2, n)
        ssr = float(np.dot(w, ((Y - X @ coef) ** 2).sum(0)))
        if best is None or ssr < best[0]:
            best = (ssr, float(p), coef[0])
    sst = float(np.dot(w, ((Y - Y.mean(0)) ** 2).sum(0)))
    return {"value": best[2].reshape(shape), "p": best[1],
            "r2": 1.0 - best[0] / sst if sst > 0 else float("nan"), "kind": "fit"}


def run(system, estimator="matched"):
    found = rungs(system)
    found = {M: p for M, p in found.items() if (system, M) not in results.STALE_RUNGS}
    if len(found) < 3:
        print(f"# {system}: {len(found)} archived rung(s) — need >= 3, skipping", flush=True)
        return
    Ms = sorted(found)
    arch = {M: np.load(found[M]) for M in Ms}
    shp = np.asarray(arch[Ms[-1]]["rho"]).shape
    w = np.asarray(arch[Ms[-1]]["w"])
    fit = {}
    if estimator == "powerlaw":
        ok = [M for M in Ms if np.asarray(arch[M]["rho"]).shape == shp
              and np.isfinite(np.asarray(arch[M]["grad"])).all()]
        lim_E = cbs.fit_powerlaw(ok, [float(arch[M]["E"]) for M in ok])
        lim_rho = powerlaw_field(ok, [arch[M]["rho"] for M in ok], weights=w)
        lim_g = powerlaw_field(ok, [arch[M]["grad"] for M in ok])
        if lim_rho["kind"] == "none":
            print(f"# {system}: {len(ok)} rung(s) {ok}; the power law needs >= 4, skipping",
                  flush=True)
            return
        lim_rho["frac_cbs"] = lim_g["frac_cbs"] = 1.0
        fit = dict(estimator="powerlaw", p_rho=lim_rho["p"], r2_rho=lim_rho["r2"],
                   p_F=lim_g["p"], r2_F=lim_g["r2"], p_E=lim_E["p"], E_lim=lim_E["value"])
        print(f"# {system}: reference is the power-law splat limit over M={ok}: "
              f"p_E={lim_E['p']:.3f} (E_inf {lim_E['value']:.8f}, r2 {lim_E['r2']:.5f}), "
              f"p_rho={lim_rho['p']:.3f} (r2 {lim_rho['r2']:.5f}), "
              f"p_F={lim_g['p']:.3f} (r2 {lim_g['r2']:.5f})", flush=True)
    else:
        match = _matched(system, Ms)
        if len(match) < 3:
            print(f"# {system}: only {len(match)} count-matched rung(s), skipping", flush=True)
            return
        rho_by = {b: np.asarray(arch[M]["rho"]) for b, M in match.items()
                  if np.asarray(arch[M]["rho"]).shape == shp}
        g_by = {b: np.asarray(arch[M]["grad"]) for b, M in match.items()
                if np.isfinite(np.asarray(arch[M]["grad"])).all()}
        lim_rho = cbs.extrapolate_field(rho_by)
        lim_g = cbs.extrapolate_field(g_by)
        if lim_rho["kind"] == "none":
            print(f"# {system}: density limit unavailable, skipping", flush=True)
            return
        print(f"# {system}: reference is the extrapolated splat limit over "
              f"{sorted(match.items(), key=lambda kv: ORDER.index(kv[0]))}; "
              f"rho frac_cbs={lim_rho['frac_cbs']:.4f}, force frac_cbs={lim_g['frac_cbs']:.4f}",
              flush=True)
    rho_r, g_r = lim_rho["value"], lim_g["value"]

    def _emit(kind, rho, g, **tags):
        if rho is None or rho.shape != shp:
            return
        dF = (float(np.abs(g - g_r).max())
              if g is not None and g_r is not None and g.shape == g_r.shape
              and np.isfinite(g).all() else float("nan"))
        print(results.result(kind, system, vs="splatcbs",
                             L1_rho_self=float(np.dot(w, np.abs(rho - rho_r))),
                             dF_self=dF, frac_cbs=lim_rho["frac_cbs"], **tags, **fit), flush=True)

    for M in Ms:
        d = arch[M]
        _emit("obs_selfref", np.asarray(d["rho"]), np.asarray(d["grad"]),
              M=int(M), E=float(d["E"]))
    for basis in ORDER:
        rho, g = _gto_fields(system, basis)
        _emit("gto_selfref", rho, g, basis=basis)


if __name__ == "__main__":
    args = sys.argv[1:]
    est = "matched"
    if args[:1] == ["--estimator"]:
        est, args = args[1], args[2:]
    for s in (args or ["water", "ethanol", "alanine_dipeptide"]):
        run(s, est)
