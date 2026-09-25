"""Complete-basis-set extrapolation for any scalar observable, reporting how the limit was obtained.

The three-point form E(X) = E_CBS + A r^X assumes a monotone, exponentially converging ladder. A
property such as the dipole need not converge that way (water's cc-pVXZ dipole rises and then
falls), so ``extrapolate`` returns what it did alongside the number:

    kind="cbs"      a decaying triple; the extrapolated limit
    kind="largest"  the triple does not decay; the largest-cardinal value, unextrapolated
    kind="none"     fewer than three cardinals; NaN, and the caller skips the system
"""
from __future__ import annotations

import math

CARD = {"cc-pvdz": 2, "cc-pvtz": 3, "cc-pvqz": 4, "cc-pv5z": 5, "cc-pv6z": 6}


def extrapolate(by_basis: dict) -> dict:
    """Extrapolate ``{basis_name: value}`` to the complete-basis limit.

    Returns ``{"value", "kind", "spread", "cardinals", "r"}``. ``kind`` is one of
    ``"cbs" | "largest" | "none"`` and MUST be checked; see the module docstring.
    """
    items = sorted((CARD[b], float(v)) for b, v in by_basis.items()
                   if b in CARD and v is not None and v == v)
    cards = [x for x, _ in items]
    if len(items) < 3:
        return {"value": math.nan, "kind": "none", "spread": math.nan,
                "cardinals": cards, "r": None}
    (_, v0), (_, v1), (_, v2) = items[-3:]          # consecutive cardinals, unit spacing
    d01, d12 = v1 - v0, v2 - v1
    # A valid extrapolation needs successive differences that shrink and keep their sign: that is
    # what "converging geometrically toward a limit" means. Anything else (a sign flip, a growing
    # step) is not this model, and the honest answer is the largest basis we actually computed.
    if abs(d01) < 1e-14 or not (0.0 < d12 / d01 < 1.0):
        return {"value": v2, "kind": "largest", "spread": 0.0,
                "cardinals": cards[-3:], "r": None}
    r = d12 / d01
    limit = v0 - d01 / (r - 1.0)
    return {"value": limit, "kind": "cbs", "spread": abs(limit - v2),
            "cardinals": cards[-3:], "r": r}


def error_vs_limit(value: float, limit: dict) -> float:
    """|value - limit|, or NaN when the limit is not usable. Never silently substitutes."""
    if limit["kind"] == "none" or limit["value"] != limit["value"]:
        return math.nan
    return abs(float(value) - limit["value"])


def summarize(limits: dict) -> str:
    """One line naming which observables got a real extrapolation and which fell back."""
    real = [k for k, v in limits.items() if v["kind"] == "cbs"]
    fell = [k for k, v in limits.items() if v["kind"] == "largest"]
    none = [k for k, v in limits.items() if v["kind"] == "none"]
    parts = []
    if real:
        parts.append("extrapolated: " + ", ".join(sorted(real)))
    if fell:
        parts.append("NOT extrapolated (non-decaying triple, largest cardinal used): "
                     + ", ".join(sorted(fell)))
    if none:
        parts.append("UNAVAILABLE (<3 cardinals): " + ", ".join(sorted(none)))
    return " | ".join(parts) if parts else "no limits computed"


def extrapolate_field(by_basis: dict) -> dict:
    """`extrapolate`, applied independently at every element of arrays that share a shape.

    Same closed form, same validity test, same fallback to the largest cardinal -- just evaluated
    elementwise, so a density on a grid or a force on the atoms extrapolates exactly as a scalar
    observable does. Returns ``{"value", "kind", "cardinals", "frac_cbs"}``."""
    import numpy as np
    items = sorted(((CARD[b], np.asarray(v, dtype=float))
                    for b, v in by_basis.items() if b in CARD and v is not None),
                   key=lambda t: t[0])
    cards = [c for c, _ in items]
    if len(items) < 3:
        return {"value": None, "kind": "none", "cardinals": cards, "frac_cbs": 0.0}
    (_, v0), (_, v1), (_, v2) = items[-3:]          # consecutive cardinals, unit spacing
    d01, d12 = v1 - v0, v2 - v1
    with np.errstate(divide="ignore", invalid="ignore"):
        r = d12 / d01
        ok = np.isfinite(r) & (np.abs(d01) >= 1e-14) & (r > 0.0) & (r < 1.0)
        limit = np.where(ok, v0 - d01 / (r - 1.0), v2)
    return {"value": limit, "kind": "cbs" if bool(ok.any()) else "largest",
            "cardinals": cards[-3:], "frac_cbs": float(ok.mean())}


def extrapolate_seq(values) -> dict:
    """`extrapolate` for a ladder indexed by POSITION rather than by cardinal name.

    The closed form only ever uses three values at UNIFORM SPACING in the index along which the
    series converges geometrically. For a Gaussian ladder that index is the cardinal number; for a
    doubling splat ladder (M, 2M, 4M, ...) it is log10(M), equally uniform. Same formula, same
    validity test, same fallback -- only the index differs."""
    vals = [float(v) for v in values if v == v]
    if len(vals) < 3:
        return {"value": math.nan, "kind": "none", "spread": math.nan, "n": len(vals), "r": None}
    v0, v1, v2 = vals[-3:]
    d01, d12 = v1 - v0, v2 - v1
    if abs(d01) < 1e-14 or not (0.0 < d12 / d01 < 1.0):
        return {"value": v2, "kind": "largest", "spread": 0.0, "n": len(vals), "r": None}
    r = d12 / d01
    return {"value": v0 - d01 / (r - 1.0), "kind": "cbs", "spread": abs(v0 - d01 / (r - 1.0) - v2),
            "n": len(vals), "r": r}


def fit_powerlaw(xs, values, drop_first=False):
    """Fit ``E(x) = E_inf + A x**(-p)`` over ALL the rungs and return the limit.

    `E_inf` is searched, and for each candidate the model is LINEAR on log-log, so the inner fit is
    a `polyfit`. The criterion is 1 - R^2, which is scale free. A plain log-residual is DEGENERATE:
    driving E_inf far below the data makes every error nearly equal, and a flat line fits equal
    values perfectly, so the search runs away to -infinity (it did, to the edge of the grid, on two
    of three systems). `E_inf` is constrained strictly below every rung, so no data point can sit
    on the limit and every error stays positive on a log axis.

    Returns ``{"value", "p", "r2", "kind"}``; ``kind`` is ``"fit"`` or ``"none"``."""
    import numpy as np
    xs = np.asarray(list(xs), dtype=float)
    es = np.asarray(list(values), dtype=float)
    ok = np.isfinite(xs) & np.isfinite(es)
    xs, es = xs[ok], es[ok]
    if drop_first and len(xs) > 3:
        o = np.argsort(xs)
        xs, es = xs[o][1:], es[o][1:]
    if len(xs) < 4:
        return {"value": math.nan, "p": math.nan, "r2": math.nan, "kind": "none"}
    lx, emin = np.log10(xs), es.min()

    def obj(delta):
        y = es - (emin - delta)
        if (y <= 0).any():
            return math.inf, None
        ly = np.log10(y)
        c = np.polyfit(lx, ly, 1)
        sst = float(((ly - ly.mean()) ** 2).sum())
        if sst <= 0.0 or c[0] >= 0.0:
            return math.inf, None       # a limit the series does not approach from above
        return float(((ly - np.polyval(c, lx)) ** 2).sum() / sst), c

    lo, hi = 1e-7, 1.0
    g = np.geomspace(lo, hi, 20000)
    j = int(np.argmin([obj(d)[0] for d in g]))
    lo, hi = g[max(j - 1, 0)], g[min(j + 1, len(g) - 1)]
    for _ in range(5):
        g = np.geomspace(lo, hi, 2000)
        j = int(np.argmin([obj(d)[0] for d in g]))
        lo, hi = g[max(j - 1, 0)], g[min(j + 1, len(g) - 1)]
    o, c = obj(g[j])
    if c is None:
        return {"value": math.nan, "p": math.nan, "r2": math.nan, "kind": "none"}
    return {"value": float(emin - g[j]), "p": float(-c[0]), "r2": float(1.0 - o), "kind": "fit"}
