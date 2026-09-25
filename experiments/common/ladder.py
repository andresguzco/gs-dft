"""Loader for the cardinal-ladder figures: Gaussian baselines, splat sweeps and the CBS limit.

Every plotter reads its baselines, sweeps and limits from here, so a figure and the table next to
it use the same rows.
"""
from __future__ import annotations

import glob
import os
import re as _re

from experiments.common import cbs
from experiments.common.results import iter_results
from experiments.common import results as _res
from experiments.exp5_basis_accuracy.cbs_fit import CARD, ORDER, fit_cbs

ALIAS = {"h2o": "water"}          # GTO baselines ran under `h2o`, the splat sweep under `water`


#: Re-exported so ladder consumers need not reach into `results`.
from experiments.common.results import XC


def load(logdir, strict=True, budget=None):
    """``{system: {"gto": {basis: (nao, E, wall_s)}, "splat": {M: [E, …per seed]}, "cbs": float|nan,
    "cards": [cardinal, …]}}`` from a data CSV or a directory of run logs.

    ``budget`` (steps) keeps only splat rows trained that long. Off by default; an iso-budget
    ladder must pass it, or a rung from a longer run enters as if it were a cheaper one."""
    gto, splat, prerebuild = {}, {}, {}
    off_budget = []
    walls = {}          # (system, basis, E) -> wall_s, so an omission cannot erase a timing
    logs = [logdir] if str(logdir).endswith(".csv") else _res.ordered_logs(logdir, "*.log")
    for d in iter_results(logs):
        sysn = ALIAS.get(d.get("system"), d.get("system"))
        # One functional per ladder; a row with no `xc` key is PBE.
        if d.get("xc", XC) != XC:
            continue
        if d.get("kind", "gto") == "gto" and {"system", "basis", "nao", "E"} <= d.keys():
            # Newest wins, except for a timing the newer row omits: timings are remembered per
            # (system, basis, energy), since a different energy is a different calculation.
            _E = float(d["E"])
            _key = (sysn, d["basis"], round(_E, 9))
            _wall = float(d.get("wall_s", "nan"))
            if _wall == _wall:
                walls[_key] = _wall
            else:
                _wall = walls.get(_key, float("nan"))
            gto.setdefault(sysn, {})[d["basis"]] = (int(d["nao"]), _E, _wall)
            # A row without `kind=` comes from an older engine version. Track the winning row only:
            # a stale row that a newer one replaced does not count.
            stale = prerebuild.setdefault(sysn, set())
            (stale.add if "kind" not in d else stale.discard)(d["basis"])
        elif d.get("kind") == "splat" and {"system", "M", "E"} <= d.keys():
            if budget is not None and int(d.get("steps", -1)) != int(budget):
                off_budget.append((sysn, int(d["M"]), d.get("steps", "none")))
                continue
            audited = next((v for k, v in d.items() if k.startswith("E_grid")), None)
            splat.setdefault(sysn, {}).setdefault(int(d["M"]), []).append(float(audited or d["E"]))

    if off_budget:
        _seen = sorted({(s, M, st) for s, M, st in off_budget})
        print(f"  ladder.load: dropped {len(off_budget)} splat row(s) off the {budget}-step "
              f"budget: " + ", ".join(f"{s} M={M} ({st} steps)" for s, M, st in _seen))

    out = {}
    for s in set(gto) | set(splat):
        g = gto.get(s, {})
        e_cbs, _, cards = fit_cbs({b: {"E": v[1]} for b, v in g.items()})
        out[s] = {"gto": g, "splat": splat.get(s, {}), "cbs": e_cbs, "cards": cards,
                  "prerebuild": prerebuild.get(s, set())}
    if strict:
        bad = variational_violations(out)
        if bad:
            raise ValueError("contaminated GTO ladder in %s:\n  %s\n(pass strict=False to "
                             "inspect it anyway)" % (logdir, "\n  ".join(bad)))
    return out


_CARDINAL = {"d": 2, "t": 3, "q": 4, "5": 5, "6": 6, "7": 7}
_BASIS_RE = _re.compile(r"^(?P<fam>.*?)cc-pv(?P<c>[dtq567])z$")

#: A rung that sits this far ABOVE a smaller rung of the same family is not noise.
VARIATIONAL_TOL_MHA = 1.0


def _family_cardinal(basis):
    """``(family, cardinal)`` for a Dunning basis, ``(None, None)`` if it is not one.

    The family is whatever precedes ``cc-pv``, so ``aug-cc-pvdz`` and ``cc-pvdz`` are DIFFERENT
    families — an anion's aug-DZ legitimately sits below plain TZ, and comparing across the two
    families by ``nao`` invents a violation where there is none (exp4's f⁻ and OH⁻ both do this).
    """
    m = _BASIS_RE.match(str(basis).lower())
    return (m.group("fam"), _CARDINAL[m.group("c")]) if m else (None, None)


def variational_violations(data, tol_mha=VARIATIONAL_TOL_MHA):
    """``[str, …]`` — one line per GTO ladder that goes UP with cardinal within a basis family.

    An inverted ladder is visible from the numbers alone, without knowing which axis was
    contaminated, which is why this check is worth running even when every row looks well-formed."""
    msgs = []
    for s in sorted(data):
        fams = {}
        for b, v in data[s]["gto"].items():
            fam, card = _family_cardinal(b)
            if fam is not None:
                fams.setdefault(fam, []).append((card, b, float(v[1])))
        for fam, rows in sorted(fams.items()):
            rows.sort()
            for (_, b0, e0), (_, b1, e1) in zip(rows, rows[1:]):
                if (e1 - e0) * 1e3 > tol_mha:
                    msgs.append(f"{s}: {b1} is {(e1 - e0) * 1e3:.3f} mHa ABOVE {b0} — a larger "
                                f"basis cannot raise the energy, so these rows are not the same "
                                f"calculation (geometry? functional? engine era?)")
    return msgs


def era_warnings(data):
    """``[str, …]`` — one line per system whose CBS mixes engine eras.

    Mixing engine eras in one ladder is small (~0.03–0.11 mHa) but invisible, so it is forbidden."""
    msgs = []
    for s in sorted(data):
        stale = data[s].get("prerebuild") or set()
        used = {b for b in ORDER if b in data[s]["gto"] and CARD.get(b) in (data[s]["cards"] or [])}
        bad = sorted(stale & used)
        if bad and data[s]["cbs"] == data[s]["cbs"]:
            msgs.append(f"{s}: CBS uses PRE-REBUILD (engine 0.4) rung(s) {bad} — mixed-era fit")
    return msgs


def cbs_alternative(entry):
    """The CBS limit refit on the LOWEST three cardinals, or NaN when only one triple exists.

    The closed-form 3-point fit is exact through its points — zero residual, zero degrees of
    freedom — so it carries no internal error bar. With four cardinals it can be run on the top
    three and on the bottom three, and the two limits bracket how sensitive the extrapolation is to
    which rungs feed it. That is the only error bar this estimator admits, and the alanine dipeptide
    (three cardinals) does not admit even that.
    """
    if entry["cbs"] != entry["cbs"]:
        return float("nan")
    cards = sorted((CARD[b], b, v[1]) for b, v in entry["gto"].items() if b in CARD)
    if len(cards) < 4:
        return float("nan")
    lo, _, _ = fit_cbs({b: {"E": e} for _, b, e in cards[:3]})
    return lo


def cbs_splat(entry, drop_first=False, xkey=None):
    """The value the SPLAT ladder's energy converges to as the representation grows without bound.

    `x` is the free-parameter count when `xkey` supplies it, else M; the two are proportional, so
    `p` is identical and only `A` absorbs the factor. `drop_first` refits without the smallest rung,
    which brackets how much the limit leans on the least converged point.

    Returns the dict `cbs.fit_powerlaw` returns; `kind` must be checked."""
    Ms = sorted(M for M, es in entry["splat"].items() if any(e == e for e in es))
    if len(Ms) < 4:
        return {"value": float("nan"), "p": float("nan"), "r2": float("nan"), "kind": "none"}
    es = [min(e for e in entry["splat"][M] if e == e) for M in Ms]
    xs = [xkey(M) for M in Ms] if xkey is not None else list(Ms)
    return cbs.fit_powerlaw(xs, es, drop_first=drop_first)


def gto_ladder(entry):
    """[(nao, gap_mHa), …] in cardinal order. Empty when the system has no CBS."""
    if entry["cbs"] != entry["cbs"]:
        return []
    return [(entry["gto"][b][0], (entry["gto"][b][1] - entry["cbs"]) * 1e3)
            for b in ORDER if b in entry["gto"]]


def gto_wall(entry):
    """[(wall_s, gap_mHa, label), …] — the GTO Pareto frontier, same machine as the energies."""
    if entry["cbs"] != entry["cbs"]:
        return []
    lab = {"cc-pvdz": "DZ", "cc-pvtz": "TZ", "cc-pvqz": "QZ", "cc-pv5z": "5Z", "cc-pv6z": "6Z"}
    return [(entry["gto"][b][2], (entry["gto"][b][1] - entry["cbs"]) * 1e3, lab[b])
            for b in ORDER if b in entry["gto"]]


def seed_uniformity(entry):
    """``(n_per_rung, uniform)`` -- how many seeds each M carries, and whether they all agree.

    A ladder whose rungs carry different seed counts cannot be read as a convergence curve when the
    plotter takes the best seed: more seeds give more chances at a low outlier.
    """
    n = {M: len(es) for M, es in entry.get("splat", {}).items()}
    return n, (len(set(n.values())) <= 1)


def splat_ladder(entry, require_uniform_seeds=False):
    """``{M: (median, lo, hi)}`` gap in mHa over seeds."""
    if entry["cbs"] != entry["cbs"]:
        return {}
    if require_uniform_seeds:
        n, ok = seed_uniformity(entry)
        if not ok:
            raise ValueError(
                "non-uniform seed counts across the ladder: "
                + ", ".join(f"M={M}:{k}" for M, k in sorted(n.items()))
                + ". This function returns the BEST seed, so a rung with more seeds is favoured by "
                  "how often it was run. Re-run the short rungs, or pass "
                  "require_uniform_seeds=False and say so in the figure.")
    out = {}
    for M, es in entry["splat"].items():
        gaps = sorted((e - entry["cbs"]) * 1e3 for e in es)
        n = len(gaps)
        med = gaps[n // 2] if n % 2 else 0.5 * (gaps[n // 2 - 1] + gaps[n // 2])
        out[M] = (med, gaps[0], gaps[-1])
    return out


def systems_with_cbs(data, order=("water", "ethanol", "alanine_dipeptide")):
    """Systems that have a REAL CBS, in a stable display order. Everything else is skipped by the
    callers rather than plotted against a proxy limit."""
    return [s for s in order if s in data and data[s]["cbs"] == data[s]["cbs"]]
