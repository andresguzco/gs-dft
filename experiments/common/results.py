"""The machine-readable result line, with its emitter and parser, used by every runner and plotter.

Grammar (``tests/unit/test_results_contract.py`` enforces it)::

    RESULT kind=<k> system=<s> [key=value ...]
    RESULT_PARTIAL kind=<k> system=<s> step=<n> E=<f> [key=value ...]

- ``kind`` is always the first key, so ``grep "^RESULT kind=ckpt"`` is stable.
- ``system=`` is always present.
- ``E=`` is the reported energy of the line; other energies keep qualified names (``E_ref=``,
  ``E_grid5=``, ``grid_bias_mha=``).
- Values contain no whitespace.

Launchers skip a finished configuration with ``grep -qa "^RESULT "``, with the trailing space
(``GUARD``), so a run that has only written ``RESULT_PARTIAL`` lines is not mistaken for a finished one.

The committed data under ``data/`` is CSV: ``write_csv`` turns run logs into one row per record
(``RESULT``, ``RESULT_PARTIAL``, ``TRACE`` or ``STEP``, the per-step monitor line) with a column per
field, and ``iter_results`` and ``read_records`` read it back as the same dictionaries.
"""
import csv
import fnmatch
import json
import glob
import os
import re

GUARD = "^RESULT "                                # the resume-guard regex payloads grep for

_KINDS = {"splat", "forces_corr", "gto", "gto_mem", "gto_min", "gtol1", "gto_force", "ckpt", "bench", "train",
          "obs_selfref",   # each splat rung against the LARGEST splat rung (Fig 3, second row)
          "gto_selfref",   # each GAUSSIAN rung against that same splat rung
          "obs", "forces", "resource", "opt",
          "cc", "mp2",
          # PBE (splat or GTO rung) differenced against the cc/mp2 archive: the Figure 3 error columns.
          "gto_wf"}


def _fmt(v):
    if isinstance(v, bool):                       # bool before int (bool is an int subclass)
        return "True" if v else "False"
    if isinstance(v, float):
        if v != 0.0 and abs(v) < 1e-6:            # tiny diagnostics (e.g. 1e-16 residuals) survive:
            return f"{v:.2e}"                     # %.8f would flatten them to 0.00000000
        return f"{v:.8f}" if abs(v) < 1e6 else f"{v:.8g}"   # energies want 8 DECIMALS, not sig-figs
    s = str(v)
    if any(c.isspace() for c in s):
        raise ValueError(f"RESULT value {s!r} contains whitespace (would break the parser)")
    return s


def _line(head, kind, system, kv):
    if kind not in _KINDS:
        raise ValueError(f"unknown result kind {kind!r} (use: {'|'.join(sorted(_KINDS))})")
    parts = [head, f"kind={kind}", f"system={system}"]
    parts += [f"{k}={_fmt(v)}" for k, v in kv.items()]
    return " ".join(parts)


def result(kind, system, **kv):
    """Format a final ``RESULT`` line (does not print — the caller owns stdout + the shell
    redirect that the `^RESULT ` guard reads)."""
    from experiments.common import tracking
    tracking.record("RESULT", kind, system, kv)
    return _line("RESULT", kind, system, kv)


def partial(kind, system, *, step, E, **kv):
    """Format a ``RESULT_PARTIAL`` snapshot line (honest E at step k, for walltime-clipped runs)."""
    from experiments.common import tracking
    tracking.record("RESULT_PARTIAL", kind, system, {"step": step, "E": E, **kv})
    return _line("RESULT_PARTIAL", kind, system, {"step": step, "E": E, **kv})


def num(row, key, default=float("nan")):
    """``row[key]`` as a float, or ``default`` when it is absent, empty or unparseable.

    ``row.get(key, fallback)`` is not enough: a key written with no value (``E_grid5=``) is
    present, so the fallback never applies and ``float("")`` raises.
    """
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def parse(line):
    """Parse one RESULT-family line into ``{"_tag": ..., key: str-value, ...}``. Returns None for a
    non-result line. Accepts any ``RESULT``-prefixed tag (``RESULT``, ``RESULT_PARTIAL``,
    ``RESULT_FLOP``) so every plotter reads with ONE order-independent key=value parser instead of a
    positional regex. Values stay strings — callers ``float()`` what they need."""
    toks = line.split()
    if not toks or not toks[0].startswith("RESULT"):
        return None
    out = {"_tag": toks[0]}
    for t in toks[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out


#: The functional every accuracy ladder is measured at. A row with any other ``xc`` belongs to the
#: functional sweep, not the ladder.
XC = "pbe"


def iter_results(paths, *, kind=None, tag="RESULT", xc=None, sources=None):
    """Yield every parsed result dict across ``paths`` (log files, or data CSVs written by
    ``write_csv``), optionally filtered to ``_tag == tag``, ``kind ==`` and ``xc ==``. The single loop
    every plotter uses. ``sources`` restricts a CSV to the rows of the named logs.

    ``xc=results.XC`` keeps the functional sweep out of a single-functional figure. A row with no
    ``xc`` key counts as PBE.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for p in paths:
        if str(p).endswith(".csv"):
            rows = (dict(r, _tag=r["_record"]) for r in read_records(p, record=tag, sources=sources))
        else:
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            rows = (d for d in map(parse, lines) if d is not None and d["_tag"] == tag)
        for d in rows:
            if kind is not None and d.get("kind") != kind:
                continue
            if xc is not None and str(d.get("xc", XC)).lower() != str(xc).lower():
                continue
            yield d


#: Per-step monitor fields, renamed to plain column names.
_STEP_FIELDS = {"|g|": "grad_norm", "worstŜ": "worst_overlap", "λmin": "lam_min", "λmax": "lam_max",
                "λmin/λmax": "lam_ratio", "cond(V)": "cond_V"}
_STEP = re.compile(r"^\s*step\s+(\d+)\s+(.*)")


def record(line):
    """``(record, fields)`` for a ``RESULT``, ``RESULT_PARTIAL``, ``TRACE`` or per-step monitor
    line, else ``None``. Values stay strings."""
    d = parse(line)
    if d is not None:
        tag = d.pop("_tag")
        return tag, d
    if line.startswith("TRACE "):
        return "TRACE", dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
    m = _STEP.match(line)
    if m:
        out = {"step": m.group(1)}
        for t in m.group(2).split():
            if "=" in t:
                k, v = t.split("=", 1)
                out[_STEP_FIELDS.get(k, k)] = v
        return "STEP", out
    return None


def write_csv(paths, dest):
    """Write the records of ``paths`` (in order) to one CSV: ``_source`` (the log's file name),
    ``_record``, then one column per field in first-seen order. Returns the number of rows."""
    rows, cols = [], {}
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                r = record(line)
                if r is None:
                    continue
                row = {"_source": os.path.basename(p), "_record": r[0], **r[1]}
                rows.append(row)
                cols.update(dict.fromkeys(row))
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_records(path, *, record=None, sources=None):
    """The rows of a data CSV, in order, as dicts of their non-empty fields (``_source`` and
    ``_record`` included). ``record`` and ``sources`` filter them; ``sources`` may hold glob patterns."""
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if record is not None and row["_record"] != record:
                continue
            if sources is not None and not any(fnmatch.fnmatchcase(row["_source"], s) for s in sources):
                continue
            yield {k: v for k, v in row.items() if v != ""}


def csv_sources(path, pattern="*"):
    """The ``_source`` names of a data CSV that match ``pattern``, in order of first appearance."""
    seen = {}
    for r in read_records(path):
        if fnmatch.fnmatchcase(r["_source"], pattern):
            seen.setdefault(r["_source"], None)
    return list(seen)


class jsonl_writer:
    """Append-or-truncate JSONL step-stream writer (exp8's per-step instrumentation). ``resume``
    appends to preserve a partial run's history; otherwise it truncates. Flushes each row so a
    preempted run keeps everything written so far."""

    def __init__(self, path, *, resume=False):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a" if resume else "w", encoding="utf-8")

    def write(self, record):
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

ORDER_FILE = "log_order.txt"


def ordered_logs(dirpath, pattern="*.log"):
    """Logs in OLDEST-FIRST order, from a committed manifest when one exists.

    `log_order.txt` is that order, written where the mtimes are still true (see `write_order`).
    Files present but unlisted are appended in mtime order, so adding a run does not require
    regenerating the manifest for the figure to stay correct."""
    found = {os.path.basename(p): p for p in glob.glob(os.path.join(dirpath, pattern))}
    manifest = os.path.join(dirpath, ORDER_FILE)
    if not os.path.exists(manifest):
        return sorted(found.values(), key=lambda p: (os.path.getmtime(p), p))
    out, seen = [], set()
    with open(manifest, errors="replace") as fh:
        for line in fh:
            name = line.strip()
            if name and not name.startswith("#") and name in found and name not in seen:
                out.append(found[name])
                seen.add(name)
    rest = [p for n, p in found.items() if n not in seen]
    return out + sorted(rest, key=lambda p: (os.path.getmtime(p), p))


def write_order(dirpath, pattern="*.log"):
    """Freeze this directory's current mtime order into `log_order.txt`. Run where the runs
    actually happened; the file is what makes the figures reproducible from a clone."""
    logs = sorted(glob.glob(os.path.join(dirpath, pattern)),
                  key=lambda p: (os.path.getmtime(p), p))
    dest = os.path.join(dirpath, ORDER_FILE)
    with open(dest, "w") as fh:
        fh.write("# Oldest first. Written by experiments.common.results.write_order because git\n"
                 "# does not preserve mtimes and these loaders resolve duplicate rungs by recency.\n")
        for p in logs:
            fh.write(os.path.basename(p) + "\n")
    return dest, len(logs)


#: Rungs whose checkpoint is not part of the ladder it would be plotted beside: ethanol M=432 was
#: trained at a different M and never polished.
STALE_RUNGS = {("ethanol", 432)}
