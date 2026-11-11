"""GPU4PySCF RKS/PBE with density fitting and with direct SCF: the GTO references dftax cannot hold,
and experiment 4 (the cyclosporin cc-pVTZ pair, with its GPU and host memory).

    $ENV/bin/python experiments/rebuttal/gpu4pyscf_df.py --systems cyclosporin \
        --out /scratch/.../ref2/cc-pvtz/cyclosporin [--basis cc-pvtz] [--timeout 10800]

Each (system, mode) runs in its own process (``--one``), so its device memory starts empty. A run
records the device-level peak GPU memory (sampled used memory, which counts CuPy's pool), the host
peak memory, the wall time, the energy and its convergence, and a failure as ``oom`` / ``oom_host`` /
``timeout`` / the exception name. Direct SCF starts from the DF density when DF converged. Every SCF
cycle saves its density matrix (``--ckpt``), so a run cut by a timeout or the job's walltime resumes
from it in the next job (chained short jobs). The driver appends one JSON line per run to
``runs.jsonl`` (resumable) and writes ``gpu4pyscf_df.csv``, with E_DF - E_direct per system. GPU4PySCF
uses every visible GPU (its J/K, DF and XC kernels split work over ``num_devices``).

Runs in its own environment (setup_gpu4pyscf.sh): PySCF, GPU4PySCF and CuPy, no JAX, so it reads
the xyz files directly instead of through experiments.common.systems.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XYZ_DIRS = [os.path.join(ROOT, "common", "geometry"),
            os.path.join(ROOT, "exp5_basis_accuracy", "peptides")]


# Net charges of the FMODB proteins (experiments/common/fmodb.py); every other system is neutral.
CHARGES = {"PJM49": 1, "9LQN2": 2, "LG5K9": 1, "38KNL": 0, "25J8R": 5}


def _xyz(name):
    for d in XYZ_DIRS:
        p = os.path.join(d, f"{name}.xyz")
        if os.path.exists(p):
            with open(p) as fh:
                lines = [ln.split() for ln in fh if ln.split()]
            return "; ".join(" ".join(t[:4]) for t in lines), len(lines)
    raise FileNotFoundError(f"{name}.xyz not in {XYZ_DIRS}")


class _PeakSampler:
    """Max device used bytes (total - free, so CUDA modules and handles count) and max CuPy pool
    bytes (the arrays alone), every ``dt`` seconds."""

    def __init__(self, dt=0.05):
        import cupy
        self.rt, self.dt, self.peak, self.stop = cupy.cuda.runtime, dt, 0, False
        self.pool, self.pool_peak = cupy.get_default_memory_pool(), 0
        free, total = self.rt.memGetInfo()
        self.base = total - free
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            free, total = self.rt.memGetInfo()
            self.peak = max(self.peak, total - free)
            self.pool_peak = max(self.pool_peak, self.pool.total_bytes())
            time.sleep(self.dt)

    def done(self):
        self.stop = True
        self.t.join()
        free, total = self.rt.memGetInfo()
        self.peak = max(self.peak, total - free)
        return self.peak / 1e6, (self.peak - self.base) / 1e6, self.pool_peak / 1e6


def one(a):
    """One SCF in this process; prints one JSON line."""
    import cupy
    import pyscf
    import gpu4pyscf
    from gpu4pyscf.dft import rks

    atom, natm = _xyz(a.system)
    row = {"system": a.system, "mode": a.mode, "basis": a.basis, "xc": a.xc, "n_atoms": natm,
           "grid_level": a.grid_level, "gpu4pyscf": gpu4pyscf.__version__,
           "pyscf": pyscf.__version__, "gpu": cupy.cuda.runtime.getDeviceProperties(0)["name"]
           .decode()}
    mol = pyscf.M(atom=atom, basis=a.basis, charge=CHARGES.get(a.system, 0), spin=0, verbose=a.verbose,
                  max_memory=a.host_mem_mb)
    row["nao"] = int(mol.nao)
    sampler = _PeakSampler()
    t0 = time.perf_counter()
    try:
        mf = rks.RKS(mol, xc=a.xc)
        if a.mode == "df":
            mf = mf.density_fit(auxbasis=a.auxbasis)
        mf.grids.level = a.grid_level
        mf.conv_tol = a.conv_tol
        mf.max_cycle = a.max_cycle
        import numpy as np
        dm0, row["init"] = None, "minao"
        if a.ckpt and os.path.exists(a.ckpt):                       # resume where the last job stopped
            dm0, row["init"] = cupy.asarray(np.load(a.ckpt)), "ckpt"
        elif a.mode == "direct" and a.dm and os.path.exists(a.dm):  # start from the DF solution
            dm0, row["init"] = cupy.asarray(np.load(a.dm)), "df_dm"
        if a.ckpt:
            def save(envs):                         # every cycle, atomically: a killed job resumes here
                np.save(a.ckpt + ".tmp.npy", cupy.asnumpy(envs["dm"]))
                os.replace(a.ckpt + ".tmp.npy", a.ckpt)
            mf.callback = save
        e = mf.kernel(dm0=dm0)
        if a.ckpt and mf.converged and os.path.exists(a.ckpt):
            os.remove(a.ckpt)
        row.update(E=float(e), converged=bool(mf.converged), failed="none",
                   n_cycles=int(getattr(mf, "cycles", -1)))
        if a.mode == "df" and a.dm and mf.converged:
            np.save(a.dm, cupy.asnumpy(mf.make_rdm1()))
        if a.mode == "df":
            row["auxbasis"] = a.auxbasis or "pyscf-default"
            aux = getattr(mf.with_df, "auxmol", None)
            row["naux"] = int(aux.nao) if aux is not None else -1
    except Exception as exc:                                       # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        oom = "OutOfMemory" in msg or "out of memory" in msg.lower()
        row.update(E=float("nan"), converged=False, failed="oom" if oom else type(exc).__name__,
                   error=msg[:300])
    row["wall_s"] = round(time.perf_counter() - t0, 1)
    import resource
    row["peak_host_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)   # KB on Linux
    row["peak_gpu_mb"], row["peak_gpu_delta_mb"], row["peak_pool_mb"] = (
        round(x) for x in sampler.done())
    print("ROW " + json.dumps(row), flush=True)


FINAL = ("none", "oom", "oom_host", "timeout")   # outcomes a rerun keeps; any other is retried


def _ckpt(out, system, basis, mode):
    return os.path.join(out, f"{system}_{basis}_{mode}_ckpt_dm.npy")


def _done(path):
    """The (system, mode) runs a rerun skips: a final outcome, except a timeout that left a checkpoint
    (the next job resumes that SCF)."""
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            r = json.loads(ln)
            if r.get("failed") == "timeout" and os.path.exists(
                    _ckpt(os.path.dirname(path), r["system"], r["basis"], r["mode"])):
                continue
            if r.get("failed") in FINAL:
                out[(r["system"], r["mode"])] = r
    return out


def drive(a):
    os.makedirs(a.out, exist_ok=True)
    runs = os.path.join(a.out, "runs.jsonl")
    done = _done(runs)
    for name in a.systems.split(","):
        for mode in ("df", "direct"):
            if (name, mode) in done:
                continue
            log = os.path.join(a.out, f"{name}_{a.basis}_{mode}.log")
            cmd = [sys.executable, os.path.abspath(__file__), "--one", "--system", name,
                   "--mode", mode, "--basis", a.basis, "--xc", a.xc,
                   "--grid_level", str(a.grid_level), "--conv_tol", str(a.conv_tol),
                   "--max_cycle", str(a.max_cycle), "--verbose", str(a.verbose),
                   "--host_mem_mb", str(a.host_mem_mb),
                   "--dm", os.path.join(a.out, f"{name}_{a.basis}_df_dm.npy"),
                   "--ckpt", _ckpt(a.out, name, a.basis, mode)] + (
                       ["--auxbasis", a.auxbasis] if a.auxbasis else [])
            print(f"# {time.strftime('%T')} {name} {mode}", flush=True)
            t0 = time.perf_counter()
            with open(log, "w") as fh:
                try:
                    p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                       timeout=a.timeout)
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            rows = [json.loads(ln[4:]) for ln in open(log) if ln.startswith("ROW ")]
            row = rows[-1] if rows else {
                "system": name, "mode": mode, "basis": a.basis, "E": float("nan"),
                "converged": False,
                "failed": "timeout" if rc == "timeout" else ("oom_host" if rc == -9 else f"rc{rc}"),
                "wall_s": round(time.perf_counter() - t0, 1)}
            print(f"  {row.get('failed')} E={row.get('E')} wall={row.get('wall_s')}s "
                  f"peak={row.get('peak_gpu_mb')} MB", flush=True)
            with open(runs, "a") as fh:
                fh.write(json.dumps(row) + "\n")
    summarize(runs, os.path.join(a.out, "gpu4pyscf_df.csv"))


def summarize(runs, path):
    by = {}
    for r in _done(runs).values():
        by.setdefault(r["system"], {})[r["mode"]] = r
    cols = ["system", "n_atoms", "nao", "basis", "naux", "auxbasis",
            "df_failed", "df_wall_s", "df_peak_gpu_mb", "df_peak_pool_mb", "df_peak_host_mb", "df_E",
            "df_converged", "direct_failed", "direct_wall_s", "direct_peak_gpu_mb",
            "direct_peak_pool_mb", "direct_peak_host_mb", "direct_E", "direct_converged",
            "dE_df_direct_uHa"]
    rows = []
    for s, m in by.items():
        d, x = m.get("df", {}), m.get("direct", {})
        ok = all(m_.get("failed") == "none" and m_.get("converged") for m_ in (d, x))
        rows.append({"system": s, "n_atoms": d.get("n_atoms", x.get("n_atoms")),
                     "nao": d.get("nao", x.get("nao")), "basis": d.get("basis", x.get("basis")),
                     "naux": d.get("naux"), "auxbasis": d.get("auxbasis"),
                     "df_failed": d.get("failed"), "df_wall_s": d.get("wall_s"),
                     "df_peak_gpu_mb": d.get("peak_gpu_mb"), "df_peak_pool_mb": d.get("peak_pool_mb"),
                     "df_peak_host_mb": d.get("peak_host_mb"), "df_E": d.get("E"),
                     "df_converged": d.get("converged"), "direct_converged": x.get("converged"),
                     "direct_peak_host_mb": x.get("peak_host_mb"),
                     "direct_failed": x.get("failed"), "direct_wall_s": x.get("wall_s"),
                     "direct_peak_gpu_mb": x.get("peak_gpu_mb"),
                     "direct_peak_pool_mb": x.get("peak_pool_mb"), "direct_E": x.get("E"),
                     "dE_df_direct_uHa": (d["E"] - x["E"]) * 1e6 if ok else None})
    rows.sort(key=lambda r: r["n_atoms"] or 0)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['system']:13s} {r['n_atoms'] or 0:4d} nao {r['nao'] or 0:5d}  "
              f"DF {r['df_failed']} {r['df_wall_s']}s {r['df_peak_gpu_mb']}MB "
              f"(pool {r['df_peak_pool_mb']})  direct {r['direct_failed']} {r['direct_wall_s']}s "
              f"{r['direct_peak_gpu_mb']}MB (pool {r['direct_peak_pool_mb']})  "
              f"dE {r['dE_df_direct_uHa']} uHa")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", default="cyclosporin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--basis", default="cc-pvtz")
    ap.add_argument("--xc", default="PBE")
    ap.add_argument("--auxbasis", default=None, help="DF aux basis; default PySCF's choice")
    ap.add_argument("--grid_level", type=int, default=3)
    ap.add_argument("--conv_tol", type=float, default=1e-9)
    ap.add_argument("--max_cycle", type=int, default=200)
    ap.add_argument("--dm", default=None, help="DF density matrix: saved by df, the start of direct")
    ap.add_argument("--ckpt", default=None, help="per-cycle density matrix: saved every cycle, resumed from")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per (system, mode)")
    ap.add_argument("--host_mem_mb", type=int, default=64000)
    ap.add_argument("--verbose", type=int, default=4)
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--system")
    ap.add_argument("--mode", choices=("df", "direct"))
    a = ap.parse_args()
    if a.one:
        one(a)
    else:
        if not a.out:
            raise SystemExit("--out is required")
        drive(a)


if __name__ == "__main__":
    main()
