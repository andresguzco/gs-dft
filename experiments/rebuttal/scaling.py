"""Experiment 2: strong scaling at G = 1, 2, 4 on one node, through the step timer.

    python -m experiments.rebuttal.scaling --systems retinal=1022,chignolin,cyclosporin \
        --out /scratch/.../exp2 [--dry]
    python -m experiments.rebuttal.plot_scaling /scratch/.../exp2/scaling.csv

Each (system, G, repeat) is one ``experiments.bench.step`` process at M = n_ao(TZ) (``name=M`` gives
M for a system data/systems.json does not list), 60 steps, the median over steps 11-59 (``--skip 11``),
refreshes timed apart. The runs go one at a time, each alone on the node (CUDA_VISIBLE_DEVICES = the
first G GPUs), repeat by repeat, so every G of a repeat sees the same node and nothing shares the
host with a timed run. A run whose log already has a RESULT row is skipped, so a rerun resumes.
Every RESULT kind=bench row lands in ``scaling.csv``.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("system", "n_atoms", "M", "G", "rep", "step_med_s", "step_p10_s", "step_p90_s",
          "refresh_med_s", "n_refresh", "peak_gpu_mb", "peak_gpu_mb_dev", "compiles_after_step0",
          "startup_s", "E0", "g0", "E_last", "log")


def _args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", default="retinal=1022,chignolin,cyclosporin",
                    help="names, or name=M where data/systems.json has no n_ao for the system")
    ap.add_argument("--G", default="1,2,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--skip", type=int, default=11)
    ap.add_argument("--refresh", type=int, default=10)
    ap.add_argument("--basis", default="cc-pvtz", help="M = n_ao of this basis")
    ap.add_argument("--out", required=True, help="directory for the logs and scaling.csv")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry", action="store_true", help="print the commands only")
    return ap.parse_args()


def _jobs(a):
    with open(os.path.join(HERE, "data", "systems.json")) as fh:
        table = json.load(fh)
    jobs = []
    for rep in range(a.reps):
        for item in a.systems.split(","):
            name, _, m = item.partition("=")
            M = int(m) if m else int(table[name]["nao"][a.basis])
            for g in sorted(int(x) for x in a.G.split(",")):
                log = os.path.join(a.out, f"bench_{name}_M{M}_G{g}_r{rep}.log")
                cmd = [a.python, "-m", "experiments.bench.step", "--system", name, "--M", str(M),
                       "--steps", str(a.steps), "--skip", str(a.skip), "--refresh", str(a.refresh),
                       "--G", str(g)]
                jobs.append({"system": name, "M": M, "G": g, "rep": rep, "log": log, "cmd": cmd})
    return jobs


def _done(log):
    if not os.path.exists(log):
        return False
    with open(log, errors="replace") as fh:
        return any(ln.startswith("RESULT ") for ln in fh)


def run(a, jobs):
    """One run at a time, alone on the node: a timed run shares neither GPUs nor the host."""
    queue = [j for j in jobs if not _done(j["log"])]
    print(f"# {len(jobs)} runs, {len(jobs) - len(queue)} already done", flush=True)
    for j in queue:
        gpus = list(range(j["G"]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
                   PYTHONUNBUFFERED="1", XLA_PYTHON_CLIENT_MEM_FRACTION="0.92")
        with open(j["log"], "a") as fh:
            fh.write(f"# {time.strftime('%F %T')} gpus={gpus} :: {' '.join(j['cmd'])}\n")
            fh.flush()
            print(f"# start {j['system']} G={j['G']} rep={j['rep']} on {gpus}", flush=True)
            rc = subprocess.run(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        print(f"# done {j['system']} G={j['G']} rep={j['rep']} rc={rc}", flush=True)


def collect(a, jobs):
    from experiments.common.results import parse
    rows = []
    for j in jobs:
        if not os.path.exists(j["log"]):
            continue
        with open(j["log"], errors="replace") as fh:
            res = [parse(ln) for ln in fh if ln.startswith("RESULT ")]
        res = [r for r in res if r and r.get("kind") == "bench"]
        if res:
            r = res[-1]
            rows.append({k: (j[k] if k in ("system", "M", "G", "rep", "log")
                             else r.get(k, "")) for k in FIELDS})
    path = os.path.join(a.out, "scaling.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"# wrote {path} ({len(rows)} rows)", flush=True)


def main():
    a = _args()
    os.makedirs(a.out, exist_ok=True)
    jobs = _jobs(a)
    if a.dry:
        for j in jobs:
            print(f"G={j['G']} {' '.join(j['cmd'])} >> {j['log']}")
        return
    run(a, jobs)
    collect(a, jobs)


if __name__ == "__main__":
    main()
