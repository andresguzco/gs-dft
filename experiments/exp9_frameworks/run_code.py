"""One RKS/PBE ground state in an external code, for the framework comparison (Appendix D).

Runs inside that code's own venv (see setup_envs.sh), so it imports nothing from gs_dft.
Prints one JSON line: energy, basis size, iterations, wall time, peak GPU memory.

    python run_code.py --code mess --system water --basis cc-pvdz --gpu 0
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
BOHR = 1.0 / 0.52917721092
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


def geometry(system):
    """Headerless xyz text in Angstrom, the same geometry GS-DFT uses (experiments.common.systems)."""
    pep = os.path.join(EXP, "exp5_basis_accuracy", "peptides", f"{system}.xyz")
    if os.path.exists(pep):
        return open(pep).read().strip()
    src = open(os.path.join(os.path.dirname(EXP), "gs_dft", "geometries.py")).read()
    name = {"water": "h2o"}.get(system, system)
    return re.search(rf'^{name}_geometry = """(.*?)"""', src, re.S | re.M).group(1).strip()


def atoms(system):
    rows = [ln.split() for ln in geometry(system).splitlines() if ln.strip()]
    return [r[0] for r in rows], [[float(x) for x in r[1:4]] for r in rows]


class GPUPeak:
    """Polls nvidia-smi for the memory used on the given cards (summed); JAX codes run with preallocation off."""

    def __init__(self, gpu, dt=0.2):
        self.gpu, self.dt, self.peak, self._stop = str(gpu), dt, 0, threading.Event()
        self.base = self._read()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _read(self):
        try:
            out = subprocess.run(["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
            return sum(int(x) for x in out.stdout.split())
        except (ValueError, FileNotFoundError):          # CPU nodes have no nvidia-smi
            return 0

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._read())
            time.sleep(self.dt)

    def stop(self):
        self._stop.set()
        self._t.join()
        return max(self.peak - self.base, 0)


def _jax_peak():
    import jax
    st = jax.devices()[0].memory_stats() or {}
    return round(st.get("peak_bytes_in_use", 0) / 2**20)


def run_pyscf(a):
    from pyscf import gto, lib
    lib.num_threads(a.threads)
    mol = gto.M(atom=geometry(a.system), basis=a.basis, cart=a.cart, verbose=0)
    import resource
    if a.xc == "hfx":
        from pyscf import scf
        mf = scf.RHF(mol)
    else:
        from pyscf import dft
        mf = dft.RKS(mol, xc="pbe")
        mf.grids.level = a.grid
        mf.max_cycle = a.max_iter or mf.max_cycle
    if a.df:
        mf = mf.density_fit()
    e = mf.kernel()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    return dict(E=float(e), nao=int(mol.nao), iters=int(mf.cycles), converged=bool(mf.converged),
                df=bool(a.df), max_rss_mb=round(rss))


def run_mess(a):
    import numpy as np
    import mess
    import optimistix as optx
    from mess.structure import Structure
    from mess.hamiltonian import Hamiltonian, minimise
    sym, pos = atoms(a.system)
    s = Structure(np.array([Z[x] for x in sym]), np.array(pos) * BOHR)
    basis = mess.basisset(s, a.basis)
    H = Hamiltonian(basis, xc_method=a.xc, backend=a.backend)
    t = time.time()
    e, _, sol = minimise(H)
    e = float(e)
    t_first = time.time() - t
    return dict(E=e, nao=int(basis.num_orbitals), iters=int(sol.stats["num_steps"]),
                converged=bool(sol.result == optx.RESULTS.successful), t_first_call_s=round(t_first, 2),
                jax_peak_mb=_jax_peak())


def run_d4ft(a):
    import jax
    import haiku as hk
    jax.config.update("jax_enable_x64", a.f64)
    from d4ft.config import get_config
    from d4ft.solver.drivers import build_mf_cgto
    from d4ft.solver.sgd import sgd
    from d4ft.types import Hamiltonian
    path = os.path.join(a.tmp, f"d4ft_{a.system}.xyz")
    with open(path, "w") as f:
        f.write(geometry(a.system) + "\n")
    cfg = get_config("KS-GD-MOL")
    cfg.sys_cfg.mol, cfg.sys_cfg.basis = path, a.basis
    cfg.method_cfg.xc_type, cfg.method_cfg.restricted = "1*gga_x_pbe+1*gga_c_pbe", True
    cfg.intor_cfg.quad_level = a.grid
    if a.max_iter:
        cfg.solver_cfg.epochs = a.max_iter
    key = jax.random.PRNGKey(cfg.method_cfg.rng_seed)
    pmol, H_factory, _, _ = build_mf_cgto(cfg)
    Ht = hk.multi_transform(H_factory)
    params = Ht.init(key)
    logger, _ = sgd(cfg.solver_cfg, Hamiltonian(*Ht.apply), params, key)
    df = logger.data_df
    steps = len(df)
    tstep = float(df.time.astype(float).iloc[min(10, steps - 1):].mean())
    return dict(E=float(df.e_total.astype(float).min()), E_last=float(df.e_total.astype(float).iloc[-1]),
                nao=int(pmol.nao), iters=steps, converged=steps < cfg.solver_cfg.epochs,
                time_per_iter_s=tstep, precision="f64" if a.f64 else "f32", jax_peak_mb=_jax_peak())


def run_dqc(a):
    import torch
    import dqc
    sym, pos = atoms(a.system)
    dev = torch.device("cuda") if a.device == "cuda" else torch.device("cpu")
    zs = torch.tensor([Z[x] for x in sym])
    ps = torch.tensor(pos, dtype=torch.float64) * BOHR
    mol = dqc.Mol((zs, ps), basis=a.basis, grid=a.grid, dtype=torch.float64, device=dev)
    qc = dqc.KS(mol, xc="gga_x_pbe + gga_c_pbe").run()
    return dict(E=float(qc.energy()), iters=None, converged=None, device=a.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", required=True, choices=["pyscf", "mess", "d4ft", "dqc"])
    p.add_argument("--system", required=True)
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--grid", type=int, default=3)
    p.add_argument("--gpu", default="0")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--cart", action="store_true")
    p.add_argument("--f64", action="store_true")
    p.add_argument("--df", action="store_true")
    p.add_argument("--max-iter", dest="max_iter", type=int, default=0,
                   help="cap iterations: the ladder needs cost per iteration, not a converged energy")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default="pyscf_cart")
    p.add_argument("--xc", default="pbe")
    p.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    p.add_argument("--tag", default="")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    mon = GPUPeak(a.gpu)
    t0 = time.time()
    rec = dict(code=a.code, system=a.system, basis=a.basis, grid=a.grid, cart=a.cart, tag=a.tag)
    try:
        fn = {"pyscf": run_pyscf, "mess": run_mess, "d4ft": run_d4ft, "dqc": run_dqc}[a.code]
        rec.update(fn(a), status="ok")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        oom = any(k in msg for k in ("RESOURCE_EXHAUSTED", "out of memory", "OutOfMemory", "MemoryError"))
        rec.update(status="oom" if oom else "error", error=msg[:400])
        traceback.print_exc()
    rec["wall_s"] = round(time.time() - t0, 2)
    rec["peak_gpu_mb"] = mon.stop()
    if rec.get("iters") and "time_per_iter_s" not in rec:
        rec["time_per_iter_s"] = rec["wall_s"] / rec["iters"]
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
