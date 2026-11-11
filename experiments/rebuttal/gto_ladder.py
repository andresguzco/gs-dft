"""GTO reference point: dftax DF-RKS at one cc-pVXZ basis, same functional and grid as the splats.

    python -m experiments.rebuttal.gto_ladder --system porphine --basis cc-pvtz
    python -m experiments.exp5_basis_accuracy.cbs_fit <log dir>          # the CBS fit afterwards

One basis per process, so each point starts on an empty device pool. The RESULT row has the
fields cbs_fit reads (system, basis, nao, E, wall_s, peak_mb, converged); ``peak_mb`` is the GPU
peak here, ``rss_mb`` the host's.

Same SCF as experiments/common/reference.py (def2-universal-jkfit, a level-shift retry), with the
3-center tensor forced materialized: dftax's auto policy streams it past a quarter of the device
pool, rebuilding the integrals every SCF iteration. With several visible GPUs it is sharded over
them (KS(mesh=mesh())). The gto_memory estimate prints first; a point it rules out writes
``RESULT kind=gto_mem ... failed=est_oom`` without running (``--force`` runs it anyway), and an
OOM writes ``failed=oom``. Neither is a ``kind=gto`` row, so cbs_fit never sees them.
"""
import argparse
import resource
import time


def _args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--xc", default="pbe")
    ap.add_argument("--grid_level", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=4096, help="XC grid chunk")
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="run even when the estimate says no")
    return ap.parse_args()


def _peak_gpu_mb():
    import jax
    peaks = []
    for d in jax.local_devices():
        try:
            peaks.append(d.memory_stats().get("peak_bytes_in_use", 0) / 1e6)
        except Exception:
            pass
    return max(peaks) if peaks else -1.0


def main():
    a = _args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import dftax
    from dftax.ks import KS, scf, df, mesh
    from experiments.common import systems, builders, results
    from experiments.rebuttal import gto_memory
    from gs_dft import nao as nao_of
    from gs_dft.ks.train import native_grid

    mol = systems.molecule(system=a.system, basis=a.basis)
    nao = int(nao_of(mol))
    G = jax.local_device_count()
    est = gto_memory.estimate(list(mol.symbols), mol.atom_coords(), a.basis, gpus=G,
                              nocc=int(mol.nelectron) // 2,
                              hybrid=builders.xc_of(a.xc).hf_coeff != 0.0)
    print(f"# dftax {dftax.__version__} from {dftax.__file__}  devices={jax.devices()}", flush=True)
    print(f"# {a.system}/{a.basis}: {len(mol.symbols)} atoms  nao={nao}  naux={est['naux']}  G={G}  "
          f"int3c {est['int3c_gib']:.1f} GiB  est. peak {est['peak_gib']:.1f} GiB "
          f"({est['verdict']})", flush=True)
    common = dict(basis=a.basis, mode="df", xc=a.xc, nao=nao, naux=est["naux"], gpus=G,
                  grid_level=a.grid_level, est_peak_gib=round(est["peak_gib"], 2))
    if est["verdict"] == "no" and not a.force:
        print(results.result("gto_mem", a.system, failed="est_oom", **common), flush=True)
        return

    t0 = time.perf_counter()
    try:
        ks = KS(mol, builders.xc_of(a.xc), grid=native_grid(mol, a.grid_level, chunk=a.chunk),
                coulomb=df("def2-universal-jkfit", chunk=None), mesh=mesh() if G > 1 else None)
        t_build = time.perf_counter() - t0
        res = scf(ks, max_iter=a.max_iter)
        shifted = False
        if not res.converged:
            res, shifted = scf(ks, max_iter=2 * a.max_iter, level_shift=0.2), True
    except Exception as exc:                                       # noqa: BLE001
        msg = str(exc)
        failed = ("oom" if "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg
                  else type(exc).__name__)
        print(f"# FAILED ({failed}): {msg[:300]}", flush=True)
        print(results.result("gto_mem", a.system, failed=failed, peak_mb=round(_peak_gpu_mb()),
                             wall_s=round(time.perf_counter() - t0, 1), **common), flush=True)
        return
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(results.result("gto", a.system, E=float(res.e_tot), converged=bool(res.converged),
                         n_iter=int(res.n_iter), level_shift=shifted,
                         wall_s=round(time.perf_counter() - t0, 1), build_s=round(t_build, 1),
                         peak_mb=round(_peak_gpu_mb()), rss_mb=round(rss), **common), flush=True)


if __name__ == "__main__":
    main()
