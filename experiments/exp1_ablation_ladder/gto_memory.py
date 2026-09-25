"""Gaussian baselines for Figure 2(b): peak memory against free parameters.

    gto_exact   exact four-center ERIs, streamed like the splat side
    gto_df      + density fitting
    gto_fast    + Cauchy-Schwarz screening, chunked RI and a chunked grid

    uv run python -m experiments.exp1_ablation_ladder.gto_memory <system> <basis> <arm>
"""
import sys
import time

from experiments.common import probes, resources, results, systems
from dftax.basis.loader import build_basis_data
from dftax.ks import KS, scf, df, exact
from dftax.energy.xc import PBE
from gs_dft import nao
from gs_dft.ks.train import native_grid

GRID = 3
SCREEN = 1e-7          # matches the splat side's pairlist(eps=1e-7)
CHUNK = 4096           # matches the splat side's grid chunk

ARMS = {
    "gto_exact": (lambda: exact(stream=True), None),
    "gto_df":    (lambda: df("def2-universal-jkfit", chunk=None), None),
    "gto_fast":  (lambda: df("def2-universal-jkfit", chunk=CHUNK, screen=SCREEN), CHUNK),
}


def main(system, basis, arm="gto_df"):
    spec, chunk = ARMS[arm]
    mol = systems.molecule(system=system, basis=basis)
    n_occ = int(mol.nelectron // 2)
    print(f"# exp1 {arm}: {system}/{basis} nao={int(nao(mol))} chunk={chunk}", flush=True)
    failed = "none"
    t0 = time.perf_counter()
    try:
        ks = KS(mol, PBE(), grid=native_grid(mol, GRID, chunk=chunk), coulomb=spec())
        res = scf(ks, max_iter=100)
        E, conv = float(res.e_tot), bool(res.converged)
        peak = round(probes.peak_gpu_mb())
    except Exception as exc:                                   # noqa: BLE001
        E, conv = float("nan"), False
        peak = round(probes.peak_gpu_mb())
        failed = ("oom" if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc)
                  else type(exc).__name__)
        print(f"# {arm} FAILED ({failed}): {str(exc)[:160]}", flush=True)
    bd = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    par = resources.gto_params(bd, int(nao(mol)), n_occ)
    print(results.result("gto_mem", system, arm=arm, basis=basis, nao=int(nao(mol)), n_occ=n_occ,
                        params=par["total"], params_basis=par["basis"], E=E, conv=conv,
                        failed=failed, peak_train_mb=peak,
                        wall_s=round(time.perf_counter() - t0, 1)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "gto_df")
