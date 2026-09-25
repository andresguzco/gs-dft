"""Peak memory of a fixed Gaussian basis under direct minimization, on the same grid, auxiliary basis
and functional as ``gto_ref.py``.
"""
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from dftax.ks import KS, df, minimize
from gs_dft import nao as nao_of
from gs_dft.ks.train import native_grid
from experiments.common import builders, probes, results, systems, tracking


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    system, basis = str(cfg.system), str(cfg.basis)
    tracking.init(cfg, __name__, name=f"gtoDM_{system}_{basis}")
    mol = systems.molecule(cfg)
    xc = builders.xc_of(cfg.xc)
    grid = native_grid(mol, cfg.grid_level, chunk=4096)
    ks = KS(mol, xc, grid=grid, coulomb=df("def2-universal-jkfit"))

    t0 = time.perf_counter()
    res = minimize(ks, max_steps=int(cfg.get("dm_max_steps", 2000)),
                   g_tol=float(cfg.get("dm_g_tol", 1e-6)))
    wall = time.perf_counter() - t0
    print(results.result("gto_min", system, basis=basis, nao=int(nao_of(mol)), xc=str(cfg.xc),
                         E=float(res.e_tot), converged=bool(res.converged),
                         n_iter=int(getattr(res, "n_iter", -1)),
                         peak_gpu_mb=round(probes.peak_gpu_mb()),
                         wall_s=round(wall, 1)), flush=True)


if __name__ == "__main__":
    main()
