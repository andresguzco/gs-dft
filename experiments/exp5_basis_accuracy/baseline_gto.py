"""Gaussian cc-pVXZ baseline for Figure 5: one RKS/PBE calculation per process, reporting the energy,
wall time and peak RSS.

Runs on dftax's own Gaussian KS-DFT with the same functional and level-3 grid as the splat runs,
so the distance from the complete-basis limit is the basis alone.

- ``mode=conv``: exact in-core ERIs.
- ``mode=df``: RI-JK with def2-universal-jkfit, for systems whose exact ERIs do not fit in memory.

A calculation that does not converge is retried once with a level shift::

    uv run python -m experiments.exp5_basis_accuracy.baseline_gto experiment=exp5_baseline_gto system=h2o basis=cc-pvqz
"""
import resource

import hydra
from omegaconf import DictConfig

from experiments.common import systems, builders, reference, results, tracking


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mol = systems.molecule(cfg)
    # run name == the on-disk log stem, so a wandb run and its log are matchable without guesswork
    tracking.init(cfg, __name__, name=f"gto_{cfg.system}_{cfg.basis}_{cfg.mode}")
    ref = reference.gto_reference(mol, builders.xc_of(cfg.xc),
                                  grid_level=cfg.grid_level, mode=cfg.mode)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # KB→MB (Linux)
    print(results.result("gto", cfg.system, basis=cfg.basis, mode=cfg.mode, xc=cfg.xc, nao=ref.nao,
                        E=ref.e, converged=ref.converged, n_iter=ref.n_iter,
                        wall_s=round(ref.wall_s, 1), peak_mb=round(peak_mb)), flush=True)


if __name__ == "__main__":
    main()
