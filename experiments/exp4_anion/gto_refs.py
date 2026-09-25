"""Gaussian RKS/PBE references (DF, def2-universal-jkfit) for the bare anions on the plain and
augmented ladders, on the same engine, functional and grid as the splat side.

    uv run python -m experiments.exp4_anion.gto_refs experiment=exp4_gto_refs system=f_anion basis=aug-cc-pvdz
"""
import hydra
from omegaconf import DictConfig

from experiments.common import systems, builders, reference, results, tracking, probes


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"gto_{cfg.system}_{cfg.basis}")
    mol = systems.molecule(cfg)                           # charge −1 from the registry
    ref = reference.gto_reference(mol, builders.xc_of(cfg.xc), grid_level=cfg.grid_level, mode="df")
    print(results.result("gto", cfg.system, basis=cfg.basis, nao=ref.nao, E=ref.e,
                        conv=ref.converged, peak_gpu_mb=round(probes.peak_gpu_mb()),
                        wall_s=round(ref.wall_s)), flush=True)


if __name__ == "__main__":
    main()
