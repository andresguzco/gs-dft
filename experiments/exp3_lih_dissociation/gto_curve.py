"""One Gaussian point on the LiH or LiF dissociation curve: RKS/PBE at (R, basis).

Restricted, so LiX dissociates on the ionic Li+X- diabat. Uses exact in-core ERIs and the level-3
grid of the splat curve.

    uv run python -m experiments.exp3_lih_dissociation.gto_curve experiment=exp3_gto_curve system=lih r=1.6 basis=aug-cc-pvdz
"""
import hydra
from omegaconf import DictConfig

from experiments.common import systems, builders, reference, results, tracking


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"gto_{cfg.system}_r{cfg.r}")
    R = float(cfg.r)
    mol = systems.molecule(cfg, r=R)
    ref = reference.gto_reference(mol, builders.xc_of(cfg.xc), grid_level=cfg.grid_level, mode="conv")
    print(results.result("gto", cfg.system, R=round(R, 3), basis=cfg.basis, nao=ref.nao,
                        E=ref.e, converged=ref.converged, wall_s=round(ref.wall_s, 2)), flush=True)


if __name__ == "__main__":
    main()
