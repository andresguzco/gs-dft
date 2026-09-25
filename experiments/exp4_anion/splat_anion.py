"""One splat run on a bare anion (charge -1, restricted closed-shell).

The cloud gets no diffuse functions and no anion-specific initialization. The final energy is
also evaluated on the level-5 grid.

    uv run python -m experiments.exp4_anion.splat_anion experiment=exp4_splat_anion system=f_anion m=28
"""
import time

import hydra
from omegaconf import DictConfig
import jax.random as jr

from experiments.common import systems, builders, results, probes, tracking
from gs_dft import init_model, nao
from gs_dft.ks.train import train, evaluate, monitor


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"splat_{cfg.system}_s{cfg.seed}")
    if cfg.m is None:
        raise ValueError("splat_anion needs an explicit splat count: m=<M>")
    M = int(cfg.m)
    mol = systems.molecule(cfg)                          # charge −1 comes from the registry
    nao_ref = int(nao(mol))
    xc = builders.xc_of(cfg.xc)
    print(f"{cfg.system} (charge {systems.charge(cfg.system)}, {mol.nelectron} e-)  M={M} "
          f"({M / nao_ref:.1f}x nao={nao_ref})  steps={cfg.steps} seed={cfg.seed}", flush=True)

    model = init_model(mol, M, jr.PRNGKey(cfg.seed))
    ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=builders.auto_screen(M),
                           df_lam=cfg.df_lam, aux_mult=cfg.aux_mult)
    t0 = time.time()
    res = train(ks, model, steps=cfg.steps,
                monitor=monitor(int(cfg.monitor)) if int(cfg.monitor) else None)
    peak_train_mb = round(probes.peak_gpu_mb())
    e = float(evaluate(ks, res.model))

    extra = {}
    e5, bias = builders.grid_audit(mol, xc, res.model, cfg.grid_check, e)
    if e5 is not None:
        extra[f"E_grid{cfg.grid_check}"] = e5
        extra["grid_bias_mha"] = bias
    print(results.result("splat", cfg.system, M=M, nao=nao_ref, steps=cfg.steps, seed=cfg.seed,
                        E=e, wall_s=round(time.time() - t0),
                        peak_gpu_mb=round(probes.peak_gpu_mb()), peak_train_mb=peak_train_mb,
                        **extra), flush=True)


if __name__ == "__main__":
    main()
