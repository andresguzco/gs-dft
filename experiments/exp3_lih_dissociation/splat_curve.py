"""One cold-start splat point on the LiH or LiF dissociation curve at fixed M, PBE with density fitting.

The cloud is re-optimized at each R with no pre-placed diffuse functions. At stretched geometries
a cold start can land in a higher basin, which ``splat_continuation.py`` avoids. The reported
energy is the exact-Coulomb evaluation of the trained model.

    uv run python -m experiments.exp3_lih_dissociation.splat_curve experiment=exp3_splat_curve system=lih r=1.6 m=38
"""
import time

import hydra
from omegaconf import DictConfig
import jax.random as jr

from experiments.common import systems, builders, results, probes, tracking
from gs_dft import init_model, nao
from gs_dft.ks.train import train, evaluate


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"splat_{cfg.system}_r{cfg.r}_s{cfg.seed}")
    if cfg.m is None:                                   # was the mandatory argv[2]
        raise ValueError("splat_curve needs an explicit splat count: m=<M> "
                         "(the payload sweeps multiples of nao(cc-pVDZ))")
    M = int(cfg.m)
    R = float(cfg.r)
    # cfg.basis (cc-pVDZ) is the coords/nelec/N_aux source here, NOT the reported basis.
    mol = systems.molecule(cfg, r=R)
    xc = builders.xc_of(cfg.xc)
    print(f"{cfg.system} R={R:.3f} / M={M}  N_aux={nao(mol)} steps={cfg.steps} "
          f"grid={cfg.grid_level} seed={cfg.seed}", flush=True)

    model = init_model(mol, M, jr.PRNGKey(cfg.seed))
    ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=builders.auto_screen(M),
                           df_lam=cfg.df_lam, aux_mult=cfg.aux_mult)
    t0 = time.time()
    res = train(ks, model, steps=cfg.steps, monitor=None)
    peak_train_mb = round(probes.peak_gpu_mb())
    e = float(evaluate(ks, res.model))
    e5, bias = builders.grid_audit(mol, xc, res.model, cfg.grid_check, e)
    extra = {} if e5 is None else {f"E_grid{int(cfg.grid_check)}": e5, "grid_bias_mha": bias}
    print(results.result("splat", cfg.system, phase="cold", R=round(R, 3), M=M, seed=cfg.seed,
                        E=e, wall_s=round(time.time() - t0),
                        peak_gpu_mb=round(probes.peak_gpu_mb()), peak_train_mb=peak_train_mb,
                        **extra), flush=True)


if __name__ == "__main__":
    main()
