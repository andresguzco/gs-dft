"""One splat run at (system, M) with screened density fitting.

Trains ``SplatKS(..., coulomb=df(), screen=pairlist())`` with the pair list and grid blocks
rebuilt every ``refresh`` steps, and reports the exact streamed energy of the trained model. Every
``partial_every`` steps a ``RESULT_PARTIAL`` line records the current energy.

    uv run python -m experiments.exp5_basis_accuracy.splat_run experiment=exp5_splat_run system=water m=96 seed=1

``experiment=exp5_refresh`` is the 12,000-step configuration of Figure 5; ``exp5_splat_run`` is
the one of Table 1.
"""
import os
import time

import hydra
from omegaconf import DictConfig
import jax.random as jr

from experiments.common import systems, builders, results, probes, tracking
from gs_dft import init_model, nao as nao_of
from gs_dft.ks.train import train, monitor, evaluate, ckpt as ckpt_spec


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.m is None:
        raise ValueError("splat_run needs an explicit splat count: m=<M> "
                         "(the launcher's M-ladder supplies it)")
    M = int(cfg.m)
    mol = systems.molecule(cfg)
    nao = int(nao_of(mol))
    xc = builders.xc_of(cfg.xc)
    tracking.init(cfg, __name__, name=f"splat_{cfg.system}_M{M}_s{cfg.seed}",
                  extra={"nao": nao, "natm": len(mol.symbols)})
    print(f"{cfg.system}/{cfg.xc} M={M} ({M / nao:.1f}x nao={nao})  steps={cfg.steps} "
          f"refresh={cfg.refresh} grid={cfg.grid_level} seed={cfg.seed}", flush=True)

    model = init_model(mol, M, jr.PRNGKey(cfg.seed))
    ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=True,  # exp2 = production fast engine
                           df_lam=cfg.df_lam, aux_mult=cfg.aux_mult)
    t0 = time.time()

    _is_hybrid = builders.hf_coeff_of(xc) != 0.0 if hasattr(builders, "hf_coeff_of") else \
        float(getattr(xc, "hf_coeff", 0.0) or 0.0) != 0.0

    def _partial(step, m, _aux, _E):                     # honest E at step k (walltime safety)
        if step == 0 or _is_hybrid:
            return
        print(results.partial("splat", cfg.system, step=step, M=M,
                              E=float(evaluate(ks, m)), wall_s=round(time.time() - t0)),
              flush=True)

    from experiments.common import paths
    _ckdir = cfg.ckpt_dir or paths.artifact_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    base = os.path.join(_ckdir, f"ckpt_{cfg.system}_M{M}_{cfg.xc}_{cfg.steps}steps_s{cfg.seed}")
    res = train(ks, model, steps=cfg.steps, refresh_every=cfg.refresh,
                monitor=monitor(cfg.monitor), snapshot_cb=_partial, snapshot_every=cfg.partial_every,
                checkpoint=(ckpt_spec(base, every=int(cfg.ckpt_every), resume=bool(cfg.resume))
                            if int(cfg.ckpt_every) > 0 else None))
    model = res.model
    e_splat = float(evaluate(ks, model, res.state))

    prelim = results.result("splat", cfg.system, nao=nao, M=M, steps=cfg.steps, seed=cfg.seed,
                            xc=cfg.xc, E=e_splat, wall_s=round(time.time() - t0),
                            peak_gpu_mb=round(probes.peak_gpu_mb()))
    print(prelim.replace("RESULT ", "RESULT_PARTIAL ", 1), flush=True)

    extra = {}
    try:
        e_ck, bias = builders.grid_audit(mol, xc, model, cfg.grid_check, e_splat,
                                         state=res.state)
    except Exception as exc:                             # OOM, walltime, unsupported path
        print(f"# grid audit FAILED ({type(exc).__name__}: {exc}); "
              f"the energy above stands, uncertified", flush=True)
        e_ck = bias = None
    if e_ck is not None:                                 # level-`grid_check` re-evaluation
        extra[f"E_grid{cfg.grid_check}"] = e_ck
        extra["grid_bias_mha"] = bias
    if cfg.ckpt_dir:
        probes.atomic_save(os.path.join(cfg.ckpt_dir, f"ckpt_{cfg.system}_M{M}_s{cfg.seed}.eqx"),
                           (model, res.state.aux))
    print(results.result("splat", cfg.system, nao=nao, M=M, steps=cfg.steps, seed=cfg.seed, xc=cfg.xc,
                        E=e_splat, wall_s=round(time.time() - t0),
                        peak_gpu_mb=round(probes.peak_gpu_mb()), **extra), flush=True)


if __name__ == "__main__":
    main()
