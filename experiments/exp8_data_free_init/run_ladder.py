"""The initialization ablation of Appendix E: schemes S0 to S4 on the production DF path.

Each scheme adds one ingredient to the start; everything else is the same as ``splat_run.py``:

  S0  random, atom-centered
  S1  + element exponent ranges from def2-SVP     (use_ref_exponents)
  S2  + bond-centered splats                      (bond_fraction > 0)
  S3  + bond-aligned anisotropy                   (bond_anisotropy)
  S4  + splats allocated in proportion to Z       (alloc='electron')

Every snapshot prints a ``RESULT_PARTIAL`` line with the exact streamed energy, which
``plot_curves.py`` reads::

    uv run python -m experiments.exp8_data_free_init.run_ladder experiment=exp8_run_ladder system=water 'schemes=[S0,S4]' 'seeds=[0]'
"""
import os
import time

import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random as jr

from experiments.common import systems, builders, results, probes, tracking
from gs_dft import init_model, chem, nao as nao_of
from gs_dft.ks.train import train, monitor, evaluate
from gs_dft.coulomb.ri import aux_metric

CHART = "spectral"        # RESULT-line tag only (the sole chart since 366502c). A LITERAL, not a
                          # knob: plot_curves.py keys on `chart=`.

# Scheme → chem-init flag set (S0 = legacy random init, handled separately).
SCHEME_FLAGS = {
    "S1": dict(use_ref_exponents=True,  bond_fraction=0.0,  bond_anisotropy=False, alloc="uniform"),
    "S2": dict(use_ref_exponents=True,  bond_fraction=0.25, bond_anisotropy=False, alloc="uniform"),
    "S3": dict(use_ref_exponents=True,  bond_fraction=0.25, bond_anisotropy=True,  alloc="uniform"),
    "S4": dict(use_ref_exponents=True,  bond_fraction=0.25, bond_anisotropy=True,  alloc="electron"),
}


def build_model(scheme, mol, M, key):
    """SplatModel for one ladder rung. S0 is the random init; S1–S4 route through the chem init."""
    if scheme == "S0":
        return init_model(mol, M, key)
    return init_model(mol, M, key, init=chem(**SCHEME_FLAGS[scheme]))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"ladder_{cfg.system}")
    mol = systems.molecule(cfg)
    nao = int(nao_of(mol))
    M = int(cfg.m) if cfg.m is not None else int(round(cfg.m_mult * nao))
    xc = builders.xc_of(cfg.xc)
    ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=True,
                           df_lam=cfg.df_lam, aux_mult=cfg.aux_mult)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                            f"ladder_{cfg.system}{cfg.log_suffix}.log")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logf = open(out_path, "a")

    def emit(line):
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    emit(f"### exp7 ladder system={cfg.system} nao={nao} M={M} (={cfg.m_mult}x) "
         f"steps={cfg.steps} grid={cfg.grid_level}")
    for scheme in cfg.schemes:
        for seed in cfg.seeds:                       # cfg.seeds (plural), swept in-process; the
            # shared singular cfg.seed is inherited but unused here.
            common = dict(chart=CHART, scheme=scheme, seed=seed, M=M)
            model = build_model(scheme, mol, M, jr.PRNGKey(seed))
            t0 = time.time()

            def _partial(step, m, _aux, _E, _c=common, _t=t0):
                if step == 0:
                    return
                emit(results.partial("splat", cfg.system, step=step, E=float(evaluate(ks, m)),
                                    wall_s=round(time.time() - _t), **_c))

            res = train(ks, model, steps=cfg.steps, refresh_every=cfg.refresh,
                        monitor=monitor(cfg.monitor), snapshot_cb=_partial,
                        snapshot_every=cfg.partial_every)
            e_splat = float(evaluate(ks, res.model))
            try:
                condV = float(jnp.linalg.cond(aux_metric(res.state.aux)))
            except Exception:
                condV = -1.0
            _gf = builders.grid_fields(mol, xc, res.model, cfg.grid_check, e_splat)
            emit(results.result("splat", cfg.system, steps=cfg.steps, E=e_splat,
                              wall_s=round(time.time() - t0), cond_V=condV,
                              peak_gpu_mb=round(probes.peak_gpu_mb()), **_gf, **common))
    logf.close()


if __name__ == "__main__":
    main()
