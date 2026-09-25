"""End-to-end time and memory for one configuration, used for the GS-DFT water check of Appendix D.

    engine=splat  m=<M>        train(), wall time, peak GPU memory, final energy and the level-5 check
    engine=gto    basis=<b>    converged DF-RKS/PBE on dftax's Gaussian KS-DFT, on the same device

    uv run python -m experiments.exp6_cost_benchmark.bench experiment=exp6_bench system=water engine=splat m=24
"""
import time

import hydra
from omegaconf import DictConfig
import jax.random as jr

from experiments.common import systems, builders, reference, results, probes, tracking
from gs_dft import init_model
from gs_dft.ks.train import train, evaluate


def run_splat(cfg):
    mol = systems.molecule(system=cfg.system, basis="cc-pvdz")   # cc-pVDZ = coords/nelec/N_aux source
    xc = builders.xc_of(cfg.xc)
    e_dz = reference.gto_reference(mol, xc, grid_level=cfg.grid_level, mode="df").e   # DZ anchor
    M = int(cfg.m)
    screened = builders.auto_screen(M, force=(cfg.system == "alanine_dipeptide"))
    model = init_model(mol, M, jr.PRNGKey(cfg.seed))
    ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=screened,
                           df_lam=cfg.df_lam, aux_mult=cfg.aux_mult)

    t0 = time.time()
    state = {"t_cross": None}

    def _snap(step, m, _aux, _E):
        if step == 0 or state["t_cross"] is not None:
            return
        if float(evaluate(ks, m)) < e_dz:                # reached cc-pVDZ quality
            state["t_cross"] = time.time() - t0
            print(f"  crossed E_DZ at step {step} ({state['t_cross']:.0f}s)", flush=True)

    res = train(ks, model, steps=cfg.steps, monitor=None, snapshot_cb=_snap, snapshot_every=1000)
    e = float(evaluate(ks, res.model))
    gf = builders.grid_fields(mol, xc, res.model, cfg.grid_check, e)
    e5, bias = gf.get(f'E_grid{cfg.grid_check}'), gf.get('grid_bias_mha')
    tc = round(state["t_cross"]) if state["t_cross"] is not None else -1
    print(results.result("bench", cfg.system, engine="splat", M=M, steps=cfg.steps, seed=cfg.seed,
                        E=e, grid_bias_mha=bias, E_dz=e_dz, t_cross_s=tc,
                        wall_s=round(time.time() - t0), peak_gpu_mb=round(probes.peak_gpu_mb())),
          flush=True)


def run_gto(cfg):
    """dftax DF-RKS/PBE single-point — the equal-engine GTO cost baseline (same GPU/XLA)."""
    import resource
    mol = systems.molecule(cfg)
    ref = reference.gto_reference(mol, builders.xc_of(cfg.xc), grid_level=cfg.grid_level, mode="df")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3   # KB→MB
    print(results.result("bench", cfg.system, engine="gto", basis=cfg.basis, nao=ref.nao, E=ref.e,
                        conv=ref.converged, n_iter=ref.n_iter, wall_s=round(ref.wall_s, 1),
                        peak_gpu_mb=round(probes.peak_gpu_mb()), max_rss_mb=round(rss)), flush=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"bench_{cfg.system}_{cfg.engine}")
    if cfg.engine == "splat":
        if cfg.m is None:
            raise ValueError("engine=splat needs a splat count: m=<M>")
        run_splat(cfg)
    elif cfg.engine == "gto":
        run_gto(cfg)
    else:
        raise SystemExit(f"unknown engine {cfg.engine!r} (use: splat | gto)")


if __name__ == "__main__":
    main()
