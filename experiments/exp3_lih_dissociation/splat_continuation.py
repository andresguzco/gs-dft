"""Warm-started splat chain along the LiH or LiF dissociation curve at fixed M.

The first R starts cold. Each later R starts from the previous converged cloud, with splats
closer to the moving partner atom than to Li shifted with it and the coefficients kept, and
re-optimizes at a reduced learning rate, so the cloud follows the density from covalent to ionic.
The whole R sequence (``systems.R_GRID[system]``) is one run. Prints ``kind=splat phase=warm``
rows; the plotters take the lower of the cold and warm energy at each (M, R).

    uv run python -m experiments.exp3_lih_dissociation.splat_continuation experiment=exp3_continuation system=lih m=95
"""
import time

import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from experiments.common import systems, builders, results, tracking
from gs_dft import init_model
from gs_dft.ks.train import train, evaluate, splat_adam


def _follow_atoms(centers, coords_old, coords_new):
    """Translate centers closer to the displaced partner (atom 1: H for LiH, F for LiF) than to Li
    (atom 0) by the partner's displacement — the 'follow-your-atom' warm-start shift."""
    d_li = jnp.linalg.norm(centers - coords_old[0], axis=-1)
    d_p = jnp.linalg.norm(centers - coords_old[1], axis=-1)
    shift = (coords_new[1] - coords_old[1])[None, :]
    return jnp.where((d_p < d_li)[:, None], centers + shift, centers)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"cont_{cfg.system}_s{cfg.seed}")
    if cfg.m is None:                                     # was the mandatory argv[1]
        raise ValueError("splat_continuation needs an explicit splat count: m=<M> "
                         "(the payload runs one chain per M, largest-first)")
    M = int(cfg.m)
    xc = builders.xc_of(cfg.xc)
    screened = builders.auto_screen(M)
    model = coords_prev = None
    r_grid = cfg.get("r_grid", None)
    Rs = [float(r) for r in r_grid] if r_grid else systems.R_GRID[cfg.system]
    print(f"chain over R = {Rs}", flush=True)
    for R in Rs:
        mol = systems.molecule(cfg, r=R)
        coords = jnp.array(mol.atom_coords())
        t0 = time.time()
        if model is None:                                 # cold start at the first R
            model = init_model(mol, M, jr.PRNGKey(cfg.seed))
            steps, lr = cfg.cold_steps, 1e-2
        else:                                             # warm start: follow-your-atom shift (splats
            model = eqx.tree_at(lambda m: m.basis.centers, model,   # only; the product aux self-rebuilds)
                                _follow_atoms(model.basis.centers, coords_prev, coords))
            steps, lr = cfg.warm_steps, cfg.warm_lr
        ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=screened,
                               df_lam=cfg.df_lam, aux_mult=cfg.aux_mult)
        res = train(ks, model, steps=steps, optimizer=splat_adam(lr, steps), monitor=None)
        model = res.model
        e = float(evaluate(ks, model))
        e5, bias = builders.grid_audit(mol, xc, model, cfg.grid_check, e)
        extra = {} if e5 is None else {f"E_grid{int(cfg.grid_check)}": e5, "grid_bias_mha": bias}
        print(results.result("splat", cfg.system, phase="warm", R=round(float(R), 3), M=M,
                            seed=cfg.seed, E=e, steps=steps,
                            wall_s=round(time.time() - t0), **extra), flush=True)
        coords_prev = coords


if __name__ == "__main__":
    main()
