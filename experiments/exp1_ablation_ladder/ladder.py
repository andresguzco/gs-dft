"""Run one rung of the representation ladder (L0 to L4) of Figure 2(a) and print its RESULT line.

Every rung trains the same functional on the same grid with the same optimizer; only the
parametrization changes, and for L0 whether the orbital coefficients are free.

    uv run python -m experiments.exp1_ablation_ladder.ladder experiment=exp1_ladder system=water rung=L4
"""
import time

import hydra
import jax.numpy as jnp
import jax.random as jr
import optax
from omegaconf import DictConfig

from gs_dft import nao as nao_of
from gs_dft.ks.energy import SplatKS, SplatModel
from gs_dft.ks.terms import exact
from gs_dft.ks.train import evaluate, native_grid, splat_adam, train
from gs_dft.basis.spectral import Splat
from experiments.common import builders, results, systems, tracking
from experiments.exp1_ablation_ladder.charts import (EllipsoidalSplat, FrostSplat, RUNGS, describe)


def build_basis(kind, mol, M, key):
    """A rung's basis at M functions, DERIVED from the production initializer."""
    from gs_dft import init_model
    basis = init_model(mol, M, key).basis                     # the production spectral cloud
    if kind == "spectral":
        return basis
    if kind == "ellipsoidal":                                  # drop the orientation, keep 3 scales
        return EllipsoidalSplat(log_scale=basis.log_scale, centers=basis.centers)
    if kind == "frost":                                        # + isotropize: one width per function
        return FrostSplat(log_width=jnp.mean(basis.log_scale, axis=-1), centers=basis.centers)
    raise ValueError(f"unknown chart {kind!r}")


def frozen_C_optimizer(opt, model):
    """``opt`` on the basis only, with the MO coefficients held fixed.

    L0 is Frost's model as published: each occupied orbital IS one Gaussian, so there is no
    coefficient matrix to optimize. Freezing C at the identity is how that is expressed inside a
    machine whose objective is defined over (basis, C) — NOT a convenience, but the rung's defining
    property, and the reason L1 (which frees exactly this) isolates what coefficients are worth.
    """
    where = optax.multi_transform(
        {"train": opt, "freeze": optax.set_to_zero()},
        SplatModel(basis="train", C="freeze", occupations="freeze"))
    return where


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    rung = str(cfg.rung)
    if rung not in RUNGS and rung.upper() in RUNGS:
        rung = rung.upper()
    chart, free_C, msize = RUNGS[rung]
    tracking.init(cfg, __name__, name=f"{rung}_{cfg.system}_s{cfg.seed}")

    mol = systems.molecule(cfg)
    n_occ = int(mol.nelectron) // 2
    M = n_occ if msize == "n_occ" else (int(cfg.m) if cfg.m is not None
                                        else int(cfg.m_mult * nao_of(mol)))
    print(f"# exp1 {rung}: {describe(rung)}", flush=True)
    print(f"#   system={cfg.system} chart={chart} free_C={free_C} M={M} "
          f"(n_occ={n_occ}, nao={int(nao_of(mol))}) steps={cfg.steps} seed={cfg.seed}", flush=True)

    key = jr.PRNGKey(int(cfg.seed))
    basis = build_basis(chart, mol, M, key)
    # C = identity on the occupied block: at L0 this IS the model (orbital k = Gaussian k); at L1+
    # it is merely the starting point and the optimizer is free to move it.
    C = jnp.eye(M, n_occ)
    model = SplatModel(basis=basis, C=C, occupations=jnp.full((n_occ,), 2.0))

    ks = SplatKS(mol, builders.xc_of(cfg.xc),
                 grid=native_grid(mol, int(cfg.grid_level), chunk=None),
                 coulomb=exact())
    opt = splat_adam(steps=int(cfg.steps))
    if not free_C:
        opt = frozen_C_optimizer(opt, model)

    t0 = time.time()
    trace_n = max(int(cfg.steps) // 400, 1)

    def _trace(step, _model, _aux, E):
        print(f"TRACE rung={rung} step={int(step)} t={time.time() - t0:.3f} "
              f"E={float(E):.8f}", flush=True)

    res = train(ks, model, steps=int(cfg.steps), refresh_every=int(cfg.refresh), monitor=None,
                optimizer=opt, snapshot_every=trace_n, snapshot_cb=_trace)
    wall = time.time() - t0
    E = float(evaluate(ks, res.model, res.state))

    row = dict(kind="splat", rung=rung, chart=chart, free_C=bool(free_C), M=M, n_occ=n_occ,
               nao=int(nao_of(mol)), steps=int(cfg.steps), seed=int(cfg.seed), xc=cfg.xc,
               E=E, wall_s=round(wall, 1), ms_step=round(1e3 * wall / max(int(cfg.steps), 1), 2))
    row.update(builders.grid_fields(mol, builders.xc_of(cfg.xc), res.model,
                                    int(cfg.grid_check), E, res.state)
               if int(cfg.grid_check) else {})
    print(results.result(row.pop("kind"), cfg.system, **row), flush=True)


if __name__ == "__main__":
    main()
