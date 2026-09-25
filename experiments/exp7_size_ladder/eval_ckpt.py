"""Evaluate a training checkpoint: the energy and gradient norm of the saved state, with no
optimization, built the same way the trainer builds it.

    uv run python -m experiments.exp7_size_ladder.eval_ckpt experiment=exp7_train system=LG5K9 \
        basis=cc-pvtz tag=LG5K9 ckpt=<checkpoint path>
"""
import os
import time

import equinox as eqx
import hydra
import jax
import jax.random as jr
import optax
from omegaconf import DictConfig

from gs_dft import chem, init_model, nao as nao_of
from gs_dft import screening as _scr
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.terms import df, pairlist
from gs_dft.ks.train import (_load_ckpt, _partition, _prepare_mesh, _refresh_state,
                                      native_grid, splat_adam)
from experiments.common import paths, probes, systems
from experiments.common.builders import aux_scales, xc_of


def _optimizer(cfg):
    """Half the serialized tree is the optimizer state, so this must match what wrote it."""
    if float(cfg.polish_lr) > 0:
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(float(cfg.polish_lr)))
    if int(cfg.warmup) > 0:
        return splat_adam(1e-2, int(cfg.steps), warmup=int(cfg.warmup))
    return splat_adam(1e-2, int(cfg.steps))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = cfg.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    tag = cfg.tag or f"{cfg.system}_M{cfg.m_mult:g}x_g{cfg.grid_level}_s{cfg.seed}"
    mol = systems.molecule(cfg)
    nao = int(nao_of(mol))
    M = max(1, int(cfg.m) if cfg.m is not None else int(cfg.m_mult * nao))
    model = init_model(mol, M, jr.PRNGKey(cfg.seed), init=chem())

    mesh = None
    if jax.local_device_count() > 1:
        from gs_dft.ks.shard import make_mesh
        mesh = make_mesh()
    ks = SplatKS(mol, xc_of(cfg.xc),
                 grid=native_grid(mol, cfg.grid_level, chunk=int(cfg.get("grid_chunk", 4096))),
                 coulomb=df(lam=cfg.df_lam, scales=aux_scales(cfg.aux_mult),
                            chunk=int(cfg.get("gamma_chunk", 512))),
                 screen=pairlist(eps=1e-7, unique=bool(cfg.get("screen_unique", False)),
                                 skip_pad=bool(cfg.get("skip_pad", False))), mesh=mesh)

    ckpt = cfg.ckpt or paths.artifact(out_dir, f"{tag}_ckpt")
    m_tr, m_st = _partition(model)
    loaded = _load_ckpt(ckpt, m_tr, _optimizer(cfg).init(m_tr))
    if loaded is None:
        raise SystemExit(f"eval_ckpt: no checkpoint at {ckpt}.eqx")
    m_tr, _opt, step = loaded
    model = eqx.combine(m_tr, m_st)
    print(f"# eval_ckpt {cfg.system} tag={tag} nao={nao} M={M} ndev={jax.device_count()} "
          f"loaded {ckpt}.eqx at step {step}", flush=True)

    ks, model, G, multinode, rank0 = _prepare_mesh(ks, model)
    t0 = time.time()
    n0 = int(_scr.neighbor_pairs(model.basis, eps=ks.screen.eps,
                                 unique=bool(cfg.get("screen_unique", False)))[0].shape[0])
    pad_to = -(-int(ks.screen.pad * n0) // G) * G
    state = _refresh_state(ks, model.basis, pad_to, multinode)
    m_tr, m_st = _partition(model)

    @eqx.filter_jit
    def f(ks_, tr, st):
        (e, aux), g = jax.value_and_grad(lambda t: ks_(eqx.combine(t, m_st), st), has_aux=True)(tr)
        return e, optax.global_norm(g), aux

    E, gnorm, aux = f(ks, m_tr, state)
    E.block_until_ready()
    if rank0:
        print(f"# step={step}  E={float(E):.10f}  |g|={float(gnorm):.6e}  "
              f"({time.time() - t0:.1f}s, peak {probes.peak_gpu_mb():.0f} MB)", flush=True)
        ref = cfg.get("e_ref", None)
        if ref is not None:
            d = float(E) - float(ref)
            print(f"# reference {float(ref):.10f}  difference {d:+.4f} Ha "
                  f"({1000 * d / max(len(mol.symbols), 1):+.3f} mHa/atom)", flush=True)
        for k, v in (aux.items() if hasattr(aux, "items") else []):
            try:
                print(f"#   {k:12s} {float(v):18.8f}", flush=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
