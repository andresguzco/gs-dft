"""Hellmann-Feynman nuclear forces from a converged size-ladder checkpoint.

The splats do not move with the nuclei, so the coordinates enter the energy only through the
electron-nucleus attraction and the nuclear repulsion, and the force is one backward pass with no
Pulay term (``gs_dft/ks/forces.py``)::

    uv run python -m experiments.exp7_size_ladder.forces experiment=exp7_train system=PJM49 tag=PJM49 polish_lr=1e-4

``tag`` names the checkpoint. The optimizer settings (``polish_lr``, ``warmup``, ``steps``) must
match the run that wrote it, because the optimizer state is part of the saved tree. Writes
``forces_<tag>.npz`` (forces, geometry, atomic numbers) and one ``RESULT kind=forces`` row.
"""
import os
import time

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from omegaconf import DictConfig

from gs_dft import chem, init_model, nao as nao_of
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.forces import splat_forces
from gs_dft.ks.terms import df, pairlist
from gs_dft.ks.train import (_load_ckpt, _partition, _prepare_mesh, _refresh_state,
                                      native_grid, splat_adam)
from experiments.common import paths, builders, results, systems, tracking
from experiments.common.builders import aux_scales, xc_of

HARTREE_BOHR = "Ha/Bohr"


def _optimizer(cfg):
    """The optimizer whose state shape the checkpoint was written with.

    Not a preference: it is half the serialized tree. A cosine-scheduled Adam and a constant-lr one
    differ in their state, so reading a checkpoint under the wrong one raises on the tree structure.
    """
    if float(cfg.polish_lr) > 0:
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(float(cfg.polish_lr)))
    if int(cfg.warmup) > 0:
        return splat_adam(1e-2, int(cfg.steps), warmup=int(cfg.warmup))
    return splat_adam(1e-2, int(cfg.steps))


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = cfg.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    tag = cfg.tag or f"{cfg.system}_M{cfg.m_mult:g}x_g{cfg.grid_level}_s{cfg.seed}"
    tracking.init(cfg, __name__, name=f"forces_{tag}")

    mol = systems.molecule(cfg)
    nao = int(nao_of(mol))
    M = max(1, int(cfg.m) if cfg.m is not None else int(cfg.m_mult * nao))
    print(f"# forces {cfg.system}: {len(mol.symbols)} atoms  nao={nao}  M={M}  tag={tag}", flush=True)

    model = init_model(mol, M, jr.PRNGKey(cfg.seed), init=chem())
    xc = xc_of(cfg.xc)
    _scales = aux_scales(cfg.aux_mult)
    # Shard exactly as the training run did. The per-device footprint is what fits: 9LQN2 peaked at
    # 35 GB on EACH of four devices, so gathering it onto one would not run.
    mesh = None
    if jax.local_device_count() > 1:
        from gs_dft.ks.shard import make_mesh
        mesh = make_mesh()
    ks = SplatKS(mol, xc,
                 grid=native_grid(mol, cfg.grid_level, chunk=int(cfg.get("grid_chunk", 4096))),
                 coulomb=df(lam=cfg.df_lam, scales=_scales), screen=pairlist(eps=1e-7), mesh=mesh)

    ckpt = cfg.ckpt or paths.artifact(out_dir, f"{tag}_ckpt")
    m_tr, m_st = _partition(model)
    opt_state = _optimizer(cfg).init(m_tr)
    loaded = _load_ckpt(ckpt, m_tr, opt_state)
    if loaded is None:
        raise SystemExit(f"forces: no checkpoint at {ckpt}.eqx")
    m_tr, _opt_state, step = loaded
    model = eqx.combine(m_tr, m_st)
    print(f"# loaded {ckpt}.eqx at step {step}", flush=True)

    ks, model, _G, multinode, _rank0 = _prepare_mesh(ks, model)
    t0 = time.time()
    state = _refresh_state(ks, model.basis, None, multinode)
    F = np.asarray(splat_forces(ks, model, state))
    wall = time.time() - t0

    if bool(cfg.get("theta_grad", True)):
        g = eqx.filter_grad(lambda m: ks(m, state)[0])(model)
        g_theta = jnp.concatenate([g.basis.centers.ravel(), g.basis.log_scale.ravel(),
                                   g.basis.quat.ravel()])
        g_rms = float(jnp.sqrt(jnp.mean(g_theta ** 2)))
        g_max = float(jnp.max(jnp.abs(g_theta)))
    else:
        g_rms = g_max = float("nan")
        print("# theta_grad=false: stationarity residual NOT measured here — take `grad_norm` "
              "from the training jsonl at this checkpoint's step", flush=True)

    coords = np.asarray(mol.atom_coords())
    charges = np.asarray(mol.atom_charges(), dtype=float)
    npz = os.path.join(out_dir, f"forces_{tag}.npz")
    np.savez(npz, F=F, coords=coords, charges=charges, step=step, M=M, nao=nao,
             g_theta_rms=g_rms, g_theta_max=g_max,
             units=HARTREE_BOHR, system=str(cfg.system), xc=str(cfg.xc))

    from experiments.common import probes
    peak = probes.peak_gpu_mb()
    fmax = float(np.abs(F).max())
    frms = float(np.sqrt(np.mean(F ** 2)))
    # The net force on an isolated molecule is zero by translational invariance, so ||sum_A F_A||
    # is a self-check on the whole evaluation rather than a physical quantity.
    fnet = float(np.linalg.norm(F.sum(axis=0)))
    print(f"# |F|max={fmax:.6e}  rms={frms:.6e}  |net|={fnet:.3e} {HARTREE_BOHR}  "
          f"in {wall:.1f}s on {jax.local_device_count()} device(s), peak {peak:.0f} MB", flush=True)
    print(f"# dE/dtheta rms={g_rms:.3e} max={g_max:.3e}  "
          f"|net|/natm={fnet / len(mol.symbols):.3e} vs F_rms={frms:.3e}", flush=True)
    print(results.result("forces", cfg.system, M=M, nao=nao, tag=tag, step=int(step),
                         natm=len(mol.symbols), F_max=fmax, F_rms=frms, F_net=fnet,
                         g_theta_rms=g_rms, g_theta_max=g_max,
                         ndev=jax.local_device_count(), peak_gpu_mb=round(peak),
                         units=HARTREE_BOHR, wall_s=round(wall, 1)), flush=True)


if __name__ == "__main__":
    main()
