"""Splat KS-DFT training for the size ladder, up to the proteins of Table 3.

Runs the standard pipeline (screened density fitting, the frozen pair-product auxiliary, the
chemistry-informed initialization, ``xc`` from the config) and logs every step to a JSONL, and to
Weights & Biases if enabled: total and per-term energies and the gradient norm, and at the monitor
cadence the DF-Coulomb energy, the condition number of the auxiliary metric, the largest
normalized splat overlap and the smallest eigenvalue of the orbital Gram matrix::

    uv run python -m experiments.exp7_size_ladder.train experiment=exp7_train system=ala_15 steps=1500

``resume=true`` continues from ``<tag>_ckpt.eqx`` and appends to the JSONL, so a run can span
several allocations. ``collapse=true`` saves the last good and the first non-finite state and stops.
``DFTAX_AOT_ANALYZE=1`` compiles the step without running it and prints its memory footprint.
"""
import math
import os
import time

import jax

# Multi-node: join the cluster before any other import, since `jax.distributed.initialize` refuses
# to run once the backend is up. Mirrors `gs_dft.ks.shard.init_distributed`, which cannot be imported
# this early. A single-task allocation keeps the single-process path.
_MULTINODE = int(os.environ.get("SLURM_NTASKS", "1")) > 1
if _MULTINODE:
    jax.distributed.initialize(
        coordinator_address=f"{os.environ['SLURM_LAUNCH_NODE_IPADDR']}:{os.environ.get('DFTAX_COORD_PORT', '29500')}",
        num_processes=int(os.environ["SLURM_NTASKS"]),
        process_id=int(os.environ["SLURM_PROCID"]),
        # one process per GPU: all of the node's devices are visible, this process owns SLURM_LOCALID
        local_device_ids=[int(os.environ["SLURM_LOCALID"])] if "SLURM_LOCALID" in os.environ and len(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) > 1 else list(range(len(
            [i for i in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if i != ""]))),
        initialization_timeout=1800)
    # Backend initialization is a collective with a timeout, so do it before any host work of
    # uneven duration (geometry, basis lookup).
    jax.config.update("jax_cpu_get_local_topology_timeout_minutes", 10)
    jax.config.update("jax_cpu_get_global_topology_timeout_minutes", 30)   # default 5: the one that fired
    jax.devices()

import hydra
from omegaconf import DictConfig, OmegaConf
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from experiments.common import paths, systems, results, probes, builders, tracking
from experiments.common.builders import xc_of, aux_scales
from gs_dft import nao as nao_of
from gs_dft import init_model, chem
from gs_dft.ks.train import (train, monitor, collapse, native_grid, splat_adam,
                                       ckpt as ckpt_spec)
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.terms import df, pairlist
from gs_dft.ks import shard as _shd





@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = cfg.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    mol = systems.molecule(cfg)
    nao = int(nao_of(mol))
    M = max(1, int(cfg.m) if cfg.m is not None else int(cfg.m_mult * nao))
    # The m_mult form is kept when m_mult sized the run: `ckpt` derives from `tag`, so changing
    # the name unconditionally would orphan existing checkpoints and restart from step 0.
    _msize = f"M{M}" if cfg.m is not None else f"M{cfg.m_mult:g}x"
    tag = cfg.tag or f"{cfg.system}_{_msize}_g{cfg.grid_level}_s{cfg.seed}"
    model = init_model(mol, M, jr.PRNGKey(cfg.seed), init=chem())
    print(f"# exp8 {cfg.system}: {len(mol.symbols)} atoms  nao={nao}  M={M} ({"absolute m" if cfg.m is not None else f"{cfg.m_mult}x"})  ndev={jax.local_device_count()}  grid_chunk={int(cfg.get("grid_chunk", 4096))}  "
          f"xc={cfg.xc}  grid={cfg.grid_level}  steps={cfg.steps}  df_lam={cfg.df_lam:g}  tag={tag}", flush=True)

    # Checkpoints and collapse dumps are ARTIFACTS, not the record: they go to DFTAX_CKPT_DIR (scratch
    # on the cluster, set by submit.sh), never into the quota-limited home checkout. Rows and logs stay in out_dir.
    ckpt_dir = paths.artifact_dir(out_dir)
    ckpt = cfg.ckpt or os.path.join(ckpt_dir, f"{tag}_ckpt")
    jsonl_path = os.path.join(out_dir, f"{tag}.jsonl")
    rank0 = jax.process_index() == 0            # multi-node: only rank 0 writes rows, wandb, RESULT
    if not rank0:
        cfg.wandb_mode = "disabled"
    jfh = results.jsonl_writer(jsonl_path, resume=cfg.resume) if rank0 else None
    wb = tracking.init(cfg, __name__, name=tag, dir=out_dir,
                       extra={"natm": len(mol.symbols), "nao": nao, "M": M})

    t0 = [time.time()]

    def step_cb(m):
        m = dict(m)
        m["wall_s"] = time.time() - t0[0]
        m["E_nn_implied"] = m["E_total"] - (m["E_kinetic"] + m["E_hartree"] + m["E_xc"] + m["E_external"])
        if rank0:
            jfh.write(m)
            wb.log(m, step=m["step"])

    collapse_ckpt = os.path.join(ckpt_dir, f"{tag}_collapse") if cfg.collapse else None
    # `jax.devices()` is the GLOBAL list under multi-controller, so ONE mesh spans every node.
    # Gate on the global count: a 2-node x 1-GPU run has local_device_count() == 1 and still needs
    # a mesh.
    mesh = None
    if jax.device_count() > 1:
        mesh = _shd.make_mesh()
    _scales = aux_scales(cfg.aux_mult)
    print(f"# aux: mult={cfg.aux_mult} scales={_scales}  naux = {len(_scales)}xM = {len(_scales) * M}", flush=True)
    xc = xc_of(cfg.xc)
    grid = native_grid(mol, cfg.grid_level, chunk=int(cfg.get('grid_chunk', 4096)))
    ks = SplatKS(mol, xc, grid=grid, coulomb=df(lam=cfg.df_lam, scales=_scales, chunk=int(cfg.get('gamma_chunk', 512))),
                 screen=pairlist(eps=1e-7, unique=bool(cfg.get('screen_unique', False)),
                                 skip_pad=bool(cfg.get('skip_pad', False))), mesh=mesh)
    _resuming = (bool(cfg.resume) and os.path.exists(ckpt + ".eqx")
                 and os.path.exists(ckpt + ".step"))
    if cfg.init_c != "random" and _resuming:
        print(f"# init_c={cfg.init_c} SKIPPED: resuming from {ckpt}.eqx, which overwrites C", flush=True)
    if cfg.init_c != "random" and not _resuming:
        import numpy as _np
        from experiments.exp8_data_free_init.minao_init import minao_C0, minao_C0_pcg
        from experiments.exp8_data_free_init.local_init import local_C0
        _fn = {"minao": minao_C0, "minao_pcg": minao_C0_pcg, "local": local_C0}[cfg.init_c]
        _t0 = time.time()
        _shape = (int(model.C.shape[0]), int(model.C.shape[1]))
        _cdir = os.environ.get("DFTAX_CKPT_DIR") or ckpt_dir
        _cpath = os.path.join(_cdir, f"initC_{cfg.system}_{cfg.basis}_{cfg.init_c}_"
                                     f"{_shape[0]}x{_shape[1]}.npy")
        _C = None
        if os.path.exists(_cpath):
            _C = _np.load(_cpath)
            if _C.shape != _shape:
                print(f"# init_c cache {_cpath} has shape {_C.shape} != {_shape}; rebuilding",
                      flush=True)
                _C = None
            else:
                print(f"# init_c={cfg.init_c}: loaded cached C from {_cpath}", flush=True)
        if _C is None:
            if _MULTINODE and not rank0:
                _C = _np.zeros(_shape, dtype=_np.float64)    # placeholder: shape/dtype must match
            else:
                _C = _np.asarray(_fn(model.basis, list(mol.symbols), _np.asarray(mol.coords),
                                     _shape[1], _np.asarray(grid.coords),
                                     _np.asarray(grid.weights)),
                                 dtype=_np.float64)          # host grid: ks's is sharded multi-node
                if rank0:
                    _tmp = _cpath + f".tmp{os.getpid()}"
                    with open(_tmp, "wb") as _fh:
                        _np.save(_fh, _C)
                    os.replace(_tmp, _cpath)
                    print(f"# init_c={cfg.init_c}: cached C to {_cpath}", flush=True)
            if _MULTINODE:
                from jax.experimental.multihost_utils import broadcast_one_to_all
                _C = _np.asarray(broadcast_one_to_all(_C))
        model = eqx.tree_at(lambda m: m.C, model, jnp.asarray(_C))
        print(f"# init_c={cfg.init_c}: C built in {time.time() - _t0:.1f}s"
              f"{' (rank 0, broadcast)' if _MULTINODE else ''}", flush=True)

    # A constant-lr `polish_lr` continues a finished run without restarting the warmup and cosine.
    optimizer = None
    if float(cfg.polish_lr) > 0:
        import optax
        optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                                optax.adam(float(cfg.polish_lr)))
        print(f"# polish: constant-lr adam lr={float(cfg.polish_lr):g} (no warmup/cosine)", flush=True)
    elif int(cfg.warmup) > 0 or float(cfg.peak_lr) != 1e-2:
        optimizer = splat_adam(float(cfg.peak_lr), int(cfg.steps),
                               warmup=(int(cfg.warmup) if int(cfg.warmup) > 0 else None))
        print(f"# splat_adam peak_lr={float(cfg.peak_lr):g} warmup={cfg.warmup or 'default'}", flush=True)
    res = train(ks, model, steps=cfg.steps, refresh_every=cfg.refresh, optimizer=optimizer,
                remat=bool(cfg.get('remat', False)),
                monitor=monitor(cfg.monitor, cond_every=cfg.cond_every),
                step_cb=step_cb,
                checkpoint=ckpt_spec(ckpt, every=cfg.ckpt_every, resume=cfg.resume),
                guard=(collapse(collapse_ckpt, nelec_rtol=float(cfg.nelec_rtol)) if collapse_ckpt else None))

    if jfh is not None:
        jfh.close()
    if not rank0:
        wb.finish()
        return
    E_final = float(res.energy[-1]) if len(res.energy) else float("nan")

    # A positive total energy is a collapse whatever the detector says: no bound electronic
    # structure has E > 0.
    collapsed = bool(res.collapsed) or not math.isfinite(E_final) or E_final > 0.0
    # Emit the row before the grid re-evaluation, which is the most expensive step at these sizes,
    # so a walltime kill inside it still leaves a result. The re-evaluated row supersedes this one.
    prelim = results.result("train", cfg.system, M=M, tag=tag, steps_run=res.n_steps, E=E_final,
                            collapsed=collapsed, grid_level=cfg.grid_level,
                            peak_gpu_mb=round(probes.peak_gpu_mb()))
    print(prelim.replace("RESULT ", "RESULT_PARTIAL ", 1), flush=True)

    extra = {}
    if int(cfg.grid_check) and math.isfinite(E_final):
        try:
            e_ck, bias = builders.grid_audit(mol, xc, res.model, int(cfg.grid_check), E_final)
        except Exception as exc:
            print(f"# grid audit FAILED ({type(exc).__name__}: {exc}); the energy above stands, "
                  f"uncertified", flush=True)
            e_ck = bias = None
        if e_ck is not None:
            extra[f"E_grid{int(cfg.grid_check)}"] = e_ck
            extra["grid_bias_mha"] = bias
            print(f"# grid audit @ level {int(cfg.grid_check)}: E={e_ck:.8f}  "
                  f"bias={bias:+.4f} mHa", flush=True)

    print(f"# ckpt={ckpt}.eqx (continue with resume=true)  jsonl={jsonl_path}", flush=True)
    print(results.result("train", cfg.system, M=M, tag=tag, steps_run=res.n_steps, E=E_final,
                        collapsed=collapsed, grid_level=cfg.grid_level,
                        peak_gpu_mb=round(probes.peak_gpu_mb()), **extra), flush=True)
    wb.summary({"E_final": E_final})
    wb.finish()


if __name__ == "__main__":
    main()
