"""A splat M-sweep at a chosen Coulomb path and functional, measured against the Gaussian reference
at the same basis and functional. Also computes the ionization potentials of Table 2 (``koopmans=true``).

    coulomb=exact xc=pbe            streaming exact Coulomb
    coulomb=df    xc=pbe            density fitting
    coulomb=df    xc=pbe0|b3lyp     hybrids, through RI-K

    uv run python -m experiments.exp5_basis_accuracy.msweep experiment=exp5_msweep system=co2 m=126 coulomb=exact xc=pbe
"""
import os
import time

import hydra
from omegaconf import DictConfig
import jax.random as jr

from experiments.common import systems, builders, reference, results, probes, tracking
from gs_dft import init_model, nao as nao_of
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.terms import pairlist
from gs_dft.ks.train import train, monitor, evaluate, native_grid, ckpt as ckpt_spec


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.m is None:
        raise ValueError("msweep needs an explicit splat count: m=<M> (the payload's M-ladder)")
    M = int(cfg.m)
    mol = systems.molecule(cfg)
    nao = int(nao_of(mol))
    xc = builders.xc_of(cfg.xc)
    ref = None if bool(cfg.get("skip_ref", False)) else \
        reference.gto_reference(mol, xc, grid_level=cfg.grid_level, mode="df")   # E_ref @ (basis, xc)

    tracking.init(cfg, __name__, extra={"nao": nao, **({} if ref is None else {"E_ref": ref.e})},
                  name=f"{cfg.coulomb}_{cfg.xc}_{cfg.system}_M{M}_s{cfg.seed}")
    coulomb = builders.coulomb_of(cfg.coulomb, lam=cfg.df_lam)
    screened = cfg.coulomb == "df" and builders.auto_screen(M)   # exact arm is dense streaming
    ks = SplatKS(mol, xc, grid=native_grid(mol, cfg.grid_level, chunk=4096 if screened else None),
                 coulomb=coulomb, screen=pairlist(eps=1e-7) if screened else None)
    print(f"{cfg.system}/{cfg.coulomb}/{cfg.xc} M={M} ({M / nao:.1f}x nao={nao})  "
          f"E_ref={'skipped' if ref is None else format(ref.e, '.6f')}  "
          f"steps={cfg.steps} seed={cfg.seed}", flush=True)

    model = init_model(mol, M, jr.PRNGKey(cfg.seed))
    checkpoint = None
    if cfg.resume or cfg.ckpt_every:                             # exact arm: chained b1 links continue
        from experiments.common import paths
        base = paths.artifact(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                              f"ckpt_{cfg.system}_{cfg.coulomb}_{cfg.xc}_M{M}_s{cfg.seed}")
        checkpoint = ckpt_spec(base, every=cfg.ckpt_every, resume=cfg.resume)

    t0 = time.time()
    res = train(ks, model, steps=cfg.steps, monitor=monitor(cfg.monitor), checkpoint=checkpoint)
    e = float(evaluate(ks, res.model, res.state))   # pass the trained state: hybrids need its aux_K
    gap = None if ref is None else (e - ref.e) * 1e3

    extra = {}
    e_ck, bias = builders.grid_audit(mol, xc, res.model, int(cfg.grid_check), e, res.state)
    if e_ck is not None:
        extra[f"E_grid{int(cfg.grid_check)}"] = e_ck
        extra["grid_bias_mha"] = bias
        if ref is not None:
            extra["beats_grid"] = bool(e_ck <= ref.e)  # the boolean that survives the finer grid

    if bool(cfg.get("koopmans", False)):
        from gs_dft.ks.canonical import homo as _homo
        from gs_dft.ks.orthonormalize import lowdin_orthonormalize
        from gs_dft.integrals.dense import overlap_matrix
        m = res.model
        C_occ = lowdin_orthonormalize(m.C, overlap_matrix(m.basis))
        eh = float(_homo(m.basis, C_occ, m.occupations, res.state.aux, ks.atom_coords,
                         ks.atom_charges, ks.grid_points, ks.grid_weights, xc, lam=cfg.df_lam))
        extra["eps_homo"] = eh
        extra["ip_eV"] = -eh * 27.211386245988          # Koopmans ionization potential

    if ref is not None:
        extra.update(E_ref=ref.e, gap_mha=gap, beats=bool(e <= ref.e))
    print(results.result("splat", cfg.system, coulomb=cfg.coulomb, xc=cfg.xc, nao=nao, M=M,
                        steps=cfg.steps, seed=cfg.seed, df_lam=cfg.df_lam, E=e,
                        wall_s=round(time.time() - t0),
                        peak_gpu_mb=round(probes.peak_gpu_mb()), **extra), flush=True)


if __name__ == "__main__":
    main()
