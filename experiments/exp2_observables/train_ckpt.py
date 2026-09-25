"""Train and save the splat checkpoint for one Figure 3 system.

12,000 steps of the standard recipe, followed by an SCF polish of the coefficients (a warm
``scf_solve``, kept if it lowers the energy). Writes ``(model, aux)`` to
``results/ckpt_<system>.eqx``; load it with the ``fresh_init`` template below.

    uv run python -m experiments.exp2_observables.train_ckpt experiment=exp2_train_ckpt system=ethanol
"""
import os
import time

import hydra
from omegaconf import DictConfig
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from experiments.common import systems, builders, results, probes, tracking
from gs_dft import init_model, nao
from gs_dft.ks.train import train, monitor, evaluate
from gs_dft.ks.energy import SplatKS, SplatState, refresh
from gs_dft.ks.terms import df
from gs_dft.integrals import dense as _full
from gs_dft.ks.orthonormalize import lowdin_orthonormalize
from experiments.exp2_observables.scf import scf_solve
from dftax.energy.xc import PBE

M_BY = {"water": 192, "ethanol": 432,   # the headline configs (8x / 6x). RESULTS, not defaults —
        "alanine_dipeptide": 800}       # `m` in the config overrides only to depart from them.

def ckpt_path(system, M=None, polished=False):
    """Where the checkpoint for ``(system, M)`` lives."""
    from experiments.common import paths
    M = int(M) if M is not None else M_BY[system]
    return paths.artifact(RES, f"ckpt_{system}_M{M}{'_polished' if polished else ''}.eqx")


AUX_MULT = 1                            # part of the checkpoint format: naux = AUX_MULT * M sets the
                                        # auxiliary leaf shape, and fresh_init takes no cfg, so
                                        # main() asserts cfg.aux_mult matches.
GRID = 3                                # the training grid level for the published checkpoints
                                        # (== cfg.grid_level's default); the analysis scripts
                                        # (polish_forces) re-load at this level.
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def fresh_init(system, mol, M=None, seed=0):
    """The production init, verbatim — also the DESERIALIZATION TEMPLATE."""
    M = int(M) if M is not None else M_BY[system]
    model = init_model(mol, M, jr.PRNGKey(seed))
    ks = builders.splat_ks(mol, builders.xc_of("pbe"), grid_level=GRID, screened=True,
                           aux_mult=AUX_MULT)
    aux = refresh(ks, model.basis).aux
    return model, aux


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    tracking.init(cfg, __name__, name=f"ckpt_{cfg.system}_s{cfg.seed}")
    system = cfg.system
    if int(cfg.aux_mult) != AUX_MULT:    # would serialize an aux fresh_init cannot rebuild
        raise ValueError(
            f"exp4 checkpoints are pinned at aux_mult={AUX_MULT} (naux = {AUX_MULT}xM); got "
            f"{cfg.aux_mult}. fresh_init is the deserialization template and takes no cfg, so this "
            f"would write a checkpoint nothing can load. Change AUX_MULT in train_ckpt.py and "
            f"RE-TRAIN every checkpoint, or leave it alone.")
    mol = systems.molecule(cfg)
    M = int(cfg.m) if cfg.m is not None else M_BY[system]
    xc = builders.xc_of(cfg.xc)
    print(f"exp4 ckpt {system}/full M={M} ({M / nao(mol):.0f}x nao={nao(mol)}) steps={cfg.steps} "
          f"seed={cfg.seed}", flush=True)
    model, aux = fresh_init(system, mol, M, cfg.seed)
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), float)

    t0 = time.time()
    ks = builders.splat_ks(mol, xc, grid_level=cfg.grid_level, screened=True,
                           df_lam=cfg.df_lam, aux_mult=cfg.aux_mult,
                           screen_pad=cfg.screen_pad)
    res = train(ks, model, steps=cfg.steps, refresh_every=cfg.refresh, monitor=monitor(2000))
    peak_train_mb = round(probes.peak_gpu_mb())
    model, aux = res.model, res.state.aux
    gp, gw = ks.grid_points, ks.grid_weights
    print(f"trained in {time.time() - t0:.0f}s; arm-D SCF polish", flush=True)

    # arm-D polish: solve C exactly at the frozen (theta, aux); accept-if-lower (DF objective)
    S = _full.one_electron_integrals(model.basis, coords, charges)[0]
    res = scf_solve(model.basis, aux, coords, charges, gp, gw, model.occupations, PBE(),
                    C0=lowdin_orthonormalize(model.C, S), max_iter=50)
    ef = SplatKS((coords, charges), xc, grid=(gp, gw), coulomb=df())
    state = SplatState(aux=aux)
    n_occ = model.occupations.shape[0]
    D = jnp.diag(1.0 + jnp.linspace(0.0, 1e-3, n_occ))   # Löwdin-invariant eigh-degeneracy guard
    cand = eqx.tree_at(lambda m: m.C, model, res.C @ D)
    E_cur = float(ef(model, state)[0])
    E_cand = float(ef(cand, state)[0])
    polished = E_cand < E_cur
    if polished:
        model = cand
    print(f"  scf polish: n_iter={res.n_iter} conv={res.converged} "
          f"E {E_cur:.8f} -> {E_cand:.8f} accepted={polished}", flush=True)

    E_exact = float(evaluate(ks, model))
    _gf = builders.grid_fields(mol, PBE(), model, cfg.grid_check, E_exact)
    path = ckpt_path(system, M)
    probes.atomic_save(path, (model, aux))
    print(results.result("ckpt", system, M=M, steps=cfg.steps, seed=cfg.seed,
                        screen_pad=float(cfg.screen_pad), **_gf,
                        E=E_exact, peak_gpu_mb=round(probes.peak_gpu_mb()), peak_train_mb=peak_train_mb, polished=polished, wall_s=round(time.time() - t0), path=path),
          flush=True)


if __name__ == "__main__":
    main()
