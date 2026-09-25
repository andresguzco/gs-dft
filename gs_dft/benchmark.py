"""Splat-vs-GTO accuracy benchmark harness (the reference helpers + CLI).

Compares a variationally-optimized Gaussian-splat KS energy against a GTO
double-zeta reference at **equal basis-function count** (M = nao of the
reference). Variational ⇒ lower energy is better; "beat" means E_splat ≤ E_ref.
Functional is held identical on both sides so the comparison isolates the
basis; ``xc_consistency`` bounds the residual density-pipeline gap between
the two engines' grid evaluations.

This module owns the native dftax SCF oracle (``build_reference``) and the
benchmark CLI; the training itself is
:func:`gs_dft.ks.train.train` on a plain dense-exact :class:`SplatKS`.

CLI:
    uv run python -m gs_dft.benchmark --molecule co2
"""

import argparse
import logging
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dftax.system import Molecule
from dftax.ks import KS, scf, df
from dftax.ks.energy import ao_on_grid
from dftax.basis.loader import build_basis_data
from dftax.grid import becke, becke_grid
from dftax.energy.xc import PBE, PBE0, B3LYP
from dftax.energy.grid import xc_energy
from gs_dft.ks.energy import SplatKS, init_model
from gs_dft.ks.train import train, monitor, splat_adam, ckpt, native_grid, _GRID_LEVELS

_XC_FUNCS = {"pbe": PBE, "pbe0": PBE0, "b3lyp": B3LYP}

log = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.509
CHEM_ACC_MHA = 1.0 / HARTREE_TO_KCAL * 1000.0   # 1 kcal/mol in mHa ≈ 1.594


# ---------------------------------------------------------------------------
#  Geometry + reference
# ---------------------------------------------------------------------------

def _geometry(name: str) -> str:
    from gs_dft import geometries
    lookup = {a[:-len("_geometry")]: getattr(geometries, a)
              for a in dir(geometries) if a.endswith("_geometry")}
    if name not in lookup:
        raise ValueError(f"Unknown molecule {name!r}. Available: {sorted(lookup)}")
    return lookup[name]


def _rotation_matrix(seed: int):
    """Deterministic proper rotation (det=+1) from a seed."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))           # fix QR sign ambiguity
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]                          # ensure proper rotation
    return Q


class Reference(NamedTuple):
    """The converged dftax reference: the KS energy functional + SCF result.

    ``res`` carries the converged density
    ``P`` / MOs, ``ks`` the built engine (grid term, integrals) so consumers
    like :func:`xc_consistency` don't rebuild the O(nao⁴) integrals.
    """
    ks: KS
    res: object          # dftax.ks.KSResult


def build_reference(molecule: str, basis: str = "cc-pvdz",
                    xc_name: str = "pbe", grid_level: int = 3, rotate_seed=None):
    """Native dftax KS reference (PySCF-free). Returns (mol, e_ref, nao, ref).

    ``mol`` is a :class:`dftax.system.Molecule` (spherical AOs, the cc-pV/def2
    convention) and ``ref`` a :class:`Reference` bundling the built ``KS``
    functional and the converged ``KSResult``.

    The reference uses **density-fitted (RI-JK) Coulomb** (``def2-universal-jkfit``,
    the standard aux) — O(nao²·naux), so it stays fast and in-memory as the system
    grows (the exact 4-center path materializes an O(nao⁴) tensor that OOMs beyond
    ~a dozen heavy atoms). DF error is ~0.1 mHa, far below any basis increment the
    splat comparison resolves, and matches the DF references the experiments use.

    ``rotate_seed`` rigidly rotates the geometry — the DFT energy is
    rotation-invariant (E_ref unchanged), certifying the splat energy is likewise
    frame-independent against the rotation-equivariant spectral chart.
    """
    mol = Molecule.from_xyz(_geometry(molecule), basis, unit="angstrom", spherical=True)
    if rotate_seed is not None:
        coords = np.asarray(mol.atom_coords()) @ _rotation_matrix(rotate_seed).T  # bohr
        mol = Molecule(list(mol.symbols), coords, basis, spherical=True)
    nr, lb = _GRID_LEVELS.get(int(grid_level), _GRID_LEVELS[3])
    ks = KS(mol, _XC_FUNCS[xc_name.lower()](), grid=becke(nr, lb),
            coulomb=df("def2-universal-jkfit"))
    res = scf(ks)
    return mol, float(res.e_tot), int(res.P.shape[-1]), Reference(ks, res)


def xc_consistency(mol, ref: Reference, grid_level: int = 3):
    """Two independent density→grid→E_xc pipelines on the converged reference.

    The "engine" side rebuilds the quadrature + AO values from scratch
    (``becke_grid`` + ``ao_on_grid``) and hand-contracts the density; the
    "ref" side is the reference engine's own grid term (``KS.e_xc``). Same
    functional code, independent grid/density plumbing — sub-mHa agreement
    certifies the density-evaluation path.
    """
    nr, lb = _GRID_LEVELS.get(int(grid_level), _GRID_LEVELS[3])
    coords = jnp.asarray(mol.atom_coords())
    gc, gw = becke_grid(list(mol.symbols), coords, n_radial=nr, lebedev=lb)
    basis = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis,
                             spherical=getattr(mol, "spherical", True))
    ao, dao = ao_on_grid(basis, gc)                            # (Ng, nao), (Ng, nao, 3)
    Ptot = jnp.sum(ref.res.P, axis=0)                          # spin-summed (nao, nao)
    rho = jnp.einsum("gm,mn,gn->g", ao, Ptot, ao)
    grad = 2.0 * jnp.einsum("gm,mn,gnd->gd", ao, Ptot, dao)
    rho_c = jnp.maximum(rho, 1e-30)
    xc = ref.ks.xc_term.xc
    exc_engine = float(xc_energy(xc, rho_c, gw, grad_rho=grad))
    exc_ref = float(ref.ks.e_xc(ref.res.P))
    return exc_engine, exc_ref


# ---------------------------------------------------------------------------
#  Benchmark driver
# ---------------------------------------------------------------------------

def run_benchmark(molecule, basis="cc-pvdz", xc_name="pbe",
                  n_steps=2000, lr=1e-2, grid_level=3, seeds=(0, 1, 2),
                  M_override=None, rotate_seed=None,
                  use_remat=False, log_every=1,
                  resume=False, ckpt_every=100, ckpt_dir="results/splat_bench"):
    mol, e_ref, nao, ref = build_reference(molecule, basis, xc_name, grid_level,
                                           rotate_seed=rotate_seed)
    M = M_override or nao
    exc_eng, exc_ref = xc_consistency(mol, ref, grid_level)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    rot = "" if rotate_seed is None else f" / rot{rotate_seed}"
    log.info(f"== {molecule} / splat / {xc_name}/{basis}{rot} ==")
    log.info(f"reference E_ref={e_ref:.6f} Ha  nao={nao}  -> M={M} splats  "
             f"(nelec={mol.nelectron})")
    log.info(f"XC-consistency (rebuilt grid vs reference-engine E_xc): "
             f"{exc_eng:.6f} vs {exc_ref:.6f}  Δ={(exc_eng-exc_ref)*1e3:.3f} mHa")

    xc = _XC_FUNCS[xc_name.lower()]()
    ks = SplatKS(mol, xc,
                 grid=native_grid(mol, grid_level))

    best = jnp.inf
    for s in seeds:
        t0 = time.time()
        model = init_model(mol, M, jr.PRNGKey(s))
        checkpoint = None
        if ckpt_dir:
            rtag = "" if rotate_seed is None else f"_rot{rotate_seed}"
            # include M so an M-sweep doesn't collide on one checkpoint
            base = os.path.join(ckpt_dir, f"{molecule}_splat_M{M}{rtag}_s{s}")
            checkpoint = ckpt(base, every=ckpt_every, resume=resume)
        res = train(ks, model, steps=n_steps,
                    optimizer=splat_adam(lr, n_steps),
                    monitor=monitor(log_every) if log_every else None, remat=use_remat,
                    checkpoint=checkpoint)
        # Fresh full forward for the converged energy + the ∫ρ electron-count sanity;
        # can OOM after a long run (allocator fragmentation) → fall back to the last
        # optimization step's energy (the converged value still lands).
        try:
            e_final, eaux = ks(res.model)
            e_splat, nelec = float(e_final), float(eaux.nelec)
        except Exception as exc:  # noqa: BLE001 - OOM/runtime; keep the result
            log.warning(f"  final sanity forward failed ({type(exc).__name__}); "
                        f"returning last-step energy, ∫ρ skipped")
            e_last = float(res.energy[-1]) if len(res.energy) else float("nan")
            e_splat, nelec = e_last, float("nan")
        # Overlap conditioning on the converged basis — diagnoses near-linear-
        # dependence (the wall for overcomplete / large-M splat bases).
        try:
            from gs_dft.integrals.dense import one_electron_integrals
            S, _, _ = one_electron_integrals(res.model.basis, jnp.array(mol.atom_coords()),
                                             jnp.array(mol.atom_charges(), dtype=float))
            ev = jnp.linalg.eigvalsh(S)
            log.info(f"  cond(S)={float(ev[-1] / jnp.clip(ev[0], 1e-30)):.3e}  "
                     f"min_eval(S)={float(ev[0]):.3e}")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"  cond(S) check failed ({type(exc).__name__})")
        log.info(f"  seed {s}: E_splat={e_splat:.6f} Ha  N={nelec:.3f}  "
                 f"({time.time()-t0:.0f}s)")
        best = min(best, e_splat)

    gap_mha = (best - e_ref) * 1e3
    log.info(f"RESULT {molecule}/splat{rot}: E_splat={best:.6f}  E_ref={e_ref:.6f}  "
             f"gap={gap_mha:+.3f} mHa  beats_ref={best <= e_ref}  "
             f"chem_acc={abs(gap_mha) <= CHEM_ACC_MHA}")
    return {
        "molecule": molecule, "basis": basis, "xc": xc_name,
        "M": M, "nao": nao, "e_ref": e_ref, "e_splat": float(best),
        "gap_mha": float(gap_mha), "beats_ref": bool(best <= e_ref),
        "xc_gap_mha": float((exc_eng - exc_ref) * 1e3),
        "rotate_seed": rotate_seed,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Persistent XLA compilation cache: DRAC/SLURM requeues re-run identical shapes,
    # so cache hits erase the ~30 s-per-function recompiles. Env var wins if set.
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update("jax_compilation_cache_dir",
                          os.path.expanduser("~/.cache/gs_dft/xla"))
    p = argparse.ArgumentParser(description="Splat-vs-GTO accuracy benchmark")
    p.add_argument("--molecule", default="co2")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--xc", default="pbe", choices=["pbe", "pbe0"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--grid_level", type=int, default=3)
    p.add_argument("--log_every", type=int, default=1,
                   help="log the energy every N steps (default 1; splat ERIs are "
                        "slow so per-step monitoring is the norm)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--M", type=int, default=None, help="override M (default = nao)")
    p.add_argument("--rotate_seed", type=int, default=None,
                   help="rigidly rotate the geometry (frame-independence control "
                        "for the full-covariance spectral chart's equivariance)")
    p.add_argument("--remat", action="store_true",
                   help="enable jax.checkpoint (lower peak memory, slower) — "
                        "only needed when a run would OOM, e.g. large M")
    p.add_argument("--resume", action="store_true",
                   help="resume from a checkpoint if one exists (preemption-safe "
                        "chaining of short DRAC allocations)")
    p.add_argument("--ckpt_every", type=int, default=20,
                   help="save a checkpoint every N steps (atomic; default 20)")
    p.add_argument("--ckpt_dir", default="results/splat_bench",
                   help="directory for checkpoints (empty string disables)")
    args = p.parse_args()
    run_benchmark(args.molecule, args.basis, args.xc,
                  args.steps, args.lr, args.grid_level, tuple(args.seeds), args.M,
                  rotate_seed=args.rotate_seed,
                  use_remat=args.remat, log_every=args.log_every,
                  resume=args.resume, ckpt_every=args.ckpt_every,
                  ckpt_dir=args.ckpt_dir)


if __name__ == "__main__":
    main()
