"""gs_dft: Gaussian-splat Kohn-Sham DFT on the dftax engine.

Nodeless full-covariance Gaussian splats (quaternion + log-eigenvalue spectral
chart) as a variational KS-DFT basis, with exact streamed / density-fitted /
screened / multi-GPU Coulomb strategies as composable values.

The canonical flow::

    from gs_dft import (SplatKS, df, pairlist, native_grid,
                                 train, evaluate, splat_forces, init_model)
    from dftax.system import Molecule
    from dftax.energy.xc import PBE

    mol   = Molecule.from_xyz("O 0 0 0; H 0 0 1; H 0 1 0", "cc-pvdz")   # native, no PySCF
    ks    = SplatKS(mol, PBE(),
                    grid=native_grid(mol, 3, chunk=4096),
                    coulomb=df(), screen=pairlist(eps=1e-7))
    model = init_model(mol, M=126, key=jr.PRNGKey(0))
    res   = train(ks, model, steps=6000)                 # TrainResult
    E     = evaluate(ks, res.model, res.state)           # exact-Coulomb energy
    F     = splat_forces(ks, res.model, res.state)       # Hellmann-Feynman

Import the subpackages directly for the full kernel surface
(``integrals.dense``, ``coulomb.stream``, ``coulomb.ri``, ``screening.pairs``,
``screening.grid``, ``ks.shard``, ``ks.orthonormalize``).

NOTE: importing this package enables ``jax_enable_x64`` globally. Splat
KS-DFT is float64-only (DFT energies need it), and several engine constants
(e.g. dftax's Boys interpolation table) are built at import time — enabling
x64 here, before any kernel module loads, is what makes their precision
independent of the caller's import order.
"""

import jax as _jax
_jax.config.update("jax_enable_x64", True)

from gs_dft.ks.energy import SplatKS, SplatModel, SplatState, refresh, init_model, nao
from gs_dft.ks.terms import exact, df, pairlist, cellgrid, hf_coeff
from gs_dft.ks.train import (
    train, evaluate, flops, TrainResult, splat_adam, ckpt, collapse, monitor,
    native_grid,
)
from gs_dft.ks.forces import splat_forces, splat_energy_and_forces
from gs_dft.basis.spectral import Splat, init_spectral_splats
from gs_dft.basis.isotropic import (
    IsotropicSplat, init_isotropic_splats, init_product_aux,
)
from gs_dft.basis.chem import build_init_sites, init_chem_splats, chem
from gs_dft.screening import neighbor_pairs
from gs_dft.ks.orthonormalize import lowdin_orthonormalize
from gs_dft.ks.shard import make_mesh
from gs_dft.ks.diagnostics import CollapseGuard

__all__ = [
    # build: model + energy functional, choices as values
    "SplatKS", "SplatModel", "SplatState", "refresh", "init_model", "nao",
    "exact", "df", "pairlist", "cellgrid", "native_grid", "make_mesh",
    # run: verbs + result
    "train", "evaluate", "flops", "TrainResult",
    "splat_adam", "ckpt", "collapse", "monitor",
    "splat_forces", "splat_energy_and_forces",
    # splat bases + initializers
    "Splat", "init_spectral_splats",
    "IsotropicSplat", "init_isotropic_splats", "init_product_aux",
    "build_init_sites", "init_chem_splats", "chem",
    # building blocks
    "neighbor_pairs", "lowdin_orthonormalize", "hf_coeff", "CollapseGuard",
]
