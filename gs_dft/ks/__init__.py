"""Build + run: the SplatKS energy builder, its Coulomb term values, the
training/evaluation verbs, forces, sharding, and diagnostics."""

from gs_dft.ks.energy import SplatKS, SplatModel, SplatState, refresh, init_model, nao
from gs_dft.ks.terms import exact, df, pairlist, cellgrid, hf_coeff
from gs_dft.ks.train import (
    train, evaluate, flops, TrainResult, splat_adam, ckpt, collapse, monitor,
    native_grid,
)
from gs_dft.ks.forces import splat_forces, splat_energy_and_forces
from gs_dft.ks.orthonormalize import (
    lowdin_orthonormalize, orthonormalize, transform, reg_inv_sqrt,
)
from gs_dft.ks.shard import make_mesh
from gs_dft.ks.diagnostics import (
    CollapseGuard,
    lindep_metrics, dense_gram_eig_ends,
)

__all__ = [
    "SplatKS", "SplatModel", "SplatState", "refresh", "init_model", "nao",
    "exact", "df", "pairlist", "cellgrid", "hf_coeff",
    "train", "evaluate", "flops", "TrainResult",
    "splat_adam", "ckpt", "collapse", "monitor", "native_grid",
    "splat_forces", "splat_energy_and_forces",
    "lowdin_orthonormalize", "orthonormalize", "transform", "reg_inv_sqrt",
    "make_mesh", "CollapseGuard",
    "lindep_metrics", "dense_gram_eig_ends",
]
