"""Dense Laplace-quadrature integral/grid kernels + closed-form 3x3 linear algebra."""

from gs_dft.integrals.dense import (
    overlap_matrix, kinetic_matrix, nuclear_attraction_matrix,
    one_electron_integrals, compute_eri_tensor,
    coulomb_matrix, exchange_matrix, coulomb_energy, exchange_energy,
    eval_basis_on_grid, eval_basis_and_grad_on_grid,
    eval_density_on_grid, eval_density_and_grad_on_grid,
)
from gs_dft.integrals.linalg3 import det3, adj3, inv3, solve3

__all__ = [
    "overlap_matrix", "kinetic_matrix", "nuclear_attraction_matrix",
    "one_electron_integrals", "compute_eri_tensor",
    "coulomb_matrix", "exchange_matrix", "coulomb_energy", "exchange_energy",
    "eval_basis_on_grid", "eval_basis_and_grad_on_grid",
    "eval_density_on_grid", "eval_density_and_grad_on_grid",
    "det3", "adj3", "inv3", "solve3",
]
