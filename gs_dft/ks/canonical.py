"""Canonical orbital energies for a DIRECTLY MINIMIZED splat model.

Direct minimization returns an occupied subspace, not orbitals: the energy is invariant under any
rotation of the occupied coefficients among themselves, so the columns of ``C`` carry no orbital
energies and ``eps_HOMO`` is not defined until the gauge is fixed.

Fixing it does NOT require an SCF. At the converged density the Fock operator is ``F = sym(dE/dP)``,
and the canonical orbitals are the ones that diagonalize it; restricted to the occupied subspace
that is an ``(n_occ, n_occ)`` eigenproblem. One Fock build, one small eigensolve, no iteration —
and, unlike warm-starting an SCF, it reports the orbital energies OF THE MINIMIZED STATE rather
than of a nearby SCF solution.

Koopmans' theorem makes this worth doing under Hartree-Fock, where ``-eps_HOMO`` is the ionization
potential and therefore comparable to CCSD(T) and experiment (see the GW100 benchmark). A
Kohn-Sham eigenvalue carries no such meaning.
"""
import equinox as eqx
import jax
import jax.numpy as jnp

from dftax.energy.grid import xc_energy
from gs_dft.coulomb import ri as _df
from gs_dft.integrals import dense as _full

__all__ = ["occupied_orbital_energies", "homo"]


def _electronic_energy(P, basis, aux, Hcore, Phi, dPhi, gw, xc, lam):
    """E(P) = Tr(P H) + E_J + hf_coeff E_K + E_xc — the same objective the trainer minimizes."""
    E = jnp.sum(P * Hcore) + _df.df_coulomb_energy(basis, P, aux, lam=lam)
    c_hf = float(getattr(xc, "hf_coeff", 0.0) or 0.0)
    if c_hf:
        E = E + c_hf * _df.df_exchange_energy_density(basis, P, aux, lam=lam)
    PhiP = Phi @ P
    rho = jnp.maximum(jnp.sum(PhiP * Phi, axis=1), 1e-30)
    grad_rho = (2.0 * jnp.einsum("gm,gmd->gd", PhiP, dPhi)) if xc.xc_type == "GGA" else None
    return E + xc_energy(xc, rho, gw, grad_rho=grad_rho)


@eqx.filter_jit
def _fock(P, basis, aux, Hcore, Phi, dPhi, gw, xc, lam):
    G = jax.grad(_electronic_energy)(P, basis, aux, Hcore, Phi, dPhi, gw, xc, lam)
    return 0.5 * (G + G.T)


def occupied_orbital_energies(basis, C_occ, occ, aux, atom_coords, atom_charges,
                              grid_points, grid_weights, xc, lam=1e-8):
    """Canonical occupied orbital energies of the minimized state, ascending.

    ``C_occ`` must be S-orthonormal (``lowdin_orthonormalize(model.C, S)``).
    """
    S, T, V = _full.one_electron_integrals(basis, atom_coords, atom_charges)
    Phi, dPhi = _full.eval_basis_and_grad_on_grid(basis, grid_points)
    occ = jnp.asarray(occ, float)
    P = (C_occ * occ[None, :]) @ C_occ.T
    F = _fock(P, basis, aux, T + V, Phi, dPhi, grid_weights, xc, lam)
    return jnp.linalg.eigvalsh(C_occ.T @ F @ C_occ)


def homo(*args, **kwargs):
    """Highest occupied orbital energy (Hartree). ``-homo`` is the Koopmans ionization potential."""
    return occupied_orbital_energies(*args, **kwargs)[-1]
