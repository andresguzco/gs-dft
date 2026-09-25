"""RI-K in the density matrix equals RI-K in the occupied coefficients.

``df_exchange_energy`` and ``df_exchange_energy_density`` contract the same three-center integrals,
so they agree to round-off for ``P = 2 C_occ C_occ^T``. Only the density form can build a Fock
matrix as ``sym(dE/dP)``.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft.basis.isotropic import init_product_aux
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.coulomb import ri as cdf


def _splats(M=24, seed=0):
    k1, k2 = jr.split(jr.PRNGKey(seed))
    centers = jr.uniform(k1, (4, 3), minval=-2.0, maxval=2.0)
    return init_spectral_splats(centers, M, key=k2, aniso_jitter=0.3)


def _c_occ(M, n_occ=4, seed=3):
    """An S-orthonormality-agnostic set of occupied coefficients: only the contraction is tested."""
    return jr.normal(jr.PRNGKey(seed), (M, n_occ)) / np.sqrt(M)


@pytest.mark.parametrize("M,n_occ", [(24, 3), (32, 5)])
def test_density_form_matches_the_occupied_form(M, n_occ):
    s = _splats(M)
    aux = init_product_aux(s, scales=(1.0,), diagonal_only=True)
    C = _c_occ(M, n_occ)
    P = 2.0 * C @ C.T                       # closed shell

    e_occ = float(cdf.df_exchange_energy(s, C, aux, lam=1e-8))
    e_den = float(cdf.df_exchange_energy_density(s, P, aux, lam=1e-8))
    assert np.isfinite(e_occ) and np.isfinite(e_den)
    assert e_occ < 0.0, "exchange energy must be negative"
    rel = abs(e_den - e_occ) / max(abs(e_occ), 1e-30)
    assert rel < 1e-9, f"density form {e_den:.10f} vs occupied form {e_occ:.10f} (rel {rel:.2e})"


def test_density_form_is_differentiable_in_P():
    """The whole point is `sym(dE/dP)`; a non-finite gradient would make the Fock matrix garbage."""
    s = _splats(24)
    aux = init_product_aux(s, scales=(1.0,), diagonal_only=True)
    C = _c_occ(24, 3)
    P = 2.0 * C @ C.T
    G = jax.grad(lambda p: cdf.df_exchange_energy_density(s, p, aux, lam=1e-8))(P)
    assert np.all(np.isfinite(np.asarray(G))), "non-finite dE_K/dP"
    assert np.max(np.abs(np.asarray(G))) > 0.0, "dE_K/dP is identically zero"
