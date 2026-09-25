"""XC grid screening (``gs_dft/screening/grid.py``).

The chunked evaluation equals the dense one, the screened evaluation equals the chunked one when no
pair is dropped and stays close at a finite cutoff, and all are differentiable.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.integrals import dense as fullmod
from gs_dft.screening import grid as gs

BACKENDS = ["splat"]
_MOD = {"splat": fullmod}

def _setup(backend, M=40, seed=0):
    k1, k2, k3 = jr.split(jr.PRNGKey(seed), 3)
    centers = jr.uniform(k1, (6, 3), minval=-3.0, maxval=3.0)
    # aniso_jitter perturbs the log-eigenvalues ℓ to lift the degenerate isotropic init
    s = init_spectral_splats(centers, M, key=k2, aniso_jitter=0.3)
    C = jr.normal(k3, (M, 5))
    occ = jnp.ones(5)
    grid = jr.uniform(jr.PRNGKey(seed + 1), (1000, 3), minval=-4.0, maxval=4.0)
    return s, C, occ, grid

@pytest.mark.parametrize("backend", BACKENDS)
def test_2a_chunked_equals_dense(backend):
    s, C, occ, grid = _setup(backend)
    mod = _MOD[backend]
    rho_d, grad_d = mod.eval_density_and_grad_on_grid(s, C, occ, grid)
    rho_c, grad_c = gs.chunked_density_and_grad(s, C, occ, grid, chunk=128)
    np.testing.assert_allclose(np.asarray(rho_c), np.asarray(rho_d), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(np.asarray(grad_c), np.asarray(grad_d), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(gs.chunked_density(s, C, occ, grid, chunk=256)),
        np.asarray(mod.eval_density_on_grid(s, C, occ, grid)), rtol=1e-11, atol=1e-11)

@pytest.mark.parametrize("backend", BACKENDS)
def test_2b_screened_equals_exact_when_unscreened(backend):
    s, C, occ, grid = _setup(backend)
    rho_x, grad_x = gs.chunked_density_and_grad(s, C, occ, grid, chunk=256)
    gi, si, v = gs.grid_neighbor_pairs(s, grid, eps=1e-80)          # rcut huge ⇒ all significant pairs
    rho_a, grad_a = gs.screened_density_and_grad(s, C, occ, grid, gi, si, v, chunk=2048)
    np.testing.assert_allclose(np.asarray(rho_a), np.asarray(rho_x), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(grad_a), np.asarray(grad_x), rtol=1e-10, atol=1e-10)

@pytest.mark.parametrize("backend", BACKENDS)
def test_2b_bounded_error_and_screens(backend):
    s, C, occ, grid = _setup(backend, M=60)
    rho_x, grad_x = gs.chunked_density_and_grad(s, C, occ, grid, chunk=256)
    gi, si, v = gs.grid_neighbor_pairs(s, grid, eps=1e-8)
    assert gi.shape[0] < grid.shape[0] * s.n_basis                  # pairs actually dropped
    rho_s, grad_s = gs.screened_density_and_grad(s, C, occ, grid, gi, si, v, chunk=2048)
    assert float(jnp.max(jnp.abs(rho_s - rho_x))) < 1e-4
    assert float(jnp.max(jnp.abs(grad_s - grad_x))) < 1e-2

@pytest.mark.parametrize("backend", BACKENDS)
def test_grid_screening_differentiable(backend):
    s, C, occ, grid = _setup(backend, M=30)
    gi, si, v = gs.grid_neighbor_pairs(s, grid, eps=1e-8)
    g = jax.grad(lambda Cc: jnp.sum(
        gs.screened_density_and_grad(s, Cc, occ, grid, gi, si, v, chunk=1024)[0] ** 2))(C)
    assert jnp.all(jnp.isfinite(g))
    g2 = jax.grad(lambda Cc: jnp.sum(
        gs.chunked_density_and_grad(s, Cc, occ, grid, chunk=128)[0] ** 2))(C)
    assert jnp.all(jnp.isfinite(g2))
