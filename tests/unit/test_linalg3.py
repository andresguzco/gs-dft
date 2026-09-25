"""Closed-form 3×3 algebra vs jnp.linalg (SPD batches, values + gradients)."""

import jax
import jax.numpy as jnp
import jax.random as jr

from gs_dft.integrals.linalg3 import det3, inv3, solve3

def _spd_batch(key, shape=(64,)):
    """Random SPD (..., 3, 3) like the kernels see: A_i + A_j + t² I."""
    B = jr.normal(key, shape + (3, 3))
    return jnp.einsum("...kl,...ml->...km", B, B) + 0.5 * jnp.eye(3)

def test_det3_inv3_solve3_match_lapack():
    key = jr.PRNGKey(0)
    A = _spd_batch(key)
    b = jr.normal(jr.split(key)[1], A.shape[:-2] + (3,))
    assert jnp.max(jnp.abs(det3(A) - jnp.linalg.det(A))) < 1e-12
    assert jnp.max(jnp.abs(inv3(A) - jnp.linalg.inv(A))) < 1e-12
    ref = jnp.linalg.solve(A, b[..., None])[..., 0]
    assert jnp.max(jnp.abs(solve3(A, b) - ref)) < 1e-12

def test_multibatch_shapes():
    A = _spd_batch(jr.PRNGKey(1), shape=(4, 5))
    b = jr.normal(jr.PRNGKey(2), (4, 5, 3))
    assert det3(A).shape == (4, 5)
    assert inv3(A).shape == (4, 5, 3, 3)
    assert solve3(A, b).shape == (4, 5, 3)
    assert jnp.max(jnp.abs(det3(A) - jnp.linalg.det(A))) < 1e-12

def test_gradients_match_lapack():
    key = jr.PRNGKey(3)
    A = _spd_batch(key, shape=(8,))
    b = jr.normal(jr.split(key)[1], (8, 3))

    for f, f_ref in [
        (lambda X: jnp.sum(jnp.log(det3(X))), lambda X: jnp.sum(jnp.log(jnp.linalg.det(X)))),
        (lambda X: jnp.sum(inv3(X) ** 2), lambda X: jnp.sum(jnp.linalg.inv(X) ** 2)),
        (lambda X: jnp.sum(solve3(X, b) ** 2),
         lambda X: jnp.sum(jnp.linalg.solve(X, b[..., None])[..., 0] ** 2)),
    ]:
        g = jax.grad(f)(A)
        g_ref = jax.grad(f_ref)(A)
        assert jnp.max(jnp.abs(g - g_ref)) < 1e-10
