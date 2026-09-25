"""The screened exact streaming Coulomb (``screened_coulomb_energy``).

With the full ordered pair list it reproduces the dense exact E_J and its gradients to machine
precision, through chunk padding and the ``valid`` mask.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from gs_dft.basis.spectral import Splat
from gs_dft.integrals.dense import compute_eri_tensor, coulomb_matrix, coulomb_energy
from gs_dft.coulomb import stream as cs
from gs_dft.screening import neighbor_pairs

def _system(M=6, seed=0, nocc=3):
    s = Splat(
        quat=jr.normal(jr.PRNGKey(seed + 3), (M, 4)),         # random orientation (R normalizes)
        log_scale=jr.normal(jr.PRNGKey(seed), (M, 3)) * 0.3,  # anisotropic precision eigenvalues
        centers=jr.normal(jr.PRNGKey(seed + 1), (M, 3)),
    )
    C = jr.normal(jr.PRNGKey(seed + 2), (M, nocc))
    return s, C, jnp.ones(nocc)

def _dense_EJ(s, C, occ):
    P = C @ jnp.diag(occ) @ C.T
    return coulomb_energy(coulomb_matrix(compute_eri_tensor(s), P), P)

@pytest.mark.parametrize("bra_chunk", [4, 128])
def test_screened_exact_full_pairs_matches_dense(bra_chunk):
    s, C, occ = _system()
    pi, pj, valid = neighbor_pairs(s, eps=-1.0)                 # all ordered pairs => exact
    E = cs.screened_coulomb_energy(s, C, occ, pi, pj, valid, bra_chunk=bra_chunk)
    np.testing.assert_allclose(float(E), float(_dense_EJ(s, C, occ)), atol=1e-10)

def test_screened_exact_padded_pairs_matches_dense():
    s, C, occ = _system()
    n = neighbor_pairs(s, eps=-1.0)[0].shape[0]
    pi, pj, valid = neighbor_pairs(s, eps=-1.0, pad_to=n + 17)  # padded => valid mask exercised
    E = cs.screened_coulomb_energy(s, C, occ, pi, pj, valid, bra_chunk=8)
    np.testing.assert_allclose(float(E), float(_dense_EJ(s, C, occ)), atol=1e-10)

def test_screened_exact_grad_matches_dense():
    s, C, occ = _system()
    pi, pj, valid = neighbor_pairs(s, eps=-1.0)

    def loss_scr(s, C):
        return cs.screened_coulomb_energy(s, C, occ, pi, pj, valid, bra_chunk=4)

    def loss_dense(s, C):
        P = C @ jnp.diag(occ) @ C.T
        return 0.5 * jnp.sum(P * jnp.einsum("pqrs,rs->pq", compute_eri_tensor(s), P))

    gs = jax.grad(loss_scr, argnums=(0, 1))(s, C)
    gd = jax.grad(loss_dense, argnums=(0, 1))(s, C)
    for a, b in zip(jtu.tree_leaves(gs[0]), jtu.tree_leaves(gd[0])):   # all splat-param leaves
        np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-9)
    np.testing.assert_allclose(np.array(gs[1]), np.array(gd[1]), atol=1e-9)
