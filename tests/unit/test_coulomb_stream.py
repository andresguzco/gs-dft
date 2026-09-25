"""The streaming ``custom_vjp`` Coulomb (``gs_dft/coulomb/stream.py``) against the dense ERI tensor.

E_J, the Coulomb matrix J and the gradients with respect to the splats and the density match to
machine precision, for ordered and unique pair enumerations and through chunk padding.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import pytest

from gs_dft.basis.spectral import Splat
from gs_dft.integrals.dense import compute_eri_tensor, coulomb_matrix, coulomb_energy
from gs_dft.coulomb import stream as cs

def _system(M=5, seed=0):
    s = Splat(
        quat=jr.normal(jr.PRNGKey(seed + 3), (M, 4)),         # random orientation (R normalizes)
        log_scale=jr.normal(jr.PRNGKey(seed), (M, 3)) * 0.3,  # anisotropic precision eigenvalues
        centers=jr.normal(jr.PRNGKey(seed + 1), (M, 3)),
    )
    C = jr.normal(jr.PRNGKey(seed + 2), (M, 3))
    return s, C @ C.T          # symmetric density

@pytest.mark.parametrize("block", ["full_eri_block", "full_eri_block_schur"])
@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("chunk", [3, 128])
def test_fused_J_and_EJ_match_full_tensor(block, unique, chunk):
    s, P = _system()
    M = s.n_basis
    block_fn = getattr(cs, block)
    eri = compute_eri_tensor(s)
    J_ref = coulomb_matrix(eri, P)
    EJ_ref = coulomb_energy(J_ref, P)
    J = cs.coulomb_J_fused(block_fn, M, unique, s, P, chunk)
    EJ = cs.coulomb_energy_fused(block_fn, M, unique, s, P, chunk)
    np.testing.assert_allclose(np.array(J), np.array(J_ref), atol=1e-10)
    np.testing.assert_allclose(float(EJ), float(EJ_ref), atol=1e-10)

@pytest.mark.parametrize("unique", [False, True])
def test_custom_vjp_grads_match_autodiff(unique):
    s, P = _system()
    M = s.n_basis

    def loss_ref(s, P):
        e = compute_eri_tensor(s)
        return 0.5 * jnp.sum(P * jnp.einsum("pqrs,rs->pq", e, P))

    def loss_fused(s, P):
        return cs.coulomb_energy_fused(cs.full_eri_block_schur, M, unique, s, P, 3)

    gr = jax.grad(loss_ref, argnums=(0, 1))(s, P)
    gf = jax.grad(loss_fused, argnums=(0, 1))(s, P)
    np.testing.assert_allclose(np.array(gf[0].log_scale), np.array(gr[0].log_scale), atol=1e-8)
    np.testing.assert_allclose(np.array(gf[0].quat), np.array(gr[0].quat), atol=1e-8)
    np.testing.assert_allclose(np.array(gf[0].centers), np.array(gr[0].centers), atol=1e-8)
    np.testing.assert_allclose(np.array(gf[1]), np.array(gr[1]), atol=1e-9)

@pytest.mark.slow                                   # build_reference co2 SCF oracle
def test_splatenergy_fused_analytic_runs():
    """End-to-end: the exact streaming Coulomb gives a finite energy and gradients. Agreement with
    the dense ERI tensor is covered by ``test_fused_J_and_EJ_match_full_tensor``."""
    from dftax.grid import becke_grid
    from gs_dft.benchmark import build_reference
    from gs_dft import init_model
    from gs_dft.ks.energy import SplatKS
    from dftax.energy.xc import PBE

    mol, _, nao, _ = build_reference("co2", "cc-pvdz", "pbe", 1)
    model = init_model(mol, nao, jr.PRNGKey(0))
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=35, lebedev=110)
    ks = SplatKS(mol, PBE(), grid=(gp, gw))
    E = float(ks(model)[0])
    g = eqx.filter_grad(lambda m: ks(m)[0])(model)
    assert np.isfinite(E)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(g.basis))
    assert np.all(np.isfinite(np.array(g.C)))


# --- the fused scalar (pair|pair) kernel ------------------------------------------------------

def _wide_system(M=24, seed=5):
    """Splats spanning core to diffuse exponents at realistic anisotropy (aspect ~e^1.5)."""
    base = jr.uniform(jr.PRNGKey(seed), (M, 1), minval=-2.0, maxval=7.0)
    return Splat(
        quat=jr.normal(jr.PRNGKey(seed + 3), (M, 4)),
        log_scale=base + 0.5 * jr.normal(jr.PRNGKey(seed + 2), (M, 3)),
        centers=jr.normal(jr.PRNGKey(seed + 1), (M, 3)) * 1.5,
    )


def test_fused_block_quadrature_converged_and_matches_schur():
    """Default nodes against 128 (quadrature convergence, values and gradients), and a loose check
    against the Schur kernel, whose 20-node half-line rule is off by up to 1e-2 on core-core pairs."""
    s = _wide_system()
    M = s.n_basis
    pi, pj, _ = cs.make_pairs(M, True)
    bi, bj = pi[:40], pj[:40]
    got = cs.full_eri_block_fused(s, bi, bj, pi, pj)
    conv = cs.full_eri_block_fused(s, bi, bj, pi, pj, n_nodes=128)
    scale = float(jnp.max(jnp.abs(conv)))
    np.testing.assert_allclose(np.array(got), np.array(conv), rtol=2e-6, atol=1e-12 * scale)
    schur = cs.full_eri_block_schur(s, bi, bj, pi, pj)
    np.testing.assert_allclose(np.array(got), np.array(schur), rtol=5e-2, atol=1e-10 * scale)
    w_ = jr.normal(jr.PRNGKey(9), got.shape)
    g16 = jax.grad(lambda t_: jnp.sum(w_ * cs.full_eri_block_fused(t_, bi, bj, pi, pj)))(s)
    g64 = jax.grad(lambda t_: jnp.sum(w_ * cs.full_eri_block_fused(t_, bi, bj, pi, pj, n_nodes=128)))(s)
    for a, b in zip(jax.tree.leaves(g16), jax.tree.leaves(g64)):
        gs = float(jnp.max(jnp.abs(b)))
        np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-6 * gs)   # near-cancelling quaternion entries


def test_fused_block_isotropic_limit_is_boys():
    """Isotropic splats: (ij|kl) has the closed Boys form F0(T) = ½√(π/T) erf(√T), exact in float64."""
    from scipy.special import erf
    M = 12
    la = jr.uniform(jr.PRNGKey(2), (M,), minval=-3.0, maxval=8.0)
    mu = jr.normal(jr.PRNGKey(3), (M, 3)) * 1.5
    s = Splat(quat=jnp.tile(jnp.array([1.0, 0, 0, 0]), (M, 1)), log_scale=jnp.repeat(la[:, None], 3, 1),
              centers=mu)
    pi, pj, _ = cs.make_pairs(M, True)
    a = np.asarray(jnp.exp(la)); mu_ = np.asarray(mu); pi_, pj_ = np.asarray(pi), np.asarray(pj)
    p = a[pi_] + a[pj_]
    P = (a[pi_, None] * mu_[pi_] + a[pj_, None] * mu_[pj_]) / p[:, None]
    N = (2 * a / np.pi) ** 0.75
    K = np.exp(-a[pi_] * a[pj_] / p * np.sum((mu_[pi_] - mu_[pj_]) ** 2, 1)) * N[pi_] * N[pj_]
    rho = p[:, None] * p[None, :] / (p[:, None] + p[None, :])
    T = rho * np.sum((P[:, None] - P[None, :]) ** 2, -1)
    F0 = np.where(T < 1e-14, 1.0 - T / 3.0, 0.5 * np.sqrt(np.pi / np.maximum(T, 1e-300)) * erf(np.sqrt(T)))
    ref = K[:, None] * K[None, :] * 2 * np.pi ** 2.5 / (p[:, None] * p[None, :] * np.sqrt(p[:, None] + p[None, :])) * F0
    got = np.asarray(cs.full_eri_block_fused(s, pi, pj, pi, pj))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-14 * np.abs(ref).max())
