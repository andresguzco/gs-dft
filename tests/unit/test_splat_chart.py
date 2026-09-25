"""The quaternion and log-eigenvalue covariance chart (``gs_dft/basis/spectral.py``).

1. Gradients are finite at isotropic and near-degenerate splats.
2. A global rotation conjugates A to Q A Q^T and leaves the overlap invariant.
3. The streaming Coulomb on the chart equals the dense E_J.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from gs_dft.integrals import dense as full
from gs_dft.basis.spectral import Splat, quat_to_rotation, quat_mul

def _random_splat(M, key):
    k = jr.split(key, 3)
    return Splat(quat=jr.normal(k[0], (M, 4)), log_scale=0.5 * jr.normal(k[1], (M, 3)),
                 centers=jr.normal(k[2], (M, 3)))

def test_backward_finite_through_chart():
    M = 4
    iso = Splat(quat=jnp.tile(jnp.array([1., 0., 0., 0.]), (M, 1)),
                log_scale=jnp.full((M, 3), 0.3), centers=jr.normal(jr.PRNGKey(1), (M, 3)))
    near = Splat(quat=jr.normal(jr.PRNGKey(2), (M, 4)),
                 log_scale=jnp.full((M, 3), 0.3) + 1e-6 * jr.normal(jr.PRNGKey(3), (M, 3)),
                 centers=jr.normal(jr.PRNGKey(4), (M, 3)))

    def scalar(ss):
        return jnp.sum(full.overlap_matrix(ss)) + jnp.sum(full.kinetic_matrix(ss))

    for ss in (iso, near):                                  # isotropic + near-degenerate: no eigh-NaN
        g = jax.grad(scalar)(ss)
        assert bool(jnp.all(jnp.isfinite(g.quat)) & jnp.all(jnp.isfinite(g.log_scale))
                    & jnp.all(jnp.isfinite(g.centers)))

def test_equivariance(M=5):
    ss = _random_splat(M, jr.PRNGKey(5))
    qQ = jr.normal(jr.PRNGKey(6), (4,)); qQ = qQ / jnp.linalg.norm(qQ)
    Q = quat_to_rotation(qQ)
    ss_rot = Splat(quat=quat_mul(jnp.broadcast_to(qQ, (M, 4)), ss.quat),
                   log_scale=ss.log_scale, centers=ss.centers @ Q.T)
    dA = float(jnp.max(jnp.abs(ss_rot.A - jnp.einsum("ij,mjk,lk->mil", Q, ss.A, Q))))   # A → Q A Qᵀ
    dS = float(jnp.max(jnp.abs(full.overlap_matrix(ss_rot) - full.overlap_matrix(ss))))  # invariant
    assert max(dA, dS) < 1e-9

def test_streaming_coulomb(M=6):
    from gs_dft.coulomb import stream as coulomb_stream
    ss = _random_splat(M, jr.PRNGKey(11))
    Pr = jr.normal(jr.PRNGKey(12), (M, M)); P = 0.5 * (Pr + Pr.T)         # symmetric density matrix
    E_stream = float(coulomb_stream.full_coulomb_energy(ss, P))
    eri = full.compute_eri_tensor(ss)
    E_dense = float(full.coulomb_energy(full.coulomb_matrix(eri, P), P))
    assert abs(E_stream - E_dense) < 1e-9 * max(1.0, abs(E_dense))
