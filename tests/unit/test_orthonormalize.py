"""The regularized-eigensolve Loewdin orthonormalization (``gs_dft/ks/orthonormalize.py``).

- Accuracy: on a well-conditioned Gram matrix it gives the same density as plain ``eigh``.
- Stability: as the smallest eigenvalue reaches zero or turns negative, the forward stays finite.
- Differentiability: the backward stays bounded at a singular or indefinite Gram matrix.
"""
import numpy as np
import jax
import jax.numpy as jnp

from gs_dft.ks.orthonormalize import transform, reg_inv_sqrt

def _eigh_ref(M):
    """Plain unregularized symmetric Löwdin M^{-1/2} — the reference the regularized path must match on
    a healthy Gram."""
    w, U = jnp.linalg.eigh(M)
    return (U * (1.0 / jnp.sqrt(w))) @ U.T

def _random_CS(key, M=40, N=8, cond=1e3):
    """A random tall C (M×N) and SPD metric S (M×M) with controlled cond(S)."""
    kc, ks = jax.random.split(key)
    C = jax.random.normal(kc, (M, N))
    Q, _ = jnp.linalg.qr(jax.random.normal(ks, (M, M)))
    ev = jnp.logspace(0, -np.log10(cond), M)
    S = (Q * ev) @ Q.T
    return C, 0.5 * (S + S.T)

def test_accuracy_healthy_matches_eigh():
    """On a healthy real Gram the floor is inert: ĈᵀSĈ = I and density = eigh's, to ~machine eps."""
    C, S = _random_CS(jax.random.PRNGKey(0))
    M = C.T @ S @ C
    Ce = C @ _eigh_ref(M)
    Cr = C @ transform(M)                                      # default floor_rel=1e-4, inert here
    I = np.asarray(Cr.T @ S @ Cr)
    assert np.allclose(I, np.eye(I.shape[0]), atol=1e-9)
    assert np.allclose(np.asarray(Ce @ Ce.T), np.asarray(Cr @ Cr.T), atol=1e-8)

def _spd(N=8, lam_min=1e-14, seed=3):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    lam = np.concatenate([[lam_min], np.sort(rng.uniform(0.1, 1.0, N - 1))])
    return jnp.asarray(Q @ np.diag(lam) @ Q.T)

def _indefinite(N=8, lam_min=-1.6, seed=4):
    """A Gram with a NEGATIVE eigenvalue — the screened-overlap case that breaks Cholesky/shift."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    lam = np.concatenate([[lam_min], np.sort(rng.uniform(0.1, 1.0, N - 1))])
    return jnp.asarray(Q @ np.diag(lam) @ Q.T)

def test_forward_finite_and_bounded_including_indefinite():
    for M in (_spd(lam_min=1e-14), _spd(lam_min=1e-8), _indefinite(lam_min=-1.6)):
        T = np.asarray(transform(M))                          # default floor_rel=1e-4
        assert np.all(np.isfinite(T))
        assert np.linalg.norm(T) < 1e3                        # floored ⇒ ‖T‖ ~ 1/√(1e-4·λmax)

def test_grad_matches_eigh_where_wellconditioned():
    M = _spd(lam_min=0.05, seed=7)                            # cond ~20: eigh backward is fine here
    g_ref = np.asarray(jax.grad(lambda Mx: jnp.sum(_eigh_ref(Mx) ** 2))(M))
    g_m = np.asarray(jax.grad(lambda Mx: jnp.sum(reg_inv_sqrt(Mx, 1e-12, 1e-4, True) ** 2))(M))
    sym = lambda G: 0.5 * (G + G.T)
    assert np.allclose(sym(g_ref), sym(g_m), atol=1e-6)

def test_grad_bounded_at_singular_and_indefinite_gram():
    """The point: regeigh's backward stays FINITE and bounded where eigh's NaNs."""
    for M in (_spd(lam_min=1e-13, seed=11), _indefinite(lam_min=-1.6, seed=13)):
        g = np.asarray(jax.grad(lambda Mx: jnp.sum(transform(Mx) ** 2))(M))   # default floor 1e-4
        assert np.all(np.isfinite(g))

def test_discard_flag_controls_nullmode_gradient():
    M = _spd(lam_min=1e-12, seed=5)
    g_keep = np.asarray(jax.grad(lambda Mx: jnp.sum(reg_inv_sqrt(Mx, 1e-4, 1e-4, False) ** 2))(M))
    g_disc = np.asarray(jax.grad(lambda Mx: jnp.sum(reg_inv_sqrt(Mx, 1e-4, 1e-4, True) ** 2))(M))
    assert np.all(np.isfinite(g_disc))
    assert np.linalg.norm(g_disc) <= np.linalg.norm(g_keep) + 1e-8

def test_kinetic_energy_bounded_by_floor():
    """E_kin = occ·Tr(M⁻¹ M_T) diverges as λmin(M)→0; the regeigh floor bounds it (the exp12 core)."""
    N, occ = 8, 2.0
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    M_T = jnp.asarray(Q @ np.diag(rng.uniform(1, 5, N)) @ Q.T)
    for lam in (1e-2, 1e-6, 1e-10, 1e-14):
        M = jnp.asarray(Q @ np.diag(np.concatenate([[lam], np.sort(rng.uniform(0.1, 1, N - 1))])) @ Q.T)
        Th = transform(M)                                     # M^{-1/2}_reg (default floor 1e-4)
        Minv_reg = Th @ Th                                    # (M^{-1/2})² = regularized M^{-1}
        ekin = float(occ * jnp.trace(Minv_reg @ M_T))
        assert np.isfinite(ekin) and abs(ekin) < 1e6          # bounded across 12 decades of λmin
