"""The density-fitted Coulomb (``gs_dft/coulomb/ri.py``) with the isotropic pair-product auxiliary.

Checks that ``coulomb_block`` matches the closed form, that the DF energy is a lower bound on the
exact E_J, and that streamed RI-K equals the dense contraction, independent of ``occ_chunk`` and
of screening.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from gs_dft.basis.isotropic import IsotropicSplat
from gs_dft.basis.spectral import Splat
from gs_dft.coulomb import stream as cs
from gs_dft.coulomb import ri as df

def _full(M, ka, kb, scale=0.3):
    """Full-covariance PRIMARY splat (exposes .A for the 'splat' kernels)."""
    return Splat(quat=jr.normal(jr.fold_in(ka, 7), (M, 4)),      # random orientation (R normalizes)
                 log_scale=jr.normal(ka, (M, 3)) * scale,         # anisotropic precision eigenvalues
                 centers=jr.normal(kb, (M, 3)) * 1.5)

def _iso(M, ka, kb, scale=0.4):
    """Isotropic AUXILIARY splat (the production DF aux)."""
    return IsotropicSplat(log_alpha=jr.normal(ka, (M,)) * scale,
                          centers=jr.normal(kb, (M, 3)) * 1.5)

def _sym_P(M, key):
    C = jr.normal(key, (M, max(2, M // 3)))
    return C @ C.T

def _EJ_exact(s, P):
    return float(cs.coulomb_energy_fused(cs.full_eri_block_schur, int(s.n_basis), True, s, P))

def test_coulomb_block_matches_iso_closed_form():
    """The aux metric block df.coulomb_block == the exact iso (ss|ss) Boys closed form (aux is iso)."""
    from dftax.energy.boys import boys
    s = _iso(8, jr.PRNGKey(0), jr.PRNGKey(1))
    pi, pj, _ = cs.make_pairs(8, True)
    a = jnp.array([0, 4, 9, 12, 2])
    b = jnp.array([1, 4, 0, 12, 7])
    ga, Pa, KNa = cs._iso_pair_quantities(cs.iso_quantities(s), pi[a], pj[a])
    gb, Pb, KNb = cs._iso_pair_quantities(cs.iso_quantities(s), pi[b], pj[b])
    # exact iso (ss|ss): 2 π^{5/2}/(γ_a γ_b √(γ_a+γ_b)) · F0(ρ |P_a−P_b|²), ρ = γ_aγ_b/(γ_a+γ_b)
    gab = ga[:, None] * gb[None, :]
    gsum = ga[:, None] + gb[None, :]
    PQ2 = jnp.sum((Pa[:, None, :] - Pb[None, :, :]) ** 2, axis=-1)
    ref = ((KNa[:, None] * KNb[None, :]) * 2.0 * jnp.pi ** 2.5
           / (gab * jnp.sqrt(gsum)) * boys(0, (gab / gsum) * PQ2))
    got = df.coulomb_block(ga, Pa, KNa, gb, Pb, KNb)
    np.testing.assert_allclose(np.array(got), np.array(ref), atol=1e-12)

def test_df_is_lower_bound():
    """E_J^DF = ‖𝒫_aux ρ‖² ≤ ½(ρ|ρ) = E_J for ANY aux (projection)."""
    M = 10
    s = _full(M, jr.PRNGKey(2), jr.PRNGKey(3))
    P = _sym_P(M, jr.PRNGKey(4))
    EJ = _EJ_exact(s, P)
    aux = _iso(20, jr.PRNGKey(5), jr.PRNGKey(6))
    EJ_df = float(df.df_coulomb_energy(s, P, aux, lam=1e-10))
    assert EJ_df <= EJ + 1e-7, f"DF not a lower bound: {EJ_df} > {EJ}"
    assert EJ_df > 0.0

def test_df_exchange_occ_chunk_invariant():
    """The streamed RI-K M-build (occ_chunk = the O(N²)-memory knob) is bit-exact across occ_chunk
    and equals the screened path on the full pair set; gradient stays finite."""
    M, nocc = 10, 5
    s = _full(M, jr.PRNGKey(0), jr.PRNGKey(1))
    aux = _iso(M, jr.PRNGKey(2), jr.PRNGKey(3))
    C = jr.normal(jr.PRNGKey(4), (M, nocc))
    ref = float(df.df_exchange_energy(s, C, aux, occ_chunk=nocc))
    for oc in (1, 2, 3):
        np.testing.assert_allclose(float(df.df_exchange_energy(s, C, aux, occ_chunk=oc)), ref,
                                   atol=1e-9, rtol=1e-9)
    ii, jj = jnp.meshgrid(jnp.arange(M), jnp.arange(M), indexing="ij")
    pi, pj = ii.reshape(-1), jj.reshape(-1)
    np.testing.assert_allclose(
        float(df.df_exchange_energy_screened(s, C, aux, pi, pj, occ_chunk=2)), ref,
        atol=1e-9, rtol=1e-9)
    g = jax.grad(lambda C: df.df_exchange_energy(s, C, aux, occ_chunk=2))(C)
    assert bool(jnp.all(jnp.isfinite(g)))

def test_df_exchange_streaming_equals_dense():
    """Streamed RI-K (O(M²) mem) == the dense (N_aux,M,M) build_three_center path, exactly and
    chunk-independently — value AND gradient w.r.t. C_occ. Guards the exp8 OOM fix."""
    M, nocc = 9, 3
    s = _full(M, jr.PRNGKey(0), jr.PRNGKey(1))
    C = jr.normal(jr.PRNGKey(2), (M, nocc))
    aux = _iso(15, jr.PRNGKey(5), jr.PRNGKey(6))

    def dense_EK(C_):
        T = df.build_three_center(aux, s)                     # (N_aux, M, M)
        D = jnp.einsum("Aik,io,kp->Aop", T, C_, C_)
        Mmat = jnp.einsum("Aop,Bop->AB", D, D)
        V = df.aux_metric(aux)
        return -jnp.trace(jnp.linalg.solve(V + 1e-8 * jnp.eye(V.shape[0]), Mmat))

    e_dense = float(dense_EK(C))
    for ch in (1, 4, M):
        np.testing.assert_allclose(float(df.df_exchange_energy(s, C, aux, chunk=ch)), e_dense,
                                   atol=1e-10, rtol=1e-10)
    g_dense = jax.grad(dense_EK)(C)
    g_stream = jax.grad(lambda C_: df.df_exchange_energy(s, C_, aux, chunk=4))(C)
    np.testing.assert_allclose(g_stream, g_dense, atol=1e-9, rtol=1e-9)

def test_df_coulomb_screened_equals_full_and_screens():
    """Screened E_J^DF over the full ordered pair list == dense df_coulomb_energy; a finite ε drops
    pairs yet stays close."""
    from gs_dft import screening as scr
    M, nocc = 24, 4
    s = _full(M, jr.PRNGKey(0), jr.PRNGKey(1))
    C = jr.normal(jr.PRNGKey(2), (M, nocc))
    occ = jnp.ones(nocc)
    P = C @ jnp.diag(occ) @ C.T
    aux = _iso(16, jr.PRNGKey(5), jr.PRNGKey(6))
    EJ_full = float(df.df_coulomb_energy(s, P, aux))

    pif, pjf, _ = scr.neighbor_pairs(s, eps=-1.0)                 # all ordered pairs
    EJ_scr_full = float(df.df_coulomb_energy_screened(s, C, occ, aux, pif, pjf))
    np.testing.assert_allclose(EJ_scr_full, EJ_full, rtol=1e-9, atol=1e-9)

    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-8)                   # screened
    assert pi.shape[0] < M * M
    EJ_scr = float(df.df_coulomb_energy_screened(s, C, occ, aux, pi, pj))
    assert abs(EJ_scr - EJ_full) < 1e-2 * abs(EJ_full)

def test_df_exchange_screened_equals_full():
    """Screened RI-K over the full ordered pair list == df_exchange_energy (dense bra-pairs)."""
    from gs_dft import screening as scr
    M, nocc = 20, 3
    s = _full(M, jr.PRNGKey(0), jr.PRNGKey(1))
    C = jr.normal(jr.PRNGKey(2), (M, nocc))
    aux = _iso(15, jr.PRNGKey(5), jr.PRNGKey(6))
    EK_full = float(df.df_exchange_energy(s, C, aux))
    pif, pjf, _ = scr.neighbor_pairs(s, eps=-1.0)
    EK_scr = float(df.df_exchange_energy_screened(s, C, aux, pif, pjf))
    np.testing.assert_allclose(EK_scr, EK_full, rtol=1e-9, atol=1e-9)
    g = jax.grad(lambda Cc: df.df_exchange_energy_screened(s, Cc, aux, pif, pjf))(C)
    assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
#  The aux metric is built a row block at a time — V is N_aux², the transient must not be 3×
# ---------------------------------------------------------------------------

def _aux_sets(A, B, seed=0):
    ks = jr.split(jr.PRNGKey(seed), 6)
    return ((jnp.exp(jr.normal(ks[0], (A,))) + 0.05, jr.normal(ks[1], (A, 3)) * 3.0,
             jr.normal(ks[2], (A,))),
            (jnp.exp(jr.normal(ks[3], (B,))) + 0.05, jr.normal(ks[4], (B, 3)) * 3.0,
             jr.normal(ks[5], (B,))))


@pytest.mark.parametrize("A", [1, 7, 64, 65, 257, 1000])
@pytest.mark.parametrize("omega", [None, 0.3])
def test_blocked_metric_matches_the_single_shot_build(A, omega, monkeypatch):
    """Blocking must not change V. Includes A not divisible by the block size, where the last
    block is CLAMPED to start at A-c and rewrites rows an earlier block already wrote."""
    B = 257
    (g1, P1, N1), (g2, P2, N2) = _aux_sets(A, B, seed=A)
    ref = df._coulomb_rows(g1, P1, N1, g2, P2, N2, omega)
    for budget in (3 * B, 3 * B * 7, 3 * B * 64):               # c = 1, 7, 64 rows per block
        monkeypatch.setattr(df, "_AUX_BLOCK_BUDGET", budget)
        got = df.coulomb_block(g1, P1, N1, g2, P2, N2, omega=omega)
        assert np.allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-11)


def test_the_metric_never_forms_the_ABx3_centre_difference(monkeypatch):
    """The single-shot build holds ~13 copies of V — seven (A,B) terms plus the (A,B,3) centre
    difference `PQ2` reduces. Correctness cannot see this, since the values are identical either
    way, so assert on the compiled temp instead."""
    A = B = 512
    (g1, P1, N1), (g2, P2, N2) = _aux_sets(A, B, seed=5)
    monkeypatch.setattr(df, "_AUX_BLOCK_BUDGET", 3 * B * 64)    # c = 64 rows per block

    def temp_bytes(fn):
        return (jax.jit(fn).lower(g1, P1, N1, g2, P2, N2)
                .compile().memory_analysis().temp_size_in_bytes)

    direct = temp_bytes(lambda *a: df._coulomb_rows(*a, None))
    blocked = temp_bytes(lambda *a: df.coulomb_block(*a))       # default budget blocks at this size
    assert df._block_rows(B) < A, "the test size no longer reaches the blocked path"
    assert blocked < direct / 2, (
        f"blocked build holds {blocked} temp bytes vs {direct} for the single shot — the (A,B,3) "
        f"difference is back")
