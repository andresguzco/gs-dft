"""Memory-efficient streaming Coulomb (J / E_J) with an exact ``custom_vjp``.

The splat KS energy only needs the Hartree energy ``E_J = ½ Σ (pq|rs) P_pq P_rs``
(and, for a Fock build, ``J_pq = Σ (pq|rs) P_rs``) — both O(M²). But letting reverse-
mode autodiff differentiate through the 4-index ERI stores the O(M⁴) tape and OOMs
(the full (M²,M²) / (M,M,M,M) arrays at moderate M). This
module computes ``E_J`` by **streaming the ERI in bra-pair chunks** (each chunk a
``(chunk, n_pair)`` block — already O(M²)) and supplies an **explicit, exact gradient**
via ``jax.custom_vjp`` so neither pass holds the M⁴ object or its tape:

    forward:  Jvec_a = Σ_b (a|b) Pw_b              (chunked over bra pairs a)
              E_J    = ½ Σ_a Jvec_a Pw_a
    backward: dE_J/dθ = ½ Σ_ab ∂(a|b)/∂θ Pw_a Pw_b (same chunked stream; per-chunk
              jax.vjp of the block kernel — exact to machine precision)
              dE_J/dP = J

Symmetry: with ``unique=True`` only the M(M+1)/2 pairs i≤j are enumerated, with weight
w=2 for i<j and 1 for i=j (Pw_b ≡ w_b·P[i,j]); the identity Σ_all = Σ_unique w·(·) makes
this *exact* and ~4× cheaper. ``Jvec_a = J[i_a,j_a]`` either way. The per-pair piece is the
full-covariance block ``eri_block_fn(splats, bra_i, bra_j, ket_i, ket_j) -> (A,B)``
(the 3×3-Schur Laplace-quadrature kernel below).
"""

import functools
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from gs_dft.integrals.dense import _QUAD_T, _QUAD_W   # Laplace half-line quadrature
from gs_dft.integrals.linalg3 import det3, inv3, solve3

__all__ = ["make_pairs", "coulomb_energy_fused", "coulomb_J_fused",
           "full_eri_block", "full_eri_block_schur", "full_eri_block_fused", "full_coulomb_energy",
           "iso_quantities", "_iso_pair_quantities", "full_quantities",
           "screened_coulomb_energy"]

_I3 = np.eye(3)          # numpy: importing the package must not initialise the JAX backend


# ---------------------------------------------------------------------------
#  Pair enumeration (+ symmetry multiplicities)
# ---------------------------------------------------------------------------

def make_pairs(M: int, unique: bool = True):
    """(i, j, w) over basis-function pairs.

    unique=False: all M² ordered pairs, w=1 (no symmetry — validation/reference).
    unique=True : the M(M+1)/2 pairs i≤j, w=2 for i<j and 1 for i=j.
    """
    if not unique:
        i = np.repeat(np.arange(M), M)
        j = np.tile(np.arange(M), M)
        w = np.ones(M * M)
    else:
        i, j = np.triu_indices(M)
        w = np.where(i == j, 1.0, 2.0)
    return (jnp.asarray(i), jnp.asarray(j), jnp.asarray(w, dtype=jnp.float64))


# ---------------------------------------------------------------------------
#  Per-splat quantities (precomputed ONCE per energy call — see note below)
# ---------------------------------------------------------------------------
#  The pair-quantity helpers take a precomputed per-splat tuple ``q`` rather
#  than the splat module: reading ``splats.A`` runs the M closed-form A-chart
#  builds (quaternion + log-eigenvalues → precision), and the helpers are called
#  inside remat'd scan bodies (DF γ/D streams, the custom_vjp backward) — module
#  access there recomputes the A build per chunk per pass. Callers build
#  ``q = *_quantities(splats)`` once, outside the scan, so the chart map (and its
#  tape) exists once.

def iso_quantities(splats):
    """(α, centers, norm) for an isotropic splat (cheap field reads)."""
    return splats.alpha, splats.centers, splats.norm


def full_quantities(splats):
    """(A, centers, norm) for the full-cov splat — ONE closed-form A build over M."""
    return splats.A, splats.centers, splats.norm


# ---------------------------------------------------------------------------
#  Isotropic (ss) pair quantities — a helper only (no iso block kernel here;
#  the streamed Coulomb runs the full-covariance Schur kernel below).
# ---------------------------------------------------------------------------

def _iso_pair_quantities(q, pi, pj):
    """Per-pair γ, Gaussian-product center P, and N_i N_j K_ij for pairs (pi,pj)."""
    a, mu, N = q                           # (M,), (M,3), (M,)
    ai, aj = a[pi], a[pj]                  # (P,)
    mui, muj = mu[pi], mu[pj]              # (P, 3)
    g = ai + aj
    p = ai * aj / g
    d2 = jnp.sum((mui - muj) ** 2, axis=-1)
    K = jnp.exp(-p * d2)
    Pc = (ai[:, None] * mui + aj[:, None] * muj) / g[:, None]
    return g, Pc, N[pi] * N[pj] * K


# ---------------------------------------------------------------------------
#  Streaming helpers
# ---------------------------------------------------------------------------

def _chunked_pairs(pi, pj, bra_chunk):
    """Pad+reshape the bra pair lists to (n_chunks, bra_chunk); return (pis, pjs, n)."""
    n = pi.shape[0]
    pad = (-n) % bra_chunk
    pis = jnp.concatenate([pi, jnp.zeros(pad, pi.dtype)]).reshape(-1, bra_chunk)
    pjs = jnp.concatenate([pj, jnp.zeros(pad, pj.dtype)]).reshape(-1, bra_chunk)
    return pis, pjs, n


def _jvec_stream(eri_block_fn, splats, pairs, Pw, bra_chunk):
    """Jvec_a = Σ_b (a|b) Pw_b, streamed over bra-pair chunks (peak = one (chunk,n_pair))."""
    pi, pj, _ = pairs
    pis, pjs, n = _chunked_pairs(pi, pj, bra_chunk)

    def body(c):
        block = eri_block_fn(splats, c[0], c[1], pi, pj)       # (chunk, n_pair)
        return block @ Pw                                      # (chunk,)

    jv = lax.map(body, (pis, pjs))                             # (n_chunks, chunk)
    return jv.reshape(-1)[:n]


def _grad_P(Jvec, pi, pj, M, unique):
    """∂E_J/∂P = J. Jvec_a = J[i_a,j_a]; scatter into the full M×M matrix."""
    if unique:
        Jm = jnp.zeros((M, M)).at[pi, pj].add(Jvec)
        Jm = Jm.at[pj, pi].add(jnp.where(pi == pj, 0.0, Jvec))
        return Jm
    return Jvec.reshape(M, M)                                  # all-ordered: one per (i,j)


# ---------------------------------------------------------------------------
#  E_J with exact streaming custom_vjp
# ---------------------------------------------------------------------------

# NB: custom_vjp forbids array-valued nondiff_argnums under jit, so we pass the
# static int M (+ unique flag) and build the pair lists *inside* the primitive
# (host np.triu_indices → trace-time constants). Only splats and P are differentiated.
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 5))
def coulomb_energy_fused(eri_block_fn, M, unique, splats, P, bra_chunk=128):
    """E_J = ½ Σ_pqrs (pq|rs) P_pq P_rs, streamed; exact gradient via custom_vjp."""
    pairs = make_pairs(M, unique)
    pi, pj, w = pairs
    Pw = P[pi, pj] * w
    Jvec = _jvec_stream(eri_block_fn, splats, pairs, Pw, bra_chunk)
    return 0.5 * jnp.sum(Jvec * Pw)


def _cef_fwd(eri_block_fn, M, unique, splats, P, bra_chunk):
    e = coulomb_energy_fused(eri_block_fn, M, unique, splats, P, bra_chunk)
    return e, (splats, P)


def _cef_bwd(eri_block_fn, M, unique, bra_chunk, res, g):
    splats, P = res
    pairs = make_pairs(M, unique)
    pi, pj, w = pairs
    Pw = P[pi, pj] * w

    # dE_J/dP = J
    Jvec = _jvec_stream(eri_block_fn, splats, pairs, Pw, bra_chunk)
    grad_P = g * _grad_P(Jvec, pi, pj, M, unique)

    # dE_J/dθ = ½ Σ_ab ∂(a|b)/∂θ Pw_a Pw_b, streamed; per-chunk jax.vjp of the block.
    pis, pjs, n = _chunked_pairs(pi, pj, bra_chunk)
    pad = pis.size - n
    Pwa = jnp.concatenate([Pw, jnp.zeros(pad)]).reshape(-1, bra_chunk)   # padded bra Pw
    zero = jax.tree.map(jnp.zeros_like, splats)

    def scan_body(acc, carry):
        bi, bj, pwa = carry
        coeff = (0.5 * g) * pwa[:, None] * Pw[None, :]         # (chunk, n_pair)
        _, vjp = jax.vjp(lambda s: eri_block_fn(s, bi, bj, pi, pj), splats)
        (gs,) = vjp(coeff)
        return jax.tree.map(lambda a, b: a + b, acc, gs), None

    grad_splats, _ = lax.scan(scan_body, zero, (pis, pjs, Pwa))
    return (grad_splats, grad_P)


coulomb_energy_fused.defvjp(_cef_fwd, _cef_bwd)


def coulomb_J_fused(eri_block_fn, M, unique, splats, P, bra_chunk=128):
    """Hartree matrix J_pq = Σ_rs (pq|rs) P_rs (for Fock builds / diagnostics)."""
    pairs = make_pairs(M, unique)
    pi, pj, w = pairs
    Pw = P[pi, pj] * w
    Jvec = _jvec_stream(eri_block_fn, splats, pairs, Pw, bra_chunk)
    return _grad_P(Jvec, pi, pj, M, unique)


# ---------------------------------------------------------------------------
#  Full-covariance block kernel — 6×6 block-precision Laplace quadrature
#  (chart-agnostic: works on any splat exposing .A/.centers/.norm/.n_basis)
# ---------------------------------------------------------------------------

def _full_pair_quantities(q, pi, pj):
    """Product-Gaussian precision Â_a, center P_a, and N_iN_jK_a for pairs (pi,pj)."""
    A, mu, N = q                                   # (M,3,3), (M,3), (M,)
    Ai, Aj = A[pi], A[pj]                          # (P,3,3)
    mui, muj = mu[pi], mu[pj]                      # (P,3)
    Aij = Ai + Aj                                  # (P,3,3) product precision
    rhs = jnp.einsum("pkl,pl->pk", Ai, mui) + jnp.einsum("pkl,pl->pk", Aj, muj)
    Pa = solve3(Aij, rhs)                          # (P,3) product center
    d = mui - muj
    Ajd = jnp.einsum("pkl,pl->pk", Aj, d)
    AiAinvAjd = jnp.einsum("pkl,pl->pk", Ai, solve3(Aij, Ajd))
    K = jnp.exp(-jnp.sum(d * AiAinvAjd, axis=-1))  # (P,)
    return Aij, Pa, N[pi] * N[pj] * K


def full_eri_block(splats, bra_i, bra_j, ket_i, ket_j):
    """(a|b) block for full-cov splats — 6×6 block-precision Laplace, q scanned."""
    q = full_quantities(splats)
    Aa, Pa, KNa = _full_pair_quantities(q, bra_i, bra_j)        # (A,3,3),(A,3),(A,)
    Ab, Pb, KNb = _full_pair_quantities(q, ket_i, ket_j)        # (B,...)
    A_, B_ = Aa.shape[0], Ab.shape[0]
    Da = jnp.broadcast_to(Aa[:, None], (A_, B_, 3, 3))          # block-diag D top-left
    Db = jnp.broadcast_to(Ab[None, :], (A_, B_, 3, 3))          # bottom-right
    c = jnp.concatenate([jnp.broadcast_to(Pa[:, None], (A_, B_, 3)),
                         jnp.broadcast_to(Pb[None, :], (A_, B_, 3))], axis=-1)   # (A,B,6)
    v = jnp.concatenate([jnp.einsum("abkl,abl->abk", Da, c[..., :3]),
                         jnp.einsum("abkl,abl->abk", Db, c[..., 3:])], axis=-1)  # (A,B,6)
    cv = jnp.sum(c * v, axis=-1)                                # cᵀDc (A,B), q-invariant

    def node(coul, tw):
        t, w = tw
        t2 = t * t
        Xa = jnp.broadcast_to((Aa + t2 * _I3)[:, None], (A_, B_, 3, 3))
        Xb = jnp.broadcast_to((Ab + t2 * _I3)[None, :], (A_, B_, 3, 3))
        OFF = jnp.broadcast_to(-t2 * _I3, (A_, B_, 3, 3))
        G = jnp.concatenate([jnp.concatenate([Xa, OFF], axis=-1),
                             jnp.concatenate([OFF, Xb], axis=-1)], axis=-2)   # (A,B,6,6)
        Ginv_v = jnp.linalg.solve(G, v[..., None])[..., 0]                    # (A,B,6)
        quad = cv - jnp.sum(v * Ginv_v, axis=-1)                             # (A,B)
        I_t = jnp.pi ** 3 / jnp.sqrt(jnp.linalg.det(G)) * jnp.exp(-quad)
        return coul + w * (2.0 / jnp.sqrt(jnp.pi)) * I_t, None

    coulomb, _ = lax.scan(node, jnp.zeros((A_, B_)), (_QUAD_T, _QUAD_W))
    return (KNa[:, None] * KNb[None, :]) * coulomb              # (A,B)


def full_eri_block_schur(splats, bra_i, bra_j, ket_i, ket_j):
    """Full-cov (a|b) block via the Schur complement of the 6×6 block precision.

    Exact algebraic reduction of ``full_eri_block``: with G=[[X,-t²I],[-t²I,Y]],
    S = Y - t⁴X⁻¹, then det G = det(X)det(S) and vᵀG⁻¹v = v_aᵀu + wᵀS⁻¹w with
    u=X⁻¹v_a, w=t²u+v_b. The a-only pieces (X⁻¹, det X, u) cost A (not A×B), and the
    A×B coupling is only the 3×3 S/solve — vs the (A,B,6,6) buffer + 6×6 solve.
    """
    q = full_quantities(splats)
    Aa, Pa, KNa = _full_pair_quantities(q, bra_i, bra_j)        # (A,3,3),(A,3),(A,)
    Ab, Pb, KNb = _full_pair_quantities(q, ket_i, ket_j)
    va = jnp.einsum("akl,al->ak", Aa, Pa)                       # Â_a P_a  (A,3)
    vb = jnp.einsum("bkl,bl->bk", Ab, Pb)                       # (B,3)
    cv = jnp.sum(Pa * va, -1)[:, None] + jnp.sum(Pb * vb, -1)[None, :]   # cᵀDc (A,B)

    def node(coul, tw):
        t, w_q = tw
        t2 = t * t
        X = Aa + t2 * _I3                                       # (A,3,3) — a only
        Xinv = inv3(X)
        u = jnp.einsum("akl,al->ak", Xinv, va)                  # X⁻¹ v_a (A,3)
        vau = jnp.sum(va * u, -1)                               # v_aᵀu (A,)
        Y = Ab + t2 * _I3                                       # (B,3,3) — b only
        S = Y[None, :] - t2 * t2 * Xinv[:, None]                # (A,B,3,3)
        w = t2 * u[:, None, :] + vb[None, :, :]                 # (A,B,3)
        z = solve3(S, w)                                       # S⁻¹w (A,B,3)
        quad = cv - (vau[:, None] + jnp.sum(w * z, -1))        # (A,B)
        detG = det3(X)[:, None] * det3(S)                      # (A,B)
        I_t = jnp.pi ** 3 / jnp.sqrt(detG) * jnp.exp(-quad)
        return coul + w_q * (2.0 / jnp.sqrt(jnp.pi)) * I_t, None

    coulomb, _ = lax.scan(node, jnp.zeros((Aa.shape[0], Ab.shape[0])), (_QUAD_T, _QUAD_W))
    return (KNa[:, None] * KNb[None, :]) * coulomb


def full_coulomb_energy(splats, P, bra_chunk=64, unique=True):
    """Streaming, O(M²)-memory, exact E_J for the full-covariance splats —
    the 3×3 Schur kernel through the fused custom-vjp stream."""
    M = int(splats.n_basis)
    return coulomb_energy_fused(full_eri_block_schur, M, unique, splats, P, bra_chunk)


# ---------------------------------------------------------------------------
#  Screened EXACT E_J — streaming ERI over the significant overlap pairs only.
#  The density-fitting-FREE Coulomb: O(M²) compute, O(bra_chunk·n_pair) memory.
# ---------------------------------------------------------------------------

def screened_coulomb_energy(splats, C_orth, occ, pi, pj, valid=None, bra_chunk=128):
    """EXACT Hartree energy ``E_J = ½ Σ_{a,b ∈ sig pairs} (a|b) P_a P_b`` over the significant overlap
    pairs — the screened, density-fitting-FREE streaming analog of ``coulomb_energy_fused``.

    ``neighbor_pairs`` returns ALL ordered significant pairs (incl. the diagonal), so the symmetry
    weight is 1 and ``P_a = P[i_a,j_a] = Σ_o occ_o C_{i_a o} C_{j_a o}`` (no dense M×M P). The bra
    pairs are streamed in chunks against the full significant ket list; ``jax.checkpoint`` on the
    chunk body bounds the backward to O(bra_chunk·n_pair) memory. Total compute is O(n_pair²)=O(M²)
    (the long-range 1/r forbids ket screening — only the significant pair COUNT, O(M²)→O(M), drops).

    Equals the dense exact ``E_J`` when (pi,pj) is the full ordered set (``neighbor_pairs`` with
    ``eps<0``); ε-controlled (overlap screening) otherwise. ``valid`` zeroes padded pairs.
    """
    block_fn = full_eri_block_schur
    Pw = jnp.sum(occ[None, :] * C_orth[pi] * C_orth[pj], axis=1)        # P_ij on the pairs (n_pair,)
    if valid is not None:
        Pw = valid.astype(Pw.dtype) * Pw           # 0 on padding; 2−δ_ij weights on a unique list
    pis, pjs, n = _chunked_pairs(pi, pj, bra_chunk)

    def body(c):
        block = block_fn(splats, c[0], c[1], pi, pj)                    # (bra_chunk, n_pair)
        return block @ Pw                                              # Jvec chunk (bra_chunk,)

    jv = lax.map(jax.checkpoint(body), (pis, pjs)).reshape(-1)[:n]      # Jvec on pairs (n_pair,)
    return 0.5 * jnp.sum(jv * Pw)


# ---------------------------------------------------------------------------
#  Fused (pair|pair) kernel: six scalars per pair-pair, then a scalar node loop.
#
#  For product Gaussians with precisions Â_a, Â_b and centres P_a, P_b, with C = (Â_a⁻¹ + Â_b⁻¹)⁻¹
#  and d = P_a − P_b,
#      (a|b) = KN_a KN_b π³ (2/√π) det(Â_a+Â_b)^{-1/2} ∫₀^∞ dt  exp(−t² N(t²)/D(t²)) / √D(t²)
#      D(s) = s³ + I₁ s² + I₂ s + I₃          (I_k = invariants of C; D = det(C + sI))
#      N(s) = q₀ + q₁ s + q₂ s²               (q₂ = dᵀCd, q₁ = I₁q₂ − |Cd|², q₀ = I₃|d|²)
#  using adj(C + sI) = adj C + s(I₁ I − C) + s² I. Nothing (A,B,3,3) is ever formed, so the whole
#  block is one elementwise fusion. The Laplace variable is rescaled per pair-pair by
#  ρ̄ = I₃^{1/3}/(1 + q₂) (t² = ρ̄ u²/(1−u²)), which keeps the u-integrand O(1)-wide across exponent
#  ratios AND pair separations, so 32 nodes reach ~1e-7 at
#  realistic anisotropy (48: 1e-9); extreme aspect ratios (e^9) converge only algebraically.
# ---------------------------------------------------------------------------

_N_FUSED_NODES = 32     # measured: 2e-8 (isotropic, exact Boys) / 4e-7 (aspect ~e^1.5) at 32; 2e-9 at 48


def _boys_nodes(n):
    x, w = np.polynomial.legendre.leggauss(n)
    u = (x + 1.0) / 2.0
    wu = w / 2.0
    c = u * u / (1.0 - u * u)                       # t² / ρ̄
    e = wu * (1.0 - u * u) ** -1.5                  # dt / (√ρ̄ du)
    return tuple(float(v) for v in c), tuple(float(v) for v in e)


@functools.lru_cache(maxsize=8)
def _make_pairpair_nodes(n):
    """The node sum for an ``n``-point Boys-scaled rule, with its analytic reverse (cached so the
    custom_vjp object is stable across traces)."""
    cs_, es_ = _boys_nodes(n)

    @jax.custom_vjp
    def nodes(I1, I2, I3, q0, q1, q2, rho):
        """Σ_k e_k exp(−t²N/D)/√D at t² = ρ̄ c_k."""
        acc = jnp.zeros_like(I1)
        for c, e in zip(cs_, es_):
            s = rho * c
            D = ((s + I1) * s + I2) * s + I3
            expo = s * (q0 + s * (q1 + s * q2)) / D
            acc = acc + e * jnp.exp(-expo) / jnp.sqrt(D)
        return acc

    def fwd(I1, I2, I3, q0, q1, q2, rho):
        return nodes(I1, I2, I3, q0, q1, q2, rho), (I1, I2, I3, q0, q1, q2, rho)

    def bwd(res, g):
        I1, I2, I3, q0, q1, q2, rho = res
        dI1 = dI2 = dI3 = dq0 = dq1 = dq2 = drho = 0.0
        for c, e in zip(cs_, es_):
            s = rho * c
            D = ((s + I1) * s + I2) * s + I3
            invD = 1.0 / D
            N = q0 + s * (q1 + s * q2)
            expo = s * N * invD
            f = g * e * jnp.exp(-expo) * jnp.sqrt(invD)
            fe = -f                                    # ∂f/∂expo
            fD = -0.5 * f * invD                       # ∂f/∂D
            e_N = s * invD                             # ∂expo/∂N
            e_D = -expo * invD                         # ∂expo/∂D
            cD = fe * e_D + fD
            dI1 = dI1 + cD * s * s
            dI2 = dI2 + cD * s
            dI3 = dI3 + cD
            dq0 = dq0 + fe * e_N
            dq1 = dq1 + fe * e_N * s
            dq2 = dq2 + fe * e_N * s * s
            D_s = (3.0 * s + 2.0 * I1) * s + I2
            N_s = q1 + 2.0 * s * q2
            e_s = N * invD + s * N_s * invD + e_D * D_s
            drho = drho + (fe * e_s + fD * D_s) * c
        return (dI1, dI2, dI3, dq0, dq1, dq2, drho)

    nodes.defvjp(fwd, bwd)
    return nodes


def _sym6(A):
    """(…,3,3) symmetric → six (…) components xx, yy, zz, xy, xz, yz."""
    return (A[..., 0, 0], A[..., 1, 1], A[..., 2, 2], A[..., 0, 1], A[..., 0, 2], A[..., 1, 2])


def full_eri_block_fused(splats, bra_i, bra_j, ket_i, ket_j, n_nodes=_N_FUSED_NODES):
    """(a|b) block for full-covariance splats — the fused scalar kernel (see the module note above).
    Equals :func:`full_eri_block_schur` to quadrature accuracy; one elementwise fusion per block."""
    q = full_quantities(splats)
    Aa, Pa, KNa = _full_pair_quantities(q, bra_i, bra_j)
    Ab, Pb, KNb = _full_pair_quantities(q, ket_i, ket_j)
    detAa, detAb = det3(Aa), det3(Ab)
    axx, ayy, azz, axy, axz, ayz = (v[:, None] for v in _sym6(Aa))
    bxx, byy, bzz, bxy, bxz, byz = (v[None, :] for v in _sym6(Ab))
    # S = Â_a + Â_b, its adjugate and determinant (symmetric)
    sxx, syy, szz, sxy, sxz, syz = axx + bxx, ayy + byy, azz + bzz, axy + bxy, axz + bxz, ayz + byz
    Jxx = syy * szz - syz * syz
    Jyy = sxx * szz - sxz * sxz
    Jzz = sxx * syy - sxy * sxy
    Jxy = sxz * syz - sxy * szz
    Jxz = sxy * syz - sxz * syy
    Jyz = sxy * sxz - sxx * syz
    detS = sxx * Jxx + sxy * Jxy + sxz * Jxz
    invS = 1.0 / detS
    # M = adj(S) Â_b (general 3x3), then C = Â_a M / detS (symmetric)
    Mxx = Jxx * bxx + Jxy * bxy + Jxz * bxz
    Mxy = Jxx * bxy + Jxy * byy + Jxz * byz
    Mxz = Jxx * bxz + Jxy * byz + Jxz * bzz
    Myx = Jxy * bxx + Jyy * bxy + Jyz * bxz
    Myy = Jxy * bxy + Jyy * byy + Jyz * byz
    Myz = Jxy * bxz + Jyy * byz + Jyz * bzz
    Mzx = Jxz * bxx + Jyz * bxy + Jzz * bxz
    Mzy = Jxz * bxy + Jyz * byy + Jzz * byz
    Mzz = Jxz * bxz + Jyz * byz + Jzz * bzz
    Cxx = (axx * Mxx + axy * Myx + axz * Mzx) * invS
    Cyy = (axy * Mxy + ayy * Myy + ayz * Mzy) * invS
    Czz = (axz * Mxz + ayz * Myz + azz * Mzz) * invS
    Cxy = (axx * Mxy + axy * Myy + axz * Mzy) * invS
    Cxz = (axx * Mxz + axy * Myz + axz * Mzz) * invS
    Cyz = (axy * Mxz + ayy * Myz + ayz * Mzz) * invS
    I1 = Cxx + Cyy + Czz
    I2 = Cxx * Cyy + Cyy * Czz + Czz * Cxx - Cxy * Cxy - Cxz * Cxz - Cyz * Cyz
    I3 = (detAa[:, None] * detAb[None, :]) * invS
    dx = Pa[:, None, 0] - Pb[None, :, 0]
    dy = Pa[:, None, 1] - Pb[None, :, 1]
    dz = Pa[:, None, 2] - Pb[None, :, 2]
    Cdx = Cxx * dx + Cxy * dy + Cxz * dz
    Cdy = Cxy * dx + Cyy * dy + Cyz * dz
    Cdz = Cxz * dx + Cyz * dy + Czz * dz
    q2 = dx * Cdx + dy * Cdy + dz * Cdz
    q1 = I1 * q2 - (Cdx * Cdx + Cdy * Cdy + Cdz * Cdz)
    q0 = I3 * (dx * dx + dy * dy + dz * dz)
    # scale of the Laplace variable: I3^{1/3} for the exponent spread, /(1+q2) so distant pairs
    # (large asymptotic exponent q2 = dᵀCd) keep an O(1)-wide integrand in u
    rho = jnp.cbrt(I3) / (1.0 + q2)
    acc = _make_pairpair_nodes(int(n_nodes))(I1, I2, I3, q0, q1, q2, rho)
    pref = 2.0 * jnp.pi ** 3 / jnp.sqrt(jnp.pi)
    return (KNa[:, None] * KNb[None, :]) * pref * jnp.sqrt(rho * invS) * acc
