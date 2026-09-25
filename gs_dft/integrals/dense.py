"""Laplace-quadrature integral/grid kernels for the splat (spectral) chart.

Each splat is a general anisotropic, nodeless Gaussian with an SPD precision matrix `A`
(arbitrary orientation, not axis-locked):

    g_i(r) = N_i · exp(-(r - μ_i)ᵀ A_i (r - μ_i)),   N_i = (det(2A_i)/π³)^{1/4}

The covariance chart itself (the spectral quaternion + log-eigenvalue parametrization, eigh/
expm-free, rotation-equivariant) lives in ``spectral.py`` (class ``Splat``). The kernels here are
**chart-agnostic**: they read `.A` / `.centers` / `.norm` / `.n_basis` off whatever splat is passed.

Integrals: overlap and kinetic are closed-form 3×3 matrix expressions; nuclear
attraction and the ERIs use Laplace quadrature (an anisotropic Coulomb has no closed form). The
ERI is **chunked over the quadrature nodes** (`lax.scan`) so the heavy `(M², M²)` intermediate is
built one node at a time. The integrals touch only `A/centers/norm/n_basis`, so the chart is fully
encapsulated in ``spectral.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jaxtyping import Float, Array
from gs_dft.integrals.linalg3 import det3, inv3

__all__ = [
    "overlap_matrix", "kinetic_matrix", "nuclear_attraction_matrix",
    "overlap_times", "kinetic_times", "overlap_times_rows", "kinetic_times_rows",
    "one_electron_integrals", "compute_eri_tensor",
    "coulomb_matrix", "exchange_matrix", "coulomb_energy", "exchange_energy",
    "eval_basis_on_grid", "eval_basis_and_grad_on_grid",
    "eval_density_on_grid", "eval_density_and_grad_on_grid",
    "eval_density_grad_tau_on_grid",
]

# ---------------------------------------------------------------------------
# Laplace half-line quadrature, shared by every ∫₀^∞ Coulomb/nuclear kernel
# (also imported by coulomb.stream / coulomb.ri / screening.pairs).
# HOST (numpy float64) arrays on purpose: a module-level ``jnp.array`` would
# freeze their dtype at import time — float32 if the package were imported
# before ``jax_enable_x64`` flips — silently degrading every Coulomb integral
# to ~1e-7. As numpy they are cast at trace time under the active precision.
# ---------------------------------------------------------------------------

def _gauss_legendre_halfline(n_quad: int = 20):
    """Gauss-Legendre on [0,∞) via t = s/(1-s)."""
    nodes_01, weights_01 = np.polynomial.legendre.leggauss(n_quad)
    s = (nodes_01 + 1.0) / 2.0
    w_s = weights_01 / 2.0
    t = s / (1.0 - s)
    jacobian = 1.0 / (1.0 - s) ** 2
    return np.asarray(t, dtype=np.float64), np.asarray(w_s * jacobian, dtype=np.float64)


def _gauss_legendre_segment(upper: float, n_quad: int = 12):
    """Gauss-Legendre on [0, upper] — the node table for the ERF-ATTENUATED Coulomb kernel.

    The whole of range separation is this change of limit. With
        1/r        = (2/√π) ∫₀^∞ e^{−t²r²} dt
        erf(ωr)/r  = (2/√π) ∫₀^ω e^{−t²r²} dt
    the long-range kernel is the SAME node integral the short-range one uses, integrated to ω
    instead of to infinity. Nothing about the integrand changes, so every kernel that consumes the
    table — the 3-centre (aux|pair) nodes, the pair screening — becomes range-separated by being
    handed a different table.

    12 nodes, not 20. The half-line map needs 20 because it must resolve a t^{-3} tail out to
    infinity; on [0, 0.3] the integrand is smooth and 12 nodes give ~1e-13 (8 give 1e-8), measured
    across exponent ratios from 0.05 to 60. So the long-range pass is CHEAPER than the short-range
    one, not an extra copy of it.
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    t = 0.5 * float(upper) * (x + 1.0)
    return np.asarray(t, dtype=np.float64), np.asarray(0.5 * float(upper) * w, dtype=np.float64)


_QUAD_T, _QUAD_W = _gauss_legendre_halfline(20)

# Python-float copies for the STATIC unrolled node loops (trace-time constants —
# coulomb.ri / screening.pairs unroll the nodes so XLA fuses the loop into one kernel).
_QUAD_T_LIST = [float(t) for t in _QUAD_T]
_QUAD_W_LIST = [float(w) for w in _QUAD_W]

_I3 = np.eye(3)          # numpy: importing the package must not initialise the JAX backend


# ---------------------------------------------------------------------------
# Coulomb/exchange contractions on a precomputed dense ERI tensor
# ---------------------------------------------------------------------------

def coulomb_matrix(eri, P):
    """J_pq = Σ_rs P_rs (pq|rs)."""
    return jnp.einsum("pqrs,rs->pq", eri, P)


def exchange_matrix(eri, P):
    """K_pq = Σ_rs P_rs (pr|qs)."""
    return jnp.einsum("prqs,rs->pq", eri, P)


def coulomb_energy(J, P):
    """E_J = ½ Tr(P J)."""
    return 0.5 * jnp.sum(P * J)


def exchange_energy(K, P):
    """E_K = -¼ Tr(P K)."""
    return -0.25 * jnp.sum(P * K)


# The kernels below are chart-agnostic: they read ``.A`` / ``.centers`` / ``.norm`` /
# ``.n_basis`` off whatever splat is passed (the production ``Splat`` supplies them).


# ---------------------------------------------------------------------------
#  Pairwise product-Gaussian quantities (3×3 matrices)
# ---------------------------------------------------------------------------

# Target element count for one row block's (c, M, 3, 3) pairwise intermediate. The result is a
# benign (M, M), but building it at once costs 2x(M,M,3,3) + 6x(M,M,3). Every (i,j) element is
# independent, so blocking the rows is exact and bounds the intermediate at c*M*9.
_PAIR_BUDGET = 2 ** 26


def _row_chunk(M: int) -> int:
    """Rows per block, so the (c, M, 3, 3) intermediate stays near :data:`_PAIR_BUDGET`."""
    return max(1, min(int(M), _PAIR_BUDGET // max(1, 9 * int(M))))


def _by_rows(row_fn, M: int):
    """``(M, M)`` from a function giving one row block ``rows -> (c, M)``.

    ``lax.map`` over blocks of row INDICES: the padded tail re-does row 0 and is sliced off, which
    keeps every block the same static shape. Only the intermediates are blocked — the arithmetic per
    (i, j) is untouched, so this is bit-identical to the all-at-once form.

    The ``jax.checkpoint`` is required. ``lax.map`` is a scan, so without it reverse mode saves
    every block's ``(c, M, 3, 3)`` intermediate and they stack straight back into the dense
    ``(M, M, 3, 3)`` this function exists to avoid. It is close to free — the dense S/T build is
    ~0.5% of a step, so there is almost nothing to recompute.

    Do not shard these row blocks over the mesh: at 0.5% of the step the ceiling is 0.3% on 4
    devices, before the all-gather where a row-sharded S meets CᵀSC.
    """
    c = _row_chunk(M)
    pad = (-M) % c
    idx = jnp.concatenate([jnp.arange(M), jnp.zeros(pad, dtype=int)]).reshape(-1, c)
    # prevent_cse=False: this sits inside a scan, which already blocks the cross-iteration CSE the
    # flag guards, and leaving it on costs compile time for nothing (as in ks/nlc.py).
    return lax.map(jax.checkpoint(row_fn, prevent_cse=False), idx).reshape(-1, M)[:M]


def _by_rows_times(row_fn, M: int, X):
    """``(M, K)`` = ``A @ X`` from a function giving one row block of ``A`` as ``rows -> (c, M)``.

    The (M, M) operator is never formed: each row block is contracted against ``X`` on the spot,
    so the result is (M, K) instead. S and T are only ever consumed as ``CᵀSC`` and ``Tr(PT)``,
    both of which factor through this, and at production sizes M is ~11.6x n_occ.
    """
    c = _row_chunk(M)
    pad = (-M) % c
    idx = jnp.concatenate([jnp.arange(M), jnp.zeros(pad, dtype=int)]).reshape(-1, c)

    def blk(r):
        return row_fn(r) @ X

    # prevent_cse=False: inside a scan, which already blocks the cross-iteration CSE the flag
    # guards. Without the checkpoint, reverse mode saves every block's (c, M, 3, 3) intermediate.
    return lax.map(jax.checkpoint(blk, prevent_cse=False), idx).reshape(-1, X.shape[-1])[:M]


def _overlap_rows(splats):
    """Row-block kernel for :func:`overlap_matrix`."""
    N = splats.norm

    def rows(r):
        _, _, _, _, detAij, _, K = _pairwise_rows(splats, r)
        return N[r][:, None] * N[None, :] * K * jnp.pi ** 1.5 / jnp.sqrt(detAij)
    return rows


def _kinetic_rows(splats):
    """Row-block kernel for :func:`kinetic_matrix`."""
    N = splats.norm

    def rows(r):
        A, mu, _, Aij_inv, detAij, mu_p, K = _pairwise_rows(splats, r)
        S = N[r][:, None] * N[None, :] * K * jnp.pi ** 1.5 / jnp.sqrt(detAij)

        AiAj = jnp.einsum("ikl,jlm->ijkm", A[r], A)       # A_i A_j, (c, M, 3, 3)
        tr_term = jnp.einsum("ijkl,ijlk->ij", AiAj, Aij_inv)   # tr(A_i A_j A_ij⁻¹)

        a = mu_p - mu[r][:, None, :]                      # (c, M, 3)
        b = mu_p - mu[None, :, :]                         # (c, M, 3)
        AiAj_b = jnp.einsum("ijkl,ijl->ijk", AiAj, b)
        quad = jnp.sum(a * AiAj_b, axis=-1)               # aᵀ A_i A_j b
        return S * (tr_term + 2.0 * quad)
    return rows


def overlap_times(splats, X) -> Float[Array, "M K"]:
    """``S @ X`` without forming S."""
    return _by_rows_times(_overlap_rows(splats), int(splats.n_basis), X)


def kinetic_times(splats, X) -> Float[Array, "M K"]:
    """``T @ X`` without forming T."""
    return _by_rows_times(_kinetic_rows(splats), int(splats.n_basis), X)


def _by_rows_times_idx(row_fn, rows, X):
    """``(A @ X)[rows]`` for an explicit row index array — a device's slice of the row-blocked
    contraction, so a mesh can split the O(M²·K) one-electron work instead of replicating it."""
    R = int(rows.shape[0])
    c = _row_chunk(int(X.shape[0]))
    pad = (-R) % c
    idx = jnp.concatenate([rows, jnp.zeros(pad, rows.dtype)]).reshape(-1, c)

    def blk(r):
        return row_fn(r) @ X

    return lax.map(jax.checkpoint(blk, prevent_cse=False), idx).reshape(-1, X.shape[-1])[:R]


def overlap_times_rows(splats, X, rows) -> Float[Array, "R K"]:
    """``(S @ X)[rows]`` without forming S."""
    return _by_rows_times_idx(_overlap_rows(splats), rows, X)


def kinetic_times_rows(splats, X, rows) -> Float[Array, "R K"]:
    """``(T @ X)[rows]`` without forming T."""
    return _by_rows_times_idx(_kinetic_rows(splats), rows, X)


def _pairwise_rows(splats, rows):
    """Pairwise product-Gaussian quantities for the splat ROWS ``rows`` against all columns.

    Identical arithmetic to the full form, with a leading ``c`` instead of ``M``; ``A``/``mu`` come
    back unsliced because the callers need the column-side values too.
    """
    A = splats.A                                          # (M, 3, 3)
    mu = splats.centers                                   # (M, 3)
    Ai = A[rows][:, None]                                 # (c, 1, 3, 3)
    Aj = A[None, :]                                       # (1, M, 3, 3)
    Aij = Ai + Aj                                         # (c, M, 3, 3)
    Aij_inv = inv3(Aij)                                   # (c, M, 3, 3) — closed-form 3×3 (linalg3),
    detAij = det3(Aij)                                    # (c, M)         batched LAPACK is overhead-bound here

    Ai_mu = jnp.einsum("imk,ik->im", A, mu)               # A_i μ_i, (M, 3)
    rhs = Ai_mu[rows][:, None, :] + Ai_mu[None, :, :]     # (c, M, 3)
    mu_p = jnp.einsum("ijkl,ijl->ijk", Aij_inv, rhs)      # (c, M, 3)

    d = mu[rows][:, None, :] - mu[None, :, :]             # μ_i - μ_j, (c, M, 3)
    # K = exp(-dᵀ A_i A_ij⁻¹ A_j d)
    Aj_d = jnp.einsum("jkl,ijl->ijk", A, d)               # A_j d  (uses A_j)
    AinvAjd = jnp.einsum("ijkl,ijl->ijk", Aij_inv, Aj_d)  # A_ij⁻¹ A_j d
    AiAinvAjd = jnp.einsum("ikl,ijl->ijk", A[rows], AinvAjd)   # A_i (...)
    K = jnp.exp(-jnp.sum(d * AiAinvAjd, axis=-1))         # (c, M)
    return A, mu, Aij, Aij_inv, detAij, mu_p, K


def _pairwise(splats):
    """The full (M, M) pairwise quantities — kept for the ERI path, which needs ``Aij`` whole."""
    return _pairwise_rows(splats, jnp.arange(splats.n_basis))


def overlap_matrix(splats) -> Float[Array, "M M"]:
    """S_ij = N_i N_j K_ij π^{3/2} / √det(A_ij).  Row-blocked (see :func:`_by_rows`).

    O(M²). Prefer :func:`overlap_times` where only ``S @ X`` is needed.
    """
    return _by_rows(_overlap_rows(splats), int(splats.n_basis))


def kinetic_matrix(splats) -> Float[Array, "M M"]:
    """T_ij = S_ij [ tr(A_i A_j A_ij⁻¹) + 2 aᵀ A_i A_j b ],  a=μ_p-μ_i, b=μ_p-μ_j.
    Row-blocked (see :func:`_by_rows`).

    O(M²). Prefer :func:`kinetic_times` where only ``T @ X`` is needed.
    """
    return _by_rows(_kinetic_rows(splats), int(splats.n_basis))


def nuclear_attraction_matrix(splats, atom_coords, atom_charges) -> Float[Array, "M M"]:
    """V_ij = -Σ_A Z_A N_iN_j K_ij (2/√π) ∫₀^∞ π^{3/2}/√det(B) exp(-dᵀ t²A_ij B⁻¹ d) dt,
    B = A_ij + t²I, d = μ_p - R_A. Laplace quadrature (matrices are 3×3, so the
    (Q,M,M,3,3) intermediate is small — no chunking needed here).
    """
    _, _, Aij, _, _, mu_p, K = _pairwise(splats)
    N = splats.norm
    NK = N[:, None] * N[None, :] * K                      # (M, M)

    t2 = (_QUAD_T ** 2)[:, None, None, None, None]        # (Q,1,1,1,1)
    B = Aij[None] + t2 * _I3                              # (Q, M, M, 3, 3)
    Binv = inv3(B)                                        # closed-form 3×3 (linalg3)
    detB = det3(B)                                        # (Q, M, M)
    # coefficient matrix t² A_ij B⁻¹  (Q,M,M,3,3)
    C = (_QUAD_T ** 2)[:, None, None, None, None] * jnp.einsum(
        "ijkl,qijlm->qijkm", Aij, Binv)

    V = jnp.zeros((splats.n_basis, splats.n_basis))
    for Aatom in range(atom_coords.shape[0]):
        Z = atom_charges[Aatom]
        d = mu_p - atom_coords[Aatom][None, None, :]      # (M, M, 3)
        Cd = jnp.einsum("qijkl,ijl->qijk", C, d)
        expo = jnp.sum(d[None] * Cd, axis=-1)             # (Q, M, M)
        integrand = (jnp.pi ** 1.5 / jnp.sqrt(detB)) * jnp.exp(-expo)  # (Q,M,M)
        vA = (2.0 / jnp.sqrt(jnp.pi)) * jnp.sum(_QUAD_W[:, None, None] * integrand, axis=0)
        V = V - Z * NK * vA
    return V


def one_electron_integrals(splats, atom_coords, atom_charges):
    S = overlap_matrix(splats)
    T = kinetic_matrix(splats)
    V = nuclear_attraction_matrix(splats, atom_coords, atom_charges)
    return S, T, V


# ---------------------------------------------------------------------------
#  ERIs — Laplace quadrature with the 6×6 block precision, chunked over nodes
# ---------------------------------------------------------------------------

def compute_eri_tensor(splats) -> Float[Array, "M M M M"]:
    """(ij|kl) via Laplace quadrature.

    For product-Gaussians p=(ij) and q=(kl) with precisions Â_p, Â_q and centers
    P_p, P_q, the node-t contribution is
        I(t) = π³/√det G(t) · exp(-cᵀ(D - D G(t)⁻¹ D) c),
    where c=[P_p;P_q], D=blkdiag(Â_p,Â_q), and
        G(t)=[[Â_p+t²I, -t²I],[-t²I, Â_q+t²I]]  (6×6).
    (ij|kl)=N_prod K_ij K_kl (2/√π) Σ_t w_t I(t). Chunked over t via lax.scan so
    the (M²,M²,6,6) intermediate is built one node at a time.
    """
    A, mu, Aij, _, _, mu_p, K = _pairwise(splats)
    N = splats.norm
    M = splats.n_basis
    P = M * M

    Ap = Aij.reshape(P, 3, 3)                             # product precisions (P,3,3)
    Pp = mu_p.reshape(P, 3)                               # product centers (P,3)
    Kf = K.reshape(P)
    Nab = (N[:, None] * N[None, :]).reshape(P)

    negI = -_I3                                           # (3,3)

    def node(carry, tw):
        t, w = tw
        t2 = t * t
        Ap_t = Ap + t2 * _I3                              # (P,3,3)
        # G blocks broadcast to (P,P,3,3)
        TL = jnp.broadcast_to(Ap_t[:, None], (P, P, 3, 3))
        BR = jnp.broadcast_to(Ap_t[None, :], (P, P, 3, 3))
        OFF = jnp.broadcast_to(t2 * negI, (P, P, 3, 3))
        top = jnp.concatenate([TL, OFF], axis=-1)         # (P,P,3,6)
        bot = jnp.concatenate([OFF, BR], axis=-1)
        G = jnp.concatenate([top, bot], axis=-2)          # (P,P,6,6)

        # D = blkdiag(Â_p, Â_q), c = [P_p; P_q]
        Dp = jnp.broadcast_to(Ap[:, None], (P, P, 3, 3))
        Dq = jnp.broadcast_to(Ap[None, :], (P, P, 3, 3))
        c = jnp.concatenate(
            [jnp.broadcast_to(Pp[:, None], (P, P, 3)),
             jnp.broadcast_to(Pp[None, :], (P, P, 3))], axis=-1)   # (P,P,6)
        # v = D c  (block-diagonal matvec)
        v = jnp.concatenate(
            [jnp.einsum("ijkl,ijl->ijk", Dp, c[..., :3]),
             jnp.einsum("ijkl,ijl->ijk", Dq, c[..., 3:])], axis=-1)  # (P,P,6)
        Ginv_v = jnp.linalg.solve(G, v[..., None])[..., 0]          # (P,P,6)
        quad = jnp.sum(c * v, axis=-1) - jnp.sum(v * Ginv_v, axis=-1)  # (P,P)
        detG = jnp.linalg.det(G)                                    # (P,P)
        I_t = jnp.pi ** 3 / jnp.sqrt(detG) * jnp.exp(-quad)
        return carry + w * (2.0 / jnp.sqrt(jnp.pi)) * I_t, None

    coulomb, _ = lax.scan(node, jnp.zeros((P, P)), (_QUAD_T, _QUAD_W))
    eri_flat = (Nab[:, None] * Nab[None, :]) * (Kf[:, None] * Kf[None, :]) * coulomb
    return eri_flat.reshape(M, M, M, M)


# ---------------------------------------------------------------------------
#  Grid evaluation
# ---------------------------------------------------------------------------

def eval_basis_on_grid(splats, grid_points) -> Float[Array, "G M"]:
    A = splats.A
    dr = grid_points[:, None, :] - splats.centers[None, :, :]   # (G, M, 3)
    Adr = jnp.einsum("mkl,gml->gmk", A, dr)                     # (G, M, 3)
    expo = jnp.sum(dr * Adr, axis=-1)                           # (G, M)
    return splats.norm[None, :] * jnp.exp(-expo)


def eval_basis_and_grad_on_grid(splats, grid_points):
    A = splats.A
    dr = grid_points[:, None, :] - splats.centers[None, :, :]   # (G, M, 3)
    Adr = jnp.einsum("mkl,gml->gmk", A, dr)                     # (G, M, 3)
    expo = jnp.sum(dr * Adr, axis=-1)
    g = splats.norm[None, :] * jnp.exp(-expo)                   # (G, M)
    dg = -2.0 * Adr * g[:, :, None]                            # ∇g = -2 A dr g
    return g, dg


def eval_density_on_grid(splats, C_orth, occupations, grid_points):
    G_mat = eval_basis_on_grid(splats, grid_points)
    psi = G_mat @ C_orth
    return jnp.sum(occupations[None, :] * psi ** 2, axis=-1)


def eval_density_and_grad_on_grid(splats, C_orth, occupations, grid_points):
    g_vals, dg = eval_basis_and_grad_on_grid(splats, grid_points)
    psi = g_vals @ C_orth
    dpsi = jnp.einsum("gmd,mn->gnd", dg, C_orth)
    rho = jnp.sum(occupations[None, :] * psi ** 2, axis=-1)
    grad_rho = 2.0 * jnp.einsum("n,gn,gnd->gd", occupations, psi, dpsi)
    return rho, grad_rho


def eval_density_grad_tau_on_grid(splats, C_orth, occupations, grid_points):
    """(ρ, ∇ρ, τ) — the meta-GGA triple, for functionals that read the kinetic energy density.

    τ = ½ Σ_i n_i |∇ψ_i|², the SAME convention the engine's Gaussian path uses
    (``dftax/ks/terms.py``: ``tau = 0.5 * einsum("mx,mn,nx->", dao_g, P, dao_g)``, which with
    P = Σ_i n_i c_i c_iᵀ is exactly this). Matching it is not cosmetic: a factor of two in τ is a
    silently wrong meta-GGA energy, not a crash, and the whole point of the reference ladder is
    that the two engines evaluate the same functional.

    Nearly free on top of the GGA path: ``dpsi`` is already built for ∇ρ, so τ is one more
    contraction of quantities in hand rather than a second pass over the grid.
    """
    g_vals, dg = eval_basis_and_grad_on_grid(splats, grid_points)
    psi = g_vals @ C_orth
    dpsi = jnp.einsum("gmd,mn->gnd", dg, C_orth)
    rho = jnp.sum(occupations[None, :] * psi ** 2, axis=-1)
    grad_rho = 2.0 * jnp.einsum("n,gn,gnd->gd", occupations, psi, dpsi)
    tau = 0.5 * jnp.einsum("n,gnd,gnd->g", occupations, dpsi, dpsi)
    return rho, grad_rho, tau
