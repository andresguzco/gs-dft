"""Phase 2 — XC grid screening for the splat density evaluation.

The XC term needs ρ(r) and ∇ρ(r) on the grid. The dense path (``integrals.dense.eval_density_*``)
forms the ``(G, M, 3)`` basis-on-grid array, which is O(G·M) in both memory and compute. Two
alternatives live here:

- ``chunked_density_*`` — the exact density streamed over grid-point chunks with remat, so memory
  is O(chunk·M) while compute stays O(G·M). This is the production path and the exact reference.
- ``screened_density_*`` — exploits grid↔splat locality via a cell list and ``segment_sum``, giving
  O(G) memory and compute. Validated (``test_grid_screening``) but deliberately not wired into
  ``SplatKS``: it is a memory tool only, since the scatter loses to dense GEMM on compute. Use
  ``screening/cells.py`` for grid-locality speed.

Backend: the single **full-covariance** splat (g_i(r) = norm_i exp(−drᵀA_i dr), ∇g = −2 A_i dr g).
The per-splat quantities (the closed-form A-chart: spectral quaternion + log-eigenvalues, no
eigh/expm) are built ONCE per call, outside the remat'd streams; per-splat cutoff radii use the
smallest precision eigenvalue λmin(A_i) (= exp(min log-eigenvalue), host-side).
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from gs_dft.coulomb.stream import full_quantities

__all__ = ["chunked_density_and_grad", "chunked_density_grad_tau", "chunked_density",
           "grid_neighbor_pairs", "screened_density_and_grad", "screened_density"]

# DFTAX_XC_GRAD_LEGACY=1 restores the (n,M,3)x(M,n_occ) contraction for ∇ρ (parity/timing only).
_LEGACY_DPSI = os.environ.get("DFTAX_XC_GRAD_LEGACY", "0") == "1"



# --------------------------------------------------------------------------------------------------
#  Per-backend basis evaluation: ``quantities(splats)`` once per call → ``gather`` per-splat params
#  (with optional dtype cast) → ``eval_block`` ((n pts) × (m splats), the GEMM-shaped path) or
#  ``eval_pairs`` (one (grid, splat) pair per row, the scatter path). Each returns (g, ∇g).
# --------------------------------------------------------------------------------------------------
def _gather(q, idx, cd):
    return tuple(x[idx].astype(cd) for x in q)


def _full_eval_block(params, pts):
    # The quadratic form is EXPANDED into elementwise broadcasts of the 6 symmetric components
    # ((m,)-vectors × (n,m) displacements) instead of an einsum over A (m,3,3): the einsum lowers
    # to a batched-contraction call that materializes (n,m,3) intermediates and breaks fusion
    # with the exp — expanded, the whole eval is one fused kernel (see coulomb.ri.full_isobra_block).
    A, ce, no = params                                          # (m,3,3), (m,3), (m,)
    dx = pts[:, None, 0] - ce[None, :, 0]                       # (n, m)
    dy = pts[:, None, 1] - ce[None, :, 1]
    dz = pts[:, None, 2] - ce[None, :, 2]
    a00, a11, a22 = A[None, :, 0, 0], A[None, :, 1, 1], A[None, :, 2, 2]
    a01, a02, a12 = A[None, :, 0, 1], A[None, :, 0, 2], A[None, :, 1, 2]
    Adr0 = a00 * dx + a01 * dy + a02 * dz                       # (A dr)_k  (n, m)
    Adr1 = a01 * dx + a11 * dy + a12 * dz
    Adr2 = a02 * dx + a12 * dy + a22 * dz
    g = no[None, :] * jnp.exp(-(dx * Adr0 + dy * Adr1 + dz * Adr2))
    dg = -2.0 * jnp.stack([Adr0, Adr1, Adr2], axis=-1) * g[:, :, None]   # ∇g = −2 A dr g
    return g, dg


def _full_eval_pairs(params, dr):
    A, _, no = params
    dx, dy, dz = dr[:, 0], dr[:, 1], dr[:, 2]
    Adr0 = A[:, 0, 0] * dx + A[:, 0, 1] * dy + A[:, 0, 2] * dz
    Adr1 = A[:, 0, 1] * dx + A[:, 1, 1] * dy + A[:, 1, 2] * dz
    Adr2 = A[:, 0, 2] * dx + A[:, 1, 2] * dy + A[:, 2, 2] * dz
    g = no * jnp.exp(-(dx * Adr0 + dy * Adr1 + dz * Adr2))
    dg = -2.0 * jnp.stack([Adr0, Adr1, Adr2], axis=-1) * g[:, None]
    return g, dg


def _host_rcut(splats, eps):
    """Per-splat cutoff radius (host numpy): g_i/norm_i > eps within
    r_cut_i = √(−ln ε / λmin(A_i)), λmin = exp(min ℓ_i) — the slowest-decaying principal
    axis (conservative); read straight off the spectral chart, no eigh."""
    loge = np.maximum(-np.log(eps), 1e-12)
    lam_min = np.exp(np.asarray(splats.log_scale).min(axis=-1))
    return np.sqrt(loge / lam_min)


# --------------------------------------------------------------------------------------------------
#  2a — chunked EXACT density (streamed over grid points + remat): O(chunk·M) memory, O(G·M) compute
# --------------------------------------------------------------------------------------------------
def chunked_density_and_grad(splats, C, occ, grid_points, chunk=4096):
    """EXACT ρ(r), ∇ρ(r), streamed over grid-point chunks of ``chunk`` + ``jax.checkpoint`` so the
    (chunk, M, 3) intermediate (not the full (G, M, 3)) is the peak — O(chunk·M) memory. Equals
    ``dense.eval_density_and_grad_on_grid`` exactly. Drop-in (avoids the large-M grid OOM)."""
    q = full_quantities(splats)                                 # ONCE, outside the remat'd scan
    G = grid_points.shape[0]
    pad = (-G) % chunk
    pts = jnp.concatenate([grid_points, jnp.zeros((pad, 3))]).reshape(-1, chunk, 3)

    def body(_, p):
        g, dg = _full_eval_block(q, p)                          # (n, M), (n, M, 3)
        psi = g @ C                                             # (n, n_occ)
        rho = jnp.sum(occ[None, :] * psi ** 2, axis=1)          # (n,)
        if _LEGACY_DPSI:
            dpsi = jnp.einsum("gmd,mn->gnd", dg, C)             # (n, n_occ, 3)
            grad = 2.0 * jnp.einsum("n,gn,gnd->gd", occ, psi, dpsi)
        else:
            # ∇ρ = 2 Σ_i φ_i ∇g_i with φ = (occ⊙ψ)Cᵀ: one (n,n_occ)x(n_occ,M) GEMM instead of the
            # (n,M,3)x(M,n_occ) contraction (3x the FLOPs) and no (n, n_occ, 3) intermediate.
            phi = (psi * occ[None, :]) @ C.T                    # (n, M)
            grad = 2.0 * jnp.einsum("gm,gmd->gd", phi, dg)
        return None, (rho, grad)

    _, (rho_c, grad_c) = lax.scan(jax.checkpoint(body), None, pts)
    return rho_c.reshape(-1)[:G], grad_c.reshape(-1, 3)[:G]


def chunked_density_grad_tau(splats, C, occ, grid_points, chunk=4096):
    """Exact (ρ, ∇ρ, τ), streamed over grid chunks + remat — the meta-GGA analogue of
    :func:`chunked_density_and_grad`, equal to ``dense.eval_density_grad_tau_on_grid``.

    τ rides along inside the same scan rather than in a second pass: ``dpsi`` is the expensive
    intermediate and it is already resident here, so computing τ elsewhere would re-derive it and
    double the grid work for a quantity that costs one contraction.
    """
    q = full_quantities(splats)
    G = grid_points.shape[0]
    pad = (-G) % chunk
    pts = jnp.concatenate([grid_points, jnp.zeros((pad, 3))]).reshape(-1, chunk, 3)

    def body(_, p):
        g, dg = _full_eval_block(q, p)
        psi = g @ C
        rho = jnp.sum(occ[None, :] * psi ** 2, axis=1)
        dpsi = jnp.einsum("gmd,mn->gnd", dg, C)
        grad = 2.0 * jnp.einsum("n,gn,gnd->gd", occ, psi, dpsi)
        tau = 0.5 * jnp.einsum("n,gnd,gnd->g", occ, dpsi, dpsi)
        return None, (rho, grad, tau)

    _, (rho_c, grad_c, tau_c) = lax.scan(jax.checkpoint(body), None, pts)
    return rho_c.reshape(-1)[:G], grad_c.reshape(-1, 3)[:G], tau_c.reshape(-1)[:G]


def chunked_density(splats, C, occ, grid_points, chunk=4096):
    """EXACT ρ(r) only (LDA path), streamed over grid chunks + remat. O(chunk·M) memory."""
    q = full_quantities(splats)
    G = grid_points.shape[0]
    pad = (-G) % chunk
    pts = jnp.concatenate([grid_points, jnp.zeros((pad, 3))]).reshape(-1, chunk, 3)

    def body(_, p):
        g, _dg = _full_eval_block(q, p)
        psi = g @ C
        return None, jnp.sum(occ[None, :] * psi ** 2, axis=1)

    _, rho_c = lax.scan(jax.checkpoint(body), None, pts)
    return rho_c.reshape(-1)[:G]


# --------------------------------------------------------------------------------------------------
#  2b — grid SCREENING: each grid point sees only O(1) significant splats → O(G) memory and compute
# --------------------------------------------------------------------------------------------------
def grid_neighbor_pairs(splats, grid_points, eps: float = 1e-8, pad_to=None):
    """Significant (grid, splat) pairs: g_i(r_g)/norm_i > eps ⇔ |r_g−μ_i| < r_cut_i (per-backend
    cutoff — see ``_host_rcut``). cKDTree on the (fixed) grid; query each splat's cutoff ball.
    Host-side, rebuilt when the splats move (the grid is static). Returns (gp_idx, splat_idx,
    valid) int32."""
    from scipy.spatial import cKDTree
    gp = np.asarray(grid_points)
    mu = np.asarray(splats.centers)
    rcut = _host_rcut(splats, eps)                             # per-splat cutoff radius
    tree = cKDTree(gp)
    res = tree.query_ball_point(mu, rcut)                      # per-splat list of grid indices
    gi, si = [], []
    for i, gids in enumerate(res):
        if len(gids):
            gi.append(np.asarray(gids, np.int32)); si.append(np.full(len(gids), i, np.int32))
    gi = np.concatenate(gi) if gi else np.zeros(0, np.int32)
    si = np.concatenate(si) if si else np.zeros(0, np.int32)

    if pad_to is not None:
        if gi.shape[0] > pad_to:
            raise ValueError(f"{gi.shape[0]} grid pairs > pad_to={pad_to}; raise pad_to.")
        n = gi.shape[0]
        valid = np.zeros(pad_to, bool); valid[:n] = True
        gi = np.concatenate([gi, np.zeros(pad_to - n, np.int32)])
        si = np.concatenate([si, np.zeros(pad_to - n, np.int32)])
        return jnp.asarray(gi), jnp.asarray(si), jnp.asarray(valid)
    return jnp.asarray(gi), jnp.asarray(si), jnp.ones(gi.shape[0], bool)


def _grid_pair_chunks(gp_idx, splat_idx, valid, chunk):
    Npair = gp_idx.shape[0]
    pad = (-Npair) % chunk
    base_valid = jnp.ones(Npair, bool) if valid is None else valid
    gic = jnp.concatenate([gp_idx, jnp.zeros(pad, gp_idx.dtype)]).reshape(-1, chunk)
    sic = jnp.concatenate([splat_idx, jnp.zeros(pad, splat_idx.dtype)]).reshape(-1, chunk)
    vvc = jnp.concatenate([base_valid, jnp.zeros(pad, bool)]).reshape(-1, chunk)
    return gic, sic, vvc


def screened_density_and_grad(splats, C, occ, grid_points,
                              gp_idx, splat_idx, valid=None, chunk=16384):
    """ρ(r), ∇ρ(r) over the significant (grid, splat) pairs only — **O(G) memory and compute**
    (avoids the dense (G,M,3)). TWO-PASS so ∇ρ stays (G,3), not (G, n_occ, 3):
      pass 1: ψ[g,o] = Σ_{i~g} g_i(r_g) C[i,o]   (segment_sum) → ρ = Σ_o occ_o ψ²
      pass 2: ∇ρ[g] = Σ_{i~g} 2·(Σ_o occ_o ψ[g,o] C[i,o])·∇g_i(r_g)   (segment_sum)
    Both passes streamed over pair chunks + ``jax.checkpoint``. Equals ``chunked_density_and_grad``
    when no pair is screened. Differentiable in the splat params and C."""
    q = full_quantities(splats)                                 # ONCE, outside the remat'd scans
    G = grid_points.shape[0]
    n_occ = C.shape[1]
    gic, sic, vvc = _grid_pair_chunks(gp_idx, splat_idx, valid, chunk)

    def _gval(gi, si, vv):
        dr = grid_points[gi] - q[1][si]                                     # (chunk, 3)
        g, dg = _full_eval_pairs(_gather(q, si, dr.dtype), dr)
        return jnp.where(vv, g, 0.0), dg

    def body1(psi, carry):
        gi, si, vv = carry
        g, _ = _gval(gi, si, vv)
        return psi + jax.ops.segment_sum(g[:, None] * C[si], gi, num_segments=G), None

    psi, _ = lax.scan(jax.checkpoint(body1), jnp.zeros((G, n_occ)), (gic, sic, vvc))
    rho = jnp.sum(occ[None, :] * psi ** 2, axis=1)

    def body2(grad, carry):
        gi, si, vv = carry
        g, dg = _gval(gi, si, vv)                                           # g masked; dg·g masked
        dg = jnp.where(vv[:, None], dg, 0.0)                                # mask ∇g too
        w = jnp.sum(occ[None, :] * psi[gi] * C[si], axis=1)                 # (chunk,)
        return grad + jax.ops.segment_sum(2.0 * w[:, None] * dg, gi, num_segments=G), None

    grad_rho, _ = lax.scan(jax.checkpoint(body2), jnp.zeros((G, 3)), (gic, sic, vvc))
    return rho, grad_rho


def screened_density(splats, C, occ, grid_points, gp_idx, splat_idx,
                     valid=None, chunk=16384):
    """ρ(r) only (LDA path) over significant (grid, splat) pairs (pass 1 of the two-pass)."""
    q = full_quantities(splats)
    G = grid_points.shape[0]
    gic, sic, vvc = _grid_pair_chunks(gp_idx, splat_idx, valid, chunk)

    def body(psi, carry):
        gi, si, vv = carry
        dr = grid_points[gi] - q[1][si]
        g, _ = _full_eval_pairs(_gather(q, si, dr.dtype), dr)
        g = jnp.where(vv, g, 0.0)
        return psi + jax.ops.segment_sum(g[:, None] * C[si], gi, num_segments=G), None

    psi, _ = lax.scan(jax.checkpoint(body), jnp.zeros((G, C.shape[1])), (gic, sic, vvc))
    return jnp.sum(occ[None, :] * psi ** 2, axis=1)

