"""Orthonormalizer for the occupied MO coefficients — the regularized Löwdin collapse fix.

The KS density of a closed-shell system with UNIFORM occupations is invariant to the *choice* of
orthonormalization: given C (M×N_occ) and the metric S (M×M), any Ĉ = C·T with ĈᵀSĈ = I spanning
the same column space gives the SAME P = Ĉ diag(occ) Ĉᵀ. So the physics is identical to a plain
symmetric-Löwdin M^{-1/2} in exact arithmetic; the choice matters ONLY for numerical conditioning +
AD stability as the occupied Gram M = CᵀSC → singular.

The mechanism: E_kin = occ·Tr(M⁻¹ M_T) with M_T = CᵀTC, so the kinetic energy carries the
occupied-Gram inverse and diverges as λmin(M) → 0. ``reg_inv_sqrt`` is symmetric Löwdin M^{-1/2}
with a custom VJP that floors λ in the forward (bounding 1/√λ, and handling the indefinite screened
Gram whose λmin can go negative) and gap-thresholds the near-null modes in the backward. The floor
is RELATIVE and must be large enough to engage before the instability bites (floor_rel ~ 1e-4, not
1e-10); it is inert on a healthy Gram, so a converged energy is unbiased.

This is the sole orthonormalization path — there is no unregularized-eigh alternative.

``orthonormalize(C, M)`` returns Ĉ = C·T; callers form ``M`` from whatever metric path they have —
dense ``M = Cᵀ(S C)`` or the screened matvec.
"""
from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["lowdin_orthonormalize", "orthonormalize", "transform", "reg_inv_sqrt"]


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def reg_inv_sqrt(M, floor_rel=1e-4, gap_rel=1e-4, discard=True):
    """T = M^{-1/2} (symmetric Löwdin) with a stabilized custom VJP. Forward floors λ at
    floor_rel·λmax (bounds 1/√λ; the floor also clamps NEGATIVE eigenvalues, so it survives the
    indefinite screened Gram). Backward (below) uses a gap-thresholded divided difference and, when
    ``discard``, zeroes the near-null modes' contribution — the anomalous collapse gradient.
    ``floor_rel`` must engage before the instability bites (~1e-4), NOT 1e-10 (never engages)."""
    w, U = jnp.linalg.eigh(M)
    wf = jnp.maximum(w, floor_rel * w[-1])
    return (U * (1.0 / jnp.sqrt(wf))) @ U.T


def _ris_fwd(M, floor_rel, gap_rel, discard):
    w, U = jnp.linalg.eigh(M)
    wf = jnp.maximum(w, floor_rel * w[-1])
    T = (U * (1.0 / jnp.sqrt(wf))) @ U.T
    return T, (w, U)


def _ris_bwd(floor_rel, gap_rel, discard, res, Tbar):
    w, U = res
    scale = w[-1]
    floor = floor_rel * scale
    gap = gap_rel * scale
    wf = jnp.maximum(w, floor)
    g = 1.0 / jnp.sqrt(wf)                                     # g(λ) = 1/√(max(λ,floor))
    gp = jnp.where(w > floor, -0.5 * wf ** -1.5, 0.0)         # g'(λ) (flat below floor ⇒ 0)
    dw = w[:, None] - w[None, :]
    dg = g[:, None] - g[None, :]
    safe = jnp.abs(dw) > gap                                   # threshold the tiny denominators
    Omega = jnp.where(safe, dg / jnp.where(safe, dw, 1.0),    # divided difference off-diagonal
                      0.5 * (gp[:, None] + gp[None, :]))       # → midpoint g' near-degenerate/diag
    G = U.T @ (0.5 * (Tbar + Tbar.T)) @ U
    if discard:                                                # drop the near-null (anomalous) modes
        keep = (w > floor).astype(w.dtype)
        G = G * keep[:, None] * keep[None, :]
    Mbar = U @ (Omega * G) @ U.T
    return (Mbar,)


reg_inv_sqrt.defvjp(_ris_fwd, _ris_bwd)


def transform(M):
    """The (N,N) orthonormalizing transform T = M^{-1/2} (Tᵀ M T ≈ I), regularized Löwdin."""
    return reg_inv_sqrt(M)


def orthonormalize(C, M):
    """Ĉ = C·T, T from ``transform``. ``M`` = the occupied Gram CᵀSC."""
    return C @ transform(M)


def lowdin_orthonormalize(C, S):
    """Löwdin-orthonormalize MO coefficients w.r.t. the primitive overlap ``S`` (M×M):
    Ĉ = C (CᵀSC)^{-1/2}, so ĈᵀSĈ = I. Forms the projected Gram and delegates to
    ``orthonormalize`` (the regularized path above)."""
    M_mo = C.T @ (S @ C)                                  # projected overlap (N_occ, N_occ)
    return orthonormalize(C, M_mo)
