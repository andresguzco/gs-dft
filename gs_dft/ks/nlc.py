"""VV10 nonlocal correlation with an **analytically streamed reverse pass**.

The engine's :func:`dftax.energy.vv10.vv10_energy` streams the double-grid pair quadrature over
outer-point chunks, so its forward peak is O(chunk·ng). Reverse mode undoes that: the ``lax.map``
saves each chunk's ``(r2, g, gp)`` residuals and they concatenate into exactly the O(ng²) tensor the
chunking avoids. ``jax.checkpoint`` bounds the memory but leaves XLA emitting one enormous fused
backward kernel, so the backward is written out analytically here instead.

**The identity that makes this short.** Write ``rw_i = ρ_i w_i`` (zero off-mask) and

    g_ij = r²_ij·w0_i + κ_i,   D_ij = g_ij · g_ji · (g_ij + g_ji).

``g_ji`` is ``g_ij`` with the indices swapped, so **D is symmetric**, and the outer weight ``ρ_i w_i``
of the energy is the same ``rw_i`` that appears in the inner sum. The whole functional is therefore
a symmetric quadratic form,

    E = β·Σ_i rw_i − ¾·Σ_ij rw_i rw_j / D_ij,

and every derivative is one reduction over ``j`` sharing the factor ``T_ij = rw_j·∂D/∂u·D⁻²``:

    ∂E/∂rw_k = β + F_k,        F_k = −1.5·Σ_j rw_j/D_kj
    ∂E/∂w0_k = 1.5·rw_k·Σ_j r²_kj·T_kj
    ∂E/∂κ_k  = 1.5·rw_k·Σ_j T_kj
    ∂E/∂R_k  = 3·rw_k·Σ_j [∂D/∂u·w0_k + ∂D/∂v·w0_j]·rw_j·D⁻²·(R_k − R_j)

with ``∂D/∂u = v(2u+v)``, ``∂D/∂v = u(u+2v)``, ``u = g_kj``, ``v = g_jk``. Each appears once because
the k-th point enters the double sum through both orderings and the two contributions are equal —
that is where the factors of 2 (and the 3 in ∂E/∂R) come from.

The residual is four ``(ng,)`` arrays; everything else is recomputed one chunk at a time, so the
reverse peak is O(chunk·ng) like the forward. The grid-coordinate cotangent is implemented (VV10
is differentiable in the moving Becke grid, which is how nuclear forces get it) — returning zeros
there would be a silent wrong answer, not a missing feature.

``tests/unit/test_nlc_remat.py`` pins the value to the engine and every cotangent to autodiff.
"""

from __future__ import annotations

import functools
import math
import os

import jax
import jax.numpy as jnp

_THRESH = 1e-8    # density threshold on both grids (PySCF convention), as in the engine

#: Outer-chunk size for both directions.
#:
#: 256, not the engine's 2048. The (chunk, ng) intermediate sets the size of the fused backward
#: kernel, and at 2048 ptxas needs tens of GB of host memory and the compile does not finish in
#: reasonable time.
#: An isolated benchmark of this term cannot see the problem; raise this only with a measurement
#: of the real training step.
DEFAULT_CHUNK = int(os.environ.get("DFTAX_VV10_CHUNK", 256))


def _params(b: float, c: float):
    pi = math.pi
    kvv = 1.5 * pi * b * (9.0 * pi) ** (-1.0 / 6.0)
    beta = (3.0 / (b * b)) ** 0.75 / 32.0
    return pi, kvv, beta


def _fields(rho, gnorm2, weights, b, c):
    """``(mask, rho_s, w0, kap, rw)`` — the per-point quantities the pair sum needs."""
    pi, kvv, _ = _params(b, c)
    mask = rho >= _THRESH
    rho_s = jnp.where(mask, rho, 1.0)                 # safe values off-mask
    w0 = jnp.sqrt(c * (gnorm2 / (rho_s * rho_s)) ** 2 + (4.0 * pi / 3.0) * rho_s)
    kap = kvv * rho_s ** (1.0 / 6.0)
    rw = jnp.where(mask, rho * weights, 0.0)
    return mask, rho_s, w0, kap, rw


def _chunked(coords, w0, kap, mask, chunk):
    """Pad the OUTER index to a multiple of ``chunk``; the inner sum stays at full ``ng``."""
    ng = w0.shape[0]
    pad = (-ng) % chunk
    return (jnp.pad(coords, ((0, pad), (0, 0))).reshape(-1, chunk, 3),
            jnp.pad(w0, (0, pad), constant_values=1.0).reshape(-1, chunk),
            jnp.pad(kap, (0, pad), constant_values=1.0).reshape(-1, chunk),
            jnp.pad(mask, (0, pad)).reshape(-1, chunk), ng)


def _pair_F(coords, w0, kap, rw, mask, chunk):
    """``F_i = -1.5·Σ_j rw_j/D_ij``, streamed. This is the forward's only pair sum."""
    co_c, w0_c, kap_c, m_c, ng = _chunked(coords, w0, kap, mask, chunk)

    def body(args):
        co, w0o, ko, mo = args
        r2 = jnp.sum((co[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        u = r2 * w0o[:, None] + ko[:, None]
        v = r2 * w0[None, :] + kap[None, :]
        f = -1.5 * jnp.sum(rw[None, :] / (u * v * (u + v)), axis=1)
        return jnp.where(mo, f, 0.0)

    return jax.lax.map(body, (co_c, w0_c, kap_c, m_c)).reshape(-1)[:ng]


def vv10_energy(rho, gnorm2, coords, weights, b: float, c: float, chunk: int = None):
    """VV10 energy from grid densities. Drop-in for ``dftax.energy.vv10.vv10_energy``."""
    return _vv10(float(b), float(c), int(DEFAULT_CHUNK if chunk is None else chunk),
                 rho, gnorm2, coords, weights)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _vv10(b, c, chunk, rho, gnorm2, coords, weights):
    _, _, beta = _params(b, c)
    mask, _, w0, kap, rw = _fields(rho, gnorm2, weights, b, c)
    F = _pair_F(coords, w0, kap, rw, mask, chunk)
    eps = jnp.where(mask, beta + 0.5 * F, 0.0)
    return jnp.sum(jnp.where(mask, weights * rho, 0.0) * eps)


def _vv10_fwd(b, c, chunk, rho, gnorm2, coords, weights):
    # The residual is the inputs, not the pair quantities. Everything on a (chunk, ng) shape is
    # recomputed in the backward, one chunk at a time — that is the whole point of the file.
    return _vv10(b, c, chunk, rho, gnorm2, coords, weights), (rho, gnorm2, coords, weights)


def _vv10_bwd(b, c, chunk, res, ct):
    rho, gnorm2, coords, weights = res
    pi, kvv, beta = _params(b, c)
    mask, rho_s, w0, kap, rw = _fields(rho, gnorm2, weights, b, c)
    co_c, w0_c, kap_c, m_c, ng = _chunked(coords, w0, kap, mask, chunk)
    rw_c = jnp.pad(rw, (0, (-ng) % chunk)).reshape(-1, chunk)

    def body(args):
        co, w0o, ko, rwo, mo = args
        r2 = jnp.sum((co[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        u = r2 * w0o[:, None] + ko[:, None]                  # g_kj
        v = r2 * w0[None, :] + kap[None, :]                  # g_jk
        D = u * v * (u + v)
        wj_invD2 = rw[None, :] / (D * D)
        T = wj_invD2 * (v * (2.0 * u + v))                   # rw_j · ∂D/∂u · D⁻²
        f = jnp.where(mo, -1.5 * jnp.sum(rw[None, :] / D, axis=1), 0.0)
        d_w0 = 1.5 * rwo * jnp.sum(T * r2, axis=1)
        d_kap = 1.5 * rwo * jnp.sum(T, axis=1)
        # ∂E/∂R_k, without ever forming a (chunk, ng, 3) difference: the sum over j splits into
        # R_k·Σ_j coef  −  coef @ coords.
        coef = T * w0o[:, None] + wj_invD2 * (u * (u + 2.0 * v)) * w0[None, :]
        d_R = 3.0 * rwo[:, None] * (co * jnp.sum(coef, axis=1)[:, None] - coef @ coords)
        return f, jnp.where(mo, d_w0, 0.0), jnp.where(mo, d_kap, 0.0), jnp.where(mo[:, None], d_R, 0.0)

    F, G_w0, G_kap, G_R = jax.lax.map(body, (co_c, w0_c, kap_c, rw_c, m_c))
    F = F.reshape(-1)[:ng]
    G_w0 = G_w0.reshape(-1)[:ng]
    G_kap = G_kap.reshape(-1)[:ng]
    G_R = G_R.reshape(-1, 3)[:ng]

    # chain the per-point fields back to (ρ, |∇ρ|²)
    G_rw = beta + F                                          # ∂E/∂rw
    q = gnorm2 / (rho_s * rho_s)
    dw0_dg = c * q / (w0 * rho_s * rho_s)
    dw0_drho = (-4.0 * c * q * gnorm2 / rho_s ** 3 + 4.0 * pi / 3.0) / (2.0 * w0)
    dkap_drho = (kvv / 6.0) * rho_s ** (-5.0 / 6.0)

    rho_bar = jnp.where(mask, G_rw * weights + G_kap * dkap_drho + G_w0 * dw0_drho, 0.0)
    gnorm2_bar = jnp.where(mask, G_w0 * dw0_dg, 0.0)
    weights_bar = jnp.where(mask, G_rw * rho, 0.0)
    return (ct * rho_bar, ct * gnorm2_bar, ct * G_R, ct * weights_bar)


_vv10.defvjp(_vv10_fwd, _vv10_bwd)
