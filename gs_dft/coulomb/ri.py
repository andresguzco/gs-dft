"""Density fitting (RI-J / RI-K) for the splat Coulomb energy — the frozen pair-product aux.

Routes ``E_J`` through the **density** (low-rank, dimension ~nao, fixed by the molecule)
instead of the O(M⁴) pairwise ERI, so the per-step Coulomb is O(M²·N_aux) = O(M²) when the
auxiliary basis size N_aux is fixed. The auxiliary basis is the **frozen pair-product** set
(``basis.isotropic.init_product_aux``): isotropic Gaussians placed deterministically at the
primary splats' own product centers, rebuilt at each ``refresh`` — no aux parameters are
optimized, so the aux cannot collapse.

Math (Dunlap's robust functional): with fitted density ρ̃ = Σ_P d_P χ_P,

    J_rob[d] = (ρ|ρ̃) − ½(ρ̃|ρ̃) = dᵀγ − ½ dᵀ V d ,   γ_P = (P|ρ) = Σ_a q_a (P|a),  V_PQ = (P|Q),

    E_J − max_d J_rob = ½ (ρ−ρ̃ | ρ−ρ̃) ≥ 0   (one-sided lower bound, 2nd-order in the residual).

``max_d J_rob = ½ γᵀ V⁻¹ γ = ‖𝒫_aux ρ‖²_Coul``; only the coefficients d = V⁻¹γ are solved
(closed-form, every step). This module provides the DF energy building blocks.

The auxiliary basis is **isotropic**, so the aux–aux metric ``V_PQ = (P|Q)`` is the exact
closed-form Boys ``(1|2)`` (no quadrature). The single primary is the full-covariance splat
(spectral chart), so the 3-center ``(P|a)`` couples the iso aux to a **full-covariance pair**
via ``full_isobra_block`` — the (iso | full) Coulomb in the pair's principal-axis (eigenframe)
**Laplace quadrature** (an anisotropic Coulomb has no closed form), reusing ``coulomb.stream``'s
pair-quantity helpers. Driven through ``df_coulomb_energy(...)``.
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg       # cho_solve for the pre-factored (frozen-aux) DF metric
from jax import lax

from dftax.energy.boys import boys
# Laplace half-line quadrature, python-float copies for the STATIC unrolls (trace-time
# constants — lets XLA fuse the whole node loop into one kernel; see full_isobra_block).
from gs_dft.integrals.dense import (_QUAD_T_LIST, _QUAD_W_LIST,
                                             _gauss_legendre_segment)
from gs_dft.coulomb.stream import (
    make_pairs, _full_pair_quantities, full_quantities,
)

__all__ = [
    "coulomb_block", "full_isobra_block",
    "aux_quantities", "aux_metric", "aux_cholesky", "shift_diag",
    "df_coulomb_energy", "build_three_center", "df_exchange_energy",
    "df_exchange_energy_density",
    "df_coulomb_energy_screened", "df_exchange_energy_screened",
]

_I3 = np.eye(3)          # numpy: importing the package must not initialise the JAX backend


_AUX_BLOCK_BUDGET = 2 ** 26     # matches `integrals/dense.py`'s `_PAIR_BUDGET`


def _block_rows(B: int) -> int:
    """Rows of set 1 per block, so the ``(c, B, 3)`` centre difference stays near the budget."""
    return max(1, _AUX_BLOCK_BUDGET // max(1, 3 * int(B)))


def _coulomb_rows(g1, P1, KN1, g2, P2, KN2, omega):
    """One row block of :func:`coulomb_block` — that docstring carries the integral."""
    gab = g1[:, None] * g2[None, :]                         # (A,B)
    gsum = g1[:, None] + g2[None, :]
    PQ2 = jnp.sum((P1[:, None, :] - P2[None, :, :]) ** 2, axis=-1)
    rho = gab / gsum
    T = rho * PQ2
    if omega:
        z = (omega ** 2) / (omega ** 2 + rho)
        F0 = jnp.sqrt(z) * boys(0, z * T)
    else:
        F0 = boys(0, T)
    coulomb = 2.0 * jnp.pi ** 2.5 / (gab * jnp.sqrt(gsum)) * F0
    return (KN1[:, None] * KN2[None, :]) * coulomb          # (A,B)


def coulomb_block(g1, P1, KN1, g2, P2, KN2, omega=None):
    """Exact (1|2) Coulomb between two sets of s-Gaussian quantities → (A, B) block.

    (1|2) = KN₁·KN₂ · 2π^{5/2}/(g₁g₂√(g₁+g₂)) · F₀(g₁g₂/(g₁+g₂)·|P₁−P₂|²).
    Each function is described only by (g=exponent, P=center, KN=norm·overlap-factor), so this
    serves both V_PQ (aux|aux) and the 3-center (aux|pair) — pass the matching quantities.

    ``omega`` (>0) switches the kernel from 1/r to the LONG-RANGE erf(ωr)/r, which for s-Gaussians
    stays closed-form — no quadrature is needed on this two-centre path:

        (1|erf(ωr)/r|2) = (1|1/r|2) with  F₀(T) → √z · F₀(z·T),   z = ω²/(ω² + ρ),  ρ = g₁g₂/(g₁+g₂)

    ρ is the reduced exponent, not the density.
    """
    A, B = int(g1.shape[0]), int(g2.shape[0])
    c = _block_rows(B)
    if c >= A:
        return _coulomb_rows(g1, P1, KN1, g2, P2, KN2, omega)

    # Fill V a row block at a time: the single-shot build holds ~13 (A,B) temporaries. `fori_loop`
    # + `dynamic_update_slice` rather than `lax.map`, whose trailing slice off the padding would
    # allocate a second (A,B). The last block is clamped to start at A-c, recomputing a few rows
    # instead of needing a second kernel shape.
    return _blocked_block(g1, P1, KN1, g2, P2, KN2, c, omega)


@functools.partial(jax.jit, static_argnums=(6, 7))
def _blocked_block(g1, P1, KN1, g2, P2, KN2, c, omega):
    """The row-block loop of :func:`coulomb_block`, compiled as one unit.

    The ``jax.jit`` is required, not cosmetic: eagerly, each ``dynamic_update_slice`` allocates a
    fresh (A,B) and the carry double-buffers. Compiled, XLA aliases it and the update is in place.
    """
    A, B = int(g1.shape[0]), int(g2.shape[0])
    dt = jnp.result_type(g1, P1, KN1, g2, P2, KN2)

    def step(k, V):
        i0 = jnp.minimum(k * c, A - c)
        blk = _coulomb_rows(lax.dynamic_slice_in_dim(g1, i0, c),
                            lax.dynamic_slice_in_dim(P1, i0, c),
                            lax.dynamic_slice_in_dim(KN1, i0, c),
                            g2, P2, KN2, omega)
        return lax.dynamic_update_slice(V, blk.astype(dt), (i0, 0))

    return lax.fori_loop(0, -(-A // c), step, jnp.zeros((A, B), dt))


def full_isobra_block(beta, R_aux, KN_aux, A_pair, P_pair, KN_pair, omega=None):
    """(P|a) for an **isotropic aux** (β) × **full-covariance pair** (Â) — via the eigenframe.

    A (iso | full) Coulomb equals a (iso | *diagonal*) Coulomb in the pair's principal-axis
    frame: eigendecompose ``Â = UΛUᵀ`` (once per pair), rotate the aux–pair displacement
    ``Uᵀ(R−P)``, then use the cheap **per-axis** quadrature (no 3×3 solve/det in the node loop)
    with pair exponents Λ. Optimization over the general 3×3-Schur Coulomb solve
    (``coulomb.stream.full_eri_block_schur``).

    The node loop is a static python unroll in a division-lean form, so XLA fuses every node into
    one kernel that reads ``dp²`` once and keeps Γ in registers. The node sum carries an analytic
    ``custom_vjp`` (``_isobra_nodes``) to avoid taping ~8 (A,B) primals per node.

    Degeneracy: differentiable ``eigh`` has 1/(λᵢ−λⱼ) backward terms, so the gradient is NaN at
    exactly degenerate Â (Â = αI, the isotropic init). Forward is always exact, and the gradient is
    well behaved for any anisotropic Â, so training only needs ``init_model``'s ``aniso_jitter`` to
    step off gap=0. Use ``coulomb.stream.full_eri_block_schur`` to differentiate through an exactly
    isotropic Â.
    """
    Lam, U = jnp.linalg.eigh(A_pair)                       # (B,3) eigenvalues, (B,3,3)
    d = R_aux[:, None, :] - P_pair[None, :, :]            # (A,B,3)
    dp2 = jnp.einsum("bki,abk->abi", U, d) ** 2           # (Uᵀd)² — the one big array
    # The only range-separation switch on this path: which node table integrates t.
    coul = (_lr_nodes(float(omega)) if omega else _isobra_nodes)(beta, dp2, Lam)
    pref = 2.0 * jnp.pi ** 3 / jnp.sqrt(jnp.pi)            # π³ from ∏π/√Γ · 2/√π
    return (KN_aux[:, None] * KN_pair[None, :]) * pref * coul


@functools.lru_cache(maxsize=8)
def _make_isobra_nodes(t_list, w_list):
    """Build the node kernel for ONE quadrature table, with its analytic reverse.

    A factory because range separation changes only the quadrature table: the short-range kernel
    integrates t over [0,∞) and the long-range one over [0,ω], with identical Γ algebra and
    derivatives.

    The tables are tuples so this is hashable and `lru_cache` returns the same function object
    for a given table. That matters: a fresh `custom_vjp` per call would defeat jit's cache and
    retrace the unrolled loop on every step.
    """
    @jax.custom_vjp
    def nodes(beta, dp2, Lam):
        """Σ_t w_t (Γ₀Γ₁Γ₂)^{-1/2} exp(−t² Σ_k W_k/Γ_k), Γ_k = βΛ_k + t²(β+Λ_k), W_k = βΛ_k dp2_k.

        Static python unroll (one fused kernel; Γ built in-register from the small per-side
        arrays — only dp2 (A,B,3) is big) + analytic backward below (no per-node tape)."""
        b = beta[:, None]                                      # (A,1)
        L0, L1, L2 = Lam[None, :, 0], Lam[None, :, 1], Lam[None, :, 2]    # (1,B)
        coul = jnp.zeros((beta.shape[0], Lam.shape[0]))
        for t, w in zip(t_list, w_list):                       # static unroll → one fused kernel
            t2 = t * t
            G0 = b * L0 + t2 * (b + L0)                        # Γ_k
            G1 = b * L1 + t2 * (b + L1)
            G2 = b * L2 + t2 * (b + L2)
            invp = 1.0 / (G0 * G1 * G2)                        # ONE division per (A,B)
            num = b * (L0 * dp2[..., 0] * G1 * G2
                       + L1 * dp2[..., 1] * G0 * G2
                       + L2 * dp2[..., 2] * G0 * G1)
            coul = coul + w * jnp.sqrt(invp) * jnp.exp(-t2 * num * invp)
        return coul

    def nodes_fwd(beta, dp2, Lam):
        return nodes(beta, dp2, Lam), (beta, dp2, Lam)

    def nodes_bwd(res, gI):
        """Analytic reverse: with I_t = w P^{-1/2} e^{−t²S}, P = ∏Γ_k, S = ΣW_k/Γ_k,
            ∂I/∂Γ_k = I·(−½ + t² W_k/Γ_k)/Γ_k ,   ∂I/∂W_k = −I·t²/Γ_k ,
            Γ_k → (β: Λ_k+t², Λ_k: β+t²),  W_k = βΛ_k dp2_k → (β: Λ_k dp2_k, Λ_k: β dp2_k,
            dp2_k: βΛ_k). One fused unrolled loop, cotangents reduced on the fly.

        The table enters only through t and w, never through a derivative, so the same reverse
        serves both ranges."""
        beta, dp2, Lam = res
        b = beta[:, None]
        Ls = (Lam[None, :, 0], Lam[None, :, 1], Lam[None, :, 2])
        d_beta = jnp.zeros_like(b)                             # (A,1) — reduce over B at the end
        d_Lam = [jnp.zeros((1, Lam.shape[0]))] * 3             # (1,B) each — reduce over A
        d_dp2 = [jnp.zeros((beta.shape[0], Lam.shape[0]))] * 3
        for t, w in zip(t_list, w_list):
            t2 = t * t
            G = [b * L + t2 * (b + L) for L in Ls]             # Γ_k, in-register
            invG = [1.0 / Gk for Gk in G]
            W = [b * L * dp2[..., k] for k, L in enumerate(Ls)]
            S = W[0] * invG[0] + W[1] * invG[1] + W[2] * invG[2]
            I = w * jnp.sqrt(invG[0] * invG[1] * invG[2]) * jnp.exp(-t2 * S)
            gIt = gI * I                                       # upstream × node value
            db = jnp.zeros_like(gIt)
            for k, L in enumerate(Ls):
                dIdG = gIt * (t2 * W[k] * invG[k] - 0.5) * invG[k]      # ∂I/∂Γ_k · gI
                dIdW = -gIt * t2 * invG[k]                              # ∂I/∂W_k · gI
                db = db + dIdG * (L + t2) + dIdW * L * dp2[..., k]
                d_Lam[k] = d_Lam[k] + jnp.sum(dIdG * (b + t2) + dIdW * b * dp2[..., k],
                                              axis=0, keepdims=True)
                d_dp2[k] = d_dp2[k] + dIdW * b * L
            d_beta = d_beta + jnp.sum(db, axis=1, keepdims=True)
        return (d_beta[:, 0], jnp.stack(d_dp2, axis=-1),
                jnp.stack([dL[0] for dL in d_Lam], axis=-1))

    nodes.defvjp(nodes_fwd, nodes_bwd)
    return nodes


_isobra_nodes = _make_isobra_nodes(tuple(_QUAD_T_LIST), tuple(_QUAD_W_LIST))


@functools.lru_cache(maxsize=8)
def _lr_nodes(omega: float, n_quad: int = 12):
    """The node kernel for erf(ωr)/r — the same kernel on the segment table [0, ω]."""
    t, w = _gauss_legendre_segment(float(omega), int(n_quad))
    return _make_isobra_nodes(tuple(float(x) for x in t), tuple(float(x) for x in w))


def aux_quantities(aux):
    """(β, R, norm) for a floating isotropic-Gaussian aux basis (each a single function)."""
    return aux.alpha, aux.centers, aux.norm                # K=1 (no pair product)


def shift_diag(V, lam):
    """``V + lam·I`` by adding to the diagonal.

    Never ``V + lam * jnp.eye(n)``: that materializes a dense identity the size of V — 20.4 GiB
    at N_aux = 52328 — plus a third array for the sum, to express a diagonal shift.
    """
    i = jnp.arange(V.shape[0])
    return V.at[i, i].add(lam)


@functools.partial(jax.jit, donate_argnums=0)
def _chol_donated(V):
    """Lower Cholesky of an ALREADY-shifted V, donating it so XLA factors into its buffer."""
    return jax.scipy.linalg.cholesky(V, lower=True)


def aux_cholesky(aux, lam, omega=None):
    """Lower Cholesky factor of ``(V + lam·I)``.

    The factorization donates its input, so V never escapes and no dense identity is built.
    """
    return _chol_donated(shift_diag(aux_metric(aux, omega=omega), lam))


def aux_metric(aux, omega=None):
    """V_PQ = (P|Q), the auxiliary Coulomb metric (N_aux, N_aux).

    Always exact Boys — the aux basis is **isotropic** for every primary backend, so the
    aux–aux Coulomb needs no quadrature (see module discussion: iso aux fits the density,
    keeps the metric exact/cheap, and the inner-max robust).
    """
    b, R, n = aux_quantities(aux)
    return coulomb_block(b, R, n, b, R, n, omega=omega)


def _gamma(aux_q, splat_q, pi, pj, q, chunk):
    """γ_P = Σ_a (P|a) q_a, streamed over pair chunks (small fused kernels → fast compile).

    ``aux_q`` = isotropic aux quantities; ``splat_q`` = per-splat quantities, built once per energy
    call outside the remat'd streams. The aux is always isotropic; only the pair side is anisotropic.
    """
    N = pi.shape[0]
    pad = (-N) % chunk
    pic = jnp.concatenate([pi, jnp.zeros(pad, pi.dtype)]).reshape(-1, chunk)
    pjc = jnp.concatenate([pj, jnp.zeros(pad, pj.dtype)]).reshape(-1, chunk)
    qc = jnp.concatenate([q, jnp.zeros(pad, q.dtype)]).reshape(-1, chunk)

    def body(acc, c):
        ci, cj, cq = c
        pair_q = _full_pair_quantities(splat_q, ci, cj)    # (chunk, ...)
        block = full_isobra_block(*aux_q, *pair_q)         # (N_aux, chunk)
        return acc + block @ cq, None

    # jax.checkpoint on the body: WITHOUT it, lax.scan stores stacked per-chunk residuals for
    # reverse-mode → O(N_pair·N_aux) memory (chunk-independent), which OOMs the anisotropic
    # (diag/full) aux-gradient at M≳400 even though the forward streams in O(M²). Remat trades
    # ~2× backward compute (recompute each chunk in reverse) for O(chunk) backward memory.
    # (A streaming custom_vjp would remove the recompute; this is the unblocking baseline.)
    gamma, _ = lax.scan(jax.checkpoint(body), jnp.zeros(aux_q[0].shape[0]), (pic, pjc, qc))
    return gamma


def df_coulomb_energy(splats, P, aux, lam=1e-8, unique=True, chunk=2048, cholV=None):
    """Robust DF estimate E_J^DF = ½ γᵀ (V+λI)⁻¹ γ  (= max_d J_rob, a lower bound on E_J).

    γ_P = Σ_a q_a (P|a) with q_a = w_a·P[i_a,j_a], streamed in pair chunks of ``chunk`` to keep each
    fused kernel small. The Tikhonov λ regularizes the solve and keeps it differentiable — in the
    splat params, the MO coeffs, and the aux params.

    ``cholV`` = the lower Cholesky factor of (V+λI), precomputed when the aux is frozen. Given it the
    per-step solve is one ``cho_solve`` (O(N_aux²)) rather than O(N_aux³); differentiability in the
    aux params is forfeited, which is the frozen-aux case.
    """
    M = int(splats.n_basis)
    pi, pj, w = make_pairs(M, unique)
    q = w * P[pi, pj]                                       # (N_pair,)
    splat_q = full_quantities(splats)                       # A-chart hoisted out of the stream
    gamma = _gamma(aux_quantities(aux), splat_q, pi, pj, q, min(chunk, pi.shape[0]))
    if cholV is not None:                                  # frozen aux: reuse the one-time factor
        d = jax.scipy.linalg.cho_solve((cholV, True), gamma)
    else:
        V = aux_metric(aux)                                # (N_aux, N_aux), exact Boys
        d = jnp.linalg.solve(shift_diag(V, lam), gamma)
    return 0.5 * jnp.dot(gamma, d)


# ---------------------------------------------------------------------------
#  RI-K: exact exchange via density fitting (dense, O(N⁴) — correctness first)
# ---------------------------------------------------------------------------
def build_three_center(aux, splats):
    """Dense DF 3-center tensor ``T[P,i,k] = (g_i g_k | χ_P)`` → ``(N_aux, M, M)``.

    Reuses the (aux|pair) block kernel over **all M² ordered
    pairs** (g_i g_k is symmetric in i,k; full M² for simplicity). The same integral the
    Coulomb γ uses — exchange is just a different *contraction* of it. O(N_aux·M²) memory; the
    correctness-first RI-K building block (stream over aux later for scaling).
    """
    M = int(splats.n_basis)
    idx = jnp.arange(M)
    pair_q = _full_pair_quantities(full_quantities(splats), jnp.repeat(idx, M), jnp.tile(idx, M))
    T = full_isobra_block(*aux_quantities(aux), *pair_q)    # (N_aux, M²)
    return T.reshape(T.shape[0], M, M)


def _exchange_M_streamed(aux_q, splat_q, M, C_occ, row_chunk, occ_chunk, omega=None):
    """M_PQ = Σ_oo' D[P,o,o'] D[Q,o,o'] in **O(N_aux²) memory** by streaming over OCCUPIED-ORBITAL
    chunks (outer) so the full D=(N_aux,n_occ,n_occ)=O(N³) accumulator is never materialized.

    Per orbital chunk the bra-rows are re-streamed to build only D[:,chunk,:], then
    M += Σ_{o∈chunk} D_o D_oᵀ. ``occ_chunk`` is the memory/recompute knob: the 3-center is recomputed
    ⌈n_occ/occ_chunk⌉ times, and ``occ_chunk ≥ n_occ`` recovers the dense single-pass D bit-exactly.
    ``jax.checkpoint`` on both bodies keeps reverse mode O(N²) too. Compute stays the intrinsic
    O(N⁴) of exact exchange; only memory is reduced."""
    N_aux = aux_q[0].shape[0]
    n_occ = C_occ.shape[1]
    allk = jnp.arange(M)
    rpad = (-M) % row_chunk
    rows = jnp.concatenate([jnp.arange(M), jnp.zeros(rpad, jnp.int32)]).reshape(-1, row_chunk)
    rvalid = jnp.concatenate([jnp.ones(M, bool), jnp.zeros(rpad, bool)]).reshape(-1, row_chunk)
    opad = (-n_occ) % occ_chunk
    o_idx = jnp.concatenate([jnp.arange(n_occ), jnp.zeros(opad, jnp.int32)]).reshape(-1, occ_chunk)
    o_val = jnp.concatenate([jnp.ones(n_occ, bool), jnp.zeros(opad, bool)]).reshape(-1, occ_chunk)

    def outer(M_acc, ocarry):
        oc, ov = ocarry                                                # (occ_chunk,)
        Coc = jnp.where(ov[None, :], C_occ[:, oc], 0.0)                # (M, occ_chunk): this chunk's MOs

        def inner(D_acc, rcarry):
            ridx, rmask = rcarry                                       # (row_chunk,)
            pair_q = _full_pair_quantities(splat_q, jnp.repeat(ridx, M), jnp.tile(allk, row_chunk))
            T = full_isobra_block(*aux_q, *pair_q, omega=omega).reshape(N_aux, row_chunk, M)
            TC = jnp.einsum("Aik,kp->Aip", T, C_occ)                   # (N_aux, row_chunk, n_occ): all o'
            Cc = jnp.where(rmask[:, None], Coc[ridx], 0.0)             # (row_chunk, occ_chunk): this o
            return D_acc + jnp.einsum("io,Aip->Aop", Cc, TC), None     # (N_aux, occ_chunk, n_occ)

        D_oc, _ = lax.scan(jax.checkpoint(inner), jnp.zeros((N_aux, occ_chunk, n_occ)), (rows, rvalid))
        return M_acc + jnp.einsum("Aoq,Boq->AB", D_oc, D_oc), None     # (N_aux, N_aux)

    Mmat, _ = lax.scan(jax.checkpoint(outer), jnp.zeros((N_aux, N_aux)), (o_idx, o_val))
    return Mmat


def df_exchange_energy_density(splats, P, aux, lam=1e-8, omega=None):
    """RI-K exact exchange as a function of the DENSITY MATRIX: ``E_K = -¼ Tr(P K)``.

    :func:`df_exchange_energy` is written in terms of ``C_occ`` because that is how occ-RI-K keeps
    its memory down, but an SCF that builds its Fock matrix as ``F = sym(dE/dP)`` needs the energy
    in ``P``. Without this the splat SCF has no exchange term at all and an ``xc=hf`` run returns
    Hartree-only eigenvalues while reporting them as Hartree-Fock.

    With ``T[P,p,r] = (g_p g_r|χ_P)`` and ``Y[P] = T[P] P``::

        K = Σ_PQ V⁻¹_PQ T_P P T_Q,    E_K = -¼ Tr(P K) = -¼ Σ_PQ V⁻¹_PQ Tr(Y_Q Y_P)

    with ``Y_P = T_P P``. O(N_aux·M²) memory — dense on purpose, for the
    small systems where orbital energies are the point. Use the streamed forms at protein scale.
    """
    T = build_three_center(aux, splats)                     # (N_aux, M, M), symmetric in (p,r)
    Y = jnp.einsum("Ppr,rs->Pps", T, P)                     # Y_P = T_P P
    # Tr(P T_P P T_Q) = Tr(Y_Q Y_P): both P and T_P are symmetric, so P T_P = Y_P^T.
    W = jnp.einsum("Pba,Qab->PQ", Y, Y)
    Vs = shift_diag(aux_metric(aux, omega=omega), lam)
    return -0.25 * jnp.trace(jnp.linalg.solve(Vs, W))


def df_exchange_energy(splats, C_occ, aux, lam=1e-8, chunk=64, occ_chunk=16, omega=None):
    """RI-K exact-exchange energy E_K ≈ −¼ Σ_pqrs P_pq P_rs (pr|qs) via DF, closed-shell
    P = 2·C_occ C_occᵀ.

    occ-RI-K: with the 3-center T[P,i,k]=(g_i g_k|χ_P) and metric V_PQ=(P|Q),
        D[P,o,o'] = Σ_ik C_io C_ko' T[P,i,k]          ( = Cᵀ T_P C, per aux )
        E_K = −Tr( V⁻¹ M ),   M_PQ = Σ_oo' D[P,o,o'] D[Q,o,o'] .
    Exact when the aux spans the splat products. Memory is O(N²) via ``_exchange_M_streamed``, which
    streams the M-build over occupied-orbital chunks so the O(N³) (N_aux,n_occ,n_occ) tensor is never
    held. ``C_occ`` = Löwdin-orthonormal occupied coefficients (occupation 2 folded into the −¼·2² =
    −1 prefactor); ``chunk`` = bra-rows per scan step; ``occ_chunk`` = the memory/recompute knob.
    Compute is the intrinsic O(N⁴) of exact exchange; only memory is addressed here.
    """
    M = int(splats.n_basis)
    n_occ = C_occ.shape[1]
    Mmat = _exchange_M_streamed(aux_quantities(aux), full_quantities(splats), M, C_occ,
                                min(chunk, M), min(occ_chunk, n_occ), omega=omega)
    V = aux_metric(aux, omega=omega)
    return -jnp.trace(jnp.linalg.solve(shift_diag(V, lam), Mmat))


# ---------------------------------------------------------------------------
#  SCREENED DF 3-center: stream only the SIGNIFICANT bra-pairs (g_i g_j vanishes when i,j are
#  far apart → screened by the SAME overlap neighbor list as the one-electron operators). Cuts the
#  M² bra-pairs of γ / D to O(M·degree). The metric solve V⁻¹ stays dense O(N_aux³) (Phase 3).
#  ``pi, pj`` = ALL ordered significant pairs (from ``screening.neighbor_pairs``); ``valid`` masks
#  the static-shape padding. The aux metric V is unchanged (the aux is dense/global).
# ---------------------------------------------------------------------------
def df_coulomb_gamma_screened(splats, C_occ, occ, aux, pi, pj, chunk=2048, valid=None,
                              skip_pad=False):
    """γ_P = Σ_a (P|a) q_a over the significant bra-pairs (q_a = Σ_o occ_o C_io C_jo, formed per
    chunk inside the stream so the (n_pair, n_occ) tape never materializes). This is the
    PAIR-LINEAR half of ``df_coulomb_energy_screened`` — a plain sum over pairs, so it shards
    across devices by splitting (pi,pj,valid) + one ``psum``; the metric solve
    (``df_coulomb_solve``) is the small replicated tail. Returns γ (N_aux,)."""
    aux_q = aux_quantities(aux)
    splat_q = full_quantities(splats)                       # A-chart hoisted out of the stream
    N_aux = aux_q[0].shape[0]
    chunk = min(chunk, pi.shape[0])
    Npair = pi.shape[0]
    pad = (-Npair) % chunk
    base_valid = jnp.ones(Npair, bool) if valid is None else valid
    pic = jnp.concatenate([pi, jnp.zeros(pad, pi.dtype)]).reshape(-1, chunk)
    pjc = jnp.concatenate([pj, jnp.zeros(pad, pj.dtype)]).reshape(-1, chunk)
    vc = jnp.concatenate([base_valid, jnp.zeros(pad, bool)]).reshape(-1, chunk)

    def chunk_gamma(ci, cj, cv):
        # cv: bool mask (0/1) or unique-pair weights (0/1/2) — see screening.neighbor_pairs
        q = cv.astype(C_occ.dtype) * jnp.sum(occ[None, :] * C_occ[ci] * C_occ[cj], axis=1)   # (chunk,)
        block = full_isobra_block(*aux_q, *_full_pair_quantities(splat_q, ci, cj))      # (N_aux, chunk)
        return block @ q

    def body(acc, carry):
        ci, cj, cv = carry
        if skip_pad:                       # appended padding ⇒ trailing chunks are all-dummy
            g = lax.cond(jnp.any(cv != 0), chunk_gamma, lambda *_: jnp.zeros(N_aux), ci, cj, cv)
        else:
            g = chunk_gamma(ci, cj, cv)
        return acc + g, None

    gamma, _ = lax.scan(jax.checkpoint(body), jnp.zeros(N_aux), (pic, pjc, vc))
    return gamma


def df_coulomb_solve(gamma, aux, lam=1e-8, cholV=None):
    """Replicated metric tail of RI-J: E_J^DF = ½ γᵀ (V+λI)⁻¹ γ, V = (P|Q) the aux metric.

    ``cholV`` = the lower Cholesky factor of (V+λI), precomputed once per refresh when
    the aux is frozen — the solve drops to O(N_aux²) cho_solve (identical value). With a factor in
    hand the energy goes through ``cho_solve``."""
    if cholV is not None:
        return 0.5 * jnp.dot(gamma, jax.scipy.linalg.cho_solve((cholV, True), gamma))
    V = aux_metric(aux)
    d = jnp.linalg.solve(shift_diag(V, lam), gamma)
    return 0.5 * jnp.dot(gamma, d)


def df_coulomb_energy_screened(splats, C_occ, occ, aux, pi, pj, lam=1e-8, chunk=2048, valid=None,
                               cholV=None, skip_pad=False):
    """E_J^DF over the significant bra-pairs only. ``q_a = P[i_a,j_a] = Σ_o occ_o C_io C_jo`` is
    computed **per chunk inside the stream** (not once over all pairs) so its (n_pair, n_occ) tape
    never materializes — the screened analog of the dense ``_gamma`` reading P[i,j]. Equals
    ``df_coulomb_energy`` when (pi,pj) is the full ordered pair set. (= the pair-shardable
    ``df_coulomb_gamma_screened`` + the replicated ``df_coulomb_solve``.)"""
    gamma = df_coulomb_gamma_screened(splats, C_occ, occ, aux, pi, pj, chunk=chunk, valid=valid,
                                      skip_pad=skip_pad)
    return df_coulomb_solve(gamma, aux, lam=lam, cholV=cholV)


def _exchange_M_streamed_screened(aux_q, splat_q, C_occ, pi, pj,
                                  pair_chunk, occ_chunk, valid, omega=None):
    """Screened analog of ``_exchange_M_streamed``: M_PQ over the significant bra-pairs only, in
    O(N_aux²) memory by streaming the M-build over occupied-orbital chunks (never holding the full
    (N_aux,n_occ,n_occ) D). Per orbital chunk, stream the pair list to build D[:,chunk,:] (the
    screened D[P,o,o']=Σ_{significant (i,k)} C_io C_ko' (g_ig_k|χ_P)) and accumulate
    M += Σ_{o∈chunk} D_o D_oᵀ. Equals ``einsum('Aop,Bop->AB', D, D)`` over the significant pairs.
    """
    N_aux, n_occ = aux_q[0].shape[0], C_occ.shape[1]
    Npair = pi.shape[0]
    ppad = (-Npair) % pair_chunk
    pic = jnp.concatenate([pi, jnp.zeros(ppad, pi.dtype)]).reshape(-1, pair_chunk)
    pjc = jnp.concatenate([pj, jnp.zeros(ppad, pj.dtype)]).reshape(-1, pair_chunk)
    if valid is None:
        valid = jnp.ones(Npair, bool)
    vc = jnp.concatenate([valid, jnp.zeros(ppad, bool)]).reshape(-1, pair_chunk)
    opad = (-n_occ) % occ_chunk
    o_idx = jnp.concatenate([jnp.arange(n_occ), jnp.zeros(opad, jnp.int32)]).reshape(-1, occ_chunk)
    o_val = jnp.concatenate([jnp.ones(n_occ, bool), jnp.zeros(opad, bool)]).reshape(-1, occ_chunk)

    def outer(M_acc, ocarry):
        oc, ov = ocarry                                                # (occ_chunk,)

        def inner(D_acc, carry):
            ci, cj, cv = carry                                         # (pair_chunk,)
            block = full_isobra_block(*aux_q, *_full_pair_quantities(splat_q, ci, cj), omega=omega)
            Ci = jnp.where((cv[:, None] & ov[None, :]), C_occ[ci][:, oc], 0.0)  # (pair_chunk, occ_chunk)
            Ck = C_occ[cj]                                             # (pair_chunk, n_occ): all o'
            return D_acc + jnp.einsum("Ap,po,pq->Aoq", block, Ci, Ck), None      # (N_aux, occ_chunk, n_occ)

        D_oc, _ = lax.scan(jax.checkpoint(inner), jnp.zeros((N_aux, occ_chunk, n_occ)), (pic, pjc, vc))
        return M_acc + jnp.einsum("Aoq,Boq->AB", D_oc, D_oc), None
    Mmat, _ = lax.scan(jax.checkpoint(outer), jnp.zeros((N_aux, N_aux)), (o_idx, o_val))
    return Mmat


def df_exchange_energy_screened(splats, C_occ, aux, pi, pj,
                                lam=1e-8, chunk=2048, valid=None, occ_chunk=16, omega=None):
    """RI-K E_K over the significant bra-pairs only, **O(N²) memory** (streamed M-build over
    occupied-orbital chunks, ``_exchange_M_streamed_screened``). Equals ``df_exchange_energy`` when
    (pi,pj) is the full ordered pair set. ``occ_chunk`` = the memory/recompute knob (O(N²) for a
    constant; ≥ n_occ builds D in a single pass)."""
    n_occ = C_occ.shape[1]
    Mmat = _exchange_M_streamed_screened(aux_quantities(aux), full_quantities(splats), C_occ,
                                         pi, pj, min(chunk, pi.shape[0]), min(occ_chunk, n_occ),
                                         valid, omega=omega)
    V = aux_metric(aux, omega=omega)
    return -jnp.trace(jnp.linalg.solve(shift_diag(V, lam), Mmat))
