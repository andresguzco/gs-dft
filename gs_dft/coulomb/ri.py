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
    "metric_factor", "df_coulomb_solve_rows",
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
    """Build the node kernel for ONE quadrature table, with its analytic reverse; returns the
    ``custom_vjp`` node sum and the one-loop ``value_and_reverse`` the RI-J stream's reverse uses.

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

        Static unroll (one fused kernel), no float64 division: 1/(Γ₀Γ₁Γ₂) is the square of the
        rsqrt the integrand takes anyway."""
        b = beta[:, None]                                      # (A,1)
        Ls = (Lam[None, :, 0], Lam[None, :, 1], Lam[None, :, 2])    # (1,B)
        bL = [b * L for L in Ls]
        bpL = [b + L for L in Ls]
        W = [bL[k] * dp2[..., k] for k in range(3)]            # node-independent
        coul = jnp.zeros((beta.shape[0], Lam.shape[0]))
        for t, w in zip(t_list, w_list):                       # static unroll → one fused kernel
            t2 = t * t
            G0, G1, G2 = (bL[k] + t2 * bpL[k] for k in range(3))
            r = lax.rsqrt(G0 * G1 * G2)
            num = W[0] * (G1 * G2) + W[1] * (G0 * G2) + W[2] * (G0 * G1)
            coul = coul + w * r * jnp.exp(-t2 * num * (r * r))
        return coul

    def nodes_fwd(beta, dp2, Lam):
        return nodes(beta, dp2, Lam), (beta, dp2, Lam)

    def value_and_reverse(beta, dp2, Lam, gI):
        """The node sum and its reverse for the cotangent gI, in one loop:
        ``(coul, d_beta, d_dp2, d_Lam)``. With I_t = w P^{-1/2} e^{−t²S}, P = ∏Γ_k, S = ΣW_k/Γ_k,
            ∂I/∂Γ_k = I·(t² W_k/Γ_k − ½)/Γ_k ,   ∂I/∂W_k = −I·t²/Γ_k ,
            Γ_k → (β: Λ_k+t², Λ_k: β+t²),  W_k = βΛ_k dp2_k → (β: Λ_k dp2_k, Λ_k: β dp2_k,
            dp2_k: βΛ_k).
        The node sums Σ I, Σ∂I/∂Γ_k, Σt²∂I/∂Γ_k and Σt²I/Γ_k are ONE variadic reduction over a
        trailing node axis (division-free: 1/Γ_k = Γ_iΓ_j/P), so the exp per node is evaluated once
        inside a single row-reduction kernel: 1.0 ms on an 832x512 block against 5.0 ms for the
        unrolled loop, which XLA cut into ~25 kernels (experiments/kernels/node_bench.py). gI and
        the chain to (β, Λ, dp2) are applied once after it, and each output is reduced once.

        The table enters only through t and w, never through a derivative, so the same reverse
        serves both ranges."""
        t2 = jnp.asarray(t_list) ** 2                          # (Q,), the trailing node axis
        w = jnp.asarray(w_list)
        b = beta[:, None]
        Ls = (Lam[None, :, 0], Lam[None, :, 1], Lam[None, :, 2])
        bL = [b * L for L in Ls]
        W = [bL[k] * dp2[..., k] for k in range(3)]
        G = [(b * L)[..., None] + t2 * (b + L)[..., None] for L in Ls]    # (A,B,Q)
        r = lax.rsqrt(G[0] * G[1] * G[2])
        r2 = r * r
        invG = (G[1] * G[2] * r2, G[0] * G[2] * r2, G[0] * G[1] * r2)
        I = w * r * jnp.exp(-t2 * (W[0][..., None] * invG[0] + W[1][..., None] * invG[1]
                                   + W[2][..., None] * invG[2]))
        terms = [I]
        for k in range(3):
            c = t2 * (I * invG[k])                             # t² I/Γ_k
            dG = c * W[k][..., None] * invG[k] - 0.5 * (I * invG[k])    # ∂I/∂Γ_k
            terms += [dG, t2 * dG, c]
        sums = lax.reduce(tuple(terms), tuple(jnp.zeros((), I.dtype) for _ in terms),
                          lambda a, c: tuple(x + y for x, y in zip(a, c)), (2,))
        coul, hG, hGt, hW = sums[0], sums[1::3], sums[2::3], sums[3::3]
        d_beta = 0.0
        d_Lam, d_dp2 = [], []
        for k, L in enumerate(Ls):
            x = dp2[..., k]
            d_beta = d_beta + L * hG[k] + hGt[k] - hW[k] * (L * x)
            d_Lam.append(jnp.sum(gI * (b * hG[k] + hGt[k] - hW[k] * (b * x)), axis=0))
            d_dp2.append(-gI * hW[k] * bL[k])
        return (coul, jnp.sum(gI * d_beta, axis=1), jnp.stack(d_dp2, axis=-1),
                jnp.stack(d_Lam, axis=-1))

    def nodes_bwd(res, gI):
        return value_and_reverse(*res, gI)[1:]

    nodes.defvjp(nodes_fwd, nodes_bwd)
    return nodes, value_and_reverse


_isobra_nodes, _isobra_value_and_reverse = _make_isobra_nodes(tuple(_QUAD_T_LIST),
                                                              tuple(_QUAD_W_LIST))


@functools.lru_cache(maxsize=8)
def _lr_nodes(omega: float, n_quad: int = 12):
    """The node kernel for erf(ωr)/r — the same kernel on the segment table [0, ω]."""
    t, w = _gauss_legendre_segment(float(omega), int(n_quad))
    return _make_isobra_nodes(tuple(float(x) for x in t), tuple(float(x) for x in w))[0]


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


@functools.partial(jax.jit, static_argnames=("omega",))
def _shifted_metric(b, R, n, lam, omega=None):
    """V + lam·I in one compiled unit, so the shift lands in V's buffer instead of a copy of it."""
    return shift_diag(coulomb_block(b, R, n, b, R, n, omega=omega), lam)


def aux_cholesky(aux, lam, omega=None):
    """Lower Cholesky factor of ``(V + lam·I)``.

    V is built and shifted in one compiled unit and the factorization donates it, so one N_aux²
    buffer serves all three, and no dense identity is built.
    """
    b, R, n = aux_quantities(aux)
    return _chol_donated(_shifted_metric(b, R, n, lam, omega=omega))


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
_ISOBRA_PREF = 2.0 * np.pi ** 2.5          # π³ · 2/√π, as in full_isobra_block


_AUX_TILE = 2048
"""Aux functions per tile of the γ stream's node kernels: each pair chunk is swept over the aux in
tiles of this many rows, so the working set is (tile, chunk), not (N_aux, chunk). Without the tile,
XLA lays the reverse's node axis out in memory at large N_aux: nine (N_aux, chunk, 64) arrays, 46 GiB
per device at insulin (N_aux 21,026, chunk 512) whatever the device count. Tiled, the same
kernels also run faster (experiments/kernels/node_bench.py)."""


def _pair_geometry(splat_q, Ci, Cj, occ, ci, cj, cv):
    """The pair half of the γ geometry for one chunk: Λ and U (Â_ij = UΛUᵀ), the product centre P
    and h = w·P_ij·N_iN_jK. Takes the chunk's coefficient rows ``Ci = C[ci]``, ``Cj = C[cj]``, so the
    reverse adds their cotangents into its running (M, n_occ) total in place, instead of making a
    dense (M, n_occ) cotangent per chunk."""
    Aij, Pa, KN = _full_pair_quantities(splat_q, ci, cj)
    Lam, U = jnp.linalg.eigh(Aij)
    h = cv.astype(Ci.dtype) * jnp.sum(occ[None, :] * Ci * Cj, axis=1) * KN
    return Lam, U, Pa, h


def _rotated_offsets(R, U, Pa):
    """dp2 = (Uᵀ(R − P))², (tile, chunk, 3): an aux tile against a pair chunk."""
    return jnp.einsum("bki,abk->abi", U, R[:, None, :] - Pa[None, :, :]) ** 2


def _aux_tiles(aux_q, tile):
    """(β, R, KN) padded to whole tiles, as (n_tile, tile, ...). A pad row has KN = 0 (and β = 1,
    R = 0, so its kernels stay finite): it adds nothing to γ and draws no cotangent."""
    beta, R, KN = aux_q
    A = beta.shape[0]
    t = int(min(tile, A))
    pad = (-A) % t
    cut = lambda x, v: jnp.concatenate([x, jnp.full((pad,) + x.shape[1:], v, x.dtype)]).reshape(
        (-1, t) + x.shape[1:])
    return cut(beta, 1.0), cut(R, 0.0), cut(KN, 0.0)


def _live_update(fn, acc, cv, skip_pad):
    """``fn(acc)``, or ``acc`` unchanged for an all-padding chunk when ``skip_pad``: the running
    total goes through the cond, so a live chunk updates it in place."""
    if skip_pad:
        return lax.cond(jnp.any(cv != 0), fn, lambda a: a, acc)
    return fn(acc)


def _live(fn, zero, cv, skip_pad):
    """``fn()``, or ``zero`` for an all-padding chunk when ``skip_pad`` (the pad is appended).
    Never differentiated through (the streams carry their own VJP), so the cond costs no memory."""
    if skip_pad:
        return lax.cond(jnp.any(cv != 0), fn, lambda: zero)
    return fn()


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8))
def _gamma_stream(aux_q, splat_q, C_occ, occ, pi, pj, valid, chunk, skip_pad):
    """γ = Σ_chunks (P|a) h_a, each chunk swept over the aux in tiles of ``_AUX_TILE`` rows. The
    reverse is one pass over the chunks too (no tape, no remat): per chunk the pair geometry's VJP,
    and per tile one node loop that makes the value and its reverse together."""
    pic, pjc, vc = _pair_chunks(pi, pj, valid, chunk)
    tiles = _aux_tiles(aux_q, _AUX_TILE)
    A = aux_q[0].shape[0]
    zero = jnp.zeros(tiles[0].size)

    def body(acc, c):
        ci, cj, cv = c

        def fn():
            Lam, U, Pa, h = _pair_geometry(splat_q, C_occ[ci], C_occ[cj], occ, ci, cj, cv)
            return lax.map(lambda t: _ISOBRA_PREF * t[2] * (
                _isobra_nodes(t[0], _rotated_offsets(t[1], U, Pa), Lam) @ h), tiles).reshape(-1)

        return acc + _live(fn, zero, cv, skip_pad), None

    return lax.scan(body, zero, (pic, pjc, vc))[0][:A]


def _gamma_stream_fwd(aux_q, splat_q, C_occ, occ, pi, pj, valid, chunk, skip_pad):
    out = _gamma_stream(aux_q, splat_q, C_occ, occ, pi, pj, valid, chunk, skip_pad)
    return out, (aux_q, splat_q, C_occ, occ, pi, pj, valid)


def _gamma_stream_bwd(chunk, skip_pad, res, gbar):
    aux_q, splat_q, C_occ, occ, pi, pj, valid = res
    pic, pjc, vc = _pair_chunks(pi, pj, valid, chunk)
    tiles = _aux_tiles(aux_q, _AUX_TILE)
    A = aux_q[0].shape[0]
    gtile = jnp.concatenate([gbar, jnp.zeros(tiles[0].size - A, gbar.dtype)]).reshape(tiles[0].shape)
    z = lambda x: jnp.zeros(jnp.shape(x), jnp.result_type(x))   # zeros_like would carry a mesh sharding into shard_map
    zero = jax.tree.map(z, (tiles, splat_q, C_occ, occ))

    def body(acc, c):
        ci, cj, cv = c

        def fn(acc):
            (Lam, U, Pa, h), pvjp = jax.vjp(
                lambda s, a, b, o: _pair_geometry(s, a, b, o, ci, cj, cv),
                splat_q, C_occ[ci], C_occ[cj], occ)

            def tile(pair_ct, t):
                beta, R, KN, gb = t
                dp2, ovjp = jax.vjp(_rotated_offsets, R, U, Pa)
                u = _ISOBRA_PREF * gb * KN                                   # (tile,)
                coul, d_beta, d_dp2, d_Lam = _isobra_value_and_reverse(beta, dp2, Lam,
                                                                       u[:, None] * h[None, :])
                d_R, d_U, d_Pa = ovjp(d_dp2)
                pair_ct = jax.tree.map(jnp.add, pair_ct, (d_Lam, d_U, d_Pa, u @ coul))
                return pair_ct, (d_beta, d_R, _ISOBRA_PREF * gb * (coul @ h))

            pair_ct, aux_ct = lax.scan(tile, jax.tree.map(z, (Lam, U, Pa, h)), (*tiles, gtile))
            d_s, d_Ci, d_Cj, d_o = pvjp(pair_ct)
            a_aux, a_s, a_C, a_o = acc
            return (jax.tree.map(jnp.add, a_aux, aux_ct), jax.tree.map(jnp.add, a_s, d_s),
                    a_C.at[ci].add(d_Ci).at[cj].add(d_Cj), a_o + d_o)

        return _live_update(fn, acc, cv, skip_pad), None

    (d_beta, d_R, d_KN), d_s, d_C, d_o = lax.scan(body, zero, (pic, pjc, vc))[0]
    flat = lambda x: x.reshape((-1,) + x.shape[2:])[:A]
    return ((flat(d_beta), flat(d_R), flat(d_KN)), d_s, d_C, d_o, None, None, None)


_gamma_stream.defvjp(_gamma_stream_fwd, _gamma_stream_bwd)


def _pair_chunks(pi, pj, valid, chunk):
    """The pair list padded to a multiple of ``chunk`` (dummy (0,0), weight 0), as (n_chunk, chunk)."""
    n = pi.shape[0]
    pad = (-n) % chunk
    v = jnp.ones(n, bool) if valid is None else valid
    return (jnp.concatenate([pi, jnp.zeros(pad, pi.dtype)]).reshape(-1, chunk),
            jnp.concatenate([pj, jnp.zeros(pad, pj.dtype)]).reshape(-1, chunk),
            jnp.concatenate([v, jnp.zeros(pad, v.dtype)]).reshape(-1, chunk))


def df_coulomb_gamma_screened(splats, C_occ, occ, aux, pi, pj, chunk=2048, valid=None,
                              skip_pad=False):
    """γ_P = Σ_a (P|a) q_a over the significant bra-pairs (q_a = w_a Σ_o occ_o C_io C_jo, formed per
    chunk inside the stream so the (n_pair, n_occ) array never materializes). This is the
    PAIR-LINEAR half of ``df_coulomb_energy_screened`` — a plain sum over pairs, so it shards
    across devices by splitting (pi,pj,valid) + one ``psum``; the metric solve
    (``df_coulomb_solve``) is the small replicated tail. ``valid``: a bool mask (0/1) or unique-pair
    weights (0/1/2) — see screening.neighbor_pairs. Returns γ (N_aux,).

    Streamed with its own VJP (``_gamma_stream``): the reverse is one more pass over the chunks,
    so memory stays at one chunk with or without ``skip_pad``."""
    if valid is None:
        valid = jnp.ones(pi.shape[0], bool)
    return _gamma_stream(aux_quantities(aux), full_quantities(splats), C_occ, occ, pi, pj, valid,
                         int(min(chunk, max(1, pi.shape[0]))), bool(skip_pad))


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


SOLVE_BLOCK = 256
"""Rows per block of the mesh's metric solves (:func:`df_coulomb_solve_rows`). The factor is padded to
whole blocks on every device, so a smaller block wastes fewer rows and takes more sequential steps."""


def factor_rows(n, G):
    """Rows of the metric factor as a mesh of ``G`` devices holds it: ``n`` padded to whole
    :data:`SOLVE_BLOCK` blocks on every device."""
    q = SOLVE_BLOCK * int(G)
    return -(-int(n) // q) * q


@functools.partial(jax.jit, static_argnums=1)
def pad_factor(L, n):
    """``[[L, 0], [0, I]]`` of size ``n``: the pad's identity rows solve to zero against a zero
    right-hand side, so they change nothing."""
    N = L.shape[0]
    i = jnp.arange(N, n)
    return jnp.pad(L, ((0, n - N), (0, n - N))).at[i, i].set(1.0)


def metric_factor(aux, lam, mesh=None):
    """The lower Cholesky factor of ``(V + λI)`` as the step reads it: whole on one device (``mesh``
    None, :func:`df_coulomb_solve`), or on a mesh padded to :func:`factor_rows` and built and factored
    by contiguous row blocks across its devices (:func:`_factor_rows`), each holding only its own
    rows (:func:`df_coulomb_solve_rows`).

    The pad: null aux functions (norm 0), whose metric rows vanish, so V + λI is block diagonal with
    a λI tail and factors into ``[[L, 0], [0, √λ I]]``; the pad's right-hand side is zero, so it
    solves to zero."""
    if mesh is None:
        return aux_cholesky(aux, lam)
    b, R, n = aux_quantities(aux)
    p = factor_rows(b.shape[0], int(mesh.size)) - b.shape[0]
    pad = lambda x, v: jnp.concatenate([x, jnp.full((p,) + x.shape[1:], v, x.dtype)])
    q = (pad(b, 1.0), pad(R, 0.0), pad(n, 0.0))
    from jax.sharding import NamedSharding, PartitionSpec
    if jax.process_count() > 1:                 # multi-node: the inputs as global replicated arrays
        from jax.experimental.multihost_utils import host_local_array_to_global_array
        q = host_local_array_to_global_array(q, mesh, PartitionSpec())
    else:                                       # the refresh builds the aux on one device
        q = jax.device_put(q, NamedSharding(mesh, PartitionSpec()))
    return _factor_rows(*q, jnp.asarray(lam, q[0].dtype), mesh=mesh)


FACTOR_PHASES = 8
"""Fixed column windows of :func:`_factor_rows`: block columns are split into this many phases, and
every update in a phase spans the columns past the phase's first panel (fixed shape, one compile)."""


@functools.partial(jax.jit, static_argnames=("mesh",))
def _factor_rows(b, R, n, lam, mesh):
    """V + λI built and Cholesky-factored by contiguous row blocks over ``mesh`` (axis "g"): the
    right-looking blocked factorization of MAGMA's multi-GPU ``potrf_mgpu``, in one dimension. Per
    block column k: the owner's diagonal block is summed to every device and factored there; each
    device solves its rows of the panel; the panel is gathered; each device updates its own rows
    past it, a row slab at a time. A device holds its rows of V (in place), the panel and one slab.

    Only the lower triangle is ever read (``symmetrize_input=False``); the windowed update writes
    over the upper one, which each later panel zeroes, so the factor ends exactly lower triangular."""
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    Np, bs = int(b.shape[0]), SOLVE_BLOCK
    nl = Np // int(mesh.size)
    nb, nbl = Np // bs, nl // bs
    slab = bs * max(q for q in (4, 2, 1) if nbl % q == 0)
    bounds = sorted({round(p * nb / FACTOR_PHASES) for p in range(FACTOR_PHASES + 1)})

    def local(b, R, n, lam):
        me = lax.axis_index("g")
        rows = lambda x: lax.dynamic_slice_in_dim(x, me * nl, nl)
        grow = me * nl + jnp.arange(nl)                          # global index of each local row
        A = coulomb_block(rows(b), rows(R), rows(n), b, R, n)   # (nl, Np): this device's rows of V
        A = A.at[jnp.arange(nl), grow].add(lam)

        def step(c0, k, A):
            mine, lk = k // nbl == me, (k % nbl) * bs
            D = lax.psum(jnp.where(mine, lax.dynamic_slice(A, (lk, k * bs), (bs, bs)), 0.0), "g")
            Lkk = jnp.tril(lax.linalg.cholesky(D, symmetrize_input=False))
            X = lax.linalg.triangular_solve(Lkk, lax.dynamic_slice_in_dim(A, k * bs, bs, axis=1),
                                            left_side=False, lower=True, transpose_a=True)
            X = jnp.where(grow[:, None] >= (k + 1) * bs, X, 0.0)
            X = jnp.where(mine, lax.dynamic_update_slice(X, Lkk, (lk, 0)), X)
            A = lax.dynamic_update_slice_in_dim(A, X, k * bs, axis=1)
            if c0 >= Np:                                         # the last panel: nothing past it
                return A
            Pn = lax.all_gather(X, "g", tiled=True)[c0:]
            Pn = jnp.where((c0 + jnp.arange(Np - c0))[:, None] >= (k + 1) * bs, Pn, 0.0)

            def update(s, A):
                blk = lax.dynamic_slice(A, (s * slab, c0), (slab, Np - c0))
                upd = blk - lax.dynamic_slice_in_dim(X, s * slab, slab) @ Pn.T
                return lax.dynamic_update_slice(A, upd, (s * slab, c0))
            return lax.fori_loop(0, nl // slab, update, A)

        for k0, k1 in zip(bounds[:-1], bounds[1:]):
            A = lax.fori_loop(k0, k1, functools.partial(step, (k0 + 1) * bs), A)
        return A

    return shard_map(local, mesh=mesh, in_specs=(P(), P(), P(), P()), out_specs=P("g"),
                     check_vma=False)(b, R, n, lam)


def _row_solves(Lrows, gamma, axis, b):
    """``y = L⁻¹γ`` and ``x = L⁻ᵀy`` with the rows of the lower-triangular ``L`` dealt over ``axis``
    in contiguous blocks (``Lrows``: this device's rows, all columns). Blocked substitution, ``b``
    rows a block: the owner of block i solves its diagonal block and broadcasts the piece, and every
    device updates its own rows (forward) or adds its rows' share of the next piece (backward).

    Never transposes ``L``: a whole-matrix transposed solve made XLA copy the factor, a second
    N_aux² on every device."""
    nlb, Np = Lrows.shape
    nl, nb = nlb // b, Np // b
    me = lax.axis_index(axis)
    g = jnp.zeros(Np, gamma.dtype).at[:gamma.shape[0]].set(gamma)
    eye = jnp.eye(b, dtype=Lrows.dtype)

    def diag(i):
        """Whether this device owns block i, its local row offset, and its diagonal block (the
        identity elsewhere, so the other devices' solves stay finite; their result is dropped)."""
        own = i // nl == me
        l0 = (i % nl) * b
        return own, l0, jnp.where(own, lax.dynamic_slice(Lrows, (l0, i * b), (b, b)), eye)

    def trsv(blk, v, transpose):
        return lax.linalg.triangular_solve(blk, v[:, None], left_side=True, lower=True,
                                           transpose_a=transpose)[:, 0]

    def forward(i, c):
        r, y = c
        own, l0, blk = diag(i)
        yi = lax.psum(jnp.where(own, trsv(blk, lax.dynamic_slice_in_dim(r, l0, b), False), 0.0),
                      axis)
        r = r - lax.dynamic_slice_in_dim(Lrows, i * b, b, axis=1) @ yi
        return r, lax.dynamic_update_slice_in_dim(y, yi, i * b, 0)

    _, y = lax.fori_loop(0, nb, forward, (lax.dynamic_slice_in_dim(g, me * nlb, nlb),
                                          jnp.zeros(Np, g.dtype)))

    def backward(k, xl):
        i = nb - 1 - k
        own, l0, blk = diag(i)
        s = lax.psum(lax.dynamic_slice_in_dim(Lrows, i * b, b, axis=1).T @ xl, axis)
        xi = trsv(blk, lax.dynamic_slice_in_dim(y, i * b, b) - s, True)
        return jnp.where(own, lax.dynamic_update_slice_in_dim(xl, xi, l0, 0), xl)

    xl = lax.fori_loop(0, nb, backward, jnp.zeros(nlb, g.dtype))
    return y, lax.all_gather(xl, axis, tiled=True), xl


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def df_coulomb_solve_rows(gamma, Lrows, axis):
    """The metric tail of RI-J on a mesh: E_J^DF = ½ γᵀ (L Lᵀ)⁻¹ γ, called inside ``shard_map`` with
    the replicated γ and this device's rows of the padded factor (:func:`metric_factor`), so each
    device holds N_aux²/G of it. The same value as :func:`df_coulomb_solve`; its gradient in γ is
    d = (L Lᵀ)⁻¹ γ, made in the forward."""
    return _solve_rows_fwd(gamma, Lrows, axis)[0]


def _solve_rows_fwd(gamma, Lrows, axis):
    y, x, xl = _row_solves(Lrows, gamma, axis, SOLVE_BLOCK)
    d = x[:gamma.shape[0]]
    return 0.5 * jnp.dot(gamma, d), (d, xl, y)


def _solve_rows_bwd(axis, res, gbar):
    """γ̄ = ḡ d; and L̄_ij = −ḡ x_i y_j on the lower triangle, this device's rows (needed only when
    the factor itself is differentiated, as on-graph forces do; the trainer's step drops it).

    The value is the same on every device, and ``shard_map`` hands each its share of the cotangent:
    γ is replicated, so its cotangent is summed over the devices on the way back, while each device
    alone owns its rows of L, which therefore take the summed ḡ here."""
    d, xl, y = res
    rows = lax.axis_index(axis) * xl.shape[0] + jnp.arange(xl.shape[0])
    lower = jnp.arange(y.shape[0])[None, :] <= rows[:, None]
    return gbar * d, jnp.where(lower, -lax.psum(gbar, axis) * xl[:, None] * y[None, :], 0.0)


df_coulomb_solve_rows.defvjp(_solve_rows_fwd, _solve_rows_bwd)


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
