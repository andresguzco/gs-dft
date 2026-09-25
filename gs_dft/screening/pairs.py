"""Pair screening for the splat one-electron operators.

The splat overlap decays as exp(−p_ij|R_i−R_j|²), so the number of *significant* neighbours per
splat is O(1) in M and the M² pair operations screen to O(M·degree) = O(M).

Pairs are held as a flat list ``(pi, pj)`` — the nonzeros of S above a normalized cutoff ε — driven
by ``segment_sum`` (CSR-style). The list is rebuilt only when the splats move, so its O(M²)
construction is amortized against the variational step.

Kernels take the PRECOMPUTED per-splat quantities from ``coulomb.stream.full_quantities`` so the
A-chart is evaluated once per energy call, outside the remat'd streams.
"""

import functools
import os

import numpy as np
import jax.numpy as jnp
import jax
from jax import lax

from gs_dft.basis.spectral import Splat
from gs_dft.integrals.linalg3 import det3, adj3
from gs_dft.coulomb.stream import full_quantities, _full_pair_quantities
from gs_dft.screening.grid import _host_rcut
# Laplace half-line quadrature: python floats so the unrolled node loops stay trace-time constants.
from gs_dft.integrals.dense import _QUAD_T_LIST, _QUAD_W_LIST

_I3 = np.eye(3)          # numpy: importing the package must not initialise the JAX backend


def _chunk_pairs(pi, pj, valid, chunk):
    """Pad the pair list to a multiple of ``chunk`` and reshape to (n_chunk, chunk) for lax.scan.
    Padding pairs are (0,0) with valid=False so they contribute nothing."""
    Npair = pi.shape[0]
    pad = (-Npair) % chunk
    base_valid = jnp.ones(Npair, bool) if valid is None else valid
    pic = jnp.concatenate([pi, jnp.zeros(pad, pi.dtype)]).reshape(-1, chunk)
    pjc = jnp.concatenate([pj, jnp.zeros(pad, pj.dtype)]).reshape(-1, chunk)
    vc = jnp.concatenate([base_valid, jnp.zeros(pad, bool)]).reshape(-1, chunk)
    return pic, pjc, vc

__all__ = ["neighbor_pairs", "screened_overlap_pairs", "screened_overlap_matvec",
           "screened_nuclear_pairs", "density_pairs", "screened_external_energy", "pair_weight"]


_SENTINEL = jnp.int64(-1)


@functools.partial(jax.jit, static_argnums=(1, 3, 4, 5))
def _scan_flat_indices(q, M, eps, pad_to, row_chunk, unique):
    """Flat indices ``i*M + j`` of the significant pairs, as ONE compiled scan over row strips.

    Two facts make the scan possible. `jnp.nonzero(sig, size=cap)` is jittable with a static `cap`,
    and the output length is already fixed because the caller pads the pair list to `pad_to`. So the
    strips compact into a preallocated buffer at a traced offset via `dynamic_update_slice`.

    `cap` is derived from the pad rather than typed: the pad is ~2x the live count and the strips
    partition the rows, so `pad_to / n_strips` is the mean occupancy and 4x that is the headroom.
    A strip that exceeds it would silently drop pairs, so the true count is accumulated and
    `n_over` reports it; the caller falls back to the exact host build. Overflow is the rare path
    (measured pair fill 0.62), and the fallback is the correctness reference.

    Returns ``(keep, n_sig, n_over)``: `keep` is `pad_to` flat indices ASCENDING (strips run in row
    order and `nonzero` returns ascending, so the concatenation is already lexsorted — the old
    `jnp.sort` over the whole set is gone), `-1` past `n_sig`."""
    n_strips = -(-M // row_chunk)
    cap = max(1024, 4 * -(-pad_to // n_strips))
    starts = jnp.arange(n_strips, dtype=jnp.int64) * row_chunk

    def body(carry, a):
        out, off, tot, over = carry
        s = _strip_abs_overlap(q, a, row_chunk, M)          # (row_chunk*M,), zero-padded rows past M
        rows = jnp.arange(s.shape[0]) // M + a
        sig = (s > eps) & (rows < M)                        # the tail strip is short: mask it
        if unique:
            sig = sig & (jnp.arange(s.shape[0]) % M >= rows)
        cnt = jnp.sum(sig)
        loc = jnp.nonzero(sig, size=cap, fill_value=0)[0].astype(jnp.int64)
        flat = jnp.where(jnp.arange(cap) < cnt, loc + a * M, _SENTINEL)
        out = lax.dynamic_update_slice(out, flat, (off,))
        return (out, off + jnp.minimum(cnt, cap), tot + cnt, over + (cnt > cap)), None

    buf = jnp.full(pad_to + cap, _SENTINEL, jnp.int64)       # +cap: the last write cannot run off
    (out, _off, tot, over), _ = lax.scan(body, (buf, jnp.int64(0), jnp.int64(0), jnp.int64(0)),
                                         starts)
    return out[:pad_to], tot, over


def _decode_flat(keep, M, pad_to, n_sig, unique):
    """``(pi, pj, valid)`` from ascending flat indices, padded to ``pad_to`` with dummy (0,0)."""
    k = min(int(n_sig), int(pad_to))
    live = keep[:k]
    pi, pj = (live // M).astype(jnp.int32), (live % M).astype(jnp.int32)
    valid = _pair_weights(pi, pj, unique)
    if pad_to > k:
        z = jnp.zeros(pad_to - k, jnp.int32)
        pi, pj = jnp.concatenate([pi, z]), jnp.concatenate([pj, z])
        valid = jnp.concatenate([valid, jnp.zeros(pad_to - k, valid.dtype)])
    return pi, pj, valid


@functools.partial(jax.jit, static_argnums=(1, 3, 4))
def _count_significant(q, M, eps, row_chunk, unique):
    """How many pairs are significant — one compiled scan, ONE host sync."""
    n_strips = -(-M // row_chunk)
    starts = jnp.arange(n_strips, dtype=jnp.int64) * row_chunk

    def body(acc, a):
        s = _strip_abs_overlap(q, a, row_chunk, M)
        rows = jnp.arange(s.shape[0]) // M + a
        sig = (s > eps) & (rows < M)                    # the tail strip is short: mask it
        if unique:
            sig = sig & (jnp.arange(s.shape[0]) % M >= rows)
        return acc + jnp.sum(sig), None

    tot, _ = lax.scan(body, jnp.int64(0), starts)
    return tot


@functools.partial(jax.jit, static_argnums=(1, 3, 4, 5))
def _scan_flat_and_vals(q, M, eps, cap, row_chunk, unique):
    """Flat indices AND their |S| values, into a buffer of STATIC size ``cap``.

    The overflow path's input. Collecting values is what makes "keep the largest" a real sort rather
    than an approximation, and pinning the buffer to `cap` (a multiple of the pad, itself frozen for
    the run) is what keeps the shape static — a buffer sized to the live count would vary between
    refreshes, and a mid-run recompile tears down the multi-GPU graphs and deadlocks the collective
    teardown. Entries past the live count are (-1, -inf), so they sort last.
    """
    n_strips = -(-M // row_chunk)
    starts = jnp.arange(n_strips, dtype=jnp.int64) * row_chunk
    ck = row_chunk * M

    def body(carry, a):
        out, val, off = carry
        sv = _strip_abs_overlap(q, a, row_chunk, M)
        rows = jnp.arange(ck) // M + a
        sig = (sv > eps) & (rows < M)
        if unique:
            sig = sig & (jnp.arange(ck) % M >= rows)
        cnt = jnp.sum(sig)
        loc = jnp.nonzero(sig, size=ck, fill_value=0)[0].astype(jnp.int64)
        live = jnp.arange(ck) < cnt
        out = lax.dynamic_update_slice(out, jnp.where(live, loc + a * M, jnp.int64(-1)), (off,))
        val = lax.dynamic_update_slice(val, jnp.where(live, jnp.take(sv, loc), -jnp.inf), (off,))
        return (out, val, off + cnt), None

    buf = jnp.full(cap + ck, jnp.int64(-1), jnp.int64)
    vbuf = jnp.full(cap + ck, -jnp.inf, jnp.float64)
    (out, val, _o), _ = lax.scan(body, (buf, vbuf, jnp.int64(0)), starts)
    return out[:cap], val[:cap]


def _keep_largest(q, M, eps, n_sig, pad_to, row_chunk, unique, headroom=2):
    """The ``pad_to`` largest-overlap pairs when the significant count overruns the pad."""
    cap = int(headroom) * int(pad_to)
    if n_sig > cap:
        raise ValueError(
            f"neighbor_pairs: {n_sig} significant pairs exceed {headroom}x the pad ({cap}). "
            f"Raise screen.pad — the run's pair list has outgrown the shape frozen at step 0.")
    flat, vals = _scan_flat_and_vals(q, M, float(eps), cap, int(row_chunk), bool(unique))
    order = jnp.argsort(-vals)[:pad_to]          # dead entries carry -inf and sort last
    return jnp.sort(flat[order])


def _spatial_candidates(splats, eps, unique, safety=1.25):
    """Flat indices ``i*M + j`` of every pair that COULD be significant, from a cKDTree ball query
    on the splat centres. A superset of ``{|S_ij| > eps}``, returned ascending.

    Cost is O(M x k) with k the neighbours inside the cutoff, against O(M^2) for the dense strips:
    ~3.4M unique pairs at M=21,026 means a mean degree near 160, so this is one to two orders of
    magnitude less overlap arithmetic."""
    from scipy.spatial import cKDTree
    mu = np.asarray(splats.centers, dtype=float)
    M = mu.shape[0]
    rc = np.asarray(_host_rcut(splats, eps), dtype=float) * float(safety)
    tree = cKDTree(mu)
    res = tree.query_ball_point(mu, rc + float(rc.max()))      # per-splat superset by one radius
    out = []
    for i, js in enumerate(res):
        if not js:
            continue
        j = np.asarray(js, dtype=np.int64)
        j = j[j >= i] if unique else j
        if j.size == 0:
            continue
        d = np.linalg.norm(mu[j] - mu[i], axis=1)              # exact per-PAIR radius test
        j = j[d < rc[i] + rc[j]]
        if j.size:
            out.append(i * np.int64(M) + j)
    if not out:
        return np.zeros(0, np.int64)
    return np.sort(np.concatenate(out))


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def _scan_candidate_indices(q, cand, eps, M, pad_to, chunk):
    """The significant subset of ``cand`` (flat indices), compacted in one scan — no host syncs.

    Same contract as `_scan_flat_indices`, but walking a CANDIDATE list instead of full M-wide row
    strips, so the overlap kernel runs O(n_cand) times rather than O(M^2). Returns
    ``(keep, n_sig, n_over)``; `keep` is ascending because `cand` is.
    """
    n = cand.shape[0]
    pad = (-n) % chunk
    cc = jnp.concatenate([cand, jnp.full(pad, -1, cand.dtype)]).reshape(-1, chunk)

    def body(carry, c):
        out, off, tot, over = carry
        live = c >= 0
        ci, cj = (c // M).astype(jnp.int32), (c % M).astype(jnp.int32)
        sig = live & (jnp.abs(_full_overlap_pairs(q, ci, cj)) > eps)
        cnt = jnp.sum(sig)
        loc = jnp.nonzero(sig, size=chunk, fill_value=0)[0]
        flat = jnp.where(jnp.arange(chunk) < cnt, jnp.take(c, loc), jnp.int64(-1))
        out = lax.dynamic_update_slice(out, flat, (off,))
        return (out, off + jnp.minimum(cnt, chunk), tot + cnt, over + (cnt > chunk)), None

    buf = jnp.full(pad_to + chunk, jnp.int64(-1), jnp.int64)
    (out, _o, tot, over), _ = lax.scan(body, (buf, jnp.int64(0), jnp.int64(0), jnp.int64(0)), cc)
    return out[:pad_to], tot, over


def neighbor_pairs(splats, eps: float = 1e-5, pad_to: int | None = None, row_chunk: int = 256,
                   unique: bool = False):
    """Significant ordered pairs (i,j) with normalized overlap |Ŝ_ij| > ``eps``, built on device.

    ``unique=True`` keeps only j >= i and returns float WEIGHTS (2 for i<j, 1 on the diagonal, 0 on
    the padding) in place of the boolean mask: every consumer here contracts a symmetric integrand,
    so Σ_all = Σ_{i<=j} w·(·) and the pair work halves. Consumers multiply by the third output
    (bool → 0/1, weights → 0/1/2) rather than masking with it.

    Splats are unit-normalized so Ŝ = |S|. Returns ``(pi, pj, valid)`` — all ordered pairs incl. the
    diagonal, lexsorted, so it directly drives the symmetric matvec S@C. With ``pad_to`` the arrays
    are padded to that fixed length (dummy (0,0) pairs, ``valid=False``) so the consuming matvec
    keeps a static shape under ``jit``; without it, ``pad_to`` is taken from the exact significant
    count.

    Overflow is not an error: if the significant count exceeds ``pad_to``, the ``pad_to``
    largest-|overlap| pairs are kept. The trainer needs constant shapes for a whole run — a mid-run
    recompile tears down the multi-GPU CUDA graphs and deadlocks the collective teardown.

    O(M²) work, but O(row_chunk·M + n_sig) resident: row strips are filtered as they are produced
    and never assembled into the full table.

    ``eps < 0`` means every ordered pair is significant, which is how ``ks.train.evaluate`` asks for
    the unscreened list.
    """
    # Duck-typed on the four properties `integrals/dense.py` documents as the splat interface, so
    # any chart can reach the screened path.
    if not all(hasattr(splats, a) for a in ("A", "centers", "norm", "n_basis")):
        raise NotImplementedError(
            f"neighbor_pairs: {type(splats).__name__} does not expose the splat interface "
            f"(.A / .centers / .norm / .n_basis)")
    M = int(splats.centers.shape[0])

    # Every pair is significant: repeat/tile is already lexsorted, so skip the M² overlaps entirely.
    # A pad below M² still needs the ranking, so it falls through.
    n_all = M * (M + 1) // 2 if unique else M * M
    if eps < 0.0 and (pad_to is None or int(pad_to) >= n_all):
        if unique:
            pi, pj = (a.astype(jnp.int32) for a in jnp.triu_indices(M))
        else:
            pi = jnp.repeat(jnp.arange(M, dtype=jnp.int32), M)
            pj = jnp.tile(jnp.arange(M, dtype=jnp.int32), M)
        valid = _pair_weights(pi, pj, unique)
        if pad_to is not None and int(pad_to) > n_all:
            z = jnp.zeros(int(pad_to) - n_all, jnp.int32)
            pi, pj = jnp.concatenate([pi, z]), jnp.concatenate([pj, z])
            valid = jnp.concatenate([valid, jnp.zeros(int(pad_to) - n_all, valid.dtype)])
        return pi, pj, valid

    q = full_quantities(splats)

    # No pad given: COUNT first (compiled, one sync), then build at exactly that size. The caller
    # wanting an exactly-sized list never has to touch the eager loop below.
    if pad_to is None and eps >= 0.0:
        pad_to = int(_count_significant(q, M, float(eps), int(row_chunk), bool(unique)))
        if pad_to == 0:
            z = jnp.zeros(0, jnp.int32)
            return z, z, _pair_weights(z, z, unique)

    if pad_to is not None and eps >= 0.0 and int(os.environ.get("DFTAX_PAIR_SPATIAL", "0")) == 1:
        try:
            cand = _spatial_candidates(splats, eps, unique)
        except Exception as exc:                       # scipy missing, or a degenerate chart
            print(f"  [screening] spatial candidates unavailable ({type(exc).__name__}: {exc}); "
                  f"using the dense strips", flush=True)
            cand = None
        if cand is not None and cand.size:
            ck = min(8192, int(cand.size))
            keep, n_sig, n_over = _scan_candidate_indices(
                full_quantities(splats), jnp.asarray(cand), float(eps), M, int(pad_to), ck)
            if int(n_over) == 0 and int(n_sig) <= int(pad_to):
                return _decode_flat(keep, M, int(pad_to), int(n_sig), unique)
            print(f"  [screening] spatial path overflowed (sig={int(n_sig)}, pad={int(pad_to)}, "
                  f"over={int(n_over)}); using the dense strips", flush=True)

    # The ONLY builder. `pad_to` is known by now (given, or counted just above), so the output shape
    # is static and the whole thing compiles: strips scanned once, significant entries compacted at a
    # traced offset, nothing crossing to the host but two integers.
    keep, n_sig, n_over = _scan_flat_indices(q, M, float(eps), int(pad_to), int(row_chunk),
                                             bool(unique))
    n_sig, n_over = int(n_sig), int(n_over)
    if n_over:
        raise ValueError(
            f"neighbor_pairs: {n_over} strip(s) overran the per-strip capacity. This is a bug in the "
            f"capacity heuristic, not a user error — report the system and pad ({pad_to}).")
    if n_sig > pad_to:
        print(f"  [screening] pair count {n_sig} > pad {pad_to}: keeping the {pad_to} "
              f"largest-overlap pairs (dropped {n_sig - pad_to})", flush=True)
        keep = _keep_largest(q, M, float(eps), n_sig, int(pad_to), int(row_chunk), bool(unique))
        return _decode_flat(keep, M, int(pad_to), int(pad_to), unique)
    return _decode_flat(keep, M, int(pad_to), n_sig, unique)


def _pair_weights(pi, pj, unique):
    """Third output of :func:`neighbor_pairs`: a bool mask, or the (2 − δ_ij) weights when unique."""
    if not unique:
        return jnp.ones(pi.shape[0], bool)
    return jnp.where(pi == pj, 1.0, 2.0)


def pair_weight(valid, dtype=jnp.float64):
    """``valid`` as a multiplicative factor: 0/1 from a bool mask, 0/1/2 from unique-pair weights."""
    return valid.astype(dtype)


# ---------------------------------------------------------------------------
#  Per-pair kernels (operate on the PRECOMPUTED per-splat quantities ``q``)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(2, 3))
def _strip_abs_overlap(q, a, n_row, M):
    """|Ŝ| for rows ``[a, a+n_row)`` against all M columns, as ONE compiled kernel."""
    pi = jnp.repeat(jnp.arange(n_row, dtype=jnp.int32) + a, M)
    pj = jnp.tile(jnp.arange(M, dtype=jnp.int32), n_row)
    return jnp.abs(_full_overlap_pairs(q, pi, pj))


def _full_overlap_pairs(q, pi, pj):
    """S_ij = N_iN_j K_ij π^{3/2}/√det(A_ij) — per-pair version of ``dense.overlap_matrix``."""
    Aij, _, NK = _full_pair_quantities(q, pi, pj)
    return NK * jnp.pi ** 1.5 / jnp.sqrt(det3(Aij))


def _full_nuclear_pairs(q, pi, pj, atom_coords, atom_charges):
    """Per-pair version of ``dense.nuclear_attraction_matrix`` — Laplace quadrature with
    B = A_ij + t²I. Degeneracy-safe (no eigh). The bra-pair g_i g_j is screened by the SAME overlap
    list, but each surviving pair still sums over ALL nuclei — V_ne's long-range 1/r tail does not
    vanish with nuclear distance, so dropping far nuclei would need multipoles (a later refinement).

    Node loop = static python unroll in a division-lean form (as in
    ``coulomb.ri.full_isobra_block``): the exponent uses ``A_ij B⁻¹ = I − t²B⁻¹`` →
    expo = t²|d|² − t⁴ (d·adj(B)d)/det(B), so each node costs one reciprocal per pair
    (adjugate = mults) and the whole loop fuses into one kernel."""
    Aij, mup, NK = _full_pair_quantities(q, pi, pj)
    d = mup[:, None, :] - atom_coords[None, :, :]             # (P, n_atoms, 3)
    d2 = jnp.sum(d * d, axis=-1)                              # (P, n_atoms)
    vA = jnp.zeros(d2.shape)
    for t, w in zip(_QUAD_T_LIST, _QUAD_W_LIST):              # static unroll → one fused kernel
        t2 = t * t
        B = Aij + t2 * _I3                                    # (P,3,3)
        invdet = 1.0 / det3(B)                                # ONE reciprocal per pair
        qf = jnp.einsum("pak,pkl,pal->pa", d, adj3(B), d)     # d·adj(B)·d  (P, n_atoms)
        expo = t2 * d2 - (t2 * t2) * qf * invdet[:, None]
        vA = vA + (w * jnp.pi ** 1.5) * jnp.sqrt(invdet)[:, None] * jnp.exp(-expo)
    vA = (2.0 / jnp.sqrt(jnp.pi)) * vA                        # (P, n_atoms)
    return -jnp.sum(atom_charges[None, :] * vA, axis=1) * NK


def screened_overlap_pairs(splats, pi, pj):
    """S_ij for the listed pairs — the overlap kernel on a pair-index list (not M×M)."""
    return _full_overlap_pairs(full_quantities(splats), pi, pj)


def screened_nuclear_pairs(splats, pi, pj, atom_coords, atom_charges):
    """V_ij for the listed pairs — the nuclear-attraction pair kernel."""
    return _full_nuclear_pairs(full_quantities(splats), pi, pj, atom_coords, atom_charges)


def screened_overlap_matvec(splats, pi, pj, C, valid=None, chunk=8192):
    """(S @ C)[i] = Σ_{significant (i,j)} S_ij C[j], **streamed** over the pair list in chunks +
    ``segment_sum``. O(n_pair·n_cols) compute, O(chunk·n_cols + M·n_cols) memory (remat keeps the
    backward O(chunk·n_cols) — without it the (n_pair, n_cols) contrib tape can exceed the dense
    M×M it replaces, since n_pair·n_cols = M·degree·n_cols ≫ M² once degree·n_cols > M). ``valid``
    zeroes dummy pairs. Differentiable in the splat params; drops into the Löwdin ``S@C``."""
    q = full_quantities(splats)                               # once, outside the remat'd scan
    M = int(splats.n_basis)
    ncol = C.shape[1]
    pic, pjc, vc = _chunk_pairs(pi, pj, valid, chunk)

    def body(acc, carry):
        ci, cj, cv = carry
        s = jnp.where(cv, _full_overlap_pairs(q, ci, cj), 0.0)   # (chunk,)
        return acc + jax.ops.segment_sum(s[:, None] * C[cj], ci, num_segments=M), None

    SC, _ = lax.scan(jax.checkpoint(body), jnp.zeros((M, ncol)), (pic, pjc, vc))
    return SC


def density_pairs(C_orth, occ, pi, pj):
    """P_ij = Σ_o occ_o C_io C_jo on the pair list — the density matrix entries needed for the
    energies, WITHOUT forming the dense M×M P (which is itself an O(M²) build)."""
    return jnp.sum(occ[None, :] * C_orth[pi] * C_orth[pj], axis=1)


def screened_external_energy(splats, C_orth, occ, pi, pj, atom_coords, atom_charges,
                             valid=None, chunk=8192, skip_pad=False):
    """E_external = Σ P_ij V_ij over the significant pairs, **streamed** over the pair list in chunks
    + remat. O(n_pair) compute, O(chunk·n_atoms) memory — the (n_pair, n_atoms) nuclear intermediate
    never fully materializes (dense V_ne's (Q,M,M,3,3) OOMs at scale). Equals Tr(P·V_ne) when no pairs
    are screened; the dense M×M V never forms.

    Only V_ne (the external energy — a trace, no positivity requirement) is screened. The overlap S
    and kinetic T are ALWAYS built dense: pairwise-screening them tips the occupied Gram G=CᵀSC and the
    kinetic Gram Θ=CᵀTC indefinite ⇒ E_kin<0. So there is no screened one-electron KINETIC — the dense
    one-electron path is the only path.
    """
    q = full_quantities(splats)                               # once, outside the remat'd scan
    pic, pjc, vc = _chunk_pairs(pi, pj, valid, chunk)

    def chunk_energy(ci, cj, cv):
        Pp = pair_weight(cv, C_orth.dtype) * density_pairs(C_orth, occ, ci, cj)   # (chunk,)
        V = _full_nuclear_pairs(q, ci, cj, atom_coords, atom_charges)
        return jnp.sum(Pp * V)

    def body(acc, carry):
        ci, cj, cv = carry
        if skip_pad:
            # The pad is appended, so trailing chunks are all-dummy: skip the kernel (static shape
            # kept, differentiable; a `where` would evaluate both branches).
            e = lax.cond(jnp.any(cv != 0), chunk_energy, lambda *_: jnp.array(0.0), ci, cj, cv)
        else:
            e = chunk_energy(ci, cj, cv)
        return acc + e, None

    Eext, _ = lax.scan(jax.checkpoint(body), jnp.array(0.0), (pic, pjc, vc))
    return Eext
