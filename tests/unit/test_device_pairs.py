"""``neighbor_pairs``, the on-device significant-pair build, against a dense reference.

Checks which pairs are kept, their order (which decides the device split under ``shard_map``), and
that padding stays at the tail when the capacity overflows.
"""
import re

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft import init_model, chem
from gs_dft.coulomb.stream import full_quantities
from gs_dft.screening.pairs import (neighbor_pairs, _full_overlap_pairs,
                                             _scan_flat_indices)
from dftax.system import Molecule


_HLO_ARRAY = re.compile(r"=\s*[sufcbp]\w*\[([\d,]+)\]")


def _largest_array_elems(text):
    """Elements in the largest array the compiled program names — the residency the docstring claims."""
    best = 0
    for m in _HLO_ARRAY.finditer(text):
        n = 1
        for d in m.group(1).split(","):
            n *= int(d)
        best = max(best, n)
    return best


def _mol():
    return Molecule(symbols=["O", "H", "H"],
                    coords_bohr=np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11],
                                          [0.0, -1.43, 1.11]]),
                    basis="cc-pvdz")


def _dense_reference(basis, M):
    """|Ŝ_ij| for every ordered pair, straight from the per-pair overlap kernel."""
    q = full_quantities(basis)
    pi = jnp.repeat(jnp.arange(M, dtype=jnp.int32), M)
    pj = jnp.tile(jnp.arange(M, dtype=jnp.int32), M)
    return np.abs(np.asarray(_full_overlap_pairs(q, pi, pj))).reshape(M, M)


@pytest.mark.parametrize("M", [24, 60])
def test_returns_exactly_the_significant_pairs_lexsorted(M):
    model = init_model(_mol(), M, jr.PRNGKey(0), init=chem())
    eps = 1e-7
    S = _dense_reference(model.basis, M)
    want = sorted((int(i), int(j)) for i, j in zip(*np.nonzero(S > eps)))

    pi, pj, valid = neighbor_pairs(model.basis, eps=eps)
    got = list(zip(np.asarray(pi).tolist(), np.asarray(pj).tolist()))
    assert bool(np.all(np.asarray(valid))), "an unpadded build must be all-valid"
    assert got == want, "wrong pair set, or not lexsorted"


def test_padding_goes_to_the_tail_and_is_masked():
    """`shard_map` splits the pair axis contiguously, so padding must stay at the END — otherwise a
    device's slice would be interleaved with dummies."""
    model = init_model(_mol(), 24, jr.PRNGKey(0), init=chem())
    eps = 1e-7
    n = int(neighbor_pairs(model.basis, eps=eps)[0].shape[0])
    pad_to = n + 37
    pi, pj, valid = neighbor_pairs(model.basis, eps=eps, pad_to=pad_to)
    v = np.asarray(valid)
    assert pi.shape[0] == pad_to and v.sum() == n
    assert v[:n].all() and not v[n:].any(), "valid entries must be contiguous at the front"
    assert (np.asarray(pi)[n:] == 0).all() and (np.asarray(pj)[n:] == 0).all()


def test_overflow_keeps_the_largest_overlaps():
    """With pad_to below the significant count the build must keep the strongest pairs — dropping
    the marginal ~eps tail — rather than raising, so the trainer's shapes stay constant."""
    model = init_model(_mol(), 48, jr.PRNGKey(1), init=chem())
    eps, M = 1e-7, 48
    S = _dense_reference(model.basis, M)
    n = int((S > eps).sum())
    pad_to = n // 2
    pi, pj, valid = neighbor_pairs(model.basis, eps=eps, pad_to=pad_to)
    assert pi.shape[0] == pad_to and bool(np.all(np.asarray(valid)))
    kept = S[np.asarray(pi), np.asarray(pj)]
    dropped_max = np.sort(S[S > eps])[::-1][pad_to:].max()
    assert kept.min() >= dropped_max, "a dropped pair outranked a kept one"


def test_pad_larger_than_the_pair_count_is_padded_not_truncated():
    """pad_to can exceed M² for a small system with a generous pad (M=60 has 3600 ordered pairs but
    1.25×n_sig = 4307). The shape must still be exactly pad_to."""
    model = init_model(_mol(), 60, jr.PRNGKey(0), init=chem())
    pad_to = 60 * 60 + 500
    pi, pj, valid = neighbor_pairs(model.basis, eps=1e-7, pad_to=pad_to)
    assert pi.shape[0] == pad_to == pj.shape[0] == valid.shape[0]


def test_the_row_strips_are_never_glued_into_one_M2_array():
    """The build must never hold the full M² overlap table.

    Every correctness test above passes whether or not it does — the pairs are identical either way
    — so only a residency assertion can catch a regression here.

    Asserts the SCALING, not an absolute size. A strip is (row_chunk·M, 3, 3) — the per-pair
    geometry carries a 3×3 — so the largest array is 9·row_chunk·M, and at unit-test sizes that
    constant alone can exceed M²; an absolute bound cannot tell the two regimes apart. What it can
    tell is the exponent: strips that are filtered as produced scale with `row_chunk`, while a table
    glued back together is pinned at M² and does not move when `row_chunk` halves.

    (The older form spied on `jnp.concatenate`, which the compiled scan builder never calls — it
    compacts with `dynamic_update_slice` — so the spy recorded nothing and the assertion silently
    inverted into a failure.)

    ε is calibrated off the dense reference so the screen actually sparsifies at unit-test size; at
    the production ε a small molecule's pair list is ~95% dense and this would be vacuous.
    """
    M = 200
    model = init_model(_mol(), M, jr.PRNGKey(0), init=chem())
    S = _dense_reference(model.basis, M)
    eps = float(np.quantile(S, 0.95))               # keep the strongest ~5% of pairs
    n_sig = int((S > eps).sum())
    assert 0 < n_sig < M * M // 8, "eps failed to sparsify — the assertion below would be vacuous"

    q = full_quantities(model.basis)

    def biggest(row_chunk):
        compiled = _scan_flat_indices.lower(q, M, eps, 2 * n_sig, row_chunk, False).compile()
        return _largest_array_elems(compiled.as_text())

    small, large = biggest(8), biggest(16)
    assert small > 0, "no arrays parsed out of the compiled HLO — the check did not take"
    assert 1.5 <= large / small <= 2.5, (
        f"halving row_chunk moved the largest array {small} -> {large} "
        f"({large / small:.2f}x, expected ~2x): it is not strip-sized, so the rows are being "
        f"assembled into one table (M²={M * M})")
    assert small < M * M, (
        f"one strip at row_chunk=8 is already {small} elements against a {M * M}-element table")

    pi, _, valid = neighbor_pairs(model.basis, eps=eps, row_chunk=8)
    assert int(np.asarray(valid).sum()) == n_sig, "the sparsified build lost pairs"


@pytest.mark.parametrize("pad_to", [None, 24 * 24, 24 * 24 + 313])
def test_negative_eps_returns_the_whole_ordered_list(pad_to):
    """``ks.train.evaluate`` asks for the unscreened list with ``eps=-1.0`` — no |Ŝ| can fail that,
    so the answer is every ordered pair, lexsorted. Taken directly rather than by thresholding M²
    computed overlaps, and the two must agree exactly."""
    M = 24
    model = init_model(_mol(), M, jr.PRNGKey(0), init=chem())
    pi, pj, valid = neighbor_pairs(model.basis, eps=-1.0, pad_to=pad_to)
    want_i = np.repeat(np.arange(M), M)
    want_j = np.tile(np.arange(M), M)
    n = M * M
    assert pi.shape[0] == (n if pad_to is None else pad_to)
    assert np.array_equal(np.asarray(pi)[:n], want_i)
    assert np.array_equal(np.asarray(pj)[:n], want_j)
    assert np.asarray(valid)[:n].all() and not np.asarray(valid)[n:].any()
