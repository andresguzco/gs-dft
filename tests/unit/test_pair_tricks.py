"""Parity for the pair-stream tricks: unique pairs with (2−δ_ij) weights and skipping all-padding
chunks. Each must reproduce the ordered / unskipped result to round-off, energies and gradients
alike."""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft.basis.isotropic import init_product_aux
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.coulomb import ri as cdf, stream as cs
from gs_dft.screening import pairs as scr


def _splats(M=48, seed=0):
    k1, k2 = jr.split(jr.PRNGKey(seed))
    centers = jr.uniform(k1, (6, 3), minval=-3.0, maxval=3.0)
    return init_spectral_splats(centers, M, key=k2, aniso_jitter=0.3)


def _coeffs(M, nocc=5, seed=2):
    return jr.normal(jr.PRNGKey(seed), (M, nocc)), jnp.ones(nocc)


def _atoms(seed=20):
    coords = jr.uniform(jr.PRNGKey(seed), (6, 3), minval=-3.0, maxval=3.0)
    return coords, jnp.array([6.0, 1.0, 8.0, 1.0, 7.0, 1.0])


def _grad(fn, *args):
    return jax.grad(fn, argnums=(0, 1))(*args)


def _assert_tree_close(a, b, rtol=1e-9, atol=1e-11):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------------------------
#  unique pairs
# --------------------------------------------------------------------------------------------

def test_unique_list_covers_the_ordered_list_with_weights():
    s = _splats()
    pi, pj, valid = scr.neighbor_pairs(s, eps=1e-6)
    ui, uj, w = scr.neighbor_pairs(s, eps=1e-6, unique=True)
    assert bool(jnp.all(uj >= ui))
    assert w.dtype == jnp.float64 and set(np.unique(np.asarray(w))) <= {1.0, 2.0}
    assert int(jnp.sum(w)) == int(pi.shape[0])              # Σ(2−δ) = ordered count
    ordered = set(zip(np.asarray(pi).tolist(), np.asarray(pj).tolist()))
    for i, j, wt in zip(np.asarray(ui), np.asarray(uj), np.asarray(w)):
        assert (int(i), int(j)) in ordered
        assert wt == (1.0 if i == j else 2.0)


def test_unique_full_list_is_the_upper_triangle():
    s = _splats(M=7)
    ui, uj, w = scr.neighbor_pairs(s, eps=-1.0, unique=True)
    assert ui.shape[0] == 7 * 8 // 2 and float(w.sum()) == 49.0


@pytest.mark.parametrize("pad", [None, 3000])
def test_unique_external_energy_and_grads_match_ordered(pad):
    s = _splats()
    C, occ = _coeffs(s.n_basis)
    coords, charges = _atoms()
    pi, pj, v = scr.neighbor_pairs(s, eps=1e-6, pad_to=pad)
    ui, uj, w = scr.neighbor_pairs(s, eps=1e-6, pad_to=pad, unique=True)

    def e_ord(sp, c):
        return scr.screened_external_energy(sp, c, occ, pi, pj, coords, charges, v, chunk=256)

    def e_uni(sp, c):
        return scr.screened_external_energy(sp, c, occ, ui, uj, coords, charges, w, chunk=256)

    np.testing.assert_allclose(float(e_uni(s, C)), float(e_ord(s, C)), rtol=1e-11)
    _assert_tree_close(_grad(e_uni, s, C), _grad(e_ord, s, C))


def test_unique_gamma_and_exact_coulomb_match_ordered():
    s = _splats()
    C, occ = _coeffs(s.n_basis)
    aux = init_product_aux(s, scales=(1.0,), diagonal_only=True)
    pi, pj, v = scr.neighbor_pairs(s, eps=1e-6, pad_to=2500)
    ui, uj, w = scr.neighbor_pairs(s, eps=1e-6, pad_to=2500, unique=True)

    def g_ord(sp, c):
        return cdf.df_coulomb_gamma_screened(sp, c, occ, aux, pi, pj, chunk=128, valid=v)

    def g_uni(sp, c):
        return cdf.df_coulomb_gamma_screened(sp, c, occ, aux, ui, uj, chunk=128, valid=w)

    np.testing.assert_allclose(np.asarray(g_uni(s, C)), np.asarray(g_ord(s, C)), rtol=1e-10, atol=1e-12)
    _assert_tree_close(_grad(lambda sp, c: jnp.sum(g_uni(sp, c) ** 2), s, C),
                       _grad(lambda sp, c: jnp.sum(g_ord(sp, c) ** 2), s, C))

    e_ord = cs.screened_coulomb_energy(s, C, occ, pi, pj, v, bra_chunk=64)
    e_uni = cs.screened_coulomb_energy(s, C, occ, ui, uj, w, bra_chunk=64)
    np.testing.assert_allclose(float(e_uni), float(e_ord), rtol=1e-10)


# --------------------------------------------------------------------------------------------
#  skip all-padding chunks
# --------------------------------------------------------------------------------------------

def test_skip_pad_is_exact_for_external_and_gamma():
    s = _splats()
    C, occ = _coeffs(s.n_basis)
    coords, charges = _atoms()
    aux = init_product_aux(s, scales=(1.0,), diagonal_only=True)
    pi, pj, v = scr.neighbor_pairs(s, eps=1e-6, pad_to=4096)     # ~2x pad ⇒ all-dummy tail chunks
    assert int(v.sum()) < 4096 - 2 * 256

    def e(sp, c, skip):
        return scr.screened_external_energy(sp, c, occ, pi, pj, coords, charges, v, chunk=256,
                                            skip_pad=skip)

    np.testing.assert_allclose(float(e(s, C, True)), float(e(s, C, False)), rtol=1e-12)
    _assert_tree_close(_grad(lambda sp, c: e(sp, c, True), s, C),
                       _grad(lambda sp, c: e(sp, c, False), s, C), rtol=1e-11)

    def g(sp, c, skip):
        return cdf.df_coulomb_gamma_screened(sp, c, occ, aux, pi, pj, chunk=256, valid=v, skip_pad=skip)

    np.testing.assert_allclose(np.asarray(g(s, C, True)), np.asarray(g(s, C, False)), rtol=1e-12, atol=1e-14)
    _assert_tree_close(_grad(lambda sp, c: jnp.sum(g(sp, c, True) ** 2), s, C),
                       _grad(lambda sp, c: jnp.sum(g(sp, c, False) ** 2), s, C), rtol=1e-11)


def test_skip_pad_does_not_inflate_the_compiled_memory_budget():
    """Skipping all-dummy chunks must not COST memory.

    `skip_pad` wraps each chunk in `lax.cond`, and a conditional inside the scan can defeat the
    `jax.checkpoint` on its body, which is what keeps the γ stream's working set to one chunk. The
    values agree either way, so only the compiled memory budget can see it.

    Read off XLA's own accounting for the gradient, which is where the residuals live.
    """
    s = _splats()
    C, occ = _coeffs(s.n_basis)
    aux = init_product_aux(s, scales=(1.0,), diagonal_only=True)
    pi, pj, v = scr.neighbor_pairs(s, eps=1e-6, pad_to=4096)     # ~2x pad ⇒ all-dummy tail chunks
    assert int(v.sum()) < 4096 - 2 * 256, "no all-dummy chunks — the skip would be a no-op"

    def budget(skip):
        def loss(sp, c):
            return jnp.sum(cdf.df_coulomb_gamma_screened(sp, c, occ, aux, pi, pj, chunk=256,
                                                         valid=v, skip_pad=skip) ** 2)
        f = jax.jit(jax.grad(loss, argnums=(0, 1)))
        return f.lower(s, C).compile().memory_analysis().temp_size_in_bytes

    on, off = budget(True), budget(False)
    # Allow a little slack for the cond's own bookkeeping; what must not happen is the skip
    # multiplying the working set because the scan body stopped being rematerialized.
    assert on <= 1.25 * off, (
        f"skip_pad reserves {on} B against {off} B without it ({on / max(off, 1):.2f}x) — "
        f"the lax.cond is defeating the scan's rematerialization")


# --------------------------------------------------------------------------------------------
#  end to end through SplatKS: single device and (when available) the sharded mesh
# --------------------------------------------------------------------------------------------

def _water_ks(mesh=None, **screen):
    from gs_dft import SplatKS, df, init_model
    from gs_dft.benchmark import build_reference
    from gs_dft.ks.terms import pairlist
    from gs_dft.ks.train import native_grid
    from dftax.energy.xc import PBE
    mol, _, _, _ = build_reference("h2o", "cc-pvdz", "pbe", 1)
    model = init_model(mol, 30, jr.PRNGKey(0))
    grid = native_grid(mol, 1, chunk=512)
    ks = SplatKS(mol, PBE(), grid=grid, coulomb=df(), screen=pairlist(eps=1e-6, **screen), mesh=mesh)
    return ks, model


def _energy_and_grad(ks, model, pad):
    import equinox as eqx
    from gs_dft.ks.train import _refresh_state
    st = _refresh_state(ks, model.basis, pad, False)
    f = lambda m: ks(m, st)[0]
    return float(f(model)), eqx.filter(eqx.filter_grad(f)(model), eqx.is_array)


@pytest.mark.parametrize("mesh_on", [False, True])
def test_splatks_unique_skip_pad_parity(mesh_on):
    if mesh_on and jax.device_count() < 2:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=4")
    from gs_dft.ks import shard as shd
    mesh = shd.make_mesh() if mesh_on else None
    ks0, model = _water_ks(mesh)
    ks1, _ = _water_ks(mesh, unique=True, skip_pad=True)
    G = int(mesh.size) if mesh is not None else 1
    n_ord = int(scr.neighbor_pairs(model.basis, eps=1e-6)[0].shape[0])
    n_uni = int(scr.neighbor_pairs(model.basis, eps=1e-6, unique=True)[0].shape[0])
    assert n_uni < n_ord
    pad0 = -(-int(2 * n_ord) // G) * G
    pad1 = -(-int(2 * n_uni) // G) * G
    E0, g0 = _energy_and_grad(ks0, model, pad0)
    E1, g1 = _energy_and_grad(ks1, model, pad1)
    np.testing.assert_allclose(E1, E0, rtol=0, atol=1e-10)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g0)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-10)



# --------------------------------------------------------------------------------------------
#  the compiled-scan pair builder must reproduce the eager host build exactly
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("row_chunk", [16, 64])
def test_scan_builder_matches_the_eager_build(unique, row_chunk):
    """The fast path is a pure host-sync optimisation: identical pairs, identical order."""
    s = _splats(M=48)
    eps = 1e-6
    ref_i, ref_j, ref_v = scr.neighbor_pairs(s, eps=eps, unique=unique, row_chunk=row_chunk)
    n = int(ref_i.shape[0])
    for pad in (n, n + 37, 2 * n):
        pi, pj, v = scr.neighbor_pairs(s, eps=eps, pad_to=pad, unique=unique, row_chunk=row_chunk)
        assert pi.shape[0] == pad
        np.testing.assert_array_equal(np.asarray(pi)[:n], np.asarray(ref_i)[:n])
        np.testing.assert_array_equal(np.asarray(pj)[:n], np.asarray(ref_j)[:n])
        np.testing.assert_allclose(np.asarray(v)[:n], np.asarray(ref_v)[:n])
        assert not np.any(np.asarray(v)[n:])          # padding carries zero weight


def test_scan_builder_is_ascending_and_matches_dense():
    """Flat index i*M+j must come out ascending, and the set must
    equal a brute-force dense threshold."""
    from gs_dft.coulomb.stream import full_quantities
    s = _splats(M=40)
    M, eps = int(s.n_basis), 1e-6
    pi, pj, _ = scr.neighbor_pairs(s, eps=eps, pad_to=M * M, row_chunk=8)
    ri, _rj, _ = scr.neighbor_pairs(s, eps=eps, row_chunk=8)
    k = int(ri.shape[0])
    flat = np.asarray(pi)[:k].astype(np.int64) * M + np.asarray(pj)[:k]
    assert np.all(np.diff(flat) > 0), "flat indices are not strictly ascending"
    q = full_quantities(s)
    dense = np.abs(np.asarray(scr._full_overlap_pairs(
        q, jnp.repeat(jnp.arange(M), M), jnp.tile(jnp.arange(M), M))))
    want = np.sort(np.where(dense > eps)[0])
    np.testing.assert_array_equal(flat, want)


def test_count_matches_the_built_list():
    """The compiled count must agree with the length of the list actually built, in both modes —
    it is what sizes the pad, so an undercount would silently truncate the run's pair list."""
    from gs_dft.coulomb.stream import full_quantities
    from gs_dft.screening.pairs import _count_significant
    s = _splats(M=48)
    for unique in (False, True):
        for rc in (16, 64):
            n = int(_count_significant(full_quantities(s), int(s.n_basis), 1e-6, rc, unique))
            pi, _pj, _v = scr.neighbor_pairs(s, eps=1e-6, unique=unique, row_chunk=rc)
            assert n == int(pi.shape[0]), f"count {n} != built {int(pi.shape[0])} (unique={unique})"


def test_overflow_keeps_the_largest_overlaps_exactly():
    """The overflow rule is a real sort: with a pad below the significant count, the kept set must be
    exactly the pad_to largest by |S|, and still lexsorted."""
    from gs_dft.coulomb.stream import full_quantities
    from gs_dft.screening.pairs import _count_significant
    s = _splats(M=40)
    M, eps = int(s.n_basis), 1e-6
    n = int(_count_significant(full_quantities(s), M, eps, 16, True))
    pad = n // 2                                   # force overflow
    pi, pj, w = scr.neighbor_pairs(s, eps=eps, pad_to=pad, unique=True, row_chunk=16)
    assert pi.shape[0] == pad and int(np.sum(np.asarray(w) != 0)) == pad
    got = np.asarray(pi).astype(np.int64) * M + np.asarray(pj)
    assert np.all(np.diff(got) > 0), "kept pairs are not lexsorted"
    # brute force: every significant pair, sorted by |S| descending, take pad
    fi, fj, _ = scr.neighbor_pairs(s, eps=eps, unique=True, row_chunk=16)
    q = full_quantities(s)
    v = np.abs(np.asarray(scr._full_overlap_pairs(q, fi, fj)))
    flat = np.asarray(fi).astype(np.int64) * M + np.asarray(fj)
    want = np.sort(flat[np.argsort(-v)[:pad]])
    np.testing.assert_array_equal(got, want)
