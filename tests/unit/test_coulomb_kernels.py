"""The rewritten Coulomb kernels against the forms they replaced (kept here only as references):
the RI-J node sum (division-free forward, node sums accumulated in the reverse) and the V_ne stream
(eigenframe point-charge kernel, reverse made in the forward). Values and gradients to round-off."""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.coulomb import ri as cdf
from gs_dft.coulomb.stream import _full_pair_quantities, full_quantities
from gs_dft.integrals import dense
from gs_dft.integrals.linalg3 import adj3, det3
from gs_dft.screening import pairs as scr

_I3 = np.eye(3)


@pytest.fixture(autouse=True)
def short_table(monkeypatch):
    """A 12-node half-line table in every kernel: the algebra under test does not depend on the
    table, and the unrolled 64-node loops (autodiff'd in the references) compile slowly."""
    t, w = dense._gauss_legendre_halfline(12)
    tl, wl = [float(x) for x in t], [float(x) for x in w]
    for mod in (dense, scr):
        monkeypatch.setattr(mod, "_QUAD_T_LIST", tl)
        monkeypatch.setattr(mod, "_QUAD_W_LIST", wl)
    nodes, vr = cdf._make_isobra_nodes(tuple(tl), tuple(wl))
    monkeypatch.setattr(cdf, "_isobra_nodes", nodes)
    monkeypatch.setattr(cdf, "_isobra_value_and_reverse", vr)


def _ref_isobra_nodes(beta, dp2, Lam):
    """The node sum as it was before (one division a node), differentiated by autodiff."""
    b = beta[:, None]
    L0, L1, L2 = Lam[None, :, 0], Lam[None, :, 1], Lam[None, :, 2]
    coul = jnp.zeros((beta.shape[0], Lam.shape[0]))
    for t, w in zip(dense._QUAD_T_LIST, dense._QUAD_W_LIST):
        t2 = t * t
        G0, G1, G2 = (b * L + t2 * (b + L) for L in (L0, L1, L2))
        invp = 1.0 / (G0 * G1 * G2)
        num = b * (L0 * dp2[..., 0] * G1 * G2 + L1 * dp2[..., 1] * G0 * G2
                   + L2 * dp2[..., 2] * G0 * G1)
        coul = coul + w * jnp.sqrt(invp) * jnp.exp(-t2 * num * invp)
    return coul


def _ref_nuclear_pairs(q, pi, pj, coords, charges):
    """V_ij in the original frame (B = Â_ij + t²I, adjugate and determinant), as it was before."""
    Aij, mup, NK = _full_pair_quantities(q, pi, pj)
    d = mup[:, None, :] - coords[None, :, :]
    d2 = jnp.sum(d * d, axis=-1)
    vA = jnp.zeros(d2.shape)
    for t, w in zip(dense._QUAD_T_LIST, dense._QUAD_W_LIST):
        t2 = t * t
        B = Aij + t2 * _I3
        invdet = 1.0 / det3(B)
        qf = jnp.einsum("pak,pkl,pal->pa", d, adj3(B), d)
        expo = t2 * d2 - (t2 * t2) * qf * invdet[:, None]
        vA = vA + (w * jnp.pi ** 1.5) * jnp.sqrt(invdet)[:, None] * jnp.exp(-expo)
    return -jnp.sum(charges[None, :] * (2.0 / jnp.sqrt(jnp.pi)) * vA, axis=1) * NK


def _close(a, b, rtol):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        scale = max(float(np.max(np.abs(y))), 1e-300)
        np.testing.assert_allclose(x, y, rtol=rtol, atol=rtol * scale)


def test_isobra_nodes_value_and_reverse_match_the_plain_loop():
    k = jr.split(jr.PRNGKey(0), 4)
    A, B = 37, 53
    beta = jnp.exp(jr.uniform(k[0], (A,), minval=-3.0, maxval=5.0))
    Lam = jnp.exp(jr.uniform(k[1], (B, 3), minval=-3.0, maxval=6.0))
    dp2 = jr.uniform(k[2], (A, B, 3), minval=0.0, maxval=20.0)
    ct = jr.normal(k[3], (A, B))
    v, vjp = jax.vjp(cdf._isobra_nodes, beta, dp2, Lam)
    v0, vjp0 = jax.vjp(_ref_isobra_nodes, beta, dp2, Lam)
    _close(v, v0, 1e-13)
    _close(vjp(ct), vjp0(ct), 1e-12)


def _system(M=40, n_atoms=6, seed=0):
    k = jr.split(jr.PRNGKey(seed), 4)
    coords = jr.uniform(k[0], (n_atoms, 3), minval=-3.0, maxval=3.0)
    charges = jnp.array([6.0, 1.0, 8.0, 1.0, 7.0, 1.0])[:n_atoms]
    s = init_spectral_splats(coords, M, key=k[1], aniso_jitter=0.3)
    C = jr.normal(k[2], (M, 5))
    occ = 2.0 * jnp.ones(5)
    return s, C, occ, coords, charges


def test_external_stream_matches_the_original_frame_kernel():
    s, C, occ, coords, charges = _system()
    pi, pj, w = scr.neighbor_pairs(s, eps=1e-8, unique=True, pad_to=1200)

    def ref(sp, c, R, Z):
        V = _ref_nuclear_pairs(full_quantities(sp), pi, pj, R, Z)
        return jnp.sum(w * scr.density_pairs(c, occ, pi, pj) * V)

    def new(sp, c, R, Z):
        return scr.screened_external_energy(sp, c, occ, pi, pj, R, Z, w, chunk=256, skip_pad=True)

    args = (s, C, coords, charges)
    _close(new(*args), ref(*args), 1e-12)
    g = jax.grad(new, argnums=(0, 1, 2, 3))(*args)
    g0 = jax.grad(ref, argnums=(0, 1, 2, 3))(*args)
    _close(g, g0, 1e-10)
    # jit, and the plain forward (no reverse) agrees with the forward that makes the reverse
    e_jit = jax.jit(new)(*args)
    e_vg = jax.jit(jax.value_and_grad(new))(*args)[0]
    _close(e_jit, e_vg, 1e-14)


def test_nuclear_pairs_match_the_original_frame_kernel():
    s, _C, _occ, coords, charges = _system(M=24)
    pi, pj, _ = scr.neighbor_pairs(s, eps=-1.0)
    got = scr.screened_nuclear_pairs(s, pi, pj, coords, charges)
    ref = _ref_nuclear_pairs(full_quantities(s), pi, pj, coords, charges)
    _close(got, ref, 1e-12)


def _ref_gamma(s, C, occ, aux, pi, pj, w):
    """γ through the plain block kernel and autodiff (the stream as it was, without its own VJP)."""
    pair_q = _full_pair_quantities(full_quantities(s), pi, pj)
    block = cdf.full_isobra_block(*cdf.aux_quantities(aux), *pair_q)
    return block @ (w * scr.density_pairs(C, occ, pi, pj))


@pytest.mark.parametrize("tile", [None, 7, 48])
def test_gamma_stream_value_and_reverse_match_autodiff(tile, monkeypatch):
    """Value and reverse (splats, coefficients and the aux) against autodiff of the plain block, in
    one aux tile, in tiles that do not divide the aux (a padded last tile), and in tiles that do."""
    from gs_dft.basis.isotropic import init_product_aux
    if tile is not None:
        monkeypatch.setattr(cdf, "_AUX_TILE", tile)
    s, C, occ, _coords, _charges = _system(M=48)
    aux = init_product_aux(s, scales=(1.0,), diagonal_only=True)
    pi, pj, w = scr.neighbor_pairs(s, eps=1e-8, unique=True, pad_to=1600)
    ct = jr.normal(jr.PRNGKey(7), (aux.centers.shape[0],))

    def new(sp, c, ax, skip):
        return cdf.df_coulomb_gamma_screened(sp, c, occ, ax, pi, pj, chunk=256, valid=w,
                                             skip_pad=skip)

    v0, vjp0 = jax.vjp(lambda sp, c, ax: _ref_gamma(sp, c, occ, ax, pi, pj, w), s, C, aux)
    for skip in (False, True):
        v, vjp = jax.vjp(lambda sp, c, ax: new(sp, c, ax, skip), s, C, aux)
        _close(v, v0, 1e-12)
        _close(vjp(ct), vjp0(ct), 1e-10)


def test_screened_coulomb_energy_is_the_same_in_ket_blocks():
    """The reported E_J streamed in ket blocks (padded with the first pair at zero weight) equals the
    one-block sum: only the summation order changes."""
    from gs_dft.coulomb.stream import screened_coulomb_energy
    s, C, occ, _coords, _charges = _system(M=12, n_atoms=4, seed=3)
    pi, pj, w = scr.neighbor_pairs(s, eps=1e-12, unique=True)
    one = screened_coulomb_energy(s, C, occ, pi, pj, w, bra_chunk=5, ket_chunk=10 ** 9)
    for kc in (7, 16):
        _close(screened_coulomb_energy(s, C, occ, pi, pj, w, bra_chunk=5, ket_chunk=kc), one, 1e-12)
