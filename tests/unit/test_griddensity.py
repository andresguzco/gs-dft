"""The chunked grid density is exact, and skips the AO gradient when it is not needed."""
import numpy as np
import jax.numpy as jnp
import pytest

from experiments.common import griddensity, systems


@pytest.fixture(scope="module")
def water_case():
    from dftax.basis.loader import build_basis_data
    mol = systems.molecule(system="water", basis="sto-3g")
    bd = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    n = int(np.asarray(bd.cart2sph).shape[1])
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n, n))
    P = jnp.asarray(A @ A.T)                       # symmetric, like a real density matrix
    pts = jnp.asarray(rng.normal(scale=1.5, size=(257, 3)))
    return bd, P, pts


def _reference(bd, P, pts):
    from dftax.ks.energy import ao_on_grid
    ao, _dao = ao_on_grid(bd, pts)
    return np.asarray(jnp.einsum("gm,mn,gn->g", ao, P, ao))


@pytest.mark.parametrize("chunk", [1024, 257, 100, 64, 1])
def test_chunking_is_exact(water_case, chunk):
    bd, P, pts = water_case
    got = griddensity.gto_density(bd, P, pts, chunk=chunk)
    assert got.shape == (pts.shape[0],)
    assert np.abs(got - _reference(bd, P, pts)).max() == pytest.approx(0.0, abs=1e-12)


def test_ragged_tail_is_not_truncated_or_padded_into_the_result(water_case):
    """257 points with chunk=100 leaves a 57-point tail. The tail is padded to the common shape so
    the jit compiles once, and the padding must be sliced back off -- a length change here would be
    silent, since the caller only ever integrates the array against weights."""
    bd, P, pts = water_case
    got = griddensity.gto_density(bd, P, pts, chunk=100)
    assert got.shape == (257,)
    assert np.isfinite(got).all()


def test_ao_values_has_no_gradient_axis(water_case):
    """The whole point: (ngrid, nao), never (ngrid, nao, 3)."""
    bd, _P, pts = water_case
    assert griddensity.ao_values(bd, pts).ndim == 2
