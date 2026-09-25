"""The grid layout follows the mesh the SplatKS was built for.

The grid is padded to a multiple of ``mesh.size``, not of the host device count, and
``refresh(like=)`` keeps the previous padding so shapes stay constant.
"""
import equinox as eqx
import jax                                     # bare `jax` used by monkeypatch.setattr below
import jax.numpy as jnp
import jax.random as jr

from gs_dft import SplatKS, SplatModel, refresh, df, pairlist
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.ks import shard as shd
from gs_dft.ks.train import _prepare_mesh
from dftax.energy.xc import PBE
from dftax.grid import points

def _tiny_ks(mesh, n_grid):
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    charges = jnp.array([1.0, 1.0])
    gp = jnp.linspace(-1.0, 1.0, 3 * n_grid).reshape(n_grid, 3)
    gw = jnp.full((n_grid,), 0.1)
    return SplatKS((coords, charges), PBE(), grid=points(gp, gw, chunk=4096),
                   coulomb=df(), screen=pairlist(eps=1e-6), mesh=mesh)

def _tiny_model(key=0):
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    basis = init_spectral_splats(coords, 4, key=jr.PRNGKey(key), aniso_jitter=0.1)
    C = 0.3 * jr.normal(jr.PRNGKey(key + 1), (4, 1))
    return SplatModel(basis=basis, C=C, occupations=jnp.array([2.0]))

def test_build_pad_follows_mesh_size_not_host_device_count(monkeypatch):
    """A 1-device mesh on a host that claims 4 local devices must pad to a
    multiple of 1 (i.e. not at all) — the mesh, not the host, drives layout,
    and it drives it at BUILD time."""
    monkeypatch.setattr(jax, "local_device_count", lambda *a, **k: 4)
    mesh = shd.make_mesh(1)
    ks = _tiny_ks(mesh, n_grid=10)          # 10 % 4 != 0: host-count padding would grow it
    assert ks.grid_points.shape[0] == 10    # laid out at construction, unpadded for G=1
    ks2, _model, G, multinode, rank0 = _prepare_mesh(ks, _tiny_model())
    assert G == mesh.size == 1
    assert not multinode and rank0
    assert ks2.grid_points.shape[0] == 10   # the verb never rewrites the layout
    assert float(jnp.sum(ks2.grid_weights)) == float(jnp.sum(ks.grid_weights))

def test_refresh_like_inherits_the_pad():
    """refresh(like=prev) keeps the pair shapes constant as the splats move."""
    ks = _tiny_ks(None, n_grid=10)
    model = _tiny_model()
    state0 = refresh(ks, model.basis, pad_to=4 * 4 + 9)     # some fixed pad
    moved = eqx.tree_at(lambda b: b.centers, model.basis,
                        model.basis.centers + 0.05)
    state1 = refresh(ks, moved, like=state0)
    assert state1.pairs[0].shape == state0.pairs[0].shape
    assert state1.pairs[2].shape == state0.pairs[2].shape
    # without `like`, the list is unpadded and may change shape as splats move
    bare = refresh(ks, moved)
    assert bare.pairs[2].all()                              # no padding entries
