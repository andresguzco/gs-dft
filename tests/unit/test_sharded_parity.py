"""The sharded step equals the single-device screened step, value and gradient, on a multi-device
mesh. Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``; skipped otherwise.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft import SplatKS, df, init_model, refresh
from gs_dft.ks import shard as shd
from gs_dft.ks.terms import pairlist
from gs_dft.ks.train import native_grid, _refresh_state
from dftax.energy.xc import PBE


@pytest.mark.skipif(jax.device_count() < 2, reason="needs a multi-device mesh (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
def test_sharded_step_matches_single_device():
    from gs_dft.benchmark import build_reference
    mol, _, _, _ = build_reference("h2o", "cc-pvdz", "pbe", 1)
    model = init_model(mol, 30, jr.PRNGKey(0))
    grid = native_grid(mol, 1, chunk=512)
    ks1 = SplatKS(mol, PBE(), grid=grid, coulomb=df(), screen=pairlist(eps=1e-6))
    mesh = shd.make_mesh()
    ksm = SplatKS(mol, PBE(), grid=grid, coulomb=df(), screen=pairlist(eps=1e-6), mesh=mesh)
    G = int(mesh.size)
    from gs_dft import screening as scr
    n0 = int(scr.neighbor_pairs(model.basis, eps=1e-6)[0].shape[0])
    pad = -(-int(2 * n0) // G) * G
    st1 = refresh(ks1, model.basis, pad_to=pad)
    stm = _refresh_state(ksm, model.basis, pad, False)

    def e1(m):
        return ks1(m, st1)[0]

    def em(m):
        return ksm(m, stm)[0]

    E1, Em = float(e1(model)), float(em(model))
    np.testing.assert_allclose(Em, E1, rtol=0, atol=1e-9)
    import equinox as eqx
    g1 = eqx.filter_grad(e1)(model)
    gm = eqx.filter_grad(em)(model)
    for a, b in zip(jax.tree.leaves(eqx.filter(gm, eqx.is_array)), jax.tree.leaves(eqx.filter(g1, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-10)
