"""The sharded step on a mesh spanning several processes equals one device's, value and gradient.

Run one process per GPU across nodes, e.g. ``srun -N2 --ntasks-per-node=4 python -m pytest
tests/unit/test_multinode_parity.py``. Skipped in a single process.
"""
from gs_dft.ks import shard as shd

MULTI = shd.init_distributed()

import importlib

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import pytest

from dftax.energy.xc import PBE
from dftax.system import Molecule
from gs_dft import SplatKS, df, init_model, refresh
from gs_dft.geometries import h2o_geometry
from gs_dft.ks.terms import pairlist
from gs_dft.screening import count_pairs

tr = importlib.import_module("gs_dft.ks.train")

pytestmark = pytest.mark.skipif(not MULTI, reason="needs a multi-process run (srun -N2 ...)")


def test_multinode_step_matches_one_device():
    mol = Molecule.from_xyz(h2o_geometry, "cc-pvdz", unit="angstrom", spherical=True)
    model = tr._on_one_device(init_model(mol, 30, jr.PRNGKey(0)))
    grid = tr.native_grid(mol, 1, chunk=512)
    ks1 = SplatKS(mol, PBE(), grid=grid, coulomb=df(lam=1e-8), screen=pairlist(eps=-1.0))
    ksm = SplatKS(mol, PBE(), grid=grid, coulomb=df(lam=1e-8), screen=pairlist(eps=-1.0),
                  mesh=shd.make_mesh())
    G = int(ksm.mesh.size)
    pad = -(-2 * count_pairs(model.basis, eps=-1.0, unique=True) // G) * G
    # ks as an argument, as train() passes it: a jit cannot close over the global grid of a multi-node ks
    step = eqx.filter_jit(eqx.filter_value_and_grad(lambda m, s, ks: ks(m, s)[0]))
    E1, g1 = step(model, refresh(ks1, model.basis, pad_to=pad), ks1)
    ksm, mm, _, multinode, _ = tr._prepare_mesh(ksm, model)
    assert multinode
    Em, gm = step(mm, tr._refresh_state(ksm, mm.basis, pad, True), ksm)
    np.testing.assert_allclose(float(Em.addressable_data(0)), float(E1), rtol=0, atol=1e-9)
    a, b = jax.tree.leaves(gm), jax.tree.leaves(g1)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(np.asarray(x.addressable_data(0)), np.asarray(y), rtol=1e-8, atol=1e-10)
