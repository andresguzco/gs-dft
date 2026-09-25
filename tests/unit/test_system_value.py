"""SplatKS takes the system as one value and checks it: a model built for a different molecule is
rejected instead of evaluated.
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from dftax.system import Molecule

from gs_dft import SplatKS, init_model, chem, evaluate, native_grid
from dftax.energy.xc import PBE
from dftax.grid import becke

def _h2():
    return Molecule.from_xyz("H 0 0 0; H 0 0 0.74", "sto-3g", spherical=True)

def test_mole_and_raw_pair_agree():
    mol = _h2()
    model = init_model(mol, 6, jr.PRNGKey(0))
    grid = native_grid(mol, 0)
    ks_mol = SplatKS(mol, PBE(), grid=grid)
    ks_raw = SplatKS((mol.atom_coords(), mol.atom_charges()), PBE(), grid=grid)
    e_mol, e_raw = float(ks_mol(model)[0]), float(ks_raw(model)[0])
    assert e_mol == e_raw                      # resolution changes nothing numerically
    assert ks_mol.nelec == 2 and ks_raw.nelec == 0

def test_electron_count_guard_fires():
    mol = _h2()
    water = Molecule.from_xyz("O 0 0 0; H 0.76 0 0.5; H 0.76 0 -0.5", "sto-3g", spherical=True)
    model_water = init_model(water, 12, jr.PRNGKey(0))     # 10 electrons
    ks_h2 = SplatKS(mol, PBE(), grid=native_grid(mol, 0))   # 2 electrons
    with pytest.raises(ValueError, match="different systems"):
        evaluate(ks_h2, model_water)

def test_chem_init_spec():
    mol = _h2()
    m = init_model(mol, 6, jr.PRNGKey(0), init=chem(bond_fraction=0.5))
    assert m.basis.n_basis == 6 and float(jnp.sum(m.occupations)) == 2.0
    with pytest.raises(ValueError, match="alloc"):
        chem(alloc="typo")
    with pytest.raises(TypeError, match="chem"):
        init_model(mol, 6, jr.PRNGKey(0), init="chem")     # init takes a value, not a string

def test_becke_grid_spec_accepted():
    mol = _h2()
    ks = SplatKS(mol, PBE(), grid=becke(20, 50))           # native PySCF-free grid
    model = init_model(mol, 6, jr.PRNGKey(0))
    assert np.isfinite(float(ks(model)[0]))
    with pytest.raises(ValueError, match="symbols"):
        SplatKS((mol.atom_coords(), mol.atom_charges()), PBE(), grid=becke(20, 50))
