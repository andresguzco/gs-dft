"""The frozen pair-product auxiliary for RI-K exchange (``gs_dft.basis.isotropic.init_product_aux``).

The pair-product auxiliary spans the occupied-orbital products, so it reproduces the exact exchange
energy of water to RI accuracy without being optimized.
"""
import pytest
import jax.numpy as jnp
import jax.random as jr
from dftax.grid import becke_grid

from gs_dft.benchmark import build_reference
from gs_dft import init_model, chem
from gs_dft.ks.energy import SplatKS, SplatState
from gs_dft.ks.orthonormalize import lowdin_orthonormalize
from gs_dft.basis.isotropic import init_product_aux
from gs_dft.coulomb import ri as df
from gs_dft.ks.terms import df as df_coulomb
from dftax.energy.xc import PBE, PBE0

pytestmark = pytest.mark.slow                       # build_reference SCF oracle on every test

from gs_dft.integrals import dense as BE

def _water():
    mol, _e, nao, mf = build_reference("h2o", basis="cc-pvdz", grid_level=1)
    model = init_model(mol, nao, jr.PRNGKey(0), init=chem())
    coords = jnp.array(mol.atom_coords()); charges = jnp.array(mol.atom_charges(), float)
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=35, lebedev=110)
    return model, coords, charges, gp, gw

def test_frozen_product_kaux_matches_exact_exchange():
    """RI-K with the frozen product K-aux is finite, correctly signed, and reproduces the exact exchange
    energy to RI accuracy — the SAME product-aux constructor serves K as J (no descent)."""
    model, coords, charges, _, _ = _water()
    S, _, _ = BE.one_electron_integrals(model.basis, coords, charges)
    C_orth = lowdin_orthonormalize(model.C, S)
    aux_K = init_product_aux(model.basis)                       # production config (frozen, diagonal, scale 1)
    E_K_rik = float(df.df_exchange_energy(model.basis, C_orth, aux_K))
    P = 2.0 * C_orth @ C_orth.T
    E_K_exact = float(BE.exchange_energy(BE.exchange_matrix(BE.compute_eri_tensor(model.basis), P), P))
    assert jnp.isfinite(E_K_rik)
    assert E_K_rik < 0.0                                        # exchange is negative
    assert abs(E_K_rik - E_K_exact) < 0.02 * abs(E_K_exact)     # RI-K within ~2% of exact (measured ~0.5%)

def test_hybrid_energy_runs_with_frozen_product_auxes():
    """The full PBE0 hybrid energy through SplatKS with the frozen product aux for BOTH J and K —
    finite total + finite per-term components."""
    model, coords, charges, gp, gw = _water()
    aux, aux_K = init_product_aux(model.basis), init_product_aux(model.basis)
    ks = SplatKS((coords, charges), PBE0(), grid=(gp, gw), coulomb=df_coulomb())
    E, eaux = ks(model, SplatState(aux=aux, aux_K=aux_K))
    assert jnp.isfinite(E)
    assert jnp.isfinite(eaux.kinetic) and jnp.isfinite(eaux.hartree) and jnp.isfinite(eaux.xc)

def test_pbe_j_only_product_aux_runs():
    """PBE (no exchange) with the frozen product J-aux — finite E (the J path is intact)."""
    model, coords, charges, gp, gw = _water()
    ks = SplatKS((coords, charges), PBE(), grid=(gp, gw), coulomb=df_coulomb())
    E, _ = ks(model, SplatState(aux=init_product_aux(model.basis)))
    assert jnp.isfinite(E)
