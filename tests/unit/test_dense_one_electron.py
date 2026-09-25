"""The screened path builds the overlap and kinetic matrices densely.

Screening a kinetic matrix can make it indefinite and the kinetic energy negative, so S and T are
always dense: the kinetic energy equals the fully dense value and is non-negative.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from dftax.system import Molecule
from dftax.grid import becke_grid
from gs_dft.integrals import dense as fullmod
from gs_dft import screening as scr, nao
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.basis.isotropic import init_isotropic_splats
from gs_dft.ks.energy import SplatModel, SplatKS, SplatState
from gs_dft.ks.terms import df, pairlist
from dftax.energy.xc import PBE

pytestmark = pytest.mark.slow                       # Molecule + becke grid + mesh build on every test
from dftax.grid import points

def _setup():
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 1.4; O 0 0 3.0", "sto-3g", unit="bohr", spherical=True)
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), float)
    M = 2 * nao(mol)
    basis = init_spectral_splats(coords, M, key=jr.PRNGKey(0), aniso_jitter=0.1)
    n_occ = 4
    C = 0.3 * jr.normal(jr.PRNGKey(1), (M, n_occ))
    model = SplatModel(basis=basis, C=C, occupations=jnp.full((n_occ,), 2.0))
    aux = init_isotropic_splats(coords, nao(mol), key=jr.PRNGKey(2))
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=20, lebedev=50)
    ef_d = SplatKS((coords, charges), PBE(), grid=(gp, gw), coulomb=df())
    ef_s = SplatKS((coords, charges), PBE(), grid=points(gp, gw, chunk=4096), coulomb=df(),
                   screen=pairlist(eps=1e-4))
    return model, aux, gp, gw, ef_d, ef_s

def test_dense_kinetic_matrix_is_psd():
    """The exact kinetic matrix T = ½⟨∇gᵢ,∇gⱼ⟩ is PSD ⇒ Θ = CᵀTC ⪰ 0 for any C ⇒ E_kin ≥ 0.
    A screened kinetic matrix, in contrast, may be indefinite."""
    model, *_ = _setup()
    T = fullmod.kinetic_matrix(model.basis)
    lam_min_T = float(jnp.linalg.eigvalsh(0.5 * (T + T.T)).min())
    Theta = model.C.T @ T @ model.C
    lam_min_theta = float(jnp.linalg.eigvalsh(0.5 * (Theta + Theta.T)).min())
    assert lam_min_T >= -1e-8, f"dense kinetic not PSD: λmin(T)={lam_min_T:.2e}"
    assert lam_min_theta >= -1e-8, f"occupied kinetic Gram not PSD: λmin(Θ)={lam_min_theta:.2e}"

def test_screened_kinetic_equals_dense_under_screening():
    """With FINITE screening (pairs dropped), the screened path's reported E_kinetic must equal the
    fully-dense path's E_kinetic — because the fix builds S and T densely in BOTH. Before the fix the
    screened path used the screened T̃ and would differ (and could go negative)."""
    model, aux, gp, gw, ef_d, ef_s = _setup()
    pi, pj, v = scr.neighbor_pairs(model.basis, eps=1e-4)          # finite ε ⇒ some pairs dropped
    assert int(jnp.sum(v)) < model.basis.n_basis ** 2, "test needs pairs actually dropped"

    _, aux_d = ef_d(model, SplatState(aux=aux))
    _, aux_s = jax.jit(lambda mm: ef_s(mm, SplatState(aux=aux, pairs=(pi, pj, v))))(model)

    E_kin_d, E_kin_s = float(aux_d.kinetic), float(aux_s.kinetic)
    assert E_kin_s > 0.0, f"E_kin should be ≥ 0 (dense T is PSD); got {E_kin_s:.4f}"
    np.testing.assert_allclose(E_kin_s, E_kin_d, rtol=1e-9, atol=1e-8)

def test_sharded_path_matches_single_device():
    """The sharded path (mesh≠None) also builds S/T densely and adds the dense kinetic (the sharded
    call computes no kinetic of its own). On a 1-device mesh (trivial sharding) it must reproduce the
    single-device screened energy AND E_kinetic exactly, with E_kin ≥ 0."""
    from gs_dft.ks import shard as shd
    model, aux, gp, gw, _ef_d, ef_s = _setup()
    pi, pj, v = scr.neighbor_pairs(model.basis, eps=1e-4)
    E_s, aux_s = ef_s(model, SplatState(aux=aux, pairs=(pi, pj, v)))

    mesh = shd.make_mesh(1)
    ef_m = SplatKS((ef_s.atom_coords, ef_s.atom_charges), PBE(),
                   grid=points(gp, gw, chunk=4096), coulomb=df(),
                   screen=pairlist(eps=1e-4), mesh=mesh)
    with jax.set_mesh(mesh):
        E_m, aux_m = ef_m(model, SplatState(aux=aux, pairs=(pi, pj, v)))

    assert float(aux_m.kinetic) >= 0.0
    np.testing.assert_allclose(float(E_m), float(E_s), rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(float(aux_m.kinetic), float(aux_s.kinetic), rtol=1e-9, atol=1e-7)
