"""Pair screening for the splat one-electron operators (``gs_dft/screening/pairs.py``).

The screened matvec is exact as the threshold goes to zero and bounded at a finite threshold, the
jitted and eager paths agree, the neighbour degree stays far below M, and the screened SplatKS
energy and gradient match the dense path.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from dftax.system import Molecule
from gs_dft.integrals import dense as fullmod
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.basis.isotropic import init_isotropic_splats
from gs_dft import screening as scr, nao

BACKENDS = ["splat"]
_MOD = {"splat": fullmod}

def _splats(backend, M=60, seed=0):
    # a spread of centers + exponents so screening actually drops pairs (not all-overlapping)
    key = jr.PRNGKey(seed)
    k1, k2 = jr.split(key)
    centers = jr.uniform(k1, (8, 3), minval=-4.0, maxval=4.0)   # 8 "atoms"
    # aniso_jitter: off the degenerate isotropic init (the established full-cov pattern)
    return init_spectral_splats(centers, M, key=k2, aniso_jitter=0.3)

def _atoms(seed=20):
    coords = jr.uniform(jr.PRNGKey(seed), (8, 3), minval=-4.0, maxval=4.0)
    charges = jnp.array([6.0, 1.0, 8.0, 1.0, 6.0, 7.0, 1.0, 8.0])
    return coords, charges

@pytest.mark.parametrize("backend", BACKENDS)
def test_nuclear_pairs_match_dense(backend):
    """Screened V_ij on the pair list equals the dense matrix entries exactly.
    (No kinetic pair kernel exists: S/T are always dense — screening them tips
    the occupied Grams indefinite.)"""
    s = _splats(backend)
    coords, charges = _atoms()
    mod = _MOD[backend]
    V = np.asarray(mod.nuclear_attraction_matrix(s, coords, charges))
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-6)
    ai, aj = np.asarray(pi), np.asarray(pj)
    np.testing.assert_allclose(
        np.asarray(scr.screened_nuclear_pairs(s, pi, pj, coords, charges)),
        V[ai, aj], rtol=1e-11, atol=1e-11)

@pytest.mark.parametrize("backend", BACKENDS)
def test_external_energy_exact_when_unscreened(backend):
    """E_ext over all pairs == Tr(P·V_ne) with the dense matrix. (Only V_ne is screened — S/T are
    always dense; screening the kinetic tips the Grams indefinite.)"""
    s = _splats(backend)
    coords, charges = _atoms()
    mod = _MOD[backend]
    C = jr.normal(jr.PRNGKey(4), (s.n_basis, 5))
    occ = jnp.abs(jr.normal(jr.PRNGKey(5), (5,))) + 0.5
    P = C @ jnp.diag(occ) @ C.T
    V = mod.nuclear_attraction_matrix(s, coords, charges)
    E_ext_ref = float(jnp.sum(P * V))
    pi, pj, _ = scr.neighbor_pairs(s, eps=-1.0)
    E_ext = scr.screened_external_energy(s, C, occ, pi, pj, coords, charges)
    np.testing.assert_allclose(float(E_ext), E_ext_ref, rtol=1e-10, atol=1e-10)

def test_external_energy_bounded_error_screened():
    """At a finite ε the screened external energy stays close to the dense Tr(P·V_ne)."""
    s = _splats("splat", M=80)
    coords, charges = _atoms()
    C = jr.normal(jr.PRNGKey(6), (s.n_basis, 5))
    occ = jnp.ones(5)
    P = C @ jnp.diag(occ) @ C.T
    Ee_ref = float(jnp.sum(P * fullmod.nuclear_attraction_matrix(s, coords, charges)))
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-7)
    Ee = scr.screened_external_energy(s, C, occ, pi, pj, coords, charges)
    assert abs(float(Ee) - Ee_ref) < 1e-3

@pytest.mark.parametrize("backend", BACKENDS)
def test_screened_overlap_pairs_match_dense(backend):
    """S_ij on the pair list equals the dense overlap entries exactly."""
    s = _splats(backend)
    S = _MOD[backend].overlap_matrix(s)
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-6)
    s_pair = scr.screened_overlap_pairs(s, pi, pj)
    np.testing.assert_allclose(np.asarray(s_pair), np.asarray(S)[np.asarray(pi), np.asarray(pj)],
                               rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize("backend", BACKENDS)
def test_matvec_exact_when_no_screening(backend):
    """With ε<0 no pair is dropped ⇒ screened S@C == dense S@C exactly (the identity baseline)."""
    s = _splats(backend)
    S = _MOD[backend].overlap_matrix(s)
    C = jr.normal(jr.PRNGKey(3), (s.n_basis, 4))
    pi, pj, valid = scr.neighbor_pairs(s, eps=-1.0)            # normalized overlap ≥ 0 ⇒ keep all
    assert pi.shape[0] == s.n_basis ** 2                       # every ordered pair kept
    got = scr.screened_overlap_matvec(s, pi, pj, C)
    np.testing.assert_allclose(np.asarray(got), np.asarray(S @ C), rtol=1e-11, atol=1e-11)

def test_matvec_bounded_error_and_locality():
    """At a finite ε the S@C error is small AND the degree is well below M (the O(M) payoff)."""
    s = _splats("splat", M=80)
    S = fullmod.overlap_matrix(s)
    C = jr.normal(jr.PRNGKey(5), (s.n_basis, 4))
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-5)
    degree = pi.shape[0] / s.n_basis
    assert degree < 0.7 * s.n_basis                            # screening actually drops pairs
    got = scr.screened_overlap_matvec(s, pi, pj, C)
    err = float(jnp.max(jnp.abs(got - S @ C)))
    assert err < 1e-3, f"screened S@C error {err:.2e} too large at eps=1e-5"

def test_full_neighbor_pairs_bounded_error():
    """Full-cov host neighbor build at finite ε: pairs dropped AND the screened S@C error ~ε."""
    s = _splats("splat", M=80)
    S = fullmod.overlap_matrix(s)
    C = jr.normal(jr.PRNGKey(5), (s.n_basis, 4))
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-5)
    assert pi.shape[0] < 0.9 * s.n_basis ** 2                  # screening actually drops pairs
    got = scr.screened_overlap_matvec(s, pi, pj, C)
    err = float(jnp.max(jnp.abs(got - S @ C)))
    assert err < 1e-3, f"screened full-cov S@C error {err:.2e} too large at eps=1e-5"

def test_padded_path_matches_eager_and_jits():
    """The padded (static-shape) list gives the same matvec as the eager one, and jits."""
    s = _splats("splat")
    C = jr.normal(jr.PRNGKey(7), (s.n_basis, 3))
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-5)
    pip, pjp, valid = scr.neighbor_pairs(s, eps=1e-5, pad_to=pi.shape[0] + 37)
    eager = scr.screened_overlap_matvec(s, pi, pj, C)
    jitted = jax.jit(lambda sp, a, b, v: scr.screened_overlap_matvec(sp, a, b, C, valid=v))(s, pip, pjp, valid)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-11, atol=1e-11)

def _energy_setup(backend):
    from dftax.grid import becke_grid
    from gs_dft.ks.energy import SplatModel, SplatKS
    from gs_dft.ks.terms import df, pairlist
    from dftax.energy.xc import PBE
    from dftax.grid import points

    mol = Molecule.from_xyz("H 0 0 0; H 0 0 1.4; O 0 0 3.0", "sto-3g", unit="bohr", spherical=True)
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), float)
    M = 2 * nao(mol)
    basis = init_spectral_splats(coords, M, key=jr.PRNGKey(0), aniso_jitter=0.1)  # off the degenerate iso init
    n_occ = 4
    C = 0.3 * jr.normal(jr.PRNGKey(1), (M, n_occ))
    model = SplatModel(basis=basis, C=C, occupations=jnp.full((n_occ,), 2.0))
    aux = init_isotropic_splats(coords, nao(mol), key=jr.PRNGKey(2))
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=20, lebedev=50)
    ef_d = SplatKS((coords, charges), PBE(), grid=(gp, gw), coulomb=df())
    ef_s = SplatKS((coords, charges), PBE(), grid=points(gp, gw, chunk=4096), coulomb=df(),
                   screen=pairlist(eps=-1.0))
    return model, aux, gp, gw, ef_d, ef_s

@pytest.mark.parametrize("backend", BACKENDS)
def test_splat_energy_screened_matches_dense(backend):
    """End-to-end: the screened SplatKS with the full pair list == the dense density_fit
    path (and the screened path jits)."""
    model, aux, gp, gw, ef_d, ef_s = _energy_setup(backend)
    pi, pj, v = scr.neighbor_pairs(model.basis, eps=-1.0)
    from gs_dft.ks.energy import SplatState
    E_d = float(ef_d(model, SplatState(aux=aux))[0])
    E_s = float(jax.jit(lambda mm: ef_s(mm, SplatState(aux=aux, pairs=(pi, pj, v)))[0])(model))
    np.testing.assert_allclose(E_s, E_d, rtol=1e-9, atol=1e-8)

@pytest.mark.parametrize("backend", BACKENDS)
def test_splat_energy_screened_gradient_matches_dense(backend):
    """The screened path's GRADIENT (splat params + C) == the dense path's on the full pair list —
    the variational-training guarantee."""
    model, aux, gp, gw, ef_d, ef_s = _energy_setup(backend)
    pi, pj, v = scr.neighbor_pairs(model.basis, eps=-1.0)
    from gs_dft.ks.energy import SplatState
    g_d = eqx.filter_grad(lambda mm: ef_d(mm, SplatState(aux=aux))[0])(model)
    g_s = eqx.filter_grad(lambda mm: ef_s(mm, SplatState(aux=aux, pairs=(pi, pj, v)))[0])(model)
    for a, b in zip(jax.tree.leaves(eqx.filter(g_d, eqx.is_inexact_array)),
                    jax.tree.leaves(eqx.filter(g_s, eqx.is_inexact_array))):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-7, atol=1e-9)

@pytest.mark.parametrize("backend", BACKENDS)
def test_matvec_differentiable(backend):
    """The screened matvec is differentiable in the splat params (needed for variational training)."""
    s = _splats(backend, M=40)
    C = jr.normal(jr.PRNGKey(9), (s.n_basis, 2))
    pi, pj, _ = scr.neighbor_pairs(s, eps=1e-5)

    def loss(sp):
        return jnp.sum(scr.screened_overlap_matvec(sp, pi, pj, C) ** 2)

    g = jax.grad(loss)(s)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(g))
