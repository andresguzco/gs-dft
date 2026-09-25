"""``refresh()`` pre-factors the frozen-auxiliary RI-J metric.

The Cholesky solve gives the plain-solve energy to round-off, on the dense and the screened DF paths.
"""
import jax.numpy as jnp
import jax.random as jr

from gs_dft.ks.energy import SplatKS, SplatModel, SplatState, refresh
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.ks.terms import df, pairlist
from dftax.energy.xc import PBE
from dftax.grid import points

def _sys(M=10, n_occ=3):
    coords = jnp.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.9]])
    charges = jnp.array([6.0, 8.0])
    basis = init_spectral_splats(coords, M, key=jr.PRNGKey(0), aniso_jitter=0.1)
    model = SplatModel(basis=basis, C=0.3 * jr.normal(jr.PRNGKey(1), (M, n_occ)),
                       occupations=2.0 * jnp.ones(n_occ))
    gp = 4.0 * jr.uniform(jr.PRNGKey(2), (300, 3), minval=-1.0, maxval=1.0)
    gw = jnp.full((300,), 0.02)
    return coords, charges, model, gp, gw

def test_dense_df_cholv_matches_solve():
    coords, charges, model, gp, gw = _sys()
    ks = SplatKS((coords, charges), PBE(), grid=(gp, gw), coulomb=df())
    state = refresh(ks, model.basis)
    assert state.cholV is not None and state.aux is not None
    e_chol = float(ks(model, state)[0])
    e_solve = float(ks(model, SplatState(aux=state.aux))[0])   # cholV=None → solve path
    assert abs(e_chol - e_solve) < 1e-10, f"{e_chol!r} vs {e_solve!r}"

def test_screened_df_cholv_matches_solve():
    coords, charges, model, gp, gw = _sys()
    ks = SplatKS((coords, charges), PBE(), grid=points(gp, gw, chunk=128),
                 coulomb=df(), screen=pairlist(eps=-1.0))
    state = refresh(ks, model.basis)
    assert state.cholV is not None and state.pairs is not None
    e_chol = float(ks(model, state)[0])
    e_solve = float(ks(model, SplatState(aux=state.aux, pairs=state.pairs))[0])
    assert abs(e_chol - e_solve) < 1e-10, f"{e_chol!r} vs {e_solve!r}"

def test_exact_state_has_no_cholv():
    coords, charges, model, gp, gw = _sys()
    ks = SplatKS((coords, charges), PBE(), grid=(gp, gw))
    state = refresh(ks, model.basis)
    assert state.cholV is None and state.aux is None and state.aux_K is None
