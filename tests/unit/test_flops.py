"""``flops()`` returns a positive per-step FLOP count."""
import jax.numpy as jnp
import jax.random as jr

from gs_dft import (SplatKS, SplatModel, df, pairlist, flops,
                             init_spectral_splats)
from dftax.energy.xc import PBE
from dftax.grid import points

def test_flops_positive_on_screened_df_step():
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    charges = jnp.array([1.0, 1.0])
    basis = init_spectral_splats(coords, 6, key=jr.PRNGKey(0), aniso_jitter=0.1)
    model = SplatModel(basis=basis, C=0.2 * jr.normal(jr.PRNGKey(1), (6, 1)),
                       occupations=jnp.array([2.0]))
    gp = jr.uniform(jr.PRNGKey(2), (100, 3), minval=-2.0, maxval=2.0)
    gw = jnp.full((100,), 0.01)
    ks = SplatKS((coords, charges), PBE(), grid=points(gp, gw, chunk=64),
                 coulomb=df(), screen=pairlist(eps=1e-6))
    f = flops(ks, model, steps=10)
    assert f > 0.0
