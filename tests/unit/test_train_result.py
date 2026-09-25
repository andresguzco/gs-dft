"""``TrainResult``: ``n_steps`` is the number of completed steps and the history arrays are recorded
at the monitor cadence, whether or not a ``step_cb`` is attached.
"""
import jax.numpy as jnp
import jax.random as jr

from gs_dft.ks.energy import SplatKS, SplatModel
from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.ks.train import train, monitor
from dftax.energy.xc import PBE

def _tiny():
    coords = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    charges = jnp.array([1.0, 1.0])
    basis = init_spectral_splats(coords, 4, key=jr.PRNGKey(0), aniso_jitter=0.1)
    model = SplatModel(basis=basis, C=0.3 * jr.normal(jr.PRNGKey(1), (4, 1)),
                       occupations=jnp.array([2.0]))
    gp = jr.uniform(jr.PRNGKey(2), (60, 3), minval=-2.0, maxval=2.0)
    gw = jnp.full((60,), 0.05)
    return SplatKS((coords, charges), PBE(), grid=(gp, gw)), model

def test_n_steps_and_monitor_cadence_history_with_step_cb():
    ks, model = _tiny()
    per_step = []
    res = train(ks, model, steps=5, monitor=monitor(2, verbose=False), refresh_every=0,
                step_cb=per_step.append)
    assert res.n_steps == 5                                # true count, not len(history)
    assert res.step.tolist() == [0, 2, 4]                  # monitor cadence, final step included
    assert len(per_step) == 5                              # step_cb fires every step
    assert res.energy.shape == res.e_hartree.shape == res.grad_norm.shape == (3,)
    assert not res.collapsed
    assert res.state.pairs is None and res.state.aux is None   # dense exact: empty state

def test_monitor_zero_yields_empty_history_but_true_count():
    ks, model = _tiny()
    res = train(ks, model, steps=3, monitor=None, refresh_every=0)
    assert res.n_steps == 3
    assert res.step.size == 0 and res.energy.size == 0
