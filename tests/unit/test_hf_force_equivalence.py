"""The Hellmann-Feynman force equals the gradient of the full energy with respect to the nuclei.

Only ``E_ext(R)`` and ``E_nn(R)`` depend on the nuclear coordinates at fixed splat parameters, so the
force can skip the density-fitting backward, which would otherwise build a dense (M, M) metric.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft import chem, init_model, nao as nao_of
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.forces import _grad_at_coords, _hf_grad_at_coords, splat_forces
from gs_dft.ks.terms import df, pairlist
from gs_dft.ks.train import native_grid
from experiments.common import builders, systems


def _setup(system="water", basis="cc-pvdz", m_mult=1.0, screen=True):
    mol = systems.molecule(system=system, basis=basis)
    M = int(m_mult * int(nao_of(mol)))
    model = init_model(mol, M, jr.PRNGKey(0), init=chem())
    ks = SplatKS(mol, builders.xc_of("pbe"),
                 grid=native_grid(mol, 2, chunk=4096),
                 coulomb=df(lam=1e-8, scales=builders.aux_scales(1)),
                 screen=pairlist(eps=1e-7) if screen else None,
                 mesh=None)
    return ks, model


@pytest.mark.parametrize("screen", [True, False])
def test_two_term_force_matches_full_gradient(screen):
    ks, model = _setup(screen=screen)
    from gs_dft.ks.train import _refresh_state
    state = _refresh_state(ks, model.basis, None, False)   # DF needs the aux either way

    g_full = np.asarray(_grad_at_coords(ks, ks.atom_coords, model, state))
    g_hf = np.asarray(_hf_grad_at_coords(ks, ks.atom_coords, model, state))

    assert np.all(np.isfinite(g_full)) and np.all(np.isfinite(g_hf))
    scale = max(np.abs(g_full).max(), 1e-12)
    rel = np.abs(g_hf - g_full).max() / scale
    assert rel < 1e-9, (
        f"two-term force differs from the full gradient by {rel:.3e} (max |g|={scale:.3e}).\n"
        f"full: {g_full[:2]}\n  hf: {g_hf[:2]}\n"
        "Something other than E_ext/E_nn moves with the nuclei — check whether the grid is "
        "atom-centered, which would be a real Pulay term the force omits either way.")
