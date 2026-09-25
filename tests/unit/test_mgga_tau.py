"""τ on the splat grid, pinned by an identity rather than by a golden number.

    for a SINGLE doubly-occupied orbital,  τ = |∇ρ|² / (8ρ)  pointwise

(ρ = 2ψ², ∇ρ = 4ψ∇ψ ⇒ |∇ρ|²/(8ρ) = |∇ψ|², and τ = ½·2·|∇ψ|² = |∇ψ|².) H₂ has exactly one occupied
orbital, so the identity holds at every grid point and a factor-of-two error shows up as ~5e-1."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dftax.grid import becke_grid

from gs_dft import init_model
from gs_dft.integrals import dense as _f
from gs_dft.integrals.dense import eval_density_grad_tau_on_grid
from gs_dft.ks.orthonormalize import lowdin_orthonormalize
from gs_dft.screening.grid import chunked_density_grad_tau
from experiments.common import systems


@pytest.fixture(scope="module")
def h2_state():
    mol = systems.molecule(system="h2", basis="cc-pvdz")
    model = init_model(mol, 8, jax.random.PRNGKey(0))
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()),
                        n_radial=50, lebedev=194)
    S = _f.one_electron_integrals(model.basis, jnp.array(mol.atom_coords()),
                                  jnp.array(mol.atom_charges(), float))[0]
    return model, lowdin_orthonormalize(model.C, S), gp, gw


def test_tau_equals_von_weizsacker_for_one_orbital(h2_state):
    model, C, gp, _ = h2_state
    assert model.occupations.shape[0] == 1, "the identity needs exactly one occupied orbital"
    rho, grad, tau = eval_density_grad_tau_on_grid(model.basis, C, model.occupations, gp)
    keep = np.asarray(rho) > 1e-8
    vw = np.asarray(jnp.sum(grad ** 2, axis=-1) / (8.0 * rho))[keep]
    got = np.asarray(tau)[keep]
    assert np.abs(got - vw).max() / np.abs(vw).max() < 1e-12


def test_chunked_tau_matches_dense(h2_state):
    """The streamed path is a memory optimisation, so it must agree to roundoff, not to tolerance."""
    model, C, gp, _ = h2_state
    _, _, tau_d = eval_density_grad_tau_on_grid(model.basis, C, model.occupations, gp)
    _, _, tau_c = chunked_density_grad_tau(model.basis, C, model.occupations, gp, chunk=128)
    assert np.allclose(np.asarray(tau_d), np.asarray(tau_c), rtol=1e-12, atol=1e-14)


def test_meta_gga_reaches_the_tau_path():
    """`_density_on_grid` branched on `== "GGA"`, which sent a meta-GGA down the LDA path with no
    ∇ρ and no τ. Guard that it now returns all three."""
    from gs_dft.ks.energy import SplatKS
    from gs_dft.ks.train import native_grid
    from experiments.common import builders
    mol = systems.molecule(system="h2", basis="cc-pvdz")
    model = init_model(mol, 8, jax.random.PRNGKey(0))
    ks = SplatKS(mol, builders.xc_of("r2scan"), grid=native_grid(mol, 1))
    S = _f.one_electron_integrals(model.basis, jnp.array(mol.atom_coords()),
                                  jnp.array(mol.atom_charges(), float))[0]
    rho, grad, tau = ks._density_on_grid(model.basis, lowdin_orthonormalize(model.C, S),
                                         model.occupations)
    assert grad is not None and tau is not None
    assert rho.shape == tau.shape


def test_long_range_exchange_limits():
    """E_K[erf(ωr)/r] must rise monotonically to E_K[1/r] as ω grows, and vanish as ω → 0."""
    from gs_dft.coulomb.ri import df_exchange_energy
    from gs_dft.ks.train import refresh
    from experiments.common import builders
    mol = systems.molecule(system="h2o", basis="cc-pvdz")
    model = init_model(mol, 24, jax.random.PRNGKey(0))
    ks = builders.splat_ks(mol, builders.xc_of("pbe0"), grid_level=1, screened=False)
    state = refresh(ks, model.basis)
    S = _f.one_electron_integrals(model.basis, jnp.array(mol.atom_coords()),
                                  jnp.array(mol.atom_charges(), float))[0]
    C = lowdin_orthonormalize(model.C, S)

    full = float(df_exchange_energy(model.basis, C, state.aux_K))
    vals = [float(df_exchange_energy(model.basis, C, state.aux_K, omega=w))
            for w in (0.1, 0.3, 1.0, 5.0)]
    assert all(v < 0 for v in vals), "exchange is negative"
    # |E_K| grows with omega and never exceeds the full-kernel value
    mags = [abs(v) for v in vals]
    assert mags == sorted(mags), f"not monotone in omega: {mags}"
    assert mags[-1] < abs(full), "attenuated exchange exceeded the full kernel"
    assert mags[0] / abs(full) < 0.1, "omega=0.1 should keep only a small fraction"


def test_two_centre_attenuation_matches_closed_form():
    """(1|erf(ωr)/r|2) / (1|1/r|2) == √z·F₀(zT)/F₀(T) — the metric has no quadrature to be wrong."""
    from scipy.special import erf as _erf
    from gs_dft.coulomb.ri import coulomb_block

    def f0(x):
        return 1.0 if x < 1e-14 else 0.5 * np.sqrt(np.pi / x) * _erf(np.sqrt(x))

    for g1, g2, d, w in [(0.8, 1.3, 1.7, 0.3), (2.5, 0.4, 0.9, 0.3), (5.0, 3.0, 0.5, 0.6)]:
        P1, P2 = jnp.array([[0.0, 0.0, 0.0]]), jnp.array([[d, 0.0, 0.0]])
        one = jnp.array([1.0])
        sr = float(coulomb_block(jnp.array([g1]), P1, one, jnp.array([g2]), P2, one)[0, 0])
        lr = float(coulomb_block(jnp.array([g1]), P1, one, jnp.array([g2]), P2, one, omega=w)[0, 0])
        rho = g1 * g2 / (g1 + g2)
        z = w ** 2 / (w ** 2 + rho)
        assert abs(lr / sr - np.sqrt(z) * f0(z * rho * d * d) / f0(rho * d * d)) < 1e-10
