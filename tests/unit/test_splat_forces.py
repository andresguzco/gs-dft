"""Nuclear forces F = -dE/dR for the splat energy (``gs_dft/ks/forces.py``).

The autodiff force equals a central finite difference of the energy at fixed splat parameters,
which holds for any splat state, and the nuclear repulsion term contributes. Translational
invariance needs a converged density and is checked by
``experiments/shared/splat_forces_validation.py``.
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from dftax.system import Molecule
from dftax.grid import becke_grid

from gs_dft.basis.spectral import init_spectral_splats
from gs_dft.ks.energy import SplatModel, SplatKS
from gs_dft.ks.forces import splat_forces, splat_energy_and_forces
from dftax.integrals.nuclear_repulsion import nuclear_repulsion
from dftax.energy.xc import PBE

pytestmark = pytest.mark.slow                       # finite-difference force loop = many energy evals


def _small_system(seed=0):
    """H2 with a handful of diagonal splats and a coarse grid — cheap, exercises V_ne + E_nn."""
    mol = Molecule.from_xyz("H 0 0 0; H 0 0 1.4", "sto-3g", unit="bohr", spherical=True)
    coords = jnp.array(mol.atom_coords())
    charges = jnp.array(mol.atom_charges(), dtype=jnp.float64)
    splats = init_spectral_splats(coords, 6, key=jr.PRNGKey(42))
    C = 0.1 * jr.normal(jr.PRNGKey(seed), (splats.n_basis, 1))
    model = SplatModel(basis=splats, C=C, occupations=jnp.array([2.0]))
    gp, gw = becke_grid(list(mol.symbols), jnp.asarray(mol.atom_coords()), n_radial=35, lebedev=110)
    return coords, charges, model, gp, gw

def _fd_forces(ks, model, eps=1e-3):
    """4th-order central finite-difference of E_total w.r.t. atom_coords, splats held fixed.

    The 5-point stencil [+1,−8,+8,−1]/12h is O(h⁴) accurate, so even on the large (~30 Ha/Bohr)
    forces of an un-converged random θ the truncation error is well below the 1e-6 gate.
    """
    coords = ks.atom_coords

    def E_at(c):
        return float(eqx.tree_at(lambda k: k.atom_coords, ks, c)(model)[0])

    F = jnp.zeros_like(coords)
    for a in range(coords.shape[0]):
        for i in range(3):
            e2 = E_at(coords.at[a, i].add(2 * eps))
            e1 = E_at(coords.at[a, i].add(eps))
            em1 = E_at(coords.at[a, i].add(-eps))
            em2 = E_at(coords.at[a, i].add(-2 * eps))
            dE = (-e2 + 8 * e1 - 8 * em1 + em2) / (12 * eps)
            F = F.at[a, i].set(-dE)
    return F

def test_forces_match_finite_difference():
    """Autodiff HF force == central FD of the total energy at fixed splats (to ~1e-6)."""
    coords, charges, model, gp, gw = _small_system()
    ks = SplatKS((coords, charges), PBE(), grid=(gp, gw))          # analytic E_nn
    F = splat_forces(ks, model)
    F_fd = _fd_forces(ks, model)
    assert jnp.all(jnp.isfinite(F))
    assert jnp.max(jnp.abs(F - F_fd)) < 1e-6, f"max |F - F_fd| = {jnp.max(jnp.abs(F - F_fd)):.2e}"

def test_enn_term_is_present():
    """The total force minus the electronic-only force equals exactly −∂E_nn/∂R (E_nn is
    always analytic on SplatKS, so its force term is always present)."""
    coords, charges, model, gp, gw = _small_system()
    ks = SplatKS((coords, charges), PBE(), grid=(gp, gw))
    F_total = splat_forces(ks, model)

    def electronic(c):
        return eqx.tree_at(lambda k: k.atom_coords, ks, c)(model)[0] - nuclear_repulsion(c, charges)

    F_elec = -jax.grad(electronic)(coords)
    F_enn = -jax.grad(lambda c: nuclear_repulsion(c, charges))(coords)
    assert jnp.max(jnp.abs((F_total - F_elec) - F_enn)) < 1e-9
    # E_nn force on H2 is repulsive along the bond ⇒ nonzero
    assert jnp.max(jnp.abs(F_enn)) > 1e-3

def test_energy_and_forces_consistent():
    """splat_energy_and_forces returns the same E and F as the separate calls."""
    coords, charges, model, gp, gw = _small_system()
    ks = SplatKS((coords, charges), PBE(), grid=(gp, gw))
    E, F = splat_energy_and_forces(ks, model)
    E_ref = float(ks(model)[0])
    F_ref = splat_forces(ks, model)
    assert abs(E - E_ref) < 1e-10
    assert jnp.max(jnp.abs(F - F_ref)) < 1e-12
