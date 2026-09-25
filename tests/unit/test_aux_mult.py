"""``aux_mult`` sizes the frozen pair-product auxiliary basis.

Changing it changes ``naux`` and the training objective, and leaves the reported exact-Coulomb
energy unchanged.
"""
import jax.random as jr
import pytest

from gs_dft import init_model
from gs_dft.ks.energy import refresh
from gs_dft.ks.train import evaluate
from experiments.common import builders, systems

M = 24


@pytest.fixture(scope="module")
def mol_model():
    mol = systems.molecule(system="h2o", basis="sto-3g")
    return mol, init_model(mol, M, jr.PRNGKey(0))


def _ks(mol, mult):
    return builders.splat_ks(mol, builders.xc_of("pbe"), grid_level=1, screened=True,
                             aux_mult=mult)


@pytest.mark.parametrize("mult", sorted(builders.AUX_LADDER))
def test_naux_is_mult_times_M(mol_model, mult):
    mol, model = mol_model
    aux = refresh(_ks(mol, mult), model.basis).aux
    assert int(aux.centers.shape[0]) == mult * M


def test_off_ladder_raises_rather_than_falling_back(mol_model):
    mol, _ = mol_model
    with pytest.raises(ValueError, match="aux_mult"):
        _ks(mol, 4)


def test_bigger_aux_moves_the_df_objective_toward_exact(mol_model):
    """The DF energy is the TRAINING objective and must improve monotonically with naux; the
    REPORTED energy comes from the exact Coulomb and must not move at all. Both halves matter — if
    evaluate() drifted with the aux, the headline numbers would depend on a fitting-set size."""
    mol, model = mol_model
    e_exact = float(evaluate(_ks(mol, 1), model))
    gaps, reported = [], []
    for mult in sorted(builders.AUX_LADDER):
        ks = _ks(mol, mult)
        st = refresh(ks, model.basis)
        gaps.append(abs(float(ks(model, st)[0]) - e_exact))
        reported.append(float(evaluate(ks, model)))
    assert gaps == sorted(gaps, reverse=True), f"DF error not monotone in naux: {gaps}"
    assert max(reported) - min(reported) < 1e-9, (
        f"the reported (exact-Coulomb) energy moved with the aux size: {reported}")
