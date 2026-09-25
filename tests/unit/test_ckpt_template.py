"""A checkpoint written by ``train_ckpt`` loads back into the template ``train_ckpt.fresh_init`` builds.

``observables.py`` and ``polish_forces.py`` rebuild their template with
``fresh_init``, so this round trip covers every reader.
"""
import equinox as eqx
import pytest

from experiments.common import systems
from experiments.exp2_observables.train_ckpt import M_BY, fresh_init


@pytest.mark.parametrize("system", ["water", "ethanol"])
def test_fresh_init_roundtrips(system, tmp_path):
    mol = systems.molecule(system=system, basis="cc-pvdz")
    template = fresh_init(system, mol)                    # (model, aux) — what train_ckpt serializes
    path = str(tmp_path / f"ckpt_{system}.eqx")
    eqx.tree_serialise_leaves(path, template)
    model, aux = eqx.tree_deserialise_leaves(path, fresh_init(system, mol))  # observables' template
    assert model.C.shape[0] == M_BY[system]              # M splats, as serialized
    assert model.occupations.shape[0] == mol.nelectron // 2
