"""The committed PubChem geometries have the composition the registry records. No network access."""
import os

import pytest

from experiments.common import pubchem, systems


@pytest.mark.parametrize("name", sorted(pubchem.SYSTEMS))
def test_committed_geometry_matches_registry(name):
    """Composition + atom count + genuinely 3-D, straight off the committed file."""
    info = pubchem.verify(name)
    assert info["formula"] == pubchem.SYSTEMS[name]["formula"]
    assert info["n_atoms"] == pubchem.SYSTEMS[name]["n_atoms"]


@pytest.mark.parametrize("name", sorted(pubchem.SYSTEMS))
def test_registry_name_resolves_through_systems(name):
    """A registered name is reachable as a SYSTEM, not merely as a file on disk — otherwise the
    geometry is committed but no runner can ask for it."""
    path = systems._xyz_path(name)
    assert path is not None and os.path.exists(path), f"{name} is not on the systems.py xyz path"
    text = systems.geometry(name)
    assert len(text.split("\n")[0].split()) == 4          # bare "El x y z", no count/comment header
    assert len([ln for ln in text.splitlines() if ln.split()]) == pubchem.SYSTEMS[name]["n_atoms"]


def test_planar_record_is_rejected():
    """A 2-D record must RAISE, not sail through. Mutating the committed file flat is the only way
    to show the check actually fires — coverage of `verify` alone would not."""
    name = sorted(pubchem.SYSTEMS)[0]
    src = systems._xyz_path(name)
    flat = "\n".join(" ".join(p[:3] + ["0.000000"]) for p in
                     (ln.split() for ln in open(src) if ln.split())) + "\n"
    tmp = src + ".flat"
    try:
        with open(tmp, "w") as fh:
            fh.write(flat)
        with pytest.raises(ValueError, match="PLANAR"):
            pubchem.verify(name, tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def test_formula_is_hill_ordered_without_a_count_of_one():
    """`_formula` must render S once, not 'S1' -- otherwise every registry entry with a lone
    heteroatom mismatches its own asserted string."""
    import collections
    assert pubchem._formula(collections.Counter("C C H H H N O S".split())) == "C2H3NOS"
