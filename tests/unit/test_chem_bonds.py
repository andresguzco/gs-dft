"""Bond detection in ``chem``: the covalent-radius table and the vectorized neighbour search.

Elements outside the table raise rather than getting a default radius, and the cKDTree search
returns exactly the pairs of a reference double loop.
"""
import numpy as np
import pytest

from gs_dft.basis.chem import _cov_radius_bohr, detect_bonds, _ANG2BOHR


def _loop_reference(coords_bohr, atom_numbers, scale=1.2):
    """The pre-B2 O(n²) double loop, verbatim, as the equivalence oracle."""
    n = coords_bohr.shape[0]
    radii = np.array([_cov_radius_bohr(int(z)) for z in atom_numbers])
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            if d < scale * (radii[i] + radii[j]):
                bonds.append((i, j))
    return bonds


# --- covalent radii -----------------------------------------------------------

@pytest.mark.parametrize("Z,ang", [(1, 0.31), (6, 0.76), (7, 0.71), (8, 0.66), (16, 1.05)])
def test_light_element_radii(Z, ang):
    """Cordero covalent radii for the first rows."""
    assert _cov_radius_bohr(Z) == pytest.approx(ang * _ANG2BOHR, rel=1e-12)


def test_metal_radii():
    """Cordero: Zn 1.22 Å, Fe 1.32 Å."""
    assert _cov_radius_bohr(30) == pytest.approx(1.22 * _ANG2BOHR, rel=1e-12)
    assert _cov_radius_bohr(26) == pytest.approx(1.32 * _ANG2BOHR, rel=1e-12)


def test_untabulated_element_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="covalent radius"):
        _cov_radius_bohr(110)


def _zn_aqua_cluster():
    """Zn(H2O)4-like: Zn at the origin, four O at 2.0 Å, two H on each O."""
    pos = [(30, (0.0, 0.0, 0.0))]
    for d in [(2.0, 0, 0), (-2.0, 0, 0), (0, 2.0, 0), (0, -2.0, 0)]:
        pos.append((8, d))
        u = np.array(d, float) / np.linalg.norm(d)
        for s in (0.6, -0.6):
            pos.append((1, tuple(np.array(d, float) + u * 0.7 + np.array([0, 0, s]))))
    Z = np.array([z for z, _ in pos])
    xyz = np.array([p for _, p in pos], float) * _ANG2BOHR
    return xyz, Z


def test_metal_bonds_are_detected():
    """All four Zn–O bonds of a Zn aqua cluster are found, so bond-centered splats are placed."""
    xyz, Z = _zn_aqua_cluster()
    zn_bonds = [(i, j) for i, j in detect_bonds(xyz, Z) if 0 in (i, j)]
    assert len(zn_bonds) == 4, "the four Zn–O bonds must be detected"


# --- B2 ---------------------------------------------------------------------

def _random_cluster(n, seed, elements=(1, 6, 7, 8)):
    rng = np.random.default_rng(seed)
    # ~2.6 Bohr nearest-neighbour spacing: dense enough that many pairs bond, not all
    coords = rng.normal(scale=2.6 * n ** (1 / 3), size=(n, 3))
    return coords, rng.choice(elements, size=n)


@pytest.mark.parametrize("n,seed", [(2, 0), (12, 1), (60, 2), (200, 3)])
def test_vectorized_matches_the_loop_exactly(n, seed):
    coords, Z = _random_cluster(n, seed)
    assert detect_bonds(coords, Z) == _loop_reference(coords, Z)


def test_matches_the_loop_with_a_metal_present():
    """Zn has the largest radius here, so it sets the global ball-query cutoff — the case where a
    too-small cutoff would silently drop real bonds."""
    coords, Z = _random_cluster(80, 7)
    Z[:4] = 30
    assert detect_bonds(coords, Z) == _loop_reference(coords, Z)


def test_output_is_lexsorted_and_upper_triangular():
    coords, Z = _random_cluster(50, 4)
    b = detect_bonds(coords, Z)
    assert all(i < j for i, j in b)
    assert b == sorted(b)


def test_degenerate_inputs():
    assert detect_bonds(np.zeros((0, 3)), np.array([], int)) == []
    assert detect_bonds(np.zeros((1, 3)), np.array([6])) == []


def test_far_apart_atoms_do_not_bond():
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    assert detect_bonds(coords, np.array([6, 6])) == []
