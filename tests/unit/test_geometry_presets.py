"""Every geometry preset matches the experimental structure it names, by bond lengths and angles."""
import numpy as np
import pytest

from gs_dft import geometries

# Experimental structures. Bond lengths in Å, angles in degrees. Equilibrium (r_e) values where the
# preset is an r_e structure, which is what a clamped-nuclei solver should be compared against.
# Tolerances are loose enough to admit r_0/r_e and source-to-source disagreement (~0.01 Å, ~0.5°)
# and far tighter than any plausible transcription error.
STRUCTURES = {
    "h2":      dict(bonds={("H", "H"): 0.7414}),
    "n2":      dict(bonds={("N", "N"): 1.0977}),
    "o2":      dict(bonds={("O", "O"): 1.2075}),
    "hf":      dict(bonds={("H", "F"): 0.9168}),
    "co2":     dict(bonds={("C", "O"): 1.1600}, angle=("O", "C", "O", 180.00)),
    "ch4":     dict(bonds={("C", "H"): 1.0870}, angle=("H", "C", "H", 109.47)),
    "h2o":     dict(bonds={("O", "H"): 0.9578}, angle=("H", "O", "H", 104.48)),
    "benzene": dict(bonds={("C", "C"): 1.3970, ("C", "H"): 1.0800}, angle=("C", "C", "C", 120.00)),
    "ethanol": dict(bonds={("C", "C"): 1.5120, ("C", "O"): 1.4310, ("O", "H"): 0.9710,
                           ("C", "H"): 1.0930}, angle=("C", "C", "O", 107.80)),
}
R_TOL, A_TOL = 0.02, 1.0

# Covalent radii (Cordero 2008) — only to decide which pairs are bonded, never a reference value.
_COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Li": 1.28}


def _parse(xyz):
    rows = [ln.split() for ln in xyz.strip().splitlines() if ln.strip()]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def _bonds(sym, xyz):
    """Bonded pairs by the 1.3x-covalent-sum rule, as {(elem, elem): [lengths]}."""
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    out = {}
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            if d[i, j] < 1.3 * (_COV[sym[i]] + _COV[sym[j]]):
                out.setdefault(tuple(sorted((sym[i], sym[j]))), []).append((d[i, j], i, j))
    return out


def _angle(xyz, i, c, j):
    v1, v2 = xyz[i] - xyz[c], xyz[j] - xyz[c]
    cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_preset_matches_the_experimental_structure(name):
    spec = STRUCTURES[name]
    sym, xyz = _parse(getattr(geometries, f"{name}_geometry"))
    found = _bonds(sym, xyz)

    for pair, ref in spec["bonds"].items():
        key = tuple(sorted(pair))
        assert key in found, f"{name}: no {pair[0]}-{pair[1]} bond found at all"
        lengths = [d for d, _, _ in found[key]]
        got = float(np.mean(lengths))
        assert got == pytest.approx(ref, abs=R_TOL), (
            f"{name}: {pair[0]}-{pair[1]} = {got:.4f} Å, experiment {ref:.4f} Å "
            f"(off by {got - ref:+.4f}) — the preset is not the molecule it is named for")

    if "angle" in spec:
        a, c, b, ref = spec["angle"]
        # every (a, c, b) angle at a centre bonded to both, averaged over equivalent centres
        vals = []
        for k, idx in enumerate(sym):
            if idx != c:
                continue
            nbr_a = [j for d, i, j in found.get(tuple(sorted((c, a))), []) if i == k] + \
                    [i for d, i, j in found.get(tuple(sorted((c, a))), []) if j == k]
            nbr_b = [j for d, i, j in found.get(tuple(sorted((c, b))), []) if i == k] + \
                    [i for d, i, j in found.get(tuple(sorted((c, b))), []) if j == k]
            vals += [_angle(xyz, x, k, y) for x in nbr_a for y in nbr_b if x != y]
        assert vals, f"{name}: no {a}-{c}-{b} angle found"
        got = float(np.mean(vals))
        assert got == pytest.approx(ref, abs=A_TOL), (
            f"{name}: {a}-{c}-{b} = {got:.2f}°, experiment {ref:.2f}° (off by {got - ref:+.2f}°) — "
            f"this is the check that {name} failed by 37° for months")


def test_water_is_the_equilibrium_structure_the_dipole_target_assumes():
    """The comparator and the geometry have to be on the same footing.

    `reference_values.py` builds water's target as the measured μ_0 plus a separately cited
    clamped-nuclei correction, i.e. it targets μ_e. A solver at clamped nuclei therefore has to sit
    at the EQUILIBRIUM geometry, not the vibrationally averaged r_0 (0.9572 Å, 104.52°). The two
    differ by less than this assertion's tolerance today; the point is that the preset is pinned to
    r_e on purpose and a future edit toward r_0 should have to argue with this test.
    """
    sym, xyz = _parse(geometries.h2o_geometry)
    assert sym == ["O", "H", "H"]
    r = [np.linalg.norm(xyz[i] - xyz[0]) for i in (1, 2)]
    assert r[0] == pytest.approx(r[1], abs=1e-6), "water is not symmetric"
    assert float(r[0]) == pytest.approx(0.9578, abs=5e-4)
    assert _angle(xyz, 1, 0, 2) == pytest.approx(104.48, abs=0.05)
