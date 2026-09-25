"""One system loader for every experiment: geometry, charge and, for the diatomics, the R grid.

A system is a name, and the registry says where its geometry comes from and what its charge is::

    from experiments.common.systems import molecule, geometry, R_GRID
    mol = molecule(cfg)                       # cfg.system + cfg.basis (+ cfg.get("r") for diatomics)
    mol = molecule(cfg, r=3.0)                # LiH or LiF at an explicit bond length
    atom = geometry("f_anion")                # the raw atom string
"""
import os

from dftax.system import Molecule
from gs_dft import geometries

_HERE = os.path.dirname(os.path.abspath(__file__))
# Directories searched for a bare `<name>.xyz` (element x y z per line, angstrom): the fetched small
# molecules and FMODB proteins in `geometry/`, and the peptides.
_XYZ_DIRS = [
    os.path.join(_HERE, "geometry"),                       # experiments/common/pubchem.py writes here
    os.path.join(_HERE, "peptides"),
    os.path.join(_HERE, "..", "exp5_basis_accuracy", "peptides"),
]

_ALIASES = {"water": "h2o"}                       # geometries names water's preset `h2o_geometry`

# Diatomic ionic-dissociation stress tests: name -> (atom-string builder of R in Å, R-scan grid).
# R_e(LiH) ≈ 1.595 Å, R_e(LiF) ≈ 1.564 Å. The grids are sampled at IDENTICAL geometries by the GTO
# and splat curves, so they live here, once, not duplicated in a curve runner and its plotter.
_DIATOMICS = {
    "lih": (lambda r: f"Li 0 0 0; H 0 0 {float(r):.6f}",
            [1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]),
    "lif": (lambda r: f"Li 0 0 0; F 0 0 {float(r):.6f}",
            [1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0]),
    # GW100 case 14452-59-6 (HCP92 experimental), R_e = 2.6729 A. Its HF HOMO is -4.95 eV, the
    # most loosely bound valence of any tractable GW100 species and six times shallower than
    # water's -13.87: an alkali dimer is where an unaugmented Gaussian set is worst and where a
    # basis that can migrate off the nuclei should show it. The default r IS the GW100 geometry,
    # so `system=li2` with no r reproduces the benchmark case exactly.
    "li2": (lambda r=2.6729: f"Li 0 0 0; Li 0 0 {float(r):.6f}",
            [2.0, 2.4, 2.6729, 3.0, 3.6, 4.5, 6.0]),
}

# Bare closed-shell anions (charge −1). The charge is a fact about the system, not a runner flag.
_ANIONS = {
    "f_anion": ("F 0 0 0", -1),
    "oh_anion": ("O 0 0 0; H 0 0 0.964", -1),
}

# FMODB proteins: readable alias -> the FMODB id, which stays the canonical system name so a RESULT
# line carries its provenance. The aliases are `fmodb_`-prefixed because `insulin` is already
# `insulin.xyz`, a different structure (784 atoms, C256H381N65O76S6, against LG5K9's 947 atoms,
# C306H463N84O88S6).
_FMODB_ALIASES = {
    "fmodb_trp_cage": "PJM49",
    "fmodb_defensin": "9LQN2",
    "fmodb_insulin": "LG5K9",
    "fmodb_spectrin_ch": "38KNL",
    "fmodb_3ifu_a": "25J8R",
}

R_GRID = {name: grid for name, (_fn, grid) in _DIATOMICS.items()}


def _fmodb_id(name):
    """The FMODB id for ``name`` (itself, or its alias), else None. Import is lazy — `fmodb` is only
    needed for these five systems and pulls in urllib/zipfile."""
    from experiments.common import fmodb
    key = _FMODB_ALIASES.get(name, name)
    return key if key in fmodb.SYSTEMS else None


def _xyz_path(name):
    for cand in (name, _FMODB_ALIASES.get(name)):
        if cand is None:
            continue
        for d in _XYZ_DIRS:
            p = os.path.join(d, f"{cand}.xyz")
            if os.path.exists(p):
                return p
    return None


def charge(name):
    """Net charge of a system (0 unless the registry says otherwise)."""
    fid = _fmodb_id(name)
    if fid is not None:
        from experiments.common import fmodb
        q = fmodb.SYSTEMS[fid].get("charge")
        if q is None:
            raise ValueError(f"FMODB {fid}: no charge in SYSTEMS. It is published in the LOG "
                             f"(`CHARGE OF MOLECULE`); re-run `python -m experiments.common.fmodb "
                             f"{fid}` and record it. Refusing to guess 0 — every entry is a cation.")
        return int(q)
    return _ANIONS.get(name, (None, 0))[1]


def geometry(name, r=None):
    """The atom string (angstrom) for ``name``. ``r`` is required for the diatomic R-scans."""
    if name in _DIATOMICS:
        fn = _DIATOMICS[name][0]
        if r is None:
            # A builder MAY carry its own reference geometry (li2 = the GW100 bond length); one
            # that does not is a pure R-scan and forgetting r would silently pick an arbitrary
            # bond length, so that still raises.
            import inspect
            default = inspect.signature(fn).parameters["r"].default
            if default is inspect.Parameter.empty:
                raise ValueError(f"system {name!r} is an R-scan diatomic; pass r=<bond length Å>")
            return fn()
        return fn(r)
    if name in _ANIONS:
        return _ANIONS[name][0]
    p = _xyz_path(name)
    if p is not None:
        with open(p) as fh:
            return fh.read()
    fid = _fmodb_id(name)
    if fid is not None:                                # known protein, not fetched yet
        raise FileNotFoundError(
            f"FMODB {fid} is a registered system but {fid}.xyz is not in {_XYZ_DIRS[0]}. "
            f"Fetch it once with: uv run python -m experiments.common.fmodb --materialize {fid}")
    return getattr(geometries, _ALIASES.get(name, name) + "_geometry")


def molecule(cfg=None, *, system=None, basis=None, r=None):
    """Build the dftax ``Molecule`` for a run.

    Reads ``cfg.system``/``cfg.basis`` when a Hydra ``cfg`` is passed (and ``cfg.r`` if present for a
    diatomic), or takes explicit ``system=``/``basis=``. The charge comes from the registry.
    """
    if cfg is not None:
        system = cfg.system if system is None else system
        basis = cfg.basis if basis is None else basis
        if r is None:
            r = cfg.get("r", None) if hasattr(cfg, "get") else getattr(cfg, "r", None)
    if system is None or basis is None:
        raise ValueError("molecule() needs system and basis (via cfg or keywords)")
    return Molecule.from_xyz(geometry(system, r=r), basis, unit="angstrom",
                             charge=charge(system), spherical=True)
