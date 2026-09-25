"""Data-free, chemistry-informed initialization of the splat basis (exp8_data_free_init).

Builds a chart-agnostic **site list** from molecular geometry + tabulated *element* constants
(covalent radii, reference-basis exponents, electron counts — **no per-molecule reference
calculation**), then converts it to any splat chart. Four independent heuristics, each a flag:

  A1  ``use_ref_exponents`` : per-element exponent range from a reference basis (def2-SVP) — fixes
        the core-cusp gap (carbon to ~10³, not the global cap of 90) and makes the spread
        element-aware.
  B   ``bond_fraction > 0``  : bond-centered splats (covalent-radius detection, valence scale from
        bond length) so the optimizer doesn't have to migrate density into bonds.
  C   ``bond_anisotropy``    : bond-aligned anisotropic covariance (full/spectral only). Also cures
        the spectral chart's orientation singularity by starting already-oriented.
  D   ``alloc='electron'``   : atom budget ∝ Z (floor 1) instead of uniform.

The site list is ``(centers (M,3), precision_eigs (M,3), quat (M,4))`` — the eigenvalues of the
precision A and the orientation quaternion (identity for isotropic sites). Chart converters turn it
into the full-covariance ``Splat`` chart (the sole splat parametrization).

See ``experiments/exp8_data_free_init/README.md``.
"""

from dataclasses import asdict, dataclass

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from gs_dft.basis.spectral import Splat

__all__ = [
    "chem", "ChemInit",
    "build_init_sites", "init_chem_splats",
    "sites_to_spectral",
    "detect_bonds",
]

_ANG2BOHR = 1.8897259886

_COV_RADIUS_CACHE: dict = {}

# Global default range = the legacy init [exp(-1), exp(4.5)] = [0.37, 90] (element-blind:
# used when ``use_ref_exponents=False``, or if BSE has no entry for an exotic element).
_GLOBAL_EXP_RANGE = (float(np.exp(-1.0)), float(np.exp(4.5)))

_REF_EXP_CACHE: dict = {}


def _ref_exponent_range(Z: int, basis_ref: str) -> tuple[float, float]:
    """(α_min, α_max) of element Z's primitive exponents in ``basis_ref`` (default def2-SVP).

    Fetched live from the **Basis Set Exchange** (``bse.get_basis`` — the same source dftax's
    own basis loader uses), so any element/basis works and NO exponent data is hardcoded. Falls
    back to the element-blind global range only if BSE has no entry for the element.
    """
    key = (Z, basis_ref)
    if key in _REF_EXP_CACHE:
        return _REF_EXP_CACHE[key]
    try:
        import basis_set_exchange as bse
        shells = (bse.get_basis(basis_ref, elements=[int(Z)], header=False)
                  ["elements"][str(int(Z))]["electron_shells"])
        exps = [float(e) for s in shells for e in s["exponents"]]
        rng = (min(exps), max(exps)) if exps else _GLOBAL_EXP_RANGE
    except Exception:
        rng = _GLOBAL_EXP_RANGE
    _REF_EXP_CACHE[key] = rng
    return rng


def _cov_radius_bohr(Z: int) -> float:
    """Covalent radius (Bohr), Cordero et al. (2008), read from ``periodictable`` (Z = 1–96).

    Raises for an element with no tabulated radius rather than substituting a default, which would
    mis-place the bond-centered splats without failing.
    """
    Z = int(Z)
    if Z in _COV_RADIUS_CACHE:
        return _COV_RADIUS_CACHE[Z]
    import periodictable as pt
    try:
        r = getattr(pt.elements[Z], "covalent_radius", None)
    except KeyError:
        r = None
    if r is None or r != r:                       # absent, or NaN for the trans-actinides
        raise ValueError(
            f"no tabulated covalent radius for Z={Z}; chem() cannot place bond-centered splats "
            f"for it. Cordero (2008) covers Z=1..96.")
    _COV_RADIUS_CACHE[Z] = float(r) * _ANG2BOHR
    return _COV_RADIUS_CACHE[Z]


def detect_bonds(coords_bohr: np.ndarray, atom_numbers: np.ndarray,
                 scale: float = 1.2) -> list[tuple[int, int]]:
    """Covalent-radius bond list: (i, j), i<j, with d_ij < scale·(r_cov_i + r_cov_j).

    A cKDTree ball query over the ONE global cutoff ``scale·2·max(r_cov)``, then the exact per-pair
    radius-sum test on the survivors — same predicate, same lexsorted output, in O(n log n + k).
    """
    coords = np.asarray(coords_bohr, float)
    n = coords.shape[0]
    if n < 2:
        return []
    radii = np.array([_cov_radius_bohr(int(z)) for z in atom_numbers], float)

    from scipy.spatial import cKDTree
    # One conservative global cutoff: no true bond can exceed it, so the ball query cannot miss a
    # pair, and the exact test below removes the extras it does return.
    cand = cKDTree(coords).query_pairs(scale * 2.0 * float(radii.max()), output_type="ndarray")
    if cand.size == 0:
        return []
    i, j = cand[:, 0], cand[:, 1]
    d = np.linalg.norm(coords[i] - coords[j], axis=1)
    keep = d < scale * (radii[i] + radii[j])
    i, j = i[keep], j[keep]
    order = np.lexsort((j, i))                    # (i, j) ascending
    return [(int(a), int(b)) for a, b in zip(i[order], j[order])]


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer counts summing EXACTLY to ``total``, apportioned ∝ weights (largest-remainder)."""
    weights = np.asarray(weights, float)
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(len(weights), int)
    exact = weights / weights.sum() * total
    floor = np.floor(exact).astype(int)
    rem = int(total - floor.sum())
    if rem > 0:
        for idx in np.argsort(-(exact - floor))[:rem]:
            floor[idx] += 1
    return floor


def _allocate_floor1(weights: np.ndarray, total: int) -> np.ndarray:
    """Apportion ∝ weights but give every unit ≥1 when ``total`` allows (else largest-remainder)."""
    n = len(weights)
    if total >= n:
        return 1 + _largest_remainder(weights, total - n)
    return _largest_remainder(weights, total)


def _atom_exponents(Z: int, n: int, use_ref_exponents: bool, basis_ref: str) -> np.ndarray:
    """``n`` geometrically-spaced exponents over the element's [α_min, α_max] (or the global range)."""
    amin, amax = _ref_exponent_range(int(Z), basis_ref) if use_ref_exponents else _GLOBAL_EXP_RANGE
    if n == 1:
        return np.array([np.sqrt(amin * amax)])           # geometric mean for a single function
    return np.exp(np.linspace(np.log(amin), np.log(amax), n))


def _quat_from_z_to(u: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) rotating the local +z axis onto unit vector ``u``.

    So ``R(quat)[:, 2] = u`` — the splat's 3rd precision-eigenaxis points along the bond.
    """
    u = u / (np.linalg.norm(u) + 1e-12)
    c = float(u[2])                                        # z · u
    if c > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])             # already aligned
    if c < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])             # antiparallel ⇒ 180° about x
    axis = np.array([-u[1], u[0], 0.0])                   # z × u
    axis /= np.linalg.norm(axis)
    half = 0.5 * np.arccos(c)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


_IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def build_init_sites(atom_coords, atom_numbers, M, *, key,
                     use_ref_exponents: bool = True, bond_fraction: float = 0.25,
                     bond_anisotropy: bool = True, alloc: str = "electron",
                     basis_ref: str = "def2-svp", bond_scale: float = 1.2,
                     bond_K: float = 5.0, bond_aspect: float = 2.5, bond_spread: float = 2.0,
                     jitter_scale: float = 0.1, aniso_jitter: float = 0.02):
    """Chart-agnostic site list for ``M`` splats from geometry + element constants.

    Returns ``(centers (M,3), precision_eigs (M,3), quat (M,4))`` as jnp arrays. The four heuristics
    are the flags: ``use_ref_exponents`` (A1), ``bond_fraction>0`` (B), ``bond_anisotropy`` (C),
    ``alloc`` (D). Atom sites are isotropic (+ tiny ``aniso_jitter`` so the spectral chart isn't
    exactly degenerate); bond sites carry the bond-aligned anisotropy when C is on.
    """
    coords = np.asarray(atom_coords, dtype=float)
    Z = np.asarray(atom_numbers).astype(int)
    n_atoms = coords.shape[0]

    bonds = detect_bonds(coords, Z, scale=bond_scale) if bond_fraction > 0 else []
    n_bonds = len(bonds)

    # --- D: budget split (atoms vs bonds), keeping ≥1 splat per atom when M allows ---
    n_bond_splats = min(int(round(bond_fraction * M)), max(0, M - n_atoms)) if n_bonds else 0
    n_atom_splats = M - n_bond_splats
    atom_w = Z.astype(float) if alloc == "electron" else np.ones(n_atoms)
    atom_counts = _allocate_floor1(atom_w, n_atom_splats)
    bond_counts = _allocate_floor1(np.ones(n_bonds), n_bond_splats) if n_bonds else np.zeros(0, int)

    centers, eigs, quats = [], [], []

    # --- atom-centered sites (A1 exponents, isotropic) ---
    for a in range(n_atoms):
        n_a = int(atom_counts[a])
        if n_a == 0:
            continue
        for alpha in _atom_exponents(Z[a], n_a, use_ref_exponents, basis_ref):
            centers.append(coords[a])
            eigs.append([alpha, alpha, alpha])
            quats.append(_IDENT_QUAT)

    # --- B: bond-centered sites (valence scale from bond length; C: bond-aligned anisotropy) ---
    for b, (i, j) in enumerate(bonds):
        k = int(bond_counts[b])
        if k == 0:
            continue
        mid = 0.5 * (coords[i] + coords[j])
        d = float(np.linalg.norm(coords[i] - coords[j]))
        a_bond = bond_K / (d * d)                          # valence-scale precision
        if k == 1:
            a_list = [a_bond]
        else:
            a_list = a_bond * bond_spread ** np.linspace(-1.0, 1.0, k)
        if bond_anisotropy:
            u = (coords[j] - coords[i])
            q = _quat_from_z_to(u)                         # local z → bond axis
        else:
            q = _IDENT_QUAT
        for a_b in a_list:
            centers.append(mid)
            if bond_anisotropy:
                # elongate ALONG the bond (3rd eigenaxis = z = bond): lower precision there
                eigs.append([a_b, a_b, a_b / bond_aspect])
            else:
                eigs.append([a_b, a_b, a_b])
            quats.append(q)

    centers = np.asarray(centers, dtype=float)
    eigs = np.asarray(eigs, dtype=float)
    quats = np.asarray(quats, dtype=float)
    assert centers.shape[0] == M, f"site builder produced {centers.shape[0]} != M={M}"

    # symmetry-breaking jitter: centers + a tiny anisotropic kick to the (isotropic) precisions
    kc, ka = jr.split(key)
    centers = jnp.asarray(centers) + jitter_scale * jr.normal(kc, (M, 3))
    log_eigs = jnp.log(jnp.asarray(eigs))
    if aniso_jitter > 0:
        log_eigs = log_eigs + aniso_jitter * jr.normal(ka, (M, 3))
    return centers, jnp.exp(log_eigs), jnp.asarray(quats)


# --------------------------------------------------------------------------------------------------
#  Chart converters: site list (centers, precision_eigs, quat) → a concrete splat module.
#  precision_eigs are the eigenvalues of A; quat rotates the local axes onto A's eigenframe.
# --------------------------------------------------------------------------------------------------
def sites_to_spectral(centers, precision_eigs, quat) -> Splat:
    """Native chart: ℓ = log(eigs), orientation = quat (A = R(q) diag(eigs) R(q)ᵀ)."""
    return Splat(quat=quat, log_scale=jnp.log(precision_eigs), centers=centers)


def init_chem_splats(atom_coords, atom_numbers, M, *, key, **kwargs):
    """Convenience: build the site list and convert to the spectral chart."""
    sites = build_init_sites(atom_coords, atom_numbers, M, key=key, **kwargs)
    return sites_to_spectral(*sites)


@dataclass(frozen=True)
class ChemInit:
    """The data-free chemistry-informed init as a value (see :func:`chem`)."""

    use_ref_exponents: bool = True
    bond_fraction: float = 0.25
    bond_anisotropy: bool = True
    alloc: str = "electron"
    basis_ref: str = "def2-svp"
    bond_scale: float = 1.2
    bond_K: float = 5.0
    bond_aspect: float = 2.5
    bond_spread: float = 2.0
    jitter_scale: float = 0.1
    aniso_jitter: float = 0.02

    def kwargs(self) -> dict:
        return asdict(self)


def chem(*, use_ref_exponents: bool = True, bond_fraction: float = 0.25,
         bond_anisotropy: bool = True, alloc: str = "electron",
         basis_ref: str = "def2-svp", bond_scale: float = 1.2,
         bond_K: float = 5.0, bond_aspect: float = 2.5, bond_spread: float = 2.0,
         jitter_scale: float = 0.1, aniso_jitter: float = 0.02) -> ChemInit:
    """The chemistry-informed initializer as a spec for
    :func:`~gs_dft.ks.energy.init_model` — geometry + tabulated
    element constants only, no per-molecule reference calculation.

    The four heuristics (each independently toggleable): per-element exponent
    ranges from ``basis_ref`` (``use_ref_exponents``), bond-centered splats
    (``bond_fraction > 0``), bond-aligned anisotropic covariance
    (``bond_anisotropy`` — also steps the spectral chart off its isotropy
    singularity), and an electron-count atom budget (``alloc="electron"``).
    """
    if alloc not in ("electron", "uniform"):
        raise ValueError(f"chem: alloc must be 'electron'|'uniform', got {alloc!r}")
    return ChemInit(use_ref_exponents=use_ref_exponents, bond_fraction=bond_fraction,
                    bond_anisotropy=bond_anisotropy, alloc=alloc, basis_ref=basis_ref,
                    bond_scale=bond_scale, bond_K=bond_K, bond_aspect=bond_aspect,
                    bond_spread=bond_spread, jitter_scale=jitter_scale,
                    aniso_jitter=aniso_jitter)
