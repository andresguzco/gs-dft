"""Block-local MinAO: fit each atom's minimal atomic orbitals from nearby splats only.

MinAO needs the global canonical orthogonalizer, a dense (M, M) overlap and an O(M^3)
eigendecomposition. The least-squares fit of one atomic orbital onto the splats is nearly local,
since ``chem()`` places splats at atoms and bonds, so it is done per atom and the inter-atomic
overlap is handled once, in the occupied space:

    A     (M, n_min)   block-sparse, from per-atom solves
    G     = A^T (S A)  via overlap_times, so S is never formed
    K     = diag(f)^1/2 G diag(f)^1/2          (n_min, n_min)
    lam,v = eigh(K);  C = A diag(f)^1/2 v / sqrt(lam)

C^T S C = I by construction and the electron count is exact; only the shape is approximated.
"""
import numpy as np
import jax.numpy as jnp

from dftax.basis.loader import build_basis_data
from gs_dft.coulomb.stream import full_quantities
from gs_dft.integrals.dense import overlap_times
from gs_dft.screening.pairs import _full_overlap_pairs
from experiments.exp8_data_free_init.minao_init import (MIN_BASIS, _minimal_occ_vector,
                                                        _cross_overlap_on_grid)


def _atom_of_minimal_ao(symbols, coords, min_basis=MIN_BASIS):
    """Which atom each minimal AO belongs to, in the order `build_basis_data` concatenates them."""
    owner = []
    for a, (sym, R) in enumerate(zip(symbols, np.asarray(coords))):
        b = build_basis_data([sym], np.asarray(R).reshape(1, 3), min_basis, spherical=True)
        n = int(_minimal_occ_vector([sym], np.asarray(R).reshape(1, 3), min_basis).shape[0])
        owner.extend([a] * n)
    return np.asarray(owner)


def _local_overlap(q, idx):
    """``S[idx, idx]`` from the per-pair kernel — the dense (M,M) is never built."""
    n = idx.shape[0]
    pi = jnp.repeat(jnp.asarray(idx, jnp.int32), n)
    pj = jnp.tile(jnp.asarray(idx, jnp.int32), n)
    return _full_overlap_pairs(q, pi, pj).reshape(n, n)


def local_C0(basis, symbols, coords, n_occ, grid_points, grid_weights, *,
             min_basis=MIN_BASIS, n_near=None, ridge=1e-8, chunk=4096):
    """MinAO coefficients from per-atom fits. Returns ``(M, n_occ)``, S-orthonormal.

    ``n_near`` splats nearest each atom take part in that atom's fit (default: scaled by M/n_atoms,
    with a floor, so the local basis grows with the splat budget rather than being capped by the
    minimal basis the way a global STO-3G bridge is).
    """
    coords = np.asarray(coords)
    n_atoms = len(symbols)
    centers = np.asarray(basis.centers)
    M = centers.shape[0]
    if n_near is None:
        n_near = max(16, int(6.0 * M / max(n_atoms, 1)))
    n_near = min(int(n_near), M)

    minb = build_basis_data(list(symbols), coords, min_basis, spherical=True)
    f = _minimal_occ_vector(symbols, coords, min_basis)
    S_cross = np.asarray(_cross_overlap_on_grid(basis, minb, grid_points, grid_weights, chunk))
    owner = _atom_of_minimal_ao(symbols, coords, min_basis)
    if S_cross.shape[1] != f.shape[0]:
        raise ValueError(f"minimal AO count {S_cross.shape[1]} != occupation count {f.shape[0]}")

    q = full_quantities(basis)
    A = np.zeros((M, f.shape[0]))
    for a in range(n_atoms):
        cols = np.flatnonzero(owner == a)
        if cols.size == 0:
            continue
        d2 = ((centers - coords[a][None, :]) ** 2).sum(1)
        idx = np.argpartition(d2, min(n_near, M - 1))[:n_near]
        S_ll = np.array(_local_overlap(q, idx))       # copy: device arrays are read-only
        S_ll[np.diag_indices_from(S_ll)] += ridge * max(np.trace(S_ll) / len(idx), 1e-300)
        A[np.ix_(idx, cols)] = np.linalg.solve(S_ll, S_cross[idx][:, cols])

    Aj = jnp.asarray(A)
    fh = jnp.sqrt(jnp.asarray(f))
    G = Aj.T @ overlap_times(basis, Aj)                    # (n_min, n_min); S never formed
    K = (fh[:, None] * G) * fh[None, :]
    lam, v = jnp.linalg.eigh(K)                            # ascending
    u = (Aj * fh[None, :]) @ v / jnp.sqrt(jnp.maximum(lam, jnp.finfo(lam.dtype).tiny))[None, :]
    return u[:, -n_occ:]
