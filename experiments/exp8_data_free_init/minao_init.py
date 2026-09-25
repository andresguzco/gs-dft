"""MinAO initialization of the orbital coefficients C on a given splat basis.

Projects tabulated minimal-basis atomic occupations onto the splats and takes the top natural
orbitals. The occupations are the spherically averaged ground-state configuration of each element,
so no reference calculation is needed.

The textbook projection P = S^-1 S_cross diag(f) S_cross^T S^-1 needs S^-1. Only the natural
orbitals of P in the S metric are needed, and with the canonical orthogonalizer X (X^T S X = I) the
inverse cancels:

    X^T S P S X = (X^T S_cross) diag(f) (X^T S_cross)^T = Z diag(f) Z^T,   Z = X^T S_cross

so C = X c from the eigenvectors of that symmetric PSD matrix. Occupations stay 2 and C is
S-orthonormal, so Tr(P S) = 2 n_occ exactly.

``S_cross`` = <splat | minimal AO> is computed by quadrature on the caller's grid, so its
quadrature error enters the guess; STO-3G is compact and the Becke grid resolves it.
"""
import numpy as np
import jax.numpy as jnp

from dftax.basis.loader import build_basis_data
from dftax.ks.energy import ao_on_grid
from dftax.ks.guess import _minimal_occupations          # engine-private; PySCF-validated
from dftax.system.molecule import symbol_to_Z

from gs_dft.integrals.dense import overlap_matrix, eval_basis_on_grid

MIN_BASIS = "sto-3g"


def _minimal_occ_vector(symbols, coords, min_basis=MIN_BASIS):
    """Concatenated spherically-averaged ground-state occupations over the molecule's minimal AOs.

    Per element, ``count/(2l+1)`` per m component of each l-shell — the tabulated configuration,
    so this stays data-free (no reference calculation anywhere).
    """
    f = []
    for sym, R in zip(symbols, np.asarray(coords)):
        minb_A = build_basis_data([sym], R.reshape(1, 3), min_basis, spherical=True)
        f.append(np.asarray(_minimal_occupations(symbol_to_Z(sym), minb_A, min_basis)))
    return np.concatenate(f)


def _cross_overlap_on_grid(basis, minb, grid_points, grid_weights, chunk=4096):
    """``S_cross[m, mu] = sum_g w_g g_m(r_g) chi_mu(r_g)`` — STREAMED over grid chunks.

    Grid-quadratured because splat x angular-momentum-GTO overlaps have no analytic path here.
    O(chunk*(M + n_min)) memory: the dense basis-on-grid is never formed.
    """
    acc = None
    gp, gw = np.asarray(grid_points), np.asarray(grid_weights)
    for s in range(0, gp.shape[0], chunk):
        pts = jnp.asarray(gp[s:s + chunk])
        w = jnp.asarray(gw[s:s + chunk])
        blk = eval_basis_on_grid(basis, pts).T @ (w[:, None] * ao_on_grid(minb, pts)[0])
        acc = blk if acc is None else acc + blk
    return np.asarray(acc)


def minao_C0(basis, symbols, coords, n_occ, grid_points, grid_weights, *,
             thresh=1e-6, min_basis=MIN_BASIS, chunk=4096):
    """MinAO guess coefficients: the top ``n_occ`` natural orbitals of the projected minimal-basis
    density. Returns ``(M, n_occ)``, S-orthonormal (``C^T S C = I``).

    ``thresh`` drops S-eigenmodes below ``thresh * lambda_max``; unlike in SAP this is free, since
    the dropped directions carry no density.
    """
    minb = build_basis_data(list(symbols), np.asarray(coords), min_basis, spherical=True)
    f = _minimal_occ_vector(symbols, coords, min_basis)
    S_cross = _cross_overlap_on_grid(basis, minb, grid_points, grid_weights, chunk)
    if S_cross.shape[1] != f.shape[0]:
        raise ValueError(f"minimal AO count {S_cross.shape[1]} != occupation count {f.shape[0]}")

    # Both eigendecompositions run on the accelerator: at insulin (M=15142) a dense 15k eigh is
    # 5.5 s on an A100 against ~26 min in CPU numpy, and there are two of them.
    S = overlap_matrix(basis)
    if S.shape[0] > 30000:
        # cuSOLVER's syevBatched workspace query overflows int32 near ~2*M^2 elements: M=39246
        # (7281K) fails INTERNAL at bufferSize while M=21026 (LG5K9) fits. Host LAPACK instead.
        sval, U = (jnp.asarray(a) for a in np.linalg.eigh(np.asarray(S)))
    else:
        sval, U = jnp.linalg.eigh(S)                      # ascending
    k = int((sval > thresh * sval[-1]).sum())             # ascending => the kept modes are a suffix
    X = U[:, -k:] / jnp.sqrt(sval[-k:])                   # (M, k), X^T S X = I
    Z = X.T @ jnp.asarray(S_cross)                        # (k, n_min)

    W = Z * jnp.sqrt(jnp.asarray(f))[None, :]             # (k, n_min)
    lam, v = jnp.linalg.eigh(W.T @ W)                     # (n_min, n_min), ascending
    # A null direction of W^T W has no density and no eigenvector to recover; its 1/sqrt(lam) would
    # be a division by ~0, so clamp before dividing and let the TOP-n_occ slice ignore it.
    u = (W @ v) / jnp.sqrt(jnp.maximum(lam, jnp.finfo(lam.dtype).tiny))[None, :]
    return X @ u[:, -n_occ:]                              # TOP, not bottom — see module docstring


# ---------------------------------------------------------------------------------------------------
#  MinAO without the (M, M) eigendecomposition. The dense path exists only to apply S^+ to the n_min
#  cross-overlap columns. S is sparse after pair screening, so those columns come from screened
#  conjugate-gradient solves of (S + ridge·I) Y = S_cross instead, and the rest is the same small
#  (n_min, n_min) eigenproblem:
#      G = Yᵀ (S Y),  K = F^½ G F^½,  K v = v λ,  C = Y F^½ v λ^{-½}      ⇒  Cᵀ S C = I exactly.
#  The ridge plays thresh's role (near-null S modes carry no density and are damped, not inverted),
#  and the CG iteration cap is a second, Krylov-type regularizer. Nothing (M, M) is ever formed:
#  O(iters · n_pair · n_min) on the accelerator, against O(M³) on the host.
# ---------------------------------------------------------------------------------------------------

def _screened_cg_solve(basis, pi, pj, valid, B, ridge, iters, tol, verbose=False):
    """Column-wise CG for ``(S + ridge·I) X = B`` with the screened overlap matvec; ``B`` is (M, c)."""
    import equinox as eqx
    from gs_dft.screening import screened_overlap_matvec

    @eqx.filter_jit
    def A(basis_, pi_, pj_, valid_, X):
        return screened_overlap_matvec(basis_, pi_, pj_, X, valid_) + ridge * X

    @eqx.filter_jit
    def step(basis_, pi_, pj_, valid_, X, R, P, rs):
        AP = A(basis_, pi_, pj_, valid_, P)
        alpha = rs / jnp.maximum(jnp.sum(P * AP, axis=0), 1e-300)
        X = X + alpha[None, :] * P
        R = R - alpha[None, :] * AP
        rs_new = jnp.sum(R * R, axis=0)
        P = R + (rs_new / jnp.maximum(rs, 1e-300))[None, :] * P
        return X, R, P, rs_new

    X = jnp.zeros_like(B)
    R = B
    P = R
    rs = jnp.sum(R * R, axis=0)
    rs0 = jnp.maximum(rs, 1e-300)
    k = 0
    for k in range(1, iters + 1):
        X, R, P, rs = step(basis, pi, pj, valid, X, R, P, rs)
        if k % 25 == 0 or k == iters:
            rel = float(jnp.sqrt(jnp.max(rs / rs0)))
            if verbose:
                print(f"    cg {k:4d}  worst rel residual {rel:.2e}", flush=True)
            if rel < tol:
                break
    return X, k, float(jnp.sqrt(jnp.max(rs / rs0)))


def minao_C0_pcg(basis, symbols, coords, n_occ, grid_points, grid_weights, *,
                 ridge=1e-3, cg_iters=200, cg_tol=1e-6, col_chunk=2048, pair_eps=1e-7,
                 min_basis=MIN_BASIS, chunk=4096, verbose=True):
    """MinAO guess coefficients without any (M, M) object — see the note above. Returns ``(M, n_occ)``
    with ``Cᵀ S C = I`` to solver precision. ``ridge`` damps the near-null overlap modes (the dense
    path's ``thresh``); ``cg_iters`` caps each column's CG (a Krylov regularizer on its own)."""
    import time
    from gs_dft.screening import neighbor_pairs, screened_overlap_matvec

    t0 = time.time()
    minb = build_basis_data(list(symbols), np.asarray(coords), min_basis, spherical=True)
    f = jnp.asarray(_minimal_occ_vector(symbols, coords, min_basis))
    S_cross = jnp.asarray(_cross_overlap_on_grid(basis, minb, grid_points, grid_weights, chunk))
    if S_cross.shape[1] != f.shape[0]:
        raise ValueError(f"minimal AO count {S_cross.shape[1]} != occupation count {f.shape[0]}")
    n_min = int(S_cross.shape[1])
    pi, pj, valid = neighbor_pairs(basis, eps=pair_eps)
    if verbose:
        print(f"  minao_pcg: M={int(basis.n_basis)} n_min={n_min} pairs={int(pi.shape[0])}  "
              f"cross-overlap + pair list in {time.time() - t0:.0f}s", flush=True)

    cols = []
    worst = 0.0
    for c0 in range(0, n_min, col_chunk):
        Yc, k, rel = _screened_cg_solve(basis, pi, pj, valid, S_cross[:, c0:c0 + col_chunk],
                                        ridge, cg_iters, cg_tol, verbose=verbose and c0 == 0)
        worst = max(worst, rel)
        cols.append(Yc)
    Y = jnp.concatenate(cols, axis=1)                                   # (M, n_min) ≈ (S+εI)⁻¹ S_cross
    SY = screened_overlap_matvec(basis, pi, pj, Y, valid)               # exact S Y
    G = Y.T @ SY                                                        # (n_min, n_min)
    sf = jnp.sqrt(f)
    K = sf[:, None] * G * sf[None, :]
    lam, v = jnp.linalg.eigh(K)                                         # ascending
    top = slice(n_min - n_occ, n_min)
    C = (Y * sf[None, :]) @ (v[:, top] / jnp.sqrt(jnp.maximum(lam[top], jnp.finfo(lam.dtype).tiny))[None, :])
    if verbose:
        print(f"  minao_pcg: done in {time.time() - t0:.0f}s (worst CG rel residual {worst:.1e}; "
              f"smallest kept density eigenvalue {float(lam[top][0]):.3e})", flush=True)
    return C
