"""Frozen-basis restricted KS-SCF on the splat trainer's own objective.

At fixed splat parameters and a frozen auxiliary basis, the coefficient subproblem is KS-SCF in a
fixed non-orthogonal basis:

    E(P) = Tr(P Hcore) + E_J^DF(P; aux, lam) + E_xc[rho_P]

with the same ``lam`` and density floor as the training energy. The Fock matrix is F = sym(dE/dP),
by autodiff. The solver uses canonical orthogonalization with an eigenvalue cutoff, Pulay DIIS,
optional damping and level shift, and aufbau occupation.

At SCF stationarity the gradient with respect to the splats is the partial derivative at the SCF
coefficients, so no implicit differentiation is needed.
"""
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from gs_dft.integrals import dense as _full
from gs_dft.coulomb import ri as _df
from dftax.energy.grid import xc_energy


class SCFResult(NamedTuple):
    C: jax.Array          # (M, n_occ) S-orthonormal aufbau MO coefficients
    E_elec: float         # electronic energy on the DF objective (no E_nn)
    n_iter: int
    converged: bool
    eps: jax.Array        # (m,) orbital energies in the retained subspace, ascending
    n_kept: int           # m = sigma-modes kept by the canonical orthogonalization


def make_energy_fn(basis, aux, atom_coords, atom_charges, grid_points, grid_weights, xc,
                   lam=1e-8):
    """Precompute the frozen-basis quantities and return ``(E_elec(P), S, Hcore)``.

    ``E_elec`` is the trainer-identical electronic objective as a pure function of the (symmetric)
    density matrix P — Fock = sym(grad(E_elec)). Exposed for tests and external use; ``scf_solve``
    builds the same closure internally through its jitted Fock step."""
    S, T, V = _full.one_electron_integrals(basis, atom_coords, atom_charges)
    Hcore = T + V
    Phi, dPhi = _full.eval_basis_and_grad_on_grid(basis, grid_points)

    def E_elec(P):
        return _electronic_energy(P, basis, aux, Hcore, Phi, dPhi, grid_weights, xc, lam)

    return E_elec, S, Hcore


def _electronic_energy(P, basis, aux, Hcore, Phi, dPhi, gw, xc, lam):
    E1 = jnp.sum(P * Hcore)
    EJ = _df.df_coulomb_energy(basis, P, aux, lam=lam)
    PhiP = Phi @ P
    rho = jnp.maximum(jnp.sum(PhiP * Phi, axis=1), 1e-30)        # same floor as SplatEnergy
    grad_rho = (2.0 * jnp.einsum("gm,gmd->gd", PhiP, dPhi)
                if xc.xc_type == "GGA" else None)                # valid for symmetric P
    return E1 + EJ + xc_energy(xc, rho, gw, grad_rho=grad_rho)


@eqx.filter_jit
def _fock_step(P, basis, aux, Hcore, S, X, Phi, dPhi, gw, xc, lam):
    """One Fock build: E, F = sym(dE/dP), and the DIIS residual in the retained subspace."""
    E, G = jax.value_and_grad(_electronic_energy)(
        P, basis, aux, Hcore, Phi, dPhi, gw, xc, lam)
    F = 0.5 * (G + G.T)
    err = X.T @ (F @ P @ S - S @ P @ F) @ X
    return E, F, err


@eqx.filter_jit
def _diag_step(F, X, occ, C_prev, level_shift, S=None):
    """Diagonalize in the retained subspace; aufbau-occupy the lowest n_occ levels.

    ``level_shift`` (static float) shifts the virtual space up by that amount using the previous
    occupied projector: in the X-subspace, C̃ = XᵀS C_prev is orthonormal (XᵀSX = I), so
    F̃ += shift·(I − C̃C̃ᵀ) leaves occupied levels in place and pushes virtuals up."""
    Ft = X.T @ F @ X
    if level_shift and C_prev is not None and S is not None:
        Ct = X.T @ (S @ C_prev)
        Ft = Ft + level_shift * (jnp.eye(Ft.shape[0]) - Ct @ Ct.T)
    eps, vecs = jnp.linalg.eigh(Ft)
    n_occ = occ.shape[0]
    C = X @ vecs[:, :n_occ]
    P = C @ jnp.diag(occ) @ C.T
    return C, P, eps


def _diis_extrapolate(F_hist, e_hist):
    """Pulay DIIS: host-side augmented B solve over the stored history, drop-oldest fallback."""
    n = len(F_hist)
    while n >= 2:
        Fs, es = F_hist[-n:], e_hist[-n:]
        B = np.empty((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                B[i, j] = B[j, i] = float(jnp.vdot(es[i], es[j]))
        B[n, :n] = B[:n, n] = -1.0
        B[n, n] = 0.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            n -= 1
            continue
        if np.all(np.isfinite(c)):
            F = c[0] * Fs[0]
            for ci, Fi in zip(c[1:], Fs[1:]):
                F = F + ci * Fi
            return F
        n -= 1
    return F_hist[-1]


def scf_solve(basis, aux, atom_coords, atom_charges, grid_points, grid_weights, occ, xc,
              C0=None, sigma_cutoff=1e-8, max_iter=150, e_tol=1e-9, r_tol=1e-6,
              diis_size=8, diis_start=2, damp=0.0, n_damp=0, level_shift=0.0,
              lam=1e-8, verbose=False) -> SCFResult:
    """Solve the frozen-basis restricted KS-SCF problem (aux frozen, see module docstring).

    ``C0=None`` → core-Hamiltonian guess; warm start by passing an (approximately) S-orthonormal
    C — e.g. ``lowdin_orthonormalize(model.C, S)`` — which SCF self-corrects after one cycle.
    Returns the best-energy iterate when not converged. ``n_kept`` reports the canonical-
    orthogonalization rank (modes with sigma > sigma_cutoff·sigma_max)."""
    S, T, V = _full.one_electron_integrals(basis, atom_coords, atom_charges)
    Hcore = T + V
    Phi, dPhi = _full.eval_basis_and_grad_on_grid(basis, grid_points)
    occ = jnp.asarray(occ, float)

    # canonical orthogonalization — eager so the retained rank m is concrete (static jit shapes)
    sig, U = jnp.linalg.eigh(S)
    sig_np = np.asarray(sig)
    idx = np.nonzero(sig_np > sigma_cutoff * sig_np[-1])[0]
    X = U[:, idx] / jnp.sqrt(sig[idx])                           # (M, m)
    n_kept = int(idx.shape[0])

    if C0 is None:                                               # core-Hamiltonian guess
        C, P, _ = _diag_step(Hcore, X, occ, None, 0.0, S)
    else:
        C = jnp.asarray(C0)
        P = C @ jnp.diag(occ) @ C.T

    F_hist, e_hist = [], []
    F_prev = None
    E_prev = np.inf
    best = (np.inf, C, None)                                     # (E, C, eps)
    converged = False
    it = 0
    eps = None
    for it in range(max_iter):
        E, F, err = _fock_step(P, basis, aux, Hcore, S, X, Phi, dPhi, grid_weights,
                               xc, lam)
        E_f, r = float(E), float(jnp.max(jnp.abs(err)))
        if verbose:
            print(f"    scf it {it:3d}  E={E_f:.10f}  max|err|={r:.2e}", flush=True)
        if E_f < best[0]:
            best = (E_f, C, eps)
        if abs(E_f - E_prev) < e_tol and r < r_tol:
            converged = True
            break
        E_prev = E_f
        if it < n_damp and F_prev is not None:                   # damping for rough (cold) starts
            F = (1.0 - damp) * F + damp * F_prev
        F_prev = F
        F_hist.append(F)
        e_hist.append(err)
        if len(F_hist) > diis_size:
            F_hist.pop(0)
            e_hist.pop(0)
        F_use = _diis_extrapolate(F_hist, e_hist) if it >= diis_start else F
        C, P, eps = _diag_step(F_use, X, occ, C, level_shift, S)

    if converged:
        return SCFResult(C=C, E_elec=E_f, n_iter=it + 1, converged=True, eps=eps, n_kept=n_kept)
    E_b, C_b, eps_b = best
    return SCFResult(C=C_b, E_elec=E_b, n_iter=max_iter, converged=False, eps=eps_b, n_kept=n_kept)
