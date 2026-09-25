"""Reusable training diagnostics for splat optimization.

``CollapseGuard`` detects the step at which an optimization run *diverges* — the energy or gradient
goes non-finite (NaN/±Inf) — and serializes the **last-good ("before")** and **first-broken ("after")**
trainable states to disk, plus a JSON of metadata, for surgical post-mortem.

Non-finite is the only trip: a large but finite energy rise is never a collapse. Both things that
cause one — the refresh discontinuity of the screened pair list / aux, and the self-resolving
splat-collision spike — are benign, so halting on an energy jump only aborts runs that would have
recovered.

Design notes:
- JAX arrays are immutable, so retaining the previous step's pytree as "last good" is **free** (a
  reference, not a copy) — only one extra state is ever held.
- Loop-agnostic: feed it ``check(step, state, energy)`` once per step (with ``energy = E(state)``) and a
  ``save_fn(path_prefix, state, step)``; it owns the detection + the before/after dump.
"""
import json
import math

import equinox as eqx
import jax.numpy as jnp

from gs_dft.coulomb import ri as _cdf
from gs_dft.coulomb.stream import full_quantities
from gs_dft.integrals.dense import overlap_times
from gs_dft.ks.orthonormalize import orthonormalize
from gs_dft.ks.terms import ProductDFCoulomb, ScreenedProductDFCoulomb
from gs_dft.screening.pairs import _full_overlap_pairs, screened_overlap_matvec

__all__ = ["CollapseGuard", "lindep_metrics", "dense_gram_eig_ends",
           "dense_gram_spectrum"]


class CollapseGuard:
    # Relative tolerance on the electron-count residual, once the run has earned it (see
    # `_nelec_bad`). Calibrated on large-system runs: two decades above healthy grid drift
    # (~1.7e-5 settled), one decade below the multi-device corruption it exists to catch (~1.3e-2).
    NELEC_RTOL = 1e-3

    def __init__(self, ckpt_base, save_fn, nelec_ref=None, nelec_rtol=None):
        """``ckpt_base``: output path prefix — writes ``{ckpt_base}_before.*`` / ``{ckpt_base}_after.*``
        (via ``save_fn``) and ``{ckpt_base}_collapse.json``. ``save_fn(path_prefix, state, step)``
        serializes one state bundle. A step is a collapse iff its energy or gradient is non-finite
        (NaN/±Inf) — the only true-divergence signal; a finite energy jump never trips it (see the
        module docstring)."""
        self.ckpt_base = ckpt_base
        self.save_fn = save_fn
        self.last_good = None        # (step, E, state) — state is a free pytree reference
        self.triggered = False
        self.nelec_ref = None if nelec_ref is None else float(nelec_ref)
        if nelec_rtol is not None:
            self.NELEC_RTOL = float(nelec_rtol)          # per-run override of the class default
        self.nelec_earned = False    # has the run ever integrated to nelec_ref within tol?
        self.reason = None

    def _nelec_bad(self, nelec):
        """High-water-mark rule on the electron-count residual.

        Löwdin orthonormalization makes ⟨ψ_o|ψ_o⟩ = 1 exactly, so ∫ρ must equal Σocc and any
        deviation is pure grid-integration error. That error is large on young runs (a plain
        threshold would fire on every one of them), but once a run has hit Σocc within tolerance a
        later excursion is not physics — the state cannot silently lose electrons while the kinetic
        energy stays continuous. This is the detector for multi-device state corruption, which no
        finiteness check catches: such runs end "converged" far below the true minimum.
        """
        if self.nelec_ref is None or nelec is None or not math.isfinite(float(nelec)):
            return False
        err = abs(float(nelec) - self.nelec_ref) / max(abs(self.nelec_ref), 1e-300)
        if err <= self.NELEC_RTOL:
            self.nelec_earned = True
            return False
        return self.nelec_earned

    def is_collapse(self, step, energy, gnorm=None, nelec=None):
        """Pure predicate (no side effects): a collapse iff ``energy`` or ``gnorm`` is non-finite,
        or the electron count regresses after having been correct (see :meth:`_nelec_bad`)."""
        if not math.isfinite(float(energy)):
            return True
        if gnorm is not None and not math.isfinite(float(gnorm)):
            return True
        return self._nelec_bad(nelec)

    def check(self, step, state, energy, gnorm=None, nelec=None):
        """Call once per step with the state whose energy is ``energy`` (energy == E(state)). On the
        first detected collapse it saves the before/after pair + metadata and returns True (so the
        caller can halt); otherwise it records ``state`` as the new last-good and returns False.
        ``gnorm`` (optional) lets the guard also trip on a non-finite gradient; ``nelec`` (optional,
        with ``nelec_ref=`` at construction) on a regressed electron count."""
        if self.triggered:
            return True
        if self.is_collapse(step, energy, gnorm, nelec):
            self.triggered = True
            self.reason = ("nelec" if (math.isfinite(float(energy))
                                       and (gnorm is None or math.isfinite(float(gnorm))))
                           else "nonfinite")
            sane = self.last_good
            if sane is not None:
                self.save_fn(f"{self.ckpt_base}_before", sane[2], sane[0])
            self.save_fn(f"{self.ckpt_base}_after", state, step)
            meta = {"collapse_step": step, "E_after": float(energy), "reason": self.reason,
                    "nelec_after": (None if nelec is None else float(nelec)),
                    "nelec_ref": self.nelec_ref,
                    "before_step": (sane[0] if sane else None),
                    "E_before": (sane[1] if sane else None)}
            with open(f"{self.ckpt_base}_collapse.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            return True
        if math.isfinite(float(energy)):
            self.last_good = (step, float(energy), state)
        return False


# ---------------------------------------------------------------------------
# Lin-dep monitor metrics (logging only, NOT part of the energy/optimization)
# ---------------------------------------------------------------------------



@eqx.filter_jit
def _gram_spectrum(basis, C, floor_rel, gap_rel):
    """Array half of the Gram diagnostics, as ONE kernel — the floats are taken on the host."""
    ev = jnp.linalg.eigvalsh(C.T @ overlap_times(basis, C))
    hi = ev[-1]
    gaps = jnp.diff(ev)
    return (ev[0], hi,
            jnp.sum(ev < floor_rel * hi),
            jnp.sum(gaps <= gap_rel * hi),
            jnp.min(gaps) if gaps.size else jnp.asarray(jnp.nan))


def dense_gram_spectrum(basis, C, *, floor_rel: float = 1e-4, gap_rel: float = 1e-4):
    """``(lo, hi, n_floored, n_degenerate, min_gap)`` of the dense MO-Gram CᵀSC.

    The ends alone cannot say whether ``reg_inv_sqrt`` engaged, and engagement is the question:
    its forward clamps every eigenvalue below ``floor_rel·λmax``, and its backward replaces the
    divided difference with a midpoint derivative for every adjacent pair closer than
    ``gap_rel·λmax``. A floored mode also has its gradient contribution zeroed (``discard=True``),
    so past that point the step is not the gradient of the energy being reported.

    Defaults mirror ``reg_inv_sqrt``'s; pass the run's values if they differ.
    """
    lo, hi, n_fl, n_dg, mgap = _gram_spectrum(basis, C, floor_rel, gap_rel)
    return float(lo), float(hi), int(n_fl), int(n_dg), float(mgap)


def dense_gram_eig_ends(basis, C):
    """(λmin, λmax) of the dense exact MO-Gram CᵀSC — the matrix ``reg_inv_sqrt`` actually floors.

    The screened variant this replaced was built from a pair list frozen at the last refresh, and
    screening does not preserve positive definiteness, so it could report a negative λmin where the
    dense Gram stays PSD. Only this ratio is comparable to ``floor_rel``. S is never formed — ``overlap_times`` contracts row
    blocks against C, so this costs (M, n_occ) work plus an (n_occ, n_occ) eigenvalue solve.
    """
    lo, hi, *_ = _gram_spectrum(basis, C, 1e-4, 1e-4)
    return float(lo), float(hi)



def lindep_metrics(ks, model, state=None, *, include_cond: bool = False) -> dict:
    """Linear-dependence / DF-conditioning diagnostics of the CURRENT model.

    The observability bundle the trainer logs at its monitor cadence, exposed
    as a free function so it can be called outside :func:`train`:
    the dense MO-Gram eigenvalue ends, the RI-K energy for hybrids, and (``include_cond=True`` — the costly N_aux²
    eigvalsh) the aux-metric condition number. Host-side; not jitted.
    """
    out: dict = {}
    is_df = isinstance(ks.coulomb, (ProductDFCoulomb, ScreenedProductDFCoulomb))
    use_K = ks.coulomb.hf_coeff != 0.0

    # Lazy, and never through a dense S. This runs at the monitor cadence, so materializing the
    # (M, M) overlap here would reintroduce it into the training loop — and only the RI-K branches
    # below consume C_orth at all, so for a non-hybrid it was built and discarded.
    def C_orth():
        return orthonormalize(model.C, model.C.T @ overlap_times(model.basis, model.C))

    if state is not None and state.pairs is not None:
        pi, pj, v = state.pairs
        # Raw MO-Gram of model.C (not the orthonormalized C, whose Gram is I); both ends because
        # the floor is relative. The dense Gram, not the screened one: screening does not preserve
        # positive definiteness, and the screened ratio was measured disagreeing with this by up
        # to 1100% at exactly the steps where |g| spikes.
        d_lo, d_hi, n_fl, n_dg, mgap = dense_gram_spectrum(model.basis, model.C)
        out["dense_min_eig"], out["dense_max_eig"] = d_lo, d_hi
        out["dense_eig_ratio"] = d_lo / d_hi if d_hi > 0 else float("nan")
        # Whether the regularizer is ACTING, which the ratio cannot report.
        out["gram_floored"], out["gram_degenerate"], out["gram_min_gap"] = n_fl, n_dg, mgap
        if use_K and is_df:
            out["E_K_DF"] = float(_cdf.df_exchange_energy_screened(
                model.basis, C_orth(), state.aux_K, pi, pj, lam=ks.coulomb.lam, valid=v))
    elif use_K and is_df:
        out["E_K_DF"] = float(_cdf.df_exchange_energy(model.basis, C_orth(), state.aux_K,
                                                      lam=ks.coulomb.lam))
    if include_cond and is_df and state is not None and state.aux is not None:
        ev = jnp.linalg.eigvalsh(_cdf.aux_metric(state.aux))
        out["cond_V"] = float(ev[-1] / jnp.clip(ev[0], 1e-30))
    return out
