"""Composable Coulomb terms for the splat KS energy.

The two-electron (Coulomb + exact exchange) piece of the splat KS energy comes
in four execution strategies: exact streamed vs density-fitted, each dense or
screened over the significant-overlap pair list. Rather than encoding that
choice as string/bool flags with if-chains on the energy class, each strategy
is a small :class:`equinox.Module` holding exactly its knobs:

- :class:`StreamedExactCoulomb` — exact E_J streamed in bra-pair chunks
  (``coulomb.stream``); hybrids materialize the dense ERI for K (small M only).
- :class:`ScreenedExactCoulomb` — exact streamed ERI over the significant
  pairs (density-fitting-free); PBE-class only (no screened exact-K).
- :class:`ProductDFCoulomb` — frozen pair-product RI-J (+ RI-K for hybrids).
- :class:`ScreenedProductDFCoulomb` — the same DF over the significant pairs.

Unlike the GTO engine (dftax), the splat basis MOVES during training, so no
term holds integral arrays: terms are static strategy values, and the
refresh products they consume (the pair list, the frozen product aux) arrive
per call in a :class:`~gs_dft.ks.energy.SplatState`.

Users select a strategy with the lowercase factories :func:`exact` and
:func:`df`, plus :func:`pairlist` for screening (Optax-style: each knob lives
on the strategy it configures, and invalid combinations raise at the builder):

    SplatKS((coords, charges), xc)                              # exact streamed E_J
    SplatKS(..., coulomb=exact(chunk=128))                    # bra-chunk knob
    SplatKS(..., coulomb=df(scales=(0.5, 1, 2)))              # frozen product RI
    SplatKS(..., coulomb=df(), screen=pairlist(eps=1e-7))     # screened DF
    SplatKS(..., coulomb=exact(), screen=pairlist(eps=1e-7))  # screened exact (PBE)
"""

import abc
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Scalar

from gs_dft.integrals import dense as _full
from gs_dft.coulomb import stream as _cs
from gs_dft.coulomb import ri as _cdf


# ---------------------------------------------------------------------------
# Specs: the user-facing currency for choosing a strategy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExactSpec:
    """Exact streamed-ERI Coulomb backend (see :func:`exact`).

    ``chunk=None`` uses the strategy's own default (64 dense, 128 screened).
    """

    chunk: int | None = None


@dataclass(frozen=True)
class DFSpec:
    """Frozen pair-product density-fitting backend (see :func:`df`)."""

    scales: tuple = (1.0,)
    diagonal: bool = True
    lam: float = 1e-8
    chunk: int = 2048


SCREEN_PAD = 2.0
"""Default pair-list pad multiplier — the one definition, importable so a run
record can state the pad it actually ran at."""


@dataclass(frozen=True)
class Pairlist:
    """Significant-overlap pair screening (see :func:`pairlist`).

    A host-side build spec: :func:`~gs_dft.ks.energy.refresh` (and the
    trainers) rebuild the padded pair list from it as the splats move.
    """

    eps: float = 1e-7
    # The pad is sized once from the pair count at step 0 and frozen for the run: a
    # trajectory-length assumption, not a performance knob. Significant pairs grow up to ~1.85x
    # over 12k steps, rising with system size; lowering it truncates the list mid-run.
    pad: float = SCREEN_PAD
    # j >= i only, with (2 − δ_ij) weights: halves every pair stream (PBE-class only — the screened
    # RI-K contraction is not symmetric in the pair index).
    unique: bool = False
    # Skip all-padding pair chunks in the streams (the pad is appended, so the tail is dummy).
    skip_pad: bool = False


def exact(*, chunk: int | None = None) -> ExactSpec:
    """Exact Coulomb: E_J streamed in bra-pair chunks with an exact custom-vjp
    gradient (O(M²) memory, no M⁴ tape).

    Args:
        chunk: bra-pair chunk of the stream (memory/launch-count knob;
            ``None`` = the strategy default, 64 dense / 128 screened).

    With ``screen=pairlist(...)`` on the builder this becomes the screened
    exact-ERI path (density-fitting-free); hybrids are rejected there (no
    screened exact exchange exists).
    """
    return ExactSpec(chunk=chunk)


def df(*, scales: tuple = (1.0,), diagonal: bool = True,
       lam: float = 1e-8, chunk: int = 2048) -> DFSpec:
    """Density-fitted Coulomb on the frozen self-constructing pair-product aux
    (RI-J, and RI-K for hybrids — the same product basis serves both).

    Args:
        scales: tempered ladder of exponent multipliers for the product aux
            (e.g. ``(0.5, 1.0, 2.0)``).
        diagonal: self-products only (default) vs screened off-diagonal
            products too.
        lam: Tikhonov regularization of the aux-metric solve.
        chunk: pair chunk of the streamed 3-center contraction.
    """
    if not scales:
        raise ValueError("df(scales=...) needs at least one exponent scale.")
    return DFSpec(scales=tuple(float(s) for s in scales), diagonal=bool(diagonal),
                  lam=float(lam), chunk=int(chunk))


def pairlist(eps: float = 1e-7, *, pad: float = SCREEN_PAD, unique: bool = False,
             skip_pad: bool = False) -> Pairlist:
    """Screen the one-electron/Coulomb/grid work to the significant-overlap
    pairs (|Ŝ_ij| > ``eps``; ``eps <= 0`` keeps every ordered pair).

    Args:
        eps: normalized-overlap cutoff.
        pad: fixed pad multiplier for the list (constant shape ⇒ the jitted
            step compiles once; overflow keeps the largest-overlap pairs).
            Must cover pair growth over the whole run, not just drift between
            refreshes.
    """
    return Pairlist(eps=float(eps), pad=float(pad), unique=bool(unique), skip_pad=bool(skip_pad))


# ---------------------------------------------------------------------------
# Coulomb terms (static strategy modules; refresh products come via the state)
# ---------------------------------------------------------------------------

class CoulombTerm(eqx.Module):
    """Coulomb + exact-exchange pieces ``(E_J, E_K)`` of a splat density.

    ``energy`` takes the basis, the Löwdin-orthonormal coefficients, the dense
    density ``P`` (``None`` on the screened paths, which never form it), the
    occupations, and the refresh state (pair list / product aux, where the
    strategy needs them). It returns the pair ``(E_hartree, E_exchange)`` with
    E_exchange unscaled (the builder adds ``hf_coeff · E_exchange`` to the
    total; ``EnergyAux.hartree`` stays the bare Coulomb energy). ``hf_coeff``
    decides whether exchange is computed at all (0 ⇒ ``E_exchange = 0``).
    """

    hf_coeff: float = eqx.field(static=True, default=0.0)
    # Range separation is additive here, not a replacement:
    #     E_x = hf_coeff · E_K[1/r]  +  hf_coeff_lr · E_K[erf(ωr)/r]
    # so `hf_coeff` keeps its meaning and `energy()` keeps returning the same unscaled
    # short-range exchange. `hf_coeff != 0` gates "does this run need exchange at all" in five
    # places (aux_K build, diagnostics, the trainer's a_x); redefining it would change all of them.
    hf_coeff_lr: float = eqx.field(static=True, default=0.0)
    omega: float = eqx.field(static=True, default=0.0)

    @abc.abstractmethod
    def energy(self, basis, C_orth, P, occ, state) -> tuple[Scalar, Scalar]:
        raise NotImplementedError

    def energy_lr(self, basis, C_orth, P, occ, state) -> Scalar:
        """Unscaled long-range exchange E_K[erf(ωr)/r], or 0 where the backend has no ω path.

        Returning zero is correct only because `hf_coeff_lr` is zero for every functional a
        backend without an ω path can be asked to run — `_make_coulomb` refuses the combination
        rather than letting a range-separated hybrid quietly lose most of its exchange."""
        return jnp.array(0.0)


class StreamedExactCoulomb(CoulombTerm):
    """Exact E_J streamed in bra-pair chunks (custom-vjp); dense-ERI K for hybrids."""

    chunk: int = eqx.field(static=True, default=64)

    def energy(self, basis, C_orth, P, occ, state):
        e_j = _cs.full_coulomb_energy(basis, P, bra_chunk=self.chunk)
        if self.hf_coeff != 0.0:
            eri = _full.compute_eri_tensor(basis)
            e_k = _full.exchange_energy(_full.exchange_matrix(eri, P), P)
        else:
            e_k = jnp.array(0.0)
        return e_j, e_k


class ScreenedExactCoulomb(CoulombTerm):
    """Exact streamed ERI over the significant pairs (DF-free); PBE-class only
    (the builder rejects hybrids — there is no screened exact-K)."""

    chunk: int = eqx.field(static=True, default=128)

    def energy(self, basis, C_orth, P, occ, state):
        pi, pj, valid = state.pairs
        e_j = _cs.screened_coulomb_energy(basis, C_orth, occ, pi, pj, valid=valid,
                                          bra_chunk=self.chunk)
        return e_j, jnp.array(0.0)


class ProductDFCoulomb(CoulombTerm):
    """Frozen pair-product RI-J (+ RI-K for hybrids) on the dense pair set.

    ``scales``/``diagonal`` are the aux-construction knobs consumed by
    :func:`~gs_dft.ks.energy.refresh` (the aux tracks the moving splats).
    """

    lam: float = eqx.field(static=True, default=1e-8)
    chunk: int = eqx.field(static=True, default=2048)
    scales: tuple = eqx.field(static=True, default=(1.0,))
    diagonal: bool = eqx.field(static=True, default=True)

    def energy(self, basis, C_orth, P, occ, state):
        e_j = _cdf.df_coulomb_energy(basis, P, state.aux, lam=self.lam, chunk=self.chunk,
                                     cholV=state.cholV)
        if self.hf_coeff != 0.0:
            e_k = _cdf.df_exchange_energy(basis, C_orth, state.aux_K, lam=self.lam)
        else:
            e_k = jnp.array(0.0)
        return e_j, e_k

    def energy_lr(self, basis, C_orth, P, occ, state):
        if self.hf_coeff_lr == 0.0:
            return jnp.array(0.0)
        return _cdf.df_exchange_energy(basis, C_orth, state.aux_K, lam=self.lam,
                                       omega=self.omega)


class ScreenedProductDFCoulomb(CoulombTerm):
    """Frozen pair-product RI-J/RI-K over the significant pairs only."""

    lam: float = eqx.field(static=True, default=1e-8)
    chunk: int = eqx.field(static=True, default=2048)
    scales: tuple = eqx.field(static=True, default=(1.0,))
    diagonal: bool = eqx.field(static=True, default=True)
    skip_pad: bool = eqx.field(static=True, default=False)     # mirrors Pairlist.skip_pad

    def energy(self, basis, C_orth, P, occ, state):
        pi, pj, valid = state.pairs
        e_j = _cdf.df_coulomb_energy_screened(basis, C_orth, occ, state.aux, pi, pj,
                                              lam=self.lam, chunk=self.chunk, valid=valid,
                                              cholV=state.cholV, skip_pad=self.skip_pad)
        if self.hf_coeff != 0.0:
            e_k = _cdf.df_exchange_energy_screened(
                basis, C_orth, state.aux_K, pi, pj, lam=self.lam, chunk=self.chunk,
                valid=valid)
        else:
            e_k = jnp.array(0.0)
        return e_j, e_k

    def energy_lr(self, basis, C_orth, P, occ, state):
        if self.hf_coeff_lr == 0.0:
            return jnp.array(0.0)
        pi, pj, valid = state.pairs
        return _cdf.df_exchange_energy_screened(
            basis, C_orth, state.aux_K, pi, pj, lam=self.lam, chunk=self.chunk,
            valid=valid, omega=self.omega)


def hf_coeff(xc) -> float:
    """Exact-exchange fraction a_x of a functional (0 for pure GGA/LDA)."""
    a = getattr(xc, "exact_exchange_fraction", None)
    return float(a) if a is not None else float(getattr(xc, "hf_coeff", 0.0))


def _make_coulomb(spec, screened: bool, a_x: float, a_x_lr: float = 0.0,
                  omega: float = 0.0, skip_pad: bool = False) -> CoulombTerm:
    """Resolve a user spec (+ screening choice + the functional's a_x) into its term."""
    if isinstance(spec, DFSpec):
        if screened:
            return ScreenedProductDFCoulomb(hf_coeff=a_x, hf_coeff_lr=a_x_lr, omega=omega,
                                            lam=spec.lam, chunk=spec.chunk, scales=spec.scales,
                                            diagonal=spec.diagonal, skip_pad=bool(skip_pad))
        return ProductDFCoulomb(hf_coeff=a_x, hf_coeff_lr=a_x_lr, omega=omega, lam=spec.lam,
                                chunk=spec.chunk, scales=spec.scales, diagonal=spec.diagonal)
    if isinstance(spec, ExactSpec):
        # Refuse, rather than drop the long-range half: the exact backends have no attenuated
        # kernel, and a range-separated hybrid on the exact path would lose most of its exchange
        # and still converge to a plausible number.
        if a_x_lr != 0.0:
            raise ValueError(
                "coulomb=exact() has no range-separated exchange path; a functional with "
                f"hf_coeff_lr={a_x_lr} needs coulomb=df()")
        if screened:
            if a_x != 0.0:
                raise ValueError(
                    "screened exact Coulomb has no exact-exchange path; use a pure "
                    "functional or coulomb=df() (screened RI-K) for hybrids."
                )
            return ScreenedExactCoulomb(hf_coeff=0.0,
                                        chunk=128 if spec.chunk is None else spec.chunk)
        return StreamedExactCoulomb(hf_coeff=a_x,
                                    chunk=64 if spec.chunk is None else spec.chunk)
    raise TypeError(f"coulomb must come from exact() or df(), got {spec!r}")


@dataclass(frozen=True)
class CellGrid:
    """Spec for the block-sparse grid density (see :mod:`gs_dft.screening.cells`)."""
    k: int = 5
    cap: int = 1024
    eps: float = 1e-8
    pt_buckets: int = 4
    sp_buckets: int = 8
    headroom: float = 1.5


def cellgrid(k: int = 5, *, cap: int = 1024, eps: float = 1e-8,
             pt_buckets: int = 4, sp_buckets: int = 8, headroom: float = 1.5) -> CellGrid:
    """Block-sparse grid density: fine Morton cells, batched.

    The dense path evaluates every splat at every grid point; only ~7% matter. Doing that per
    (grid, splat) PAIR loses ~110x to lost data reuse, but at CELL granularity — tight lists from
    fine cells, GEMM efficiency from a batch axis over many small cells.

    Worth it only at scale — it is a loss on small systems.

    Args:
        k: Morton level; cell side = extent/2^k.
        cap: max points per cell. Splits oversized cells into contiguous Morton runs, which also
            tightens occupancy (a sub-run's bounding sphere beats its parent cell's).
        eps: splat cutoff for significance.
        pt_buckets: point-count buckets, so one dense cell does not pad every other.
        sp_buckets: significant-splat-count buckets. Load-bearing: with a single bucket ``S`` is
            the worst cell in the whole point-bucket and the kernel pads away its own advantage.
            Cells are rank-ordered and cut at fixed sizes, so shapes stay constant across refreshes.
        headroom: fixed splat-list width = headroom x the first refresh's max, so the step keeps
            constant shapes and compiles once; overflow keeps the nearest splats.
    """
    return CellGrid(k=int(k), cap=int(cap), eps=float(eps), pt_buckets=int(pt_buckets),
                    sp_buckets=int(sp_buckets), headroom=float(headroom))
