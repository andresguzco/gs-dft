"""The representation rungs L0 to L4 of Figure 2(a): floating Gaussian parametrizations from the
literature, and ours.

Every rung is an ``eqx.Module`` exposing ``.A``, ``.centers``, ``.norm`` and ``.n_basis``, the whole
interface ``gs_dft.integrals.dense`` reads, so rungs differ in the parametrization and nothing else.

- ``FrostSplat`` (L0, L1, L2): Frost's FSGO, J. Chem. Phys. 47, 3707 (1967). One spherical Gaussian
  per function, optimizing its width and centre.
- ``EllipsoidalSplat`` (L3): the ellipsoidal generalization of Vescelius and Frost (1974). Three
  axis-aligned widths, with the axes fixed to the lab frame.
- ``Splat`` (L4, ``gs_dft/basis/spectral.py``): quaternion orientation and log-eigenvalues. The
  eigenframe is free, so the chart is rotation-equivariant, and A, A^-1, det A and lambda_min are
  closed form.
"""
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["FrostSplat", "EllipsoidalSplat", "RUNGS", "describe"]


class FrostSplat(eqx.Module):
    """L0-L2 — Frost's floating SPHERICAL Gaussian: a centre and one width. 4 params/function."""

    log_width: Float[Array, "M"]        # log of the precision (Frost's rho enters as its reciprocal)
    centers: Float[Array, "M 3"]

    @property
    def n_basis(self) -> int:
        return self.centers.shape[0]

    @property
    def A(self) -> Float[Array, "M 3 3"]:
        a = jnp.exp(self.log_width)                                  # (M,)
        return a[:, None, None] * jnp.eye(3)[None, :, :]

    @property
    def norm(self) -> Float[Array, "M"]:
        detA = jnp.exp(3.0 * self.log_width)                         # isotropic: det = a^3
        return (8.0 * detA / jnp.pi ** 3) ** 0.25


class EllipsoidalSplat(eqx.Module):
    """L3 — the ellipsoidal FSGO: three AXIS-ALIGNED widths, no orientation. 6 params/function."""

    log_scale: Float[Array, "M 3"]
    centers: Float[Array, "M 3"]

    @property
    def n_basis(self) -> int:
        return self.centers.shape[0]

    @property
    def A(self) -> Float[Array, "M 3 3"]:
        return jnp.einsum("mj,jk->mjk", jnp.exp(self.log_scale), jnp.eye(3))

    @property
    def norm(self) -> Float[Array, "M"]:
        detA = jnp.exp(jnp.sum(self.log_scale, axis=-1))
        return (8.0 * detA / jnp.pi ** 3) ** 0.25


# Figure 2(a): the floating-Gaussian model as the literature ran it ("old": axis-aligned widths in the
# lab frame, one function per electron pair, no coefficient matrix) against GS-DFT ("new": a free
# eigenframe, an overcomplete cloud and free coefficients). The panel compares the two methods.
#
# rung -> (chart, free coefficients?, M relative to N_occ)
RUNGS = {
    "old": ("ellipsoidal", False, "n_occ"),   # Vescelius & Frost, ellipsoidal FSGO
    "new": ("spectral",    True,  "over"),    # ours
}

RUNGS.update({
    "L0": ("frost",       False, "n_occ"),   # Frost 1967 as published (ISOTROPIC, the true origin)
    "L1": ("frost",       True,  "n_occ"),   # + free MO coefficients
    "L2": ("frost",       True,  "over"),    # + overcompleteness
    "L3": ("ellipsoidal", True,  "over"),    # + axis-aligned anisotropy
    "L4": ("spectral",    True,  "over"),    # + free orientation
})

_WHY = {
    "old": "ellipsoidal FSGO as published: axis-aligned widths, one per pair, no coefficients",
    "new": "ours: quaternion eigenframe + overcomplete cloud + free coefficients",
    "L0": "Frost FSGO as published: one spherical Gaussian per pair, no coefficients",
    "L1": "+ free MO coefficients (Lowdin-orthonormalized)",
    "L2": "+ overcompleteness (M > N_occ)",
    "L3": "+ axis-aligned anisotropy (ellipsoidal FSGO)",
    "L4": "+ free orientation via the quaternion chart (ours)",
}


def describe(rung: str) -> str:
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; the ladder is {sorted(RUNGS)}")
    return _WHY[rung]
