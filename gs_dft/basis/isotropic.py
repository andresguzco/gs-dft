"""Isotropic floating Gaussian splat — the density-fitting AUXILIARY basis.

Each splat is a single isotropic Gaussian (one scalar exponent) with a free center:

    g_i(r) = N_i · exp(-α_i |r - μ_i|²),   N_i = (2α_i/π)^{3/4}

This is the auxiliary basis for the RI-J Coulomb fit — the (P|a)/(P|Q) integrals are
built in ``coulomb.ri`` from ``coulomb.stream.iso_quantities``. The *primary*
variational basis is the full-covariance ``spectral.Splat``; this module keeps only
the class + initializer the auxiliary needs.
"""

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from jaxtyping import Float, Array

__all__ = ["IsotropicSplat", "init_isotropic_splats", "init_product_aux"]


class IsotropicSplat(eqx.Module):
    """Learnable isotropic Gaussian splat basis (one scalar exponent each)."""

    log_alpha: Float[Array, "M"]     # scalar log-exponent per splat
    centers: Float[Array, "M 3"]

    @property
    def alpha(self) -> Float[Array, "M"]:
        return jnp.exp(self.log_alpha)

    @property
    def n_basis(self) -> int:
        return self.log_alpha.shape[0]

    @property
    def norm(self) -> Float[Array, "M"]:
        """N_i = (2α_i/π)^{3/4}."""
        return (2.0 * self.alpha / jnp.pi) ** 0.75


def init_isotropic_splats(
    atom_coords: Float[Array, "n_atoms 3"],
    n_splats: int,
    *,
    key: jr.PRNGKey,
    log_alpha_min: float = -1.0,
    log_alpha_max: float = 4.5,
    jitter_scale: float = 0.1,
) -> IsotropicSplat:
    """Place exactly ``n_splats`` isotropic splats, distributed across atoms.

    Each atom gets a near-equal share with geometrically spaced exponents
    (log-uniform in [log_alpha_min, log_alpha_max]); centers sit on the atom
    plus a small Gaussian jitter. The remainder (n_splats mod n_atoms) is given
    to the first atoms so the total is exactly ``n_splats``.
    """
    n_atoms = atom_coords.shape[0]
    base, rem = divmod(n_splats, n_atoms)

    all_log_alpha = []
    all_centers = []
    for a in range(n_atoms):
        n_a = base + (1 if a < rem else 0)
        if n_a == 0:
            continue
        las = jnp.linspace(log_alpha_min, log_alpha_max, n_a)
        for e in range(n_a):
            all_log_alpha.append(float(las[e]))
            all_centers.append([float(atom_coords[a][i]) for i in range(3)])

    log_alpha = jnp.array(all_log_alpha)        # (M,)
    centers = jnp.array(all_centers)            # (M, 3)

    k1, _ = jr.split(key)
    centers = centers + jitter_scale * jr.normal(k1, centers.shape)

    return IsotropicSplat(log_alpha=log_alpha, centers=centers)


def init_product_aux(basis, pi=None, pj=None, valid=None, *, scales=(1.0,),
                     diagonal_only=True):
    """Frozen density-fitting aux from the primary splats' own pair products.

    The splat density is exactly a Gaussian mixture over pairs — ρ = Σ_ij P_ij φ_iφ_j — and the
    product of two Gaussians is a single Gaussian at the product center with precision A_i+A_j, so
    the pair products are the natural auxiliary functions: an aux placed there spans ρ by
    construction, leaving the robust-DF lower bound E_J^DF = E_J − ½‖ρ−ρ̃‖² no blind spot for the
    primary to exploit. Only the coefficients d = V⁻¹γ are fit (closed-form, every step); the
    placement is a deterministic function of the primary, so there are no aux parameters to
    optimize and the aux cannot collapse.

    Each kept pair (i,j) → one isotropic aux at the product center with the volume-matched exponent
    α_ij = det(A_i+A_j)^{1/3} (geometric mean of the product-precision eigenvalues — exact for an
    isotropic primary, where the self-product φ_i² has exponent 2α_i). Keeping the aux isotropic
    reuses the existing exact-Boys metric V and the (iso|full) 3-center unchanged (see ``coulomb_df``).
    ``scales`` adds a tempered ladder (α_ij·s for s in scales) so the isotropic set spans the
    product's anisotropy and absorbs drift between refreshes. ``diagonal_only`` keeps just the i==j
    self-products {φ_i²} — the O(M) floor that tracks the splats exactly; pass (pi, pj[, valid]) with
    ``diagonal_only=False`` to also place the screened off-diagonal (bonding) products.

    Returns an ``IsotropicSplat`` (placement frozen; rebuild it as the splats move).
    """
    from gs_dft.coulomb.stream import full_quantities, _full_pair_quantities
    from gs_dft.integrals.linalg3 import det3
    A, mu, N = full_quantities(basis)                          # (M,3,3), (M,3), (M,) precision/center/norm
    if diagonal_only or pi is None:
        idx = jnp.arange(A.shape[0])
        pi, pj, valid = idx, idx, None                        # the self-products φ_i² (centers = μ_i, α = det(2A_i)^{1/3})
    Aij, Pa, _ = _full_pair_quantities((A, mu, N), pi, pj)     # product precision + center per kept pair
    alpha0 = det3(Aij) ** (1.0 / 3.0)                          # (P,) volume-matched isotropic exponent
    if valid is not None:
        alpha0 = jnp.where(valid, alpha0, 1.0)                # padded (invalid) pairs → inert exponent
    log_alpha = jnp.concatenate([jnp.log(alpha0 * float(s)) for s in scales])
    centers = jnp.concatenate([Pa for _ in scales], axis=0)
    return IsotropicSplat(log_alpha=log_alpha, centers=centers)
