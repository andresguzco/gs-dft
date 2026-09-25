"""The Gaussian-splat covariance chart (quaternion orientation + log-eigenvalue scale) — eigendecomposition-free.

Each splat's SPD **precision** is parametrized in its own eigenframe:

    A_i = R(q_i) · diag(exp ℓ_i) · R(q_i)ᵀ,

with a unit quaternion ``q_i`` (orientation) and log-precision-eigenvalues ``ℓ_i ∈ ℝ³`` (anisotropic
scale) → 4 + 3 + 3 = 10 raw params/splat (9 after the quaternion's unit-norm gauge). This is the
3D-Gaussian-Splatting-native chart. Unlike a log-Euclidean chart ``A = expm(S)``, it
needs **no per-splat matrix exponential / eigendecomposition** and **no eigh-backward**: ``A``, ``A⁻¹``,
``det A`` and ``λ_min`` are all closed-form in ``(q, ℓ)``:

    A⁻¹ = R diag(exp −ℓ) Rᵀ,   det A = exp(Σ ℓ),   λ_min(A) = exp(min ℓ).

It is a drop-in for every ``integrals.dense`` integral/grid routine — those consume only ``.A /
.centers / .norm / .n_basis`` — so ``dense.overlap_matrix(spectral_splat)`` etc. just work.

**Coordinate singularity at isotropy.** When the three ``exp ℓ`` coincide, ``A = a·I`` for *every*
``q`` ⇒ ``∂A/∂q ≡ 0``: the orientation gradient vanishes. Off the singular set the chart is regular
and SE(3)-equivariant (spatial rotation Q acts as ``q → q_Q ⊗ q`` ⇒ ``A → Q A Qᵀ``). In practice an
anisotropic init — ``aniso_jitter`` on ``ℓ``, or the chem init's bond-aligned anisotropy
(``basis.chem``) — seeds a nonzero orientation gradient from the first step.
"""
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from jaxtyping import Float, Array

__all__ = ["Splat", "init_spectral_splats", "quat_to_rotation", "quat_mul"]


def quat_to_rotation(q: Float[Array, "... 4"]) -> Float[Array, "... 3 3"]:
    """Unit-quaternion (w, x, y, z) → rotation matrix. ``q`` is normalized first (smooth for |q|>0)."""
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r0 = jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1)
    r1 = jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1)
    r2 = jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    return jnp.stack([r0, r1, r2], axis=-2)


def quat_mul(a: Float[Array, "... 4"], b: Float[Array, "... 4"]) -> Float[Array, "... 4"]:
    """Hamilton product a⊗b. Left-multiply rotates: R(q_Q ⊗ q) = R(q_Q) R(q)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


class Splat(eqx.Module):
    """The splat: a full-covariance Gaussian in the eigendecomposition-free chart
    A = R(q) diag(exp ℓ) R(q)ᵀ — quaternion orientation q + log-precision-eigenvalues ℓ."""

    quat: Float[Array, "M 4"]          # orientation (normalized to a unit quaternion inside R)
    log_scale: Float[Array, "M 3"]     # ℓ: log eigenvalues of the precision A
    centers: Float[Array, "M 3"]

    @property
    def n_basis(self) -> int:
        return self.centers.shape[0]

    @property
    def R(self) -> Float[Array, "M 3 3"]:
        return quat_to_rotation(self.quat)

    @property
    def A(self) -> Float[Array, "M 3 3"]:
        """Precision A = R diag(exp ℓ) Rᵀ — closed-form, NO expm/eigh."""
        R = self.R
        a = jnp.exp(self.log_scale)                          # precision eigenvalues (M, 3)
        return jnp.einsum("mij,mj,mkj->mik", R, a, R)

    @property
    def norm(self) -> Float[Array, "M"]:
        """N_i = (8 det A_i / π³)^{1/4}, det A = Π exp ℓ = exp(Σ ℓ)."""
        detA = jnp.exp(jnp.sum(self.log_scale, axis=-1))
        return (8.0 * detA / jnp.pi ** 3) ** 0.25



def init_spectral_splats(atom_coords, n_splats, *, key,
                         log_alpha_min=-1.0, log_alpha_max=4.5,
                         jitter_scale=0.1, aniso_jitter=0.0):
    """Place ``n_splats`` spectral splats across atoms (near-equal share per atom).

    Isotropic by default: identity quaternion + ``ℓ = log α · 1`` (so ``A = α I``), α geometrically
    spaced. ``aniso_jitter > 0`` adds ``N(0, σ)`` to ``ℓ`` to **break the isotropic orientation-gradient
    singularity** (the spectral chart's stall point) before optimization — see the module docstring.
    """
    n_atoms = atom_coords.shape[0]
    base, rem = divmod(n_splats, n_atoms)
    log_scale, centers = [], []
    for a in range(n_atoms):
        n_a = base + (1 if a < rem else 0)
        if n_a == 0:
            continue
        las = jnp.linspace(log_alpha_min, log_alpha_max, n_a)
        for e in range(n_a):
            la = float(las[e])
            log_scale.append([la, la, la])                  # ℓ = log α · 1  ⇒  A = α I
            centers.append([float(atom_coords[a][i]) for i in range(3)])
    k1, k2 = jr.split(key)
    log_scale = jnp.array(log_scale)
    if aniso_jitter > 0:
        log_scale = log_scale + aniso_jitter * jr.normal(k2, log_scale.shape)
    centers = jnp.array(centers) + jitter_scale * jr.normal(k1, (n_splats, 3))
    quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (n_splats, 1))   # identity orientation
    return Splat(quat=quat, log_scale=log_scale, centers=centers)

