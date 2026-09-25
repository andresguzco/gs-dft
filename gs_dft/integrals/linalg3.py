"""Closed-form 3×3 linear algebra (batched, differentiable).

The full-covariance hot kernels do thousands-to-millions of *tiny* `(…, 3, 3)`
inverses/determinants/solves per step. Routing those through ``jnp.linalg.*``
means batched LAPACK-style routines (LU with pivoting) — overhead-dominated at
this size and opaque to XLA fusion. The adjugate/cofactor closed forms below
are a handful of FMAs, fuse into the surrounding kernel, and are exactly
differentiable (plain arithmetic, no ``stop_gradient``/branches).

All inputs here are SPD (sums of splat precisions ``A_i + A_j (+ t²I)``), so no
pivoting is needed and the determinant is safely positive.
"""

import jax.numpy as jnp


def det3(A):
    """det(A) for (..., 3, 3) via cofactor expansion along the first row."""
    return (
        A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
        - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
        + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
    )


def adj3(A):
    """Adjugate (transposed cofactor matrix) of (..., 3, 3): A⁻¹ = adj(A)/det(A)."""
    c00 = A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1]
    c01 = A[..., 0, 2] * A[..., 2, 1] - A[..., 0, 1] * A[..., 2, 2]
    c02 = A[..., 0, 1] * A[..., 1, 2] - A[..., 0, 2] * A[..., 1, 1]
    c10 = A[..., 1, 2] * A[..., 2, 0] - A[..., 1, 0] * A[..., 2, 2]
    c11 = A[..., 0, 0] * A[..., 2, 2] - A[..., 0, 2] * A[..., 2, 0]
    c12 = A[..., 0, 2] * A[..., 1, 0] - A[..., 0, 0] * A[..., 1, 2]
    c20 = A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0]
    c21 = A[..., 0, 1] * A[..., 2, 0] - A[..., 0, 0] * A[..., 2, 1]
    c22 = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    return jnp.stack([
        jnp.stack([c00, c01, c02], axis=-1),
        jnp.stack([c10, c11, c12], axis=-1),
        jnp.stack([c20, c21, c22], axis=-1),
    ], axis=-2)


def inv3(A):
    """A⁻¹ for (..., 3, 3) = adj(A)/det(A) (no pivoting — SPD inputs)."""
    return adj3(A) / det3(A)[..., None, None]


def solve3(A, b):
    """A⁻¹ b for A (..., 3, 3), b (..., 3) — adjugate matvec, one division."""
    return jnp.einsum("...kl,...l->...k", adj3(A), b) / det3(A)[..., None]
