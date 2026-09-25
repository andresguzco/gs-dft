"""Density on a grid in bounded memory: grid points are processed in chunks and concatenated, so the
result is exact.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dftax.energy.gto import eval_gto   # dftax 0.8 moved it off ks.energy; ks.terms re-exports
                                        # it too, but energy.gto is where it is defined.

DEFAULT_CHUNK = 8192


def ao_values(basis, coords):
    """AO values at ``coords``, WITHOUT the gradient `ao_on_grid` would also build."""
    return jax.vmap(lambda r: eval_gto(basis, r))(coords)


def gto_density(basis, P, coords, chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """rho(r) = sum_mn ao_m(r) P_mn ao_n(r), evaluated in blocks of ``chunk`` points.

    ``P`` is the SPIN-SUMMED density matrix. Returns a numpy array of length ``len(coords)``.
    """
    coords = jnp.asarray(coords)
    P = jnp.asarray(P)
    n = coords.shape[0]
    step = max(1, int(chunk))

    @jax.jit
    def block(c):
        ao = ao_values(basis, c)                       # (b, nao) -- no Jacobian
        return jnp.einsum("gm,mn,gn->g", ao, P, ao)

    out = []
    for s in range(0, n, step):
        c = coords[s:s + step]
        if c.shape[0] < step and n > step:             # pad the tail to the common shape
            pad = jnp.zeros((step - c.shape[0], 3), dtype=c.dtype)
            out.append(np.asarray(block(jnp.concatenate([c, pad])))[:c.shape[0]])
        else:
            out.append(np.asarray(block(c)))
    return np.concatenate(out) if out else np.zeros(0)
