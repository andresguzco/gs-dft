"""The native grid's resolution as a cap on how sharp a splat may be, and the width limits: no splat is sharper than
the grid integrating the exchange–correlation energy resolves, none wider than WIDE_VAR, none more anisotropic than
ANISO.

Along a principal axis of precision α a splat's density g² ∝ exp(−2αx²) has the spectrum exp(−k²/(8α)). A grid that
resolves wavenumbers up to k_max leaves at most a fraction ε of it unresolved when

    α ≤ k_max² / (8 ln(1/ε)).

Around each atom the native grid is radial shells times a Lebedev rule. At a shell of radius r, Δr from its
neighbours, with n angular points (a rule of degree L ≈ √(3n) − 1), it resolves k_max = min(π/Δr, L/r): the radial
spacing and the angular one along the sphere. A splat takes the k_max of its nearest atom's grid at its centre's
distance d, interpolated in log–log between shells and scaling as 1/d inside the first shell and past the last, so a
splat on a nucleus is uncapped and the cap tightens outward. Each axis is capped alike, which bounds needles and
pancakes as well as round splats.

The shells are read off the grid points: an element's shells are the spheres around one of its atoms that carry a whole
Lebedev rule. They stay fixed along a run (the atoms do not move).
"""
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

RESOLUTION_EPS = 1e-6     # the fraction of a splat's density spectrum the grid may leave unresolved
ANISO = 16.0              # a splat's precisions at most this far apart
WIDE_VAR = 100.0          # bohr²: a splat's widest variance; the log-precision floor is −log(2 WIDE_VAR)
SHELL_RTOL = 1e-9         # distances within this relative gap are one sphere
STRAY = 4                 # other atoms' points a sphere may also hold
MIN_SHELL = 38            # the smallest Lebedev rule taken for a shell
PRESSED = 1e-9            # a log precision this close to a limit counts as pressed against it
TIE = 1e-3                # axis k is clipped to its limit ± k TIE: no two tie (the DF kernel's eigh has no gradient there)


class GridResolution(NamedTuple):
    """Each atom's shells as (A, S) arrays, ascending: log radius and the log wavenumber the grid resolves there (an
    atom with fewer shells continued past its last at doubling radii, the wavenumber as 1/d)."""
    log_radius: jax.Array
    log_wavenumber: jax.Array


def _host(x):
    if isinstance(x, jax.Array) and not x.is_fully_addressable:
        from jax.experimental import multihost_utils
        x = multihost_utils.process_allgather(x, tiled=True)
    return np.asarray(x)


def _lebedev():
    from dftax.grid.lebedev import available_lebedev, lebedev_grid
    return {n: lebedev_grid(n)[0] for n in available_lebedev() if n >= MIN_SHELL}


def _shells(points, center, rules):
    """The spheres around ``center`` that carry a whole Lebedev rule: (radius, points) per shell, ascending."""
    off = points - center
    d = np.linalg.norm(off, axis=1)
    o = np.argsort(d, kind="stable")
    ds = d[o]
    cut = np.concatenate([[0], np.flatnonzero(np.diff(ds) > SHELL_RTOL * ds[1:]) + 1, [ds.size]])
    big = np.flatnonzero(np.diff(cut) >= MIN_SHELL)
    out = []
    for s, e in zip(cut[big], cut[big + 1]):
        if ds[s] == 0.0:
            continue
        u = off[o[s:e]] / ds[s:e, None]
        for n in sorted((n for n in rules if e - s - STRAY <= n <= e - s), reverse=True):
            hit = u @ rules[n].T > 1.0 - 1e-10                  # direction ↔ rule point
            if hit.any(axis=1).sum() == n and np.all(hit.sum(axis=0) == 1):
                out.append((float(np.mean(ds[s:e][hit.any(axis=1)])), n))
                break
    return out


def _wavenumbers(shells):
    """(log r, log k_max) per shell."""
    r, n = np.array([s[0] for s in shells]), np.array([s[1] for s in shells], float)
    dr = np.gradient(r) if r.size > 1 else r
    return np.log(r), np.log(np.minimum(np.pi / dr, (np.sqrt(3.0 * n) - 1.0) / r))


def grid_resolution(ks, verbose=False):
    """The :class:`GridResolution` of ``ks``'s grid, or None for a grid that is not atom-centred shells (then only the
    width limits apply). The mesh's padding points (zero-weight copies of the first, at the end or, with the cells
    dealt over the devices, at the end of each device's block) are left out; the shells must account for every other
    point (a warning says when they do not)."""
    gp, gw, atoms = _host(ks.grid_points), _host(ks.grid_weights), _host(ks.atom_coords)
    Z = np.rint(_host(ks.atom_charges)).astype(int)
    pad = (gw == 0.0) & np.all(gp == gp[0], axis=1)
    pad[0] = False
    gp = gp[~pad]
    n = gp.shape[0]
    rules = _lebedev()
    by_z = {}
    for z in np.unique(Z):
        by_z[z] = _shells(gp, atoms[int(np.flatnonzero(Z == z)[0])], rules)
        if not by_z[z]:
            return None
    covered = sum(sum(m for _, m in by_z[z]) for z in Z)
    if verbose or covered != n:
        print("  resolution: " + ", ".join(f"Z={z} {len(s)} shells" for z, s in by_z.items())
              + (f"; all {n} points on them" if covered == n
                 else f"; WARNING: they hold {covered} of the grid's {n} points"), flush=True)
    shells = [_wavenumbers(by_z[z]) for z in Z]
    width = max(r.size for r, _ in shells)

    def widen(log_r, log_k):
        step = np.log(2.0) * np.arange(1, width - log_r.size + 1)
        return np.concatenate([log_r, log_r[-1] + step]), np.concatenate([log_k, log_k[-1] - step])
    log_r, log_k = (np.stack(x) for x in zip(*(widen(*s) for s in shells)))
    return GridResolution(jnp.asarray(log_r), jnp.asarray(log_k))


def log_precision_cap(res, centers, atoms, eps=RESOLUTION_EPS):
    """The largest log precision each splat may have along any axis, (M,): 2 log k_max − log(8 ln(1/ε)) at its centre
    (``centers`` (M, 3)) on its nearest atom's grid (``atoms`` (A, 3))."""
    d2 = jnp.sum((centers[:, None, :] - atoms[None]) ** 2, -1)
    a = jnp.argmin(d2, axis=1)
    log_d = 0.5 * jnp.log(jnp.maximum(jnp.min(d2, axis=1), jnp.finfo(centers.dtype).tiny))
    lr, lk = res.log_radius[a], res.log_wavenumber[a]
    log_k = jax.vmap(jnp.interp)(log_d, lr, lk)
    log_k = jnp.where(log_d < lr[:, 0], lk[:, 0] + lr[:, 0] - log_d, log_k)          # the grid scales with d
    log_k = jnp.where(log_d > lr[:, -1], lk[:, -1] + lr[:, -1] - log_d, log_k)
    return 2.0 * log_k - jnp.log(8.0 * jnp.log(1.0 / eps))


def capped(basis, res, atoms, eps=RESOLUTION_EPS):
    """``basis`` with each splat's log precisions clipped to :func:`log_precision_cap` (with ``res``), then raised to
    no less than its largest less log ANISO and no less than −log(2 WIDE_VAR); axis k's limits moved inward by k TIE.
    Centres and orientations unchanged."""
    off = TIE * jnp.arange(3.0)
    ls = basis.log_scale
    if res is not None:
        ls = jnp.minimum(ls, log_precision_cap(res, basis.centers, atoms, eps)[:, None] - off)
    floor = jnp.maximum(jnp.max(ls, -1, keepdims=True) - math.log(ANISO), -math.log(2.0 * WIDE_VAR))
    return eqx.tree_at(lambda b: b.log_scale, basis, jnp.maximum(ls, floor + off))


def cap_counts(basis, res, atoms, eps=RESOLUTION_EPS):
    """How many splats have an axis at or past each limit of :func:`capped`, (3,) int32: the sharpness cap, the width
    floor, the anisotropy."""
    off, ls = TIE * jnp.arange(3.0), basis.log_scale
    hi = jnp.max(ls, -1, keepdims=True)
    sharp = (jnp.zeros(ls.shape, bool) if res is None
             else ls >= log_precision_cap(res, basis.centers, atoms, eps)[:, None] - off - PRESSED)
    wide = ls <= -math.log(2.0 * WIDE_VAR) + off + PRESSED
    aniso = ls <= hi - math.log(ANISO) + off + PRESSED
    return jnp.stack([jnp.sum(jnp.any(x, -1), dtype=jnp.int32) for x in (sharp, wide, aniso)])
