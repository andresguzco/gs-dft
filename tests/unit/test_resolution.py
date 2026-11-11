"""gs_dft.ks.resolution: the native grid's resolution as a cap on splat precision, the width limits, and the cap in
the Adam step.

The mesh tests need several devices: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from dftax.energy.xc import PBE
from dftax.grid.grid import _atom_shell_blocks
from dftax.system import Molecule

from gs_dft import SplatKS, df, init_model
from gs_dft.basis.spectral import Splat
from gs_dft.geometries import ethanol_geometry, h2o_geometry
from gs_dft.ks import shard as shd
from gs_dft.ks.resolution import (ANISO, TIE, WIDE_VAR, _wavenumbers, cap_counts, capped, grid_resolution,
                                  log_precision_cap)
from gs_dft.ks.terms import cellgrid, pairlist
from tests.compiles import compiles

tr = importlib.import_module("gs_dft.ks.train")


def _mol(geometry=h2o_geometry):
    return Molecule.from_xyz(geometry, "cc-pvdz", unit="angstrom", spherical=True)


def _ks(level=3, mol=None, mesh=None, screen=False):
    mol = mol or _mol()
    return SplatKS(mol, PBE(), grid=tr.native_grid(mol, level, chunk=512), coulomb=df() if screen else None,
                   screen=pairlist(eps=1e-6) if screen else None, mesh=mesh)


def _splats(ls, centers):
    ls, centers = jnp.asarray(ls, float), jnp.asarray(centers, float)
    return Splat(quat=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (ls.shape[0], 1)), log_scale=ls, centers=centers)


# ------------------------------------------------------------------------------------------------
#  The shells
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("geometry,level", [(h2o_geometry, 1), (h2o_geometry, 3), (ethanol_geometry, 3)])
def test_the_shells_are_the_native_grids_own(geometry, level):
    """Every atom's shells, read off the points, are dftax's radial rule and pruned Lebedev orders for its element."""
    mol = _mol(geometry)
    ks = _ks(level, mol)
    res = grid_resolution(ks)
    nr, lb = tr._GRID_LEVELS[level]
    for a, z in enumerate(np.rint(np.asarray(ks.atom_charges)).astype(int)):
        blocks = _atom_shell_blocks(int(z), nr, lb, "nwchem", 45.0)
        shells = sorted((r, n) for rs, _, n in blocks for r in rs)
        log_r, log_k = _wavenumbers(shells)
        np.testing.assert_allclose(np.asarray(res.log_radius[a, :log_r.size]), log_r, rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(res.log_wavenumber[a, :log_k.size]), log_k, rtol=0, atol=1e-8)
    assert sum(len(r) * n for z in np.asarray(ks.atom_charges)
               for r, _, n in _atom_shell_blocks(int(z), nr, lb, "nwchem", 45.0)) == ks.grid_points.shape[0]


def test_the_shells_do_not_depend_on_the_point_order_or_the_mesh_padding():
    """A permuted grid (the cell path's Morton sort) with zero-weight copies of its first point appended (the mesh
    pad) has the same shells."""
    mol = _mol()
    ks = _ks(1, mol)
    perm = np.random.default_rng(0).permutation(ks.grid_points.shape[0])
    gp, gw = ks.grid_points[perm], ks.grid_weights[perm]
    gp = jnp.concatenate([gp, jnp.broadcast_to(gp[:1], (3, 3))])
    gw = jnp.concatenate([gw, jnp.zeros(3)])
    a, b = grid_resolution(ks), grid_resolution(SplatKS(mol, PBE(), grid=(gp, gw)))
    np.testing.assert_array_equal(np.asarray(a.log_radius), np.asarray(b.log_radius))
    np.testing.assert_array_equal(np.asarray(a.log_wavenumber), np.asarray(b.log_wavenumber))


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices (xla_force_host_platform_device_count=4)")
def test_the_shells_do_not_depend_on_the_cells_dealt_over_a_mesh():
    """With the Morton cells dealt over G = 2, 4 devices (zero-weight copies of the first point closing each device's
    block), the shells are those of one device."""
    mol = _mol()
    ks = lambda mesh: SplatKS(mol, PBE(), grid=tr.native_grid(mol, 1, chunk=512), coulomb=df(),
                              screen=pairlist(eps=1e-6), mesh=mesh, cells=cellgrid())
    a = grid_resolution(ks(None))
    for G in (2, 4):
        b = grid_resolution(ks(shd.make_mesh(G)))
        np.testing.assert_array_equal(np.asarray(a.log_radius), np.asarray(b.log_radius))
        np.testing.assert_array_equal(np.asarray(a.log_wavenumber), np.asarray(b.log_wavenumber))


def test_a_grid_without_shells_has_no_resolution():
    coords, charges = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), jnp.array([1.0, 1.0])
    gp = jr.uniform(jr.PRNGKey(2), (200, 3), minval=-2.0, maxval=2.0)
    assert grid_resolution(SplatKS((coords, charges), PBE(), grid=(gp, jnp.full((200,), 0.05)))) is None


# ------------------------------------------------------------------------------------------------
#  The cap
# ------------------------------------------------------------------------------------------------

def test_a_splat_on_a_nucleus_is_uncapped_and_the_cap_tightens_outward():
    """At a nucleus the cap is past any physical precision (cc-pVDZ's tightest oxygen exponent is ~1.2e4), so a sharp
    core splat there is unchanged; along a ray from it the cap falls."""
    ks = _ks()
    res, atoms = grid_resolution(ks), ks.atom_coords
    ray = atoms[0] + jnp.array([[0.0, 0.0, 1.0]]) * jnp.array([0.0, 0.1, 0.5, 1.0, 2.0, 4.0])[:, None]
    cap = np.asarray(log_precision_cap(res, ray, atoms))
    assert cap[0] > np.log(1e6) and (np.diff(cap) < 0).all()
    core = _splats([[np.log(1.2e4)] * 3], atoms[:1])
    np.testing.assert_array_equal(np.asarray(capped(core, res, atoms).log_scale), np.asarray(core.log_scale))
    assert np.asarray(cap_counts(core, res, atoms)).tolist() == [0, 0, 0]


def test_a_needle_and_a_pancake_are_capped():
    """Off the nucleus a needle loses its sharp axes to the cap, and a pancake its sharp axis, its broad ones then
    raised to ANISO below (axis k's limit k TIE inward, so none tie); a splat inside the limits is unchanged; centres
    and orientations are untouched."""
    ks = _ks()
    res, atoms = grid_resolution(ks), ks.atom_coords
    centers = atoms[0] + jnp.array([[0.0, 0.0, 3.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0]])
    cap = np.asarray(log_precision_cap(res, centers, atoms))
    ls = jnp.array([[cap[0] + 5.0, cap[0] + 5.0, cap[0] - 2.0],             # needle
                    [cap[1] + 3.0, cap[1] - 6.0, cap[1] - 6.0],             # pancake
                    [cap[2] - 1.0, cap[2] - 2.0, cap[2] - 2.5]])            # inside
    b = _splats(ls, centers)
    out = capped(b, res, atoms)
    lo = math.log(ANISO)
    np.testing.assert_allclose(np.asarray(out.log_scale[0]), [cap[0], cap[0] - TIE, cap[0] - 2.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.log_scale[1]), [cap[1], cap[1] - lo + TIE, cap[1] - lo + 2 * TIE],
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(out.log_scale[2]), np.asarray(ls[2]))
    np.testing.assert_array_equal(np.asarray(out.centers), np.asarray(centers))
    np.testing.assert_array_equal(np.asarray(out.quat), np.asarray(b.quat))
    assert np.asarray(cap_counts(out, res, atoms)).tolist() == [2, 0, 1]


def test_the_cap_is_idempotent_and_bounds_the_anisotropy_and_the_variance():
    """Random splats around water, from needles to 1e4-bohr² blobs: capped once, every axis is within the grid's cap,
    the precisions at most ANISO apart and every variance 1/(2α) at most WIDE_VAR; capped again, nothing moves."""
    ks = _ks()
    res, atoms = grid_resolution(ks), ks.atom_coords
    r = np.random.default_rng(1)
    M = 400
    ls = r.uniform(-10.0, 12.0, size=(M, 3))
    q = r.normal(size=(M, 4))
    centers = np.asarray(atoms)[r.integers(0, 3, M)] + r.normal(size=(M, 3)) * r.choice([0.0, 0.5, 2.0, 8.0], (M, 1))
    b = Splat(quat=jnp.asarray(q / np.linalg.norm(q, axis=1, keepdims=True)), log_scale=jnp.asarray(ls),
              centers=jnp.asarray(centers))
    once = capped(b, res, atoms)
    twice = capped(once, res, atoms)
    np.testing.assert_array_equal(np.asarray(twice.log_scale), np.asarray(once.log_scale))
    lo, cap = np.asarray(once.log_scale), np.asarray(log_precision_cap(res, once.centers, atoms))
    beyond = cap < -math.log(2.0 * WIDE_VAR) + 4 * TIE          # tens of bohr out: the width floor wins
    assert (lo[~beyond] <= cap[~beyond, None]).all()
    moved = np.asarray(once.log_scale != b.log_scale)
    gaps = np.diff(np.sort(np.where(moved, lo, np.nan), 1), axis=1)           # between a splat's moved axes
    assert (moved.sum(1) >= 2).sum() > 100 and np.nanmin(gaps) >= 0.5 * TIE   # no tie the eigh cannot differentiate
    assert (lo.max(1) - lo.min(1) <= math.log(ANISO) + 1e-12).all()
    var = 1.0 / (2.0 * np.exp(lo))
    assert var.max() <= WIDE_VAR * (1.0 + 1e-12)
    eig = np.linalg.eigvalsh(np.asarray(once.A))
    assert (eig.max(1) <= ANISO * eig.min(1) * (1.0 + 1e-9)).all() and (1.0 / (2.0 * eig)).max() <= WIDE_VAR * 1.000001


# ------------------------------------------------------------------------------------------------
#  The cap in training
# ------------------------------------------------------------------------------------------------

LOOSE = tr.collapse(nelec_tol_per_atom=1e-2)


def _push(at, value):
    """An optimizer that moves nothing, except every log precision by ``value`` at step ``at``."""
    def init(params):
        return jnp.zeros([], jnp.int32)

    def update(g, count, params=None):
        u = jax.tree.map(jnp.zeros_like, g)
        u = eqx.tree_at(lambda m: m.basis.log_scale, u,
                        jnp.where(count == at, value, 0.0) * jnp.ones_like(u.basis.log_scale))
        return u, count + 1
    return optax.GradientTransformation(init, update)


def _within(model, ks, res):
    lo = np.asarray(model.basis.log_scale)
    cap = np.asarray(log_precision_cap(res, model.basis.centers, ks.atom_coords))
    return ((lo <= cap[:, None] + 1e-12).all() and (lo >= -math.log(2.0 * WIDE_VAR) - 1e-12).all()
            and (lo.max(1) - lo.min(1) <= math.log(ANISO) + 1e-12).all())


@pytest.mark.parametrize("mesh_on", [False, True])
def test_training_keeps_the_splats_capped_and_compiles_nothing_after_step_0(mesh_on):
    """Every log precision pushed up by 12 at step 1: the step projects them back onto the cap (counted), the energy
    and its gradient stay finite, and no step after the first compiles anything, across three refreshes."""
    ks = _ks(1, screen=True, mesh=shd.make_mesh(1) if mesh_on else None)
    model = init_model(_mol(), 30, jr.PRNGKey(0))
    step = [-1]
    with compiles(lambda: step[0] >= 0) as seen:
        res = tr.train(ks, model, steps=8, refresh_every=2, monitor=tr.monitor(2, verbose=False), guard=LOOSE,
                       optimizer=_push(1, 12.0), step_cb=lambda m: step.__setitem__(0, m["step"]))
    assert res.n_steps == 8 and res.stop_reason is None and not seen, (res.stop_reason, seen[:5])
    assert np.isfinite(res.energy).all() and np.isfinite(res.grad_norm).all()
    assert res.n_cap_sharp > 0
    assert _within(res.model, ks, grid_resolution(ks))


def test_the_start_is_capped():
    ks = _ks(1, screen=True)
    model = init_model(_mol(), 30, jr.PRNGKey(0))
    wide = eqx.tree_at(lambda m: m.basis.log_scale, model, model.basis.log_scale - 12.0)
    res = tr.train(ks, wide, steps=1, refresh_every=0, monitor=None, guard=None, optimizer=_push(99, 0.0))
    assert _within(res.model, ks, grid_resolution(ks)) and res.n_cap_wide == 30


@pytest.mark.skipif(jax.device_count() < 2, reason="needs 2 devices (xla_force_host_platform_device_count=2)")
def test_capped_training_on_a_mesh_of_two_matches_one():
    """Adam from splats sharpened past the cap: six steps on a mesh of two give one's energies and splats."""
    model = init_model(_mol(), 30, jr.PRNGKey(0))
    model = eqx.tree_at(lambda m: m.basis.log_scale, model, model.basis.log_scale + 4.0)
    out = []
    for G in (1, 2):
        ks = _ks(1, screen=True, mesh=shd.make_mesh(G))
        out.append(tr.train(ks, model, steps=6, refresh_every=2, monitor=tr.monitor(1, verbose=False),
                            guard=None, optimizer=tr.splat_adam(1e-2, 6)))
    a, b = out
    assert a.n_cap_sharp > 0 and (a.n_cap_sharp, a.n_cap_wide, a.n_cap_aniso) == (b.n_cap_sharp, b.n_cap_wide,
                                                                                  b.n_cap_aniso)
    rel = np.abs(b.energy - a.energy) / np.abs(a.energy)
    dl = float(np.abs(np.asarray(b.model.basis.log_scale) - np.asarray(a.model.basis.log_scale)).max())
    print(f"\nG=2 vs G=1: E rel max {rel.max():.1e}, log precision max diff {dl:.1e}")
    assert rel.max() <= 1e-9 and dl <= 1e-9
