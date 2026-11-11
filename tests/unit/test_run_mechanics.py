"""Run mechanics: compile once, the guards, the production pair setting and the mesh at every G.

The mesh tests need several devices: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from dftax.energy.xc import PBE, PBE0
from dftax.system import Molecule
from gs_dft import SplatKS, df, init_model, refresh
from gs_dft import screening as scr
from gs_dft.geometries import h2o_geometry
from gs_dft.ks import shard as shd
from gs_dft.ks import train as _train_verb                     # noqa: F401  (the verb; module below)
from gs_dft.ks.terms import pairlist
from gs_dft.ks.diagnostics import grid_norm_fields
from gs_dft.ks.resolution import grid_resolution
from tests.compiles import compiles

import importlib
tr = importlib.import_module("gs_dft.ks.train")


def _mol(geometry=h2o_geometry):
    return Molecule.from_xyz(geometry, "cc-pvdz", unit="angstrom", spherical=True)


def _ks(mesh=None, xc=None, level=1, mol=None, eps=1e-6, **screen):
    mol = mol or _mol()
    return SplatKS(mol, xc or PBE(), grid=tr.native_grid(mol, level, chunk=512), coulomb=df(),
                   screen=pairlist(eps=eps, **screen), mesh=mesh)


def _model(M=30, seed=0, mol=None):
    return init_model(mol or _mol(), M, jr.PRNGKey(seed))


@eqx.filter_jit
def _vg(model, ks, st):
    return eqx.filter_value_and_grad(lambda m: ks(m, st)[0])(model)


def _value_and_grad(ks, model, pad):
    E, g = _vg(model, ks, tr._refresh_state(ks, model.basis, pad, False))
    return float(E), np.concatenate([np.ravel(np.asarray(x)) for x in
                                     jax.tree.leaves(eqx.filter(g, eqx.is_inexact_array))])


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


# The level-1 test grid's own electron error reaches ~2e-4/atom on a young run, past the production
# tolerance (calibrated at level 3); the tests not about that guard loosen it.
LOOSE = tr.collapse(nelec_tol_per_atom=1e-2)


def _kick(at, where, value):
    """An optimizer that moves nothing, except ``where(model)`` by ``value`` at step ``at``."""
    def init(params):
        return jnp.zeros([], jnp.int32)

    def update(g, count, params=None):
        u = jax.tree.map(jnp.zeros_like, g)
        u = eqx.tree_at(where, u, jnp.where(count == at, value, 0.0) * jnp.ones_like(where(u)))
        return u, count + 1
    return optax.GradientTransformation(init, update)


# ------------------------------------------------------------------------------------------------
#  A: compile once
# ------------------------------------------------------------------------------------------------

def test_pair_weights_are_strong_typed_even_when_the_list_fills_its_pad():
    model = _model()
    n = scr.count_pairs(model.basis, eps=1e-6, unique=True)
    for pad in (n, n + 7):
        _pi, _pj, w = scr.neighbor_pairs(model.basis, eps=1e-6, pad_to=pad, unique=True)
        assert w.dtype == jnp.float64 and not w.weak_type


def test_a_new_pair_count_decodes_without_compiling():
    model, moved = _model(seed=0), _model(seed=1)
    pad = 2 * scr.count_pairs(model.basis, eps=1e-6, unique=True)
    assert scr.count_pairs(moved.basis, eps=1e-6, unique=True) != pad // 2
    scr.neighbor_pairs(model.basis, eps=1e-6, pad_to=pad, unique=True)
    with compiles() as seen:
        scr.neighbor_pairs(moved.basis, eps=1e-6, pad_to=pad, unique=True)
    assert not seen, seen


@pytest.mark.parametrize("mesh_on", [False, True])
def test_training_compiles_nothing_after_step_0(mesh_on):
    ks = _ks(shd.make_mesh(1) if mesh_on else None)
    step = [-1]
    with compiles(lambda: step[0] >= 0) as seen:
        res = tr.train(ks, _model(), steps=8, refresh_every=2, monitor=tr.monitor(2, verbose=False),
                       guard=LOOSE, step_cb=lambda m: step.__setitem__(0, m["step"]))
    assert res.n_steps == 8 and res.stop_reason is None
    assert not seen, seen[:5]


@pytest.mark.parametrize("G", [1, 2])
def test_a_resumed_run_steps_on_its_compiled_step(tmp_path, G):
    """Loaded arrays are placed where the compiled step expects them, the optimizer state split over
    the mesh as the step was compiled for."""
    if jax.device_count() < G:
        pytest.skip(f"needs {G} devices")
    ks, base = _ks(shd.make_mesh(G)), str(tmp_path / "run")
    tr.train(ks, _model(), steps=4, refresh_every=2, monitor=None, guard=LOOSE,
             checkpoint=tr.ckpt(base, every=2))
    res = tr.train(ks, _model(), steps=6, refresh_every=2, monitor=None, guard=LOOSE,
                   checkpoint=tr.ckpt(base, every=2, resume=True))
    assert res.n_steps == 6 and res.stop_reason is None


def test_the_compiled_step_raises_on_a_shape_change():
    ks, model = _ks(), _model()
    pad = 2 * scr.count_pairs(model.basis, eps=1e-6, unique=True)
    m_tr, m_st = tr._partition(model)
    opt = tr.splat_adam(1e-2, 10)
    res = grid_resolution(ks)
    step = tr._compiled_step(tr._step_fn(opt, m_st, False),
                             (ks, refresh(ks, model.basis, pad_to=pad), res), m_tr, opt.init(m_tr))
    with pytest.raises((TypeError, ValueError)):
        step((ks, refresh(ks, model.basis, pad_to=pad + 1), res), m_tr, opt.init(m_tr))


# ------------------------------------------------------------------------------------------------
#  B: the guards, each forced
# ------------------------------------------------------------------------------------------------

def test_a_pair_pad_overflow_raises_naming_the_pad(monkeypatch):
    real = scr.count_pairs
    monkeypatch.setattr(scr, "count_pairs", lambda *a, **k: real(*a, **k) // 4)   # pad = half the list
    with pytest.raises(scr.PairPadOverflow, match="pad"):
        tr.train(_ks(), _model(), steps=2, refresh_every=1, monitor=None)


def test_an_overflow_at_a_refresh_saves_the_point_and_a_rerun_resumes_there(tmp_path, monkeypatch, capsys):
    real, calls = tr._refresh_state, []

    def overflows_at_the_first_refresh(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise scr.PairPadOverflow("forced")
        return real(*a, **k)

    spec = tr.ckpt(str(tmp_path / "run"), every=100, resume=True)       # only the overflow saves
    monkeypatch.setattr(tr, "_refresh_state", overflows_at_the_first_refresh)
    with pytest.raises(scr.PairPadOverflow):
        tr.train(_ks(), _model(), steps=4, refresh_every=2, monitor=None, checkpoint=spec)
    monkeypatch.setattr(tr, "_refresh_state", real)
    tr.train(_ks(), _model(), steps=4, refresh_every=2, monitor=None, checkpoint=spec)
    assert "at step 2/4" in capsys.readouterr().out


def test_a_non_finite_step_stops_the_run_by_default():
    res = tr.train(_ks(), _model(), steps=6, refresh_every=0, monitor=None,
                   optimizer=_kick(2, lambda m: m.C, jnp.nan))
    assert res.stop_reason == "nonfinite" and res.collapsed and res.n_steps == 4


def test_the_grid_electron_error_stops_the_run_once_earned():
    ks = _ks()
    probe = tr.train(ks, _model(), steps=1, refresh_every=0, monitor=None, guard=None,
                     optimizer=_kick(99, lambda m: m.C, 0.0))
    e0 = abs(float(probe.nelec_err[0]))
    # Splats moved 40 bohr off the grid at step 2: the grid then misses every electron.
    res = tr.train(ks, _model(), steps=6, refresh_every=0, monitor=tr.monitor(1, verbose=False),
                   guard=tr.collapse(nelec_tol_per_atom=2 * e0 + 1e-9),
                   optimizer=_kick(2, lambda m: m.basis.centers, 40.0))
    assert res.stop_reason == "nelec" and res.n_steps == 4
    assert abs(res.nelec_err[-1]) > 0.5


def test_the_grid_electron_error_is_recorded_at_monitor_and_refresh_points():
    res = tr.train(_ks(), _model(), steps=7, refresh_every=3, monitor=tr.monitor(5, verbose=False),
                   guard=LOOSE)
    assert res.nelec_step.tolist() == [0, 3, 5, 6]
    assert np.all(np.isfinite(res.nelec_err))


def test_the_splat_grid_norm_flags_a_splat_the_grid_cannot_integrate():
    ks, model = _ks(level=1), _model()
    healthy = grid_norm_fields(ks, model)
    k = int(jnp.argmax(ks.grid_weights))                      # sit a sharp splat on one grid point
    basis = eqx.tree_at(lambda b: (b.centers, b.log_scale), model.basis,
                        (model.basis.centers.at[0].set(ks.grid_points[k]),
                         model.basis.log_scale.at[0].set(jnp.log(1e4))))
    sharp = eqx.tree_at(lambda m: (m.basis, m.C), model, (basis, model.C.at[0].set(1.0)))
    bad = grid_norm_fields(ks, sharp)
    assert bad["splat_norm_max"] > 10.0 and bad["splat_norm_n_over"] >= 1
    assert bad["splat_norm_excess"] > 10 * healthy["splat_norm_excess"]


# ------------------------------------------------------------------------------------------------
#  A: the production pair setting and the mesh at every G
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("system", [
    "water", pytest.param("ethanol", marks=pytest.mark.slow)])
def test_the_production_path_matches_the_release_setting(system):
    """Unique pairs, padding skipped, on a one-device mesh, against ordered pairs, unskipped, with no
    mesh: E and gradient to 1e-10 relative. Ethanol at M=144 on the level-3 grid, eps 1e-7 (the
    production setting) is the slow case."""
    from gs_dft.geometries import ethanol_geometry
    mol, M, level, eps = ((_mol(), 30, 1, 1e-6) if system == "water" else
                          (_mol(ethanol_geometry), 144, 3, 1e-7))
    model = _model(M, mol=mol)
    old = _ks(mol=mol, level=level, eps=eps, unique=False, skip_pad=False)
    new = _ks(shd.make_mesh(1), mol=mol, level=level, eps=eps)
    assert new.screen.unique and new.screen.skip_pad
    E0, g0 = _value_and_grad(old, model, 2 * scr.count_pairs(model.basis, eps=eps))
    E1, g1 = _value_and_grad(new, model, 2 * scr.count_pairs(model.basis, eps=eps, unique=True))
    print(f"\nidentity {system}: E {E1:.12f} vs {E0:.12f}, rel {abs(E1 - E0) / abs(E0):.1e}; "
          f"gradient rel {_rel(g1, g0):.1e}")
    assert abs(E1 - E0) <= 1e-10 * abs(E0), (E1, E0)
    assert _rel(g1, g0) <= 1e-10, _rel(g1, g0)


def test_hybrids_keep_the_ordered_list_and_one_device():
    from experiments.common import builders
    ks = builders.splat_ks(_mol(), PBE0(), grid_level=0, screened=True)
    assert ks.mesh is None and ks.screen.unique is False
    with pytest.raises(ValueError, match="one device"):
        builders.splat_ks(_mol(), PBE0(), grid_level=0, screened=True, G=2)
    with pytest.raises(ValueError, match="PBE-class"):
        _ks(xc=PBE0(), unique=True)
    assert builders.splat_ks(_mol(), PBE(), grid_level=0, screened=True).mesh.size == 1


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices (xla_force_host_platform_device_count=4)")
def test_the_mesh_at_g_1_2_4_agrees():
    model = _model()
    pad = 4 * scr.count_pairs(model.basis, eps=1e-6, unique=True)          # a multiple of every G
    E1, g1 = _value_and_grad(_ks(shd.make_mesh(1)), model, pad)
    for G in (2, 4):
        EG, gG = _value_and_grad(_ks(shd.make_mesh(G)), model, pad)
        print(f"\nG={G} vs G=1: E rel {abs(EG - E1) / abs(E1):.1e}, gradient rel {_rel(gG, g1):.1e}")
        assert abs(EG - E1) <= 1e-9 * abs(E1), (G, EG, E1)
        assert _rel(gG, g1) <= 1e-9, (G, _rel(gG, g1))


@pytest.mark.skipif(jax.device_count() < 2, reason="needs 2 devices (xla_force_host_platform_device_count=4)")
def test_training_on_a_mesh_of_two_compiles_nothing_after_step_0():
    step = [-1]
    with compiles(lambda: step[0] >= 0) as seen:
        res = tr.train(_ks(shd.make_mesh(2)), _model(), steps=6, refresh_every=2, monitor=None,
                       guard=LOOSE, step_cb=lambda m: step.__setitem__(0, m["step"]))
    assert res.n_steps == 6 and not seen, seen[:5]


def test_the_chem_init_breaks_ties_by_index():
    from gs_dft.basis.chem import _largest_remainder
    assert _largest_remainder(np.ones(5), 3).tolist() == [1, 1, 1, 0, 0]
