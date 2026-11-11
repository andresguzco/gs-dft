"""The grid density by Morton cells: against the dense path, compile once across refreshes, a list
overflow raises, and the mesh at every G.

The mesh tests need several devices: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from dftax.energy.xc import PBE
from dftax.system import Molecule
from gs_dft import SplatKS, df, init_model, refresh
from gs_dft import screening as scr
from gs_dft.geometries import h2o_geometry
from gs_dft.ks import shard as shd
from gs_dft.ks.terms import pairlist, cellgrid
from gs_dft.screening import cells as cl
from tests.compiles import compiles

tr = importlib.import_module("gs_dft.ks.train")
LOOSE = tr.collapse(nelec_tol_per_atom=1e-2)


def _mol():
    return Molecule.from_xyz(h2o_geometry, "cc-pvdz", unit="angstrom", spherical=True)


def _ks(mesh=None, cells=cellgrid()):
    mol = _mol()
    return SplatKS(mol, PBE(), grid=tr.native_grid(mol, 1, chunk=512), coulomb=df(),
                   screen=pairlist(eps=1e-6), mesh=mesh, cells=cells)


def _model(seed=0):
    return init_model(_mol(), 30, jr.PRNGKey(seed))


@eqx.filter_jit
def _vg(model, ks, st):
    return eqx.filter_value_and_grad(lambda m: ks(m, st)[0])(model)


def _value_and_grad(ks, model, pad):
    E, g = _vg(model, ks, tr._refresh_state(ks, model.basis, pad, False))
    return float(E), np.concatenate([np.ravel(np.asarray(x)) for x in
                                     jax.tree.leaves(eqx.filter(g, eqx.is_inexact_array))])


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def _pad(model):
    return 4 * scr.count_pairs(model.basis, eps=1e-6, unique=True)     # a multiple of every G


def _host_lists(splats, partition, eps, layout=None, headroom=2.0, margin=64, n_sp_buckets=8,
                align=8):
    """The host build the device build replaced (cells.cell_lists before rb-par), as the reference."""
    from gs_dft.screening.grid import _host_rcut
    mu, rcut = np.asarray(splats.centers), np.asarray(_host_rcut(splats, eps))
    M = mu.shape[0]
    width = lambda n: cl._align(min(M, int(np.ceil(max(1, n) * headroom)) + margin), align)
    todo = None if layout is None else list(layout)
    groups = []
    for (pi, pm), (c, R) in zip(partition, partition.geom):
        ndev, nc = pi.shape[:2]
        d2 = sum((mu[None, :, a] - c.reshape(-1, 3)[:, None, a]) ** 2 for a in range(3))
        keep = (np.sqrt(d2) <= R.reshape(-1)[:, None] + rcut[None]).reshape(ndev, nc, M)
        keep &= pm.any(-1)[..., None]
        cnt = keep.sum(-1)
        order = np.argsort(cnt, axis=1, kind="stable")
        if todo is None:
            shapes = [(n, None) for n in cl._cut(np.sort(cnt, axis=1).max(0), n_sp_buckets, width)]
        else:
            shapes, tot = [], 0
            while tot < nc:
                shapes.append(todo.pop(0))
                tot += shapes[-1][0]
        lo = 0
        for n, S in shapes:
            sel = order[:, lo:lo + n]
            lo += n
            n_sel = np.take_along_axis(cnt, sel, 1)
            S = width(int(n_sel.max())) if S is None else int(S)
            k_sel = np.take_along_axis(keep, sel[..., None], 1).reshape(-1, M)
            rows, cols = np.nonzero(k_sel)
            pos = np.arange(rows.size) - np.repeat(np.cumsum(n_sel.ravel()) - n_sel.ravel(),
                                                   n_sel.ravel())
            si = np.zeros((ndev * n, S), np.int32)
            si[rows, pos] = cols
            sm = np.arange(S)[None] < n_sel.reshape(-1, 1)
            groups.append((np.take_along_axis(pi, sel[..., None], 1),
                           np.take_along_axis(pm, sel[..., None], 1),
                           si.reshape(ndev, n, S), sm.reshape(ndev, n, S)))
    return groups


@pytest.mark.parametrize("ndev", [1, 2, 4])
def test_the_device_lists_equal_the_host_build(ndev):
    """Every list array equal to the host build's, at the first build and at a pinned layout."""
    ks = _ks(cells=None)
    gp = np.asarray(ks.grid_points)
    gp = gp[cl.morton_order(gp)[0]]
    gp = np.concatenate([gp, np.repeat(gp[:1], (-len(gp)) % ndev, 0)])
    part = cl.cell_partition(gp, ndev=ndev)
    for seed in (0, 1):
        splats = _model(seed).basis
        lists, layout = cl.cell_lists(splats, part, eps=1e-8)
        again, _ = cl.cell_lists(splats, part, eps=1e-8, layout=[(n, S + 8) for n, S in layout])
        for got, lay in ((lists, None), (again, [(n, S + 8) for n, S in layout])):
            want = _host_lists(splats, part, 1e-8, layout=lay)
            assert len(got) == len(want)
            for g, w in zip(got, want):
                for a, b in zip(g, w):
                    assert np.array_equal(np.asarray(a), b), (ndev, seed)


@pytest.mark.parametrize("mesh_on", [False, True])
def test_cells_match_the_dense_path(mesh_on):
    model = _model()
    mesh = shd.make_mesh(1) if mesh_on else None
    E0, g0 = _value_and_grad(_ks(mesh, cells=None), model, _pad(model))
    E1, g1 = _value_and_grad(_ks(mesh), model, _pad(model))
    print(f"\ncells vs dense: dE {E1 - E0:.1e} Ha, gradient rel {_rel(g1, g0):.1e}")
    assert abs(E1 - E0) <= 1e-8 and _rel(g1, g0) <= 1e-7


def test_dealt_cells_cover_the_grid_once_in_whole_runs():
    """``deal_order``: the identity on one device; on G, every point once, each device's block whole
    Morton runs in Morton order, and the per-device point counts within two runs of each other."""
    gp = np.asarray(_ks(cells=None).grid_points)
    gp = gp[cl.morton_order(gp)[0]]
    assert np.array_equal(cl.deal_order(gp, 1), np.arange(len(gp)))
    for G in (2, 3, 4):
        order = cl.deal_order(gp, G, cap=64).reshape(G, -1)
        live = order[order >= 0]
        assert np.array_equal(np.sort(live), np.arange(len(gp))), G
        n = (order >= 0).sum(1)
        assert n.max() - n.min() <= 2 * 64, (G, n)
        for row in order:
            r = row[row >= 0]
            assert np.all(np.diff(r) > 0) and np.all(row[len(r):] < 0), G


def test_the_lists_are_unchanged_by_their_padding():
    """headroom, margin and align only widen the lists: the energy and gradient stay."""
    model = _model()
    E0, g0 = _value_and_grad(_ks(cells=cellgrid(headroom=1.0, margin=0, align=1)), model, _pad(model))
    E1, g1 = _value_and_grad(_ks(cells=cellgrid(headroom=1.5, margin=7, align=32)), model, _pad(model))
    assert abs(E1 - E0) <= 1e-12 * abs(E0) and _rel(g1, g0) <= 1e-12


@pytest.mark.parametrize("mesh_on", [False, True])
def test_training_on_cells_compiles_nothing_after_step_0(mesh_on):
    ks = _ks(shd.make_mesh(1) if mesh_on else None)
    step = [-1]
    with compiles(lambda: step[0] >= 0) as seen:
        res = tr.train(ks, _model(), steps=8, refresh_every=2, monitor=None, guard=LOOSE,
                       step_cb=lambda m: step.__setitem__(0, m["step"]))
    assert res.n_steps == 8 and res.stop_reason is None and res.state.cells is not None
    assert not seen, seen[:5]


def test_a_cell_list_overflow_raises():
    ks, model = _ks(), _model()
    st = refresh(ks, model.basis)
    narrow = [(int(g[2].shape[1]), 1) for g in st.cells]
    with pytest.raises(cl.CellOverflow, match="headroom"):
        cl.cell_lists(model.basis, ks.cell_part, eps=ks.cell_spec.eps, layout=narrow)


def test_the_cut_minimizes_the_padded_work():
    srt = np.sort(np.r_[np.full(900, 3), np.arange(100) * 10 + 50])
    width = lambda n: n + 1
    sizes = cl._cut(srt, 4, width)
    cost = lambda sz: sum(n * width(srt[e - 1]) for n, e in zip(sz, np.cumsum(sz)))
    assert sum(sizes) == len(srt) and len(sizes) <= 4
    assert cost(sizes) <= cost([250] * 4)


def test_a_cell_list_overflow_in_training_raises_naming_the_step():
    """Widths fixed at step 0 without room; the splats widen at step 1, the refresh at step 2 raises."""
    def widen(params):
        return jnp.zeros([], jnp.int32)

    def update(g, count, params=None):
        u = jax.tree.map(jnp.zeros_like, g)
        u = eqx.tree_at(lambda m: m.basis.log_scale, u,
                        jnp.where(count == 1, -3.0, 0.0) * jnp.ones_like(u.basis.log_scale))
        return u, count + 1

    ks = _ks(cells=cellgrid(headroom=1.0, margin=0, align=1))
    with pytest.raises(cl.CellOverflow, match="step 2"):
        tr.train(ks, _model(), steps=4, refresh_every=2, monitor=None, guard=None,
                 optimizer=optax.GradientTransformation(widen, update))


@pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices (xla_force_host_platform_device_count=4)")
def test_cells_on_the_mesh_at_g_1_2_4_agree():
    model = _model()
    E1, g1 = _value_and_grad(_ks(shd.make_mesh(1)), model, _pad(model))
    for G in (2, 4):
        EG, gG = _value_and_grad(_ks(shd.make_mesh(G)), model, _pad(model))
        print(f"\ncells G={G} vs G=1: E rel {abs(EG - E1) / abs(E1):.1e}, gradient rel {_rel(gG, g1):.1e}")
        assert abs(EG - E1) <= 1e-9 * abs(E1), (G, EG, E1)
        assert _rel(gG, g1) <= 1e-9, (G, _rel(gG, g1))


@pytest.mark.skipif(jax.device_count() < 2, reason="needs 2 devices (xla_force_host_platform_device_count=4)")
def test_training_on_cells_on_a_mesh_of_two_compiles_nothing_after_step_0():
    step = [-1]
    with compiles(lambda: step[0] >= 0) as seen:
        res = tr.train(_ks(shd.make_mesh(2)), _model(), steps=8, refresh_every=2, monitor=None,
                       guard=LOOSE, step_cb=lambda m: step.__setitem__(0, m["step"]))
    assert res.n_steps == 8 and not seen, seen[:5]


def test_the_builder_takes_cells_by_size():
    """Dense below builders.CELLS_FROM_ATOMS, cells from it (penicillin, 41 atoms), never for a meta-GGA."""
    from dftax.energy.xc import R2SCAN
    from experiments.common import builders, systems
    water, pen = _mol(), systems.molecule(system="penicillin", basis="cc-pvdz")
    assert not builders.use_cells(water, PBE()) and builders.use_cells(pen, PBE())
    assert not builders.use_cells(pen, R2SCAN())                   # the cells have no τ
    assert builders.splat_ks(water, PBE(), grid_level=0, screened=True, cells=True).cell_spec is not None
    assert builders.splat_ks(water, PBE(), grid_level=0, screened=True).cell_spec is None


@pytest.mark.parametrize("budget", [cl.CHUNK_BYTES, 1 << 16])
def test_the_cell_reverse_matches_autodiff(budget):
    """The cell density's own reverse (one running total per input) against autodiff of its forward,
    in one chunk a list and in many: splats, coefficients, occupations and grid points."""
    from gs_dft.coulomb.stream import full_quantities
    ks, model = _ks(), _model()
    st = refresh(ks, model.basis)
    local = tuple((a[0], b[0], c[0], d[0]) for a, b, c, d in st.cells)
    args = (full_quantities(model.basis), model.C, model.occupations, ks.grid_points)
    k = jr.split(jr.PRNGKey(5), 2)
    n = ks.grid_points.shape[0]
    ct = (jr.normal(k[0], (n,)), jr.normal(k[1], (n, 3)))
    run = lambda f: jax.vjp(lambda q, C, o, x: f(q, C, o, x, local, budget), *args)
    (v, vjp), (v0, vjp0) = run(cl._cell_density), run(cl._cell_density.fun)
    for a, b in zip(jax.tree.leaves((v, vjp(ct))), jax.tree.leaves((v0, vjp0(ct)))):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * max(1.0, float(np.max(np.abs(b)))))
