"""The sharded step equals the single-device screened step, value and gradient, on a multi-device
mesh. Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``; skipped otherwise.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from gs_dft import SplatKS, df, init_model, refresh
from gs_dft.ks import shard as shd
from gs_dft.ks.terms import pairlist
from gs_dft.ks.train import native_grid, _refresh_state
from dftax.energy.xc import PBE


@pytest.mark.skipif(jax.device_count() < 2, reason="needs a multi-device mesh (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
def test_sharded_step_matches_single_device():
    from gs_dft.benchmark import build_reference
    mol, _, _, _ = build_reference("h2o", "cc-pvdz", "pbe", 1)
    model = init_model(mol, 30, jr.PRNGKey(0))
    grid = native_grid(mol, 1, chunk=512)
    ks1 = SplatKS(mol, PBE(), grid=grid, coulomb=df(), screen=pairlist(eps=1e-6))
    mesh = shd.make_mesh()
    ksm = SplatKS(mol, PBE(), grid=grid, coulomb=df(), screen=pairlist(eps=1e-6), mesh=mesh)
    G = int(mesh.size)
    from gs_dft import screening as scr
    n0 = int(scr.neighbor_pairs(model.basis, eps=1e-6)[0].shape[0])
    pad = -(-int(2 * n0) // G) * G
    st1 = refresh(ks1, model.basis, pad_to=pad)
    stm = _refresh_state(ksm, model.basis, pad, False)

    def e1(m):
        return ks1(m, st1)[0]

    def em(m):
        return ksm(m, stm)[0]

    E1, Em = float(e1(model)), float(em(model))
    np.testing.assert_allclose(Em, E1, rtol=0, atol=1e-9)
    import equinox as eqx
    g1 = eqx.filter_grad(e1)(model)
    gm = eqx.filter_grad(em)(model)
    for a, b in zip(jax.tree.leaves(eqx.filter(gm, eqx.is_array)), jax.tree.leaves(eqx.filter(g1, eqx.is_array))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("G", [1, 2, 4])
def test_the_row_dealt_metric_solve_matches_cho_solve(G, monkeypatch):
    """E_J's metric tail with the factor's rows dealt over G devices (padded to whole blocks; 64-row
    blocks here, so each device owns several) against cho_solve on the whole factor: value, gradient
    in γ, and the gradient in the factor (on-graph forces differentiate it)."""
    if jax.device_count() < G:
        pytest.skip(f"needs {G} devices (xla_force_host_platform_device_count=4)")
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    from gs_dft.coulomb import ri
    monkeypatch.setattr(ri, "SOLVE_BLOCK", 64)
    N = 300
    k = jr.split(jr.PRNGKey(3), 3)
    Q, _ = jnp.linalg.qr(jr.normal(k[0], (N, N)))
    V = (Q * jnp.logspace(-7, 0, N)) @ Q.T                      # a metric-like spread, κ = 1e7
    L = jnp.linalg.cholesky(0.5 * (V + V.T) + 1e-8 * jnp.eye(N))
    gamma = jr.normal(k[1], (N,))
    mesh = shd.make_mesh(G)
    Np = ri.factor_rows(N, G)
    rows = shard_map(lambda g, Lr: ri.df_coulomb_solve_rows(g, Lr, "g"), mesh=mesh,
                     check_vma=False, in_specs=(P(), P("g")), out_specs=P())

    def new(g, L_):
        return rows(g, ri.pad_factor(L_, Np))

    def ref(g, L_):
        return ri.df_coulomb_solve(g, None, cholV=L_)

    np.testing.assert_allclose(float(new(gamma, L)), float(ref(gamma, L)), rtol=1e-11)
    for a, b in zip(jax.grad(new, argnums=(0, 1))(gamma, L), jax.grad(ref, argnums=(0, 1))(gamma, L)):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9 * np.max(np.abs(b)))


def _aux(N, key):
    """N isotropic s-Gaussians spread over a 10-bohr box: a metric-like, well-posed V."""
    from types import SimpleNamespace
    k = jr.split(key, 2)
    return SimpleNamespace(alpha=jnp.exp(jr.uniform(k[0], (N,), minval=-1.0, maxval=1.5)),
                           centers=10.0 * jr.uniform(k[1], (N, 3)), norm=jnp.ones(N))


@pytest.fixture
def small_factor_blocks(monkeypatch):
    """64-row blocks and 3 phases (several blocks per device, unequal windows), with the traced
    factorization dropped before and after, since its block size is read at trace time."""
    from gs_dft.coulomb import ri
    monkeypatch.setattr(ri, "SOLVE_BLOCK", 64)
    monkeypatch.setattr(ri, "FACTOR_PHASES", 3)
    ri._factor_rows.clear_cache()
    yield ri
    ri._factor_rows.clear_cache()


@pytest.mark.parametrize("G", [1, 2, 4])
def test_the_row_built_factor_is_the_metric_cholesky(G, small_factor_blocks):
    """ri.metric_factor on a mesh of G devices: exactly lower triangular, L Lᵀ = V + λI to roundoff,
    the pad's √λ I tail, and E_J's metric tail equal to cho_solve with the one-device factor (to the
    metric's conditioning: two correct factors differ by roundoff times κ)."""
    if jax.device_count() < G:
        pytest.skip(f"needs {G} devices (xla_force_host_platform_device_count=4)")
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    ri = small_factor_blocks
    N, lam = 300, 1e-8
    aux = _aux(N, jr.PRNGKey(5))
    mesh = shd.make_mesh(G)
    Lr = ri.metric_factor(aux, lam, mesh)
    L = np.asarray(Lr)
    assert L.shape == (ri.factor_rows(N, G),) * 2
    assert np.all(np.triu(L, 1) == 0.0)
    A = np.asarray(ri.shift_diag(ri.aux_metric(aux), lam))
    assert np.linalg.norm(L[:N, :N] @ L[:N, :N].T - A) / np.linalg.norm(A) < 1e-14
    tail = L[N:, :]
    assert np.all(tail[:, :N] == 0.0)
    np.testing.assert_allclose(tail[:, N:], np.sqrt(lam) * np.eye(L.shape[0] - N), rtol=1e-15, atol=0)

    gamma = jr.normal(jr.PRNGKey(6), (N,))
    rows = shard_map(lambda g, L_: ri.df_coulomb_solve_rows(g, L_, "g"), mesh=mesh,
                     check_vma=False, in_specs=(P(), P("g")), out_specs=P())
    e_new = float(rows(gamma, Lr))
    e_ref = float(ri.df_coulomb_solve(gamma, None, cholV=ri.aux_cholesky(aux, lam)))
    kappa = float(np.linalg.cond(A))
    np.testing.assert_allclose(e_new, e_ref, rtol=max(1e-13, 50 * kappa * np.finfo(float).eps))


def test_the_row_built_factor_keeps_one_copy_of_its_rows(small_factor_blocks, monkeypatch):
    """Compiled per device, the build and factorization hold this device's rows of V once (in place)
    plus the panel, one update slab and the build's row blocks, never a second (nl, Np) copy. Compile
    only: the size where one copy and two copies are told apart is too slow to run on CPU."""
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices (xla_force_host_platform_device_count=4)")
    ri = small_factor_blocks
    N, G = 8000, 4
    Np = ri.factor_rows(N, G)
    nl, c = Np // G, 16
    monkeypatch.setattr(ri, "_AUX_BLOCK_BUDGET", 3 * Np * c)       # the build in 16-row blocks
    q = (jnp.ones(Np), jnp.zeros((Np, 3)), jnp.ones(Np), jnp.asarray(1e-8))
    mem = ri._factor_rows.lower(*q, mesh=shd.make_mesh(G)).compile().memory_analysis()
    used = mem.temp_size_in_bytes + mem.output_size_in_bytes
    slab = 64 * 4
    one_copy = (nl + slab + 64 + 16 * c) * Np * 8
    assert used < 1.25 * one_copy, (used / 2**20, one_copy / 2**20, (2 * nl * Np * 8) / 2**20)


@pytest.mark.parametrize("G", [2, 4])
def test_the_split_optimizer_state_takes_the_same_steps(G):
    """train.zero_sharded (the trainer's clipped Adam with its moments split over a mesh of G) makes
    the plain optimizer's updates to roundoff, from moments each device holds 1/G of, on leaves whose
    sizes do not divide by G."""
    if jax.device_count() < G:
        pytest.skip(f"needs {G} devices (xla_force_host_platform_device_count=4)")
    import optax
    from jax.sharding import NamedSharding, PartitionSpec as P
    import importlib
    tr = importlib.import_module("gs_dft.ks.train")    # the module: gs_dft.ks.train is the verb
    mesh = shd.make_mesh(G)
    k = jr.split(jr.PRNGKey(7), 2)
    params = jax.device_put({"a": jr.normal(k[0], (37, 5)), "b": jr.normal(k[1], (13,))},
                            NamedSharding(mesh, P()))
    opt = tr.splat_adam(1e-2, 50)
    zopt = tr.zero_sharded(opt, mesh)
    s, zs = opt.init(params), tr._opt_initializer(zopt, mesh, params)(params)
    moments = [x for x in jax.tree.leaves(zs) if x.ndim]
    assert moments and all(x.sharding.spec == P("g") and x.addressable_shards[0].data.shape[0] * G == x.shape[0]
                           for x in moments)
    up, zup = jax.jit(opt.update), jax.jit(zopt.update)
    p = zp = params
    for t in range(3):
        g = jax.tree.map(lambda x: jnp.sin((t + 1) * x), p)
        u, s = up(g, s, p)
        zu, zs = zup(g, zs, zp)
        for a, b in zip(jax.tree.leaves(zu), jax.tree.leaves(u)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-16)
        p, zp = optax.apply_updates(p, u), optax.apply_updates(zp, zu)
