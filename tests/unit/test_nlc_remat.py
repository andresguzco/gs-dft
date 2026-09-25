"""The analytic VV10 backward pass in ``gs_dft/ks/nlc.py`` matches the engine's VV10.

The forward is identical to ``dftax.energy.vv10``, and every cotangent, including the one with
respect to the grid coordinates that nuclear forces need, matches autodiff. The custom backward
exists because reverse mode through the streamed pair quadrature would store an O(ng^2) tensor.
"""
import jax
import jax.numpy as jnp
import pytest

from dftax.energy.vv10 import vv10_energy as engine_vv10
from gs_dft.ks.nlc import vv10_energy as ours

B, C = 6.0, 0.01          # ωB97M-V / ωB97X-V parameters


def _synthetic(ng, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    coords = jax.random.normal(k1, (ng, 3)) * 2.0
    rho = jnp.abs(jax.random.normal(k2, (ng,))) * 0.3 + 1e-9
    gnorm2 = jnp.abs(jax.random.normal(k3, (ng,))) * 0.1
    return rho, gnorm2, coords, jnp.full((ng,), 0.01)


@pytest.mark.parametrize("ng", [1024, 4096])          # one sub-chunk, one multi-chunk
def test_forward_matches_engine_bitwise(ng):
    rho, gnorm2, coords, w = _synthetic(ng)
    a = engine_vv10(rho, gnorm2, coords, w, b=B, c=C)
    b = ours(rho, gnorm2, coords, w, b=B, c=C)
    assert float(a) == float(b), "the transcription drifted from the engine"


@pytest.mark.parametrize("arg,name", [(0, "rho"), (1, "gnorm2"), (2, "coords"), (3, "weights")])
def test_every_cotangent_matches_autodiff(arg, name):
    """Each of the four, separately, so a failure names the term that is wrong.

    The reference is autodiff through the ENGINE's forward — an independent implementation of the
    same function, so this checks the derivation, not just self-consistency.
    """
    rho, gnorm2, coords, w = _synthetic(1536, seed=1)
    ref = jax.grad(lambda *a: engine_vv10(*a, b=B, c=C), argnums=arg)(rho, gnorm2, coords, w)
    got = jax.grad(lambda *a: ours(*a, b=B, c=C), argnums=arg)(rho, gnorm2, coords, w)
    scale = float(jnp.max(jnp.abs(ref)))
    rel = float(jnp.max(jnp.abs(ref - got))) / scale
    assert rel < 1e-10, f"d/d{name} disagrees with autodiff by {rel:.2e} (relative)"


def test_coordinate_cotangent_is_not_zero():
    """Guard the easy silent failure: returning zeros for `coords` would pass nothing else here.

    VV10 is differentiable in the grid coordinates because the Becke grid moves with the nuclei —
    that is how forces pick the term up. A zero there is a wrong force, not a missing feature.
    """
    rho, gnorm2, coords, w = _synthetic(1024, seed=2)
    g = jax.grad(lambda cc: ours(rho, gnorm2, cc, w, b=B, c=C))(coords)
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_reverse_does_not_grow_with_the_chunk_count():
    """A saved-residual scan's live set grows with the number of chunks; a streamed one does not."""
    ng = 8192
    rho, gnorm2, coords, w = _synthetic(ng, seed=3)
    dev = jax.devices()[0]

    # `memory_stats()` is None on backends that do not report it (CPU), and this test asserts a
    # property OF device memory -- there is nothing to measure there, so skip rather than pass
    # vacuously.
    if dev.memory_stats() is None:
        pytest.skip(f"{dev.platform} reports no memory_stats; this bound is device-memory only")

    def peak(chunk):
        f = jax.jit(jax.grad(lambda r, g: ours(r, g, coords, w, b=B, c=C, chunk=chunk)))
        f(rho, gnorm2).block_until_ready()
        return dev.memory_stats()["peak_bytes_in_use"]

    small, large = peak(2048), peak(512)          # 4x the chunks
    assert large <= small * 1.1, "the reverse pass is holding something per-chunk"
