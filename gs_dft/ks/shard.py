"""Multi-GPU single-system sharding for the screened splat KS step.

We do NOT partition the splat *parameter* buffer across GPUs — we **replicate the params**
(``basis``, ``C`` — only O(M·n_occ), ~0.5 GB at insulin) and **shard the WORK** on one device
axis ``"g"``:

  - XC over GRID BLOCKS         (each device sums its block slice; ``psum`` the partial E_xc)
  - 1e energies over PAIRS      (``psum`` the partial E_kin, E_ext)
  - Coulomb γ over PAIRS        (``psum`` the partial γ), then the (N_aux,N_aux) metric solve
                                ½γᵀ(V+λI)⁻¹γ runs REPLICATED on the psum'd γ
  - E_nn replicated; the Löwdin eigh (replicated, ~0.5%) is the caller's, done before this.

Autodiff flows through every ``psum`` (its transpose is ``psum``), so the sharded value AND
gradient equal the single-device reference to ~machine precision — the property gated by
the in-suite 1-device shard_map parity test.
"""
from functools import partial

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P
from jax import shard_map
from jax.experimental.multihost_utils import host_local_array_to_global_array

from gs_dft.integrals.dense import overlap_times_rows, kinetic_times_rows
from gs_dft.ks.orthonormalize import transform as _lowdin_transform
from gs_dft.screening.grid import chunked_density_and_grad
from gs_dft.screening.cells import cell_density_and_grad
from gs_dft.screening import screened_external_energy, density_pairs
from gs_dft.coulomb.ri import df_coulomb_gamma_screened, df_coulomb_solve
from gs_dft.coulomb.stream import full_eri_block_schur, _chunked_pairs
from dftax.energy.grid import xc_energy
from dftax.energy.xc import PBE
from dftax.integrals.nuclear_repulsion import nuclear_repulsion

__all__ = ["init_distributed", "make_mesh", "to_global", "tree_to_global", "pad_pairs_to_multiple",
           "stripe_pairs",
           "sharded_screened_energy", "row_partition", "sync_replicas"]


def sync_replicas(tree, mesh):
    """Force every device's copy of a REPLICATED pytree back into agreement (``pmean``).

    Required, not defensive. The dense one-electron path (``overlap_matrix``, ``kinetic_matrix``,
    ``regeigh``) runs OUTSIDE ``shard_map`` on replicated arrays, so XLA computes it redundantly per
    device. ``regeigh`` eigendecomposes a near-degenerate overlap, so a last-bit difference between
    devices becomes a large ``C_orth`` difference, hence different gradients and updates, and the
    replicas separate geometrically. A refresh then freezes the mismatch into the frozen aux.

    Cost: one all-reduce of the params per call, negligible against the step.
    """
    if mesh is None:
        return tree
    leaves, treedef = jax.tree.flatten(eqx.filter(tree, eqx.is_inexact_array))

    @partial(shard_map, mesh=mesh, check_vma=False,
             in_specs=(tuple(P() for _ in leaves),), out_specs=tuple(P() for _ in leaves))
    def _mean(xs):
        return tuple(lax.pmean(x, "g") for x in xs)

    synced = _mean(tuple(leaves))
    return eqx.combine(jax.tree.unflatten(treedef, list(synced)), tree)


def init_distributed():
    """Join the multi-controller cluster from SLURM's environment, once, before any JAX call.

    Returns True when this process is part of a multi-node run. Without this every ``multinode``
    branch in the trainer is unreachable: ``jax.process_count()`` is 1 on a single controller no
    matter how many nodes SLURM allocated, so the code takes the single-process path and each node
    silently optimizes its own copy.

    A single-task allocation is not an error: SLURM sets ``SLURM_NTASKS=1`` for every ordinary job,
    and this returns False so the caller keeps the single-process path."""
    import os
    n = int(os.environ.get("SLURM_NTASKS", "1"))
    if n <= 1:
        return False
    jax.distributed.initialize(
        coordinator_address=f"{os.environ['SLURM_LAUNCH_NODE_IPADDR']}:{os.environ.get('DFTAX_COORD_PORT', '29500')}",
        num_processes=n,
        process_id=int(os.environ["SLURM_PROCID"]),
        # one process per GPU: all of the node's devices are visible, this process owns SLURM_LOCALID
        local_device_ids=[int(os.environ["SLURM_LOCALID"])] if "SLURM_LOCALID" in os.environ and len(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) > 1 else list(range(len(
            [i for i in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if i != ""]))),
        initialization_timeout=1800,
    )
    jax.config.update("jax_cpu_get_local_topology_timeout_minutes", 10)
    jax.config.update("jax_cpu_get_global_topology_timeout_minutes", 30)   # default 5: the one that fired
    jax.devices()            # backend init is collective; do it before any uneven host work
    return True


def make_mesh(n=None):
    """1-D device mesh over axis ``"g"`` of the first ``n`` of ``jax.devices()`` (default: all). In a
    multi-controller (multi-node) run ``jax.devices()`` is the GLOBAL device list, so the same call
    spans every node."""
    devs = jax.devices()
    n = len(devs) if n is None else int(n)
    return Mesh(np.array(devs[:n]), ("g",))


def to_global(x, mesh, sharded=False):
    """Promote a per-process array to a GLOBAL ``jax.Array`` for multi-controller (multi-node) runs.

    ``sharded=True`` shards dim 0 over axis ``"g"``: the array is assumed IDENTICAL on every process
    (deterministic host build) and each process contributes its contiguous slice. ``sharded=False``
    replicates. Non-arrays pass through. Callers must guard with ``jax.process_count() > 1`` — on a
    single process this would needlessly round-trip through the host."""
    if not hasattr(x, "shape"):
        return x
    if sharded:
        npr = jax.process_count()
        per = x.shape[0] // npr
        pid = jax.process_index()
        return host_local_array_to_global_array(x[pid * per:(pid + 1) * per], mesh, P("g"))
    return host_local_array_to_global_array(x, mesh, P())


def tree_to_global(tree, mesh):
    """Replicate every array leaf of a pytree (e.g. an Equinox model / aux) as a GLOBAL ``jax.Array``;
    static / non-array leaves pass through. Multi-controller only."""
    return jax.tree.map(lambda a: to_global(a, mesh, False), tree)


def _roundup(n, g):
    return -(-int(n) // int(g)) * int(g)


def pad_pairs_to_multiple(pi, pj, valid, mult):
    """Pad the pair axis to a multiple of ``mult`` with valid=False entries (q→0 ⇒ no contribution;
    pairs never reach the GGA functional, so no NaN risk) — required for even ``P("g")`` sharding."""
    n = pi.shape[0]
    vv = jnp.ones(n, bool) if valid is None else valid
    pad = (-n) % mult
    if pad == 0:
        return pi, pj, vv
    z = lambda a, fill: jnp.concatenate([a, jnp.full(pad, fill, a.dtype)])
    return z(pi, 0), z(pj, 0), jnp.concatenate([vv, jnp.zeros(pad, bool)])


def stripe_pairs(pi, pj, valid, G):
    """Reorder a padded pair list so each of the ``G`` contiguous device blocks holds the SAME
    valid:dummy ratio, with its valid entries first.

    ``P("g")`` splits the pair axis into contiguous blocks, and the pad is APPENDED, so a plain
    split hands the last devices nothing but dummies. That is harmless while every device still
    walks every chunk, but with ``skip_pad`` those devices finish early and sit at the next
    collective, where they can exceed the collective timeout.

    Dealing pair ``i`` to device ``i % G`` fixes both halves at once. Writing ``n = G·d``, the map
    ``new[g·d + j] = old[j·G + g]`` is exactly ``reshape(d, G).T.reshape(-1)``; old index ``j·G+g``
    is valid iff it is ``< n_valid``, so block ``g`` gets ``⌈(n_valid-g)/G⌉`` valid entries — even to
    within one — and they land contiguously at the block's front, so whole trailing chunks are still
    skippable. Every consumer of the list (γ, E_ext, the exact-ERI path) is an order-independent sum
    over pairs, so the energy is unchanged; only the work distribution moves.
    """
    n = pi.shape[0]
    if G <= 1 or n % G:
        return pi, pj, valid
    f = lambda a: a.reshape(-1, G).T.reshape(-1)
    return f(pi), f(pj), f(valid)


def sharded_screened_energy(basis, C, occ, aux, coords, charges, pairs, gp, gw, *,
                            mesh, rows, xc=None, gga=True, lam=1e-8, cholV=None,
                            hartree="df", bra_chunk=128, grid_chunk=4096, cells=None,
                            skip_pad=False, gamma_chunk=2048):
    """Full screened KS energy with every reduction sharded over ``mesh`` axis ``"g"``. ``C`` is the
    RAW (replicated) coefficient matrix: the one-electron work is sharded too — each device forms
    its row slice of ``S·C`` and ``T·C`` (``rows`` = ``(index, valid)`` sharded over ``"g"``), the
    occupied Gram is ``psum``'d, and the Löwdin retraction runs replicated on that small matrix. Before
    this every device recomputed the whole O(M²·n_occ) one-electron path. The grid points ``gp`` /
    weights ``gw`` are sharded over the device grid — each device runs **2a chunked density** on its
    slice → partial E_xc → psum (O(grid_chunk·M) memory, grid-size-independent). ``pairs`` =
    (pi, pj, valid), pair count a multiple of G.

    ``hartree="df"`` routes E_J through the frozen product RI-J aux (γ pair-sharded, metric
    solve replicated); ``gamma_chunk`` is that stream's pair chunk, and it sets the step's PEAK
    MEMORY — the per-chunk temporaries are (N_aux, chunk) and (N_aux, chunk, 3). ``hartree="exact"`` is the density-fitting-FREE path: the EXACT streamed ERI with the
    bra pairs sharded (this device's slice) and the ket pairs ALL-GATHERED, each device contributing
    ½ Σ_{a∈slice} Jvec_a P_a and the partial E_J psum'd — O(M) memory / O(M²) compute, no aux, no
    O(N_aux³) metric. Differentiable in (basis, C_orth); equals the single-device screened energy to
    ~machine precision."""
    xc = xc if xc is not None else PBE()
    grid_spec = (P("g"), P("g"))
    pairs_spec = (P("g"), P("g"), P("g"))
    rows_spec = (P("g"), P("g"))
    # Every cell array carries a leading DEVICE axis, so it shards on "g" exactly like the grid it
    # indexes into — each device then holds the cells for its own contiguous grid slice, with
    # point indices already local to that slice.
    cells_spec = tuple((P("g"),) * 4 for _ in (cells or ()))

    @partial(shard_map, mesh=mesh, check_vma=False, out_specs=(P(),) * 6,
             in_specs=(P(), P(), grid_spec, pairs_spec, cells_spec, rows_spec))
    def _energy(basis, Craw, grid, pairs, cells_, rows_):
        # Returns (E_total, E_kin, E_ext, E_hartree, E_xc, nelec) — replicated scalars (psum'd).
        gp_, gw_ = grid                                         # this device's grid-point slice
        pi, pj, vm = pairs                                      # this device's pair slice
        ridx, rok = rows_                                       # this device's rows of S/T
        # One-electron path, sharded by rows: partial Grams psum'd, retraction replicated.
        rmask = rok.astype(Craw.dtype)[:, None]
        Cr = Craw[ridx] * rmask
        gram = lax.psum(Cr.T @ overlap_times_rows(basis, Craw, ridx), "g")        # (n_occ, n_occ)
        C = Craw @ _lowdin_transform(gram)                                          # Löwdin, replicated
        Ekin = lax.psum(occ @ jnp.sum((C[ridx] * rmask) * kinetic_times_rows(basis, C, ridx), axis=0), "g")
        # XC on this device's grid slice. Block-sparse cells when built (6.2× on the density term
        # at ala_15), else the 2a chunked dense path (remat internal → O(grid_chunk·M) memory).
        # shard_map leaves a length-1 device axis on each cell array; drop it to get this device's
        # own cells, whose point indices are already local to `gp_`.
        if cells_:
            local = tuple((a[0], b[0], c[0], d[0]) for a, b, c, d in cells_)
            rho, grad = cell_density_and_grad(basis, C, occ, gp_, local)
        else:
            rho, grad = chunked_density_and_grad(basis, C, occ, gp_, chunk=grid_chunk)
        rho_c = jnp.maximum(rho, 1e-30)
        Exc = xc_energy(xc, rho_c, gw_, grad_rho=grad if gga else None)
        nel_part = jnp.dot(gw_, rho_c)                          # integrated electron count (this slice)
        ex = screened_external_energy(basis, C, occ, pi, pj, coords, charges, vm, skip_pad=skip_pad)
        Enn = nuclear_repulsion(coords, charges)               # replicated constant
        if hartree == "exact":                                 # DF-FREE exact streamed ERI
            pif = lax.all_gather(pi, "g", tiled=True)           # full ket pair list on EVERY device
            pjf = lax.all_gather(pj, "g", tiled=True)
            vmf = lax.all_gather(vm, "g", tiled=True)
            Pw_full = vmf.astype(C.dtype) * density_pairs(C, occ, pif, pjf)   # full ket densities
            Pw_loc = vm.astype(C.dtype) * density_pairs(C, occ, pi, pj)        # local bra densities
            pis, pjs, n = _chunked_pairs(pi, pj, bra_chunk)     # chunk the LOCAL bra pairs

            def _body(c):                                       # (chunk local bra) × (full ket)
                return full_eri_block_schur(basis, c[0], c[1], pif, pjf) @ Pw_full
            jv = lax.map(jax.checkpoint(_body), (pis, pjs)).reshape(-1)[:n]   # Jvec on local bra
            Ej_part = 0.5 * jnp.sum(jv * Pw_loc)                # E_J = Σ_g ½ Σ_{a∈slice} Jvec_a P_a
            Exc, ex, nel, Ej = lax.psum((Exc, ex, nel_part, Ej_part), "g")
            return Ekin + ex + Ej + Exc + Enn, Ekin, ex, Ej, Exc, nel
        gamma = df_coulomb_gamma_screened(basis, C, occ, aux, pi, pj, valid=vm, skip_pad=skip_pad,
                                          chunk=gamma_chunk)
        Exc, ex, nel, gamma = lax.psum((Exc, ex, nel_part, gamma), "g")
        Ej = df_coulomb_solve(gamma, aux, lam=lam, cholV=cholV)  # replicated metric tail
        return Ekin + ex + Ej + Exc + Enn, Ekin, ex, Ej, Exc, nel

    return _energy(basis, C, (gp, gw), pairs, tuple(cells or ()), rows)


def sharded_hf_energy(basis, C, occ, coords, charges, pairs, *, mesh, rows, skip_pad=False):
    """E_ext + E_nn only, sharded over ``mesh`` axis ``"g"`` -- the two terms of the total energy
    that depend on the NUCLEAR COORDINATES.

    The Löwdin retraction is reproduced exactly as ``sharded_screened_energy`` does it, because
    ``E_ext`` is a trace against the ORTHONORMALIZED coefficients; using ``C`` raw would silently
    change the density it is contracted with."""
    pairs_spec = (P("g"), P("g"), P("g"))
    rows_spec = (P("g"), P("g"))

    @partial(shard_map, mesh=mesh, check_vma=False, out_specs=P(),
             in_specs=(P(), P(), pairs_spec, rows_spec))
    def _hf(basis, Craw, pairs, rows_):
        pi, pj, vm = pairs
        ridx, rok = rows_
        rmask = rok.astype(Craw.dtype)[:, None]
        Cr = Craw[ridx] * rmask
        gram = lax.psum(Cr.T @ overlap_times_rows(basis, Craw, ridx), "g")
        C_ = Craw @ _lowdin_transform(gram)
        ex = screened_external_energy(basis, C_, occ, pi, pj, coords, charges, vm,
                                      skip_pad=skip_pad)
        return lax.psum(ex, "g") + nuclear_repulsion(coords, charges)

    return _hf(basis, C, pairs, rows)


def row_partition(M, G):
    """``(index, valid)`` over the M rows of S/T, padded to a multiple of the mesh size ``G`` so the
    row axis shards evenly; padding rows point at row 0 and are masked out."""
    Mp = -(-int(M) // int(G)) * int(G)
    idx = jnp.concatenate([jnp.arange(M, dtype=jnp.int32), jnp.zeros(Mp - M, jnp.int32)])
    ok = jnp.concatenate([jnp.ones(M, bool), jnp.zeros(Mp - M, bool)])
    return idx, ok
