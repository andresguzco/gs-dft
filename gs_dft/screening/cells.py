"""Block-sparse grid density: fine Morton cells, batched.

The dense path evaluates every splat at every grid point, though only the splats whose cutoff ball
reaches a point contribute. Exploiting that per (grid, splat) PAIR loses far more to lost data reuse
than it saves, which is why ``screening/grid.py``'s ``screened_density_and_grad`` is slower than
dense at every size.

This module exploits the same locality at cell granularity: fine Morton cells give tight
significant-splat lists, and GEMM efficiency comes from a batch axis over many small cells rather
than from few large blocks.

Do not merge fine cells into bigger work groups. Merging enough k=5 cells to fill a large block
spans the molecule, the significant lists degenerate, and it is worse than dense. Keep cells fine
and batch them.

**Constant shapes.** The jitted step must compile once (see
:func:`~gs_dft.screening.pairs.neighbor_pairs` for why a mid-run recompile is fatal on a
mesh), so the split here is:

* :func:`cell_partition` — depends only on the grid, which is static, so the point groupings and
  their ``P`` pads are fixed for the whole run. Built once.
* :func:`cell_lists` — depends on the splats, which move, so it is rebuilt on the refresh cadence.
  Only the significant-splat sets change; they are padded to a fixed ``S`` (chosen once with
  headroom) and overflow keeps the nearest splats, exactly as the pair list caps its overflow.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gs_dft.coulomb.stream import full_quantities
from gs_dft.screening.grid import _full_eval_block, _host_rcut

__all__ = ["morton_order", "cell_partition", "cell_lists", "cell_density_and_grad", "Partition"]


class Partition:
    """Host-side cell grouping, hashed by IDENTITY so it can ride in an ``eqx.field(static=True)``.

    ``SplatKS.cell_part`` must be static (it is a build-time property of the grid, not a traced
    value), but a bare list of numpy arrays trips equinox's "a JAX array is being set as static"
    warning and — worse — numpy arrays have no usable ``__hash__``/``__eq__`` for the jit cache key,
    so every call risks a recompile. A recompile mid-run tears down the multi-GPU CUDA graphs and
    deadlocks the collectives, which is exactly what the constant-shape contract exists to prevent.

    Identity semantics are the right ones here: the partition is consumed only by :func:`cell_lists`
    on the HOST and never traced, so structural equality would buy nothing, and one ``SplatKS`` holds
    one partition for its whole life.
    """

    __slots__ = ("groups",)

    def __init__(self, groups):
        self.groups = tuple(groups)

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.groups[i]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return other is self


def morton_order(grid_points, bits: int = 10):
    """Z-order permutation, so a contiguous slice of the grid is a compact box.

    Required: Becke grids are ordered by radial shell out to ``r_max=45 bohr``, so a contiguous
    slice is a spherical *shell* whose bounding sphere spans the molecule — every splat counts and
    the significant list degenerates to all of them. Sort the grid once at construction and carry
    weights along; the XC energy is a sum over points, so nothing needs un-permuting.
    """
    gp = np.asarray(grid_points)
    x = gp - gp.min(0)
    s = float(x.max()) or 1.0
    q = np.clip((x / s * ((1 << bits) - 1)).astype(np.int64), 0, (1 << bits) - 1)
    key = np.zeros(gp.shape[0], np.int64)
    for b in range(bits):
        for a in range(3):
            key |= ((q[:, a] >> b) & 1) << (3 * b + a)
    return np.argsort(key, kind="stable"), key


def cell_partition(grid_points, *, k: int = 5, cap: int = 1024, bits: int = 10, n_pt_buckets: int = 4,
                   group_points: int = 250_000, ndev: int = 1):
    """Static point grouping — depends only on the grid, so build it once.

    Returns ``[(pt_idx (ndev,nc,P) int32, pt_mask (ndev,nc,P) bool), ...]``, one entry per
    point-count bucket. Assumes ``grid_points`` is already in :func:`morton_order`.

    ``group_points`` caps points per emitted group: the live tensors are ``(nc, P, S)`` and its
    gradient, and the backward holds several, so an uncapped bucket OOMs. Capping bounds each launch
    without changing what is computed.

    ``cap`` splits oversized cells into contiguous Morton runs. That is not only a memory guard:
    cell occupancy is wildly skewed (k=4 ranges 1 … 44,886 points), and a sub-run's bounding sphere
    is TIGHTER than its parent cell's, so capping *improves* work-weighted occupancy (k=5:
    16.7% → 13.3%).

    **``ndev`` = the mesh size.** ``shard_map`` hands each device a CONTIGUOUS slice of the grid, and
    because the grid is Morton-sorted first, that slice is itself spatially coherent — so each device
    can carry its own cells. Indices are therefore **LOCAL to the device block** (``gp_[pi]`` inside
    the shard_map just works). Every emitted array carries a leading device axis so the same layout
    serves both paths: sharded, ``shard_map`` slices it; single-device, ``ndev=1`` and the caller
    takes ``[0]``.

    The device axis forces uniform shapes: buckets are cut on global quantiles (so every device
    agrees on which bucket a run belongs to) and ``nc`` is padded to the max over devices. Without
    that, devices would trace different shapes and ``shard_map`` would reject the call.
    """
    gp = np.asarray(grid_points)
    G = gp.shape[0]
    if G % ndev:
        raise ValueError(f"cell_partition: grid {G} not a multiple of ndev={ndev} "
                         "(SplatKS pads the grid to the mesh size before partitioning).")
    blk = G // ndev
    _, key = morton_order(gp, bits)
    cell = key >> np.int64(3 * (bits - k))

    # runs, per device, in LOCAL indices. Sorting by the cell prefix within a block is stable, so
    # runs stay contiguous Morton spans; a cell straddling a device boundary is simply cut there.
    runs_dev = []
    for d in range(ndev):
        c_blk = cell[d * blk:(d + 1) * blk]
        order = np.argsort(c_blk, kind="stable")
        _, starts = np.unique(c_blk[order], return_index=True)
        rs = []
        for grp in np.split(order, starts[1:]):
            for s0 in range(0, len(grp), cap):
                rs.append(grp[s0:s0 + cap])
        runs_dev.append(rs)

    npts_all = np.array([len(r) for rs in runs_dev for r in rs])
    edges = np.unique(np.quantile(npts_all, np.linspace(0, 1, n_pt_buckets + 1)[1:-1])) \
        if n_pt_buckets > 1 else np.array([])
    sel_dev = [np.digitize([len(r) for r in rs], edges) for rs in runs_dev]

    out = []
    for b in sorted(set(np.digitize(npts_all, edges).tolist())):
        picks = [np.flatnonzero(sd == b) for sd in sel_dev]
        P = max((len(runs_dev[d][i]) for d in range(ndev) for i in picks[d]), default=1)
        P = max(int(P), 1)
        nc_tot = max((len(p) for p in picks), default=0)     # pad every device to the widest
        if nc_tot == 0:
            continue
        nc_cap = max(1, int(group_points // P))
        for lo in range(0, nc_tot, nc_cap):
            nc = min(nc_cap, nc_tot - lo)
            pi = np.zeros((ndev, nc, P), np.int32)
            pm = np.zeros((ndev, nc, P), bool)
            for d in range(ndev):
                for r_, i in enumerate(picks[d][lo:lo + nc]):
                    run = runs_dev[d][i]
                    pi[d, r_, :len(run)] = run
                    pm[d, r_, :len(run)] = True
            out.append((pi, pm))
    return Partition(out)


def cell_lists(splats, grid_points, partition, *, eps: float = 1e-8, s_pad=None,
               headroom: float = 1.5, n_sp_buckets: int = 4):
    """Per-cell significant splats at FIXED width — rebuilt on the refresh cadence.

    ``s_pad`` (a list, one width per partition bucket) pins the shape for the whole run; pass the
    previous one back to keep the step compiled once. When a cell exceeds its width the NEAREST
    splats are kept, mirroring the pair list's overflow policy — it drops only the marginal tail.

    Carries :func:`cell_partition`'s leading DEVICE axis through: point indices are local to a
    device block, splat indices are global (the splats are replicated, only the grid is sharded).
    ``S`` is the max over every device AND cell, so all devices trace one shape.
    """
    gp = np.asarray(grid_points)
    mu = np.asarray(splats.centers)
    rcut = np.asarray(_host_rcut(splats, eps))
    ndev = int(partition[0][0].shape[0])
    blk = gp.shape[0] // ndev
    groups, widths = [], []
    out_b = 0
    for pi, pm in partition:
        nc = pi.shape[1]
        keeps, dists = {}, {}
        for d in range(ndev):
            for r_ in range(nc):
                sel = pi[d, r_][pm[d, r_]]
                if sel.size == 0:                             # device padded past its own run count
                    keeps[(d, r_)] = np.zeros(0, np.int64)    # nothing significant; mask stays False
                    continue
                pts = gp[d * blk + sel]                       # LOCAL index -> global grid row
                c = pts.mean(0)
                dd = np.sqrt(((mu - c) ** 2).sum(1))
                keeps[(d, r_)] = np.flatnonzero(
                    dd <= np.sqrt(((pts - c) ** 2).sum(1)).max() + rcut)
                dists[(d, r_)] = dd

        # Split each device's cells by significant-splat count before padding, or S is the worst
        # cell in the whole point-bucket and the padding costs more than the dense path it replaces.
        # `cell_partition` cannot do this (it is static); `cell_lists` can, since it sees the splats.
        #
        # Constant shapes are preserved by construction: cells are rank-ordered by count and cut at
        # FIXED sizes, so every refresh emits the same (nc, P, S) even as membership churns. Cells
        # whose count drifts past their range's S fall back to the keep-the-nearest truncation below.
        order = {d: sorted(range(nc), key=lambda r_: len(keeps[(d, r_)])) for d in range(ndev)}
        bounds = [(nc * j // n_sp_buckets, nc * (j + 1) // n_sp_buckets)
                  for j in range(n_sp_buckets)]
        for lo, hi in bounds:
            if hi <= lo:
                continue
            n = hi - lo
            want = max(1, max((len(keeps[(d, order[d][r_])]) for d in range(ndev)
                               for r_ in range(lo, hi)), default=1))
            S = int(s_pad[out_b]) if s_pad is not None else int(np.ceil(want * headroom))
            gi = np.zeros((ndev, n, pi.shape[2]), np.int32)
            gm = np.zeros((ndev, n, pi.shape[2]), bool)
            si = np.zeros((ndev, n, S), np.int32)
            sm = np.zeros((ndev, n, S), bool)
            for d in range(ndev):
                for k, r_ in enumerate(order[d][lo:hi]):
                    gi[d, k] = pi[d, r_]
                    gm[d, k] = pm[d, r_]
                    sig = keeps[(d, r_)]
                    if len(sig) > S:                          # overflow: keep the NEAREST
                        sig = sig[np.argsort(dists[(d, r_)][sig])[:S]]
                    si[d, k, :len(sig)] = sig
                    sm[d, k, :len(sig)] = True
            groups.append((jnp.asarray(gi), jnp.asarray(gm), jnp.asarray(si), jnp.asarray(sm)))
            widths.append(S)
            out_b += 1
    return tuple(groups), widths


@jax.jit
def cell_density_and_grad(splats, C, occ, grid_points, cells):
    """ρ(r), ∇ρ(r) by batched evaluation over fine cells. Equals the dense path to the ``eps`` tail."""
    q = full_quantities(splats)
    rho = jnp.zeros(grid_points.shape[0])
    grad = jnp.zeros((grid_points.shape[0], 3))
    def _one(q, C, pts, pm, si, sm):
        sub = tuple(x[si] for x in q)                           # (nc, S, ...) gathered per CELL
        g, dg = jax.vmap(_full_eval_block)(sub, pts)            # (nc,P,S), (nc,P,S,3)
        keep = pm[:, :, None] & sm[:, None, :]
        g = jnp.where(keep, g, 0.0)
        dg = jnp.where(keep[..., None], dg, 0.0)
        Cb = C[si]                                              # (nc, S, n_occ) ONCE per cell
        psi = jnp.einsum("cps,cso->cpo", g, Cb)
        r_c = jnp.sum(occ[None, None, :] * psi ** 2, axis=-1)
        # Contract the orbital index before the gradient index: `einsum("cpsd,cso->cpod")`
        # materializes (nc,P,n_occ,3). Algebraically ∇ρ = 2 Σ_i (Σ_o occ_o ψ_o C_io) ∇g_i, whose
        # intermediate is (nc,P,S) — the size of `g`, and `g` is never re-evaluated.
        w = jnp.einsum("o,cpo,cso->cps", occ, psi, Cb)
        return r_c, 2.0 * jnp.einsum("cps,cpsd->cpd", w, dg)

    for pi, pm, si, sm in cells:
        # Remat per group, unconditionally: without it the backward keeps every group's (nc,P,S)
        # and (nc,P,S,3) intermediates live at once. Costs ~1%, so there is no trade to expose.
        r_c, g_c = jax.checkpoint(_one)(q, C, grid_points[pi], pm, si, sm)
        rho = rho.at[pi].add(jnp.where(pm, r_c, 0.0))
        grad = grad.at[pi].add(jnp.where(pm[..., None], g_c, 0.0))
    return rho, grad
