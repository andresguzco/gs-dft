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
  headroom), and a cell past its ``S`` raises :class:`CellOverflow`.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gs_dft.coulomb.stream import full_quantities
from gs_dft.screening.grid import _host_rcut

__all__ = ["morton_order", "cell_partition", "cell_lists", "cell_density_and_grad", "Partition",
           "CellOverflow"]


class CellOverflow(RuntimeError):
    """A cell has more significant splats than its list's fixed width (fixed for the run)."""


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

    __slots__ = ("groups", "geom")

    def __init__(self, groups, geom=()):
        self.groups = tuple(groups)
        self.geom = tuple(geom)          # per group: cell centres (ndev,nc,3), radii (ndev,nc)

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


def _align(n, align):
    return -(-int(n) // int(align)) * int(align)


def _cell_geometry(gp, pi, pm, blk):
    """Per cell: the mean of its points and their largest distance from it (0 for an empty cell)."""
    pts = gp[np.arange(pi.shape[0])[:, None, None] * blk + pi]          # (ndev, nc, P, 3)
    n = pm.sum(-1)
    c = np.where(pm[..., None], pts, 0.0).sum(2) / np.maximum(n, 1)[..., None]
    R = np.where(pm, np.sqrt(((pts - c[:, :, None]) ** 2).sum(-1)), 0.0).max(-1)
    return c, R


def _pt_edges(cap, n_pt_buckets):
    """Point-bucket edges: (cap/2, cap], (cap/4, cap/2], ... (a cell pads to at most twice its points)."""
    return cap / 2.0 ** np.arange(n_pt_buckets - 1, 0, -1)


def deal_order(grid_points, ndev, *, k: int = 5, cap: int = 1024, bits: int = 10,
               n_pt_buckets: int = 6):
    """The order of a Morton-sorted grid that deals its cells round-robin over ``ndev`` devices.

    Returns ``order`` (ndev·blk,): device d's block of the grid is ``order[d·blk:(d+1)·blk]``,
    indices into ``grid_points``, −1 for a padding slot (a zero-weight point). The cells are the runs
    :func:`cell_partition` makes (Morton cell, split at ``cap``); taken bucket by bucket, each in
    Morton order, run i goes to device i mod ndev, and each device keeps its runs in Morton order.
    ``ndev = 1`` is the identity.

    Contiguous blocks would give each device one region of space, and a cell's work follows the
    splats there; the lists are padded to the widest device (one shape for shard_map), so every
    device would pay the busiest one's work. Dealt cells give every device a sample of the whole
    molecule, so the lists match and the padding vanishes, however the splats move.
    """
    gp = np.asarray(grid_points)
    n = gp.shape[0]
    if ndev == 1:
        return np.arange(n)
    _, key = morton_order(gp, bits)
    if np.any(np.diff(key) < 0):
        raise ValueError("deal_order: the grid must be in Morton order")
    cell = key >> np.int64(3 * (bits - k))
    _, first = np.unique(cell, return_index=True)
    last = np.append(first[1:], n)
    starts = np.concatenate([np.arange(a, b, cap) for a, b in zip(first, last)])
    lens = np.minimum(cap, np.repeat(last, -(-(last - first) // cap)) - starts)
    bucket = np.digitize(lens, _pt_edges(cap, n_pt_buckets), right=True)
    rank = np.empty(len(starts), np.int64)                     # position within its bucket, carried
    off = 0                                                     # across buckets so no device is first
    for b in np.unique(bucket):                                 # in all of them
        sel = np.flatnonzero(bucket == b)
        rank[sel] = off + np.arange(sel.size)
        off += sel.size
    dev = rank % ndev
    blocks = [np.concatenate([np.arange(s, s + m) for s, m in zip(starts[dev == d], lens[dev == d])])
              for d in range(ndev)]
    order = np.full((ndev, max(len(b) for b in blocks)), -1, np.int64)
    for d, b in enumerate(blocks):
        order[d, :len(b)] = b
    return order.reshape(-1)


def cell_partition(grid_points, *, k: int = 5, cap: int = 1024, bits: int = 10, n_pt_buckets: int = 6,
                   group_points: int | None = None, ndev: int = 1, align: int = 8):
    """Static point grouping — depends only on the grid, so build it once.

    Returns ``[(pt_idx (ndev,nc,P) int32, pt_mask (ndev,nc,P) bool), ...]``, one entry per
    point-count bucket, ``P`` rounded up to a multiple of ``align`` (masked) for the batched GEMMs.
    Assumes ``grid_points`` is already in :func:`morton_order`.

    ``group_points`` caps points per emitted group (``None``: one group per bucket). Memory does not
    need it, :func:`cell_density_and_grad` runs each list in chunks of cells, and every group adds
    lists to compile.

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

    The device axis forces uniform shapes: buckets are cut at fixed widths (so every device
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

    # Point buckets halve from cap down, (cap/2, cap], (cap/4, cap/2], ...: a cell pads to at most
    # twice its points. Quantiles of the run lengths are set by the many near-empty far cells, and
    # pad every run of 7 … 1024 points to 1024.
    npts_all = np.array([len(r) for rs in runs_dev for r in rs])
    edges = _pt_edges(cap, n_pt_buckets)
    sel_dev = [np.digitize([len(r) for r in rs], edges, right=True) for rs in runs_dev]

    out = []
    for b in sorted(set(np.digitize(npts_all, edges, right=True).tolist())):
        picks = [np.flatnonzero(sd == b) for sd in sel_dev]
        P = max((len(runs_dev[d][i]) for d in range(ndev) for i in picks[d]), default=1)
        P = _align(max(int(P), 1), align)
        nc_tot = max((len(p) for p in picks), default=0)     # pad every device to the widest
        if nc_tot == 0:
            continue
        nc_cap = nc_tot if group_points is None else max(1, int(group_points // P))
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
    return Partition(out, [_cell_geometry(gp, pi, pm, blk) for pi, pm in out])


def _keep(c, R, live, mu, rcut):
    """(ndev, nc, M) bool: the splat's cutoff ball meets the cell's bounding sphere, in a cell that
    holds points (a device's padding cells hold nothing)."""
    d2 = (mu[:, 0] - c[..., 0:1]) ** 2                      # by axis: no (.., M, 3) array
    d2 = d2 + (mu[:, 1] - c[..., 1:2]) ** 2
    d2 = d2 + (mu[:, 2] - c[..., 2:3]) ** 2
    return (jnp.sqrt(d2) <= R[..., None] + rcut) & live[..., None]


@jax.jit
def _counts(c, R, live, mu, rcut):
    """(ndev, nc) significant splats per cell."""
    return jnp.sum(_keep(c, R, live, mu, rcut), axis=-1, dtype=jnp.int32)


@functools.partial(jax.jit, static_argnums=(7,))
def _lists(c, R, live, pi, pm, mu, rcut, shapes):
    """One point group's lists at the static ``shapes`` ``((cells, width), ...)``: its cells ranked
    by count (stable), cut at those sizes, each cell's significant splats ascending in its first
    slots. Returns the lists and each list's widest cell (past its width = an overflow; the excess
    is dropped here, and the caller raises)."""
    keep = _keep(c, R, live, mu, rcut)
    ndev, nc, M = keep.shape
    cnt = jnp.sum(keep, axis=-1, dtype=jnp.int32)
    order = jnp.argsort(cnt, axis=1, stable=True)
    out, want, lo = [], [], 0
    for n, S in shapes:
        sel = order[:, lo:lo + n]                                          # (ndev, n)
        lo += n
        n_sel = jnp.take_along_axis(cnt, sel, 1)
        k_sel = jnp.take_along_axis(keep, sel[..., None], 1)               # (ndev, n, M)
        slot = jnp.where(k_sel, jnp.cumsum(k_sel, axis=-1, dtype=jnp.int32) - 1, S)
        si = jnp.zeros((ndev, n, S), jnp.int32).at[
            jnp.arange(ndev)[:, None, None], jnp.arange(n)[None, :, None], slot].set(
            jnp.broadcast_to(jnp.arange(M, dtype=jnp.int32), k_sel.shape), mode="drop")
        out.append((jnp.take_along_axis(pi, sel[..., None], 1), jnp.take_along_axis(pm, sel[..., None], 1),
                    si, jnp.arange(S) < n_sel[..., None]))
        want.append(jnp.max(n_sel))
    return tuple(out), jnp.stack(want)


def _cut(srt, n_lists, width, candidates=64):
    """List sizes over the cells ranked by count (``srt``: the count at each rank, ascending) that
    minimize Σ size × width(count at the list's top rank), over boundaries at ``candidates`` ranks."""
    nc = len(srt)
    B = np.unique(np.linspace(0, nc, min(candidates, nc) + 1).round().astype(int))
    nb = len(B)
    w = [width(int(srt[b - 1])) if b else 0 for b in B]
    INF = float("inf")
    f = [[INF] * nb for _ in range(n_lists + 1)]
    arg = [[0] * nb for _ in range(n_lists + 1)]
    f[0][0] = 0.0
    for k in range(1, n_lists + 1):
        for j in range(1, nb):
            for i in range(j):
                v = f[k - 1][i] + (B[j] - B[i]) * w[j]
                if v < f[k][j]:
                    f[k][j], arg[k][j] = v, i
    k = min(range(1, n_lists + 1), key=lambda k: (f[k][-1], k))
    sizes, j = [], nb - 1
    while k:
        i = arg[k][j]
        sizes.append(int(B[j] - B[i]))
        j, k = i, k - 1
    return sizes[::-1]


def cell_lists(splats, partition, *, eps: float = 1e-8, layout=None,
               headroom: float = 2.0, margin: int = 64, n_sp_buckets: int = 8, align: int = 8):
    """Per-cell significant splats at FIXED shapes — rebuilt on the refresh cadence.

    Each point group's cells are ranked by their significant-splat count and cut into at most
    ``n_sp_buckets`` lists, at the sizes that minimize the padded work Σ cells x width; a list's
    width is ``headroom`` x its widest cell + ``margin``, rounded up to a multiple of ``align``
    (masked slots). ``layout`` (``[(cells, width)]`` per list, the previous build's) pins the shapes
    for the whole run, so the step compiles once. A cell past its width raises
    :class:`CellOverflow`: truncating would drop density silently.

    Carries :func:`cell_partition`'s leading DEVICE axis through: point indices are local to a
    device block, splat indices are global (the splats are replicated, only the grid is sharded).
    Shapes are the max over every device, so all devices trace one shape.

    Built on the splats' device, one compiled call per point group (the host only cuts the sizes at
    the first build); the host build was most of a refresh (0.83 of 0.85 s at chignolin, M 3354).

    Returns the lists and their layout.
    """
    mu = splats.centers
    rcut = jnp.asarray(_host_rcut(splats, eps))
    M = int(mu.shape[0])
    width = lambda n: _align(min(M, int(np.ceil(max(1, n) * headroom)) + margin), align)
    todo = None if layout is None else list(layout)
    groups, out, over = [], [], []
    for (pi, pm), (c, R) in zip(partition, partition.geom):
        ndev, nc = pi.shape[:2]
        live = pm.any(-1)
        # Cells ranked by count and cut at FIXED sizes, so every refresh emits the same shapes as
        # membership churns; without the split S is the worst cell of the whole point group.
        if todo is None:
            srt = np.sort(np.asarray(_counts(c, R, live, mu, rcut)), axis=1)
            shapes, lo = [], 0
            for n in _cut(srt.max(0), n_sp_buckets, width):
                shapes.append((n, width(int(srt[:, lo:lo + n].max()))))
                lo += n
        else:
            shapes, tot = [], 0
            while tot < nc:
                shapes.append(tuple(int(x) for x in todo.pop(0)))
                tot += shapes[-1][0]
        lists, want = _lists(c, R, live, pi, pm, mu, rcut, tuple(shapes))
        for (n, S), w, g in zip(shapes, np.asarray(want), lists):
            if w > S:
                over.append((len(out), int(w), S))
            groups.append(g)
            out.append((n, S))
    if over:
        b, want, S = max(over, key=lambda o: o[1] - o[2])
        raise CellOverflow(
            f"cell lists overflow: {len(over)} of {len(out)} lists have a cell past their fixed "
            f"width (worst: list {b}, {want} significant splats > width {S}); the widths are fixed "
            f"at the first build, so raise cellgrid(headroom=..., margin=...)")
    return tuple(groups), out


def _eval(params, pts):
    """g (n,m) and t = A(r − μ) by component, the dense block's expressions; ∇g = −2 t g is never
    formed."""
    A, ce, no = params
    dx = pts[:, None, 0] - ce[None, :, 0]
    dy = pts[:, None, 1] - ce[None, :, 1]
    dz = pts[:, None, 2] - ce[None, :, 2]
    a00, a11, a22 = A[None, :, 0, 0], A[None, :, 1, 1], A[None, :, 2, 2]
    a01, a02, a12 = A[None, :, 0, 1], A[None, :, 0, 2], A[None, :, 1, 2]
    t = (a00 * dx + a01 * dy + a02 * dz, a01 * dx + a11 * dy + a12 * dz,
         a02 * dx + a12 * dy + a22 * dz)
    return no[None, :] * jnp.exp(-(dx * t[0] + dy * t[1] + dz * t[2])), t


def _take(x, idx):
    # Out-of-range rows read 0, and their cotangents drop instead of piling onto row 0.
    return jnp.take(x, idx, axis=0, mode="fill", fill_value=0)


CHUNK_BYTES = 1 << 30           # live bytes per chunk of cells


def _chunked(arrays, per_cell, budget):
    """Cell arrays as (chunks, cells per chunk, ...), padded with empty cells, each chunk within
    ``budget`` bytes."""
    nc = int(arrays[0].shape[0])
    nchunk = -(-nc // max(1, min(nc, budget // max(per_cell, 1))))
    n = -(-nc // nchunk)
    pad = nchunk * n - nc
    return tuple(jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:], x.dtype)])
                 .reshape((nchunk, n) + x.shape[1:]) for x in arrays)


def _one(sub, Cb, occ, pts, pm, sm):
    """ρ and ∇ρ on a chunk of cells from its gathered splats ``sub`` (nc, S, ...) and coefficient
    rows ``Cb`` (nc, S, n_occ)."""
    g, t = jax.vmap(_eval)(sub, pts)                       # (nc,P,S), 3 x (nc,P,S)
    g = jnp.where(pm[:, :, None] & sm[:, None, :], g, 0.0)
    psi = jnp.einsum("cps,cso->cpo", g, Cb)
    r_c = jnp.sum(occ[None, None, :] * psi ** 2, axis=-1)
    # ∇ρ = 2 Σ_i w_i ∇g_i, w = Σ_o occ_o ψ_o C_io, with ∇g = −2 t g: reductions over the splats
    # of (w g) t_k, so neither (nc,P,S,3) nor (nc,P,n_occ,3) is formed.
    h = jnp.einsum("o,cpo,cso->cps", occ, psi, Cb) * g
    return r_c, -4.0 * jnp.stack([jnp.sum(h * tk, -1) for tk in t], -1)


def _gathered(q, C, grid_points, pi, si, sm):
    """A chunk's points, splats and coefficient rows; padded slots point past the end and read 0."""
    si = jnp.where(sm, si, C.shape[0])
    return si, grid_points[pi], tuple(_take(x, si) for x in q), _take(C, si)


def _groups(cells, n_occ, budget):
    """Each list's cells as chunks within ``budget`` bytes: ``(pi, pm, si, sm)``, (chunks, n, ...)."""
    return [_chunked(group, 8 * (8 * group[0].shape[-1] * group[2].shape[-1]
                                 + (group[0].shape[-1] + group[2].shape[-1]) * n_occ), budget)
            for group in cells]


@functools.partial(jax.jit, static_argnames="budget")
def cell_density_and_grad(splats, C, occ, grid_points, cells, budget=CHUNK_BYTES):
    """ρ(r), ∇ρ(r) by batched evaluation over fine cells, ``budget`` live bytes of cells at a time.
    Equals the dense path to the ``eps`` tail."""
    return _cell_density(full_quantities(splats), C, occ, grid_points, cells, budget)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _cell_density(q, C, occ, grid_points, cells, budget):
    """The cell loop, with its own reverse: one pass over the same chunks that recomputes each
    chunk and adds its cotangents into ONE running total per input, in place. Autodiff of the loop
    gave each of the ~36 lists its own dense (M, n_occ) cotangent of C, all live until summed: about
    ten (M, n_occ) arrays per device at insulin, the step's largest floor once the RI-J stream was
    tiled."""
    n_pts = grid_points.shape[0]
    rho = jnp.zeros(n_pts)
    grad = jnp.zeros((n_pts, 3))
    for pi, pm, si, sm in _groups(cells, C.shape[1], budget):
        def chunk(a):
            pi_, pm_, si_, sm_ = a
            _, pts, sub, Cb = _gathered(q, C, grid_points, pi_, si_, sm_)
            return _one(sub, Cb, occ, pts, pm_, sm_)

        r_c, g_c = lax.map(chunk, (pi, pm, si, sm))
        P = pi.shape[-1]
        pi, pm = pi.reshape(-1, P), pm.reshape(-1, P)
        # Padded points scatter past the end, each to its own index (dropped): a real point is in
        # one cell, so the indices are unique and nothing piles onto point 0.
        dst = jnp.where(pm, pi, n_pts + jnp.arange(pi.size, dtype=pi.dtype).reshape(pi.shape))
        rho = rho.at[dst].add(r_c.reshape(-1, P), mode="drop", unique_indices=True)
        grad = grad.at[dst].add(g_c.reshape(-1, P, 3), mode="drop", unique_indices=True)
    return rho, grad


def _cell_density_fwd(q, C, occ, grid_points, cells, budget):
    return _cell_density(q, C, occ, grid_points, cells, budget), (q, C, occ, grid_points, cells)


def _cell_density_bwd(budget, res, cts):
    q, C, occ, grid_points, cells = res
    rho_bar, grad_bar = cts
    z = lambda x: jnp.zeros(jnp.shape(x), jnp.result_type(x))   # zeros_like would carry a mesh sharding into shard_map
    acc = (tuple(z(x) for x in q), z(C), z(occ), z(grid_points))
    for pi, pm, si, sm in _groups(cells, C.shape[1], budget):
        def body(acc, a):
            pi_, pm_, si_, sm_ = a
            si_, pts, sub, Cb = _gathered(q, C, grid_points, pi_, si_, sm_)
            _, vjp = jax.vjp(lambda s, c, o, x: _one(s, c, o, x, pm_, sm_), sub, Cb, occ, pts)
            rb = jnp.where(pm_, _take(rho_bar, pi_), 0.0)                    # (nc, P)
            gb = jnp.where(pm_[..., None], _take(grad_bar, pi_), 0.0)        # (nc, P, 3)
            d_sub, d_Cb, d_o, d_pts = vjp((rb, gb))
            a_q, a_C, a_o, a_x = acc
            # straight into the running totals, in place; padded slots (index M) drop
            return (tuple(a.at[si_].add(d, mode="drop") for a, d in zip(a_q, d_sub)),
                    a_C.at[si_].add(d_Cb, mode="drop"), a_o + d_o,
                    a_x.at[pi_].add(jnp.where(pm_[..., None], d_pts, 0.0), mode="drop")), None

        acc = lax.scan(body, acc, (pi, pm, si, sm))[0]
    d_q, d_C, d_o, d_x = acc
    return d_q, d_C, d_o, d_x, None


_cell_density.defvjp(_cell_density_fwd, _cell_density_bwd)
