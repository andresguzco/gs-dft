"""Step timer: the real ``train`` loop on the exp5 path, timed.

Reports the median step time without step 0, the refresh time separately, the compiles after step 0,
peak GPU memory (per device on a mesh), E/|g| at step 0 and the run's guard diagnostics. ``--profile``
adds a per-term fwd+bwd profile: without a mesh the four terms; on a mesh (G = 1 included) each sharded
term as its own shard_map, each device's share of it alone on one device (the imbalance), and the whole
energy; and the refresh by piece::

    python -m experiments.bench.step --system ethanol --M 144
    python -m experiments.bench.step --system penicillin --m_mult 2 --profile
    python -m experiments.bench.step --system <s> --basis cc-pvtz --m_mult 1 --skip 11 --G 4   # experiment 2

Steps are timed between consecutive ``step_cb`` calls (each forces a host sync); a refresh is timed by
wrapping ``train._refresh_state`` and is subtracted from the step it precedes.
"""
import argparse
import importlib
import logging
import time

import numpy as np
import jax


def _args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--system", default="ethanol")
    ap.add_argument("--basis", default="cc-pvdz", help="reference basis: sizes M through nao")
    ap.add_argument("--xc", default="pbe")
    ap.add_argument("--M", type=int, default=None, help="splat count; default m_mult x nao")
    ap.add_argument("--m_mult", type=float, default=2.0)
    ap.add_argument("--init", default="random", choices=("random", "chem"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--refresh", type=int, default=10)
    ap.add_argument("--skip", type=int, default=1, help="steps left out of the median (step 0 = compile)")
    ap.add_argument("--grid_level", type=int, default=3)
    ap.add_argument("--gamma_chunk", type=int, default=512)
    ap.add_argument("--G", type=int, default=None, help="mesh size; default: the builder's")
    ap.add_argument("--cells", default="auto", choices=("auto", "on", "off"),
                    help="the grid density by Morton cells (on), dense (off) or by size (builders.use_cells)")
    ap.add_argument("--headroom", type=float, default=None, help="cellgrid headroom; default cellgrid()'s")
    ap.add_argument("--margin", type=int, default=None, help="cellgrid margin")
    ap.add_argument("--align", type=int, default=None, help="cellgrid align")
    ap.add_argument("--profile", action="store_true", help="per-term fwd+bwd times")
    ap.add_argument("--reps", type=int, default=10, help="calls per term in the profile")
    return ap.parse_args()


class _Compiles(logging.Handler):
    """Counts the "Compiling" log records once ``armed``."""

    def __init__(self):
        super().__init__()
        self.armed, self.seen = False, []

    def emit(self, record):
        msg = record.getMessage()
        if self.armed and msg.startswith("Compiling"):
            self.seen.append(msg[:160])


def _sync(tree):
    import equinox as eqx
    jax.block_until_ready(jax.tree.leaves(eqx.filter(tree, eqx.is_array)))
    return tree


def _profile(ks, model, state, reps):
    """Median fwd+bwd seconds of each energy term, on one device, state and ks as arguments."""
    import equinox as eqx
    import jax.numpy as jnp
    from dftax.energy.grid import xc_energy
    from gs_dft import screening as scr
    from gs_dft.integrals import dense as full
    from gs_dft.ks.orthonormalize import orthonormalize

    occ = model.occupations
    C_orth = orthonormalize(model.C, model.C.T @ full.overlap_times(model.basis, model.C))

    def ortho_kin(ks_, st, basis, C):
        Co = orthonormalize(C, C.T @ full.overlap_times(basis, C))
        return occ @ jnp.sum(Co * full.kinetic_times(basis, Co), axis=0)

    def external(ks_, st, basis, Co):
        pi, pj, v = st.pairs
        return scr.screened_external_energy(basis, Co, occ, pi, pj, ks_.atom_coords, ks_.atom_charges,
                                            v, skip_pad=ks_.screen.skip_pad)

    def coulomb(ks_, st, basis, Co):
        ej, ek = ks_.coulomb.energy(basis, Co, None, occ, st)
        return ej + ks_.coulomb.hf_coeff * ek

    def xc(ks_, st, basis, Co):
        rho, grad, tau = ks_._density_on_grid(basis, Co, occ, st)
        return xc_energy(ks_.xc, jnp.maximum(rho, 1e-30), ks_.grid_weights, grad_rho=grad, tau=tau)

    out = {}
    for name, f, X in (("ortho_kin", ortho_kin, model.C), ("ext", external, C_orth),
                       ("coulomb", coulomb, C_orth), ("xc", xc, C_orth)):
        vg = eqx.filter_jit(eqx.filter_value_and_grad(lambda bx, ks_, st, f=f: f(ks_, st, *bx)))
        t0 = time.perf_counter()
        _sync(vg((model.basis, X), ks, state))
        t_first = time.perf_counter() - t0
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            _sync(vg((model.basis, X), ks, state))
            ts.append(time.perf_counter() - t0)
        out[name] = (float(np.median(ts)), t_first)
    return out


def _timed(fn, *a, reps):
    """(median seconds over ``reps`` synced calls, seconds of the first call)."""
    t0 = time.perf_counter()
    _sync(fn(*a))
    first = time.perf_counter() - t0
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _sync(fn(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), first


def _profile_mesh(ks, model, state, reps):
    """Per-term fwd+bwd on the run's mesh, each term its own shard_map with the step's specs, and each
    device's share of it alone on one device.

    Returns ``{term: (t_mesh, t_first, [t_share per device])}``. ``max(t_share)`` is the term's
    critical path and ``sum(t_share)`` its total work; ``t_mesh − max(t_share)`` is what the mesh adds
    (collectives, waiting). ``total`` is the whole energy's fwd+bwd on the mesh (no shares); the
    terms do not include the Löwdin retraction or the optimizer, which run replicated."""
    import equinox as eqx
    import jax.numpy as jnp
    from jax import lax, shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from dftax.energy.grid import xc_energy
    from gs_dft import screening as scr
    from gs_dft.ks import shard as shd
    from gs_dft.ks.orthonormalize import orthonormalize
    from gs_dft.integrals import dense as full
    from gs_dft.screening.cells import cell_density_and_grad
    from gs_dft.screening.grid import chunked_density_and_grad
    from gs_dft.coulomb.ri import df_coulomb_gamma_screened, df_coulomb_solve, df_coulomb_solve_rows
    tr = importlib.import_module("gs_dft.ks.train")

    mesh, G = ks.mesh, int(ks.mesh.size)
    occ = model.occupations
    gga = ks.xc.xc_type in ("GGA", "MGGA")
    skip, lam = ks.screen.skip_pad, ks.coulomb.lam
    chunk = getattr(ks.coulomb, "chunk", 2048)
    M = int(model.C.shape[0])
    rows = tuple(jax.device_put(x, NamedSharding(mesh, P())) for x in shd.row_partition(M, G))
    cells = tuple(state.cells or ())
    C_orth = orthonormalize(model.C, model.C.T @ full.overlap_times(model.basis, model.C))

    def t_rows(basis, C, rows_):              # the row-sharded S·C and T·C products
        ridx, rok = rows_
        Cr = C[ridx] * rok.astype(C.dtype)[:, None]
        return (jnp.sum(Cr * full.overlap_times_rows(basis, C, ridx))
                + occ @ jnp.sum(Cr * full.kinetic_times_rows(basis, C, ridx), axis=0))

    def t_xc(basis, C, grid, cells_):
        gp_, gw_ = grid
        if cells_:
            rho, grad = cell_density_and_grad(basis, C, occ, gp_,
                                              tuple((a[0], b[0], c[0], d[0]) for a, b, c, d in cells_))
        else:
            rho, grad = chunked_density_and_grad(basis, C, occ, gp_, chunk=ks.grid_chunk or 4096)
        return xc_energy(ks.xc, jnp.maximum(rho, 1e-30), gw_, grad_rho=grad if gga else None)

    def t_ext(basis, C, pairs):
        pi, pj, vm = pairs
        return scr.screened_external_energy(basis, C, occ, pi, pj, ks.atom_coords, ks.atom_charges,
                                            vm, skip_pad=skip)

    def t_gamma(basis, C, pairs, aux):
        pi, pj, vm = pairs
        return df_coulomb_gamma_screened(basis, C, occ, aux, pi, pj, valid=vm, skip_pad=skip,
                                         chunk=chunk)

    # One device: the whole factor (the mesh's is padded with an identity tail). The mesh: its rows.
    solve = lambda g, aux, cholV: df_coulomb_solve(g, aux, lam=lam,
                                                   cholV=cholV[:g.shape[0], :g.shape[0]])
    solve_rows = shard_map(lambda g, L: df_coulomb_solve_rows(g, L, "g"), mesh=mesh, check_vma=False,
                           in_specs=(P(), P("g")), out_specs=P())
    S_pairs, S_grid = (P("g"),) * 3, (P("g"), P("g"))
    S_cells = tuple((P("g"),) * 4 for _ in cells)
    pairs, grid = state.pairs, (ks.grid_points, ks.grid_weights)
    # term -> (local fn, its arguments after (basis, C), their specs, the replicated tail's extra args)
    terms = {"rows": (t_rows, (rows,), ((P("g"),) * 2,), None),
             "xc": (t_xc, (grid, cells), (S_grid, S_cells), None),
             "ext": (t_ext, (pairs,), (S_pairs,), None),
             "gamma": (t_gamma, (pairs, state.aux), (S_pairs, P()), (state.cholV,))}

    def share(x, g):                          # device g's block of a "g"-sharded leading axis
        n = x.shape[0] // G
        return x[g * n:(g + 1) * n]

    def shares(g):
        cells_g = tuple(tuple(x[g:g + 1] for x in grp) for grp in cells)   # keeps the device axis
        return {"rows": (tuple(share(x, g) for x in rows),),
                "xc": (tuple(share(x, g) for x in grid), cells_g),
                "ext": (tuple(share(x, g) for x in pairs),),
                "gamma": (tuple(share(x, g) for x in pairs), state.aux)}

    out = {}
    for name, (fn, args, specs, tail) in terms.items():
        n = len(args)

        def on_mesh(basis, C, *a, fn=fn, tail=tail, n=n, specs=specs):
            body = lambda basis, C, *b: lax.psum(fn(basis, C, *b), "g")
            v = shard_map(body, mesh=mesh, check_vma=False, out_specs=P(),
                          in_specs=(P(), P(), *specs))(basis, C, *a[:n])
            return v if tail is None else solve_rows(v, *a[n:])

        def alone(basis, C, *a, fn=fn, tail=tail, n=n):
            v = fn(basis, C, *a[:n])
            return v if tail is None else solve(v, a[n - 1], *a[n:])

        extra = tuple(tail or ())
        vg = eqx.filter_jit(eqx.filter_value_and_grad(lambda bc, *a, f=on_mesh: f(*bc, *a)))
        t, first = _timed(vg, (model.basis, C_orth), *args, *extra, reps=reps)
        vg1 = eqx.filter_jit(eqx.filter_value_and_grad(lambda bc, *a, f=alone: f(*bc, *a)))
        bc1 = tr._on_one_device((model.basis, C_orth))
        per = [_timed(vg1, bc1, *tr._on_one_device((*shares(g)[name], *extra)), reps=reps)[0]
               for g in range(G)]
        out[name] = (t, first, per)

    m_tr, m_st = tr._partition(model)
    vg = eqx.filter_jit(eqx.filter_value_and_grad(lambda m, s, k, st: k(eqx.combine(m, s), st)[0]))
    t, first = _timed(vg, m_tr, m_st, ks, state, reps=reps)
    out["total"] = (t, first, [])
    return out


def _profile_refresh(ks, basis, state, reps):
    """Seconds of each piece of a refresh as ``train`` runs it (median of ``reps``): the pair list on
    one device, the product aux and its Cholesky, the cell lists (host), the whole refresh on one
    device, and its broadcast to the mesh."""
    from gs_dft import screening as scr
    from gs_dft.ks import shard as shd
    from gs_dft.ks.energy import refresh
    from gs_dft.basis.isotropic import init_product_aux
    from gs_dft.coulomb import ri
    from gs_dft.screening import cells as cl
    tr = importlib.import_module("gs_dft.ks.train")
    b1 = tr._on_one_device(basis)
    pad = int(state.pairs[0].shape[0])
    G = 1 if ks.mesh is None else int(ks.mesh.size)

    def pairs():
        return shd.stripe_pairs(*scr.neighbor_pairs(b1, eps=ks.screen.eps, pad_to=pad,
                                                    unique=ks.screen.unique), G)

    def aux():
        a = init_product_aux(b1, scales=ks.coulomb.scales, diagonal_only=ks.coulomb.diagonal)
        return a, ri.aux_cholesky(a, ks.coulomb.lam)

    def cells():
        if ks.cell_part is None:
            return ()
        layout = [tuple(int(x) for x in g[2].shape[1:]) for g in state.cells]
        return cl.cell_lists(b1, ks.cell_part, eps=ks.cell_spec.eps, layout=layout,
                             headroom=ks.cell_spec.headroom, margin=ks.cell_spec.margin,
                             align=ks.cell_spec.align, n_sp_buckets=ks.cell_spec.sp_buckets)[0]

    whole = lambda: refresh(ks, b1, pad_to=pad, like=state)
    parts = {"pairs": pairs, "aux_chol": aux, "cells": cells, "whole_1dev": whole}
    if ks.mesh is not None:
        st1 = whole()
        parts["broadcast"] = lambda: tr._replicated(st1, ks.mesh)
    return {k: _timed(f, reps=reps)[0] for k, f in parts.items()}


def main():
    args = _args()
    jax.config.update("jax_enable_x64", True)
    import jax.random as jr
    from experiments.common import systems, builders, results, probes
    import gs_dft
    from gs_dft import init_model, nao as nao_of, chem
    tr = importlib.import_module("gs_dft.ks.train")         # the module (gs_dft.ks exports the verb)

    print(f"# gs_dft from {gs_dft.__file__}  devices={jax.devices()}", flush=True)
    mol = systems.molecule(system=args.system, basis=args.basis)
    nao = int(nao_of(mol))
    M = int(args.M) if args.M else int(round(args.m_mult * nao))
    xc = builders.xc_of(args.xc)
    kw = {} if args.G is None else {"G": int(args.G)}
    from gs_dft.ks.terms import cellgrid
    knobs = {k: getattr(args, k) for k in ("headroom", "margin", "align") if getattr(args, k) is not None}
    kw["cells"] = {"on": cellgrid(**knobs), "off": False, "auto": None}[args.cells]
    ks = builders.splat_ks(mol, xc, grid_level=args.grid_level, screened=True,
                           gamma_chunk=args.gamma_chunk, **kw)
    model = init_model(mol, M, jr.PRNGKey(args.seed), init=chem() if args.init == "chem" else None)
    G = 1 if ks.mesh is None else int(ks.mesh.size)
    n_atoms = int(ks.atom_coords.shape[0])
    print(f"# {args.system}: {n_atoms} atoms  nao={nao}  M={M}  G={G}  grid={args.grid_level} "
          f"({int(ks.grid_points.shape[0])} pts)  screen={ks.screen}", flush=True)

    # Refreshes, timed and synced.
    t_ref, orig, fills = [], tr._refresh_state, []

    def timed_refresh(*a, **k):
        t0 = time.perf_counter()
        st = _sync(orig(*a, **k))
        t_ref.append(time.perf_counter() - t0)
        if st.cells is not None:                    # per list: the widest cell and the width
            fills.append([(int(np.asarray(g[3]).sum(-1).max()), int(g[3].shape[-1])) for g in st.cells])
        return st

    tr._refresh_state = timed_refresh

    comp = _Compiles()
    logging.getLogger("jax").addHandler(comp)
    jax.config.update("jax_log_compiles", True)

    marks, first, last = [], {}, {}

    def step_cb(m):
        marks.append((m["step"], time.perf_counter(), len(t_ref)))
        last.update(E=m["E_total"], g=m["grad_norm"])
        if not first:
            first.update(E=m["E_total"], g=m["grad_norm"], nelec=m["nelec"])
            comp.armed = True                       # everything after step 0 counts

    t_call = time.perf_counter()
    # No guard: a measurement is not stopped (its electron error per atom is in the row instead),
    # and nothing outside the step is held.
    res = tr.train(ks, model, steps=args.steps, refresh_every=args.refresh, monitor=None,
                   step_cb=step_cb, guard=None)
    comp.armed = False
    jax.config.update("jax_log_compiles", False)
    tr._refresh_state = orig

    # Step i's interval holds the refresh that precedes it (if any); subtract it.
    dts, ref_steps = [], []
    for k in range(1, len(marks)):
        i, t, nref = marks[k]
        dt = t - marks[k - 1][1]
        if nref > marks[k - 1][2]:
            dt -= t_ref[nref - 1]
            ref_steps.append(t_ref[nref - 1])
        if i >= args.skip:
            dts.append(dt)
    dts = np.asarray(dts)
    startup = marks[0][1] - t_call
    row = dict(M=M, nao=nao, n_atoms=n_atoms, G=G, xc=args.xc, grid=args.grid_level, init=args.init,
               cells="on" if ks.cell_spec is not None else "off",
               steps=len(marks), refresh=args.refresh,
               step_med_s=float(np.median(dts)), step_p10_s=float(np.percentile(dts, 10)),
               step_p90_s=float(np.percentile(dts, 90)),
               refresh_med_s=float(np.median(ref_steps)) if ref_steps else float("nan"),
               n_refresh=len(ref_steps), startup_s=float(startup), refresh0_s=float(t_ref[0]),
               compiles_after_step0=len(comp.seen),
               peak_gpu_mb=round(probes.peak_gpu_mb()),
               E0=first["E"], g0=first["g"], nelec0=first["nelec"],
               E_last=last["E"], g_last=last["g"],
               n_pairs=int(res.state.pairs[0].shape[0]) if res.state.pairs is not None else 0)
    if fills:
        # fill: the widest cell over its list's width; growth: over the list's widest at step 0
        row.update(cell_lists=len(fills[0]), cell_S_max=max(S for _, S in fills[0]),
                   cell_fill_max=max(c / S for f in fills for c, S in f),
                   cell_growth_max=max(c / max(c0, 1) for f in fills for (c, _), (c0, _) in zip(f, fills[0])),
                   cell_growth_abs=max(c - c0 for f in fills for (c, _), (c0, _) in zip(f, fills[0])))
    row.update(builders.run_fields(ks, res))
    if fills:
        print("# list growth (first -> widest, width): " + " ".join(
            f"{c0}->{max(f[k][0] for f in fills)}/{S}" for k, (c0, S) in enumerate(fills[0])), flush=True)
    for msg in comp.seen[:20]:
        print(f"# compile after step 0: {msg}", flush=True)

    if ks.mesh is not None:                         # per device: the refresh runs on device 0
        row["peak_gpu_mb_dev"] = "/".join(str(round((d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6))
                                          for d in ks.mesh.devices.flat)
        print(f"# peak MB per device: {row['peak_gpu_mb_dev']}", flush=True)

    if args.profile and ks.mesh is not None:
        pks, pmodel, *_ = tr._prepare_mesh(ks, res.model)
        prof = _profile_mesh(pks, pmodel, res.state, args.reps)
        for name, (t, t_first, per) in prof.items():
            row[f"t_{name}_s"] = t
            msg = f"# profile {name:6s} mesh {t * 1e3:9.2f} ms  (first call {t_first:.1f} s)"
            if per:
                row[f"t_{name}_max_s"], row[f"t_{name}_sum_s"] = max(per), sum(per)
                msg += (f"  shares max {max(per) * 1e3:8.2f} mean {np.mean(per) * 1e3:8.2f} ms "
                        f"[{' '.join(f'{p * 1e3:.1f}' for p in per)}]  sum {sum(per) * 1e3:8.2f}")
            print(msg, flush=True)
        row["t_terms_s"] = float(sum(t for k, (t, _, _) in prof.items() if k != "total"))
    elif args.profile:
        prof = _profile(ks, res.model, res.state, args.reps)
        for name, (t, t_first) in prof.items():
            row[f"t_{name}_s"] = t
            print(f"# profile {name:10s} {t * 1e3:9.2f} ms  (first call {t_first:.2f} s)", flush=True)
        row["t_terms_s"] = float(sum(t for t, _ in prof.values()))
    if args.profile and ks.screen is not None:
        for name, t in _profile_refresh(ks, res.model.basis, res.state, max(1, args.reps // 3)).items():
            row[f"t_refresh_{name}_s"] = t
            print(f"# refresh {name:10s} {t * 1e3:9.2f} ms", flush=True)
    print(results.result("bench", args.system, **row), flush=True)


if __name__ == "__main__":
    main()
