"""The splat Kohn-Sham total energy as a differentiable function of the model.

The :class:`SplatKS` builder is constructed once from a system + functional +
choices-as-values (the dftax 0.2 style):

    ks = SplatKS((coords, charges), PBE(),
                 grid    = (grid_points, grid_weights),      # or points(gp, gw, chunk=…)
                 coulomb = df(),                             # or exact()
                 screen  = pairlist(eps=1e-7),               # or None (dense)
                 mesh    = None)                             # or a jax Mesh (screened only)

Because the splat basis moves during training, nothing heavy is precomputed at
build time — construction is cheap, and the per-geometry work products (the
significant-pair list, the frozen product aux) live in a :class:`SplatState`
rebuilt by :func:`refresh` as the splats move:

    state  = refresh(ks, model.basis)
    E, aux = ks(model, state)                                # aux = per-term EnergyAux

The energy is

    E_KS = Tr(P·T) + Tr(P·V_ne) + E_J + a_x·E_K + E_xc + E_nn

with the Coulomb strategy a term value from :func:`~gs_dft.ks.terms.exact`
/ :func:`~gs_dft.ks.terms.df` (see :mod:`gs_dft.ks.terms`), the
overlap/kinetic always dense (screening S/T tips the occupied Grams indefinite
⇒ E_kin < 0; only V_ne / Coulomb / the grid density are screened), and the XC
quadrature dense or streamed by the grid spec's ``chunk``.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import equinox as eqx

from jaxtyping import Float, Array, Scalar
from dftax.energy.xc import XCFunctional
from dftax.energy.grid import xc_energy
from dftax.grid import Becke, Points, becke_grid
# Vendored: see gs_dft/ks/energy_aux.py.
from gs_dft.ks.energy_aux import EnergyAux, pack_energy_aux
from dftax.integrals.nuclear_repulsion import nuclear_repulsion as _nuclear_repulsion

from gs_dft.ks.orthonormalize import lowdin_orthonormalize, orthonormalize
from gs_dft.integrals import dense as _full
from gs_dft import screening as _scr
from gs_dft.screening import grid as _gscr
from gs_dft.screening import cells as _cells
from gs_dft.basis.isotropic import init_product_aux
from gs_dft.ks.terms import (
    CoulombTerm,
    DFSpec,
    ExactSpec,
    Pairlist,
    ProductDFCoulomb,
    ScreenedExactCoulomb,
    ScreenedProductDFCoulomb,
    _make_coulomb,
    exact,
    hf_coeff,
)

__all__ = ["SplatModel", "SplatState", "SplatKS", "refresh", "init_model"]


def _resolve_system(system):
    """Normalize a system input to ``(coords, charges, nelec, symbols)``.

    Accepts a dftax :class:`~dftax.system.molecule.Molecule` or a raw
    ``(coords, charges)`` pair. The raw pair carries no electron count or
    element identities (``nelec = 0``, ``symbols = None``), which forecloses
    the conveniences that need them: the electron-count guard and Becke-grid
    construction.
    """
    if isinstance(system, (tuple, list)) and len(system) == 2:
        coords, charges = system
        return jnp.asarray(coords), jnp.asarray(charges, dtype=float), 0, None
    if hasattr(system, "atom_coords") and hasattr(system, "atom_charges"):
        symbols = list(system.symbols) if hasattr(system, "symbols") else None
        return (jnp.asarray(system.atom_coords()),
                jnp.asarray(system.atom_charges(), dtype=float),
                int(getattr(system, "nelectron", 0)), symbols)
    raise TypeError(
        f"system must be a dftax Molecule or a (coords, charges) pair, "
        f"got {type(system)!r}")


def nao(system) -> int:
    """AO count of a system's GTO basis (the M-sizing reference).

    Resolved through :func:`dftax.basis.loader.build_basis_data` (BSE-backed,
    cached per element) and counted in the molecule's own spherical/Cartesian
    convention."""
    import numpy as _np

    from dftax.basis.loader import build_basis_data
    bd = build_basis_data(list(system.symbols), system.atom_coords(), system.basis,
                          spherical=False)
    if not getattr(system, "spherical", True):
        return int(bd.centers.shape[0])
    l = _np.asarray(bd.angular).sum(axis=1)             # Cartesian component triples -> l
    total = 0
    for lv in _np.unique(l):
        n_shell = int((l == lv).sum()) // ((int(lv) + 1) * (int(lv) + 2) // 2)
        total += n_shell * (2 * int(lv) + 1)
    return int(total)


class SplatModel(eqx.Module):
    """Trainable model: Gaussian splats + MO coefficients."""

    basis: eqx.Module                       # Splat (full-cov primary); aux is IsotropicSplat
    C: Float[Array, "M N_occ"]
    occupations: Float[Array, "N_occ"]

    @property
    def n_electrons(self) -> float:
        return float(jnp.sum(self.occupations))


class SplatState(eqx.Module):
    """Refresh products for the current splats — built by :func:`refresh`.

    One value carries everything the configured strategies need per energy
    call; a field is ``None`` exactly when the strategy does not use it
    (``pairs`` without ``screen=``, ``aux``/``aux_K`` without ``coulomb=df()``).
    Rebuild when the splats move (the trainers do this every ``refresh`` steps).
    """

    pairs: tuple | None = None              # (pi, pj, valid) from screening.neighbor_pairs
    aux: eqx.Module | None = None           # frozen pair-product J-aux (IsotropicSplat)
    aux_K: eqx.Module | None = None         # frozen pair-product K-aux (hybrids)
    # Lower Cholesky factor of (V + λI) for the frozen aux metric: computed once per
    # refresh, so the per-step RI-J solve is O(N_aux²) cho_solve instead of O(N_aux³).
    cholV: Float[Array, "Naux Naux"] | None = None
    # Per-cell significant-splat lists for the block-sparse grid density
    # (screening/cells.py). None = the dense/chunked path. Rebuilt on the refresh
    # cadence at fixed width, so the jitted step still compiles once.
    cells: tuple | None = None


class SplatKS(eqx.Module):
    """Splat KS total energy ``E(model, state)`` — build once, call as a verb.

    See the module docstring for the canonical flow. Invalid combinations
    raise here: ``mesh`` requires ``screen`` (the sharded step is the screened
    step), the sharded path is pure-functional only, and screened ``exact()``
    has no exact-exchange path.
    """

    atom_coords: Float[Array, "n_atoms 3"]
    atom_charges: Float[Array, "n_atoms"]
    xc: XCFunctional
    grid_points: Float[Array, "G 3"]
    grid_weights: Float[Array, "G"]
    coulomb: CoulombTerm
    nelec: int = eqx.field(static=True)
    grid_chunk: int | None = eqx.field(static=True)
    screen: Pairlist | None = eqx.field(static=True)
    mesh: object = eqx.field(static=True)
    # Block-sparse grid density. `cell_part` is the static point grouping (the grid does not move,
    # so it is built once); the splat-dependent lists live in SplatState and are refreshed.
    cell_part: object = eqx.field(static=True, default=None)
    cell_spec: object = eqx.field(static=True, default=None)

    def __init__(self,
                 system: "object | tuple",
                 xc: XCFunctional, *,
                 grid: "Becke | Points | tuple",
                 coulomb: "ExactSpec | DFSpec | None" = None,
                 screen: Pairlist | None = None,
                 mesh: object = None,
                 cells: "CellGrid | None" = None):
        """Build the energy functional.

        Args:
            system: a dftax ``Molecule`` or a raw ``(coords, charges)`` pair
                (Bohr). A molecule carries its electron count, enabling the
                occupation guard in the verbs.
            xc: the exchange-correlation functional (e.g. ``PBE()``); hybrids
                carry their exact-exchange fraction, consumed here.
            grid: the XC quadrature — an explicit ``(points, weights)`` pair,
                a :func:`dftax.grid.points` spec whose ``chunk`` streams the
                grid density in point-chunks (O(chunk·M) memory), or a
                :func:`dftax.grid.becke` spec (needs a system with element
                symbols).
            coulomb: :func:`~gs_dft.ks.terms.exact` (default) or
                :func:`~gs_dft.ks.terms.df`.
            screen: a :func:`~gs_dft.ks.terms.pairlist` spec to restrict
                V_ne + Coulomb to the significant-overlap pairs (``None`` =
                dense). S/T stay dense either way (PSD requirement).
            mesh: a 1-D device mesh (``shard.make_mesh()``) to shard the
                screened step across devices; ``None`` = single device.
            cells: a :func:`~gs_dft.ks.terms.cellgrid` spec — the
                block-sparse grid density (6.2x on the density term at ala_15).
                MORTON-SORTS the grid here, which is safe because the XC energy
                is a sum over points. ``None`` = the dense/chunked path.
        """
        coords, charges, nelec, symbols = _resolve_system(system)
        self.atom_coords = coords
        self.atom_charges = charges
        self.nelec = nelec
        self.xc = xc
        if isinstance(grid, Becke):
            if symbols is None:
                raise ValueError("a becke() grid needs element symbols; pass a "
                                 "Molecule system or an explicit grid.")
            # Forward the spec's pruning fields when the engine has them; without this a
            # becke(prune=None) spec is SILENTLY ignored and the grid prunes anyway.
            extra = {k: getattr(grid, k) for k in ("prune", "r_max") if hasattr(grid, k)}
            gc, gw_ = becke_grid(symbols, coords, grid.n_radial, grid.lebedev, **extra)
            gp, gw, chunk = gc, gw_, grid.chunk
        elif isinstance(grid, Points):
            gp, gw, chunk = grid.coords, grid.weights, grid.chunk
        else:
            gp, gw = grid
            chunk = None
        gp, gw = jnp.asarray(gp), jnp.asarray(gw)
        self.cell_part = self.cell_spec = None
        if cells is not None:
            # Morton-sort the grid ONCE. Required: Becke grids are radial-shell ordered, so a
            # contiguous slice is a spherical shell spanning the molecule and the significant list
            # degenerates to every splat. Sorting is safe — E_xc is a sum over points, so `gp`/`gw`
            # just travel together and nothing needs un-permuting.
            #
            # Sort here but partition below, after the mesh pad: the partition must cover the
            # final grid, and it needs `ndev` to cut cells at the device boundaries.
            perm, _ = _cells.morton_order(gp)
            gp, gw = gp[perm], gw[perm]
            self.cell_spec = cells
        # chunk may be "auto" (the engine's default since its grid rewrite), an int, or None.
        # "auto" = let the engine size it; we hold no budget model here, so treat it as unset.
        self.grid_chunk = None if chunk is None or chunk == "auto" else int(chunk)
        spec = exact() if coulomb is None else coulomb
        self.screen = screen
        self.coulomb = _make_coulomb(spec, screened=screen is not None, a_x=hf_coeff(xc),
                                     a_x_lr=float(getattr(xc, "hf_coeff_lr", 0.0) or 0.0),
                                     omega=float(getattr(xc, "omega", 0.0) or 0.0),
                                     skip_pad=bool(screen is not None and screen.skip_pad))
        if screen is not None and screen.unique and self.coulomb.hf_coeff != 0.0:
            raise ValueError("pairlist(unique=True) is PBE-class only: the screened RI-K "
                             "contraction is not symmetric in the pair index")
        if mesh is not None:
            if screen is None:
                raise ValueError("mesh= shards the SCREENED step; pass screen=pairlist(...).")
            if self.coulomb.hf_coeff != 0.0:
                raise ValueError("the sharded screened path is pure-functional only "
                                 "(no sharded RI-K); use a non-hybrid xc with mesh=.")
            # Build ≠ run: the mesh is a BUILD choice, so the grid is laid out for it
            # here — padded to a mesh-size multiple (clone point 0, zero weight ⇒ no XC
            # contribution) and, multi-node, promoted to a "g"-sharded global array with
            # the nuclear arrays replicated. The verbs never rewrite the ks.
            from gs_dft.ks import shard as _shd
            G = int(mesh.size)
            if gp.shape[0] % G:
                npad = G - gp.shape[0] % G
                gp = jnp.concatenate([gp, jnp.broadcast_to(gp[:1], (npad, 3))])
                gw = jnp.concatenate([gw, jnp.zeros(npad, gw.dtype)])
            if jax.process_count() > 1:
                gp = _shd.to_global(gp, mesh, sharded=True)
                gw = _shd.to_global(gw, mesh, sharded=True)
                self.atom_coords = _shd.to_global(self.atom_coords, mesh)
                self.atom_charges = _shd.to_global(self.atom_charges, mesh)
        if self.cell_spec is not None:
            # After the mesh pad, so the partition covers the FINAL grid and its cells are cut at
            # the device boundaries. ndev=1 (no mesh) still emits the leading device axis.
            self.cell_part = _cells.cell_partition(
                gp, k=cells.k, cap=cells.cap, n_pt_buckets=cells.pt_buckets,
                ndev=(int(mesh.size) if mesh is not None else 1))
        self.grid_points = gp
        self.grid_weights = gw
        self.mesh = mesh

    # -- state validation (python-level, so the error beats the trace error) --
    def _check_state(self, state):
        if self.screen is not None and state.pairs is None:
            raise ValueError("this SplatKS screens: pass the refresh state "
                             "(state.pairs) — see refresh(ks, basis).")
        if isinstance(self.coulomb, (ProductDFCoulomb, ScreenedProductDFCoulomb)):
            if state.aux is None:
                raise ValueError("coulomb=df() needs the frozen product aux: "
                                 "pass the refresh state — see refresh(ks, basis).")
            if self.coulomb.hf_coeff != 0.0 and state.aux_K is None:
                raise ValueError("hybrid df() needs the K-aux: pass the refresh state "
                                 "(state.aux_K) — see refresh(ks, basis).")

    def __call__(self, model: SplatModel, state: SplatState | None = None
                 ) -> tuple[Scalar, EnergyAux]:
        state = SplatState() if state is None else state
        self._check_state(state)
        occ = model.occupations

        if self.screen is not None:
            pi, pj, valid = state.pairs
            if self.mesh is not None:
                # Refuse a meta-GGA here rather than approximate one: `sharded_screened_energy`
                # has no tau, so a meta-GGA would silently be evaluated as a GGA — plausible,
                # wrong, and invisible to every diagnostic the trainer prints.
                if self.xc.xc_type == "MGGA":
                    raise NotImplementedError(
                        f"{type(self.xc).__name__} is a meta-GGA and the sharded path has no tau "
                        f"term; run it on one device (mesh=None) until sharded tau lands")
                # One shard_map for the whole step: rows of S/T, grid points and pairs are sharded,
                # parameters replicated, the Löwdin retraction runs replicated on the psum'd Gram.
                from gs_dft.ks import shard as _sharded
                E_total, E_kinetic, E_ext, E_J, E_xc, nel = _sharded.sharded_screened_energy(
                    model.basis, model.C, occ, state.aux, self.atom_coords, self.atom_charges,
                    (pi, pj, valid), self.grid_points, self.grid_weights, mesh=self.mesh,
                    rows=_sharded.row_partition(int(model.C.shape[0]), int(self.mesh.size)),
                    xc=self.xc, gga=(self.xc.xc_type in ("GGA", "MGGA")), cholV=state.cholV,
                    hartree=("exact" if isinstance(self.coulomb, ScreenedExactCoulomb)
                             else "df"),
                    grid_chunk=(self.grid_chunk or 4096),
                    gamma_chunk=getattr(self.coulomb, "chunk", 2048),
                    cells=state.cells, skip_pad=self.screen.skip_pad)
                return E_total, pack_energy_aux(
                    kinetic=E_kinetic, hartree=E_J, xc=E_xc, external=E_ext, nelec=nel)

            # Dense one-electron S, T (closed-form) so the occupied Gram G=CᵀSC and the kinetic Gram
            # stay exactly PSD — screening them tips both indefinite. S, T, P are never formed: the
            # blocked contractions carry (M, n_occ).
            C_orth = orthonormalize(model.C, model.C.T @ _full.overlap_times(model.basis, model.C))
            E_kinetic = occ @ jnp.sum(C_orth * _full.kinetic_times(model.basis, C_orth), axis=0)
            # E_external stays screened: a trace (no positivity requirement) and dense V_ne's
            # (Q,M,M,3,3) intermediate OOMs at scale.
            E_external = _scr.screened_external_energy(
                model.basis, C_orth, occ, pi, pj,
                self.atom_coords, self.atom_charges, valid, skip_pad=self.screen.skip_pad)
            E_hartree, E_exchange = self.coulomb.energy(model.basis, C_orth, None, occ, state)
            E_exchange_lr = self.coulomb.energy_lr(model.basis, C_orth, None, occ, state)
        else:
            S, T, V_ne = _full.one_electron_integrals(
                model.basis, self.atom_coords, self.atom_charges)
            C_orth = lowdin_orthonormalize(model.C, S)
            P = C_orth @ jnp.diag(occ) @ C_orth.T
            E_kinetic = jnp.sum(P * T)
            E_external = jnp.sum(P * V_ne)
            E_hartree, E_exchange = self.coulomb.energy(model.basis, C_orth, P, occ, state)
            E_exchange_lr = self.coulomb.energy_lr(model.basis, C_orth, P, occ, state)

        return self._finish(model, C_orth, E_kinetic, E_external, E_hartree, E_exchange,
                            E_exchange_lr,
                            state)

    def _density_on_grid(self, basis, C_orth, occ, state=None):
        """(ρ, ∇ρ|None, τ|None) on the grid — block-sparse cells, dense, or ``grid_chunk``-streamed.

        Branch on the functional's own `xc_type`, never on a hardcoded list: a meta-GGA reports
        "MGGA" and needs both ∇ρ and τ, and `xc_energy` will evaluate it on ρ alone rather than
        raising.
        """
        kind = self.xc.xc_type
        mgga = kind == "MGGA"
        gga = kind in ("GGA", "MGGA")           # a meta-GGA needs ∇ρ too, not only τ
        if state is not None and getattr(state, "cells", None) is not None:
            # `cells` carries a leading DEVICE axis (see cells.cell_partition). This is the
            # single-device path, so ndev == 1 and we take that slice; the sharded path lets
            # shard_map do the slicing instead.
            local = tuple((pi[0], pm[0], si[0], sm[0]) for pi, pm, si, sm in state.cells)
            if mgga:
                raise NotImplementedError(
                    "the block-sparse cell grid has no tau path; run a meta-GGA with cells=None "
                    "(the cellgrid() spec is opt-in and off by default)")
            rho, grad = _cells.cell_density_and_grad(basis, C_orth, occ,
                                                    self.grid_points, local)
            return ((rho, grad, None) if gga else (rho, None, None))
        if self.grid_chunk is not None:                    # 2a chunked exact (memory-streamed)
            if mgga:
                return _gscr.chunked_density_grad_tau(basis, C_orth, occ, self.grid_points,
                                                      chunk=self.grid_chunk)
            if gga:
                return (*_gscr.chunked_density_and_grad(basis, C_orth, occ, self.grid_points,
                                                        chunk=self.grid_chunk), None)
            return _gscr.chunked_density(basis, C_orth, occ, self.grid_points,
                                         chunk=self.grid_chunk), None, None
        if mgga:
            return _full.eval_density_grad_tau_on_grid(basis, C_orth, occ, self.grid_points)
        if gga:
            return (*_full.eval_density_and_grad_on_grid(basis, C_orth, occ, self.grid_points),
                    None)
        return _full.eval_density_on_grid(basis, C_orth, occ, self.grid_points), None, None

    def _finish(self, model, C_orth, E_kinetic, E_external, E_hartree, E_exchange,
                E_exchange_lr=0.0, state=None):
        """Shared tail: XC (grid), nuclear repulsion, total + per-term aux."""
        rho, grad_rho, tau = self._density_on_grid(model.basis, C_orth, model.occupations, state)
        rho = jnp.maximum(rho, 1e-30)

        nelec = jnp.dot(self.grid_weights, rho)
        E_xc = xc_energy(self.xc, rho, self.grid_weights, grad_rho=grad_rho, tau=tau)
        # The "-V" is a separate term, not part of the semilocal functional: `xc_energy` does
        # not add VV10 nonlocal correlation for ωB97M-V/ωB97X-V, so without this line either
        # converges to a plausible energy that is simply missing a term. It is a pure grid
        # functional of (ρ, |∇ρ|²), both already in hand.
        nlc_b = float(getattr(self.xc, "nlc_b", 0.0) or 0.0)
        if nlc_b:
            if grad_rho is None:
                raise ValueError(f"{type(self.xc).__name__} needs |∇ρ|² for its VV10 term but the "
                                 f"density path returned no gradient")
            from gs_dft.ks.nlc import vv10_energy
            E_xc = E_xc + vv10_energy(rho, jnp.sum(grad_rho ** 2, axis=-1),
                                      self.grid_points, self.grid_weights,
                                      b=nlc_b, c=float(getattr(self.xc, "nlc_c", 0.0) or 0.0))
        E_nn = _nuclear_repulsion(self.atom_coords, self.atom_charges)

        E_total = (E_kinetic + E_external + E_hartree
                   + self.coulomb.hf_coeff * E_exchange
                   + self.coulomb.hf_coeff_lr * E_exchange_lr
                   + E_xc + E_nn)

        aux = pack_energy_aux(
            kinetic=E_kinetic,
            hartree=E_hartree,
            xc=E_xc,
            external=E_external,
            nelec=nelec,
        )
        return E_total, aux


def refresh(ks: SplatKS, basis, *, pad_to: int | None = None,
            like: "SplatState | None" = None) -> SplatState:
    """Rebuild the refresh products for the CURRENT splats.

    Host-side; call at init and every time the splats have moved enough (the
    trainers do it on a fixed cadence). Builds exactly what ``ks`` needs:

    - the significant-pair list when ``ks`` screens (padded to ``pad_to`` with
      overflow keeping the largest-overlap pairs, so the jitted step keeps a
      constant shape and compiles once — required on multi-GPU);
    - the frozen product aux (and the K-aux for hybrids) when ``ks`` density-fits.

    ``like=previous_state`` inherits the previous pair pad, so a hand-rolled
    refresh loop keeps constant shapes (and the compile-once contract) without
    threading the pad explicitly:

        state = refresh(ks, basis0, pad_to=...)      # fix the pad once
        ...
        state = refresh(ks, basis_t, like=state)     # same shapes forever
    """
    if pad_to is None and like is not None and like.pairs is not None:
        pad_to = int(like.pairs[0].shape[0])
    cells = None
    if ks.cell_part is not None:
        # Inherit the splat-list widths from the previous state, the same contract `pad_to` gives
        # the pair list: constant shapes for the whole run (see screening.neighbor_pairs for why
        # a mid-run recompile is fatal on a mesh).
        s_pad = ([int(g[2].shape[-1]) for g in like.cells]     # (ndev, nc, S) -> S
                 if like is not None and like.cells is not None else None)
        cells, _ = _cells.cell_lists(basis, ks.grid_points, ks.cell_part,
                                     eps=ks.cell_spec.eps, s_pad=s_pad,
                                     headroom=ks.cell_spec.headroom,
                                     n_sp_buckets=ks.cell_spec.sp_buckets)
    pairs = None
    if ks.screen is not None:
        pairs = _scr.neighbor_pairs(basis, eps=ks.screen.eps, pad_to=pad_to,
                                    unique=ks.screen.unique)
        if ks.mesh is not None:
            # Deal the pairs round-robin across the mesh BEFORE P("g") splits them, so no device
            # inherits a block of pure padding — see shard.stripe_pairs.
            from gs_dft.ks import shard as _sharded
            pairs = _sharded.stripe_pairs(*pairs, int(ks.mesh.size))
    aux = aux_K = cholV = None
    if isinstance(ks.coulomb, (ProductDFCoulomb, ScreenedProductDFCoulomb)):
        aux = init_product_aux(basis, scales=ks.coulomb.scales,
                               diagonal_only=ks.coulomb.diagonal)
        if ks.coulomb.hf_coeff != 0.0:
            aux_K = aux                     # the SAME product basis serves K (identical build)
        # Frozen aux ⇒ the RI-J metric is constant until the next refresh: factor it ONCE
        # (the per-step solve becomes an O(N_aux²) cho_solve — the only O(N_aux³) piece
        # of the step otherwise, ~13% of an insulin-scale step).
        from gs_dft.coulomb import ri as _cdf
        cholV = _cdf.aux_cholesky(aux, ks.coulomb.lam)
    return SplatState(pairs=pairs, aux=aux, aux_K=aux_K, cholV=cholV, cells=cells)


def _check_electrons(ks: SplatKS, model: SplatModel):
    """Host-boundary guard: the model's electron count must match the system's.

    A model built for one molecule silently evaluates on another's ``ks``
    otherwise — the dropped-identity failure class. Only active when the
    system carried an electron count (``nelec > 0``) and the occupations are
    concrete (host side)."""
    if ks.nelec:
        try:
            n = float(jnp.sum(model.occupations))
        except Exception:                      # traced occupations: skip (jit interior)
            return
        if abs(n - ks.nelec) > 1e-6:
            raise ValueError(
                f"model carries {n:g} electrons but the system has {ks.nelec}; "
                f"the model and SplatKS describe different systems.")


def init_model(system: object, M: int, key: "jax.Array", *,
               init: "object | None" = None) -> SplatModel:
    """Build a :class:`SplatModel`: ``M`` splats + random MO coefficients.

    Args:
        system: a dftax ``Molecule`` (the electron count sets the closed-shell
            occupations).
        M: number of splats.
        key: PRNG key.
        init: ``None`` — the atom-centered random init (anisotropically
            jittered off the spectral chart's isotropy singularity); or a
            :func:`~gs_dft.basis.chem.chem` spec for the data-free
            chemistry-informed init (element exponents, bond-centered splats,
            bond-aligned anisotropy — cuts optimization 3-4x).
    """
    import jax.random as jr
    from gs_dft.basis.spectral import init_spectral_splats
    from gs_dft.basis.chem import ChemInit, init_chem_splats

    coords, charges, nelec, _symbols = _resolve_system(system)
    if not nelec:
        raise ValueError("init_model needs a system with an electron count "
                         "(a Molecule, not a raw (coords, charges) pair).")
    n_occ = nelec // 2
    k1, k2 = jr.split(key)
    if init is None:
        basis = init_spectral_splats(coords, M, key=k1, aniso_jitter=0.02)
    elif isinstance(init, ChemInit):
        basis = init_chem_splats(coords, charges, M, key=k1, **init.kwargs())
    else:
        raise TypeError(f"init must be None or a chem() spec, got {init!r}")
    C = 0.1 * jr.normal(k2, (M, n_occ))
    return SplatModel(basis=basis, C=C, occupations=2.0 * jnp.ones(n_occ))
