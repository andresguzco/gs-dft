"""Turn config values into library values: XC and Coulomb factories, the automatic screening rule,
SplatKS assembly and the level-5 grid re-evaluation.
"""
import jax.numpy as jnp
from dftax.energy.xc import (PBE, PBE0, B3LYP, R2SCAN, WB97MV, WB97X, XCFunctional,
                             DensityFunctional)
from gs_dft.ks.energy import SplatKS
from gs_dft.ks.train import native_grid, evaluate
from gs_dft.ks.terms import df, exact, pairlist, cellgrid, SCREEN_PAD

class _NoDensityFunctional(DensityFunctional):
    """The zero density functional. HF has no exchange-correlation term, and the composite base
    requires an `exchange` and a `correlation` component, so this is what they are."""

    name = "zero"
    xc_type = "LDA"

    def __call__(self, *args):
        return jnp.zeros_like(args[0])


class HartreeFock(XCFunctional):
    """Pure Hartree-Fock: full exact exchange, no density functional at all.

    Here so the orbital energies MEAN something. Koopmans' theorem holds for HF —
    ``-eps_HOMO`` is the ionization potential — while a KS eigenvalue is not an observable, which
    is why the GW100 benchmark tabulates HF, CCSD(T) and experimental HOMOs but no PBE one."""

    name = "HF"
    xc_type = "LDA"                      # no gradients needed: there is no density functional
    hf_coeff = 1.0

    exchange = _NoDensityFunctional()
    correlation = _NoDensityFunctional()

    def __call__(self, density, grad_density=None):
        return jnp.zeros_like(density)


_XC = {"pbe": PBE, "pbe0": PBE0, "b3lyp": B3LYP, "r2scan": R2SCAN,
       "wb97x": WB97X, "wb97m-v": WB97MV, "hf": HartreeFock}


def xc_of(name):
    """XC functional object from a schema string. The ONE reader of ``cfg.xc`` — its existence is
    what lets ``xc`` live in the shared schema honestly (see conf/config.yaml)."""
    try:
        return _XC[str(name).lower()]()
    except KeyError:
        raise ValueError(f"unknown xc {name!r} (use: {'|'.join(_XC)})")


def use_streamed_vv10():
    """Point the ENGINE's VV10 at our streamed-reverse implementation. Call before building a `KS`.

    `gs_dft.ks.nlc.vv10_energy` is a drop-in with the same signature whose forward is
    **bit-identical** to the engine's and whose four cotangents match autodiff-through-the-engine to
    better than 1e-10 relative (`tests/unit/test_nlc_remat.py`). The engine imports the symbol
    lazily inside the function body (`dftax/ks/terms.py`), so rebinding the module attribute is
    picked up on the next call.

    Returns the original, so a caller can restore it."""
    import dftax.energy.vv10 as _engine_vv10
    from gs_dft.ks.nlc import vv10_energy as _streamed
    original = _engine_vv10.vv10_energy
    _engine_vv10.vv10_energy = _streamed
    return original


def has_nlc(xc):
    """Does this functional carry the VV10 nonlocal correlation (the "-V" in ωB97M-V)?"""
    return float(getattr(xc, "nlc_b", 0.0) or 0.0) != 0.0


def coulomb_of(name, *, lam=1e-8):
    """Coulomb backend value: ``df`` (frozen pair-product RI-J/RI-K) or ``exact`` (streaming O(M²)
    exact E_J). Only the msweep experiment exposes this as a knob; everywhere else DF is the
    experiment's identity and the runner calls ``df()`` directly.

    ``lam`` is the aux-metric Tikhonov (``cfg.df_lam``); ``exact`` has no metric and ignores it."""
    n = str(name).lower()
    if n == "df":
        return df(lam=float(lam))
    if n == "exact":
        return exact()
    raise ValueError(f"unknown coulomb {name!r} (use: df|exact)")


def auto_screen(M, *, force=False):
    """Whether to build the screened (O(M) pair) path: M ≥ 300. ``force=True`` screens regardless,
    for a system whose pair count rather than M is the driver."""
    return bool(force or M >= 300)


def use_cells(M, force=False):
    """Block-sparse Morton-cell grid density — **OFF by default; opt in with ``force``.**

    TODO: vectorize `cell_lists` and cut the ``(nc,P,S)`` peak, then enable this by size."""
    return bool(force)


AUX_LADDER = {1: (1.0,), 2: (0.7, 1.4), 3: (0.5, 1.0, 2.0), 5: (0.25, 0.5, 1.0, 2.0, 4.0)}
"""``aux_mult`` -> the tempered exponent ladder for the frozen pair-product aux; ``naux = mult * M``.

An integer multiplier rather than a scales string because Hydra reads bare commas as list syntax, so
``aux_scales="0.5,1.0,2.0"`` needs shell-hostile quoting through a launcher.
"""


def aux_scales(mult):
    """Validate ``aux_mult`` and return its ladder. Raises rather than silently falling back to 1×."""
    m = int(mult)
    if m not in AUX_LADDER:
        raise ValueError(f"aux_mult must be one of {sorted(AUX_LADDER)}, got {mult}")
    return AUX_LADDER[m]


def splat_ks(mol, xc, *, grid_level, screened, chunk=4096, cells=False, df_lam=1e-8,
             aux_mult=1, screen_pad=SCREEN_PAD, gamma_chunk=512):
    """The production SplatKS every splat runner builds: native Becke grid + DF Coulomb, screened
    pairs when ``screened`` (grid chunked only then — the dense path takes the whole grid at once).

    ``cells`` (from :func:`use_cells`) swaps the grid density for the block-sparse Morton-cell path.
    Energy-identical and mesh-safe; default False so existing callers are unchanged."""
    return SplatKS(mol, xc,
                   grid=native_grid(mol, grid_level, chunk=chunk if screened else None),
                   coulomb=df(lam=float(df_lam), scales=aux_scales(aux_mult), chunk=int(gamma_chunk)),
                   screen=pairlist(eps=1e-7, pad=float(screen_pad)) if screened else None,
                   cells=cellgrid() if cells else None)


GRID_CONVERGED_MHA = 0.1
"""|E(train grid) − E(level-N)| above which a point is NOT grid-converged, in mHa.

Use the audit as a criterion, not a correction: subtracting the bias and keeping the point is not
sound. The variational principle bounds E only when the functional is
evaluated exactly; E_xc is a quadrature, and quadrature error has no sign, so an unconverged integral
produces a number that is not an upper bound on anything.

Separation measured on exp7's 48 rows (below-CBS vs not):

    below CBS   n=35   |bias| median 0.952 mHa   max 3.875
    above CBS   n=13   |bias| median 0.021 mHa   max 0.075

    |bias| > 0.10 mHa  flags 32/35 bad and 0/13 good     <- this default
    |bias| > 0.05 mHa  flags 34/35 bad and 2/13 good

Reference-free by construction, so it works on systems with no GTO ladder — unlike a CBS check.
It also flags ethanol M=432 (|bias| 4.55 mHa), which is exp4's observables checkpoint and an exp2
headline point. That is the criterion working, not misfiring.
"""


def grid_audit(mol, xc, model, level, e_train, state=None):
    """Level-``level`` honest-grid re-evaluation of a trained model: rebuild the KS on the finer
    grid, evaluate, and return ``(E_gridN, bias_mHa)`` with ``bias = (E_train − E_gridN)·1e3``.
    ``level`` of 0 means no re-evaluation and returns ``(None, None)``."""
    if not level:
        return None, None
    needs_df = float(getattr(xc, "hf_coeff_lr", 0.0) or 0.0) != 0.0
    ks_ck = SplatKS(mol, xc, grid=native_grid(mol, level),
                    coulomb=df() if needs_df else None)
    e_ck = float(evaluate(ks_ck, model, state))
    return e_ck, (e_train - e_ck) * 1e3


def grid_fields(mol, xc, model, level, e_train, state=None, tol=GRID_CONVERGED_MHA):
    """The audit as RESULT fields: ``E_grid<N>``, ``grid_bias_mha``, ``grid_converged``.

    One helper so every runner reports the criterion identically. ``grid_converged=False`` means the
    XC quadrature has not converged, so the energy is NOT a variational upper bound and must not be
    quoted — see :data:`GRID_CONVERGED_MHA`. Returns ``{}`` when the audit is off.
    """
    e_ck, bias = grid_audit(mol, xc, model, level, e_train, state)
    if e_ck is None:
        return {}
    return {f"E_grid{int(level)}": e_ck, "grid_bias_mha": bias,
            "grid_converged": bool(abs(bias) <= float(tol))}
