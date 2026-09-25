"""Parameters, FLOPs and peak memory, defined once for every experiment.

Parameters are counted on both sides. A Gaussian basis is fixed, but its exponents and contraction
coefficients are still numbers the representation needs, about 6.6 to 7.5 per basis function
against the splat chart's 9; the orbital coefficients add n_functions x n_occ on both sides. For
water at cc-pVDZ this gives 278 parameters for the Gaussian basis and 336 for the matched splat
cloud. The basis share falls as 1/n_occ, so it matters on small molecules and vanishes on large ones.
"""
from __future__ import annotations

import numpy as np

SPLAT_PARAMS_PER_FN = 9        # 3 center + 3 log-eigenvalue + 3 orientation (unit quaternion)


def splat_params(M: int, n_occ: int) -> dict:
    """Free parameters of a splat cloud: the chart plus the MO coefficients."""
    M, n_occ = int(M), int(n_occ)
    basis = SPLAT_PARAMS_PER_FN * M
    coeff = M * n_occ
    return {"basis": basis, "coeff": coeff, "total": basis + coeff}


def gto_basis_params(basis_data) -> int:
    """Exponents + contraction coefficients of a contracted Gaussian basis."""
    e = np.asarray(basis_data.exponents)
    n_prim = int((e > 0).sum())
    return 2 * n_prim                       # one exponent and one coefficient per primitive


def gto_params(basis_data, nao: int, n_occ: int) -> dict:
    """Parameters of a Gaussian calculation: the tabulated basis plus the MO coefficients."""
    basis = gto_basis_params(basis_data)
    coeff = int(nao) * int(n_occ)
    return {"basis": basis, "coeff": coeff, "total": basis + coeff}


def params_from_row(row: dict, n_occ: int, basis_data=None) -> dict | None:
    """Parameters for a parsed `RESULT` row, splat or GTO, or None if it carries neither size.

    Lets a plotter work straight off the committed record: a splat row carries `M`, a GTO row
    carries `nao` and needs its `basis_data` to price the tabulated half.
    """
    if row.get("M") is not None:
        return splat_params(int(row["M"]), n_occ)
    if row.get("nao") is not None and basis_data is not None:
        return gto_params(basis_data, int(row["nao"]), n_occ)
    return None


_PEAK_KEYS = ("peak_train_mb", "peak_gpu_mb", "peak_mb")


def peak_mb_from_row(row: dict) -> float | None:
    """Peak device memory in MB from a parsed `RESULT` row, under either spelling.

    Returns None when absent and when the probe reported its -1.0 sentinel (CPU, or a JAX without
    the stat), so an unavailable measurement cannot be plotted as "this run used no memory".
    """
    for k in _PEAK_KEYS:
        if row.get(k) is not None:
            v = float(row[k])
            return None if v < 0 else v
    return None


def breakeven_fraction(n_occ: int, gto_basis_per_fn: float = 7.0) -> float:
    """The M/nao below which a splat cloud uses FEWER total parameters than the Gaussian basis.

    Splats win when ``M*(9 + n_occ) < nao*(gto_basis_per_fn + n_occ)``. Returns that ratio. The
    default per-function figure is the measured cc-pVXZ range (6.6-7.5); pass the exact value from
    `gto_basis_params` when a real comparison is being made rather than a rule of thumb.
    """
    return (gto_basis_per_fn + n_occ) / (SPLAT_PARAMS_PER_FN + n_occ)
