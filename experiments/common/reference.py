"""The Gaussian reference: one converged dftax DF-RKS calculation, retried with a level shift if
the SCF does not converge.

Runs on the process's default device, on the same engine, functional and grid as the splat side.
Unlike ``gs_dft.benchmark.build_reference`` it takes an arbitrary Molecule (peptides, charged
anions, points on a dissociation curve).
"""
import time
from typing import NamedTuple

from dftax.ks import KS, scf, df
from gs_dft import nao
from gs_dft.ks.train import native_grid


class RefResult(NamedTuple):
    e: float
    nao: int
    converged: bool
    n_iter: int
    wall_s: float


def gto_reference(mol, xc, *, grid_level=3, mode="df", chunk=4096, max_iter=100, rescue=True):
    """Converged RKS energy for ``mol`` with functional ``xc`` on a level-``grid_level`` Becke grid.

    ``mode`` picks the Coulomb path: ``"df"`` (default) uses the standard ``def2-universal-jkfit``
    RI-JK aux — O(nao²·naux), the only feasible route past ~a dozen heavy atoms; ``"conv"`` uses the
    exact in-core 4-center ERI (the most accurate reference, for the small water/ethanol ladders).
    On non-convergence, retry once with a 0.2 virtual level shift (the diffuse-anion / stretched-bond
    salvage) unless ``rescue=False``.
    """
    coulomb = df("def2-universal-jkfit") if mode == "df" else None
    ks = KS(mol, xc, grid=native_grid(mol, grid_level, chunk=chunk), coulomb=coulomb)
    t0 = time.perf_counter()
    res = scf(ks, max_iter=max_iter)
    if rescue and not res.converged:
        res = scf(ks, max_iter=2 * max_iter, level_shift=0.2)
    return RefResult(e=float(res.e_tot), nao=int(nao(mol)), converged=bool(res.converged),
                     n_iter=int(res.n_iter), wall_s=time.perf_counter() - t0)
