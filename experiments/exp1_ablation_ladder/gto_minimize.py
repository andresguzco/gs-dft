"""The Gaussian baseline of Figure 2(a), optimized the same way as the splats.

``dftax.ks.minimize`` minimizes the KS energy directly over orthonormalized coefficients with an
Optax optimizer, so the Gaussian basis follows the same kind of update as the splat cloud and the
comparison isolates the representation.

    uv run python -m experiments.exp1_ablation_ladder.gto_minimize <system> <basis> [steps]
"""
import contextlib
import io
import re
import sys
import time

import optax

from dftax.basis.loader import build_basis_data
from dftax.energy.xc import PBE
from dftax.ks import KS, df, minimize
from gs_dft import nao
from gs_dft.ks.train import native_grid
from experiments.common import probes, resources, results, systems

GRID = 3
AUX = "def2-universal-jkfit"

# Verbatim from `dftax.ks.minimize`'s verbose branch:
#     print(f"  min {step:4d}: E={float(e):.10f}  |g|={gnorm:.2e}")
_MIN_LINE = re.compile(r"^\s*min\s+(\d+):\s+E=(-?[\d.]+)\s+\|g\|=(\S+)")


class _Retrace(io.TextIOBase):
    """Turn the engine's per-step line into our `TRACE` line, streaming, as it is printed."""

    def __init__(self, out, rung, t0):
        self.out, self.rung, self.t0, self._buf = out, rung, t0, ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = _MIN_LINE.match(line)
            if m:
                self.out.write(f"TRACE rung={self.rung} step={int(m.group(1))} "
                               f"t={time.time() - self.t0:.3f} E={float(m.group(2)):.8f}\n")
            else:
                self.out.write(line + "\n")
            self.out.flush()
        return len(s)


def main(system, basis, steps=6000, rung="p1_gto", lr=None):
    steps = int(steps)
    mol = systems.molecule(system=system, basis=basis)
    n_occ = int(mol.nelectron // 2)
    print(f"# exp1 {rung}: {system}/{basis} nao={int(nao(mol))} DIRECT MINIMIZATION "
          f"(not SCF) steps={steps}", flush=True)
    ks = KS(mol, PBE(), grid=native_grid(mol, GRID, chunk=None), coulomb=df(AUX, chunk=None))
    opt = optax.adam(0.3 if lr is None else float(lr))
    t0 = time.time()
    failed, E, conv, n_iter = "none", float("nan"), False, 0
    try:
        with contextlib.redirect_stdout(_Retrace(sys.stdout, rung, t0)):
            res = minimize(ks, opt, max_steps=steps, verbose=True)
        E, conv, n_iter = float(res.e_tot), bool(res.converged), int(res.n_iter)
    except Exception as exc:                                   # noqa: BLE001
        failed = ("oom" if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc)
                  else type(exc).__name__)
        print(f"# {rung} FAILED ({failed}): {str(exc)[:160]}", flush=True)
    bd = build_basis_data(mol.symbols, mol.atom_coords(), mol.basis, spherical=True)
    par = resources.gto_params(bd, int(nao(mol)), n_occ)
    print(results.result("gto_min", system, rung=rung, basis=basis, nao=int(nao(mol)), n_occ=n_occ,
                         params=par["total"], params_basis=par["basis"], E=E, conv=conv,
                         n_iter=n_iter, steps=steps, opt=f"adam{0.3 if lr is None else lr}",
                         failed=failed, peak_train_mb=round(probes.peak_gpu_mb()),
                         wall_s=round(time.time() - t0, 1)), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
