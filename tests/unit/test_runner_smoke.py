"""Each experiment runner imports and runs a few steps to a parseable ``RESULT`` line.

Every case runs the runner as a subprocess on a two-atom system and checks exit code 0 and one
``RESULT`` line of the expected kind. Marked slow: each case compiles from scratch.
"""
import subprocess
import sys

import pytest

from experiments.common.results import parse

pytestmark = pytest.mark.slow

_BASE = ["basis=sto-3g", "grid_level=1", "wandb_mode=disabled"]
RUNNERS = [
    ("exp5_splat_run", "experiments.exp5_basis_accuracy.splat_run",
     ["experiment=exp5_splat_run", "system=h2", "m=6", "steps=2", "monitor=1", "partial_every=0"], "splat"),
    ("exp5_baseline_gto", "experiments.exp5_basis_accuracy.baseline_gto",
     ["experiment=exp5_baseline_gto", "system=h2", "mode=conv"], "gto"),
    ("exp3_gto_curve", "experiments.exp3_lih_dissociation.gto_curve",
     ["experiment=exp3_gto_curve", "system=lih", "r=1.6"], "gto"),
    ("exp3_splat_curve", "experiments.exp3_lih_dissociation.splat_curve",
     ["experiment=exp3_splat_curve", "system=lih", "r=1.6", "m=8", "steps=2"], "splat"),
    ("exp2_train_ckpt", "experiments.exp2_observables.train_ckpt",
     ["experiment=exp2_train_ckpt", "system=h2", "m=6", "steps=2"], "ckpt"),
    ("exp2_gto_ref", "experiments.exp2_observables.gto_ref",
     ["experiment=exp2_gto_ref", "system=h2"], "gto"),
    ("exp6_bench", "experiments.exp6_cost_benchmark.bench",
     ["experiment=exp6_bench", "system=h2", "engine=splat", "m=6", "steps=2"], "bench"),
    ("exp4_splat_anion", "experiments.exp4_anion.splat_anion",
     ["experiment=exp4_splat_anion", "system=f_anion", "m=8", "steps=2", "grid_check=0"], "splat"),
    ("exp4_gto_refs", "experiments.exp4_anion.gto_refs",
     ["experiment=exp4_gto_refs", "system=f_anion"], "gto"),
    ("exp8_run_ladder", "experiments.exp8_data_free_init.run_ladder",
     ["experiment=exp8_run_ladder", "system=h2", "schemes=[S0]", "seeds=[0]", "steps=2"], "splat"),
    ("exp7_train", "experiments.exp7_size_ladder.train",
     ["experiment=exp7_train", "system=h2", "steps=5", "monitor=1"], "train"),
    # msweep: both the exact arm and a hybrid arm, on a tiny system (covers coulomb= and xc= wiring)
    ("exp5_msweep_exact", "experiments.exp5_basis_accuracy.msweep",
     ["experiment=exp5_msweep", "system=h2", "m=6", "coulomb=exact", "xc=pbe", "steps=2"], "splat"),
    ("exp5_msweep_hybrid", "experiments.exp5_basis_accuracy.msweep",
     ["experiment=exp5_msweep", "system=h2", "m=6", "coulomb=df", "xc=pbe0", "steps=2"], "splat"),
]


@pytest.mark.parametrize("label,module,overrides,kind", RUNNERS, ids=[r[0] for r in RUNNERS])
def test_runner_emits_result(label, module, overrides, kind, tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", module, *_BASE, *overrides, f"hydra.run.dir={tmp_path}"],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"{label} exited {proc.returncode}\nSTDERR:\n{proc.stderr[-3000:]}"
    results = [d for ln in proc.stdout.splitlines()
               if (d := parse(ln)) is not None and d["_tag"] == "RESULT"]
    assert results, f"{label} printed no RESULT line\nSTDOUT:\n{proc.stdout[-2000:]}"
    assert results[-1]["kind"] == kind and "E" in results[-1], results[-1]
