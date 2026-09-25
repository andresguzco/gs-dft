"""Every ``experiment=`` config supplies every key its driver reads.

The check is static: compose the config, scan the driver for ``cfg.<key>`` accesses, and compare.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf" / "experiment"

#: experiment name -> the driver module that runs under it, as `launcher: experiment=` pairs it.
#: A config with no driver here is unused by any launcher and should be retired, not exempted.
DRIVERS = {
    "exp1_ladder": "experiments/exp1_ablation_ladder/ladder.py",
    "exp1_machinery": "experiments/exp1_ablation_ladder/machinery.py",
    "exp1_optsweep": "experiments/exp1_ablation_ladder/optimizers.py",
    "exp2_gto_ref": "experiments/exp2_observables/gto_ref.py",
    "exp2_observables": "experiments/exp2_observables/observables.py",
    "exp2_train_ckpt": "experiments/exp2_observables/train_ckpt.py",
    "exp3_gto_curve": "experiments/exp3_lih_dissociation/gto_curve.py",
    "exp3_splat_curve": "experiments/exp3_lih_dissociation/splat_curve.py",
    "exp3_continuation": "experiments/exp3_lih_dissociation/splat_continuation.py",
    "exp4_gto_refs": "experiments/exp4_anion/gto_refs.py",
    "exp4_splat_anion": "experiments/exp4_anion/splat_anion.py",
    "exp5_baseline_gto": "experiments/exp5_basis_accuracy/baseline_gto.py",
    "exp5_msweep": "experiments/exp5_basis_accuracy/msweep.py",
    "exp5_refresh": "experiments/exp5_basis_accuracy/splat_run.py",
    "exp5_splat_run": "experiments/exp5_basis_accuracy/splat_run.py",
    "exp6_bench": "experiments/exp6_cost_benchmark/bench.py",
    "exp7_train": "experiments/exp7_size_ladder/train.py",
    "exp7_forces": "experiments/exp7_size_ladder/forces.py",
    # Driven from Python and the smoke tests rather than a .sh launcher, so a scan of
    # experiments/**/*.sh does not find them.
    "exp8_run_ladder": "experiments/exp8_data_free_init/run_ladder.py",
}

#: Keys a driver reads only under a branch the launchers never take, or supplies itself.
ALLOWED_MISSING = {
    "exp7_train": {"m"},        # exp7 sizes from m_mult unless a launcher passes m= explicitly
    "exp7_forces": {"m"},       # same
}


def _keys_read(driver: Path) -> set[str]:
    """`cfg.<key>` and `cfg.get("<key>")` accesses in a driver."""
    src = driver.read_text(errors="replace")
    keys = set(re.findall(r"cfg\.([a-z_][a-z0-9_]*)", src))
    keys |= set(re.findall(r"""cfg\.get\(\s*['"]([a-z_][a-z0-9_]*)['"]""", src))
    return keys - {"get", "keys", "items", "values"}


def test_every_experiment_config_has_a_driver():
    """A config no launcher uses is dead weight; adding one here is how it gets covered."""
    on_disk = {p.stem for p in CONF.glob("*.yaml")}
    assert on_disk - set(DRIVERS) == set(), (
        "experiment configs with no driver mapped — retire them or add them to DRIVERS"
    )


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_config_supplies_every_key_its_driver_reads(name):
    hydra = pytest.importorskip("hydra")
    driver = ROOT / DRIVERS[name]
    if not driver.exists():
        pytest.skip(f"driver missing: {DRIVERS[name]}")

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])

    missing = sorted(k for k in _keys_read(driver) if k not in cfg)
    missing = [k for k in missing if k not in ALLOWED_MISSING.get(name, set())]
    assert not missing, (
        f"experiment={name} does not supply {missing}, which {DRIVERS[name]} reads. "
        f"Every run under this experiment would die on ConfigAttributeError before taking a step."
    )
