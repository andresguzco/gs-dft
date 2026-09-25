"""Weights & Biases tracking: one project per experiment directory, and no entry point raises when
wandb is missing, broken or offline.
"""
import pytest

from experiments.common import tracking


@pytest.mark.parametrize("module,project", [
    ("experiments.exp5_basis_accuracy.msweep", "gs-dft-exp5-basis-accuracy"),
    ("experiments.exp5_basis_accuracy.baseline_gto", "gs-dft-exp5-basis-accuracy"),
    ("experiments.exp5_basis_accuracy.splat_run", "gs-dft-exp5-basis-accuracy"),   # same experiment
    ("experiments.exp7_size_ladder.train", "gs-dft-exp7-size-ladder"),
    ("experiments.common.something", "gs-dft-common"),
    ("not_an_experiment", "gs-dft-misc"),
])
def test_project_is_derived_from_the_module(module, project):
    assert tracking.project_for(module) == project


def test_every_runner_maps_to_its_own_experiment_project():
    """The whole point: sibling runners share a project, different experiments never collide."""
    import glob, os
    projects = {}
    for path in glob.glob("experiments/exp*/*.py"):
        exp = os.path.basename(os.path.dirname(path))
        mod = f"experiments.{exp}.{os.path.basename(path)[:-3]}"
        projects.setdefault(tracking.project_for(mod), set()).add(exp)
    assert projects, "no runners found"
    for project, exps in projects.items():
        assert len(exps) == 1, f"{project} would collect runs from {exps}"


def test_project_survives_python_dash_m():
    import subprocess, sys, os, textwrap, tempfile, pathlib
    root = pathlib.Path(tempfile.mkdtemp())
    pkg = root / "experiments" / "exp5_basis_accuracy"
    pkg.mkdir(parents=True)
    (root / "experiments" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    # tracking.py is import-standalone (stdlib only at module level), so load it BY PATH: making the
    # temp tree a real `experiments` package would shadow the repo's own and break the import.
    src = os.path.join(os.getcwd(), "experiments", "common", "tracking.py")
    (pkg / "probe.py").write_text(textwrap.dedent(f"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("tracking", {src!r})
        tracking = importlib.util.module_from_spec(spec); spec.loader.exec_module(tracking)
        tracking.init({{"wandb_mode": "disabled"}}, __name__, name="probe")
    """))
    out = subprocess.run([sys.executable, "-m", "experiments.exp5_basis_accuracy.probe"],
                         cwd=root, capture_output=True, text=True, timeout=120)
    assert "gs-dft-exp5-basis-accuracy" in out.stdout, (
        f"resolved the wrong project under -m\nSTDOUT:{out.stdout}\nSTDERR:{out.stderr[-800:]}")
    assert "gs-dft-misc" not in out.stdout


def test_resolve_module_prefers_spec_then_package():
    """The two things `-m` fills in, checked directly."""
    class Spec:
        name = "experiments.exp3_lih_dissociation.gto_curve"
    assert tracking._resolve_module("__main__", {"__spec__": Spec()}) == Spec.name
    assert tracking._resolve_module("__main__", {"__package__": "experiments.exp4_anion"}) \
        == "experiments.exp4_anion"
    # an already-real module name is passed through untouched
    assert tracking._resolve_module("experiments.exp5_basis_accuracy.msweep", {}) \
        == "experiments.exp5_basis_accuracy.msweep"


def test_disabled_returns_a_working_no_op():
    run = tracking.init({"wandb_mode": "disabled"}, "experiments.exp5_basis_accuracy.msweep", name="x")
    assert run.enabled is False
    run.log({"a": 1}, step=0); run.summary({"b": 2}); run.finish()      # must not raise


def test_init_survives_a_broken_wandb(monkeypatch):
    """Simulate the real failure: wandb present but `init` raising (bad key, network, version)."""
    import sys, types
    mod = types.ModuleType("wandb")
    mod.Settings = lambda **kw: None
    def boom(**kw):
        raise RuntimeError("no API key")
    mod.init = boom
    monkeypatch.setitem(sys.modules, "wandb", mod)
    run = tracking.init({"wandb_mode": "online"}, "experiments.exp5_basis_accuracy.msweep", name="x")
    assert run.enabled is False                       # degraded, not raised
    run.log({"a": 1}); run.summary({"b": 2}); run.finish()


def test_record_is_a_no_op_without_an_active_run():
    """`results.result()` calls this on EVERY emit, including in plotters and tests. It must be inert
    when nothing started a run — otherwise the shared emitter becomes a liability."""
    tracking._ACTIVE = None
    tracking.record("RESULT", "gto", "water", {"E": -76.0})            # must not raise


def test_result_emitter_still_returns_the_line_with_tracking_hooked():
    """The hook must not change the RESULT grammar — that grammar is a parsing contract."""
    from experiments.common import results
    tracking._ACTIVE = None
    line = results.result("gto", "water", basis="cc-pvdz", nao=24, E=-76.281316)
    assert line.startswith("RESULT kind=gto system=water ")
    assert "E=-76.28131600" in line
