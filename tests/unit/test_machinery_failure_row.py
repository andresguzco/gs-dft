"""A failed machinery rung still emits a parseable ``RESULT`` row carrying ``failed=``.

In Figure 2(b) the dense arm is expected to run out of memory, and that ceiling is the measurement.
"""
import os

import pytest
from hydra import compose, initialize_config_dir

from experiments.common.results import parse
import pathlib

from experiments.exp1_ablation_ladder import machinery

_CONF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "conf")


def _cfg(**over):
    with initialize_config_dir(version_base=None, config_dir=_CONF):
        return compose(config_name="config",
                       overrides=["experiment=exp1_machinery", "system=h2o", "basis=sto-3g",
                                  "rung=P2_df", "m=8", "steps=2", "grid_level=1",
                                  "wandb_mode=disabled"] + [f"{k}={v}" for k, v in over.items()])


def test_free_mb_is_defined_and_never_raises():
    """The context helper runs ON the failure path, so it must not be the thing that fails."""
    assert callable(machinery._free_mb)
    assert isinstance(machinery._free_mb(), int)


def test_failed_rung_still_emits_a_parseable_result(monkeypatch, capsys):
    """Force the trainer to raise; the runner must report rather than crash."""
    def boom(*a, **k):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 115.13GiB")
    monkeypatch.setattr(machinery, "train", boom)

    machinery.run(_cfg())

    rows = [parse(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT ")]
    assert rows, "a failed rung emitted NO RESULT row — the ceiling is lost"
    row = rows[0]
    assert row["failed"] == "oom", f"an allocation failure must be classified: {row}"
    assert row["finite"] == "False" and row["kinetic_ok"] == "False"
    assert "gpu_free_mb" in row


def test_ms_step_counts_only_the_steps_this_process_ran(monkeypatch, capsys, tmp_path):
    """    A job clipped at 2400/3000 and requeued walks 600 steps; dividing its wall by 3000 understates
    the cost 5x. The denominator has to come from the checkpoint, which is the only thing that
    knows how far the previous attempt got."""
    monkeypatch.setattr(machinery, "_CKPT_DIR", tmp_path)
    base = pathlib.Path(machinery.ckpt_base("P2_df", "h2o", 8, 0, 2, _cfg().grid_level))
    (base.with_suffix(".step")).write_text("1")          # 1 of 2 steps already done

    machinery.run(_cfg())

    row = [parse(ln) for ln in capsys.readouterr().out.splitlines()
           if ln.startswith("RESULT ")][0]
    assert row["resumed_from"] == "1", row
    # steps=2, resumed at 1 => the denominator is 1, so ms_step == wall_s * 1000 (not half of it).
    assert float(row["ms_step"]) == pytest.approx(1e3 * float(row["wall_s"]), rel=0.05), row


def test_the_checkpoint_name_separates_runs_that_differ(tmp_path, monkeypatch):
    """    A different `M` or a different rung announces itself: `tree_deserialise_leaves` raises on the
    shape mismatch, the run dies, and the row says so. `grid_level` changes no array shape at all,
    so a grid-3 checkpoint loads cleanly into a grid-2 run and the job carries on under a label for
    a quadrature it never used. `steps` is the same kind of quiet: the Adam schedule rides along
    with the moments, so a resumed run is simply on the wrong learning rate."""
    monkeypatch.setattr(machinery, "_CKPT_DIR", tmp_path)
    ref = machinery.ckpt_base("P2_df", "h2o", 8, 0, 2, 3)
    for kwargs in ({"rung": "P2_fast"}, {"M": 16}, {"seed": 1}, {"steps": 4}, {"grid_level": 2}):
        args = dict(rung="P2_df", system="h2o", M=8, seed=0, steps=2, grid_level=3) | kwargs
        assert machinery.ckpt_base(**args) != ref, \
            f"{list(kwargs)[0]} does not reach the checkpoint name — runs that differ would collide"


def test_lindep_row_carries_both_spectrum_ends():
    from gs_dft.ks import dense_gram_eig_ends
    import jax.random as jr
    from gs_dft import init_model
    from experiments.common import systems

    mol = systems.molecule(_cfg())
    model = init_model(mol, 8, jr.PRNGKey(0))
    lo, hi = dense_gram_eig_ends(model.basis, model.C)
    assert lo <= hi and hi > 0


def test_a_nondescript_exception_is_classified_by_type(monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("something else entirely")
    monkeypatch.setattr(machinery, "train", boom)
    machinery.run(_cfg())
    rows = [parse(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("RESULT ")]
    assert rows and rows[0]["failed"] == "ValueError", rows
