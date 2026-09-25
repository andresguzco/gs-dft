"""Heavy artifacts follow DFTAX_CKPT_DIR / DFTAX_WANDB_DIR; unset, they stay where they were."""
import os

from experiments.common import paths


def test_artifact_dir_follows_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DFTAX_CKPT_DIR", str(tmp_path / "ckpt"))
    assert paths.artifact("results", "x_ckpt") == str(tmp_path / "ckpt" / "x_ckpt")
    assert os.path.isdir(tmp_path / "ckpt")
    monkeypatch.setenv("DFTAX_WANDB_DIR", str(tmp_path / "wb"))
    assert paths.wandb_dir("results") == str(tmp_path / "wb")


def test_artifact_dir_default_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("DFTAX_CKPT_DIR", raising=False)
    monkeypatch.delenv("DFTAX_WANDB_DIR", raising=False)
    d = str(tmp_path / "results")
    assert paths.artifact(d, "x_ckpt") == os.path.join(d, "x_ckpt")
    assert paths.wandb_dir(d) == d
    assert paths.wandb_dir(None) is None
