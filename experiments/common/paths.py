"""Where heavy artifacts go.

Checkpoints, collapse dumps and W&B run directories follow ``DFTAX_CKPT_DIR`` and
``DFTAX_WANDB_DIR`` when set, and otherwise stay in the experiment's ``results/``.
"""
import os


def artifact_dir(default: str) -> str:
    """The checkpoint directory: ``$DFTAX_CKPT_DIR`` if set, else ``default``; created."""
    d = os.environ.get("DFTAX_CKPT_DIR") or default
    os.makedirs(d, exist_ok=True)
    return d


def artifact(default_dir: str, name: str) -> str:
    """Full path of an artifact file ``name`` (see :func:`artifact_dir`)."""
    return os.path.join(artifact_dir(default_dir), name)


def wandb_dir(default: str | None) -> str | None:
    """The W&B run directory: ``$DFTAX_WANDB_DIR`` if set, else ``default``."""
    d = os.environ.get("DFTAX_WANDB_DIR") or default
    if d:
        os.makedirs(d, exist_ok=True)
    return d
