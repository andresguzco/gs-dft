"""Weights & Biases logging for every runner, with one project per experiment directory.

    experiments.exp5_basis_accuracy.baseline_gto  ->  project "gs-dft-exp5-basis-accuracy"

Logging is off unless ``wandb_mode`` is ``online`` or ``offline``, ``cfg.wandb_project`` overrides
the project, and no entry point raises when wandb is missing or unreachable.
"""
from __future__ import annotations

import os
import re

_PREFIX = "gs-dft"


def project_for(module_name: str) -> str:
    """``experiments.exp5_basis_accuracy.baseline_gto`` -> ``gs-dft-exp5-basis-accuracy``.

    Also accepts the PACKAGE form ``experiments.exp5_basis_accuracy`` — see :func:`_resolve_module`.
    Falls back to ``gs-dft-misc`` for anything not under an ``experiments.<dir>`` package, so a stray
    caller lands somewhere obvious instead of inventing a project per module.
    """
    parts = [p for p in str(module_name).split(".") if p]
    if len(parts) >= 2 and parts[0] == "experiments":
        return f"{_PREFIX}-{parts[1].replace('_', '-')}"
    return f"{_PREFIX}-misc"


def _resolve_module(module_name: str, frame_globals: dict) -> str:
    """Turn what a runner passes into a real dotted module path."""
    if module_name and module_name != "__main__":
        return module_name
    spec = frame_globals.get("__spec__")
    name = getattr(spec, "name", None) if spec is not None else None
    return name or frame_globals.get("__package__") or module_name


class _NullRun:
    """What every caller gets when logging is off or unavailable. Same surface, does nothing."""
    enabled = False

    def log(self, *_a, **_k):
        pass

    def summary(self, *_a, **_k):
        pass

    def finish(self, *_a, **_k):
        pass


class _Run:
    def __init__(self, run):
        self._run = run
        self.enabled = True

    def log(self, metrics: dict, step: int | None = None):
        try:
            self._run.log(dict(metrics), **({"step": int(step)} if step is not None else {}))
        except Exception:
            pass

    def summary(self, metrics: dict):
        """Final scalars — what you want on the runs TABLE, not the step chart."""
        try:
            for k, v in dict(metrics).items():
                self._run.summary[k] = v
        except Exception:
            pass

    def finish(self):
        try:
            self._run.finish()
        except Exception:
            pass


def init(cfg, module_name: str, *, name: str, extra: dict | None = None, dir: str | None = None):
    """Start a run for ``module_name``'s experiment. Returns a handle that is always safe to call.

    ``cfg.wandb_mode`` — ``disabled`` (default) | ``offline`` | ``online``.
    ``cfg.wandb_project`` — optional override; ``None`` means derive from ``module_name``.
    ``name`` is the run name: make it the same string the experiment uses on disk (its tag / log
    stem), so a wandb run and its local log can be matched up without guesswork.
    """
    global _ACTIVE
    import inspect
    caller = inspect.currentframe().f_back
    module_name = _resolve_module(module_name, caller.f_globals if caller else {})
    project = _get(cfg, "wandb_project", None) or project_for(module_name)
    mode = str(_get(cfg, "wandb_mode", "disabled"))
    print(f"# wandb: {project}/{_slug(name)} ({mode})", flush=True)
    if mode == "disabled":
        _ACTIVE = _NullRun()
        return _ACTIVE
    try:
        import wandb
        from omegaconf import OmegaConf

        conf = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        from experiments.common import paths
        run = wandb.init(project=project, name=_slug(name), mode=mode, dir=paths.wandb_dir(dir),
                         config={**conf, **(extra or {})})
        handle = _Run(run)
    except Exception as e:                       # never let logging kill a run — see module note
        print(f"# wandb unavailable ({type(e).__name__}: {e}); continuing without it", flush=True)
        handle = _NullRun()
    _ACTIVE = handle                             # what results.result()/partial() mirror into
    return handle


_ACTIVE: "_Run | _NullRun | None" = None


def record(tag: str, kind: str, system: str, kv: dict) -> None:
    """Mirror a RESULT-family record to the active run. Called by ``results.result``/``partial``.

    A final ``RESULT`` becomes wandb **summary** (the scalars that belong on the runs table); a
    ``RESULT_PARTIAL`` becomes a **logged step** (the training curve). No-op when no run is active,
    which is what makes the hook safe to sit inside the shared emitter.
    """
    run = _ACTIVE
    if run is None or not getattr(run, "enabled", False):
        return
    num = {}
    for k, v in kv.items():
        if isinstance(v, bool):
            num[k] = int(v)
        elif isinstance(v, (int, float)):
            num[k] = v
    if tag == "RESULT_PARTIAL":
        run.log(num, step=int(kv.get("step", 0)))
    else:
        run.summary({"kind": kind, "system": system, **num})


def _get(cfg, key, default):
    try:
        v = cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
    except Exception:
        return default
    return default if v is None and default is not None else v


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "run"
