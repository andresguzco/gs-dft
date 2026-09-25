"""Every ``conf/experiment/*.yaml`` composes against the shared schema, and every key it adds is read.

1. ``experiment=<name>`` composes through the defaults list.
2. Every key not in ``conf/config.yaml`` is read as ``cfg.<key>``, ``cfg["<key>"]`` or
   ``cfg.get("<key>")`` under that experiment's package or ``experiments/common``.
"""
import glob
import os
import re

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONF = os.path.join(_REPO, "conf")
_EXP = os.path.join(_REPO, "experiments")

_SCHEMA_KEYS = set(OmegaConf.load(os.path.join(_CONF, "config.yaml")).keys()) - {"defaults", "hydra"}

_YAMLS = sorted(os.path.basename(p)[:-5]
                for p in glob.glob(os.path.join(_CONF, "experiment", "*.yaml")))


def _exp_dir(name):
    """conf/experiment/expN_x.yaml -> the experiments/expN_<slug>/ package dir."""
    prefix = re.match(r"(exp\d+)_", name).group(1)
    hits = [d for d in glob.glob(os.path.join(_EXP, f"{prefix}_*")) if os.path.isdir(d)]
    assert len(hits) == 1, f"{name}: expected one {prefix}_* dir, got {hits}"
    return hits[0]


def _reader_sources(name):
    """The .py files that may read this experiment's cfg: its own package + experiments/common."""
    return (glob.glob(os.path.join(_exp_dir(name), "*.py"))
            + glob.glob(os.path.join(_EXP, "common", "*.py")))


@pytest.mark.parametrize("name", _YAMLS)
def test_experiment_composes(name):
    with initialize_config_dir(version_base=None, config_dir=_CONF):
        cfg = compose(config_name="config", overrides=[f"experiment={name}"])
    assert cfg.system is not None                          # a real merged config, not an empty stub


@pytest.mark.parametrize("name", _YAMLS)
def test_no_ghost_keys(name):
    keys = set(OmegaConf.load(os.path.join(_CONF, "experiment", f"{name}.yaml")).keys())
    extras = keys - _SCHEMA_KEYS
    if not extras:
        return
    src = "\n".join(open(p, encoding="utf-8", errors="replace").read()
                    for p in _reader_sources(name))
    unread = [k for k in extras
              if not re.search(rf"""cfg\.{k}\b|cfg\[["']{k}["']\]|cfg\.get\(["']{k}["']""", src)]
    assert not unread, f"{name}: keys declared but never read ({unread}) — a config that lies"


# --- shared-schema keys ------------------------------------------------------------------------
# A runner that has a Hydra cfg and builds the production SplatKS must forward the shared keys that
# change that object. A static text check: no GPU, no imports.
_KS_FORWARD = ("df_lam", "aux_mult")


def _runner_sources():
    return sorted(p for p in glob.glob(os.path.join(_EXP, "exp*", "*.py")))


def test_splat_ks_callers_forward_the_shared_knobs():
    problems = []
    for path in _runner_sources():
        src = open(path, encoding="utf-8", errors="replace").read()
        if "builders.splat_ks(" not in src:
            continue
        for call in re.finditer(r"builders\.splat_ks\((?:[^()]|\([^()]*\))*\)", src):
            code = call.group(0)
            # A call is EXEMPT only if it pins the knob to an explicit module constant — the
            # checkpoint-template case (exp4), where the aux shape is part of a serialized contract
            # and must NOT follow the config.
            for key in _KS_FORWARD:
                if re.search(rf"\b{key}\s*=", code):
                    continue
                if "cfg." not in code and "cfg" not in code:
                    continue                    # no config in the call at all (constants-only helper)
                line = src[:call.start()].count("\n") + 1
                problems.append(f"{os.path.relpath(path, _EXP)}:{line} does not forward {key}")
    assert not problems, (
        "shared-schema knobs that change the SplatKS are not reaching it:\n  "
        + "\n  ".join(problems)
        + "\nA key the schema declares and the runner ignores is a config that lies.")
