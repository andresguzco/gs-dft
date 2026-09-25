"""The shell launchers invoke modules, scripts, flags and config keys that exist.

A static text check: no GPU and no heavy imports. A launcher whose invocation cannot be resolved
is reported rather than skipped.
"""
import pathlib
import re

from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[2]

# `-m pkg.mod` for either the library or an experiment module.
_INVOKE_M = re.compile(r"-m\s+((?:gs_dft|experiments)(?:\.[A-Za-z_0-9]+)+)")
# a .py path token, possibly with $VARS in it: "$EXP/run_cost.py", experiments/exp3/gto_curve.py
_INVOKE_PY = re.compile(r"""["']?((?:\$\{?[A-Za-z_][A-Za-z_0-9]*\}?|[\w.\-/])+\.py)["']?""")
# literal VAR=value assignments only (no command substitution) — enough to expand $EXP / $D
_ASSIGN = re.compile(r"""^\s*([A-Za-z_][A-Za-z_0-9]*)=["']?([\w.\-/]+)["']?\s*$""", re.M)
_FLAG = re.compile(r"(--[A-Za-z_][A-Za-z_0-9-]*)")
_ADD_ARG = re.compile(r"add_argument\(\s*['\"](--[A-Za-z_][A-Za-z_0-9-]*)['\"]")
# a Hydra override token: key=value / key.sub=value, but NOT $VARS and NOT --flags
_OVERRIDE = re.compile(r"(?<![\w$/.-])([a-z_][a-z_0-9.]*)=(?!=)")


def _vars_of(text: str) -> dict[str, str]:
    return dict(_ASSIGN.findall(text))


def _logical_lines(text: str) -> list[str]:
    """Join `\\`-continuations, THEN strip comments.

    Launcher invocations wrap across continuation lines, with the script or module on the first
    and its arguments on the rest, so reading line by line would never see the arguments.
    """
    joined = re.sub(r"\\\n\s*", " ", text)
    return [ln.split("#", 1)[0] for ln in joined.splitlines()]


def _expand(token: str, env: dict[str, str]) -> str:
    def sub(m):
        return env.get(m.group(1) or m.group(2), "\0")      # \0 => unresolvable, never a real path
    return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)", sub, token)


def _module_source(module: str) -> pathlib.Path | None:
    p = REPO / (module.replace(".", "/") + ".py")
    return p if p.exists() else None


def _hydra_keys(experiment: str | None) -> set[str] | None:
    """Keys a Hydra script may be handed: the shared schema + the selected experiment's own."""
    keys = {k for k in OmegaConf.load(REPO / "conf/config.yaml") if k not in ("defaults", "hydra")}
    if experiment:
        p = REPO / "conf/experiment" / f"{experiment}.yaml"
        if not p.exists():
            return None                                     # unknown experiment= selector
        keys |= set(OmegaConf.load(p))
    return keys


def _check_target(sh_name, code_after, src: pathlib.Path, problems: list[str]) -> None:
    """`code_after` is the invocation line AFTER the script/module token — i.e. its arguments."""
    text = src.read_text()
    if "@hydra.main" in text:                               # Hydra: args are key=value overrides
        exp = re.search(r"(?<![\w+])experiment=([\w]+)", code_after)
        if "+experiment=" in code_after:
            problems.append(f"{sh_name}: `+experiment=` errors (the group is in the defaults "
                            f"list); use `experiment=`")
        known = _hydra_keys(exp.group(1) if exp else None)
        if known is None:
            problems.append(f"{sh_name}: experiment={exp.group(1)} has no conf/experiment yaml")
            return
        for key in _OVERRIDE.findall(code_after):
            if key != "experiment" and key not in known:
                problems.append(f"{sh_name}: {src.name} config has no `{key}` "
                                f"(experiment={exp.group(1) if exp else '-'})")
        for flag in _FLAG.findall(code_after):              # argparse flags at a Hydra script
            problems.append(f"{sh_name}: {src.name} is Hydra — `{flag}` is not a thing")
    elif "add_argument" in text:                            # argparse: args are --flags
        known = set(_ADD_ARG.findall(text))
        for flag in _FLAG.findall(code_after):
            if flag not in known:
                problems.append(f"{sh_name}: {src.name} has no {flag}")


def test_launcher_invocations_are_live():
    launchers = sorted((REPO / "experiments").rglob("*.sh"))
    assert launchers, "no launchers found — repo layout changed?"
    problems: list[str] = []
    checked: set[str] = set()

    for sh in launchers:
        text = sh.read_text()
        env = _vars_of(text)
        for code in _logical_lines(text):                   # continuations JOINED — see _logical_lines
            if not re.search(r"\bpython\b|\$\{?PY\[@\]\}?|\$PYRUN", code):
                continue
            if "PY=(" in code:                              # the `PY=(python -u)` selector itself
                continue

            m = _INVOKE_M.search(code)
            if m:
                src = _module_source(m.group(1))
                if src is None:
                    problems.append(f"{sh.name}: references missing module {m.group(1)}")
                    continue
                checked.add(sh.name)
                _check_target(sh.name, code[m.end():], src, problems)
                continue

            p = _INVOKE_PY.search(code)
            if p:
                cand = REPO / _expand(p.group(1), env)
                if not cand.exists():
                    # A launcher that builds its script path from `"$@"` picks it at runtime —
                    # genuinely dynamic, so a non-resolving path there is not a defect.
                    if "$@" not in code:
                        problems.append(f"{sh.name}: invokes {p.group(1)}, which does not resolve "
                                        f"to a file (expanded: {_expand(p.group(1), env)})")
                    continue
                checked.add(sh.name)
                _check_target(sh.name, code[p.end():], cand, problems)

    assert not problems, "\n".join(problems)
    # Coverage guard: every launcher that runs a Python module or script must be reachable by the
    # resolver above, so a new invocation form fails here instead of going unchecked.
    runs_python = {sh.name for sh in launchers
                   if re.search(r"""^[^#\n]*(\bpython[0-9.]*["']?\s+(-u\s+)?(-m\s|\S+\.py)|\$\{?PY\[@\]\}?|\$PYRUN)""",
                                sh.read_text(), re.M)
                   and "$@" not in sh.read_text()}
    assert not (runs_python - checked), (
        f"launcher(s) invoke python but no invocation was resolved/checked: "
        f"{sorted(runs_python - checked)}")
