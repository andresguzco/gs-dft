"""The quickstart snippet in ``README.md`` runs.

They run verbatim except for the molecule, basis, M and step count, which are shrunk so a CPU
finishes in seconds.
"""
import math
import pathlib
import re

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

# (budget pattern -> tiny replacement). Physics-free: nothing here changes an API surface.
SHRINK = [
    (r'build_reference\("co2", "cc-pvdz", "pbe", 3\)', 'build_reference("h2o", "sto-3g", "pbe", 1)'),
    (r"native_grid\(mol, 3, chunk=4096\)", "native_grid(mol, 1, chunk=4096)"),
    (r"init_model\(mol, M=126, key=jr\.PRNGKey\(0\)\)", "init_model(mol, M=14, key=jr.PRNGKey(0))"),
    (r"train\(ks, model, steps=6000\)", "train(ks, model, steps=2, monitor=None)"),
]


def _first_python_block(path):
    text = (REPO / path).read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, f"{path}: no ```python block — did the quickstart move?"
    return blocks[0]


def _shrink(src, path):
    for pat, repl in SHRINK:
        if re.search(pat, src):
            src = re.sub(pat, repl, src)
    assert "steps=2" in src, (
        f"{path}: the training call no longer matches any budget pattern in SHRINK, so this test "
        f"would run the FULL quickstart. Update SHRINK alongside the snippet.")
    return src


@pytest.mark.parametrize("path", ["README.md"])
def test_quickstart_snippet_executes(path):
    src = _shrink(_first_python_block(path), path)
    ns: dict = {}
    try:
        exec(compile(src, f"<{path} quickstart>", "exec"), ns)
    except Exception as exc:                                 # noqa: BLE001 — report, don't swallow
        pytest.fail(f"{path} quickstart raised {type(exc).__name__}: {exc}\n"
                    f"The snippet a first-time reader copies does not run.\n--- as executed ---\n"
                    f"{src}")
    assert "E" in ns and "F" in ns, f"{path}: snippet defined {sorted(ns)} — no E / F"
    assert math.isfinite(float(ns["E"])), f"{path}: E = {ns['E']!r}"
    assert ns["F"].shape == (len(ns["mol"].symbols), 3), f"{path}: F has shape {ns['F'].shape}"
    assert np.isfinite(np.asarray(ns["F"])).all(), f"{path}: forces contain non-finite entries"
