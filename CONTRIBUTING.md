# Contributing

## Layout

`gs_dft/` is the library and `experiments/` holds one directory per result in the paper. Library
docstrings are a one-line summary plus `Args:`/`Returns:` where the signature needs it, and
comments state constraints the code cannot show (a dtype fixed at import time, an allocation that
must not materialize, an invariant a refactor could break) in a sentence or two.

A new public symbol needs a docstring, and a new experiment needs a directory with a README and a
row in `experiments/README.md`.

## Checks

```bash
make test                                           # the suite
uv run pytest tests/unit/test_docs_quickstart.py    # the README quickstart runs verbatim
```

## Artifacts

Checkpoints, collapse dumps and W&B run directories follow `$DFTAX_CKPT_DIR` and `$DFTAX_WANDB_DIR`
when set (`experiments/common/paths.py`); logs and `RESULT` lines go to each experiment's
`results/`, which is not tracked. The data behind every figure and table is committed as CSV under
`data/` (see `data/README.md`); `experiments.common.results.write_csv` converts run logs to that
format.
