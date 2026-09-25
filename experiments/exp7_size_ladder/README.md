# Protein size ladder

Table 3: the five FMODB proteins, from 304 to 2,742 atoms, at M = nao(cc-pVTZ) on one four-GPU
node, with peak memory and forces. The same trainer runs the alanine chains.

The structures are fetched from FMODB (CC BY-SA 4.0) rather than committed:

```bash
uv run python -m experiments.common.fmodb --materialize             # once
bash experiments/exp7_size_ladder/run_fmodb_suite.sh --submit       # one whole-node job per protein
bash experiments/exp7_size_ladder/run_forces.sh                     # the "Simple forces" column
uv run python -m experiments.exp7_size_ladder.train experiment=exp7_train system=ala_15
```

`train.py` writes a per-step JSONL (energy terms, gradient norm) next to its log, and a final
`RESULT kind=train` line with the energy, step count and peak GPU memory.
