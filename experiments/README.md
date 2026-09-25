# Experiments

One directory per result in the paper. Every runner prints machine-readable `RESULT` lines (see
`common/results.py`), and the rows the figures are built from are committed under
[`data/`](../data/) as CSV, so every figure rebuilds from a clean checkout with `make figures`.

| Directory | Paper |
|---|---|
| `exp1_ablation_ladder` | Figure 2; eigensolve, optimizer, refresh and screening ablations (Appendix E) |
| `exp2_observables` | Figure 3; ionization potentials (Table 2) |
| `exp3_lih_dissociation` | Figure 4(b); dissociation ladders (Appendix F) |
| `exp4_anion` | Figure 4(a); anion table (Appendix F) |
| `exp5_basis_accuracy` | Figure 5; functional coverage (Table 1) |
| `exp6_cost_benchmark` | the GS-DFT water check of Appendix D |
| `exp7_size_ladder` | FMO proteins (Table 3) |
| `exp8_data_free_init` | initialization ablation (Appendix E) |
| `exp9_frameworks` | comparison with differentiable DFT codes (Appendix D) |

`common/` holds the shared helpers (systems, builders, the result line), `shared/` the plotters of
figures that combine several experiments, and `slurm/submit.sh` the job launcher. Runners are
configured with Hydra (`conf/`) and invoked as modules:

```bash
uv run python -m experiments.<directory>.<runner> experiment=<config> key=value
```

Logs go to each directory's `results/`, which is not tracked. For a cluster, set `SLURM_ACCOUNT`
and `SLURM_PARTITION` and wrap a launcher with `experiments/slurm/submit.sh`.
