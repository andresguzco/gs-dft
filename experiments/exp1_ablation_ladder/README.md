# Ablation ladder

Figure 2 and four ablations of Appendix E.

- `ladder.py` with `charts.py`: the representation rungs of Figure 2(a), from Frost's FSGO to the
  splat chart; `gto_minimize.py` gives the Gaussian trajectory on the same step axis.
- `machinery.py`: adds one engineering piece at a time, for Figure 2(b) (peak memory against free
  parameters, with `gto_memory.py` for the Gaussian side), Figure 2(c) (sharding), and the
  regularized-eigensolve, refresh-period and screening-threshold tables of Appendix E.
- `optimizers.py`: the optimizer comparison of Appendix E.

```bash
PHASE=p1 bash experiments/exp1_ablation_ladder/run_panels.sh        # Figure 2(a)
PHASE=p2 bash experiments/exp1_ablation_ladder/run_panels.sh        # Figure 2(b)
PHASE=p3par bash experiments/exp1_ablation_ladder/run_panels.sh     # Figure 2(c)
ARM_INDEX=0 bash experiments/exp1_ablation_ladder/run_optsweep.sh   # one optimizer
uv run python -m experiments.exp1_ablation_ladder.plot_ablation     # Figure 2 from data/figures/
```
