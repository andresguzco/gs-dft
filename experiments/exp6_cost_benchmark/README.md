# Cost benchmark

End-to-end time and memory for one configuration, splat or Gaussian, on the same engine and grid.
Used for the GS-DFT water check of Appendix D (`exp9_frameworks/run_gsdft_node.sh`).

```bash
uv run python -m experiments.exp6_cost_benchmark.bench experiment=exp6_bench system=water engine=splat m=24
uv run python -m experiments.exp6_cost_benchmark.bench experiment=exp6_bench system=water engine=gto basis=cc-pvdz
```
