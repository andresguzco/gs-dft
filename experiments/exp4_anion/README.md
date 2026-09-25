# Bare anions

Fluoride and hydroxide, whose density extends far from the nuclei: Figure 4(a) and the anion
table of Appendix F. The splat cloud gets no diffuse functions.

```bash
bash experiments/exp4_anion/run_local.sh                            # Gaussian references and splat sweep
uv run python -m experiments.exp4_anion.splat_anion experiment=exp4_splat_anion system=f_anion m=28
```

Figure 4 is drawn by `experiments/shared/plot_fidelity.py` from `data/figures/fig4_anions.csv`.
