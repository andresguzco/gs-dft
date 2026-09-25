# Dissociation curves

LiH and LiF along the ionic diabat, splats against plain and augmented Gaussian bases at matched
function count: Figure 4(b) and the dissociation ladders of Appendix F.

- `gto_curve.py`: one Gaussian point (R, basis).
- `splat_curve.py`: one cold-start splat point (R, M).
- `splat_continuation.py`: a warm-started chain along R at fixed M.

```bash
SYSTEM=lif bash experiments/exp3_lih_dissociation/run.sh            # Gaussian points and cold starts
SYSTEM=lif bash experiments/exp3_lih_dissociation/run_warm.sh       # warm-started chains
bash experiments/exp3_lih_dissociation/run_equilibrium.sh           # LiF near equilibrium
bash experiments/exp3_lih_dissociation/gen_gto_ladder.sh            # Appendix F, Gaussian ladder
bash experiments/exp3_lih_dissociation/gen_splat_matched.sh         # Appendix F, matched splat chains
MOL=both uv run python -m experiments.exp3_lih_dissociation.plot_energy_curve   # Appendix F figure
```

Figure 4 itself is drawn by `experiments/shared/plot_fidelity.py`.
