# Basis accuracy and scaling

Figure 5, the splat ladder against the Gaussian cc-pVXZ ladder with both limits and the scaling
exponents, and the functional coverage of Table 1.

- `baseline_gto.py` and `cbs_fit.py`: the Gaussian baselines and their complete-basis limit.
- `splat_run.py`: one splat run at (system, M).
- `msweep.py`: M-sweeps at a chosen Coulomb path and functional, and the ionization potentials.

```bash
bash experiments/exp5_basis_accuracy/run_baselines.sh               # Gaussian baselines
JOB=water bash experiments/exp5_basis_accuracy/run_isoparams.sh     # Figure 5 splat ladder
NGPU=4 bash experiments/exp5_basis_accuracy/run_functionals.sh      # Table 1
uv run python -m experiments.shared.plot_isoparams                  # Figure 5 from data/figures/
```
