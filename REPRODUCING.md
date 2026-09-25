# Reproducing the results

## Figures

Every figure in the paper rebuilds from the CSV files committed in `data/figures/`, without a GPU:

```bash
uv sync
make figures
```

`make figures` uses LaTeX text when `latex` is installed (with `mhchem` and `amsmath`), as in the
paper, and matplotlib's mathtext otherwise. Each plotter writes its PDF next to itself.

| Figure | Built by | Data | Runs that produced the data |
|---|---|---|---|
| 1, visual abstract | `experiments/shared/visual_abstract.py` | `data/figures/fig1_*.csv` | `exp5_basis_accuracy` |
| 2, components | `experiments.exp1_ablation_ladder.plot_ablation` | `fig2_ablation.csv` | `exp1_ablation_ladder/run_panels.sh` |
| 3, observables | `experiments.shared.plot_observables` | `fig3_observables.csv`, `fig3_gto_gradients.csv` | `exp2_observables/run_ladder.sh`, `run_obs_dipeptide.sh`, `run_forces_dipeptide.sh`, `run_obs_wf.sh` |
| 4, anion and LiF | `experiments.shared.plot_fidelity` | `fig4_anions.csv`, `fig4_dissociation.csv` | `exp4_anion/run_local.sh`, `exp3_lih_dissociation/run_equilibrium.sh` |
| 5, accuracy scaling | `experiments.shared.plot_isoparams` | `fig5_isoparams.csv` | `exp5_basis_accuracy/run_isoparams.sh`, `run_baselines.sh` |
| Appendix F, dissociation ladders | `experiments.exp3_lih_dissociation.plot_energy_curve` (`MOL=both`) | `fig4_dissociation.csv` | `exp3_lih_dissociation/gen_gto_ladder.sh`, `gen_splat_matched.sh`, `run_warm.sh` |

`data/README.md` describes the columns. To plot new runs, convert their logs with
`experiments.common.results.write_csv(logs, "data/figures/<file>.csv")`.

## Tables

`data/tables/` holds every results table as printed, and these runs produce them:

| Table | Data | Runs |
|---|---|---|
| 1, functional coverage | `table1_functional_coverage.csv` | `exp5_basis_accuracy/run_functionals.sh` |
| 2, ionization potentials | `table2_ionization_potentials.csv` | `exp2_observables/run_koopmans.sh` |
| 3, FMO proteins | `table3_fmo_proteins.csv` | `exp7_size_ladder/run_fmodb_suite.sh`, then `run_forces.sh` |
| Appendix D | `appD_*.csv` | `exp9_frameworks/run_water_node.sh`, `run_ladder.sh`, `run_gsdft_node.sh` |
| Appendix E, eigensolve, refresh, screening | `appE_eigensolve.csv`, `appE_refresh.csv`, `appE_screening.csv` | `exp1_ablation_ladder/run_panels.sh` (`PHASE=p3`) |
| Appendix E, optimizers | `appE_optimizer.csv` | `exp1_ablation_ladder/run_optsweep.sh` |
| Appendix E, initialization | `appE_initialization.csv` | `exp8_data_free_init/run_ladder.py`, summarized by `plot_curves.py` |
| Appendix F, anions | `appF_anions.csv` | `exp4_anion/run_local.sh` |

## Running the calculations

The training runs need GPUs. Each experiment directory has a README with its entry points; a single
run looks like

```bash
uv run python -m experiments.exp7_size_ladder.train experiment=exp7_train system=ala_15
```

and cluster jobs go through one launcher, configured for your site by `SLURM_ACCOUNT`,
`SLURM_PARTITION` and the other variables described at the top of the script:

```bash
experiments/slurm/submit.sh --gpus 4 --time 12:00:00 -- \
  bash experiments/exp7_size_ladder/run_ladder.sh ala_15
```

The protein structures of Table 3 come from the FMO database and are fetched once with
`uv run python -m experiments.common.fmodb --materialize`. The external codes of Appendix D each
need their own environment, built by `experiments/exp9_frameworks/setup_envs.sh`.

## Tests

```bash
make test
```

The suite covers the integrals, the Coulomb paths, screening, sharding, the result grammar and the
experiment configurations, and runs on CPU.
