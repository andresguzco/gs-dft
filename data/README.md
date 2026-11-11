# Data

The data behind every figure and table of the paper, as CSV.

## `figures/`

The inputs of the submitted paper's six figures; `rebuttal/` holds the rest. `make figures` rebuilds
all of them from these two folders alone.

| File | Figure | Contents |
|---|---|---|
| `fig1_atoms.csv`, `fig1_splats.csv`, `fig1_gto_primitives.csv` | 1 | water's nuclei, the trained splat cloud (centre, precision matrix `A`, normalization, mean orbital coefficients) and the cc-pVDZ primitives |
| `fig1_scaling.csv`, `fig1_memory.csv` | 1 | energy error (mHa) against parameters for alanine dipeptide, and peak memory (MB) against parameters |
| `fig1_training_energy.csv` | 1 | the training energy (Ha) of the splat cloud |
| `fig2_ablation.csv` | 2, Appendix E | the ablation runs: representation rungs, memory arms, sharded runs, eigensolve, optimizer, refresh and screening sweeps |
| `fig3_observables.csv` | 3 | Gaussian and splat observables of water, ethanol and alanine dipeptide |
| `fig3_gto_gradients.csv` | 3 | nuclear gradients (Ha/Bohr) of the Gaussian references |
| `fig4_anions.csv` | 4(a), Appendix F | fluoride and hydroxide, splats and the plain and augmented Gaussian ladders |
| `fig4_dissociation.csv` | 4(b), Appendix F | LiH and LiF along the dissociation curve |
| `fig5_isoparams.csv` | 5, 1 | the splat M-ladders and Gaussian baselines of water, ethanol and alanine dipeptide |

The run files (`fig2`–`fig5`) hold one row per recorded result, with a column per field:

- `_source` is the run log the row came from, and rows are in the order the runs were made. When a
  configuration was run more than once, the later row wins.
- `_record` is the kind of line: `RESULT` (a run's final result), `RESULT_PARTIAL` (a snapshot during
  training), `TRACE` (energy against time or step, for the trajectory panels) or `STEP` (the
  per-step monitor: energy, gradient norm `grad_norm`, orbital Gram eigenvalues `lam_min` and
  `lam_max`, and the auxiliary metric condition number `cond_V`).
- Energies are in Hartree (`E`, `E_ref`, `E_grid5`), differences in mHa (`gap_mha`, `grid_bias_mha`),
  memory in MB (`peak_*_mb`) and times in seconds (`wall_s`, `t`).

`experiments.common.results.iter_results` and `read_records` read these files, and `write_csv`
converts the logs of new runs to the same format.

## `rebuttal/`

The accuracy scaling on the new systems (Figures 5 and 6, Appendix F) and the parallel efficiency
(Appendix F). `accuracy.csv` is a run file in the format above.

| File | Contents |
|---|---|
| `accuracy.csv` | the splat ladders (`kind=splat`) and Gaussian ladders (`kind=gto`) of the new systems; GPU4PySCF references have `mode=g4p_direct` and are shifted onto the `dftax` scale by `offset_mha` |
| `offsets.csv` | the `dftax` − GPU4PySCF energy offset per system, at the bases both codes ran |
| `exp2_scaling.csv` | step time (s; median, 10th and 90th percentiles) against the number of GPUs `G`, with repeats `rep` |
| `exp2_memory.csv` | peak memory per GPU (MB, one value per GPU separated by `/`) against `G` |
| `exp4.csv` | GPU4PySCF density fitting against direct SCF on cyclosporin A at cc-pVTZ |

## `tables/`

Every results table, with the values as printed in the paper.

| File | Table |
|---|---|
| `table1_functional_coverage.csv` | 1, functional coverage on ethanol (differences from the CBS limit in mHa) |
| `table2_ionization_potentials.csv` | 2, Hartree–Fock ionization potentials (eV) |
| `table3_fmo_proteins.csv` | 3, the FMO proteins; Gaussian memory is extrapolated from measured runs |
| `appD_capabilities.csv`, `appD_water.csv`, `appD_scaling.csv` | Appendix D, differentiable DFT codes |
| `appE_eigensolve.csv`, `appE_optimizer.csv`, `appE_initialization.csv`, `appE_refresh.csv`, `appE_screening.csv` | Appendix E, ablations |
| `appF_anions.csv` | Appendix F, bare anions |

Empty cells are entries the paper leaves blank or marks with a dash.
