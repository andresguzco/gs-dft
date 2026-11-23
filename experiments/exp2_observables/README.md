# Observables

Figure 3: energy, density and force errors of water, ethanol and alanine dipeptide against the
Gaussian ladder and against the splat ladder's own limit, and the ionization potentials of Table 2.

- `gto_ref.py`: Gaussian reference observables per rung; `pyscf_forces.py` supplies the alanine
  dipeptide force references and `cc_ref.py` the coupled-cluster densities.
- `train_ckpt.py`, `polish_forces.py`, `observables.py`: train a splat checkpoint, converge its
  forces, and measure its observables; `ref_l1.py` and `selfref.py` add the Gaussian density error
  and the bottom row.
- Appendix F repeats Figure 3 on SF6, cubane, the A–T pair, porphine, retinal and cholesterol, from
  the ladders of `experiments/rebuttal/ladder.py`: `rb_obs.py` polishes each rung's forces and
  measures it, `g4p_ref.py` gives the GPU4PySCF references where dftax cannot hold the basis or its
  forces (on the grid of `obs_grid.py`; cc-pV5Z, past GPU4PySCF's g functions, runs on the CPU with
  PySCF), `rb_post.py grads` exports the Gaussian gradients, and
  `selfref.py --estimator powerlaw` the bottom row. `rb_post.py ccfloor` gives the CCSD floor.

```bash
bash experiments/exp2_observables/run_ladder.sh water ethanol
bash experiments/exp2_observables/run_obs_dipeptide.sh
bash experiments/exp2_observables/run_forces_dipeptide.sh
bash experiments/exp2_observables/run_koopmans.sh                   # Table 2
uv run python -m experiments.shared.plot_observables                # Figure 3 from data/figures/
uv run python -m experiments.shared.plot_observables --rebuttal     # Appendix F from data/rebuttal/
```
