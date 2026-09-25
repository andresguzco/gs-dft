# Differentiable framework comparison

Appendix D: D4FT, MESS and DQC against GS-DFT, with PySCF as the reference energy, on the same
geometries and basis sets. Each external code runs in its own environment through `run_code.py`,
which imports nothing from `gs_dft`.

```bash
bash experiments/exp9_frameworks/setup_envs.sh d4ft                 # once per code: pyscf, mess, d4ft, dqc
bash experiments/exp9_frameworks/run_water_node.sh                  # the water checks
CAP=3 bash experiments/exp9_frameworks/run_ladder.sh pyscf          # the cc-pVTZ ladder, one code
GPUS=0,1,2,3 bash experiments/exp9_frameworks/run_gsdft_node.sh     # the GS-DFT rows
```

Environments go to `$APPD_ENVS` (default `$SCRATCH/appD_envs`) and logs to `results/`.
