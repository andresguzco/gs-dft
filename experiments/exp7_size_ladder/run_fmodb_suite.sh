#!/usr/bin/env bash
# Table 3: the five FMODB proteins, one whole four-GPU node each, at M = nao(cc-pVTZ), with the
# chemistry-informed initialization and minimal-basis coefficients (Appendix C). Prints the five
# submit lines; pass --submit to submit them.
#
#   uv run python -m experiments.common.fmodb --materialize                # fetch the structures once
#   bash experiments/exp7_size_ladder/run_fmodb_suite.sh                   # dry run: show the jobs
#   bash experiments/exp7_size_ladder/run_fmodb_suite.sh --submit
#
# Steps per system as in Appendix C. 3IFU (25J8R) peaks near 480 GB, so it needs four GPUs with at
# least 120 GB each; the others fit on four 80 GB GPUs. Set SLURM_ACCOUNT and SLURM_PARTITION for
# your site (see experiments/slurm/submit.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
SUBMIT=${1:-}

declare -A STEPS TIME
COMMON="basis=cc-pvtz m_mult=1.0 init_c=minao"
STEPS[PJM49]=4500; TIME[PJM49]=24:00:00      # Trp-cage
STEPS[9LQN2]=1500; TIME[9LQN2]=24:00:00      # defensin
STEPS[LG5K9]=1000; TIME[LG5K9]=24:00:00      # insulin
STEPS[38KNL]=1000; TIME[38KNL]=24:00:00      # beta-spectrin CH domain
STEPS[25J8R]=600;  TIME[25J8R]=24:00:00      # 3IFU chain A

for S in PJM49 9LQN2 LG5K9 38KNL 25J8R; do
  cmd=(env GPUS_PER=4 HYDRA="$COMMON steps=${STEPS[$S]}"
       experiments/slurm/submit.sh --name "fmo_$S" --gpus 4 --cpus 24 --mem 0
       --time "${TIME[$S]}" --out experiments/exp7_size_ladder/results
       -- bash experiments/exp7_size_ladder/run_ladder.sh "$S")
  if [ "$SUBMIT" = "--submit" ]; then "${cmd[@]}"; else printf '%q ' "${cmd[@]}"; echo; fi
done
