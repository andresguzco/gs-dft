#!/usr/bin/env bash
# Score the Figure 3 ladders against the coupled-cluster (or MP2) reference: observables.py writes
# the `*_wf` fields on each `obs` row and one `kind=gto_wf` row per Gaussian rung. One GPU.
#
#   bash experiments/exp2_observables/run_obs_wf.sh <system>:<M> [<system>:<M> ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PAIRS=("$@")
[ ${#PAIRS[@]} -eq 0 ] && PAIRS=(water:24 water:58 water:115 ethanol:72 ethanol:174 ethanol:345
                                alanine_dipeptide:200 alanine_dipeptide:468 alanine_dipeptide:910)
for p in "${PAIRS[@]}"; do
  S=${p%%:*}; M=${p##*:}
  echo "=== obs $S M=$M $(date -Is) ==="
  python -m experiments.exp2_observables.observables experiment=exp2_observables system="$S" m="$M" wandb_mode=disabled
done
