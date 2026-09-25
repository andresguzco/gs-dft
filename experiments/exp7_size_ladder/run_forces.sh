#!/usr/bin/env bash
# Table 3, "Simple forces": Hellmann-Feynman forces from the converged protein checkpoints.
#
#   bash experiments/exp7_size_ladder/run_forces.sh              # every system listed below
#   bash experiments/exp7_size_ladder/run_forces.sh PJM49        # one of them
#
# Each system's HYDRA line must match the configuration its checkpoint was written under, because
# the optimizer state is part of the saved tree.
set -uo pipefail
cd "$(dirname "$0")/../.."
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || { echo "run_forces: no uv found" >&2; exit 127; }
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=("$UV" run python -u); fi

# NGPU limits the GPUs the job sees, for a single-device force measurement on a whole-node allocation.
if [ -n "${NGPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NGPU - 1)))"
  echo "# NGPU=$NGPU -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# The tags and step budgets of run_fmodb_suite.sh.
declare -A HY
HY[PJM49]="tag=PJM49 steps=4500"
HY[9LQN2]="tag=9LQN2 steps=1500"
HY[LG5K9]="tag=LG5K9 steps=1000"
HY[38KNL]="tag=38KNL steps=1000 theta_grad=false"
HY[25J8R]="tag=25J8R steps=600 theta_grad=false"

SYSTEMS=("$@"); [ ${#SYSTEMS[@]} -eq 0 ] && SYSTEMS=(PJM49 9LQN2 LG5K9 38KNL 25J8R)
RES=experiments/exp7_size_ladder/results; mkdir -p "$RES"

rc=0
for S in "${SYSTEMS[@]}"; do
  echo "=== forces $S ==="
  if PYTHONPATH="$PWD" "${PY[@]}" -m experiments.exp7_size_ladder.forces \
       experiment=exp7_forces system="$S" basis=cc-pvtz m_mult=1.0 ${HY[$S]} \
       2>&1 | tee -a "$RES/forces_$S.log"; then
    echo "--- ok: $S"
  else
    echo "--- FAILED: $S" >&2; rc=1
  fi
done
exit $rc
