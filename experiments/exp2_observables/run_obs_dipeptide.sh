#!/usr/bin/env bash
# Figure 3: the alanine dipeptide observables, measured from the polished checkpoints.
#
#   bash experiments/exp2_observables/run_obs_dipeptide.sh
#
# Env: SYS(alanine_dipeptide) MS("200 468 910")
set -uo pipefail
cd "$(dirname "$0")/../.."
SYS="${SYS:-alanine_dipeptide}"; MS="${MS:-200 468 910}"
RES=experiments/exp2_observables/results
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

g=0
for M in $MS; do
  log="$RES/obs_${SYS}_M${M}.log"
  # key on ckpt=polished: the stale rows are ckpt=raw with dF_max=nan
  if grep -qa "^RESULT kind=obs.*ckpt=polished" "$log" 2>/dev/null; then echo "--- obs M=$M done"; continue; fi
  echo "GPU $g <- obs $SYS M=$M  [$(date +%H:%M:%S)]"
  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
    "${PY[@]}" -m experiments.exp2_observables.observables experiment=exp2_observables \
    system="$SYS" m="$M" >> "$log" 2>&1 &
  g=$((g+1))
done
wait
echo "=== dipeptide observables drained ==="
n=0; want=$(wc -w <<<"$MS")
for M in $MS; do
  grep -qa "^RESULT kind=obs.*ckpt=polished" "$RES/obs_${SYS}_M${M}.log" 2>/dev/null && n=$((n+1))
done
echo "=== $n/$want rungs carry a polished obs row ==="
[ "$n" -eq "$want" ]
