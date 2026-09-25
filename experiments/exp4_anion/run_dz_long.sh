#!/bin/bash
# The anion runs at the cc-pVDZ-matched M with a longer budget.
#
#   bash experiments/exp4_anion/run_dz_long.sh
#
# Env: STEPS (15000).
set -uo pipefail
cd "$(dirname "$0")/../.."
STEPS="${STEPS:-15000}"
RES=experiments/exp4_anion/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

for spec in "f_anion 14" "oh_anion 19"; do
  set -- $spec; a=$1; m=$2
  for s in 0 1 2; do
    log="$RES/dz${STEPS}_${a}_M${m}_s${s}.log"
    grep -qa "^RESULT " "$log" 2>/dev/null && continue
    echo "--- $a M=$m seed=$s steps=$STEPS  [$(date +%H:%M:%S)]"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
      "${PY[@]}" -m experiments.exp4_anion.splat_anion experiment=exp4_splat_anion \
        system="$a" m="$m" seed="$s" steps="$STEPS" >> "$log" 2>&1
    grep -a "^RESULT" "$log" | tail -1
  done
done
echo "=== dz long drained ==="
grep -ah "^RESULT" "$RES"/dz${STEPS}_*.log 2>/dev/null
