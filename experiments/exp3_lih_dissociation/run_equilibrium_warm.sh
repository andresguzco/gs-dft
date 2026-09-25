#!/bin/bash
# Figure 4(b): LiF near equilibrium, warm-started splat chains swept up and down in R.
#
#   bash experiments/exp3_lih_dissociation/run_equilibrium_warm.sh
#
# Env: M (60), SEEDS ("0 1").
set -uo pipefail
cd "$(dirname "$0")/../.."
M="${M:-60}"
SEEDS="${SEEDS:-0 1}"
UP="[1.50,1.55,1.60,1.65,1.70,1.75]"
DOWN="[1.75,1.70,1.65,1.60,1.55,1.50]"
RES=experiments/exp3_lih_dissociation/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

for S in $SEEDS; do
  for DIR in up down; do
    [ "$DIR" = up ] && G="$UP" || G="$DOWN"
    log="$RES/eqw_lif_M${M}_s${S}_${DIR}.log"
    grep -qa "^RESULT " "$log" 2>/dev/null && { echo "--- $DIR s$S done"; continue; }
    echo "=== warm chain lif M=$M seed=$S $DIR  [$(date +%H:%M:%S)]"
    PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 "${PY[@]}" \
      -m experiments.exp3_lih_dissociation.splat_continuation experiment=exp3_continuation \
      system=lif m="$M" seed="$S" +r_grid="$G" >> "$log" 2>&1
    grep -ac "^RESULT" "$log" | xargs -I{} echo "    {} points"
  done
done
echo "=== warm equilibrium chains drained ==="
