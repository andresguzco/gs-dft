#!/bin/bash
# Warm-started splat chains along the dissociation curve, one chain per M and per GPU; each
# chain sweeps R upward from the previous converged cloud (splat_continuation.py).
#
#   SYSTEM=lih bash experiments/exp3_lih_dissociation/run_warm.sh
#
# Env: NGPU(4)
set -uo pipefail

NGPU="${NGPU:-4}"
RES=experiments/exp3_lih_dissociation/results
EXP=experiments/exp3_lih_dissociation
mkdir -p "$RES"

SYS="${SYSTEM:-lih}"
case "$SYS" in
  lih) R_GRID=(1.0 1.3 1.6 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0); CONFIGS=(85 69 44 32 19) ;;
  lif) R_GRID=(1.2 1.5 1.8 2.2 2.8 3.5 4.5 6.0);             CONFIGS=(110 92 60 46 28) ;;
  *) echo "unknown SYSTEM=$SYS (lih|lif)" >&2; exit 1 ;;
esac                                # one continuation chain per M, LARGEST FIRST (load balance)

# submit.sh already runs us inside `uv run`, so python IS the venv's; standalone, go through uv.
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

idx=0; declare -A pid2gpu
launch_on() {  # $1 = gpu id. Pulls chains until one launches, or falls through when drained.
  local g=$1
  while [ "$idx" -lt "${#CONFIGS[@]}" ]; do
    local M="${CONFIGS[$idx]}"; idx=$((idx+1))
    local log="$RES/warm_${SYS}_M${M}.log"
    # A chain is complete only when every R point has a RESULT line.
    local done_pts; done_pts=$(grep -ac "^RESULT kind=splat " "$log" 2>/dev/null) || true
    [ "${done_pts:-0}" -ge "${#R_GRID[@]}" ] && continue
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
      "${PY[@]}" -m experiments.exp3_lih_dissociation.splat_continuation experiment=exp3_continuation \
        system="$SYS" m="$M" >> "$log" 2>&1 &
    pid2gpu[$!]=$g; echo "GPU $g <- chain M=$M (pid $!)"; return 0
  done
}
for g in $(seq 0 $((NGPU-1))); do launch_on "$g"; done
# Refill pool: `wait -n` wakes on ANY child, so a freed GPU pulls the next chain immediately rather
# than idling until the wave ends. `|| true` keeps one crashed chain from killing the queue.
while [ "${#pid2gpu[@]}" -gt 0 ]; do
  wait -n || true
  for pid in "${!pid2gpu[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then g=${pid2gpu[$pid]}; unset 'pid2gpu[$pid]'; launch_on "$g"; fi
  done
done
echo "=== exp3 warm chains drained ==="
grep -aH "^RESULT" "$RES"/warm_M*.log 2>/dev/null | sed 's#.*/##'
