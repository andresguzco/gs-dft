#!/bin/bash
# Figure 5: four M per system, doubling so the power-law fit gets even log spacing, three seeds
# each, 12,000 steps, as a work queue over the GPUs. Logs go to results_isoparams/.
#
#   JOB=water bash experiments/exp5_basis_accuracy/run_isoparams.sh
#   JOB=ethanol experiments/slurm/submit.sh --gpus 4 --time 24:00:00 -- bash experiments/exp5_basis_accuracy/run_isoparams.sh
#
# Env: JOB (water|ethanol|dipeptide), NGPU (4), SEEDS ("0 1 2").
set -uo pipefail
cd "$(dirname "$0")/../.."

JOB="${JOB:-water}"
NGPU="${NGPU:-4}"
SEEDS="${SEEDS:-0 1 2}"
RES=experiments/exp5_basis_accuracy/results_isoparams
mkdir -p "$RES"

case "$JOB" in
  water)     SYS=water;             MS="24 48 96 192" ;;
  ethanol)   SYS=ethanol;           MS="72 144 288 576" ;;
  dipeptide) SYS=alanine_dipeptide; MS="125 250 500 1000" ;;
  *) echo "unknown JOB=$JOB (water|ethanol|dipeptide)" >&2; exit 1 ;;
esac

# Smallest first: a walltime clip must still leave a usable ladder rather than only its cheap end.
CONFIGS=()
for m in $MS; do for s in $SEEDS; do CONFIGS+=("$SYS $m $s"); done; done

if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

idx=0; nfail=0; declare -A pid2gpu
launch_on() {
  local g=$1
  while [ "$idx" -lt "${#CONFIGS[@]}" ]; do
    local cfg="${CONFIGS[$idx]}"; idx=$((idx+1))
    local a b c; read -r a b c <<< "$cfg"
    local log="$RES/iso_${a}_M${b}_s${c}.log"
    [ -f "$log" ] && grep -qa "^RESULT " "$log" && { echo "skip $cfg (done)"; continue; }
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
      "${PY[@]}" -m experiments.exp5_basis_accuracy.splat_run experiment=exp5_refresh \
        system="$a" m="$b" seed="$c" ckpt_dir="$RES" >> "$log" 2>&1 &
    pid2gpu[$!]=$g
    echo "[pool] launch $cfg -> gpu $g (pid $!)"
    return 0
  done
  return 1
}

echo "[pool] JOB=$JOB  $NGPU GPUs  ${#CONFIGS[@]} configs: $SYS M={$MS} seeds={$SEEDS}"
for ((g = 0; g < NGPU; g++)); do launch_on "$g" || break; done
while [ "${#pid2gpu[@]}" -gt 0 ]; do
  wait -n 2>/dev/null
  for p in "${!pid2gpu[@]}"; do
    kill -0 "$p" 2>/dev/null && continue
    g=${pid2gpu[$p]}; unset 'pid2gpu[$p]'
    if wait "$p"; then echo "[pool] done (gpu $g freed)"
    else nfail=$((nfail+1)); echo "[pool] FAILED rc=$? (gpu $g freed)" >&2; fi
    launch_on "$g" || true
  done
done
# Exit nonzero if anything failed: a job whose every config died must not report COMPLETED.
if [ "$nfail" -gt 0 ]; then
  echo "[pool] $JOB: $nfail config(s) FAILED" >&2; exit 1
fi
echo "[pool] $JOB complete"
