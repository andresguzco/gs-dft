#!/bin/bash
# Figure 5: the Gaussian cc-pVXZ baselines as a work queue over the GPUs, largest first.
#
#   bash experiments/exp5_basis_accuracy/run_baselines.sh
#   GROUP=refmode bash experiments/exp5_basis_accuracy/run_baselines.sh   # exact vs DF at the top rungs
#
# Env: GROUP (ladder|refmode), NGPU (4), SYSTEMS.
set -uo pipefail

GROUP="${GROUP:-ladder}"
NGPU="${NGPU:-4}"
RES=experiments/exp5_basis_accuracy/results
mkdir -p "$RES"

mem_fraction_for() {   # $1 = system (kept as a hook; every system now takes the default)
  case "$1" in
    *) echo "${MEM_FRACTION_DEFAULT:-0.95}" ;;
  esac
}

case "$GROUP" in
  ladder)   # the CBS tier, largest-first so the long poles start on wave 1
    CONFIGS=()
    for b in cc-pv5z cc-pvqz cc-pvtz cc-pvdz; do
      for s in ${SYSTEMS:-alanine_dipeptide penicillin ethanol h2o}; do CONFIGS+=("$s $b df"); done
    done ;;
  refmode)  # conv (exact in-core ERI) vs df (RI-JK) at the top rungs where BOTH still fit.
    CONFIGS=("h2o cc-pvqz conv" "h2o cc-pvqz df"
             "h2o cc-pv5z conv" "h2o cc-pv5z df"
             "ethanol cc-pvtz conv" "ethanol cc-pvtz df") ;;
  *) echo "unknown GROUP=$GROUP (ladder|refmode)" >&2; exit 1 ;;
esac

# submit.sh already runs us inside `uv run`, so python IS the venv's; standalone, go through uv.
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

idx=0; n_running=0; declare -A pid2gpu
launch_on() {  # $1 = gpu id. Pulls configs until one launches, or falls through when drained.
  local g=$1
  while [ "$idx" -lt "${#CONFIGS[@]}" ]; do
    local spec="${CONFIGS[$idx]}"; idx=$((idx+1))
    local sys basis mode; read -r sys basis mode <<< "$spec"
    local log="$RES/${GROUP}_${sys}_${basis}_${mode}.log"
    if grep -qa "^RESULT " "$log" 2>/dev/null; then echo "GPU $g -- $sys $basis $mode done, skip"; continue; fi
    local mf; mf="$(mem_fraction_for "$sys")"
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
      XLA_PYTHON_CLIENT_MEM_FRACTION="$mf" \
      "${PY[@]}" -m experiments.exp5_basis_accuracy.baseline_gto experiment=exp5_baseline_gto \
      system="$sys" basis="$basis" mode="$mode" >> "$log" 2>&1 &
    pid2gpu[$!]=$g; n_running=$((n_running+1))
    echo "GPU $g <- $sys $basis $mode memfrac=$mf (pid $!)"; return 0
  done
}
for g in $(seq 0 $((NGPU-1))); do launch_on "$g"; done
while [ "$n_running" -gt 0 ]; do
  wait -n || true
  for pid in "${!pid2gpu[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      g=${pid2gpu[$pid]}; unset 'pid2gpu[$pid]'; n_running=$((n_running-1))
      launch_on "$g"      # re-launch bumps n_running again when it finds work
    fi
  done
done
echo "=== exp2 ${GROUP} GTO baselines drained ==="
grep -aH "^RESULT " "$RES"/gto_*.log 2>/dev/null | sed 's#.*/##'
