#!/bin/bash
# Table 1: ethanol at M = nao(cc-pVTZ) = 174 against cc-pVTZ, for each functional. One work item
# per (Gaussian or splat, functional), as a queue over the GPUs.
#
#   NGPU=4 bash experiments/exp5_basis_accuracy/run_functionals.sh
#
# Env: SYS (ethanol), BASIS (cc-pvtz), M (174), STEPS (48000), XCS, NGPU (1).
set -uo pipefail
cd "$(dirname "$0")/../.."
SYS="${SYS:-ethanol}"; BASIS="${BASIS:-cc-pvtz}"; M="${M:-174}"
STEPS="${STEPS:-48000}"; XCS="${XCS:-pbe b3lyp r2scan wb97m-v}"
NGPU="${NGPU:-1}"
RES=experiments/exp5_basis_accuracy/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

# One work item per (kind, functional); a freed GPU pulls the next, as in run_baselines.sh.
work=()
for XC in $XCS; do work+=("gto:$XC" "splat:$XC"); done
next=0
declare -A pid2gpu; n_running=0

launch_on() {
  local g="$1" item kind XC log
  while [ "$next" -lt "${#work[@]}" ]; do
    item="${work[$next]}"; next=$((next+1))
    kind="${item%%:*}"; XC="${item##*:}"

    if [ "$kind" = gto ]; then
      log="$RES/func_gto_${SYS}_${BASIS}_${XC}.log"
      if grep -qa "^RESULT .*xc=${XC}\b" "$log" 2>/dev/null; then
        echo "--- gto $SYS/$BASIS xc=$XC done"; continue; fi
      echo "GPU $g <- gto  $SYS/$BASIS xc=$XC  [$(date +%H:%M:%S)]"
      CUDA_VISIBLE_DEVICES=$g "${PY[@]}" -m experiments.exp2_observables.gto_ref \
          experiment=exp2_gto_ref system="$SYS" basis="$BASIS" xc="$XC" force_max_nao=0 \
          > "$log" 2>&1 &
    else
      log="$RES/func_splat_${SYS}_M${M}_${XC}_${STEPS}steps.log"
      if grep -qa "^RESULT .*steps=${STEPS} .*xc=${XC}\b" "$log" 2>/dev/null; then
        echo "--- splat $SYS M=$M xc=$XC steps=$STEPS done"; continue; fi
      echo "GPU $g <- splat $SYS M=$M xc=$XC steps=$STEPS  [$(date +%H:%M:%S)]"
      CUDA_VISIBLE_DEVICES=$g "${PY[@]}" -m experiments.exp5_basis_accuracy.splat_run \
          experiment=exp5_splat_run system="$SYS" m="$M" xc="$XC" steps="$STEPS" \
          > "$log" 2>&1 &
    fi
    pid2gpu[$!]=$g; n_running=$((n_running+1))
    return 0
  done
  return 1
}

for g in $(seq 0 $((NGPU-1))); do launch_on "$g" || break; done
# Refill pool; an explicit counter because an empty associative array is unbound under `set -u`.
while [ "$n_running" -gt 0 ]; do
  wait -n || true
  for pid in "${!pid2gpu[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      g=${pid2gpu[$pid]}; unset "pid2gpu[$pid]"; n_running=$((n_running-1))
      launch_on "$g" || true
    fi
  done
done

for XC in $XCS; do
  grep -ah "^RESULT" "$RES/func_gto_${SYS}_${BASIS}_${XC}.log" \
       "$RES/func_splat_${SYS}_M${M}_${XC}_${STEPS}steps.log" 2>/dev/null
done
echo "=== functional coverage drained ==="
grep -ah "^RESULT" "$RES"/func_*.log 2>/dev/null | wc -l
