#!/usr/bin/env bash
# The runs behind Figure 2 and the eigensolve ablation of Appendix E, one phase per call.
#
#   PHASE=p1    bash experiments/exp1_ablation_ladder/run_panels.sh   # (a) FSGO, GS-DFT and Gaussian trajectories
#   PHASE=p2    bash experiments/exp1_ablation_ladder/run_panels.sh   # (b) peak memory against free parameters
#   PHASE=p3    bash experiments/exp1_ablation_ladder/run_panels.sh   # (c) and Appendix E: one arm at a time
#   PHASE=p3par bash experiments/exp1_ablation_ladder/run_panels.sh   # (c) all three arms at once on one node
#
# P3_SYS picks the system for p3 (ala_45 for Figure 2(c); ch4, benzene, co2 and n2 with
# P3_ARMS="P3_nofloor P3_floor" for Appendix E). A run is skipped when its log already holds a
# RESULT line; logs go to experiments/exp1_ablation_ladder/results/.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1

PHASE="${PHASE:-p3}"
STEPS="${STEPS:-300}"
R=experiments/exp1_ablation_ladder/results
mkdir -p "$R"
SUMMARY="$R/panel_${PHASE}.log"

if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

run() {  # run <tag> <module> <overrides...>       env: MATCH=<substring the RESULT must contain>
  local tag="$1"; shift; local mod="$1"; shift
  local log="$R/${tag}.log"
  if grep -q "^RESULT " "$log" 2>/dev/null; then
    local tok ok=1
    for tok in ${MATCH:-}; do
      grep -q "^RESULT .*${tok}" "$log" 2>/dev/null || { ok=0; break; }
    done
    if [ "$ok" = 1 ]; then
      echo "=== skip $tag (already has a RESULT${MATCH:+ matching: ${MATCH}}) ==="
      grep -h "^RESULT " "$log" >> "$SUMMARY"; return
    fi
    echo "=== re-run $tag: its RESULT lacks '${tok}' ==="
  fi
  echo "=== [$(date +%H:%M:%S)] $tag :: $* ==="
  [ -n "${APPEND:-}" ] || : > "$log"
  local rc
  if [ -n "${TIMEOUT:-}" ]; then
    timeout "$TIMEOUT" "${PY[@]}" -m "experiments.exp1_ablation_ladder.$mod" "$@" >> "$log" 2>&1
    rc=$?; [ "$rc" = 124 ] && echo "    TIMED OUT after ${TIMEOUT}s — no ceiling measured, point absent"
  else
    "${PY[@]}" -m "experiments.exp1_ablation_ladder.$mod" "$@" >> "$log" 2>&1; rc=$?
  fi
  echo "    exit=$rc"
  grep -h "^RESULT " "$log" >> "$SUMMARY" || echo "    (no RESULT — the runner died; read $log)"
}

case "$PHASE" in
p1)  # ---- Figure 2(a): three trajectories on one optimizer-step axis ----------------------
  P1_SYS="${P1_SYS:-h2o}"; P1_BASIS="${P1_BASIS:-cc-pvtz}"; P1_STEPS="${FSTEPS:-6000}"
  P1_NAO="$(NAO_SYS="$P1_SYS" NAO_BASIS="$P1_BASIS" "${PY[@]}" - <<'EOF'
import os
from gs_dft import nao
from experiments.common import systems
print(int(nao(systems.molecule(system=os.environ["NAO_SYS"], basis=os.environ["NAO_BASIS"]))))
EOF
)"
  echo "=== panel 1: $P1_SYS/$P1_BASIS nao=$P1_NAO, $P1_STEPS steps ==="
  for arm in old new; do
    CUDA_VISIBLE_DEVICES=0 MATCH="nao=$P1_NAO steps=$P1_STEPS" run "p1_${arm}" ladder \
      experiment=exp1_ladder system="$P1_SYS" basis="$P1_BASIS" rung="$arm" \
      m_mult=1.0 steps="$P1_STEPS" grid_level=3 grid_check=5
  done
  CUDA_VISIBLE_DEVICES=0 MATCH="nao=$P1_NAO steps=$P1_STEPS" \
    run "p1_gto" gto_minimize "$P1_SYS" "$P1_BASIS" "$P1_STEPS" ;;

p2)  # ---- Figure 2(b): peak memory against free parameters, up to each arm's ceiling ----
  P2_SYS="${P2_SYS:-alanine_dipeptide}"
  P2_STEPS="${P2_STEPS:-5}"; P2_REFRESH="${P2_REFRESH:-2}"
  P2_DENSE_STEPS="${P2_DENSE_STEPS:-2}"
  P2_DENSE_TIMEOUT="${P2_DENSE_TIMEOUT:-3600}"
  P2_TAG="${P2_TAG:+_$P2_TAG}"
  for M in ${P2_MS-200 468 910 1820 3640}; do
    for arm in P2_df P2_fast; do
      CUDA_VISIBLE_DEVICES=0 MATCH="steps=$P2_STEPS " run "p2_${arm}_M${M}${P2_TAG}" machinery \
        experiment=exp1_machinery system="$P2_SYS" m="$M" rung="$arm" \
        steps="$P2_STEPS" refresh="$P2_REFRESH" grid_level=3
    done
  done
  for arm in ${P2_GTO_ARMS-gto_exact gto_df gto_fast}; do
    for B in ${P2_BASES-cc-pvdz cc-pvtz cc-pvqz}; do
      CUDA_VISIBLE_DEVICES=0 TIMEOUT="${P2_GTO_TIMEOUT:-2400}" \
        run "p2_${arm}_${B}" gto_memory "$P2_SYS" "$B" "$arm"
    done
  done
  for M in ${P2_MS_DENSE-200 300 468 560 660}; do
    CUDA_VISIBLE_DEVICES=0 TIMEOUT="$P2_DENSE_TIMEOUT" MATCH="steps=$P2_DENSE_STEPS " \
      run "p2_P2_dense_M${M}${P2_TAG}" machinery \
      experiment=exp1_machinery system="$P2_SYS" m="$M" rung=P2_dense \
      steps="$P2_DENSE_STEPS" grid_level=3
  done
  ;;

p3)  # ---- Figure 2(c) and Appendix E: energy against wall-clock time -------------------
  P3_SYS="${P3_SYS:-ala_45}"
  P3_BASIS="${P3_BASIS:-cc-pvdz}"       # ala_45: nao(DZ)=4299 · alanine_dipeptide: pass cc-pvtz (468)
  MSPEC=(basis="$P3_BASIS" m_mult=1.0)  # M := nao(cardinal), the matched-count rule
  echo "=== panel 3: $P3_SYS / $P3_BASIS, $STEPS steps ==="
  ALL_DEVS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
  ALL_DEVS="${ALL_DEVS:-0}"
  echo "=== devices in this allocation: ${ALL_DEVS} ==="
  for arm in ${P3_ARMS:-P3_nofloor P3_floor P3_multi}; do
    case "$arm" in
      P3_multi) DEV="$ALL_DEVS" ;;                         # every device in the allocation
      *)        DEV="0" ;;                                 # single device
    esac
    CUDA_VISIBLE_DEVICES="$DEV" MATCH="steps=$STEPS seed=${P3_SEED:-0} " \
      CKPT_SUFFIX="${P3_REP:+r$P3_REP}" \
      run "p3_${arm}_${P3_SYS}_s${P3_SEED:-0}${P3_REP:+_r$P3_REP}" machinery \
      experiment=exp1_machinery system="$P3_SYS" "${MSPEC[@]}" rung="$arm" \
      seed="${P3_SEED:-0}" steps="$STEPS" refresh="${P3_REFRESH:-50}" \
      screen_pad="${P3_PAD:-2.0}" screen_eps="${P3_EPS:-1e-7}" grid_level="${P3_GRID:-3}"
  done ;;

p3par)  # ---- Figure 2(c), all three arms at once on one node ------------------------------
  P3_SYS="${P3_SYS:-ala_45}"
  P3_BASIS="${P3_BASIS:-cc-pvdz}"
  MSPEC=(basis="$P3_BASIS" m_mult=1.0)
  DEVMAP="${P3_DEVMAP:-P3_nofloor:0 P3_floor:1 P3_multi:2,3}"
  export APPEND="${APPEND:-1}"
  echo "=== panel 3 (concurrent): $P3_SYS / $P3_BASIS, $STEPS steps ==="
  echo "=== devices in this allocation: $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || echo '?') ==="
  pids=()
  for spec in $DEVMAP; do
    arm="${spec%%:*}"; DEV="${spec##*:}"
    echo "=== arm $arm -> CUDA_VISIBLE_DEVICES=$DEV"
    CUDA_VISIBLE_DEVICES="$DEV" MATCH="steps=$STEPS seed=${P3_SEED:-0} " \
      CKPT_SUFFIX="${P3_REP:+r$P3_REP}" \
      run "p3_${arm}_${P3_SYS}_s${P3_SEED:-0}${P3_REP:+_r$P3_REP}" machinery \
      experiment=exp1_machinery system="$P3_SYS" "${MSPEC[@]}" rung="$arm" \
      seed="${P3_SEED:-0}" steps="$STEPS" refresh="${P3_REFRESH:-50}" \
      screen_pad="${P3_PAD:-2.0}" screen_eps="${P3_EPS:-1e-7}" grid_level="${P3_GRID:-3}" &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then echo "    ARM FAILED: $(echo $DEVMAP | cut -d' ' -f$((i+1)))"; fail=1; fi
  done
  echo "=== all arms returned (fail=$fail) ===" ;;

*)
  echo "unknown PHASE=$PHASE (p1|p2|p3|p3par)" >&2; exit 2 ;;
esac

echo "=== [$(date +%H:%M:%S)] $PHASE complete -> $SUMMARY ==="
