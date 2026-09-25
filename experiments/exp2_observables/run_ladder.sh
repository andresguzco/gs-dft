#!/bin/bash
# Figure 3 for water and ethanol: the Gaussian references, the Gaussian ladder's own density error,
# then per rung the splat checkpoint, its force polish and its observables.
#
#   bash experiments/exp2_observables/run_ladder.sh water ethanol
#
# Env: STEPS (12000), PAD (2.0), FORCE_MAX_NAO (200).
set -uo pipefail
cd "$(dirname "$0")/../.."
STEPS="${STEPS:-12000}"
PAD="${PAD:-2.0}"
PAD_FMT=$(printf "%.8f" "$PAD")
SYSTEMS=("${@:-water ethanol}")
FORCE_MAX_NAO="${FORCE_MAX_NAO:-200}"
RES=experiments/exp2_observables/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi
run() { PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 "${PY[@]}" "$@"; }

declare -A REF_BASES=(
  [water]="cc-pvdz cc-pvtz cc-pvqz cc-pv5z"
  [ethanol]="cc-pvdz cc-pvtz cc-pvqz cc-pv5z"
  [alanine_dipeptide]="cc-pvdz cc-pvtz cc-pvqz"
)
declare -A LADDER=( [water]="24 58 115" [ethanol]="72 174 345" [alanine_dipeptide]="200 468 910" )

for S in ${SYSTEMS[@]}; do
  MS=(${LADDER[$S]})
  # --- Gaussian references first: they are the reference the splat side is scored against, and
  # they fail in ways (host OOM at QZ) that are better discovered before spending training time.
  for B in ${REF_BASES[$S]}; do
    log="$RES/ref_${S}_${B}.log"
    if grep -qa "^RESULT " "$log" 2>/dev/null; then echo "--- ref $S $B done"; continue; fi
    echo "=== ref $S $B  [$(date +%H:%M:%S)]"
    run -m experiments.exp2_observables.gto_ref experiment=exp2_gto_ref \
        system="$S" basis="$B" force_max_nao="$FORCE_MAX_NAO" >> "$log" 2>&1
    grep -a "^RESULT" "$log" | tail -1
  done
  # --- the Gaussian ladder's own density error, against its top rung (needs every reference)
  l1log="$RES/refl1_${S}.log"
  if ! grep -qa "^RESULT kind=gtol1" "$l1log" 2>/dev/null; then
    echo "=== refL1 $S  [$(date +%H:%M:%S)]"
    run -m experiments.exp2_observables.ref_l1 experiment=exp2_observables system="$S" \
        >> "$l1log" 2>&1
    grep -a "^RESULT" "$l1log" | tail -1
  fi

  # --- the cloud ladder: train, then observables, per rung
  for M in "${MS[@]}"; do
    log="$RES/ckpt_${S}_M${M}.log"
    if ! grep -qa "^RESULT kind=ckpt.*steps=$STEPS .*screen_pad=$PAD_FMT" "$log" 2>/dev/null; then
      echo "=== train $S M=$M  [$(date +%H:%M:%S)]"
      run -m experiments.exp2_observables.train_ckpt experiment=exp2_train_ckpt \
          system="$S" m="$M" steps="$STEPS" screen_pad="$PAD" >> "$log" 2>&1
      grep -a "^RESULT" "$log" | tail -1
    fi
    plog="$RES/polish_${S}_M${M}.log"
    if ! grep -qa "^RESULT kind=forces" "$plog" 2>/dev/null; then
      echo "=== polish $S M=$M  [$(date +%H:%M:%S)]"
      run -m experiments.exp2_observables.polish_forces "$S" "$M" >> "$plog" 2>&1
      grep -a "^RESULT" "$plog" | tail -1
    fi
    olog="$RES/obs_${S}_M${M}.log"
    if ! grep -qa "^RESULT kind=obs" "$olog" 2>/dev/null; then
      echo "=== obs   $S M=$M  [$(date +%H:%M:%S)]"
      run -m experiments.exp2_observables.observables experiment=exp2_observables \
          system="$S" m="$M" >> "$olog" 2>&1
      grep -a "^RESULT" "$olog" | tail -1
    fi
  done
done
echo "=== ladder drained ==="
grep -ah "^RESULT" "$RES"/ckpt_*_M*.log "$RES"/obs_*_M*.log "$RES"/ref_*.log "$RES"/refl1_*.log 2>/dev/null | wc -l
