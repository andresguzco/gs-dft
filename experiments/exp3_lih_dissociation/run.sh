#!/bin/bash
# Figure 4 and Appendix F: the Gaussian points and the cold-start splat M-sweep along the LiH or
# LiF dissociation curve, as a work queue over the GPUs.
#
#   SYSTEM=lih bash experiments/exp3_lih_dissociation/run.sh
#
# Env: SYSTEM (lih|lif), NGPU (4).
set -uo pipefail

STEPS="${STEPS:-4000}"
GRID_LEVEL="${GRID_LEVEL:-3}"
NGPU="${NGPU:-4}"
RES=experiments/exp3_lih_dissociation/results
EXP=experiments/exp3_lih_dissociation
mkdir -p "$RES"

SYS="${SYSTEM:-lih}"
case "$SYS" in
  lih) R_GRID=(1.0 1.3 1.6 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0)   # mirrors common.systems.R_GRID["lih"]
       M_GRID=(19 32 44 69 85) ;;                             # DZ augDZ TZ augTZ QZ function counts
  lif) R_GRID=(1.2 1.5 1.8 2.2 2.8 3.5 4.5 6.0)               # mirrors common.systems.R_GRID["lif"]
       M_GRID=(28 46 60 92 110) ;;
  *) echo "unknown SYSTEM=$SYS (lih|lif)" >&2; exit 1 ;;
esac
GTO_BASES=(cc-pvdz aug-cc-pvdz)

# submit.sh already runs us inside `uv run`, so python IS the venv's; standalone, go through uv.
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi
run() { PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 "${PY[@]}" "$@"; }

# --- GTO curves (instant, GPU, serial) ---
for R in "${R_GRID[@]}"; do for B in "${GTO_BASES[@]}"; do
  log="$RES/gto_${SYS}_R${R}_${B}.log"
  grep -qa "^RESULT " "$log" 2>/dev/null && continue
  run -m experiments.exp3_lih_dissociation.gto_curve experiment=exp3_gto_curve \
    system="$SYS" r="$R" basis="$B" >> "$log" 2>&1
done; done
echo "=== GTO curves done ==="; grep -ah "^RESULT " "$RES"/gto_*.log | sed 's/  */ /g'

# --- splat M-sweep (work-queue over 4 GPUs) ---
CONFIGS=(); for R in "${R_GRID[@]}"; do for M in "${M_GRID[@]}"; do CONFIGS+=("$R $M"); done; done
idx=0; declare -A pid2gpu
launch_on() {  # $1 = gpu id. Pulls configs until one launches, or falls through when drained.
  local g=$1
  while [ "$idx" -lt "${#CONFIGS[@]}" ]; do
    local spec="${CONFIGS[$idx]}"; idx=$((idx+1))
    local R M; read -r R M <<< "$spec"
    local log="$RES/splat_${SYS}_R${R}_M${M}.log"
    # "^RESULT " with the space: the splat runner also writes RESULT_PARTIAL lines.
    if grep -qa "^RESULT " "$log" 2>/dev/null; then continue; fi
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
      "${PY[@]}" -m experiments.exp3_lih_dissociation.splat_curve experiment=exp3_splat_curve \
      system="$SYS" r="$R" m="$M" steps="$STEPS" grid_level="$GRID_LEVEL" >> "$log" 2>&1 &
    pid2gpu[$!]=$g; echo "GPU $g <- R$R M$M (pid $!)"; return 0
  done
}
for g in $(seq 0 $((NGPU-1))); do launch_on "$g"; done
# Refill pool: `wait -n` wakes on ANY child, so a freed GPU pulls the next config immediately rather
# than idling until the wave ends. `|| true` keeps one crashed config from killing the queue.
while [ "${#pid2gpu[@]}" -gt 0 ]; do
  wait -n || true
  for pid in "${!pid2gpu[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then g=${pid2gpu[$pid]}; unset 'pid2gpu[$pid]'; launch_on "$g"; fi
  done
done
echo "=== exp3 $SYS splat sweep drained ==="
grep -aH "^RESULT" "$RES"/splat_*.log 2>/dev/null | sed 's#.*/##'
