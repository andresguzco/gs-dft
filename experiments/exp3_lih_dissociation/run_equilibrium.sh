#!/bin/bash
# Figure 4(b): LiF near equilibrium, the Gaussian ladder and cold-start splats at M = nao(cc-pVTZ).
#
#   bash experiments/exp3_lih_dissociation/run_equilibrium.sh
#
# Env: M (60), STEPS (12000), SEEDS ("0 1 2").
set -uo pipefail
cd "$(dirname "$0")/../.."
M="${M:-60}"                       # nao(cc-pVTZ) for LiF: the matched-count rung
STEPS="${STEPS:-12000}"
SEEDS="${SEEDS:-0 1 2}"
RGRID="1.50 1.55 1.60 1.65 1.70 1.75"
BASES="cc-pvdz cc-pvtz cc-pvqz"
RES=experiments/exp3_lih_dissociation/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi
run() { PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 "${PY[@]}" "$@"; }

for R in $RGRID; do
  for B in $BASES; do
    log="$RES/eq_gto_lif_r${R}_${B}.log"
    grep -qa "^RESULT " "$log" 2>/dev/null && continue
    run -m experiments.exp3_lih_dissociation.gto_curve experiment=exp3_gto_curve \
        system=lif r="$R" basis="$B" >> "$log" 2>&1
    grep -a "^RESULT" "$log" | tail -1
  done
done
for R in $RGRID; do
  for S in $SEEDS; do
    log="$RES/eq_splat_lif_r${R}_M${M}_s${S}.log"
    grep -qa "^RESULT " "$log" 2>/dev/null && continue
    echo "--- splat lif r=$R M=$M seed=$S  [$(date +%H:%M:%S)]"
    run -m experiments.exp3_lih_dissociation.splat_curve experiment=exp3_splat_curve \
        system=lif r="$R" m="$M" seed="$S" steps="$STEPS" >> "$log" 2>&1
    grep -a "^RESULT" "$log" | tail -1
  done
done
echo "=== equilibrium scan drained ==="
