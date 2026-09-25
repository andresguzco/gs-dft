#!/usr/bin/env bash
# Figure 3: the splat forces on the alanine dipeptide ladder, one rung per GPU. The Gaussian force
# references for this system come from pyscf_forces.py.
#
#   bash experiments/exp2_observables/run_forces_dipeptide.sh
#
# Env: SYS(alanine_dipeptide) MS("200 468 910")
set -uo pipefail
cd "$(dirname "$0")/../.."
SYS="${SYS:-alanine_dipeptide}"; MS="${MS:-200 468 910}"
RES=experiments/exp2_observables/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

# --- splat side: one rung per GPU, they are independent
g=0
for M in $MS; do
  log="$RES/polish_${SYS}_M${M}.log"
  if grep -qa "^RESULT kind=forces" "$log" 2>/dev/null; then echo "--- splat M=$M done"; continue; fi
  echo "GPU $g <- splat forces $SYS M=$M  [$(date +%H:%M:%S)]"
  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
    "${PY[@]}" -m experiments.exp2_observables.polish_forces "$SYS" "$M" > "$log" 2>&1 &
  g=$((g+1))
done
wait
echo "=== splat forces drained ==="

echo "=== dipeptide force column done ==="
grep -h "^RESULT kind=forces" "$RES"/polish_${SYS}_M*.log 2>/dev/null
