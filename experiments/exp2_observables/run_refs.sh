#!/bin/bash
# Gaussian reference observables for one system across the cc-pVXZ ladder, one basis at a time.
#
#   bash experiments/exp2_observables/run_refs.sh water [cc-pvdz cc-pvtz ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
SYSTEM="${1:?usage: run_refs.sh <system> [basis ...]}"
shift
BASES=("${@:-cc-pvdz cc-pvtz cc-pvqz cc-pv5z}")
RES=experiments/exp2_observables/results
mkdir -p "$RES"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

for B in ${BASES[@]}; do
  log="$RES/ref_${SYSTEM}_${B}.log"
  if grep -qa "^RESULT " "$log" 2>/dev/null; then
    echo "--- $SYSTEM $B already done, skipping"
    continue
  fi
  echo "--- $SYSTEM $B  [$(date +%H:%M:%S)]"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
    "${PY[@]}" -m experiments.exp2_observables.gto_ref experiment=exp2_gto_ref \
      system="$SYSTEM" basis="$B" >> "$log" 2>&1
  grep -a "^RESULT" "$log" | tail -1
done
echo "=== $SYSTEM references drained ==="
