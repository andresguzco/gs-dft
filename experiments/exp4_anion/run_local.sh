#!/bin/bash
# Figure 4(a) and Appendix F: the Gaussian references and the splat M-sweep for both anions, on
# one GPU. Finished runs are skipped, so a re-run resumes.
#
#   bash experiments/exp4_anion/run_local.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
RES=experiments/exp4_anion/results
mkdir -p "$RES"
# submit.sh runs us inside `uv run` (VIRTUAL_ENV set); `uv` is not on a compute node's PATH.
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

# Gaussian references
for a in f_anion oh_anion; do
  for b in cc-pvdz cc-pvtz cc-pvqz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz; do
    log="$RES/gto_${a}_${b}.log"
    grep -qa "^RESULT " "$log" 2>/dev/null && continue
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
      "${PY[@]}" -m experiments.exp4_anion.gto_refs experiment=exp4_gto_refs \
        system="$a" basis="$b" >> "$log" 2>&1
  done
done

declare -A MS=([f_anion]="14 30 55 91" [oh_anion]="19 44 85 146")
for a in f_anion oh_anion; do
  for m in ${MS[$a]}; do
    for s in 0 1 2; do
      log="$RES/splat_${a}_M${m}_s${s}.log"
      grep -qa "^RESULT " "$log" 2>/dev/null && continue
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
        "${PY[@]}" -m experiments.exp4_anion.splat_anion experiment=exp4_splat_anion \
          system="$a" m="$m" seed="$s" >> "$log" 2>&1
    done
  done
done
echo "exp6 drained"; grep -ahc "^RESULT " $RES/*.log | wc -l
