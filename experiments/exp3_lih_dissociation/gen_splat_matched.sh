#!/usr/bin/env bash
# Appendix F: warm-started splat chains at M = nao(cc-pVTZ) and nao(cc-pVQZ); the cc-pVDZ-matched
# chains (LiH 19, LiF 28) come from run_warm.sh.
#   LiH: M=44 (TZ), M=85 (QZ)   |   LiF: M=60 (TZ), M=110 (QZ)
set -u
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
RES=experiments/exp3_lih_dissociation/results
chain() {  # <MOL> <prefix> <M>
  local log="$RES/${2}splat_match_M${3}.log"
  echo "=== $1 M=$3 ($(date +%H:%M:%S)) ===" | tee -a "$RES/splat_matched.log"
  uv run python -u -m experiments.exp3_lih_dissociation.splat_continuation \
    experiment=exp3_continuation system="$1" m="$3" \
    2>&1 | tee "$log" | grep -a "^RESULT " >> "$RES/splat_matched.log"
}
: > "$RES/splat_matched.log"
chain lih ""    44
chain lih ""    85
chain lif "lif_" 60
chain lif "lif_" 110
echo "=== SPLAT MATCHED DONE ($(date +%H:%M:%S)) ===" | tee -a "$RES/splat_matched.log"
