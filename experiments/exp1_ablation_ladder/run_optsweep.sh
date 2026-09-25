#!/usr/bin/env bash
# The optimizer comparison of Appendix E: one optimizer per call, chosen by index into ARMS.
#
#   ARM_INDEX=0 bash experiments/exp1_ablation_ladder/run_optsweep.sh     # adam
#   experiments/slurm/submit.sh --array 0-20 --gpus 1 --time 12:00:00 -- bash experiments/exp1_ablation_ladder/run_optsweep.sh
#
# Env: OPT_SYS (alanine_dipeptide), OPT_BASIS (cc-pvtz), OPT_STEPS (12000), OPT_SEED (0), OPT_ARMS.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1

OPT_SYS="${OPT_SYS:-alanine_dipeptide}"
OPT_BASIS="${OPT_BASIS:-cc-pvtz}"
OPT_STEPS="${OPT_STEPS:-12000}"
OPT_SEED="${OPT_SEED:-0}"
R=experiments/exp1_ablation_ladder/results
mkdir -p "$R"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

ARMS=(${OPT_ARMS:-adam adamax nadam radam amsgrad adabelief yogi adan novograd \
      adagrad rmsprop adadelta lars lamb fromage lion sign_sgd rprop sgd adamw_wd lbfgs})
IDX="${ARM_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ARM="${ARMS[$IDX]}"
TAG="opt_${ARM}_${OPT_SYS}_s${OPT_SEED}"
LOG="$R/${TAG}.log"

if grep -q "^RESULT .*steps=$OPT_STEPS " "$LOG" 2>/dev/null; then
  echo "=== skip $TAG (already has a RESULT at steps=$OPT_STEPS) ==="; exit 0
fi

echo "=== [$(date +%H:%M:%S)] $TAG :: $OPT_SYS/$OPT_BASIS steps=$OPT_STEPS ==="
CUDA_VISIBLE_DEVICES=0 "${PY[@]}" -m experiments.exp1_ablation_ladder.optimizers \
  experiment=exp1_optsweep system="$OPT_SYS" basis="$OPT_BASIS" \
  opt="$ARM" seed="$OPT_SEED" steps="$OPT_STEPS" > "$LOG" 2>&1
echo "    exit=$?"
grep -a "^RESULT " "$LOG" | tail -1
