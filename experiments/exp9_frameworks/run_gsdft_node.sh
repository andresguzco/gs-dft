#!/bin/bash
# Appendix D, the GS-DFT rows: the water check at M = nao(cc-pVDZ) = 24, then the cc-pVTZ ladder
# with the production trainer at M = nao(cc-pVTZ), grid level 3, sharded over $GPUS. 100 steps per
# rung, so the per-step time is measured past compilation.
#
#   GPUS=0,1,2,3 bash experiments/exp9_frameworks/run_gsdft_node.sh
cd "$(dirname "$0")/../.."; R=experiments/exp9_frameworks/results; mkdir -p "$R"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "${PY[@]}" -m experiments.exp6_cost_benchmark.bench experiment=exp6_bench \
  system=water engine=splat m=24 > "$R/gsdft_water_dz_M24.log" 2>&1
for s in water ethanol alanine_dipeptide ala_5 ala_15 ala_45 insulin; do
  CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "${PY[@]}" \
    -m experiments.exp7_size_ladder.train experiment=exp7_train system="$s" tag="${s}_appD" out_dir="$R" \
    basis=cc-pvtz m_mult=1.0 grid_level=3 steps=100 monitor=10 grid_check=0 collapse=false \
    resume=false > "$R/gsdft_${s}.log" 2>&1
done
