#!/bin/bash
# The LiF dissociation curve on one GPU: the Gaussian points and the warm-started splat chains.
# Finished points are skipped, so a re-run resumes.
#
#   bash experiments/exp3_lih_dissociation/run_lif_local.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
RES=experiments/exp3_lih_dissociation/results
mkdir -p "$RES"
RGRID="1.2 1.5 1.8 2.2 2.8 3.5 4.5 6.0"
export PYTHONPATH=$PWD
D=experiments/exp3_lih_dissociation

echo "=== GTO baselines (GPU): cc-pVDZ (cookie-cutter) | aug-cc-pVDZ (truth) | cc-pVTZ (ref) ==="
for R in $RGRID; do
  for b in cc-pvdz aug-cc-pvdz cc-pvtz; do
    log="$RES/lif_gto_R${R}_${b}.log"
    [ -f "$log" ] && grep -qa "^RESULT " "$log" && { echo "skip gto $R $b"; continue; }
    uv run python -m experiments.exp3_lih_dissociation.gto_curve \
      experiment=exp3_gto_curve system=lif r=$R basis=$b > "$log" 2>&1
    grep -a "^RESULT " "$log" || echo "FAILED gto $R $b"
  done
done

echo "=== splat full-cov warm-continuation chains (GPU): M = 1x/2x/3x nao(28) ==="
for M in 28 56 84; do
  log="$RES/lif_splat_M${M}.log"
  # skips a chain with any RESULT line; run_warm.sh checks that every R point landed
  [ -f "$log" ] && grep -qa "^RESULT kind=splat " "$log" && { echo "skip splat M$M"; continue; }
  CUDA_VISIBLE_DEVICES=0 \
    uv run python -u -m experiments.exp3_lih_dissociation.splat_continuation \
      experiment=exp3_continuation system=lif m=$M > "$log" 2>&1
  grep -a "^RESULT kind=splat " "$log" | tail -3 || echo "FAILED splat M$M"
done
echo "=== LiF dissociation run complete ==="
grep -ah "^RESULT" $RES/lif_*.log | wc -l
