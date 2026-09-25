#!/bin/bash
# Appendix D, Table "scaling": the cc-pVTZ size ladder for one external code, climbing until the
# first failure. The ladder measures memory and cost per iteration, not converged energies, so
# CAP=<n> caps the iterations on rungs that would otherwise run for hours.
#
#   CAP=3 bash experiments/exp9_frameworks/run_ladder.sh <pyscf|d4ft|mess|dqc>
C="$1"; CAP="${CAP:-0}"; E="${APPD_ENVS:-${SCRATCH:-$HOME}/appD_envs}"
cd "$(dirname "$0")/../.."; L="${APPD_LOGS:-experiments/exp9_frameworks/results}/ladder"; mkdir -p "$L"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
case "$C" in
  pyscf) env=pyscf; args="--code pyscf --threads $OMP_NUM_THREADS" ;;
  d4ft)  env=d4ft;  args="--code d4ft --f64 --gpu ${GPUS:-0}" ;;
  mess)  env=mess;  args="--code mess --gpu ${GPUS:-0}" ;;
  dqc)   env=dqc;   args="--code dqc --device cpu --threads $OMP_NUM_THREADS" ;;
  *) echo "unknown code $C (pyscf|d4ft|mess|dqc)" >&2; exit 2 ;;
esac
for s in water ethanol alanine_dipeptide ala_5 ala_15 ala_45 insulin; do
  "$E/$env/bin/python" experiments/exp9_frameworks/run_code.py --system "$s" --basis cc-pvtz --max-iter "$CAP" $args > "$L/${C}_$s.log" 2>&1
  grep -h RESULT "$L/${C}_$s.log"
  grep -q '"status": "ok"' "$L/${C}_$s.log" || break
done
