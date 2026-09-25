#!/usr/bin/env bash
# Table 2: Hartree-Fock ionization potentials, -eps_HOMO, at M = nao(def2-QZVPP), and the
# def2-QZVPP Gaussian HOMO of each system. Under Hartree-Fock, -eps_HOMO is the Koopmans ionization
# potential, comparable to the CCSD(T) and experimental values of GW100. One GPU per system.
#
#   bash experiments/exp2_observables/run_koopmans.sh
#
# Env: KOOP_STEPS (12000).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi
STEPS=${KOOP_STEPS:-12000}
RES=experiments/exp2_observables/results; mkdir -p "$RES"

run_system() {                                  # $1=gpu  $2=system  $3=M
  local log="$RES/koop_$2_M$3.log"
  CUDA_VISIBLE_DEVICES="$1" "${PY[@]}" -m experiments.exp5_basis_accuracy.msweep experiment=exp5_msweep \
    system="$2" basis=def2-qzvpp m="$3" steps="$STEPS" koopmans=true \
    xc=hf coulomb=df grid_check=0 > "$log" 2>&1
  echo "[koop] splat $2 M=$3 rc=$?"; grep -aE "^RESULT" "$log" | tail -1
  CUDA_VISIBLE_DEVICES="$1" "${PY[@]}" -m experiments.exp2_observables.gto_ref experiment=exp2_gto_ref \
    system="$2" basis=def2-qzvpp xc=hf grid_check=0 >> "$log" 2>&1
  echo "[koop] gto $2 rc=$?"; grep -aE "^RESULT kind=gto" "$log" | tail -1
}

run_system 0 li2 70 &
run_system 1 water 117 &
run_system 2 ethanol 351 &
run_system 3 benzene 522 &
wait
