#!/bin/bash
# Appendix D, Table "water": the correctness checks on water, one GPU per code. PBE, plus
# Hartree-Fock (--xc hfx) where a code's PBE result needs a cross-check.
#
#   bash experiments/exp9_frameworks/run_water_node.sh
E="${APPD_ENVS:-${SCRATCH:-$HOME}/appD_envs}"; L="${APPD_LOGS:-experiments/exp9_frameworks/results}/water"
cd "$(dirname "$0")/../.."; mkdir -p "$L"
w() { local env=$1 name=$2; shift 2
  "$E/$env/bin/python" experiments/exp9_frameworks/run_code.py --system water "$@" > "$L/$name.log" 2>&1
  grep -h RESULT "$L/$name.log"; }
( for b in sto-3g 6-31g cc-pvdz cc-pvtz; do w pyscf pyscf_$b --code pyscf --basis $b; w pyscf pyscf_cart_$b --code pyscf --basis $b --cart; done
  w pyscf pyscf_cart_hf --code pyscf --basis cc-pvdz --cart --xc hfx
  w dqc dqc_dz --code dqc --basis cc-pvdz --device cpu; w dqc dqc_tz --code dqc --basis cc-pvtz --device cpu ) &
( for b in sto-3g 6-31g cc-pvdz; do w mess mess_$b --code mess --basis $b --gpu 1; done
  w mess mess_hf --code mess --basis cc-pvdz --xc hfx --gpu 1 ) &
( w d4ft d4ft_sto3g --code d4ft --basis sto-3g --f64 --gpu 2; w d4ft d4ft_dz_f64 --code d4ft --basis cc-pvdz --f64 --gpu 2 ) &
( w d4ft d4ft_dz_f32 --code d4ft --basis cc-pvdz --gpu 3; w d4ft d4ft_tz_f64 --code d4ft --basis cc-pvtz --f64 --gpu 3 ) &
wait
