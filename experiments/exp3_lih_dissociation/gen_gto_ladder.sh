#!/usr/bin/env bash
# Appendix F: the Gaussian ladder (cc-pVDZ/TZ/QZ and aug-cc-pVXZ) along both dissociation curves.
set -u
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=$PWD                       # GPU-first: dftax's own RKS reference runs on the GPU
RES=experiments/exp3_lih_dissociation/results
gto() {  # <MOL> <prefix> <R> <basis>
  local log="$RES/${2}gto_R${3}_${4}.log"
  [ -s "$log" ] && grep -q "^RESULT " "$log" && { echo "skip $2 R${3} $4"; return; }
  uv run python -m experiments.exp3_lih_dissociation.gto_curve experiment=exp3_gto_curve \
    system="$1" r="$3" basis="$4" > "$log" 2>&1
  grep -a "^RESULT " "$log"
}
for R in 1.0 1.3 1.6 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0; do
  for B in cc-pvtz cc-pvqz aug-cc-pvtz aug-cc-pvqz; do gto lih "" "$R" "$B"; done
done
for R in 1.2 1.5 1.8 2.2 2.8 3.5 4.5 6.0; do
  for B in cc-pvqz aug-cc-pvtz aug-cc-pvqz; do gto lif "lif_" "$R" "$B"; done
done
echo "=== GTO LADDER DONE ==="
