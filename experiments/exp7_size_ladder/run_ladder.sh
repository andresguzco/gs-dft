#!/usr/bin/env bash
# Run each named system through train.py on a pool of GPUs, GPUS_PER contiguous GPUs per run
# (1 = single GPU, 2 or 4 = sharded). A block is refilled as soon as its run finishes or stops.
# Used by run_fmodb_suite.sh for Table 3 and for the alanine chains.
#
#   bash experiments/exp7_size_ladder/run_ladder.sh ala_5 ala_15            # one GPU per system
#   GPUS_PER=4 bash experiments/exp7_size_ladder/run_ladder.sh PJM49         # one sharded run
#   HYDRA="steps=500 monitor=10" bash experiments/exp7_size_ladder/run_ladder.sh ala_5
#
# Env: NGPU (default: the nvidia-smi count), GPUS_PER (1), SYSTEMS (if no arguments),
#      TAG_SUFFIX (appended to the system name to form the run tag), HYDRA (extra overrides).
# Everything else is Hydra config, see conf/experiment/exp7_train.yaml.
set -uo pipefail
GPUS_PER=${GPUS_PER:-1}
[ "${SLURM_NTASKS:-1}" -gt 1 ] && GPUS_PER=1           # per-rank task owns exactly one GPU
HYDRA=${HYDRA:-}                                     # extra key=value overrides, passed through verbatim
RES=experiments/exp7_size_ladder/results; mkdir -p "$RES"
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}; [ "${NGPU:-0}" -lt 1 ] && NGPU=1
NBLOCK=$(( NGPU / GPUS_PER )); [ "$NBLOCK" -lt 1 ] && NBLOCK=1

systems=("$@")                                       # positional wins; else the SYSTEMS env list
if [ "${#systems[@]}" -eq 0 ]; then
  read -ra systems <<< "${SYSTEMS:-ala_5 ala_15 ala_25 ala_35 ala_45 ala_55 ala_65 insulin}"
fi

# submit.sh already runs us inside `uv run`, so python IS the venv's; standalone, go through uv.
if [ -n "${VIRTUAL_ENV:-}" ]; then PY=(python -u); else PY=(uv run python -u); fi

devs_of_block() {                                    # comma-separated GPU ids for block $1
  local start=$(( $1 * GPUS_PER )) i out=""
  for ((i = 0; i < GPUS_PER; i++)); do out="${out:+$out,}$(( start + i ))"; done
  echo "$out"
}
# train.py names the checkpoint, JSONL and log after the tag, and resume=true appends to them, so a
# new set of runs needs a new TAG_SUFFIX.
tag_of() { echo "$1${TAG_SUFFIX:-}"; }
run_one() {                                          # $1=block  $2=system
  local devs tag log; devs=$(devs_of_block "$1"); tag=$(tag_of "$2"); log="$RES/${tag}.log"
  # Multi-node (one SLURM task per GPU): keep SLURM's device binding and give every rank its own log.
  if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
    devs="${CUDA_VISIBLE_DEVICES:-0}"
    [ "${SLURM_PROCID:-0}" -gt 0 ] && log="$RES/${tag}.rank${SLURM_PROCID}.log"
  fi
  # $HYDRA is unquoted on purpose: it is a list of space-separated key=value overrides.
  CUDA_VISIBLE_DEVICES="$devs" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" \
    "${PY[@]}" -m experiments.exp7_size_ladder.train experiment=exp7_train \
      system="$2" tag="$tag" $HYDRA \
      > "$log" 2>&1
}

free=($(seq 0 $((NBLOCK - 1)))); declare -a running; qi=0
echo "[pool] $NGPU GPUs · $NBLOCK block(s) of $GPUS_PER · ${#systems[@]} systems : ${systems[*]}"
while [ "$qi" -lt "${#systems[@]}" ] || [ "${#running[@]}" -gt 0 ]; do
  while [ "${#free[@]}" -gt 0 ] && [ "$qi" -lt "${#systems[@]}" ]; do
    b=${free[0]}; free=("${free[@]:1}"); s=${systems[$qi]}; qi=$((qi + 1))
    run_one "$b" "$s" & running+=("$!:$b:$s")
    echo "[pool] launch $s -> block $b (gpus $(devs_of_block "$b"), pid $!)"
  done
  sleep 15
  newrun=()
  if [ "${#running[@]}" -gt 0 ]; then                # guard: don't iterate/expand an empty array under set -u
    for e in "${running[@]}"; do
      p=${e%%:*}; bs=${e#*:}; b=${bs%%:*}; s=${bs#*:}
      if kill -0 "$p" 2>/dev/null; then newrun+=("$e"); continue; fi
      free+=("$b")
      # Output goes to a per-run log, so report the exit status here.
      if wait "$p"; then echo "[pool] done $s (block $b freed)"
      else echo "[pool] FAILED $s rc=$? (block $b freed) -- see $RES/$(tag_of "$s").log" >&2; fi
    done
  fi
  running=()
  [ "${#newrun[@]}" -gt 0 ] && running=("${newrun[@]}")
done
echo "[pool] all systems complete"
