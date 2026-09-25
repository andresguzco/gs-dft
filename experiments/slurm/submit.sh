#!/usr/bin/env bash
# Submit any command as a SLURM batch job, run from the repository root inside `uv run`.
#
#   experiments/slurm/submit.sh [--gpus N] [--cpus N] [--mem M] [--time HH:MM:SS] [--name NAME]
#       [--partition P] [--gpu-type T] [--array SPEC] [--nodes N] [--out DIR] [--dry-run] -- <command>
#
# Site settings (account, partition, GPU flag) come from the environment; see "site profile" below.
set -euo pipefail

GPUS=1; CPUS=""; MEM=""; TIME="3:00:00"; NAME=""; ARRAY=""; OUT="results"; DRY=0
PART=""; DEP=""; PARSABLE=0; NODES=1; GPU_TYPE_OPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)       GPUS="$2";    shift 2 ;;
    --nodes)      NODES="$2";   shift 2 ;;
    --gpu-type)   GPU_TYPE_OPT="$2"; shift 2 ;;   # GPU model, e.g. a100, h100
    --cpus)       CPUS="$2";    shift 2 ;;
    --mem)        MEM="$2";     shift 2 ;;
    --time)       TIME="$2";    shift 2 ;;
    --name)       NAME="$2";    shift 2 ;;
    --array)      ARRAY="$2";   shift 2 ;;
    --out)        OUT="$2";     shift 2 ;;
    --partition)  PART="$2";    shift 2 ;;
    --dependency) DEP="$2";     shift 2 ;;
    --parsable)   PARSABLE=1;   shift ;;
    --dry-run)    DRY=1;        shift ;;
    --)           shift; break ;;
    *) echo "submit.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "submit.sh: no command given (put it after --)" >&2; exit 2; }

# --- site profile ------------------------------------------------------------
# Sites differ in how they name accounts, partitions and GPUs, so they are configured by
# environment rather than hardcoded. Set these once in your shell profile:
#
#   SLURM_ACCOUNT    charge account, if your site requires one
#   SLURM_PARTITION  default partition, overridden by --partition
#   SLURM_GPU_TYPE   GPU model to request, e.g. a100, h100 (default: any)
#   SLURM_GPU_FLAG   gres (default) or gpus-per-node, whichever your site accepts
#   SLURM_PRE        commands to run before the payload, e.g. "module load proxy"
ACCOUNT_LINE="${SLURM_ACCOUNT:+#SBATCH --account=${SLURM_ACCOUNT}}"
PART="${PART:-${SLURM_PARTITION:-}}"
PART_LINE="${PART:+#SBATCH --partition=${PART}}"
GPU_TYPE="${GPU_TYPE_OPT:-${SLURM_GPU_TYPE:-}}"
if [[ "$GPUS" -gt 0 ]]; then
  spec="${GPU_TYPE:+${GPU_TYPE}:}${GPUS}"
  case "${SLURM_GPU_FLAG:-gres}" in
    gpus-per-node) GPU_LINE="#SBATCH --gpus-per-node=${spec}" ;;
    *)             GPU_LINE="#SBATCH --gres=gpu:${spec}" ;;
  esac
else
  GPU_LINE=""
fi
NET_LINE="${SLURM_PRE:-}"
# Node-local when the scheduler gives a local disk: XLA's autotune files fail on some networked
# filesystems, which can kill a job at its first jit.
JAX_CACHE='${SLURM_TMPDIR:-${SCRATCH:-$PWD}}/.jax_cache'
CPUS="${CPUS:-8}"; MEM="${MEM:-0}"       # mem=0 requests all the memory of the allocated node
if ! grep -qs "api.wandb.ai" "${HOME}/.netrc" && [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "submit.sh: no WandB credential (\$HOME/.netrc has no api.wandb.ai, WANDB_API_KEY unset)." >&2
  echo "submit.sh:   Runs default to wandb_mode=online and will log NOTHING. Fix: \`wandb login\` on" >&2
  echo "submit.sh:   this host, or pass wandb_mode=disabled if that is what you want." >&2
fi

if [[ "$NODES" -gt 1 ]]; then
  [[ "$GPUS" -gt 0 ]] || { echo "submit.sh: --nodes>1 needs GPUs" >&2; exit 2; }
  TASKS_PER_NODE="$GPUS"
  CPUS=$(( ${CPUS:-16} / GPUS )); [[ "$CPUS" -lt 1 ]] && CPUS=1
  LAUNCH="srun --ntasks=$((NODES * GPUS)) "
else
  TASKS_PER_NODE=1
  LAUNCH=""
fi

NAME="${NAME:-dftax_$(date +%H%M%S)}"
[[ -n "$ARRAY" ]] && ARRAY_LINE="#SBATCH --array=${ARRAY}" || ARRAY_LINE=""
DEP_LINE="${DEP:+#SBATCH --dependency=${DEP}}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# An absolute --out is used as is and a relative one is taken from the repository root, for both
# mkdir and #SBATCH --output (SLURM would resolve a relative --output against the submitter's cwd).
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
mkdir -p "$OUT"

# Shell-quote the payload so it survives the heredoc intact: `$*` splats unquoted, so
# `-- bash -c "echo a b"` would arrive as four words. printf %q keeps it one argument.
CMD="$(printf '%q ' "$@")"

SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}
${ACCOUNT_LINE}
${PART_LINE}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${TASKS_PER_NODE}
${GPU_LINE}
${ARRAY_LINE}
${DEP_LINE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --requeue
#SBATCH --output=${OUT}/%x_%j.log
set -uo pipefail
${NET_LINE}
cd "${REPO}"
export PYTHONUNBUFFERED=1 PYTHONPATH="\$PWD" TF_CPP_MIN_LOG_LEVEL=3
export JAX_COMPILATION_CACHE_DIR="${JAX_CACHE}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# Resolve uv on the compute node: a batch job does not inherit an interactive PATH.
UV="\$(command -v uv || echo "\$HOME/.local/bin/uv")"
[ -x "\$UV" ] || { echo "submit.sh: no uv on PATH and none at \$HOME/.local/bin/uv" >&2; exit 127; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo "=== \$(date -Is) :: ${CMD}==="
"\$UV" sync --frozen --inexact          # once, before the ranks start (concurrent syncs corrupt .venv); --inexact keeps the checkout's extras
${LAUNCH}"\$UV" run --frozen --no-sync ${CMD}
echo "=== DONE exit=\$? \$(date -Is) ==="
EOF
)

if [[ "$DRY" == "1" ]]; then echo "$SCRIPT"; exit 0; fi
# --parsable prints the bare job id on stdout, so a caller can chain:
#   prev=""; for i in 1 2 3 4; do
#     prev=$(submit.sh --parsable ${prev:+--dependency afterany:$prev} ... -- <cmd>); done
if [[ "$PARSABLE" == "1" ]]; then echo "$SCRIPT" | sbatch --parsable
else                                echo "$SCRIPT" | sbatch; fi
