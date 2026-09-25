#!/bin/bash
# One virtual environment per external code, under $APPD_ENVS (default: $SCRATCH/appD_envs, or
# $HOME/appD_envs). Run on a machine with the target CUDA stack.
#
#   bash experiments/exp9_frameworks/setup_envs.sh <pyscf|mess|d4ft|dqc>
set -u
ROOT="${APPD_ENVS:-${SCRATCH:-$HOME}/appD_envs}"
mkdir -p "$ROOT" && cd "$ROOT"
name="$1"
case "$name" in
  pyscf)
    uv venv -q --python 3.11 pyscf
    VIRTUAL_ENV=$ROOT/pyscf uv pip install -q pyscf ;;
  mess)
    uv venv -q --python 3.11 mess
    VIRTUAL_ENV=$ROOT/mess uv pip install -q "jax[cuda12]" "git+https://github.com/graphcore-research/mess.git" tabulate ;;
  d4ft)
    # D4FT installs editable from a clone, and needs its pinned JAX, Haiku, SciPy and cuDNN.
    [ -d "$ROOT/d4ft_src" ] || git clone -q https://github.com/sail-sg/d4ft.git "$ROOT/d4ft_src"
    uv venv -q --python 3.9 --clear d4ft
    VIRTUAL_ENV=$ROOT/d4ft uv pip install -q -e "$ROOT/d4ft_src" "jax[cuda12_pip]==0.4.23" "dm-haiku==0.0.12" "scipy<1.13" "nvidia-cudnn-cu12==8.9.7.29" \
      --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html ;;
  dqc)
    # DQC publishes wheels up to Python 3.9.
    uv venv -q --python 3.9 --clear dqc
    VIRTUAL_ENV=$ROOT/dqc uv pip install -q torch --index-url https://download.pytorch.org/whl/cu121
    VIRTUAL_ENV=$ROOT/dqc uv pip install -q dqc ;;
esac
echo "$name exit=$?"
"$ROOT/$name/bin/python" -c "import sys; print(sys.version)"
