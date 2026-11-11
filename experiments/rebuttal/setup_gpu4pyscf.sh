#!/bin/bash
# The GPU4PySCF environment for experiment 4 (no JAX). Build it on a compute node (CPU only: the
# GPU is first touched by the smoke test, gpu4pyscf_df.py on two small systems):
#
#   bash experiments/rebuttal/setup_gpu4pyscf.sh [ENV]      # default $SCRATCH/rebuttal/env-gpu4pyscf
set -eu
ENV="${1:-${SCRATCH:-$HOME}/rebuttal/env-gpu4pyscf}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME}/rebuttal/uv_cache}"
"$UV" venv -q --python 3.11 --clear "$ENV"
# cupy[ctk] brings the CUDA libraries (cuBLAS, cuSOLVER, ...) as wheels: no module load needed.
VIRTUAL_ENV="$ENV" "$UV" pip install -q gpu4pyscf-cuda12x "cupy-cuda12x[ctk]" cutensor-cu12
# gpu4pyscf loads nvidia/cusolver/lib/libcusolver.so (unversioned) when no system cuSOLVER is found;
# the wheel ships only the versioned soname.
SP="$("$ENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
ln -sf libcusolver.so.11 "$SP/nvidia/cusolver/lib/libcusolver.so"
"$ENV/bin/python" -c "import pyscf, cupy; from importlib.metadata import version as v; print('pyscf', pyscf.__version__, 'gpu4pyscf', v('gpu4pyscf-cuda12x'), 'cupy', cupy.__version__)"
