#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for cuda_dir in /usr/local/cuda /usr/local/cuda-12.4 /usr/local/cuda-12; do
    if [[ -x "${cuda_dir}/bin/nvcc" ]]; then
      export CUDA_HOME="${cuda_dir}"
      break
    fi
  done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

export MAX_JOBS="${MAX_JOBS:-4}"

python -m pip install --upgrade pip wheel packaging ninja
python -m pip install setuptools==68.2.2
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --constraint "${REPO_ROOT}/requirements/constraints-cu124.txt" \
  -r "${REPO_ROOT}/requirements.txt"

cp "${REPO_ROOT}/requirements/mamba_simple.py" \
  "${REPO_ROOT}/requirements/Mamba/mamba/mamba_ssm/modules/mamba_simple.py"

pushd "${REPO_ROOT}/requirements/Mamba/causal-conv1d" >/dev/null
CAUSAL_CONV1D_FORCE_BUILD=TRUE python setup.py install
popd >/dev/null

pushd "${REPO_ROOT}/requirements/Mamba/mamba" >/dev/null
MAMBA_FORCE_BUILD=TRUE python setup.py install
popd >/dev/null

python "${REPO_ROOT}/scripts/check_tdr_mamba.py"
