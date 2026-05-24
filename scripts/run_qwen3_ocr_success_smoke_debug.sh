#!/usr/bin/env bash
set -euo pipefail

cd /hpc2hdd/home/rtang906/AgentOCR

CUDA_ROOT_CANDIDATE="/hpc2ssd/softwares/cuda/cuda-12.6"
if [ -x "${CUDA_ROOT_CANDIDATE}/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT_CANDIDATE}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ -x /hpc2ssd/softwares/anaconda3/bin/conda ]; then
    # shellcheck disable=SC1091
    set +u
    eval "$(/hpc2ssd/softwares/anaconda3/bin/conda shell.bash hook)"
    conda activate AgentOCR
    set -u
fi

export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

output_path="${1:?usage: run_qwen3_ocr_success_smoke_debug.sh <output_json> <log_path>}"
log_path="${2:?usage: run_qwen3_ocr_success_smoke_debug.sh <output_json> <log_path>}"

python -u tools/qwen3_alfworld_ocr_success_smoke.py \
    --device cuda:0 \
    --output "${output_path}" \
    2>&1 | tee "${log_path}"
