#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --job-name=flashattn_r
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/rtang906/AgentOCR/logs_tr/flash_attn_build_%j.log
#SBATCH --signal=B:TERM@120

set -euo pipefail

ROOT_DIR=/hpc2hdd/home/rtang906/AgentOCR
SRC_DIR="${ROOT_DIR}/flash-attention"
CONDA_ENV_DIR=/hpc2hdd/home/rtang906/.conda/envs/AgentOCR
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
SUCCESS_MARKER="${SRC_DIR}/.flash_attn_build_success"
STATE_FILE="${SRC_DIR}/.flash_attn_build_state.env"
ATTEMPT="${FLASH_ATTN_ATTEMPT:-1}"
MAX_ATTEMPTS="${FLASH_ATTN_MAX_ATTEMPTS:-20}"
RUN_LOG="${ROOT_DIR}/logs_tr/flash_attn_resume_attempt_${ATTEMPT}.log"

submit_next_attempt() {
    local next_attempt=$((ATTEMPT + 1))
    if [[ -f "${SUCCESS_MARKER}" ]]; then
        echo "[flash-attn] success marker exists, not resubmitting"
        return 0
    fi
    if (( next_attempt > MAX_ATTEMPTS )); then
        echo "[flash-attn] reached max attempts (${MAX_ATTEMPTS}), stopping"
        return 0
    fi
    echo "[flash-attn] resubmitting attempt ${next_attempt}/${MAX_ATTEMPTS}"
    sbatch --export=ALL,FLASH_ATTN_ATTEMPT="${next_attempt}",FLASH_ATTN_MAX_ATTEMPTS="${MAX_ATTEMPTS}" "$0"
}

handle_term() {
    echo "[flash-attn] received TERM, preserving build dir and resubmitting"
    submit_next_attempt
    exit 0
}

trap handle_term TERM

mkdir -p "${ROOT_DIR}/logs_tr"
touch "${RUN_LOG}"
exec >>"${RUN_LOG}" 2>&1
echo "[flash-attn] bootstrap $(date -Iseconds)"

cat > "${STATE_FILE}" <<EOF
FLASH_ATTN_ATTEMPT=${ATTEMPT}
FLASH_ATTN_MAX_ATTEMPTS=${MAX_ATTEMPTS}
FLASH_ATTN_LAST_START=$(date -Iseconds)
EOF

export PYTHONNOUSERSITE=1
export CUDA_HOME=/hpc2ssd/softwares/cuda/cuda-12.6
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

cd "${SRC_DIR}"

echo "[flash-attn] attempt ${ATTEMPT}/${MAX_ATTEMPTS}"
echo "[flash-attn] host $(hostname)"
echo "[flash-attn] cwd ${PWD}"
echo "[flash-attn] CUDA_HOME=${CUDA_HOME}"
echo "[flash-attn] FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}"
echo "[flash-attn] MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")
PY

if [[ -f "${SUCCESS_MARKER}" ]]; then
    echo "[flash-attn] already built successfully, exiting"
    exit 0
fi

set +e
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps -v .
build_status=$?
set -e

if (( build_status == 0 )); then
    echo "[flash-attn] pip install succeeded, verifying import from /tmp"
    pushd /tmp >/dev/null
    set +e
    "${PYTHON_BIN}" - <<'PY'
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_func
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
print("flash_attn_func_ok", callable(flash_attn_func))
PY
    verify_status=$?
    set -e
    popd >/dev/null
    if (( verify_status == 0 )); then
        touch "${SUCCESS_MARKER}"
        echo "[flash-attn] build and import verification succeeded"
        exit 0
    fi
    echo "[flash-attn] import verification failed"
    build_status=1
fi

echo "[flash-attn] build status=${build_status}; build dir preserved for resume"
submit_next_attempt
exit "${build_status}"
