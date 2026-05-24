#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs_qwen3

timestamp="${1:-$(date +%Y%m%d_%H%M%S)}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

launch_job() {
    local job_name="$1"
    local gpu_set="$2"
    local cpu_set="$3"
    local train_script="$4"
    local prepare_root="$5"

    local log_file="logs_qwen3/${job_name}_${timestamp}.log"
    local pid_file="logs_qwen3/${job_name}_${timestamp}.pid"

    mkdir -p "${prepare_root}"

    (
        export CUDA_VISIBLE_DEVICES="${gpu_set}"
        export TRAINER_N_GPUS_PER_NODE=4
        export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-32}"
        export OCR_MAX_WORKERS="${OCR_MAX_WORKERS:-16}"
        export NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
        export AGENTOCR_TASKRUNNER_NUM_CPUS="${AGENTOCR_TASKRUNNER_NUM_CPUS:-1}"
        export PREPARE_LOCAL_DIR="${prepare_root}"
        export TRAIN_FILE="${prepare_root}/visual/train.parquet"
        export VAL_FILE="${prepare_root}/visual/test.parquet"
        export ACTOR_ENABLE_ACTIVATION_OFFLOAD=False
        export ACTOR_FSDP_PARAM_OFFLOAD=False
        export ACTOR_FSDP_OPTIMIZER_OFFLOAD=False
        export ACTOR_FSDP_OFFLOAD_POLICY=False
        export TRAINER_EXPERIMENT_NAME="${job_name}"
        exec taskset -c "${cpu_set}" bash "${train_script}"
    ) >"${log_file}" 2>&1 &

    local pid=$!
    printf '%s\n' "${pid}" >"${pid_file}"
    printf '%s\n' "launched ${job_name} pid=${pid} log=${log_file}"
}

launch_job \
    "q3vl4b_ocr4x_isolated" \
    "0,1,2,3" \
    "0-31" \
    "./train_alfworld_qwen3vl4b_ocr_4xa800.sh" \
    "${HOME}/data/verl-agent-q3vl4b-ocr4x-${timestamp}"

launch_job \
    "q3vl4b_ocrtrue4x_isolated" \
    "4,5,6,7" \
    "32-63" \
    "./train_alfworld_qwen3vl4b_ocr_slotaware_4xa800.sh" \
    "${HOME}/data/verl-agent-q3vl4b-ocrtrue4x-${timestamp}"

printf '%s\n' "timestamp=${timestamp}"
