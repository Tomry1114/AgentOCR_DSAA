#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs_qwen3

ts="${1:-$(date +%Y%m%d_%H%M%S)}"
node_ip="$(hostname -I | awk '{print $1}')"

launch_job() {
    local session_name="$1"
    local master_port="$2"
    local gpu_set="$3"
    local cpu_set="$4"
    local train_script="$5"
    local prepare_root="$6"
    local log_file="logs_qwen3/${session_name}.log"

    tmux kill-session -t "${session_name}" >/dev/null 2>&1 || true
    mkdir -p "${prepare_root}"

    local cmd
    cmd="cd /hpc2hdd/home/rtang906/AgentOCR && "
    cmd+="export MASTER_ADDR=${node_ip} MASTER_PORT=${master_port} "
    cmd+="CUDA_VISIBLE_DEVICES=${gpu_set} "
    cmd+="TRAINER_N_GPUS_PER_NODE=4 "
    cmd+="RAY_INIT_NUM_CPUS=32 OCR_MAX_WORKERS=16 NUM_CPUS_PER_ENV_WORKER=0.1 AGENTOCR_TASKRUNNER_NUM_CPUS=1 "
    cmd+="PREPARE_LOCAL_DIR=${prepare_root} "
    cmd+="TRAIN_FILE=${prepare_root}/visual/train.parquet "
    cmd+="VAL_FILE=${prepare_root}/visual/test.parquet "
    cmd+="TRAINER_EXPERIMENT_NAME=${session_name} "
    cmd+="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false "
    cmd+="ACTOR_ENABLE_ACTIVATION_OFFLOAD=False ACTOR_FSDP_PARAM_OFFLOAD=False ACTOR_FSDP_OPTIMIZER_OFFLOAD=False ACTOR_FSDP_OFFLOAD_POLICY=False; "
    cmd+="exec taskset -c ${cpu_set} bash ${train_script} > ${log_file} 2>&1"

    tmux new-session -d -s "${session_name}" "${cmd}"
    printf 'launched %s log=%s\n' "${session_name}" "${log_file}"
}

launch_job \
    "q3vl4b_ocr4x_${ts}" \
    "29611" \
    "0,1,2,3" \
    "0-31" \
    "./train_alfworld_qwen3vl4b_ocr_4xa800.sh" \
    "/hpc2hdd/home/rtang906/data/verl-agent-q3vl4b-ocr4x-${ts}"

launch_job \
    "q3vl4b_ocrtrue4x_${ts}" \
    "29621" \
    "4,5,6,7" \
    "32-63" \
    "./train_alfworld_qwen3vl4b_ocr_slotaware_4xa800.sh" \
    "/hpc2hdd/home/rtang906/data/verl-agent-q3vl4b-ocrtrue4x-${ts}"
