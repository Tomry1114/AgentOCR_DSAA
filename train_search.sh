#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "$0")"

CUDA_ROOT_CANDIDATE="/hpc2ssd/softwares/cuda/cuda-12.6"
CONDA_ROOT="/hpc2ssd/softwares/anaconda3"

if [ -f /etc/profile ]; then
    set +u
    source /etc/profile >/dev/null 2>&1 || true
    set -u
fi
if [ -f /etc/profile.d/modules.sh ]; then
    set +u
    source /etc/profile.d/modules.sh >/dev/null 2>&1 || true
    set -u
fi

if command -v module >/dev/null 2>&1; then
    module load cuda/12.6 >/dev/null 2>&1 || true
fi

if [ -x "${CUDA_ROOT_CANDIDATE}/bin/nvcc" ]; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT_CANDIDATE}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v nvcc >/dev/null 2>&1; then
    cuda_bin_dir="$(dirname "$(command -v nvcc)")"
    export CUDA_HOME="${CUDA_HOME:-$(cd "${cuda_bin_dir}/.." && pwd)}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

slurm_gpu_count=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    slurm_gpu_count="$(
        env -u LD_LIBRARY_PATH scontrol show job "${SLURM_JOB_ID}" 2>/dev/null \
            | sed -n 's/.*gres\/gpu=\([0-9]\+\).*/\1/p' \
            | head -n 1 \
            || true
    )"
fi

search_variant="${SEARCH_VARIANT:-idea_adapter_only}"
case "${search_variant}" in
    idea|idea_full|idea_no_trust|idea_followanchor_notrust|idea_targeted_notrust)
        search_config_stem="qwen3vl4b_search_idea"
        ;;
    ocr|noidea|idea_adapter_only|idea_force_search_only|idea_strict_prompt_only|idea_trust_policy_only)
        search_config_stem="qwen3vl4b_search_ocr"
        ;;
    *)
        echo "Unsupported SEARCH_VARIANT=${search_variant}. Expected one of: idea, idea_full, idea_no_trust, idea_followanchor_notrust, idea_targeted_notrust, ocr, noidea, idea_adapter_only, idea_force_search_only, idea_strict_prompt_only, idea_trust_policy_only" >&2
        exit 1
        ;;
esac

default_search_config="$(pwd)/configs/${search_config_stem}_4xa800.env"
if [ "${slurm_gpu_count:-0}" -ge 8 ] 2>/dev/null && [ -f "$(pwd)/configs/${search_config_stem}_8xa800.env" ]; then
    default_search_config="$(pwd)/configs/${search_config_stem}_8xa800.env"
fi

config_file="${QWEN3_SEARCH_CONFIG:-${QWEN3_SEARCH_OCR_CONFIG:-$default_search_config}}"
if [ -f "$config_file" ]; then
    # shellcheck disable=SC1090
    source "$config_file"
fi

set_variant_env() {
    if [ "${AGENTOCR_RESPECT_PRESET_VARIANT_ENVS:-0}" = "1" ] && [ "${!1+x}" = x ]; then
        export "$1=${!1}"
    else
        export "$1=$2"
    fi
}

case "${search_variant}" in
    ocr|noidea)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 0
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        ;;
    idea|idea_full)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 1
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE True
        set_variant_env OCR_TRUST_POLICY_QUERY_CONDITIONED True
        set_variant_env OCR_TRUST_POLICY_STATE_AWARE False
        set_variant_env OCR_TRUST_POLICY_CONTEXT_MODE query
        set_variant_env OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT 0.30
        set_variant_env OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT 50
        set_variant_env OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY True
        set_variant_env OCR_TRUST_POLICY_USE_PROMPT_SUMMARY False
        set_variant_env OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS True
        set_variant_env OCR_TRUST_POLICY_MIN_COMPACTION_LINES 2
        set_variant_env OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION 1
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL 1
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES 2
        ;;
    idea_no_trust)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 1
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        set_variant_env OCR_TRUST_POLICY_QUERY_CONDITIONED False
        set_variant_env OCR_TRUST_POLICY_STATE_AWARE False
        set_variant_env OCR_TRUST_POLICY_CONTEXT_MODE auto
        set_variant_env OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT 0.30
        set_variant_env OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT 50
        set_variant_env OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY False
        set_variant_env OCR_TRUST_POLICY_USE_PROMPT_SUMMARY False
        set_variant_env OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS False
        set_variant_env OCR_TRUST_POLICY_MIN_COMPACTION_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION 0
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL 4
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES 8
        ;;
    idea_followanchor_notrust)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 1
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        set_variant_env OCR_TRUST_POLICY_QUERY_CONDITIONED False
        set_variant_env OCR_TRUST_POLICY_STATE_AWARE False
        set_variant_env OCR_TRUST_POLICY_CONTEXT_MODE auto
        set_variant_env OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT 0.30
        set_variant_env OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT 50
        set_variant_env OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY False
        set_variant_env OCR_TRUST_POLICY_USE_PROMPT_SUMMARY False
        set_variant_env OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS False
        set_variant_env OCR_TRUST_POLICY_MIN_COMPACTION_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION 0
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL 4
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES 8
        ;;
    idea_targeted_notrust)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 1
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 1
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        set_variant_env OCR_TRUST_POLICY_QUERY_CONDITIONED False
        set_variant_env OCR_TRUST_POLICY_STATE_AWARE False
        set_variant_env OCR_TRUST_POLICY_CONTEXT_MODE auto
        set_variant_env OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT 0.30
        set_variant_env OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT 50
        set_variant_env OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY False
        set_variant_env OCR_TRUST_POLICY_USE_PROMPT_SUMMARY False
        set_variant_env OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS False
        set_variant_env OCR_TRUST_POLICY_MIN_COMPACTION_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION 0
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL 4
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES 8
        ;;
    idea_adapter_only)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 1
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        ;;
    idea_force_search_only)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 0
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        ;;
    idea_strict_prompt_only)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 0
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 1
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE False
        ;;
    idea_trust_policy_only)
        set_variant_env AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH 0
        set_variant_env AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE 0
        set_variant_env AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD 0
        set_variant_env OCR_TRUST_POLICY_ENABLE True
        set_variant_env OCR_TRUST_POLICY_QUERY_CONDITIONED True
        set_variant_env OCR_TRUST_POLICY_STATE_AWARE False
        set_variant_env OCR_TRUST_POLICY_CONTEXT_MODE query
        set_variant_env OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT 0.30
        set_variant_env OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT 50
        set_variant_env OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY True
        set_variant_env OCR_TRUST_POLICY_USE_PROMPT_SUMMARY False
        set_variant_env OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS True
        set_variant_env OCR_TRUST_POLICY_MIN_COMPACTION_LINES 2
        set_variant_env OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES 8
        set_variant_env OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION 1
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL 1
        set_variant_env OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES 2
        ;;
esac

if [ -z "${AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP+x}" ] && [ -n "${AGENTOCR_SEARCH_NO_HISTORY_FORCE_SEARCH+x}" ]; then
    export AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP="${AGENTOCR_SEARCH_NO_HISTORY_FORCE_SEARCH}"
fi
if [ -z "${AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT+x}" ] && [ -n "${AGENTOCR_SEARCH_STRICT_ENTITY_GROUNDING+x}" ]; then
    export AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT="${AGENTOCR_SEARCH_STRICT_ENTITY_GROUNDING}"
fi
export AGENTOCR_SEARCH_NO_HISTORY_FORCE_SEARCH="${AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP:-0}"
export AGENTOCR_SEARCH_STRICT_ENTITY_GROUNDING="${AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT:-0}"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export RUST_BACKTRACE=1
export _FAISS_WHEEL_DISABLE_CUDA_PRELOAD="${_FAISS_WHEEL_DISABLE_CUDA_PRELOAD:-1}"
ulimit -n 65535

for ray_var in RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV RAY_JOB_CONFIG_JSON; do
    if [ -n "${!ray_var:-}" ]; then
        echo "Unsetting inherited ${ray_var}=${!ray_var}"
    fi
    unset "${ray_var}"
done

if [ -x "${CONDA_ROOT}/bin/conda" ]; then
    set +u
    eval "$(${CONDA_ROOT}/bin/conda shell.bash hook)"
    conda activate AgentOCR
    set -u
fi

launcher_script="$(pwd)/tools/run_module_sanitized.py"
python_bin="${PYTHON_BIN:-/hpc2hdd/home/rtang906/AgentOCR/tools/python_glibc234}"
seed="${SEED:-0}"
model_path="${MODEL_PATH:-/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct}"
project_name="${TRAINER_PROJECT_NAME:-AgentOCR_search}"
experiment_name="${TRAINER_EXPERIMENT_NAME:-search_idea_qwen3vl4b_s${seed}}"

search_processed_dir="${SEARCH_PROCESSED_DIR:-/hpc2hdd/home/rtang906/data/searchR1_processed_direct}"
search_index_dir="${SEARCH_INDEX_DIR:-/hpc2hdd/home/rtang906/data/searchR1}"
train_file="${TRAIN_FILE:-${search_processed_dir}/train.parquet}"
val_file="${VAL_FILE:-${search_processed_dir}/test.parquet}"
index_file="${SEARCH_INDEX_FILE:-${search_index_dir}/e5_Flat.index}"
corpus_file="${SEARCH_CORPUS_FILE:-${search_index_dir}/wiki-18.jsonl}"

search_port="${SEARCH_PORT:-8000}"
search_host="${SEARCH_HOST:-127.0.0.1}"
search_url="${SEARCH_URL:-http://${search_host}:${search_port}/retrieve}"
search_request_timeout="${SEARCH_REQUEST_TIMEOUT:-300}"
search_max_concurrent_requests="${SEARCH_MAX_CONCURRENT_REQUESTS:-4}"
search_env_max_workers="${SEARCH_ENV_MAX_WORKERS:-}"
search_batch_request_size="${SEARCH_BATCH_REQUEST_SIZE:-}"
search_server_gpu="${SEARCH_SERVER_CUDA_VISIBLE_DEVICES:-7}"
search_server_log="${SEARCH_SERVER_LOG:-$(pwd)/logs_debug/search_ocr_retriever_${search_port}.log}"
search_server_pid_file="${SEARCH_SERVER_PID_FILE:-$(pwd)/logs_debug/search_ocr_retriever_${search_port}.pid}"
search_server_start_timeout="${SEARCH_SERVER_START_TIMEOUT:-900}"
search_server_launch_mode="${SEARCH_SERVER_LAUNCH_MODE:-auto}"
search_server_use_faiss_gpu="${SEARCH_SERVER_USE_FAISS_GPU:-True}"
search_server_device="${SEARCH_SERVER_DEVICE:-auto}"

train_data_size="${TRAIN_DATA_SIZE:-14}"
val_data_size="${VAL_DATA_SIZE:-128}"
group_size="${GROUP_SIZE:-8}"
data_max_prompt_length="${DATA_MAX_PROMPT_LENGTH:-4096}"
data_max_response_length="${DATA_MAX_RESPONSE_LENGTH:-512}"
env_max_steps="${ENV_MAX_STEPS:-4}"
env_history_length="${ENV_HISTORY_LENGTH:-4}"
actor_lr="${ACTOR_LR:-1e-6}"
actor_ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE:-112}"
actor_ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
rollout_log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
rollout_enforce_eager="${ROLLOUT_ENFORCE_EAGER:-False}"
rollout_free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE:-False}"
vllm_skip_mm_profiling="${VLLM_SKIP_MM_PROFILING:-True}"
actor_fsdp_param_offload="${ACTOR_FSDP_PARAM_OFFLOAD:-False}"
actor_fsdp_optimizer_offload="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-False}"
actor_fsdp_offload_policy="${ACTOR_FSDP_OFFLOAD_POLICY:-False}"
actor_strategy="${ACTOR_STRATEGY:-fsdp2}"
actor_use_torch_compile="${ACTOR_USE_TORCH_COMPILE:-False}"
actor_enable_activation_offload="${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-False}"
trainer_nnodes="${TRAINER_NNODES:-1}"
trainer_save_freq="${TRAINER_SAVE_FREQ:-10}"
trainer_test_freq="${TRAINER_TEST_FREQ:-150}"
trainer_total_training_steps="${TRAINER_TOTAL_TRAINING_STEPS:-150}"
trainer_val_before_train="${TRAINER_VAL_BEFORE_TRAIN:-False}"
invalid_action_penalty_coef="${INVALID_ACTION_PENALTY_COEF:-0.01}"
data_shuffle="${DATA_SHUFFLE:-True}"

ocr_use_parallel="${OCR_USE_PARALLEL:-True}"
ocr_max_workers="${OCR_MAX_WORKERS:-32}"
ocr_font_size="${OCR_FONT_SIZE:-${AGENTOCR_QWEN3_OCR_RENDER_FONT_SIZE:-12}}"
ocr_max_width="${OCR_MAX_WIDTH:-${AGENTOCR_QWEN3_OCR_RENDER_MAX_WIDTH:-392}}"
compression_reward_coef="${OCR_COMPRESSION_REWARD_COEF:-0.01}"
compression_failure_penalty_coef="${OCR_COMPRESSION_FAILURE_PENALTY_COEF:-0.0}"
compression_reward_every_n_steps="${OCR_COMPRESSION_REWARD_EVERY_N_STEPS:-8}"

ocr_trust_policy_enable="${OCR_TRUST_POLICY_ENABLE:-False}"
ocr_trust_policy_query_conditioned="${OCR_TRUST_POLICY_QUERY_CONDITIONED:-False}"
ocr_trust_policy_state_aware="${OCR_TRUST_POLICY_STATE_AWARE:-False}"
ocr_trust_policy_context_mode="${OCR_TRUST_POLICY_CONTEXT_MODE:-auto}"
ocr_trust_policy_query_relevance_weight="${OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT:-0.30}"
ocr_trust_policy_context_budget_percent="${OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT:-50}"
ocr_trust_policy_use_compressed_history="${OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY:-False}"
ocr_trust_policy_use_prompt_summary="${OCR_TRUST_POLICY_USE_PROMPT_SUMMARY:-False}"
ocr_trust_policy_collect_diagnostics="${OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS:-False}"
ocr_trust_policy_min_compaction_lines="${OCR_TRUST_POLICY_MIN_COMPACTION_LINES:-8}"
ocr_trust_policy_min_prompt_summary_lines="${OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES:-8}"
ocr_trust_policy_min_history_steps_for_compaction="${OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION:-0}"
ocr_trust_policy_feedback_update_interval="${OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL:-4}"
ocr_trust_policy_feedback_min_history_lines="${OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES:-8}"

if [ -n "${TRAINER_N_GPUS_PER_NODE_OVERRIDE:-}" ]; then
    trainer_n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE_OVERRIDE}"
elif [ -n "${TRAINER_N_GPUS_PER_NODE:-}" ]; then
    trainer_n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}"
elif [ -n "${slurm_gpu_count}" ]; then
    trainer_n_gpus_per_node="${slurm_gpu_count}"
else
    trainer_n_gpus_per_node="4"
fi

if [ -n "${RAY_INIT_NUM_CPUS_OVERRIDE:-}" ]; then
    ray_init_num_cpus="${RAY_INIT_NUM_CPUS_OVERRIDE}"
elif [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    ray_init_num_cpus="${SLURM_CPUS_PER_TASK}"
else
    ray_init_num_cpus="${RAY_INIT_NUM_CPUS:-64}"
fi

###############
# Highlight configs: use environment variable to avoid Hydra parsing issues with < > characters
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
###############

csv_list_contains() {
    local needle="$1"
    local csv="$2"
    local item=""
    local old_ifs="$IFS"
    IFS=','
    read -r -a items <<<"$csv"
    IFS="$old_ifs"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

is_non_negative_integer() {
    case "$1" in
        ''|*[!0-9]*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

search_server_requires_gpu=1
if ! is_truthy "$search_server_use_faiss_gpu"; then
    case "${search_server_device}" in
        cpu|CPU)
            search_server_requires_gpu=0
            ;;
    esac
fi

resolve_search_server_launch_mode() {
    if [ "$search_server_launch_mode" = "local" ] || [ "$search_server_launch_mode" = "slurm_step" ]; then
        echo "$search_server_launch_mode"
        return 0
    fi
    if [ "$search_server_launch_mode" != "auto" ]; then
        echo "Unsupported SEARCH_SERVER_LAUNCH_MODE=${search_server_launch_mode}" >&2
        return 1
    fi

    if [ "$search_server_requires_gpu" -eq 0 ]; then
        echo "local"
        return 0
    fi

    if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ! csv_list_contains "$search_server_gpu" "${CUDA_VISIBLE_DEVICES}"; then
        echo "slurm_step"
        return 0
    fi

    echo "local"
}

launch_search_server() {
    local launch_mode="$1"
    local workdir_q=""
    local inner_cmd=""
    local search_server_cuda_visible_devices="$search_server_gpu"
    local -a retriever_cmd=(
        "$python_bin"
        examples/search/retriever/retrieval_server.py
        --index_path "$index_file"
        --corpus_path "$corpus_file"
        --topk 3
        --retriever_name e5
        --retriever_model intfloat/e5-base-v2
        --device "$search_server_device"
        --port "$search_port"
    )

    if is_truthy "$search_server_use_faiss_gpu"; then
        retriever_cmd+=(--faiss_gpu)
    fi
    if [ "$search_server_requires_gpu" -eq 0 ]; then
        search_server_cuda_visible_devices=""
    fi

    rm -f "$search_server_pid_file"
    if [ "$launch_mode" = "slurm_step" ]; then
        if ! command -v srun >/dev/null 2>&1; then
            echo "srun is required for SEARCH_SERVER_LAUNCH_MODE=slurm_step" >&2
            return 1
        fi
        printf -v workdir_q '%q' "$(pwd)"
        printf -v inner_cmd '%q ' \
            env \
            "CUDA_VISIBLE_DEVICES=${search_server_cuda_visible_devices}" \
            "_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=${_FAISS_WHEEL_DISABLE_CUDA_PRELOAD}" \
            "${retriever_cmd[@]}"
        nohup srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=1 --ntasks=1 \
            bash -lc "cd ${workdir_q} && exec ${inner_cmd}" \
            >"$search_server_log" 2>&1 &
    else
        nohup env \
            "CUDA_VISIBLE_DEVICES=${search_server_cuda_visible_devices}" \
            "_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=${_FAISS_WHEEL_DISABLE_CUDA_PRELOAD}" \
            "${retriever_cmd[@]}" \
            >"$search_server_log" 2>&1 &
    fi
    echo $! > "$search_server_pid_file"
}

mkdir -p "$(dirname "$search_server_log")"
mkdir -p "$search_processed_dir" "$search_index_dir"

if ! is_truthy "${DRY_RUN:-0}"; then
    if [ ! -f "$train_file" ] || [ ! -f "$val_file" ]; then
        echo "Preparing Search-R1 parquet into ${search_processed_dir}"
        "$python_bin" examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir "$search_processed_dir"
    fi

    if [ ! -f "$index_file" ] || [ ! -f "$corpus_file" ]; then
        echo "Downloading Search-R1 retrieval assets into ${search_index_dir}"
        "$python_bin" examples/search/searchr1_download.py --local_dir "$search_index_dir"
        if [ ! -f "$index_file" ]; then
            cat "${search_index_dir}"/part_* > "$index_file"
        fi
        if [ -f "${search_index_dir}/wiki-18.jsonl.gz" ] && [ ! -f "$corpus_file" ]; then
            gzip -df "${search_index_dir}/wiki-18.jsonl.gz"
        fi
    fi

    if [ ! -f "$train_file" ] || [ ! -f "$val_file" ]; then
        echo "Search parquet files are still missing: train=${train_file} val=${val_file}" >&2
        exit 1
    fi
    if [ ! -f "$index_file" ] || [ ! -f "$corpus_file" ]; then
        echo "Search retrieval assets are still missing: index=${index_file} corpus=${corpus_file}" >&2
        exit 1
    fi

    if ! python3 - <<PY
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("${search_host}", ${search_port}))
except Exception:
    sys.exit(1)
finally:
    s.close()
PY
then
    search_server_launch_mode_resolved="$(resolve_search_server_launch_mode)"
    if [ "$search_server_requires_gpu" -eq 1 ] && [ -n "${slurm_gpu_count:-}" ] && is_non_negative_integer "$search_server_gpu"; then
        if [ "$search_server_gpu" -ge "$slurm_gpu_count" ]; then
            echo "SEARCH_SERVER_CUDA_VISIBLE_DEVICES=${search_server_gpu} is outside the allocated GPU range 0..$((slurm_gpu_count - 1))" >&2
            exit 1
        fi
        if [ "$trainer_n_gpus_per_node" -ge "$slurm_gpu_count" ]; then
            echo "trainer_n_gpus_per_node=${trainer_n_gpus_per_node} leaves no GPU free for the local Search retriever; reduce training GPUs or point SEARCH_URL at an external retriever" >&2
            exit 1
        fi
    fi
    echo "Starting retrieval server on ${search_host}:${search_port} device=${search_server_device} faiss_gpu=${search_server_use_faiss_gpu} via ${search_server_launch_mode_resolved}"
    launch_search_server "$search_server_launch_mode_resolved"

    python3 - <<PY
import socket, sys, time
import os
host = "${search_host}"
port = ${search_port}
pid_file = "${search_server_pid_file}"
deadline = time.time() + ${search_server_start_timeout}
while time.time() < deadline:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect((host, port))
        print("retrieval_ready")
        sys.exit(0)
    except Exception:
        if os.path.exists(pid_file):
            try:
                with open(pid_file, "r") as fh:
                    pid = int((fh.read() or "0").strip())
                os.kill(pid, 0)
            except OSError:
                print("retrieval_process_exited")
                sys.exit(1)
            except ValueError:
                pass
        time.sleep(2)
    finally:
        s.close()
print("retrieval_not_ready")
sys.exit(1)
PY
    fi
fi

echo "Launching Search run: variant=${search_variant} seed=${seed} experiment=${experiment_name}"
echo "model_path=${model_path}"
echo "train_file=${train_file}"
echo "val_file=${val_file}"
echo "search_url=${search_url}"
echo "search_request_timeout=${search_request_timeout}"
echo "search_max_concurrent_requests=${search_max_concurrent_requests}"
echo "search_env_max_workers=${search_env_max_workers:-auto}"
echo "search_batch_request_size=${search_batch_request_size:-default}"
echo "trainer_n_gpus_per_node=${trainer_n_gpus_per_node}"
echo "ray_init_num_cpus=${ray_init_num_cpus}"
echo "ocr_max_workers=${ocr_max_workers}"
echo "ocr_font_size=${ocr_font_size}"
echo "ocr_max_width=${ocr_max_width}"
echo "compression_reward_coef=${compression_reward_coef}"
echo "compression_failure_penalty_coef=${compression_failure_penalty_coef}"
echo "compression_reward_every_n_steps=${compression_reward_every_n_steps}"
echo "ocr_trust_policy_enable=${ocr_trust_policy_enable}"
echo "ocr_trust_policy_context_mode=${ocr_trust_policy_context_mode}"
echo "ocr_trust_policy_query_relevance_weight=${ocr_trust_policy_query_relevance_weight}"
echo "ocr_trust_policy_context_budget_percent=${ocr_trust_policy_context_budget_percent}"
echo "ocr_trust_policy_collect_diagnostics=${ocr_trust_policy_collect_diagnostics}"
echo "ocr_trust_policy_min_history_steps_for_compaction=${ocr_trust_policy_min_history_steps_for_compaction}"
echo "agentocr_search_force_search_first_step=${AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP:-0}"
echo "agentocr_search_enable_strict_grounding_prompt=${AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT:-0}"
echo "agentocr_search_special_idea_branch=${AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH:-0}"
echo "agentocr_search_enable_query_rewrite=${AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE:-0}"
echo "agentocr_search_enable_relation_rewrite=${AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE:-0}"
echo "agentocr_search_enable_exact_qualifier_rewrite=${AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE:-0}"
echo "agentocr_search_enable_bridge_attribute_rewrite=${AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE:-0}"
echo "agentocr_search_enable_member_rewrite=${AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE:-0}"
echo "agentocr_search_enable_followup_anchor_rewrite=${AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE:-0}"
echo "agentocr_search_enable_generic_query_rewrite=${AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE:-0}"
echo "agentocr_search_enable_answer_rewrite=${AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE:-0}"
echo "agentocr_search_enable_answer_to_search_rewrite=${AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE:-0}"
echo "agentocr_search_enable_explicit_attribute_answer_guard=${AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD:-0}"

cmd=(
    "$python_bin" "$launcher_script" verl.trainer.main_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=${train_file}"
    "data.val_files=${val_file}"
    "data.train_batch_size=${train_data_size}"
    "data.val_batch_size=${val_data_size}"
    "data.shuffle=${data_shuffle}"
    "data.max_prompt_length=${data_max_prompt_length}"
    "data.max_response_length=${data_max_response_length}"
    "data.filter_overlong_prompts=False"
    "data.truncation=right"
    "data.return_raw_chat=True"
    "+data.apply_chat_template_kwargs.enable_thinking=False"
    "actor_rollout_ref.model.path=${model_path}"
    "actor_rollout_ref.actor.optim.lr=${actor_lr}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${actor_ppo_mini_batch_size}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_ppo_micro_batch_size_per_gpu}"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${actor_fsdp_param_offload}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_fsdp_optimizer_offload}"
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${actor_fsdp_offload_policy}"
    "actor_rollout_ref.actor.strategy=${actor_strategy}"
    "actor_rollout_ref.actor.use_torch_compile=${actor_use_torch_compile}"
    "actor_rollout_ref.model.enable_activation_offload=${actor_enable_activation_offload}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}"
    "actor_rollout_ref.rollout.free_cache_engine=${rollout_free_cache_engine}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=${vllm_skip_mm_profiling}"
    "actor_rollout_ref.actor.use_invalid_action_penalty=True"
    "actor_rollout_ref.actor.invalid_action_penalty_coef=${invalid_action_penalty_coef}"
    "algorithm.use_kl_in_reward=False"
    "ray_init.num_cpus=${ray_init_num_cpus}"
    "env.env_name=search"
    "env.seed=${seed}"
    "env.max_steps=${env_max_steps}"
    "env.rollout.n=${group_size}"
    "env.history_length=${env_history_length}"
    "env.search.search_url=${search_url}"
    "env.search.timeout=${search_request_timeout}"
    "env.search.max_concurrent_requests=${search_max_concurrent_requests}"
    "ocr.use_ocr=True"
    "ocr.use_parallel=${ocr_use_parallel}"
    "ocr.max_workers=${ocr_max_workers}"
    "ocr.font_size=${ocr_font_size}"
    "ocr.max_width=${ocr_max_width}"
    "ocr.agent_select_compression.enable=True"
    "ocr.agent_select_compression.compression_reward_coef=${compression_reward_coef}"
    "ocr.agent_select_compression.compression_failure_penalty_coef=${compression_failure_penalty_coef}"
    "ocr.agent_select_compression.compression_reward_every_n_steps=${compression_reward_every_n_steps}"
    "ocr.trust_policy.enable=${ocr_trust_policy_enable}"
    "ocr.trust_policy.query_conditioned=${ocr_trust_policy_query_conditioned}"
    "ocr.trust_policy.state_aware=${ocr_trust_policy_state_aware}"
    "ocr.trust_policy.context_mode=${ocr_trust_policy_context_mode}"
    "ocr.trust_policy.query_relevance_weight=${ocr_trust_policy_query_relevance_weight}"
    "++ocr.trust_policy.context_budget_percent=${ocr_trust_policy_context_budget_percent}"
    "++ocr.trust_policy.use_compressed_history=${ocr_trust_policy_use_compressed_history}"
    "++ocr.trust_policy.use_prompt_summary=${ocr_trust_policy_use_prompt_summary}"
    "++ocr.trust_policy.collect_diagnostics=${ocr_trust_policy_collect_diagnostics}"
    "++ocr.trust_policy.min_compaction_lines=${ocr_trust_policy_min_compaction_lines}"
    "++ocr.trust_policy.min_prompt_summary_lines=${ocr_trust_policy_min_prompt_summary_lines}"
    "++ocr.trust_policy.min_history_steps_for_compaction=${ocr_trust_policy_min_history_steps_for_compaction}"
    "++ocr.trust_policy.feedback_update_interval=${ocr_trust_policy_feedback_update_interval}"
    "++ocr.trust_policy.feedback_min_history_lines=${ocr_trust_policy_feedback_min_history_lines}"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${project_name}"
    "trainer.experiment_name=${experiment_name}"
    "trainer.n_gpus_per_node=${trainer_n_gpus_per_node}"
    "trainer.nnodes=${trainer_nnodes}"
    "trainer.save_freq=${trainer_save_freq}"
    "trainer.test_freq=${trainer_test_freq}"
    "trainer.total_training_steps=${trainer_total_training_steps}"
    "trainer.val_before_train=${trainer_val_before_train}"
)

if [ -n "${search_env_max_workers}" ]; then
    cmd+=("env.search.max_workers=${search_env_max_workers}")
fi

if [ -n "${search_batch_request_size}" ]; then
    cmd+=("env.search.batch_request_size=${search_batch_request_size}")
fi

cmd+=("$@")

if [ "${DRY_RUN:-0}" = "1" ] || [ "${DRY_RUN:-false}" = "true" ] || [ "${DRY_RUN:-False}" = "True" ]; then
    printf 'DRY_RUN command:\n'
    printf '  %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
