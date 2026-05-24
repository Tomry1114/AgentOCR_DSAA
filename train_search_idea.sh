#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs_debug

suffix="${SEARCH_IDEA_DEBUG_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
log_path="${SEARCH_IDEA_DEBUG_LOG:-logs_debug/search_idea_debug_${suffix}.log}"

search_idea_mode="${SEARCH_IDEA_MODE:-prod}"
case "${search_idea_mode}" in
    debug|prod)
        ;;
    *)
        echo "Unsupported SEARCH_IDEA_MODE=${search_idea_mode}. Expected debug or prod." >&2
        exit 1
        ;;
esac

maybe_export_default() {
    local var_name="$1"
    local default_value="$2"
    if [ "${!var_name+x}" = x ]; then
        export "${var_name}=${!var_name}"
    elif [ "${search_idea_mode}" = "debug" ]; then
        export "${var_name}=${default_value}"
    fi
}

export SEARCH_VARIANT="${SEARCH_VARIANT:-$([ "${search_idea_mode}" = "debug" ] && echo idea_adapter_only || echo idea)}"
export TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-AgentOCR_search}"
if [ "${search_idea_mode}" = "debug" ]; then
    export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-search_idea_debug_${suffix}}"
fi
export PYTHON_BIN="${PYTHON_BIN:-/hpc2hdd/home/rtang906/AgentOCR/tools/python_glibc234}"
if [ "${search_idea_mode}" = "debug" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
else
    export WANDB_MODE="${WANDB_MODE:-online}"
fi

trainer_logger="${TRAINER_LOGGER:-}"
if [ -z "${trainer_logger}" ]; then
    case "${WANDB_MODE}" in
        online|ONLINE|Online)
            trainer_logger="['console','wandb']"
            ;;
        *)
            trainer_logger="['console']"
            ;;
    esac
fi

if [ "${search_idea_mode}" = "debug" ]; then
    # Debug-only single-GPU defaults. Keep the retriever off the training GPU so
    # this can run safely inside `srun --gres=gpu:1` without touching main jobs.
    maybe_export_default TRAINER_N_GPUS_PER_NODE 1
    maybe_export_default TRAINER_NNODES 1
    maybe_export_default TRAIN_DATA_SIZE 1
    maybe_export_default VAL_DATA_SIZE 1
    maybe_export_default GROUP_SIZE 2
    maybe_export_default RAY_INIT_NUM_CPUS 4

    maybe_export_default DATA_MAX_PROMPT_LENGTH 2048
    maybe_export_default DATA_MAX_RESPONSE_LENGTH 128
    maybe_export_default ENV_MAX_STEPS 2
    maybe_export_default ENV_HISTORY_LENGTH 4

    maybe_export_default ACTOR_PPO_MINI_BATCH_SIZE 2
    maybe_export_default ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU 1
    maybe_export_default ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 1
    maybe_export_default ROLLOUT_GPU_MEMORY_UTILIZATION 0.20
    maybe_export_default ROLLOUT_MAX_MODEL_LEN 1536
    maybe_export_default ROLLOUT_MAX_NUM_BATCHED_TOKENS 1536
    maybe_export_default ROLLOUT_MAX_NUM_SEQS 1
    maybe_export_default ROLLOUT_ENFORCE_EAGER True
    maybe_export_default ROLLOUT_FREE_CACHE_ENGINE True
    maybe_export_default VLLM_SKIP_MM_PROFILING True

    maybe_export_default ACTOR_ENABLE_ACTIVATION_OFFLOAD True
    maybe_export_default ACTOR_FSDP_OFFLOAD_POLICY True

    maybe_export_default SEARCH_PORT 18081
    maybe_export_default SEARCH_MAX_CONCURRENT_REQUESTS 1
    maybe_export_default SEARCH_ENV_MAX_WORKERS 1
    maybe_export_default SEARCH_BATCH_REQUEST_SIZE 1
    maybe_export_default SEARCH_SERVER_USE_FAISS_GPU False
    maybe_export_default SEARCH_SERVER_DEVICE cpu
    maybe_export_default SEARCH_SERVER_LAUNCH_MODE local
    maybe_export_default AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP 0
    maybe_export_default AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT 0
    maybe_export_default AGENTOCR_SEARCH_NO_HISTORY_FORCE_SEARCH "${AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP:-0}"
    maybe_export_default AGENTOCR_SEARCH_STRICT_ENTITY_GROUNDING "${AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT:-0}"

    maybe_export_default OCR_USE_PARALLEL False
    maybe_export_default OCR_MAX_WORKERS 1
    maybe_export_default OCR_FONT_SIZE 12
    maybe_export_default OCR_MAX_WIDTH 392
    maybe_export_default OCR_COMPRESSION_REWARD_COEF 0.01
    maybe_export_default OCR_COMPRESSION_FAILURE_PENALTY_COEF 0.0
    maybe_export_default OCR_COMPRESSION_REWARD_EVERY_N_STEPS 8

    maybe_export_default OCR_TRUST_POLICY_ENABLE False
    maybe_export_default OCR_TRUST_POLICY_QUERY_CONDITIONED False
    maybe_export_default OCR_TRUST_POLICY_STATE_AWARE False
    maybe_export_default OCR_TRUST_POLICY_CONTEXT_MODE auto
    maybe_export_default OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT 0.30
    maybe_export_default OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT 50
    maybe_export_default OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY False
    maybe_export_default OCR_TRUST_POLICY_USE_PROMPT_SUMMARY False
    maybe_export_default OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS False
    maybe_export_default OCR_TRUST_POLICY_MIN_COMPACTION_LINES 8
    maybe_export_default OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES 8
    maybe_export_default OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION 0
    maybe_export_default OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL 4
    maybe_export_default OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES 8

    maybe_export_default TRAINER_SAVE_FREQ -1
    maybe_export_default TRAINER_TEST_FREQ -1
    maybe_export_default TRAINER_TOTAL_TRAINING_STEPS 2
    maybe_export_default TRAINER_VAL_BEFORE_TRAIN False
fi

if [ "${search_idea_mode}" = "debug" ]; then
    debug_asset_dir="${SEARCH_DEBUG_ASSET_DIR:-$(pwd)/tmp_debug_data/search_idea_debug}"
    debug_train_path="${debug_asset_dir}/train.parquet"
    debug_val_path="${debug_asset_dir}/test.parquet"
    debug_corpus_path="${debug_asset_dir}/wiki-18.jsonl"
    debug_index_path="${debug_asset_dir}/e5_Flat.index"

    if [ ! -f "${debug_train_path}" ] || [ ! -f "${debug_val_path}" ] || [ ! -f "${debug_corpus_path}" ] || [ ! -f "${debug_index_path}" ]; then
        echo "[SEARCH_IDEA_DEBUG] preparing tiny debug assets at ${debug_asset_dir}"
        "${PYTHON_BIN}" tools/prepare_search_debug_assets.py \
            --train-source /hpc2hdd/home/rtang906/data/searchR1_processed_direct/train.parquet \
            --val-source /hpc2hdd/home/rtang906/data/searchR1_processed_direct/test.parquet \
            --output-dir "${debug_asset_dir}" \
            --num-train "${SEARCH_DEBUG_NUM_TRAIN:-2}" \
            --num-val "${SEARCH_DEBUG_NUM_VAL:-2}" \
            --retriever-model "${SEARCH_DEBUG_RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
            --device "${SEARCH_DEBUG_RETRIEVER_DEVICE:-cpu}"
    fi

    export TRAIN_FILE="${TRAIN_FILE:-${debug_train_path}}"
    export VAL_FILE="${VAL_FILE:-${debug_val_path}}"
    export SEARCH_INDEX_FILE="${SEARCH_INDEX_FILE:-${debug_index_path}}"
    export SEARCH_CORPUS_FILE="${SEARCH_CORPUS_FILE:-${debug_corpus_path}}"
fi

echo "[SEARCH_IDEA_MODE] ${search_idea_mode}"
echo "[SEARCH_IDEA_DEBUG] experiment=${TRAINER_EXPERIMENT_NAME:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] log=${log_path}"
echo "[SEARCH_IDEA_DEBUG] search_variant=${SEARCH_VARIANT}"
echo "[SEARCH_IDEA_DEBUG] trainer_n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] trainer_logger=${trainer_logger}"
echo "[SEARCH_IDEA_DEBUG] search_server_device=${SEARCH_SERVER_DEVICE:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] search_server_use_faiss_gpu=${SEARCH_SERVER_USE_FAISS_GPU:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] search_port=${SEARCH_PORT:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] agentocr_search_force_search_first_step=${AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] agentocr_search_enable_strict_grounding_prompt=${AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] actor_enable_activation_offload=${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] actor_fsdp_offload_policy=${ACTOR_FSDP_OFFLOAD_POLICY:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] train_file=${TRAIN_FILE:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] val_file=${VAL_FILE:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] search_index_file=${SEARCH_INDEX_FILE:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] search_corpus_file=${SEARCH_CORPUS_FILE:-<inherit>}"
echo "[SEARCH_IDEA_DEBUG] ocr_trust_policy_min_history_steps_for_compaction=${OCR_TRUST_POLICY_MIN_HISTORY_STEPS_FOR_COMPACTION:-<inherit>}"

cmd=(
    bash train_search.sh
    "trainer.logger=${trainer_logger}"
    "$@"
)

if [ "${ROLLOUT_MAX_MODEL_LEN+x}" = x ]; then
    cmd+=("actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}")
fi
if [ "${ROLLOUT_MAX_NUM_BATCHED_TOKENS+x}" = x ]; then
    cmd+=("actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}")
fi
if [ "${ROLLOUT_MAX_NUM_SEQS+x}" = x ]; then
    cmd+=("actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}")
fi

if [ "${DRY_RUN:-0}" = "1" ] || [ "${DRY_RUN:-false}" = "true" ] || [ "${DRY_RUN:-False}" = "True" ]; then
    "${cmd[@]}"
    exit 0
fi

"${cmd[@]}" 2>&1 | tee "${log_path}"
