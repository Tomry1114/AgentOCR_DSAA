#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
mkdir -p logs_tr

# Ensure batch shells can resolve CUDA toolchain for flashinfer JIT.
CUDA_ROOT_CANDIDATE="/hpc2ssd/softwares/cuda/cuda-12.6"

if [ -f /etc/profile ]; then
  # shellcheck disable=SC1091
  set +u
  source /etc/profile >/dev/null 2>&1 || true
  set -u
fi
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
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

export MODEL_PATH="${MODEL_PATH:-/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct}"
export VERL_HF_ATTN_IMPLEMENTATION="${VERL_HF_ATTN_IMPLEMENTATION:-sdpa}"
export AGENTOCR_BOOT_DEBUG="${AGENTOCR_BOOT_DEBUG:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHON_BIN="${PYTHON_BIN:-$(cd "$(dirname "$0")" && pwd)/tools/python_glibc234}"
export AGENTOCR_TASKRUNNER_NUM_CPUS="${AGENTOCR_TASKRUNNER_NUM_CPUS:-0}"
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-4}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export GROUP_SIZE="${GROUP_SIZE:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
export OCR_MAX_WORKERS="${OCR_MAX_WORKERS:-1}"
export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-2}"
export TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-True}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-1}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-console_wandb}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-1}"
export TRAINER_NNODES="${TRAINER_NNODES:-1}"
export DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-768}"
export DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-96}"
export ENV_MAX_STEPS="${ENV_MAX_STEPS:-24}"
export ENV_HISTORY_LENGTH="${ENV_HISTORY_LENGTH:-16}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-1536}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1536}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-1}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
export ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-1}"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
export ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
export ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-False}"
export ACTOR_ENABLE_ACTIVATION_OFFLOAD="${ACTOR_ENABLE_ACTIVATION_OFFLOAD:-True}"
export ACTOR_FSDP_PARAM_OFFLOAD="${ACTOR_FSDP_PARAM_OFFLOAD:-False}"
export ACTOR_FSDP_OPTIMIZER_OFFLOAD="${ACTOR_FSDP_OPTIMIZER_OFFLOAD:-False}"
export ACTOR_FSDP_OFFLOAD_POLICY="${ACTOR_FSDP_OFFLOAD_POLICY:-True}"
export ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-0}"
export ACTOR_LORA_ALPHA="${ACTOR_LORA_ALPHA:-16}"
export ACTOR_LORA_TARGET_MODULES="${ACTOR_LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
export VLLM_SKIP_MM_PROFILING="${VLLM_SKIP_MM_PROFILING:-True}"

export OCR_TRUST_POLICY_ENABLE="${OCR_TRUST_POLICY_ENABLE:-True}"
export OCR_TRUST_POLICY_QUERY_CONDITIONED="${OCR_TRUST_POLICY_QUERY_CONDITIONED:-True}"
export OCR_TRUST_POLICY_STATE_AWARE="${OCR_TRUST_POLICY_STATE_AWARE:-True}"
export OCR_TRUST_POLICY_CONTEXT_MODE="${OCR_TRUST_POLICY_CONTEXT_MODE:-auto}"
export OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT="${OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT:-0.30}"

suffix="${FULLIDEA_DEBUG_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-agentocr_fullidea_qwen3vl4b_debug_${suffix}}"
log_path="${FULLIDEA_DEBUG_LOG:-logs_tr/fullidea_qwen3vl4b_debug_${suffix}.log}"

echo "[FULLIDEA_DEBUG] experiment=${TRAINER_EXPERIMENT_NAME}"
echo "[FULLIDEA_DEBUG] model=${MODEL_PATH}"
echo "[FULLIDEA_DEBUG] log=${log_path}"
echo "[FULLIDEA_DEBUG] gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
echo "[FULLIDEA_DEBUG] data_max_prompt_length=${DATA_MAX_PROMPT_LENGTH}"
echo "[FULLIDEA_DEBUG] data_max_response_length=${DATA_MAX_RESPONSE_LENGTH}"
echo "[FULLIDEA_DEBUG] env_max_steps=${ENV_MAX_STEPS}"
echo "[FULLIDEA_DEBUG] env_history_length=${ENV_HISTORY_LENGTH}"
echo "[FULLIDEA_DEBUG] actor_strategy=${ACTOR_STRATEGY}"
echo "[FULLIDEA_DEBUG] actor_activation_offload=${ACTOR_ENABLE_ACTIVATION_OFFLOAD}"
echo "[FULLIDEA_DEBUG] actor_fsdp_offload_policy=${ACTOR_FSDP_OFFLOAD_POLICY}"
echo "[FULLIDEA_DEBUG] lora_rank=${ACTOR_LORA_RANK}"
echo "[FULLIDEA_DEBUG] lora_target_modules=${ACTOR_LORA_TARGET_MODULES}"
echo "[FULLIDEA_DEBUG] skip_mm_profiling=${VLLM_SKIP_MM_PROFILING}"
echo "[FULLIDEA_DEBUG] pythonnousersite=${PYTHONNOUSERSITE}"
echo "[FULLIDEA_DEBUG] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-}"
echo "[FULLIDEA_DEBUG] python_bin=${PYTHON_BIN}"
echo "[FULLIDEA_DEBUG] cuda_home=${CUDA_HOME:-}"
echo "[FULLIDEA_DEBUG] nvcc=$(command -v nvcc || echo missing)"

cmd=(
  bash train_alfworld_qwen3vl4b_ocr_slotaware_4xa800.sh
  data.max_prompt_length="${DATA_MAX_PROMPT_LENGTH}"
  data.max_response_length="${DATA_MAX_RESPONSE_LENGTH}"
  env.max_steps="${ENV_MAX_STEPS}"
  env.history_length="${ENV_HISTORY_LENGTH}"
  actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY}"
  actor_rollout_ref.actor.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}"
  actor_rollout_ref.model.enable_activation_offload="${ACTOR_ENABLE_ACTIVATION_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_FSDP_PARAM_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_FSDP_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.offload_policy="${ACTOR_FSDP_OFFLOAD_POLICY}"
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
)

if [ "${ACTOR_LORA_RANK}" -gt 0 ]; then
  cmd+=(
    actor_rollout_ref.model.lora_rank="${ACTOR_LORA_RANK}"
    actor_rollout_ref.model.lora_alpha="${ACTOR_LORA_ALPHA}"
    actor_rollout_ref.model.target_modules="${ACTOR_LORA_TARGET_MODULES}"
  )
fi

cmd+=("$@")

"${cmd[@]}" 2>&1 | tee "${log_path}"
