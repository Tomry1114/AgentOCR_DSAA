#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs_tr

export WANDB_MODE="${WANDB_MODE:-offline}"
export AGENTOCR_BOOT_DEBUG="${AGENTOCR_BOOT_DEBUG:-1}"
export AGENTOCR_TASKRUNNER_NUM_CPUS="${AGENTOCR_TASKRUNNER_NUM_CPUS:-0}"
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-4}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
export OCR_MAX_WORKERS="${OCR_MAX_WORKERS:-1}"
export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-2}"
export TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-False}"
export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:--1}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-console}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-1}"
export TRAINER_NNODES="${TRAINER_NNODES:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.2}"

suffix="${TP0_DEBUG_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-tp0_u_alfworld_trust_off_${suffix}}"
log_path="${TP0_DEBUG_LOG:-logs_tr/tp0_u_debug_${suffix}.log}"

echo "[TP0_DEBUG] experiment=${TRAINER_EXPERIMENT_NAME}"
echo "[TP0_DEBUG] log=${log_path}"
echo "[TP0_DEBUG] gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"

bash train_alfworld_local.sh \
  data.max_prompt_length=1024 \
  data.max_response_length=128 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.max_model_len=1152 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1152 \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  "$@" 2>&1 | tee "${log_path}"
