#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "$0")"

config_file="${QWEN3VL4B_SEARCH_TEXT_4XA800_CONFIG:-$(pwd)/configs/qwen3vl4b_search_text_4xa800.env}"
if [ ! -f "$config_file" ]; then
    echo "Missing config file: $config_file" >&2
    exit 1
fi
source "$config_file"

###############
# Highlight configs: use environment variable to avoid Hydra parsing issues with < > characters
# Format: "context1:r,g,b;context2:r,g,b"
# <search> and </search> are highlighted in blue (0,0,255)
# <information> and </information> are highlighted in red (255,0,0)
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'
###############

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

echo "Launching 4xA800 Search text Qwen3-VL-4B run: seed=${SEED} experiment=${TRAINER_EXPERIMENT_NAME}"
echo "Config file: ${config_file}"
echo "model_path=${MODEL_PATH}"
echo "train_file=${TRAIN_FILE}"
echo "val_file=${VAL_FILE}"

"${PYTHON_BIN}" tools/run_module_sanitized.py verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_DATA_SIZE}" \
    data.val_batch_size="${VAL_DATA_SIZE}" \
    data.max_prompt_length="${DATA_MAX_PROMPT_LENGTH}" \
    data.max_response_length="${DATA_MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_FSDP_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_FSDP_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.offload_policy="${ACTOR_FSDP_OFFLOAD_POLICY}" \
    actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY}" \
    actor_rollout_ref.actor.use_torch_compile="${ACTOR_USE_TORCH_COMPILE}" \
    actor_rollout_ref.model.enable_activation_offload="${ACTOR_ENABLE_ACTIVATION_OFFLOAD}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling="${VLLM_SKIP_MM_PROFILING}" \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    ray_init.num_cpus="${RAY_INIT_NUM_CPUS}" \
    env.env_name=search \
    env.seed="${SEED}" \
    env.max_steps="${ENV_MAX_STEPS}" \
    env.rollout.n="${GROUP_SIZE}" \
    env.history_length="${ENV_HISTORY_LENGTH}" \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    ocr.use_ocr=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${TRAINER_PROJECT_NAME}" \
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
    trainer.nnodes="${TRAINER_NNODES}" \
    trainer.save_freq="${TRAINER_SAVE_FREQ}" \
    trainer.test_freq="${TRAINER_TEST_FREQ}" \
    trainer.total_training_steps="${TRAINER_TOTAL_TRAINING_STEPS}" \
    trainer.val_before_train="${TRAINER_VAL_BEFORE_TRAIN}" \
    "$@"
