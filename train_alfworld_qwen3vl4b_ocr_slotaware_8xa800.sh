#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "$0")"

CUDA_ROOT_CANDIDATE="/hpc2ssd/softwares/cuda/cuda-12.6"

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

config_file="${QWEN3VL4B_8XA800_CONFIG:-$(pwd)/configs/qwen3vl4b_slotaware_8xa800.env}"
if [ ! -f "$config_file" ]; then
    echo "Missing config file: $config_file" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$config_file"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HIGHLIGHT_CONFIGS="${HIGHLIGHT_CONFIGS:-[Observation]:0,0,255;[Action]:255,0,0}"
export AGENTOCR_QWEN3_OCR_MAX_PIXELS="${AGENTOCR_QWEN3_OCR_MAX_PIXELS:-131072}"
export AGENTOCR_QWEN3_OCR_MIN_PIXELS="${AGENTOCR_QWEN3_OCR_MIN_PIXELS:-16384}"
export AGENTOCR_TASKRUNNER_NUM_CPUS="${AGENTOCR_TASKRUNNER_NUM_CPUS:-0}"

for ray_var in RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV RAY_JOB_CONFIG_JSON; do
    if [ -n "${!ray_var:-}" ]; then
        echo "Unsetting inherited ${ray_var}=${!ray_var}"
    fi
    unset "${ray_var}"
done

if [ -x /hpc2ssd/softwares/anaconda3/bin/conda ]; then
    set +u
    eval "$(/hpc2ssd/softwares/anaconda3/bin/conda shell.bash hook)"
    conda activate AgentOCR
    set -u
fi

launcher_script="$(pwd)/tools/run_module_sanitized.py"
python_bin="${PYTHON_BIN}"
seed="${SEED}"
query_weight="${OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT}"
experiment_name="${TRAINER_EXPERIMENT_NAME:-agentocr_qwen3vl4b_ocr_slotaware_8xa800_s${seed}}"
prepare_local_dir="${PREPARE_LOCAL_DIR:-$HOME/data/verl-agent/}"
skip_data_prepare="${SKIP_DATA_PREPARE:-0}"
prepared_local_root="${prepare_local_dir/#\~/$HOME}"
prepared_visual_dir="${prepared_local_root%/}/visual"

if [ "$skip_data_prepare" = "1" ] || [ "$skip_data_prepare" = "true" ] || [ "$skip_data_prepare" = "True" ]; then
    echo "Skipping data prepare; using existing parquet files: train=${TRAIN_FILE} val=${VAL_FILE}"
else
    echo "Preparing shared visual parquet with original ALFWorld split logic: train=${TRAIN_DATA_SIZE} test=$((VAL_DATA_SIZE * 2))"
    "$python_bin" "$launcher_script" examples.data_preprocess.prepare \
        --mode visual \
        --local_dir "$prepare_local_dir" \
        --train_data_size "$TRAIN_DATA_SIZE" \
        --val_data_size "$((VAL_DATA_SIZE * 2))"
    export TRAIN_FILE="${prepared_visual_dir}/train.parquet"
    export VAL_FILE="${prepared_visual_dir}/test.parquet"
    echo "Using freshly prepared parquet files: train=${TRAIN_FILE} val=${VAL_FILE}"
fi

echo "Launching 8xA800 Qwen3-VL-4B OCR+slotaware run: seed=${seed} experiment=${experiment_name}"
echo "Config file: ${config_file}"
echo "python_bin=${python_bin}"
echo "cuda_home=${CUDA_HOME:-}"
echo "nvcc=$(command -v nvcc || echo missing)"
echo "taskrunner_num_cpus=${AGENTOCR_TASKRUNNER_NUM_CPUS}"
echo "actor_lr=${ACTOR_LR}"
echo "qwen3_ocr_pixels=${AGENTOCR_QWEN3_OCR_MIN_PIXELS}-${AGENTOCR_QWEN3_OCR_MAX_PIXELS}"

cmd=(
    "$python_bin" "$launcher_script" verl.trainer.main_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.train_batch_size=${TRAIN_DATA_SIZE}"
    "data.val_batch_size=${VAL_DATA_SIZE}"
    "data.max_prompt_length=${DATA_MAX_PROMPT_LENGTH}"
    "data.max_response_length=${DATA_MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=False"
    "data.truncation=right"
    "data.return_raw_chat=True"
    "+data.apply_chat_template_kwargs.enable_thinking=False"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_FSDP_PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_FSDP_OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_FSDP_OFFLOAD_POLICY}"
    "actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY}"
    "actor_rollout_ref.actor.use_torch_compile=${ACTOR_USE_TORCH_COMPILE}"
    "actor_rollout_ref.model.enable_activation_offload=${ACTOR_ENABLE_ACTIVATION_OFFLOAD}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=${VLLM_SKIP_MM_PROFILING}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.4"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "actor_rollout_ref.actor.use_invalid_action_penalty=True"
    "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1"
    "algorithm.use_kl_in_reward=False"
    "ray_init.num_cpus=${RAY_INIT_NUM_CPUS}"
    "env.env_name=alfworld/AlfredTWEnv"
    "env.seed=${seed}"
    "env.max_steps=${ENV_MAX_STEPS}"
    "env.history_length=${ENV_HISTORY_LENGTH}"
    "env.rollout.n=${GROUP_SIZE}"
    "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER}"
    "ocr.use_ocr=True"
    "ocr.max_workers=${OCR_MAX_WORKERS}"
    "ocr.font_size=10"
    "ocr.max_width=392"
    "ocr.agent_select_compression.enable=True"
    "ocr.agent_select_compression.compression_reward_coef=0.01"
    "ocr.agent_select_compression.compression_reward_every_n_steps=8"
    "ocr.trust_policy.enable=${OCR_TRUST_POLICY_ENABLE}"
    "ocr.trust_policy.query_conditioned=${OCR_TRUST_POLICY_QUERY_CONDITIONED}"
    "ocr.trust_policy.state_aware=${OCR_TRUST_POLICY_STATE_AWARE}"
    "ocr.trust_policy.context_mode=${OCR_TRUST_POLICY_CONTEXT_MODE}"
    "ocr.trust_policy.query_relevance_weight=${query_weight}"
    "++ocr.trust_policy.context_budget_percent=${OCR_TRUST_POLICY_CONTEXT_BUDGET_PERCENT}"
    "++ocr.trust_policy.use_compressed_history=${OCR_TRUST_POLICY_USE_COMPRESSED_HISTORY}"
    "++ocr.trust_policy.use_prompt_summary=${OCR_TRUST_POLICY_USE_PROMPT_SUMMARY}"
    "++ocr.trust_policy.collect_diagnostics=${OCR_TRUST_POLICY_COLLECT_DIAGNOSTICS}"
    "++ocr.trust_policy.min_compaction_lines=${OCR_TRUST_POLICY_MIN_COMPACTION_LINES}"
    "++ocr.trust_policy.min_prompt_summary_lines=${OCR_TRUST_POLICY_MIN_PROMPT_SUMMARY_LINES}"
    "++ocr.trust_policy.feedback_update_interval=${OCR_TRUST_POLICY_FEEDBACK_UPDATE_INTERVAL}"
    "++ocr.trust_policy.feedback_min_history_lines=${OCR_TRUST_POLICY_FEEDBACK_MIN_HISTORY_LINES}"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${TRAINER_PROJECT_NAME}"
    "trainer.experiment_name=${experiment_name}"
    "trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE}"
    "trainer.nnodes=${TRAINER_NNODES}"
    "trainer.save_freq=${TRAINER_SAVE_FREQ}"
    "trainer.test_freq=${TRAINER_TEST_FREQ}"
    "trainer.total_epochs=${TRAINER_TOTAL_EPOCHS}"
    "trainer.val_before_train=${TRAINER_VAL_BEFORE_TRAIN}"
)

cmd+=("$@")

if [ "${DRY_RUN:-0}" = "1" ] || [ "${DRY_RUN:-false}" = "true" ] || [ "${DRY_RUN:-False}" = "True" ]; then
    printf 'DRY_RUN command:\n'
    printf '  %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
