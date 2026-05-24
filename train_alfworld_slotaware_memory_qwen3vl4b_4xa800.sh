#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "$0")"

# Ensure batch shells can resolve CUDA toolchain for flashinfer/flash-attention JIT.
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

config_file="${QWEN3VL4B_4XA800_CONFIG:-$(pwd)/configs/qwen3vl4b_slotaware_4xa800.env}"
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
# Do not let the head TaskRunner reserve a full Ray CPU slot. This only affects
# Ray scheduling and keeps more CPU resources available for env/OCR workers.
export AGENTOCR_TASKRUNNER_NUM_CPUS="${AGENTOCR_TASKRUNNER_NUM_CPUS:-0}"

# Avoid inheriting a pre-existing Ray cluster address from login shells or
# previous debug sessions. This launcher expects to start its own local Ray.
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

variant="${EXPERIMENT_VARIANT}"
seed="${SEED}"
query_weight="${OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT}"

case "$variant" in
    baseline|slotaware)
        ;;
    *)
        echo "Unsupported variant: $variant (expected: baseline|slotaware)" >&2
        exit 1
        ;;
esac

if [ "$variant" = "baseline" ]; then
    extra_args=()
else
    extra_args=(
        "ocr.trust_policy.enable=${OCR_TRUST_POLICY_ENABLE}"
        "ocr.trust_policy.query_conditioned=${OCR_TRUST_POLICY_QUERY_CONDITIONED}"
        "ocr.trust_policy.state_aware=${OCR_TRUST_POLICY_STATE_AWARE}"
        "ocr.trust_policy.context_mode=${OCR_TRUST_POLICY_CONTEXT_MODE}"
        "ocr.trust_policy.query_relevance_weight=${query_weight}"
    )
fi

echo "Preparing shared visual parquet with original ALFWorld split logic: train=${TRAIN_DATA_SIZE} test=$((VAL_DATA_SIZE * 2))"
"$python_bin" "$launcher_script" examples.data_preprocess.prepare \
    --mode visual \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$((VAL_DATA_SIZE * 2))"

echo "Launching 4xA800 Qwen3-VL-4B run: variant=${variant} seed=${seed} experiment=${TRAINER_EXPERIMENT_NAME}"
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
    "trainer.critic_warmup=0"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${TRAINER_PROJECT_NAME}"
    "trainer.experiment_name=${TRAINER_EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${TRAINER_N_GPUS_PER_NODE}"
    "trainer.nnodes=${TRAINER_NNODES}"
    "trainer.save_freq=${TRAINER_SAVE_FREQ}"
    "trainer.test_freq=${TRAINER_TEST_FREQ}"
    "trainer.total_epochs=${TRAINER_TOTAL_EPOCHS}"
    "trainer.val_before_train=${TRAINER_VAL_BEFORE_TRAIN}"
)

cmd+=("${extra_args[@]}")
cmd+=("$@")

if [ "${DRY_RUN:-0}" = "1" ] || [ "${DRY_RUN:-false}" = "true" ] || [ "${DRY_RUN:-False}" = "True" ]; then
    printf 'DRY_RUN command:\n'
    printf '  %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
