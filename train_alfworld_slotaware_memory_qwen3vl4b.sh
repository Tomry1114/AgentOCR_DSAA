#!/usr/bin/env bash
set -eo pipefail
set -x

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

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export AGENTOCR_QWEN3_OCR_MAX_PIXELS="${AGENTOCR_QWEN3_OCR_MAX_PIXELS:-131072}"
export AGENTOCR_QWEN3_OCR_MIN_PIXELS="${AGENTOCR_QWEN3_OCR_MIN_PIXELS:-16384}"

variant="${EXPERIMENT_VARIANT:-slotaware}"
seed="${SEED:-0}"
query_weight="${OCR_TRUST_POLICY_QUERY_RELEVANCE_WEIGHT:-0.30}"
model_path="${MODEL_PATH:-/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct}"
export VERL_HF_ATTN_IMPLEMENTATION="${VERL_HF_ATTN_IMPLEMENTATION:-sdpa}"
train_file="${TRAIN_FILE:-$HOME/data/verl-agent/visual/train.parquet}"
val_file="${VAL_FILE:-$HOME/data/verl-agent/visual/test.parquet}"
trainer_project_name="${TRAINER_PROJECT_NAME:-AgentOCR_alfworld}"
actor_lr="${ACTOR_LR:-5e-6}"

case "$variant" in
  baseline|slotaware)
    ;;
  *)
    echo "Unsupported variant: $variant (expected: baseline|slotaware)" >&2
    exit 1
    ;;
esac

export HIGHLIGHT_CONFIGS='[Observation]:0,0,255;[Action]:255,0,0'

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128
group_size=8

if [ "$variant" = "baseline" ]; then
    experiment_name="${TRAINER_EXPERIMENT_NAME:-agentocr_slotaware_memory_qwen3vl4b_baseline_s${seed}}"
    extra_args=()
else
    experiment_name="${TRAINER_EXPERIMENT_NAME:-agentocr_slotaware_memory_qwen3vl4b_s${seed}}"
    extra_args=(
        ocr.trust_policy.enable=True
        ocr.trust_policy.query_conditioned=True
        ocr.trust_policy.state_aware=True
        ocr.trust_policy.context_mode=auto
        ocr.trust_policy.query_relevance_weight="${query_weight}"
    )
fi

echo "Launching slotaware-memory Qwen3-VL-4B run: variant=${variant} seed=${seed} experiment=${experiment_name}"
echo "actor_lr=${actor_lr}"
echo "qwen3_ocr_pixels=${AGENTOCR_QWEN3_OCR_MIN_PIXELS}-${AGENTOCR_QWEN3_OCR_MAX_PIXELS}"

python3 -m examples.data_preprocess.prepare \
    --mode "visual" \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed="${seed}" \
    env.max_steps=50 \
    env.history_length=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    ocr.use_ocr=True \
    ocr.max_workers=64 \
    ocr.font_size=10 \
    ocr.max_width=392 \
    ocr.agent_select_compression.enable=True \
    ocr.agent_select_compression.compression_reward_coef=0.01 \
    ocr.agent_select_compression.compression_reward_every_n_steps=8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${trainer_project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    "${extra_args[@]}" \
    "$@"
