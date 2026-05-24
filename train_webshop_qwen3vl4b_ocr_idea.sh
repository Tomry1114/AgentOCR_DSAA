#!/usr/bin/env bash
set -euo pipefail
set -x

cd "$(dirname "$0")"

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

PATCH_ROOT="${PATCH_ROOT:-/hpc2hdd/home/rtang906/AgentOCR/isolated_webshop_ocr_idea_patch}"
PATCH_VENDOR_ROOT="${PATCH_VENDOR_ROOT:-${PATCH_ROOT}/_vendor}"
WEBSHOP_DATA_ROOT="${WEBSHOP_DATA_ROOT:-/hpc2hdd/home/rtang906/AgentOCR/supplement-data/webshop/data}"
PROJECT_ROOT="${PROJECT_ROOT:-/hpc2hdd/home/rtang906/AgentOCR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export AGENTOCR_ENABLE_WEBSHOP_OCR_PATCH=1
export PYTHONPATH="${PATCH_VENDOR_ROOT}:${PATCH_ROOT}:${PYTHONPATH:-}"
export WEBSHOP_DATA_ROOT
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-webshop_qwen3vl4b_ocr_idea}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-AgentOCR_webshop}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-2}"
TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-150}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"
RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-28}"
OCR_MAX_WORKERS="${OCR_MAX_WORKERS:-24}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.08}"

echo "python_bin=${PYTHON_BIN}"
echo "cuda_home=${CUDA_HOME:-}"
echo "nvcc=$(command -v nvcc || echo missing)"
echo "vllm_attention_backend=${VLLM_ATTENTION_BACKEND}"
echo "vllm_use_flashinfer_sampler=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "vllm_disable_flashinfer_prefill=${VLLM_DISABLE_FLASHINFER_PREFILL}"

"${PYTHON_BIN}" -m examples.data_preprocess.prepare \
    --mode visual \
    --train_data_size "${TRAIN_DATA_SIZE}" \
    --val_data_size "${VAL_DATA_SIZE}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" tools/run_module_sanitized.py verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$HOME/data/verl-agent/visual/train.parquet" \
    data.val_files="$HOME/data/verl-agent/visual/test.parquet" \
    data.train_batch_size="${TRAIN_DATA_SIZE}" \
    data.val_batch_size="${VAL_DATA_SIZE}" \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    ray_init.num_cpus="${RAY_INIT_NUM_CPUS}" \
    env.env_name=WebshopOCR \
    env.seed=0 \
    env.max_steps=15 \
    env.history_length=15 \
    env.rollout.n="${GROUP_SIZE}" \
    env.webshop.use_small=True \
    env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
    ocr.use_ocr=True \
    ocr.use_parallel=True \
    ocr.max_workers="${OCR_MAX_WORKERS}" \
    ocr.font_size=10 \
    ocr.max_width=392 \
    ocr.agent_select_compression.enable=True \
    ocr.agent_select_compression.compression_reward_coef=0.01 \
    ocr.agent_select_compression.compression_reward_every_n_steps=5 \
    ocr.trust_policy.enable=True \
    ocr.trust_policy.query_conditioned=True \
    ocr.trust_policy.state_aware=False \
    ocr.trust_policy.context_mode=auto \
    ocr.trust_policy.query_relevance_weight=0.30 \
    ++ocr.trust_policy.context_budget_percent=50 \
    ++ocr.trust_policy.use_compressed_history=True \
    ++ocr.trust_policy.use_prompt_summary=False \
    ++ocr.trust_policy.collect_diagnostics=False \
    ++ocr.trust_policy.min_compaction_lines=8 \
    ++ocr.trust_policy.min_prompt_summary_lines=8 \
    ++ocr.trust_policy.feedback_update_interval=4 \
    ++ocr.trust_policy.feedback_min_history_lines=8 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${TRAINER_PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq="${TRAINER_TEST_FREQ}" \
    trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}" \
    trainer.val_before_train=True \
    "$@"
