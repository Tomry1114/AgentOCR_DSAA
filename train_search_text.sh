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
    slurm_gpu_count="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*gres\/gpu=\([0-9]\+\).*/\1/p' | head -n 1)"
fi

default_search_config="$(pwd)/configs/qwen3vl4b_search_text_4xa800.env"
if [ "${slurm_gpu_count:-0}" -ge 8 ] 2>/dev/null && [ -f "$(pwd)/configs/qwen3vl4b_search_text_8xa800.env" ]; then
    default_search_config="$(pwd)/configs/qwen3vl4b_search_text_8xa800.env"
fi

config_file="${QWEN3_SEARCH_TEXT_CONFIG:-$default_search_config}"
if [ -f "$config_file" ]; then
    # shellcheck disable=SC1090
    source "$config_file"
fi

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
experiment_name="${TRAINER_EXPERIMENT_NAME:-search_text_qwen3vl4b_s${seed}}"

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
search_max_concurrent_requests="${SEARCH_MAX_CONCURRENT_REQUESTS:-8}"
search_env_max_workers="${SEARCH_ENV_MAX_WORKERS:-}"
search_batch_request_size="${SEARCH_BATCH_REQUEST_SIZE:-}"
search_server_gpu="${SEARCH_SERVER_CUDA_VISIBLE_DEVICES:-7}"
search_server_log="${SEARCH_SERVER_LOG:-$(pwd)/logs_debug/search_retriever_${search_port}.log}"
search_server_pid_file="${SEARCH_SERVER_PID_FILE:-$(pwd)/logs_debug/search_retriever_${search_port}.pid}"
search_server_start_timeout="${SEARCH_SERVER_START_TIMEOUT:-900}"
search_server_launch_mode="${SEARCH_SERVER_LAUNCH_MODE:-auto}"

train_data_size="${TRAIN_DATA_SIZE:-128}"
val_data_size="${VAL_DATA_SIZE:-512}"
group_size="${GROUP_SIZE:-8}"
data_max_prompt_length="${DATA_MAX_PROMPT_LENGTH:-14000}"
data_max_response_length="${DATA_MAX_RESPONSE_LENGTH:-512}"
env_max_steps="${ENV_MAX_STEPS:-4}"
env_history_length="${ENV_HISTORY_LENGTH:-4}"
actor_lr="${ACTOR_LR:-1e-6}"
actor_ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE:-128}"
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

if [ -n "${TRAINER_N_GPUS_PER_NODE_OVERRIDE:-}" ]; then
    trainer_n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE_OVERRIDE}"
elif [ -n "${slurm_gpu_count}" ]; then
    trainer_n_gpus_per_node="${slurm_gpu_count}"
else
    trainer_n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE:-8}"
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
# Format: "context1:r,g,b;context2:r,g,b"
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

resolve_search_server_launch_mode() {
    if [ "$search_server_launch_mode" = "local" ] || [ "$search_server_launch_mode" = "slurm_step" ]; then
        echo "$search_server_launch_mode"
        return 0
    fi
    if [ "$search_server_launch_mode" != "auto" ]; then
        echo "Unsupported SEARCH_SERVER_LAUNCH_MODE=${search_server_launch_mode}" >&2
        return 1
    fi

    # When the training step narrows CUDA_VISIBLE_DEVICES (for example 0-6),
    # the retriever's reserved GPU must be launched from a separate SLURM step.
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

    rm -f "$search_server_pid_file"
    if [ "$launch_mode" = "slurm_step" ]; then
        if ! command -v srun >/dev/null 2>&1; then
            echo "srun is required for SEARCH_SERVER_LAUNCH_MODE=slurm_step" >&2
            return 1
        fi
        printf -v workdir_q '%q' "$(pwd)"
        printf -v inner_cmd '%q ' \
            env \
            "CUDA_VISIBLE_DEVICES=${search_server_gpu}" \
            "_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=${_FAISS_WHEEL_DISABLE_CUDA_PRELOAD}" \
            "$python_bin" \
            examples/search/retriever/retrieval_server.py \
            --index_path "$index_file" \
            --corpus_path "$corpus_file" \
            --topk 3 \
            --retriever_name e5 \
            --retriever_model intfloat/e5-base-v2 \
            --faiss_gpu \
            --port "$search_port"
        nohup srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=1 --ntasks=1 \
            bash -lc "cd ${workdir_q} && exec ${inner_cmd}" \
            >"$search_server_log" 2>&1 &
    else
        nohup env \
            "CUDA_VISIBLE_DEVICES=${search_server_gpu}" \
            "_FAISS_WHEEL_DISABLE_CUDA_PRELOAD=${_FAISS_WHEEL_DISABLE_CUDA_PRELOAD}" \
            "$python_bin" examples/search/retriever/retrieval_server.py \
            --index_path "$index_file" \
            --corpus_path "$corpus_file" \
            --topk 3 \
            --retriever_name e5 \
            --retriever_model intfloat/e5-base-v2 \
            --faiss_gpu \
            --port "$search_port" \
            >"$search_server_log" 2>&1 &
    fi
    echo $! > "$search_server_pid_file"
}

mkdir -p "$(dirname "$search_server_log")"
mkdir -p "$search_processed_dir" "$search_index_dir"

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
    if [ -n "${slurm_gpu_count:-}" ] && is_non_negative_integer "$search_server_gpu"; then
        if [ "$search_server_gpu" -ge "$slurm_gpu_count" ]; then
            echo "SEARCH_SERVER_CUDA_VISIBLE_DEVICES=${search_server_gpu} is outside the allocated GPU range 0..$((slurm_gpu_count - 1))" >&2
            exit 1
        fi
        if [ "$trainer_n_gpus_per_node" -ge "$slurm_gpu_count" ]; then
            echo "trainer_n_gpus_per_node=${trainer_n_gpus_per_node} leaves no GPU free for the local Search retriever; reduce training GPUs or point SEARCH_URL at an external retriever" >&2
            exit 1
        fi
    fi
    echo "Starting retrieval server on ${search_host}:${search_port} using GPU ${search_server_gpu} via ${search_server_launch_mode_resolved}"
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

echo "Launching Search text run: seed=${seed} experiment=${experiment_name}"
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

cmd=(
    "$python_bin" "$launcher_script" verl.trainer.main_ppo
    "algorithm.adv_estimator=grpo"
    "data.train_files=${train_file}"
    "data.val_files=${val_file}"
    "data.train_batch_size=${train_data_size}"
    "data.val_batch_size=${val_data_size}"
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
    "ocr.use_ocr=False"
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
