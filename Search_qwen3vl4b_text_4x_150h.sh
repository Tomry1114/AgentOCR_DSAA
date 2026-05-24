#!/usr/bin/env bash
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=150:00:00
#SBATCH --job-name=q3vl4b_srchtxt4x
#SBATCH -o /hpc2hdd/home/rtang906/AgentOCR/logs_qwen3/%x-%j.out
#SBATCH -e /hpc2hdd/home/rtang906/AgentOCR/logs_qwen3/%x-%j.err

set -euo pipefail
set -x

cd /hpc2hdd/home/rtang906/AgentOCR
mkdir -p logs_qwen3

if [ -x /hpc2ssd/softwares/anaconda3/bin/conda ]; then
    set +u
    eval "$(/hpc2ssd/softwares/anaconda3/bin/conda shell.bash hook)"
    set -u
else
    source ~/.bashrc
fi

set +u
conda activate AgentOCR
set -u

export QWEN3VL4B_SEARCH_TEXT_4XA800_CONFIG="${QWEN3VL4B_SEARCH_TEXT_4XA800_CONFIG:-/hpc2hdd/home/rtang906/AgentOCR/configs/qwen3vl4b_search_text_4xa800.env}"
export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-search_text_qwen3vl4b_4x_150h_s0}"
export TRAINER_N_GPUS_PER_NODE=4
export TRAINER_TOTAL_TRAINING_STEPS=150

export ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-128}"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-28}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash /hpc2hdd/home/rtang906/AgentOCR/train_search_qwen3vl4b_text_4xa800.sh "$@"
