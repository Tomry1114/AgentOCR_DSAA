#!/usr/bin/env bash
#SBATCH -p i64m1tga800u
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=150:00:00
#SBATCH --job-name=q3vl4b_ocrbase4x
#SBATCH -o /hpc2hdd/home/rtang906/AgentOCR/logs_qwen3/%x-%j.out
#SBATCH -e /hpc2hdd/home/rtang906/AgentOCR/logs_qwen3/%x-%j.err

set -euo pipefail
set -x

cd /hpc2hdd/home/rtang906/AgentOCR

export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-agentocr_q3vl4b_ocrbase4x_150h_mb128_s0}"
export TRAINER_N_GPUS_PER_NODE=4
export TRAINER_TOTAL_EPOCHS=150

export TRAIN_FILE=/hpc2hdd/home/rtang906/data/verl-agent-q3vl4b-ocr/visual/train.parquet
export VAL_FILE=/hpc2hdd/home/rtang906/data/verl-agent-q3vl4b-ocr/visual/test.parquet
export SKIP_DATA_PREPARE=1

export AGENTOCR_TASKRUNNER_NUM_CPUS=1
export RAY_INIT_NUM_CPUS=28
export OCR_MAX_WORKERS=24
export NUM_CPUS_PER_ENV_WORKER=0.08

export ACTOR_PPO_MINI_BATCH_SIZE=128
export ACTOR_ENABLE_ACTIVATION_OFFLOAD=False
export ACTOR_FSDP_OFFLOAD_POLICY=False
export ACTOR_FSDP_PARAM_OFFLOAD=False
export ACTOR_FSDP_OPTIMIZER_OFFLOAD=False

./train_alfworld_qwen3vl4b_ocr_4xa800.sh
