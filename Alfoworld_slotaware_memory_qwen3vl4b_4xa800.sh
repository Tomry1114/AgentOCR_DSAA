#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=60:00:00
#SBATCH --job-name=agentocr_q3vl4b_mem4x
#SBATCH -o /hpc2hdd/home/rtang906/AgentOCR/logs_tr_2/%x-%j.out
#SBATCH -e /hpc2hdd/home/rtang906/AgentOCR/logs_tr_2/%x-%j.err

set -eo pipefail

if [ -x /hpc2ssd/softwares/anaconda3/bin/conda ]; then
    eval "$(/hpc2ssd/softwares/anaconda3/bin/conda shell.bash hook)"
else
    source ~/.bashrc
fi

set +u
conda activate AgentOCR
set -u

cd /hpc2hdd/home/rtang906/AgentOCR
mkdir -p logs_tr_2

export WANDB_MODE="${WANDB_MODE:-online}"
export QWEN3VL4B_4XA800_CONFIG="${QWEN3VL4B_4XA800_CONFIG:-/hpc2hdd/home/rtang906/AgentOCR/configs/qwen3vl4b_slotaware_4xa800.env}"

bash /hpc2hdd/home/rtang906/AgentOCR/train_alfworld_slotaware_memory_qwen3vl4b_4xa800.sh "$@"
