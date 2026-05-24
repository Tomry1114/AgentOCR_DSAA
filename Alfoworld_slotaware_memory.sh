#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=32:00:00
#SBATCH --job-name=agentocr_slotaware_memory
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

export WANDB_MODE="${WANDB_MODE:-online}"

bash /hpc2hdd/home/rtang906/AgentOCR/train_alfworld_slotaware_memory.sh "$@"
