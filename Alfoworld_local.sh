#!/bin/bash
#SBATCH -p i64m1tga800ue
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=32:00:00
#SBATCH --job-name=agentocr_ppo_local
#SBATCH -o /hpc2hdd/home/rtang906/AgentOCR/logs_tr/%x-%j.out
#SBATCH -e /hpc2hdd/home/rtang906/AgentOCR/logs_tr/%x-%j.err

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
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-32}}"

bash /hpc2hdd/home/rtang906/AgentOCR/train_alfworld_local.sh "$@"
