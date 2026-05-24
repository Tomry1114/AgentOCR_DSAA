#!/bin/bash
#SBATCH -p i64m1tga800ue          # 你之前用的 A800 分区
#SBATCH --gres=gpu:4              # 【重要】申请 4 张 A800！
#SBATCH --cpus-per-task=32        # 【重要】申请 32 个 CPU 核心，保障 64 个 OCR worker 并发不卡死
#SBATCH --time=32:00:00           # 强烈建议 24 小时保底！
#SBATCH --job-name=agentocr_ppo_v5_v2   # 任务名字
#SBATCH -o /hpc2hdd/home/rtang906/AgentOCR/logs_tr/%x-%j.out
#SBATCH -e /hpc2hdd/home/rtang906/AgentOCR/logs_tr/%x-%j.err

# 1. 激活你的 Conda 环境
source ~/.bashrc
conda activate AgentOCR

# 2. 进入你的代码目录
cd ~/AgentOCR

# 3. 【修改】开启 WandB 在线实时同步！
export WANDB_MODE=online

# 4. 运行你的启动命令！
bash /hpc2hdd/home/rtang906/AgentOCR/train_alfworld.sh
