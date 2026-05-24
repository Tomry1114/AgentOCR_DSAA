#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <alfworld|search> [extra hydra overrides...]" >&2
    exit 1
fi

task="$1"
shift

case "$task" in
    alfworld)
        export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-1}"
        export TRAINER_NNODES="${TRAINER_NNODES:-1}"
        export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
        export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:--1}"
        export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:--1}"
        export TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-True}"
        export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-1}"
        export VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
        export GROUP_SIZE="${GROUP_SIZE:-4}"
        export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-8}"
        export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-text_valonly}"
        export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-agentocr_qwen3vl4b_text_valonly_s${SEED:-0}}"

        exec bash ./train_alfworld_qwen3vl4b_text_4xa800.sh \
            trainer.val_only=True \
            "$@"
        ;;
    search)
        export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-1}"
        export TRAINER_NNODES="${TRAINER_NNODES:-1}"
        export TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-1}"
        export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:--1}"
        export TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:--1}"
        export TRAINER_VAL_BEFORE_TRAIN="${TRAINER_VAL_BEFORE_TRAIN:-True}"
        export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-1}"
        export VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
        export GROUP_SIZE="${GROUP_SIZE:-4}"
        export RAY_INIT_NUM_CPUS="${RAY_INIT_NUM_CPUS:-8}"
        export SEARCH_SERVER_CUDA_VISIBLE_DEVICES="${SEARCH_SERVER_CUDA_VISIBLE_DEVICES:-0}"
        export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-search_text_valonly}"
        export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-search_text_qwen3vl4b_valonly_s${SEED:-0}}"

        exec bash ./train_search_text.sh \
            trainer.val_only=True \
            "$@"
        ;;
    *)
        echo "Unsupported task: $task" >&2
        echo "Expected one of: alfworld, search" >&2
        exit 1
        ;;
esac
