#!/usr/bin/env bash
set -euo pipefail

job_id="${1:-9753601}"
workdir="/hpc2hdd/home/rtang906/AgentOCR"
watch_log="${WATCH_LOG:-${workdir}/logs_qwen3/search_ocr_qwen3vl4b_7train1retr_noidea_8xa800_s0.watch.log}"
launch_log="${LAUNCH_LOG:-${workdir}/logs_qwen3/search_ocr_qwen3vl4b_7train1retr_noidea_8xa800_s0.launch.log}"
poll_seconds="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "$watch_log")"

echo "[watcher] started $(date) job_id=${job_id}" >>"$watch_log"

while rtk srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 bash -lc "pgrep -af 'bash train_search_text.sh' >/dev/null"; do
    echo "[watcher] search_text still running $(date)" >>"$watch_log"
    sleep "${poll_seconds}"
done

echo "[watcher] launching search_ocr $(date)" >>"$watch_log"

rtk srun --jobid="${job_id}" --overlap --nodes=1 --ntasks=1 \
    bash -lc "cd ${workdir} && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 && export SEARCH_VARIANT=noidea && export TRAINER_EXPERIMENT_NAME=search_ocr_qwen3vl4b_7train1retr_noidea_8xa800_s0 && bash train_search.sh" \
    >>"$launch_log" 2>&1

rc=$?
echo "[watcher] launch command exited rc=${rc} $(date)" >>"$watch_log"
exit "${rc}"
