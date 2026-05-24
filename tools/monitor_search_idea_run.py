#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import wandb


BASELINE_STEP1_SUCCESS_RATE = 0.348
BASELINE_STEP1_MEMORY_TOKENS = 280.571
BASELINE_VAL_SUCCESS_RATE = 0.36159336419753085
BASELINE_VAL_MEMORY_TOKENS = 284.0824279785156

WATCH_KEYS = [
    "episode/success_rate",
    "val/success_rate",
    "prompt_length/memory_tokens/mean",
    "val/memory_tokens/mean",
    "prompt_length/mean",
    "ocr/image_height/mean",
    "trust_policy/render_compacted_rate",
]

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
LOG_FALLBACK_KEYS = set(WATCH_KEYS + ["training/global_step"])
TRAIN_PROGRESS_RE = re.compile(r"Training Progress:.*?(\d+)/(\d+)\s+\[")
VAL_PROGRESS_RE = re.compile(r"Validation Progress:.*?(\d+)/(\d+)\s+\[")


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def maybe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_metric_line(line: str) -> Dict[str, Optional[float]]:
    clean_line = ANSI_ESCAPE_RE.sub("", line).strip()
    if "step:" not in clean_line and "val/success_rate" not in clean_line:
        return {}

    metric_start = clean_line.find("step:")
    if metric_start >= 0:
        clean_line = clean_line[metric_start:]

    parsed: Dict[str, Optional[float]] = {}
    for chunk in clean_line.split(" - "):
        key, sep, raw_value = chunk.partition(":")
        if not sep:
            continue
        key = key.strip()
        raw_value = raw_value.strip()
        if key == "step":
            value = maybe_float(raw_value)
            if value is not None:
                parsed["_step"] = int(value)
            continue
        if key not in LOG_FALLBACK_KEYS:
            continue
        value = maybe_float(raw_value)
        if value is not None:
            parsed[key] = value
    return parsed


def latest_log_snapshot(log_path: str) -> Dict[str, Optional[float]]:
    if not log_path:
        return {}

    path = Path(log_path)
    if not path.is_file():
        return {}

    latest: Dict[str, Optional[float]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            clean_line = ANSI_ESCAPE_RE.sub("", line).strip()

            if "Validation Progress:" in clean_line:
                match = VAL_PROGRESS_RE.search(clean_line)
                if match:
                    latest["_phase"] = "val"
                    latest["_progress_current"] = int(match.group(1))
                    latest["_progress_total"] = int(match.group(2))
                continue

            if "Training Progress:" in clean_line:
                match = TRAIN_PROGRESS_RE.search(clean_line)
                if match:
                    latest["_phase"] = "train"
                    latest["_progress_current"] = int(match.group(1))
                    latest["_progress_total"] = int(match.group(2))
                continue

            if "step:" not in clean_line and "val/success_rate" not in clean_line:
                continue
            parsed = parse_metric_line(clean_line)
            if parsed:
                latest.update(parsed)
    return latest


def fetch_latest_rows(run: "wandb.apis.public.Run", keys: List[str]) -> List[Dict]:
    rows = list(run.scan_history(keys=["_step"] + keys, page_size=200))
    filtered = []
    for row in rows:
        if any(row.get(key) is not None for key in keys):
            filtered.append(row)
    return filtered


def current_snapshot(
    run: "wandb.apis.public.Run",
    log_snapshot: Optional[Dict[str, Optional[float]]] = None,
    prefer_log: bool = False,
) -> Dict[str, Optional[float]]:
    summary = run.summary
    snapshot = {key: summary.get(key) for key in WATCH_KEYS}
    if snapshot.get("val/success_rate") is None:
        snapshot["val/success_rate"] = summary.get("success_rate")
    if log_snapshot:
        for key in WATCH_KEYS:
            if log_snapshot.get(key) is None:
                continue
            if prefer_log or snapshot.get(key) is None:
                snapshot[key] = log_snapshot.get(key)
    return snapshot


def status_from_snapshot(
    snapshot: Dict[str, Optional[float]],
    current_step: Optional[int],
    *,
    step1_min_success_rate: float,
    step1_max_prompt_memory: float,
    val_min_success_rate: float,
    val_max_memory: float,
) -> str:
    ep_sr = snapshot.get("episode/success_rate")
    prompt_mem = snapshot.get("prompt_length/memory_tokens/mean")
    if current_step is not None and current_step >= 1 and ep_sr is not None and prompt_mem is not None:
        if ep_sr < step1_min_success_rate and prompt_mem >= step1_max_prompt_memory:
            return "step1_worse_both"
        if ep_sr < step1_min_success_rate:
            return "step1_sr_not_enough"
        if prompt_mem >= step1_max_prompt_memory:
            return "step1_token_not_enough"

    val_sr = snapshot.get("val/success_rate")
    val_mem = snapshot.get("val/memory_tokens/mean")
    if val_sr is None or val_mem is None:
        return "waiting_for_val"
    if val_sr > val_min_success_rate and val_mem < val_max_memory:
        return "target_met"
    if val_sr <= val_min_success_rate and val_mem >= val_max_memory:
        return "worse_both"
    if val_sr <= val_min_success_rate:
        return "sr_not_enough"
    return "token_not_enough"


def should_stop_for_status(status: str, *, kill_on_step1_fail: bool, kill_on_val_fail: bool) -> bool:
    if status.startswith("step1_"):
        return kill_on_step1_fail
    if status in {"worse_both", "sr_not_enough", "token_not_enough"}:
        return kill_on_val_fail
    return False


def _run_command(argv: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def find_experiment_pgid(jobid: str, experiment_name: str) -> Optional[int]:
    result = _run_command(
        [
            "srun",
            f"--jobid={jobid}",
            "--overlap",
            "-N1",
            "-n1",
            "ps",
            "-eo",
            "pid,pgid,cmd",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "failed to inspect process table")

    matched_pgids: List[int] = []
    for line in result.stdout.splitlines():
        if experiment_name not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pgid = int(parts[1])
        except ValueError:
            continue
        matched_pgids.append(pgid)
    if not matched_pgids:
        return None
    return matched_pgids[0]


def kill_experiment_group(jobid: str, experiment_name: str, signal_name: str) -> Dict[str, Optional[str]]:
    pgid = find_experiment_pgid(jobid, experiment_name)
    if pgid is None:
        return {"ok": False, "reason": "pgid_not_found", "pgid": None}

    signal_name = str(signal_name or "INT").strip().upper()
    kill_cmd = f"kill -{shlex.quote(signal_name)} -- -{int(pgid)}"
    result = _run_command(
        [
            "srun",
            f"--jobid={jobid}",
            "--overlap",
            "-N1",
            "-n1",
            "bash",
            "-lc",
            kill_cmd,
        ]
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "reason": result.stderr.strip() or result.stdout.strip() or "kill_failed",
            "pgid": str(pgid),
        }
    return {"ok": True, "reason": "signal_sent", "pgid": str(pgid)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a search OCR idea W&B run against the OCR baseline.")
    parser.add_argument("--run", required=True, help="W&B run path: entity/project/run_id")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--max-polls", type=int, default=0, help="Optional cap on polling iterations; 0 means unlimited")
    parser.add_argument("--jsonl", default="", help="Optional JSONL output path")
    parser.add_argument("--log", default="", help="Optional local training log path for early-step fallback parsing")
    parser.add_argument("--step1-min-success-rate", type=float, default=BASELINE_STEP1_SUCCESS_RATE)
    parser.add_argument("--step1-max-prompt-memory", type=float, default=BASELINE_STEP1_MEMORY_TOKENS)
    parser.add_argument("--val-min-success-rate", type=float, default=BASELINE_VAL_SUCCESS_RATE)
    parser.add_argument("--val-max-memory", type=float, default=BASELINE_VAL_MEMORY_TOKENS)
    parser.add_argument("--kill-on-step1-fail", action="store_true")
    parser.add_argument("--kill-on-val-fail", action="store_true")
    parser.add_argument("--jobid", default="", help="Interactive Slurm job id used to locate the training process group")
    parser.add_argument("--experiment-name", default="", help="Exact TRAINER_EXPERIMENT_NAME used to match the training process group")
    parser.add_argument("--kill-signal", default="INT", help="Signal name passed to kill, e.g. INT or TERM")
    args = parser.parse_args()

    api = wandb.Api()
    polls = 0
    last_seen_marker = None
    stop_already_attempted = False

    while True:
        polls += 1
        run = api.run(args.run)
        rows = fetch_latest_rows(run, WATCH_KEYS)
        latest_row = rows[-1] if rows else {}
        log_snapshot = latest_log_snapshot(args.log)
        current_step = latest_row.get("_step")
        if current_step is None:
            current_step = log_snapshot.get("_step")
        snapshot = current_snapshot(run, log_snapshot=log_snapshot, prefer_log=not rows)
        status = status_from_snapshot(
            snapshot,
            current_step,
            step1_min_success_rate=args.step1_min_success_rate,
            step1_max_prompt_memory=args.step1_max_prompt_memory,
            val_min_success_rate=args.val_min_success_rate,
            val_max_memory=args.val_max_memory,
        )

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run": args.run,
            "status": status,
            "latest_step": current_step,
            "phase": log_snapshot.get("_phase"),
            "progress_current": log_snapshot.get("_progress_current"),
            "progress_total": log_snapshot.get("_progress_total"),
            "rows": len(rows),
            "log_step": log_snapshot.get("_step"),
            "episode_success_rate": snapshot.get("episode/success_rate"),
            "val_success_rate": snapshot.get("val/success_rate"),
            "prompt_memory_tokens_mean": snapshot.get("prompt_length/memory_tokens/mean"),
            "val_memory_tokens_mean": snapshot.get("val/memory_tokens/mean"),
            "prompt_length_mean": snapshot.get("prompt_length/mean"),
            "ocr_image_height_mean": snapshot.get("ocr/image_height/mean"),
            "delta_val_success_rate": None
            if snapshot.get("val/success_rate") is None
            else snapshot.get("val/success_rate") - args.val_min_success_rate,
            "delta_val_memory_tokens_mean": None
            if snapshot.get("val/memory_tokens/mean") is None
            else snapshot.get("val/memory_tokens/mean") - args.val_max_memory,
            "delta_step1_success_rate": None
            if snapshot.get("episode/success_rate") is None
            else snapshot.get("episode/success_rate") - args.step1_min_success_rate,
            "delta_step1_prompt_memory_tokens_mean": None
            if snapshot.get("prompt_length/memory_tokens/mean") is None
            else snapshot.get("prompt_length/memory_tokens/mean") - args.step1_max_prompt_memory,
        }

        kill_result: Optional[Dict[str, Optional[str]]] = None
        if (
            not stop_already_attempted
            and should_stop_for_status(
                status,
                kill_on_step1_fail=args.kill_on_step1_fail,
                kill_on_val_fail=args.kill_on_val_fail,
            )
        ):
            if args.jobid and args.experiment_name:
                kill_result = kill_experiment_group(args.jobid, args.experiment_name, args.kill_signal)
                payload["kill_result"] = kill_result
                stop_already_attempted = bool(kill_result.get("ok"))
            else:
                payload["kill_result"] = {"ok": False, "reason": "missing_jobid_or_experiment_name", "pgid": None}

        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=True) + "\n")

        step_prefix = f"step={current_step}" if current_step is not None else "step=NA"
        phase = payload["phase"] or "unknown"
        if payload["progress_current"] is not None and payload["progress_total"] is not None:
            phase = f"{phase}:{int(payload['progress_current'])}/{int(payload['progress_total'])}"
        print(
            "[SEARCH_IDEA_MONITOR]"
            f" {payload['ts']}"
            f" {step_prefix}"
            f" phase={phase}"
            f" rows={len(rows)}"
            f" status={status}"
            f" val_sr={format_float(snapshot.get('val/success_rate'))}"
            f" val_mem={format_float(snapshot.get('val/memory_tokens/mean'))}"
            f" ep_sr={format_float(snapshot.get('episode/success_rate'))}"
            f" prompt_mem={format_float(snapshot.get('prompt_length/memory_tokens/mean'))}"
            f" prompt_len={format_float(snapshot.get('prompt_length/mean'))}"
            f" img_h={format_float(snapshot.get('ocr/image_height/mean'))}"
            f" kill={kill_result.get('reason') if kill_result else 'NA'}",
            flush=True,
        )

        current_marker = (
            current_step,
            payload["phase"],
            payload["progress_current"],
            payload["progress_total"],
        )
        if current_marker != last_seen_marker:
            print(json.dumps(payload, ensure_ascii=True), flush=True)
            last_seen_marker = current_marker

        if args.once:
            return 0
        if args.max_polls > 0 and polls >= args.max_polls:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    sys.exit(main())
