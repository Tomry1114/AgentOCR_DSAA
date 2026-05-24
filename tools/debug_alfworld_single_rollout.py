#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld import alfworld_projection, build_alfworld_envs
from tools.qwen3_alfworld_ocr_success_smoke import load_model, run_generation, score_output


DEFAULT_MODEL_PATH = "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct"


def _as_image_list(image_like):
    if image_like is None:
        return []
    if isinstance(image_like, (list, tuple)):
        return list(image_like)
    return [image_like]


def build_config(model_path: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "env": {
                "history_length": 50,
                "max_steps": 50,
            },
            "ocr": {
                "use_ocr": True,
                "font_size": 10,
                "max_width": 392,
                "padding": 2,
                "use_parallel": False,
                "max_workers": 1,
                "agent_select_compression": {
                    "enable": True,
                    "compression_reward_coef": 0.01,
                    "compression_failure_penalty_coef": 0.0,
                    "compression_reward_every_n_steps": 8,
                },
                "trust_policy": {
                    "enable": True,
                    "query_conditioned": True,
                    "state_aware": True,
                    "context_mode": "auto",
                    "query_relevance_weight": 0.30,
                    "use_compressed_history": True,
                    "use_prompt_summary": False,
                    "collect_diagnostics": True,
                    "min_compaction_lines": 8,
                    "min_prompt_summary_lines": 8,
                    "feedback_update_interval": 4,
                    "feedback_min_history_lines": 8,
                },
            },
            "actor_rollout_ref": {
                "model": {
                    "path": model_path,
                }
            },
        }
    )


def build_manager(seed: int, model_path: str, eval_dataset: str) -> AlfWorldEnvironmentManager:
    root = Path(__file__).resolve().parents[1]
    alf_config_path = root / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    envs = build_alfworld_envs(
        str(alf_config_path),
        seed=seed,
        env_num=1,
        group_n=1,
        is_train=False,
        env_kwargs={"eval_dataset": eval_dataset},
        resources_per_worker={"num_cpus": 0.2, "num_gpus": 0.0},
    )
    return AlfWorldEnvironmentManager(envs, alfworld_projection, build_config(model_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one or more real ALFWorld eval seeds with the current Qwen3 OCR prompt/runtime."
    )
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--eval-dataset", default="eval_in_distribution")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    processor, model = load_model(args.model_path, args.device)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = []
    for seed in args.seed:
        manager = build_manager(seed=seed, model_path=args.model_path, eval_dataset=args.eval_dataset)
        trace = []
        try:
            obs, infos = manager.reset([{}])
            task = manager.tasks[0]
            gamefile = infos[0].get("extra.gamefile")
            print(f"SEED: {seed}", flush=True)
            print(f"TASK: {task}", flush=True)
            print(f"GAMEFILE: {gamefile}", flush=True)

            for step in range(1, manager.config.env.max_steps + 1):
                prompt_text = obs["text"][0]
                current_observation = obs["anchor"][0]
                admissible_actions = list(manager.envs.get_admissible_commands[0])
                raw_output, mm_diagnostics, _ = run_generation(
                    model=model,
                    processor=processor,
                    prompt_text=prompt_text,
                    images=_as_image_list(obs["image"][0]),
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                )
                scored = score_output(raw_output, admissible_actions)
                next_obs, rewards, dones, step_infos = manager.step([scored["normalized_output"]])
                trust_diagnostics = (
                    dict(manager.trust_policy_last_metrics[0])
                    if manager.trust_policy_last_metrics and manager.trust_policy_last_metrics[0]
                    else {}
                )
                record = {
                    "step": step,
                    "task": task,
                    "gamefile": gamefile,
                    "current_observation": current_observation,
                    "admissible_actions": admissible_actions,
                    "raw_output": raw_output,
                    "normalized_output": scored["normalized_output"],
                    "parsed_action": scored["parsed_action"],
                    "is_valid": bool(scored["is_valid"]),
                    "selected_compression": float(scored["compression_factor"]),
                    "mm_diagnostics": mm_diagnostics,
                    "trust_diagnostics": trust_diagnostics,
                    "reward": float(rewards[0]),
                    "done": bool(dones[0]),
                    "won": bool(step_infos[0].get("won", False)),
                    "next_observation": next_obs["anchor"][0],
                }
                trace.append(record)
                print(
                    f"STEP {step} | action={record['parsed_action']} | valid={record['is_valid']} | "
                    f"reward={record['reward']} | done={record['done']} | won={record['won']}",
                    flush=True,
                )
                if dones[0]:
                    break
                obs = next_obs

            payloads.append(
                {
                    "seed": seed,
                    "task": task,
                    "gamefile": gamefile,
                    "won": bool(trace[-1]["won"]) if trace else False,
                    "steps": len(trace),
                    "trace": trace,
                }
            )
        finally:
            manager.close()

    output_path.write_text(json.dumps(payloads, indent=2), encoding="utf-8")
    print(f"WROTE_JSON {output_path}", flush=True)


if __name__ == "__main__":
    main()
