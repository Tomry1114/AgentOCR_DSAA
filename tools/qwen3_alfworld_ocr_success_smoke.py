#!/usr/bin/env python3
import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.prompts.alfworld import ALFWORLD_QWEN3VL_SYSTEM_PROMPT
from agent_system.multi_turn_rollout.rollout_loop import (
    _build_qwen3_multimodal_model_inputs,
    _looks_like_alfworld_react_prompt,
    _normalize_action_reasoning_format,
    _strip_duplicate_react_system_header,
)
from agent_system.multi_turn_rollout.utils import get_agentocr_image_metadata
from examples.agentocr_pilots.run_alfworld_multitemplate_query_conditioned_pilot import (
    make_multitemplate_task,
)
from examples.agentocr_pilots.run_alfworld_query_conditioned_pilot import TinyScriptedAlfWorldEnv


DEFAULT_MODEL_PATH = "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct"
DEFAULT_OUTPUT = "outputs/qwen3_alfworld_ocr_success_smoke/results.json"


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def temporary_env(overrides: Dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def build_config(model_path: str, variant: str) -> Any:
    trust_enabled = variant != "baseline"
    query_conditioned = variant == "idea"
    return OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 2048,
                "truncation": "right",
                "return_raw_chat": True,
                "apply_chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            "env": {
                "history_length": 50,
                "max_steps": 8,
                "rollout": {"n": 1},
            },
            "ocr": {
                "use_ocr": True,
                "font_size": 10,
                "max_width": 392,
                "padding": 2,
                "use_parallel": False,
                "agent_select_compression": {
                    "enable": True,
                    "compression_factor_max": 10.0,
                    "compression_reward_coef": 0.01,
                    "compression_failure_penalty_coef": 0.0,
                    "compression_reward_every_n_steps": 8,
                },
                "trust_policy": {
                    "enable": trust_enabled,
                    "query_conditioned": query_conditioned,
                    "state_aware": query_conditioned,
                    "context_mode": "auto" if query_conditioned else None,
                    "query_relevance_weight": 0.30,
                    "use_compressed_history": trust_enabled,
                    "use_prompt_summary": False,
                    "collect_diagnostics": True,
                },
            },
            "actor_rollout_ref": {
                "model": {
                    "path": model_path,
                }
            },
        }
    )


def build_cases() -> List[Dict[str, Any]]:
    specs = [
        (0, 1, 5),
        (2, 1, 5),
        (3, 1, 5),
        (4, 1, 5),
    ]
    cases = []
    for template_id, seed, distractor_count in specs:
        task = make_multitemplate_task(template_id=template_id, seed=seed, distractor_count=distractor_count)
        extra_actions = []
        extra_observations = []
        for extra_idx in range(4):
            place = f"filler locker {template_id}-{extra_idx + 1}"
            obj = f"filler object {template_id}-{extra_idx + 1}"
            extra_actions.append(f"examine {place}")
            extra_observations.append(f"You inspect the {place}. The {obj} is in the {place}.")
        task["scripted_actions"] = list(task["scripted_actions"]) + extra_actions
        task["step_observations"] = list(task["step_observations"]) + extra_observations
        task["case_name"] = f"template{template_id}_seed{seed}_d{distractor_count}"
        cases.append(task)
    return cases


def _response_for_action(action: str) -> str:
    return f"<thinking>setup</thinking><action>{action}</action><compression>1.0</compression>"


def _as_image_list(images: Any) -> List[Any]:
    if images is None:
        return []
    if isinstance(images, (list, tuple)):
        return list(images)
    return [images]


def _save_image(image_like: Any, path: Path) -> Image.Image:
    if isinstance(image_like, Image.Image):
        image = image_like
    else:
        image = Image.fromarray(image_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image


def collect_case_observation(
    task: Dict[str, Any],
    *,
    model_path: str,
    variant: str,
) -> Dict[str, Any]:
    config = build_config(model_path=model_path, variant=variant)
    manager = AlfWorldEnvironmentManager(TinyScriptedAlfWorldEnv(task), alfworld_projection, config)
    try:
        obs, _ = manager.reset([{}])
        infos = []
        for scripted_action in task["scripted_actions"]:
            obs, _rewards, _dones, infos = manager.step([_response_for_action(scripted_action)])

        final_text = obs["text"][0]
        final_images = _as_image_list(obs["image"][0] if obs["image"] else None)
        admissible_actions = list(manager.envs.get_admissible_commands[0])
        diagnostics = (
            dict(manager.trust_policy_last_metrics[0])
            if manager.trust_policy_last_metrics and manager.trust_policy_last_metrics[0]
            else {}
        )
        effective_compression = (
            float(manager.last_effective_compression_factors[0])
            if manager.last_effective_compression_factors
            else 1.0
        )
        last_info = dict(infos[0]) if infos else {}
        image_metadata = [get_agentocr_image_metadata(image) for image in final_images]
        return {
            "prompt_text": final_text,
            "images": final_images,
            "admissible_actions": admissible_actions,
            "diagnostics": diagnostics,
            "effective_compression": effective_compression,
            "last_info": last_info,
            "image_metadata": image_metadata,
        }
    finally:
        manager.close()


def build_chat_prompt(processor, prompt_text: str, image_count: int) -> str:
    obs_content = str(prompt_text or "")
    if image_count > 0:
        obs_content = obs_content.replace("<image>\n", "", 1).replace("<image>", "", 1)

    if _looks_like_alfworld_react_prompt(obs_content):
        obs_content = _strip_duplicate_react_system_header(obs_content)
        chat = [
            {
                "role": "system",
                "content": [{"type": "text", "text": ALFWORLD_QWEN3VL_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *([{"type": "image"}] * max(1, image_count)),
                    {"type": "text", "text": obs_content},
                ],
            },
        ]
    else:
        chat = [
            {
                "role": "user",
                "content": [
                    *([{"type": "image"}] * max(1, image_count)),
                    {"type": "text", "text": obs_content},
                ],
            }
        ]

    return processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def load_model(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return processor, model


def run_generation(
    *,
    model,
    processor,
    prompt_text: str,
    images: List[Any],
    max_prompt_length: int,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any], List[Image.Image]]:
    raw_prompt = build_chat_prompt(processor, prompt_text=prompt_text, image_count=len(images))
    processed_images, _processor_kwargs, model_inputs, mm_diagnostics = _build_qwen3_multimodal_model_inputs(
        raw_prompt=raw_prompt,
        obs_image=images,
        processor=processor,
        max_prompt_length=max_prompt_length,
    )
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = model_inputs["input_ids"].shape[1]
    output_tokens = generated[0][input_len:]
    output_text = processor.decode(output_tokens, skip_special_tokens=False)
    mm_diagnostics = dict(mm_diagnostics)
    mm_diagnostics["prompt_char_count"] = len(raw_prompt)
    mm_diagnostics["input_token_count"] = int(input_len)
    return output_text, mm_diagnostics, processed_images


def score_output(output_text: str, admissible_actions: List[str]) -> Dict[str, Any]:
    normalized = _normalize_action_reasoning_format(output_text)
    parsed_actions, valids, compression_factors = alfworld_projection(
        [normalized],
        [admissible_actions],
        check_compression_tag=True,
    )
    return {
        "normalized_output": normalized,
        "parsed_action": parsed_actions[0],
        "is_valid": bool(valids[0]),
        "compression_factor": float(compression_factors[0]),
    }


def summarize(rows: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    variant_rows = [row for row in rows if row["variant"] == variant]
    success_count = sum(1 for row in variant_rows if row["success"])
    visual_tokens = [float(row["mm_diagnostics"]["visual_tokens"]) for row in variant_rows]
    return {
        "variant": variant,
        "num_cases": len(variant_rows),
        "success_rate": (success_count / len(variant_rows)) if variant_rows else 0.0,
        "success_count": success_count,
        "mean_visual_tokens": (sum(visual_tokens) / len(visual_tokens)) if visual_tokens else 0.0,
    }


def save_case_artifacts(
    case_dir: Path,
    *,
    prompt_text: str,
    raw_output: str,
    normalized_output: str,
    processed_images: Iterable[Image.Image],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (case_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
    (case_dir / "normalized_output.txt").write_text(normalized_output, encoding="utf-8")
    for index, image in enumerate(processed_images):
        image.save(case_dir / f"history_{index}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3 ALFWorld OCR success on real single-image OCR inputs.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()

    output_path = Path(args.output)
    artifact_root = output_path.parent / f"{output_path.stem}_artifacts"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(args.model_path, args.device)
    rows: List[Dict[str, Any]] = []

    env_overrides = {
        "AGENTOCR_QWEN3_HISTORY_PAGES": "0",
    }
    with temporary_env(env_overrides):
        for task in build_cases():
            for variant in ("baseline", "idea"):
                collected = collect_case_observation(
                    task,
                    model_path=args.model_path,
                    variant=variant,
                )
                raw_output, mm_diagnostics, processed_images = run_generation(
                    model=model,
                    processor=processor,
                    prompt_text=collected["prompt_text"],
                    images=collected["images"],
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                )
                scored = score_output(raw_output, collected["admissible_actions"])
                success = scored["parsed_action"] == task["ground_truth_action"]
                case_dir = artifact_root / variant / task["case_name"]
                save_case_artifacts(
                    case_dir,
                    prompt_text=collected["prompt_text"],
                    raw_output=raw_output,
                    normalized_output=scored["normalized_output"],
                    processed_images=processed_images,
                )
                rows.append(
                    {
                        "variant": variant,
                        "case_name": task["case_name"],
                        "task_description": task["task_description"],
                        "ground_truth_action": task["ground_truth_action"],
                        "admissible_actions": collected["admissible_actions"],
                        "parsed_action": scored["parsed_action"],
                        "success": success,
                        "is_valid": scored["is_valid"],
                        "selected_compression": scored["compression_factor"],
                        "raw_output": raw_output,
                        "prompt_text": collected["prompt_text"],
                        "mm_diagnostics": mm_diagnostics,
                        "trust_diagnostics": collected["diagnostics"],
                        "effective_render_compression": collected["effective_compression"],
                        "image_metadata": collected["image_metadata"],
                        "artifact_dir": str(case_dir),
                    }
                )

    payload = {
        "summary": {
            "model_path": args.model_path,
            "device": args.device,
            "baseline": summarize(rows, "baseline"),
            "idea": summarize(rows, "idea"),
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote smoke results to {output_path}")


if __name__ == "__main__":
    main()
