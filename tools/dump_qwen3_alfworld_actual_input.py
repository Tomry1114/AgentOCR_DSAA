#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from agentocr import OCRTool
from agent_system.environments.base import parse_highlight_configs
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_QWEN3VL_SYSTEM_PROMPT,
    ALFWORLD_TEMPLATE_OCR_QWEN3,
    build_alfworld_rules_for_model,
)
from agent_system.multi_turn_rollout.utils import process_qwen3_ocr_image


DEFAULT_HISTORY = """[Observation]: you are in the kitchen.
[Action]: look
[Observation]: you notice a countertop 1 and fridge 1 nearby.
[Action]: go to countertop 1
[Observation]: you see apple 1 on countertop 1.
[Action]: take apple 1 from countertop 1
[Observation]: you are carrying apple 1.
[Action]: go to fridge 1
[Observation]: you arrive at fridge 1."""


def _looks_like_alfworld_react_prompt(obs_content: str) -> bool:
    if not obs_content:
        return False
    markers = [
        "Task:",
        "Current step:",
        "Current textual observation:",
        "Admissible actions:",
        "Rules:",
    ]
    return all(marker in obs_content for marker in markers)


def _strip_duplicate_react_system_header(obs_content: str) -> str:
    normalized = str(obs_content).lstrip()
    system_header = ALFWORLD_QWEN3VL_SYSTEM_PROMPT.strip()
    if not normalized.startswith(system_header):
        return obs_content
    remainder = normalized[len(system_header):]
    if remainder.startswith("\n"):
        remainder = remainder[1:]
    return remainder.lstrip()


def _count_mm_visual_tokens(image_grid_thw, merge_size: int) -> int:
    if image_grid_thw is None:
        return 0
    merge_length = max(1, int(merge_size) ** 2)
    total = 0
    for idx in range(len(image_grid_thw)):
        total += int((image_grid_thw[idx].prod() // merge_length).item())
    return total


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _build_qwen3_multimodal_model_inputs(raw_prompt: str, obs_image, processor, max_prompt_length: int):
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
    temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2) or 2)
    merge_size = int(getattr(image_processor, "merge_size", 2) or 2)

    requested_max_pixels = _env_int("AGENTOCR_QWEN3_OCR_MAX_PIXELS", 131072)
    min_pixels = _env_int("AGENTOCR_QWEN3_OCR_MIN_PIXELS", 16384)
    safety_margin = max(0, _env_int("AGENTOCR_QWEN3_MM_PROMPT_SAFETY_MARGIN", 256))
    preferred_visual_floor = max(256, _env_int("AGENTOCR_QWEN3_MM_MIN_VISUAL_TOKENS", 896))
    hard_visual_floor = max(256, _env_int("AGENTOCR_QWEN3_MM_HARD_MIN_VISUAL_TOKENS", 384))
    max_shrink_steps = max(1, _env_int("AGENTOCR_QWEN3_MM_MAX_SHRINK_STEPS", 4))
    safe_prompt_len = max(512, int(max_prompt_length) - safety_margin)

    processor_kwargs = {"do_resize": False}
    diagnostics = {
        "safe_prompt_len": int(safe_prompt_len),
        "requested_max_pixels": int(requested_max_pixels),
        "shrink_steps": 0,
    }

    obs_images = list(obs_image) if isinstance(obs_image, (list, tuple)) else [obs_image]
    current_max_pixels = max(min_pixels * max(1, len(obs_images)), requested_max_pixels)
    per_image_max_pixels = max(min_pixels, int(current_max_pixels / max(1, len(obs_images))))
    processed_images = [
        process_qwen3_ocr_image(
            image,
            max_pixels=per_image_max_pixels,
            min_pixels=min_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        for image in obs_images
    ]

    final_model_inputs = None
    final_visual_tokens = 0
    final_input_len = 0
    final_text_tokens = 0

    for attempt in range(max_shrink_steps):
        model_inputs = processor(
            text=[raw_prompt],
            images=processed_images,
            return_tensors="pt",
            **processor_kwargs,
        )
        image_grid_thw = model_inputs.get("image_grid_thw")
        visual_tokens = _count_mm_visual_tokens(image_grid_thw, merge_size)
        input_len = int(model_inputs["input_ids"].shape[1])
        text_tokens = max(0, input_len - visual_tokens)

        final_model_inputs = model_inputs
        final_visual_tokens = visual_tokens
        final_input_len = input_len
        final_text_tokens = text_tokens

        if input_len <= safe_prompt_len:
            break

        visual_floor = preferred_visual_floor if attempt < max_shrink_steps - 1 else hard_visual_floor
        if visual_tokens <= visual_floor:
            break

        target_visual_tokens = max(
            visual_floor,
            min(
                visual_tokens - 32,
                int(max(visual_floor, visual_tokens * (safe_prompt_len / max(1, input_len)) * 0.97)),
            ),
        )
        if target_visual_tokens >= visual_tokens:
            target_visual_tokens = max(visual_floor, visual_tokens - 64)
        if target_visual_tokens >= visual_tokens:
            break

        current_pixels = max(1, sum(image.width * image.height for image in processed_images))
        target_pixels = max(
            min_pixels,
            min(current_pixels - 1024, int(current_pixels * (target_visual_tokens / max(1, visual_tokens)))),
        )
        if target_pixels >= current_max_pixels:
            target_pixels = max(min_pixels, current_max_pixels - 1024)
        if target_pixels >= current_max_pixels:
            break

        current_max_pixels = target_pixels
        diagnostics["shrink_steps"] = attempt + 1
        per_image_max_pixels = max(min_pixels, int(current_max_pixels / max(1, len(obs_images))))
        processed_images = [
            process_qwen3_ocr_image(
                image,
                max_pixels=per_image_max_pixels,
                min_pixels=min_pixels,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )
            for image in obs_images
        ]

    diagnostics.update(
        {
            "input_len": int(final_input_len),
            "text_tokens": int(final_text_tokens),
            "visual_tokens": int(final_visual_tokens),
        }
    )
    return processed_images, processor_kwargs, final_model_inputs, diagnostics


def _save_image(image_like, path: Path) -> Image.Image:
    if isinstance(image_like, Image.Image):
        image = image_like
    else:
        image = Image.fromarray(np.asarray(image_like).astype("uint8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image


def _build_obs_text(
    history_text: str,
    current_observation: str,
    admissible_actions: list[str],
) -> str:
    return ALFWORLD_TEMPLATE_OCR_QWEN3.format(
        task_description="put a cooled apple in fridge 1",
        goal_hints="",
        step_count=4,
        history_length=len([line for line in history_text.splitlines() if line.strip()]),
        action_history=history_text,
        history_hint="",
        failure_hint="",
        current_step=5,
        current_observation=current_observation,
        admissible_actions="\n".join(admissible_actions),
        compression_factor=1.0,
        rules=build_alfworld_rules_for_model(
            "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct",
            include_compression=True,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump the exact Qwen3 ALFWorld OCR input payload.")
    parser.add_argument(
        "--model-path",
        default="/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/qwen3_actual_input_dump",
    )
    parser.add_argument(
        "--history-file",
        default=None,
        help="Optional path to a raw history text file.",
    )
    parser.add_argument(
        "--current-observation",
        default="you are next to fridge 1 while carrying apple 1. fridge 1 is closed.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_text = (
        Path(args.history_file).read_text(encoding="utf-8").strip()
        if args.history_file
        else DEFAULT_HISTORY.strip()
    )
    admissible_actions = [
        "open fridge 1",
        "go to countertop 1",
        "go to sinkbasin 1",
    ]
    obs_text = _build_obs_text(
        history_text=history_text,
        current_observation=args.current_observation,
        admissible_actions=admissible_actions,
    )
    (out_dir / "history_raw.txt").write_text(history_text + "\n", encoding="utf-8")
    (out_dir / "obs_text_prompt.txt").write_text(obs_text + "\n", encoding="utf-8")

    highlight_configs = parse_highlight_configs(
        "[Observation]:0,0,255;[Action]:255,0,0"
    )
    ocr = OCRTool(
        enabled=True,
        font_size=12,
        padding=0,
        min_width=28,
        max_width=512,
        min_height=0,
        max_height=4096,
        max_workers=1,
        use_parallel=False,
        use_precise=True,
        enable_cache=False,
        highlight_configs=highlight_configs,
    )
    raw_render = ocr.convert_texts_to_images(
        trajectory_contexts=[history_text],
        batch_size=1,
        active_masks=[True],
        compression_factor=[1.0],
        current_steps=[5],
        enable_cache=False,
        save_img=False,
        qwen3_history_pages=False,
        qwen3_history_structured=True,
    )[0]
    raw_image = _save_image(raw_render, out_dir / "ocr_history_single_raw.png")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    obs_content = obs_text.replace("<image>\n", "", 1).replace("<image>", "", 1)
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
                    {"type": "image"},
                    {"type": "text", "text": obs_content},
                ],
            },
        ]
    else:
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": obs_content},
                ],
            }
        ]

    prompt_with_chat_template = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    (out_dir / "chat_template_prompt.txt").write_text(
        prompt_with_chat_template + "\n", encoding="utf-8"
    )

    processed_images, image_processor_kwargs, model_inputs, diagnostics = _build_qwen3_multimodal_model_inputs(
        raw_prompt=prompt_with_chat_template,
        obs_image=[np.asarray(raw_image)],
        processor=processor,
        max_prompt_length=2048,
    )
    processed_image = _save_image(processed_images[0], out_dir / "ocr_history_single_processed.png")

    image_grid_thw = model_inputs.get("image_grid_thw")
    image_grid = image_grid_thw[0].tolist() if image_grid_thw is not None else None

    summary = {
        "history_line_count": len([line for line in history_text.splitlines() if line.strip()]),
        "chat": chat,
        "raw_prompt_text_path": str(out_dir / "chat_template_prompt.txt"),
        "ocr_raw_image_size": [raw_image.width, raw_image.height],
        "ocr_processed_image_size": [processed_image.width, processed_image.height],
        "image_grid_thw": image_grid,
        "mm_budget_diagnostics": diagnostics,
        "input_token_count": int(model_inputs["input_ids"].shape[1]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
