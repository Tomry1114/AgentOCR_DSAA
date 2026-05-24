#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from agentocr import OCRTool
from agentocr.utils import _get_cached_font, get_font_metrics, wrap_text_fast, wrap_text_precise
from agent_system.environments.base import parse_highlight_configs
from agent_system.environments.prompts.alfworld import ALFWORLD_QWEN3VL_SYSTEM_PROMPT
from agent_system.multi_turn_rollout.utils import process_qwen3_ocr_image


CASES: Dict[str, Dict[str, Any]] = {
    "fridge_carry": {
        "task_description": "put a cooled apple in fridge 1",
        "history_text": (
            "[Observation]: you are in the kitchen.\n"
            "[Action]: look\n"
            "[Observation]: you notice a countertop 1 and fridge 1 nearby.\n"
            "[Action]: go to countertop 1\n"
            "[Observation]: you see apple 1 on countertop 1.\n"
            "[Action]: take apple 1 from countertop 1\n"
            "[Observation]: you are carrying apple 1.\n"
            "[Action]: go to fridge 1\n"
            "[Observation]: you arrive at fridge 1."
        ),
        "current_observation": "you are next to fridge 1 while carrying apple 1. fridge 1 is closed.",
        "qa": [
            {
                "name": "latest_observation",
                "question": "According to the history image, what was the latest observation?",
                "expected": "you arrive at fridge 1",
            },
            {
                "name": "last_action",
                "question": "According to the history image, what was the last action?",
                "expected": "go to fridge 1",
            },
            {
                "name": "inventory",
                "question": "According to the history image, what object is the agent carrying or holding?",
                "expected": "apple 1",
            },
            {
                "name": "recent_room",
                "question": "According to the history image, what location was visited most recently?",
                "expected": "fridge 1",
            },
        ],
    },
    "sink_clean": {
        "task_description": "put a clean mug in cabinet 1",
        "history_text": (
            "[Observation]: you are in the kitchen.\n"
            "[Action]: go to diningtable 1\n"
            "[Observation]: you see mug 1 on diningtable 1.\n"
            "[Action]: take mug 1 from diningtable 1\n"
            "[Observation]: you are carrying mug 1.\n"
            "[Action]: go to sinkbasin 1\n"
            "[Observation]: you arrive at sinkbasin 1."
        ),
        "current_observation": "you are next to sinkbasin 1 while carrying mug 1.",
        "qa": [
            {
                "name": "latest_observation",
                "question": "According to the history image, what was the latest observation?",
                "expected": "you arrive at sinkbasin 1",
            },
            {
                "name": "last_action",
                "question": "According to the history image, what was the last action?",
                "expected": "go to sinkbasin 1",
            },
            {
                "name": "inventory",
                "question": "According to the history image, what object is the agent carrying or holding?",
                "expected": "mug 1",
            },
            {
                "name": "recent_room",
                "question": "According to the history image, what location was visited most recently?",
                "expected": "sinkbasin 1",
            },
        ],
    },
}


def _extract_prefixed_text(line: str, prefix: str) -> str:
    marker = f"[{prefix}]:"
    if line.startswith(marker):
        return line[len(marker):].strip()
    return line.strip()


def _history_lines(history_text: str) -> List[str]:
    return [line.strip() for line in str(history_text or "").splitlines() if line.strip()]


def _pair_history_steps(history_text: str) -> List[List[Tuple[str, str]]]:
    steps: List[List[Tuple[str, str]]] = []
    current_step: List[Tuple[str, str]] = []
    for raw_line in _history_lines(history_text):
        if raw_line.startswith("[Observation]:"):
            if current_step:
                steps.append(current_step)
            current_step = [("Observation", _extract_prefixed_text(raw_line, "Observation"))]
        elif raw_line.startswith("[Action]:"):
            if not current_step:
                current_step = []
            current_step.append(("Action", _extract_prefixed_text(raw_line, "Action")))
        else:
            if not current_step:
                current_step = []
            current_step.append(("Text", raw_line))
    if current_step:
        steps.append(current_step)
    return steps


def _format_layout(history_text: str, layout: str, recent_window: int) -> str:
    if layout == "plain":
        return history_text.strip()

    steps = _pair_history_steps(history_text)
    if not steps:
        return history_text.strip()

    if layout == "step_blocks":
        lines: List[str] = []
        for idx, step in enumerate(steps, start=1):
            lines.append(f"[STEP {idx:02d}]")
            for tag, text in step:
                if tag in ("Observation", "Action"):
                    lines.append(f"[{tag}]: {text}")
                else:
                    lines.append(text)
        return "\n".join(lines).strip()

    if layout in {"sectioned", "recent_first"}:
        split_idx = max(0, len(steps) - max(1, recent_window))
        older_steps = steps[:split_idx]
        recent_steps = steps[split_idx:]

        ordered_sections = []
        if layout == "recent_first":
            ordered_sections = [
                ("[RECENT HISTORY]", recent_steps),
                ("[OLDER HISTORY]", older_steps),
            ]
        else:
            ordered_sections = [
                ("[OLDER HISTORY]", older_steps),
                ("[RECENT HISTORY]", recent_steps),
            ]

        lines = []
        display_idx = 1
        for header, section_steps in ordered_sections:
            if not section_steps:
                continue
            lines.append(header)
            for step in section_steps:
                lines.append(f"[STEP {display_idx:02d}]")
                for tag, text in step:
                    if tag in ("Observation", "Action"):
                        lines.append(f"[{tag}]: {text}")
                    else:
                        lines.append(text)
                display_idx += 1
        return "\n".join(lines).strip()

    raise ValueError(f"Unsupported layout: {layout}")


def _build_ocr_tool(args) -> OCRTool:
    highlight_configs = parse_highlight_configs(
        os.environ.get("HIGHLIGHT_CONFIGS", "[Observation]:0,0,255;[Action]:255,0,0")
    )
    return OCRTool(
        enabled=True,
        font_size=args.font_size,
        padding=args.padding,
        min_width=28,
        max_width=args.max_width,
        min_height=28,
        max_height=args.max_height,
        max_workers=1,
        use_parallel=False,
        use_precise=args.use_precise,
        enable_cache=False,
        highlight_configs=highlight_configs,
    )


def _save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _build_diagnostic_chat(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        ALFWORLD_QWEN3VL_SYSTEM_PROMPT
                        + " Read the provided history image carefully and answer only from the image."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def _prepare_processed_image(raw_image: Image.Image, processor) -> Tuple[Image.Image, Dict[str, Any]]:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
    temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2) or 2)
    processed_image = process_qwen3_ocr_image(
        raw_image,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
    )
    return processed_image, {"do_resize": False}


def _collect_text_stats(history_text: str) -> Dict[str, Any]:
    lines = _history_lines(history_text)
    line_lengths = [len(line) for line in lines]
    return {
        "history_line_count": len(lines),
        "avg_chars_per_line": (sum(line_lengths) / len(line_lengths)) if line_lengths else 0.0,
        "max_chars_per_line": max(line_lengths) if line_lengths else 0,
        "observation_line_count": sum(1 for line in lines if line.startswith("[Observation]:")),
        "action_line_count": sum(1 for line in lines if line.startswith("[Action]:")),
    }


def _estimate_render_stats(
    history_text: str,
    args,
    target_latest_line: str | None = None,
    target_earliest_line: str | None = None,
) -> Dict[str, Any]:
    font = _get_cached_font(None, args.font_size)
    avg_char_width, line_height = get_font_metrics(font, args.font_size)
    available_width = args.max_width - 2 * args.padding
    if args.use_precise:
        wrapped_lines = wrap_text_precise(history_text, available_width, font, args.font_size)
        max_chars_per_line = None
    else:
        max_chars_per_line = int(available_width / avg_char_width)
        wrapped_lines = wrap_text_fast(history_text, max_chars_per_line)

    required_height = len(wrapped_lines) * line_height + 2 * args.padding
    available_height = max(0, args.max_height - 2 * args.padding)
    max_lines = int(available_height / line_height) if line_height > 0 else len(wrapped_lines)
    overflow_flag = required_height > args.max_height
    kept_lines = wrapped_lines[:max_lines] if overflow_flag else wrapped_lines
    dropped_lines = max(0, len(wrapped_lines) - len(kept_lines))

    source_lines = _history_lines(history_text)
    latest_source_line = target_latest_line or (source_lines[-1] if source_lines else "")
    earliest_source_line = target_earliest_line or (source_lines[0] if source_lines else "")
    kept_text = "\n".join(line for line, _ in kept_lines)

    return {
        "available_width": available_width,
        "avg_char_width": avg_char_width,
        "line_height": line_height,
        "max_chars_per_line": max_chars_per_line,
        "wrapped_line_count": len(wrapped_lines),
        "required_height_before_clamp": required_height,
        "max_render_lines": max_lines,
        "overflow_flag": overflow_flag,
        "kept_wrapped_line_count": len(kept_lines),
        "dropped_wrapped_line_count": dropped_lines,
        "latest_source_line_prefix": latest_source_line[:80],
        "latest_source_line_visible_in_kept_text": latest_source_line[:40] in kept_text if latest_source_line else False,
        "earliest_source_line_prefix": earliest_source_line[:80],
        "earliest_source_line_visible_in_kept_text": earliest_source_line[:40] in kept_text if earliest_source_line else False,
    }


def _repeat_history_prefix(history_text: str, prefix_lines: int, repeat_count: int) -> str:
    lines = _history_lines(history_text)
    if repeat_count <= 1 or prefix_lines <= 0 or len(lines) <= prefix_lines:
        return history_text.strip()
    prefix = lines[:prefix_lines]
    remainder = lines[prefix_lines:]
    expanded = prefix * repeat_count + remainder
    return "\n".join(expanded).strip()


def _run_single_question(
    processor,
    model,
    image: Image.Image,
    image_processor_kwargs: Dict[str, Any],
    question_spec: Dict[str, str],
    max_new_tokens: int,
) -> Dict[str, Any]:
    chat = _build_diagnostic_chat(question_spec["question"])
    prompt = processor.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_token_count = int(processor.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
    model_inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        **image_processor_kwargs,
    )
    input_ids = model_inputs["input_ids"]
    total_input_tokens = int(input_ids.shape[1])
    image_grid_thw = model_inputs.get("image_grid_thw")
    visual_tokens = None
    if image_grid_thw is not None:
        grid = image_grid_thw[0].tolist()
        visual_tokens = int(grid[0] * grid[1] * grid[2])
    else:
        grid = None

    model_inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    output_tokens = generated[0][input_ids.shape[1]:]
    answer = processor.decode(output_tokens, skip_special_tokens=False).strip()

    return {
        "name": question_spec["name"],
        "question": question_spec["question"],
        "expected": question_spec["expected"],
        "answer": answer,
        "prompt_token_count": prompt_token_count,
        "total_input_tokens": total_input_tokens,
        "image_grid_thw": grid,
        "visual_tokens": visual_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Qwen3 OCR history image readability.")
    parser.add_argument("--case", default="fridge_carry", choices=sorted(CASES.keys()))
    parser.add_argument("--layout", default="plain", choices=["plain", "step_blocks", "sectioned", "recent_first"])
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--prefix-lines", type=int, default=3)
    parser.add_argument("--prefix-repeat", type=int, default=1)
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--max-height", type=int, default=4096)
    parser.add_argument("--use-precise", action="store_true")
    parser.add_argument("--qwen3-history-pages", action="store_true")
    parser.add_argument("--qwen3-page-width", type=int, default=448)
    parser.add_argument("--qwen3-page-height", type=int, default=512)
    parser.add_argument("--qwen3-page-padding", type=int, default=10)
    parser.add_argument("--qwen3-page-gap", type=int, default=24)
    parser.add_argument("--qwen3-page-budget", type=int, default=4)
    parser.add_argument("--qwen3-page-columns", type=int, default=2)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="diagnostics/qwen3_ocr_history")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct"),
    )
    args = parser.parse_args()

    case = CASES[args.case]
    repeated_history = _repeat_history_prefix(
        case["history_text"],
        prefix_lines=args.prefix_lines,
        repeat_count=args.prefix_repeat,
    )
    layout_history = _format_layout(repeated_history, args.layout, args.recent_window)
    text_stats = _collect_text_stats(layout_history)
    repeated_history_lines = _history_lines(repeated_history)
    render_stats = _estimate_render_stats(
        layout_history,
        args,
        target_latest_line=repeated_history_lines[-1] if repeated_history_lines else "",
        target_earliest_line=repeated_history_lines[0] if repeated_history_lines else "",
    )

    run_name = (
        f"rep{args.prefix_repeat}_pref{args.prefix_lines}"
        f"_w{args.max_width}_fs{args.font_size}"
        f"_prec{int(bool(args.use_precise))}"
    )
    output_dir = Path(args.output_dir) / args.case / args.layout / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "history_layout.txt"
    text_path.write_text(layout_history + "\n", encoding="utf-8")

    ocr = _build_ocr_tool(args)
    raw_array = ocr.convert_texts_to_images(
        trajectory_contexts=[layout_history],
        batch_size=1,
        active_masks=[True],
        compression_factor=[1.0],
        current_steps=[1],
        enable_cache=False,
        save_img=False,
        qwen3_history_pages=args.qwen3_history_pages,
        use_precise=args.use_precise or args.qwen3_history_pages,
        qwen3_history_page_width=args.qwen3_page_width,
        qwen3_history_page_height=args.qwen3_page_height,
        qwen3_history_page_padding=args.qwen3_page_padding,
        qwen3_history_page_gap=args.qwen3_page_gap,
        qwen3_history_page_budget=args.qwen3_page_budget,
        qwen3_history_page_columns=args.qwen3_page_columns,
    )[0]
    raw_image = Image.fromarray(raw_array.astype("uint8"))
    raw_path = output_dir / "raw_history.png"
    _save_image(raw_image, raw_path)

    result: Dict[str, Any] = {
        "case": args.case,
        "layout": args.layout,
        "prefix_lines": args.prefix_lines,
        "prefix_repeat": args.prefix_repeat,
        "task_description": case["task_description"],
        "text_stats": text_stats,
        "render_stats": render_stats,
        "raw_image_width": raw_image.width,
        "raw_image_height": raw_image.height,
        "raw_image_pixels": raw_image.width * raw_image.height,
        "raw_image_path": str(raw_path),
        "history_layout_path": str(text_path),
        "qwen3_history_pages": bool(args.qwen3_history_pages),
        "layout_diagnostics": ocr.get_last_qwen3_history_layout_diagnostics(),
    }

    if args.stats_only:
        result_path = output_dir / "diagnostic_stats.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    processor = AutoProcessor.from_pretrained(args.model_path)
    processed_image, image_processor_kwargs = _prepare_processed_image(raw_image, processor)
    processed_path = output_dir / "processed_history.png"
    _save_image(processed_image, processed_path)

    result.update(
        {
            "processed_image_width": processed_image.width,
            "processed_image_height": processed_image.height,
            "processed_image_pixels": processed_image.width * processed_image.height,
            "processed_image_path": str(processed_path),
        }
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    qa_results = []
    for question_spec in case["qa"]:
        qa_results.append(
            _run_single_question(
                processor=processor,
                model=model,
                image=processed_image,
                image_processor_kwargs=image_processor_kwargs,
                question_spec=question_spec,
                max_new_tokens=args.max_new_tokens,
            )
        )

    result["qa_results"] = qa_results
    result_path = output_dir / "diagnostic_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
