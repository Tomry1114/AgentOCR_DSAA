#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from agentocr import OCRTool
from agent_system.environments.base import parse_highlight_configs
from agent_system.multi_turn_rollout.utils import process_image, process_qwen3_ocr_image


DEFAULT_HISTORY_TEXT = """[Observation]: you are in the kitchen.
[Action]: look
[Observation]: you notice a countertop 1 and fridge 1 nearby.
[Action]: go to countertop 1
[Observation]: you see apple 1 on countertop 1.
[Action]: take apple 1 from countertop 1
[Observation]: you are carrying apple 1.
[Action]: go to fridge 1
[Observation]: you arrive at fridge 1."""


def _load_history_text(history_file: str | None) -> str:
    if history_file:
        return Path(history_file).read_text(encoding="utf-8").strip()
    return DEFAULT_HISTORY_TEXT.strip()


def _save_array_image(array, path: Path) -> Image.Image:
    image = Image.fromarray(np.asarray(array).astype("uint8"))
    image.save(path)
    return image


def _count_visual_tokens_from_grid(image_grid_thw, merge_size: int) -> int:
    if image_grid_thw is None:
        return 0
    merge_length = max(1, int(merge_size) ** 2)
    total = 0
    for idx in range(len(image_grid_thw)):
        total += int((image_grid_thw[idx].prod() // merge_length).item())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ALFWorld OCR history layouts for Qwen2.5 vs Qwen3.")
    parser.add_argument("--history-file", default=None, help="Optional path to a raw history text file.")
    parser.add_argument(
        "--output-dir",
        default="diagnostics/ocr_layout_compare/latest",
        help="Directory to write images and summary JSON.",
    )
    parser.add_argument(
        "--qwen25-model-path",
        default="/hpc2hdd/home/rtang906/hf_models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--qwen3-model-path",
        default="/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_text = _load_history_text(args.history_file)
    (out_dir / "raw_history.txt").write_text(history_text + "\n", encoding="utf-8")

    highlight_configs = parse_highlight_configs("[Observation]:0,0,255;[Action]:255,0,0")

    qwen25_ocr = OCRTool(
        enabled=True,
        font_size=10,
        padding=0,
        min_width=28,
        max_width=392,
        min_height=0,
        max_height=4096,
        max_workers=1,
        use_parallel=False,
        use_precise=False,
        enable_cache=False,
        highlight_configs=highlight_configs,
    )
    qwen25_array = qwen25_ocr.convert_texts_to_images(
        trajectory_contexts=[history_text],
        batch_size=1,
        active_masks=[True],
        compression_factor=[1.0],
        current_steps=[1],
        enable_cache=False,
        save_img=False,
    )[0]
    qwen25_image = _save_array_image(qwen25_array, out_dir / "qwen25_single_history.png")
    qwen25_processor = AutoProcessor.from_pretrained(args.qwen25_model_path, trust_remote_code=True)
    qwen25_processed = process_image(qwen25_array)
    qwen25_processed.save(out_dir / "qwen25_single_history_processed.png")
    qwen25_grid = qwen25_processor.image_processor([qwen25_processed], return_tensors="pt").get("image_grid_thw")
    qwen25_visual_tokens = _count_visual_tokens_from_grid(
        qwen25_grid, getattr(qwen25_processor.image_processor, "merge_size", 2)
    )

    os.environ["AGENTOCR_QWEN3_OCR_MIN_PIXELS"] = "16384"
    os.environ["AGENTOCR_QWEN3_OCR_MAX_PIXELS"] = "262144"
    os.environ["AGENTOCR_QWEN3_OCR_MIN_ASPECT"] = "0.75"
    os.environ["AGENTOCR_QWEN3_OCR_MAX_ASPECT"] = "2.50"

    qwen3_ocr = OCRTool(
        enabled=True,
        font_size=12,
        padding=0,
        min_width=448,
        max_width=512,
        min_height=512,
        max_height=4096,
        max_workers=1,
        use_parallel=False,
        use_precise=True,
        enable_cache=False,
        highlight_configs=highlight_configs,
    )
    qwen3_pages = qwen3_ocr.convert_texts_to_images(
        trajectory_contexts=[history_text],
        batch_size=1,
        active_masks=[True],
        compression_factor=[1.0],
        current_steps=[1],
        enable_cache=False,
        save_img=False,
        qwen3_history_pages=True,
        use_precise=True,
        qwen3_history_page_width=448,
        qwen3_history_page_height=512,
        qwen3_history_page_padding=8,
        qwen3_history_page_gap=16,
        qwen3_history_page_budget=0,
        qwen3_history_page_columns=2,
    )[0]
    qwen3_processor = AutoProcessor.from_pretrained(args.qwen3_model_path, trust_remote_code=True)
    qwen3_patch_size = int(getattr(qwen3_processor.image_processor, "patch_size", 16) or 16)
    qwen3_temporal_patch_size = int(getattr(qwen3_processor.image_processor, "temporal_patch_size", 2) or 2)

    qwen3_processed_pages = []
    qwen3_page_infos = []
    for page_idx, page_array in enumerate(qwen3_pages, start=1):
        raw_image = _save_array_image(page_array, out_dir / f"qwen3_history_page_{page_idx:02d}.png")
        processed_image = process_qwen3_ocr_image(
            raw_image,
            patch_size=qwen3_patch_size,
            temporal_patch_size=qwen3_temporal_patch_size,
        )
        processed_image.save(out_dir / f"qwen3_history_page_{page_idx:02d}_processed.png")
        qwen3_processed_pages.append(processed_image)
        qwen3_page_infos.append(
            {
                "page_index": page_idx,
                "raw_size": [raw_image.width, raw_image.height],
                "processed_size": [processed_image.width, processed_image.height],
            }
        )

    qwen3_grid = qwen3_processor.image_processor(qwen3_processed_pages, return_tensors="pt").get("image_grid_thw")
    qwen3_merge_size = int(getattr(qwen3_processor.image_processor, "merge_size", 2) or 2)
    qwen3_visual_tokens_total = 0
    if qwen3_grid is not None:
        merge_length = max(1, qwen3_merge_size ** 2)
        for idx in range(len(qwen3_grid)):
            visual_tokens = int((qwen3_grid[idx].prod() // merge_length).item())
            qwen3_visual_tokens_total += visual_tokens
            if idx < len(qwen3_page_infos):
                qwen3_page_infos[idx]["visual_tokens"] = visual_tokens

    summary = {
        "history_line_count": len([line for line in history_text.splitlines() if line.strip()]),
        "qwen25": {
            "raw_image_size": [qwen25_image.width, qwen25_image.height],
            "processed_image_size": [qwen25_processed.width, qwen25_processed.height],
            "visual_tokens_total": qwen25_visual_tokens,
            "image_path": str(out_dir / "qwen25_single_history.png"),
            "processed_image_path": str(out_dir / "qwen25_single_history_processed.png"),
        },
        "qwen3": {
            "page_count": len(qwen3_pages),
            "pages": qwen3_page_infos,
            "visual_tokens_total": qwen3_visual_tokens_total,
            "layout_diagnostics": qwen3_ocr.get_last_qwen3_history_layout_diagnostics(),
        },
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
