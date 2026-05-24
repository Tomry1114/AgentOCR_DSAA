#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tools.qwen3_alfworld_ocr_success_smoke import (
    collect_case_observation,
    load_model,
    run_generation,
    save_case_artifacts,
    score_output,
    temporary_env,
)


DEFAULT_MODEL_PATH = "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct"
DEFAULT_OUTPUT = "outputs/debug_smoke/qwen3_alfworld_lookat_smoke.json"


def build_case() -> Dict[str, Any]:
    return {
        "case_name": "lookat_statue_light",
        "task_description": "look at the statue under the light",
        "reset_observation": "You are in the gallery. Your task is to: look at the statue under the light",
        "scripted_actions": [
            "go to statue pedestal",
            "toggle desk lamp",
            "examine side table",
        ],
        "step_observations": [
            "You reach the statue pedestal. The statue is here, but it is too dark to inspect details.",
            "You flip the desk lamp switch. The switch is on and the pedestal is illuminated.",
            "You inspect the side table. A second statue brochure is on the side table.",
        ],
        "final_admissible": [
            "examine statue with lamp on",
            "examine statue in the dark",
            "read side-table brochure",
            "look",
        ],
        "ground_truth_action": "examine statue with lamp on",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3 ALFWorld look-at-in-light OCR behavior.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()

    output_path = Path(args.output)
    artifact_root = output_path.parent / f"{output_path.stem}_artifacts"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor, model = load_model(args.model_path, args.device)
    task = build_case()
    rows: List[Dict[str, Any]] = []

    env_overrides = {
        "AGENTOCR_QWEN3_HISTORY_PAGES": "0",
    }
    with temporary_env(env_overrides):
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
