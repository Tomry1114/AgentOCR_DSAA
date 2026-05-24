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
DEFAULT_OUTPUT = "outputs/debug_smoke/qwen3_alfworld_weakfam_smoke.json"


def _with_fillers(task: Dict[str, Any], filler_prefix: str, filler_count: int = 5) -> Dict[str, Any]:
    task = dict(task)
    scripted_actions = list(task["scripted_actions"])
    step_observations = list(task["step_observations"])
    for index in range(filler_count):
        place = f"{filler_prefix} filler locker {index + 1}"
        obj = f"{filler_prefix} filler object {index + 1}"
        scripted_actions.append(f"examine {place}")
        step_observations.append(f"You inspect the {place}. The {obj} is in the {place}.")
    task["scripted_actions"] = scripted_actions
    task["step_observations"] = step_observations
    return task


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    cases.append(
        _with_fillers(
            {
                "case_name": "pick_and_place_search",
                "task_family": "pick_and_place",
                "task_description": "put the apple in the fridge",
                "reset_observation": "You are in the kitchen. Your task is to: put the apple in the fridge",
                "scripted_actions": [
                    "examine blue crate",
                    "examine red crate",
                ],
                "step_observations": [
                    "You inspect the blue crate. The apple is in the blue crate.",
                    "You inspect the red crate. The banana is in the red crate.",
                ],
                "final_admissible": [
                    "open blue crate",
                    "open red crate",
                    "inventory",
                    "look",
                ],
                "ground_truth_action": "open blue crate",
            },
            filler_prefix="pick_and_place_search",
        )
    )
    cases.append(
        _with_fillers(
            {
                "case_name": "pick_two_progress",
                "task_family": "pick_two_obj_and_place",
                "task_description": "put the apple and knife in the drawer",
                "reset_observation": "You are in the kitchen. Your task is to: put the apple and knife in the drawer",
                "scripted_actions": [
                    "take apple from counter",
                    "open drawer",
                    "put apple in drawer",
                    "examine knife rack",
                    "examine fruit bowl",
                ],
                "step_observations": [
                    "You pick up the apple from the counter.",
                    "The drawer is open.",
                    "You put the apple in the drawer.",
                    "You inspect the knife rack. The knife is on the knife rack.",
                    "You inspect the fruit bowl. Another apple is in the fruit bowl.",
                ],
                "final_admissible": [
                    "take knife from knife rack",
                    "take apple from fruit bowl",
                    "inventory",
                    "look",
                ],
                "ground_truth_action": "take knife from knife rack",
            },
            filler_prefix="pick_two_progress",
        )
    )
    cases.append(
        _with_fillers(
            {
                "case_name": "pick_two_holding",
                "task_family": "pick_two_obj_and_place",
                "task_description": "put the apple and knife in the drawer",
                "reset_observation": "You are in the kitchen. Your task is to: put the apple and knife in the drawer",
                "scripted_actions": [
                    "take apple from counter",
                    "open drawer",
                    "put apple in drawer",
                    "take knife from knife rack",
                    "examine spoon rack",
                ],
                "step_observations": [
                    "You pick up the apple from the counter.",
                    "The drawer is open.",
                    "You put the apple in the drawer.",
                    "You take the knife from the knife rack.",
                    "You inspect the spoon rack while carrying the knife. A spoon is on the spoon rack.",
                ],
                "final_admissible": [
                    "put knife in open drawer",
                    "take spoon from spoon rack",
                    "open drawer",
                    "inventory",
                ],
                "ground_truth_action": "put knife in open drawer",
            },
            filler_prefix="pick_two_holding",
        )
    )
    cases.append(
        _with_fillers(
            {
                "case_name": "lookat_state_ready",
                "task_family": "look_at_obj_in_light",
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
            },
            filler_prefix="lookat_state_ready",
        )
    )
    cases.append(
        _with_fillers(
            {
                "case_name": "heat_post_state",
                "task_family": "pick_heat_then_place_in_recep",
                "task_description": "heat the potato then place it in the serving bowl",
                "reset_observation": "You are in the kitchen. Your task is to: heat the potato then place it in the serving bowl",
                "scripted_actions": [
                    "take potato from counter",
                    "heat potato with microwave",
                    "examine produce shelf",
                ],
                "step_observations": [
                    "You pick up the potato from the counter.",
                    "A moment later, it is steaming.",
                    "You inspect the produce shelf. Another potato is on the produce shelf.",
                ],
                "final_admissible": [
                    "put steaming potato in serving bowl",
                    "put room-temperature potato in serving bowl",
                    "take potato from produce shelf",
                    "inventory",
                ],
                "ground_truth_action": "put steaming potato in serving bowl",
            },
            filler_prefix="heat_post_state",
        )
    )
    cases.append(
        _with_fillers(
            {
                "case_name": "cool_post_state",
                "task_family": "pick_cool_then_place_in_recep",
                "task_description": "cool the soda then place it in the ice bucket",
                "reset_observation": "You are in the kitchen. Your task is to: cool the soda then place it in the ice bucket",
                "scripted_actions": [
                    "take soda from pantry",
                    "cool soda in fridge",
                    "examine drinks shelf",
                ],
                "step_observations": [
                    "You take the soda from the pantry.",
                    "After waiting, it is chilled.",
                    "You inspect the drinks shelf. Another soda is on the drinks shelf.",
                ],
                "final_admissible": [
                    "put chilled soda in ice bucket",
                    "put room-temperature soda in ice bucket",
                    "take soda from drinks shelf",
                    "inventory",
                ],
                "ground_truth_action": "put chilled soda in ice bucket",
            },
            filler_prefix="cool_post_state",
        )
    )
    cases.append(
        _with_fillers(
            {
                "case_name": "clean_post_state",
                "task_family": "pick_clean_then_place_in_recep",
                "task_description": "clean the mug then place it on the drying rack",
                "reset_observation": "You are in the kitchen. Your task is to: clean the mug then place it on the drying rack",
                "scripted_actions": [
                    "take mug from sink",
                    "clean mug with faucet",
                    "examine cabinet shelf",
                ],
                "step_observations": [
                    "You take the mug from the sink.",
                    "Water runs over it until it is rinsed.",
                    "You inspect the cabinet shelf. Another mug is on the cabinet shelf.",
                ],
                "final_admissible": [
                    "put rinsed mug on drying rack",
                    "put dirty mug on drying rack",
                    "take mug from cabinet shelf",
                    "inventory",
                ],
                "ground_truth_action": "put rinsed mug on drying rack",
            },
            filler_prefix="clean_post_state",
        )
    )
    return cases


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


def summarize_by_family(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    families = sorted({str(row["task_family"]) for row in rows})
    summary: List[Dict[str, Any]] = []
    for family in families:
        family_rows = [row for row in rows if row["task_family"] == family]
        for variant in ("baseline", "idea"):
            variant_rows = [row for row in family_rows if row["variant"] == variant]
            success_count = sum(1 for row in variant_rows if row["success"])
            visual_tokens = [float(row["mm_diagnostics"]["visual_tokens"]) for row in variant_rows]
            summary.append(
                {
                    "task_family": family,
                    "variant": variant,
                    "num_cases": len(variant_rows),
                    "success_rate": (success_count / len(variant_rows)) if variant_rows else 0.0,
                    "success_count": success_count,
                    "mean_visual_tokens": (sum(visual_tokens) / len(visual_tokens)) if visual_tokens else 0.0,
                }
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen3 ALFWorld weak families with OCR baseline vs idea.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=160)
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
                        "task_family": task["task_family"],
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
            "by_family": summarize_by_family(rows),
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote smoke results to {output_path}")


if __name__ == "__main__":
    main()
