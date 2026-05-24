#!/usr/bin/env python3
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_ACTION_RULES,
    ALFWORLD_COMPRESSION_RULE,
    ALFWORLD_QWEN3VL_SYSTEM_PROMPT,
    ALFWORLD_TEMPLATE_OCR,
)
from agent_system.multi_turn_rollout.rollout_loop import (
    _is_qwen3_vl_processor,
)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None


DEFAULT_MODELS = [
    "/hpc2hdd/home/rtang906/hf_models/Qwen2.5-VL-3B-Instruct",
    "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-2B-Instruct",
    "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-2B-Thinking",
]


@dataclass
class Case:
    name: str
    task_description: str
    step_count: int
    current_step: int
    current_observation: str
    admissible_actions: List[str]
    target_keywords: List[str]
    distractor_keywords: List[str]
    history_note: str = "History is intentionally omitted in this probe to isolate goal anchoring from local distractors."


CASES = [
    Case(
        name="box_vs_book",
        task_description="put some box on dresser.",
        step_count=2,
        current_step=3,
        current_observation="You pick up the book 1 from the dresser 1.",
        admissible_actions=[
            "examine book 1",
            "examine dresser 1",
            "go to armchair 1",
            "go to cabinet 1",
            "go to drawer 1",
            "go to drawer 10",
            "go to drawer 11",
            "go to drawer 12",
            "go to drawer 13",
            "go to drawer 14",
            "go to drawer 15",
            "go to drawer 16",
            "go to drawer 17",
            "go to drawer 18",
            "go to drawer 19",
            "go to drawer 2",
            "go to drawer 20",
            "go to drawer 21",
            "go to drawer 3",
            "go to drawer 4",
            "go to drawer 5",
            "go to drawer 6",
            "go to drawer 7",
            "go to drawer 8",
            "go to drawer 9",
            "go to garbagecan 1",
            "go to sidetable 1",
            "go to sidetable 2",
            "go to sidetable 3",
            "go to sidetable 4",
            "go to sidetable 5",
            "go to sofa 1",
            "inventory",
            "look",
            "move book 1 to dresser 1",
        ],
        target_keywords=["box", "drawer"],
        distractor_keywords=["book"],
    ),
    Case(
        name="fork_vs_peppershaker",
        task_description="clean some fork and put it in drawer.",
        step_count=6,
        current_step=7,
        current_observation="You move the peppershaker 2 to the cabinet 1.",
        admissible_actions=[
            "close cabinet 1",
            "examine cabinet 1",
            "go to cabinet 10",
            "go to cabinet 11",
            "go to cabinet 12",
            "go to cabinet 13",
            "go to cabinet 14",
            "go to cabinet 15",
            "go to cabinet 16",
            "go to cabinet 17",
            "go to cabinet 18",
            "go to cabinet 2",
            "go to cabinet 3",
            "go to cabinet 4",
            "go to cabinet 5",
            "go to cabinet 6",
            "go to cabinet 7",
            "go to cabinet 8",
            "go to cabinet 9",
            "go to coffeemachine 1",
            "go to countertop 1",
            "go to countertop 2",
            "go to countertop 3",
            "go to countertop 4",
            "go to drawer 1",
            "go to drawer 2",
            "go to drawer 3",
            "go to drawer 4",
            "go to drawer 5",
            "go to drawer 6",
            "go to drawer 7",
            "go to fridge 1",
            "go to garbagecan 1",
            "go to microwave 1",
            "go to sinkbasin 1",
            "go to stoveburner 1",
            "go to stoveburner 2",
            "go to stoveburner 3",
            "go to stoveburner 4",
            "go to toaster 1",
            "inventory",
            "look",
            "take peppershaker 2 from cabinet 1",
        ],
        target_keywords=["fork", "sinkbasin", "drawer"],
        distractor_keywords=["peppershaker"],
    ),
    Case(
        name="alarmclock_vs_generic_look",
        task_description="look at alarmclock under the desklamp.",
        step_count=13,
        current_step=14,
        current_observation="You are in the middle of a room. Looking quickly around you, you see nothing.",
        admissible_actions=[
            "go to bed 1",
            "go to bed 2",
            "go to desk 1",
            "go to drawer 1",
            "go to drawer 10",
            "go to drawer 11",
            "go to drawer 2",
            "go to drawer 3",
            "go to drawer 4",
            "go to drawer 5",
            "go to drawer 6",
            "go to drawer 7",
            "go to drawer 8",
            "go to drawer 9",
            "go to dresser 1",
            "go to garbagecan 1",
            "go to safe 1",
            "go to sidetable 1",
            "go to sidetable 2",
            "inventory",
            "look",
        ],
        target_keywords=["alarmclock", "desk", "desklamp"],
        distractor_keywords=["look"],
    ),
]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _build_current_prompt(case: Case) -> str:
    admissible_actions = "\n".join(case.admissible_actions)
    return ALFWORLD_TEMPLATE_OCR.format(
        task_description=case.task_description,
        step_count=case.step_count,
        history_length=case.step_count,
        action_history=case.history_note,
        current_step=case.current_step,
        current_observation=case.current_observation,
        admissible_actions=admissible_actions,
        rules=f"{ALFWORLD_ACTION_RULES}\n{ALFWORLD_COMPRESSION_RULE}",
    )


def _build_goal_explicit_prompt(case: Case) -> str:
    admissible_actions = "\n".join(case.admissible_actions)
    target_keywords = ", ".join(case.target_keywords)
    distractors = ", ".join(case.distractor_keywords) if case.distractor_keywords else "None"
    return f"""
You are an expert agent operating in the ALFRED Embodied Environment.

Task:
{case.task_description}

Current step:
{case.current_step}

Current textual observation:
{case.current_observation}

Goal-critical keywords:
{target_keywords}

Likely distractor keywords:
{distractors}

Rules:
1. Prefer actions that help achieve the task goal, even if another visible object is immediately interactable.
2. If the visible/interactable object does not match the goal object, prefer exploration actions that help find the goal object.
3. Choose exactly one action copied verbatim from the admissible list.
4. Put brief reasoning inside <thinking>...</thinking>.
5. Put the final action inside <action>...</action>.
6. Put the next image compression factor inside <compression>...</compression>.

Admissible actions:
{admissible_actions}
""".strip()


def _build_chat(processor, prompt_text: str):
    if _is_qwen3_vl_processor(processor):
        chat = [
            {
                "role": "system",
                "content": [{"type": "text", "text": ALFWORLD_QWEN3VL_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, None

    chat = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, None


def _load_model(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("Qwen3VLForConditionalGeneration is unavailable in this environment.")
        model_cls = Qwen3VLForConditionalGeneration
    elif model_type == "qwen2_5_vl":
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError("Qwen2_5_VLForConditionalGeneration is unavailable in this environment.")
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model_type={model_type} for {model_path}")

    return model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    )


def _extract_action(text: str, action_pool: List[str]) -> Optional[str]:
    actions, valids = alfworld_projection([text], [action_pool])
    if valids[0]:
        return actions[0]
    lower_text = text.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = lower_text.find(start_tag)
    end_idx = lower_text.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx + len(start_tag):end_idx].strip()
        if candidate in action_pool:
            return candidate
    return None


def _score_action(action: Optional[str], case: Case) -> dict:
    normalized_action = _normalize_whitespace(action or "")
    target_hit = any(keyword.lower() in normalized_action for keyword in case.target_keywords)
    distractor_hit = any(keyword.lower() in normalized_action for keyword in case.distractor_keywords)
    generic_look = normalized_action == "look"
    return {
        "target_keyword_hit": target_hit,
        "distractor_keyword_hit": distractor_hit,
        "generic_look": generic_look,
    }


def _run_case(model, processor, case: Case, prompt_variant: str, image: Image.Image, max_tokens: int) -> dict:
    if prompt_variant == "current":
        prompt_text = _build_current_prompt(case)
    elif prompt_variant == "goal_explicit":
        prompt_text = _build_goal_explicit_prompt(case)
    else:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant}")

    prompt, response_prefix_text = _build_chat(processor, prompt_text)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    output_tokens = generated[0][input_len:]
    raw_text = processor.decode(output_tokens, skip_special_tokens=False)
    full_text = f"{response_prefix_text or ''}{raw_text}"
    action = _extract_action(full_text, case.admissible_actions)
    scores = _score_action(action, case)
    return {
        "case": case.name,
        "prompt_variant": prompt_variant,
        "prompt_preview": prompt_text[:1200],
        "raw_text": raw_text,
        "full_text": full_text,
        "action": action,
        "valid": action is not None and action in case.admissible_actions,
        **scores,
    }


def main() -> None:
    model_paths = [path for path in os.environ.get("MODEL_PATHS", ":".join(DEFAULT_MODELS)).split(":") if path]
    prompt_variants = [item for item in os.environ.get("PROMPT_VARIANTS", "current").split(",") if item]
    max_tokens = int(os.environ.get("MAX_TOKENS", "96"))
    image_size = int(os.environ.get("IMAGE_SIZE", "448"))
    image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))

    all_results = []
    for model_path in model_paths:
        print(f"\n=== MODEL {model_path} ===", flush=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        model = _load_model(model_path)
        model.eval()
        try:
            for prompt_variant in prompt_variants:
                print(f"\n--- PROMPT_VARIANT {prompt_variant} ---", flush=True)
                for case in CASES:
                    result = _run_case(
                        model=model,
                        processor=processor,
                        case=case,
                        prompt_variant=prompt_variant,
                        image=image,
                        max_tokens=max_tokens,
                    )
                    result["model_path"] = model_path
                    all_results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== SUMMARY TABLE ===", flush=True)
    for result in all_results:
        print(
            json.dumps(
                {
                    "model": os.path.basename(result["model_path"]),
                    "prompt_variant": result["prompt_variant"],
                    "case": result["case"],
                    "action": result["action"],
                    "valid": result["valid"],
                    "target_keyword_hit": result["target_keyword_hit"],
                    "distractor_keyword_hit": result["distractor_keyword_hit"],
                    "generic_look": result["generic_look"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
