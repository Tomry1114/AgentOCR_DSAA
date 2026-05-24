#!/usr/bin/env python3
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.prompts.alfworld import ALFWORLD_QWEN3VL_SYSTEM_PROMPT


CASES = [
    {
        "name": "pickup_from_counter",
        "task_description": "put a cooled apple in fridge 1",
        "step_count": 2,
        "history_length": 2,
        "action_history": (
            "[Observation]: you are in the kitchen.\n"
            "[Action]: go to countertop 1\n"
            "[Observation]: you see an apple 1 on the countertop."
        ),
        "current_step": 3,
        "current_observation": "you see apple 1 on countertop 1 and fridge 1 nearby.",
        "action_pool": ["go to fridge 1", "open fridge 1", "go to countertop 1"],
    },
    {
        "name": "open_fridge_after_arrival",
        "task_description": "put a cooled apple in fridge 1",
        "step_count": 4,
        "history_length": 4,
        "action_history": (
            "[Observation]: you are in the kitchen.\n"
            "[Action]: go to countertop 1\n"
            "[Observation]: you see an apple 1 on the countertop.\n"
            "[Action]: take apple 1 from countertop 1\n"
            "[Observation]: you are carrying apple 1.\n"
            "[Action]: go to fridge 1"
        ),
        "current_step": 5,
        "current_observation": "you are next to fridge 1 while carrying apple 1. fridge 1 is closed.",
        "action_pool": ["open fridge 1", "go to countertop 1", "put apple 1 in/on fridge 1"],
    },
    {
        "name": "move_to_sink",
        "task_description": "put a clean mug in cabinet 1",
        "step_count": 2,
        "history_length": 2,
        "action_history": (
            "[Observation]: you are in the kitchen.\n"
            "[Action]: go to dining table 1\n"
            "[Observation]: you see mug 1 on dining table 1."
        ),
        "current_step": 3,
        "current_observation": "you see mug 1 on dining table 1. sinkbasin 1 is nearby.",
        "action_pool": ["take mug 1 from dining table 1", "go to sinkbasin 1", "go to cabinet 1"],
    },
]


def build_prompt_text(case: dict, action_list_style: str) -> tuple[str, list[str]]:
    if action_list_style == "quoted_lines":
        admissible_actions = "\n ".join(f"'{action}'" for action in case["action_pool"])
    elif action_list_style == "plain_lines":
        admissible_actions = "\n".join(case["action_pool"])
    else:
        raise ValueError(f"Unsupported ACTION_LIST_STYLE={action_list_style}")
    prompt_text = f"""
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {case['task_description']}

Prior to this step, you have already taken {case['step_count']} step(s). The provided image shows the most recent {case['history_length']} observations and the corresponding actions you took.

You are now at step {case['current_step']} and your current textual observation is: {case['current_observation']}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thinking> </thinking> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
""".strip()
    return prompt_text, case["action_pool"]


def main() -> None:
    model_path = os.environ.get(
        "MODEL_PATH", "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-2B-Thinking"
    )
    action_list_style = os.environ.get("ACTION_LIST_STYLE", "quoted_lines")
    max_tokens = int(os.environ.get("MAX_TOKENS", "48"))

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    image = Image.new("RGB", (448, 448), color=(255, 255, 255))

    valid_count = 0
    for case in CASES:
        prompt_text, action_pool = build_prompt_text(case, action_list_style=action_list_style)
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
        text = processor.decode(output_tokens, skip_special_tokens=False)
        actions, valids = alfworld_projection([text], [action_pool])
        valid = valids[0]
        valid_count += valid
        print(f"CASE {case['name']}")
        print("RAW_OUTPUT", repr(text))
        print("PARSED_ACTION", actions[0])
        print("VALID", valid)
        print("---")

    print(f"SUMMARY valid={valid_count}/{len(CASES)} action_list_style={action_list_style} max_tokens={max_tokens}")


if __name__ == "__main__":
    main()
