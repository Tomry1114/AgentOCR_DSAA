#!/usr/bin/env python3
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from vllm import LLM, SamplingParams

from agentocr import OCRTool
from agent_system.environments.base import parse_highlight_configs
from agent_system.environments.env_package.alfworld.projection import alfworld_projection
from agent_system.environments.prompts.alfworld import (
    ALFWORLD_ACTION_RULES,
    ALFWORLD_COMPRESSION_RULE,
    ALFWORLD_QWEN3VL_SYSTEM_PROMPT,
    ALFWORLD_TEMPLATE_NO_HIS_OCR,
    ALFWORLD_TEMPLATE_OCR,
)
from agent_system.multi_turn_rollout.utils import process_qwen3_ocr_image


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def build_prompt_text(with_history: bool = True, action_list_style: str = "quoted_lines") -> tuple[str, list[str]]:
    action_pool = ["go to fridge 1", "open fridge 1", "go to countertop 1"]
    if action_list_style == "quoted_lines":
        admissible_actions = "\n ".join(f"'{action}'" for action in action_pool)
    elif action_list_style == "plain_lines":
        admissible_actions = "\n".join(action_pool)
    else:
        raise ValueError(f"Unsupported ACTION_LIST_STYLE={action_list_style}")

    if with_history:
        prompt_text = ALFWORLD_TEMPLATE_OCR.format(
            task_description="put a cooled apple in fridge 1",
            goal_hints="",
            step_count=2,
            history_length=2,
            action_history=(
                "[Observation]: you are in the kitchen.\n"
                "[Action]: go to countertop 1\n"
                "[Observation]: you see an apple 1 on the countertop."
            ),
            history_hint="",
            failure_hint="",
            current_step=3,
            current_observation="you see apple 1 on countertop 1 and fridge 1 nearby.",
            admissible_actions=admissible_actions,
            rules=f"{ALFWORLD_ACTION_RULES}\n{ALFWORLD_COMPRESSION_RULE}",
        )
    else:
        prompt_text = ALFWORLD_TEMPLATE_NO_HIS_OCR.format(
            task_description="put a cooled apple in fridge 1",
            goal_hints="",
            current_step=1,
            current_observation="you see apple 1 on countertop 1 and fridge 1 nearby.",
            failure_hint="",
            admissible_actions=admissible_actions,
            rules=f"{ALFWORLD_ACTION_RULES}\n{ALFWORLD_COMPRESSION_RULE}",
        )
    prompt_text = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1)
    return prompt_text, action_pool


def build_test_image(with_history: bool, use_ocr_history_image: bool) -> Image.Image:
    if not with_history or not use_ocr_history_image:
        return Image.new("RGB", (448, 448), color=(255, 255, 255))

    history_repeat = _env_int("HISTORY_REPEAT", 1)
    highlight_configs = parse_highlight_configs(
        os.environ.get("HIGHLIGHT_CONFIGS", "[Observation]:0,0,255;[Action]:255,0,0")
    )
    ocr = OCRTool(
        enabled=True,
        font_size=10,
        padding=5,
        min_width=28,
        max_width=392,
        min_height=28,
        max_height=4096,
        max_workers=1,
        use_parallel=False,
        use_precise=False,
        enable_cache=False,
        highlight_configs=highlight_configs,
    )
    prefix_block = (
        "[Observation]: you are in the kitchen.\n"
        "[Action]: look\n"
        "[Observation]: you notice a countertop and a fridge nearby.\n"
    )
    terminal_block = (
        "[Observation]: you are in the kitchen.\n"
        "[Action]: go to countertop 1\n"
        "[Observation]: you see an apple 1 on the countertop.\n"
        "[Action]: take apple 1 from countertop 1\n"
        "[Observation]: you pick up the apple 1.\n"
        "[Action]: go to fridge 1\n"
        "[Observation]: you arrive at fridge 1.\n"
    )
    history_context = prefix_block * max(0, history_repeat - 1) + terminal_block
    image_array = ocr.convert_texts_to_images(
        trajectory_contexts=[history_context],
        batch_size=1,
        active_masks=[True],
        compression_factor=[1.0],
        current_steps=[3],
        enable_cache=False,
        save_img=False,
    )[0]
    return process_qwen3_ocr_image(image_array)


def build_chat(prompt_text: str, prefill_think: bool, assistant_prefill_text: str | None) -> list[dict]:
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
    if assistant_prefill_text is not None:
        chat.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_prefill_text}],
            }
        )
    elif prefill_think:
        chat.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<thinking>"}],
            }
        )
    return chat


def build_prompt(
    processor,
    chat: list[dict],
    prefill_think: bool,
    enable_thinking: bool,
    assistant_prefill_text: str | None,
) -> str:
    kwargs = {"tokenize": False}
    if prefill_think or assistant_prefill_text is not None:
        kwargs["add_generation_prompt"] = False
        kwargs["continue_final_message"] = True
    else:
        kwargs["add_generation_prompt"] = True
        kwargs["enable_thinking"] = enable_thinking
    return processor.apply_chat_template(chat, **kwargs)


def normalize_and_report(
    text: str,
    action_pool: list[str],
    assistant_prefill_text: str | None,
) -> None:
    print("RAW_OUTPUT", repr(text))
    full_text = f"{assistant_prefill_text}{text}" if assistant_prefill_text else text
    if "<action>" in full_text.lower():
        normalized = full_text
    elif full_text.lstrip().lower().startswith("<thinking>"):
        normalized = full_text
    else:
        normalized = "<thinking>" + full_text
    actions, valids = alfworld_projection([normalized], [action_pool])
    print("NORMALIZED", repr(normalized))
    print("PARSED_ACTION", actions[0])
    print("VALID", valids[0])


def run_vllm(model_path: str, prompt: str, image: Image.Image) -> str:
    llm = LLM(
        model=model_path,
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "2048")),
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=False,
        enforce_eager=_env_flag("ENFORCE_EAGER", True),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.55")),
        max_num_batched_tokens=int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "2048")),
        max_num_seqs=1,
    )
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        sampling_params=SamplingParams(
            temperature=float(os.environ.get("TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("MAX_TOKENS", "96")),
        ),
    )
    return outputs[0].outputs[0].text


def run_transformers(model_path: str, processor, prompt: str, image: Image.Image) -> str:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    print("IMAGE_SIZE", image.size)
    if "image_grid_thw" in inputs:
        image_grid = inputs["image_grid_thw"][0].tolist()
        print("IMAGE_GRID_THW", image_grid)
        print("IMAGE_VISUAL_TOKENS", int(image_grid[0] * image_grid[1] * image_grid[2]))
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MAX_TOKENS", "96")),
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    output_tokens = generated[0][input_len:]
    return processor.decode(output_tokens, skip_special_tokens=False)


def main() -> None:
    model_path = os.environ.get(
        "MODEL_PATH", "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-2B-Thinking"
    )
    with_history = _env_flag("WITH_HISTORY", True)
    prefill_think = _env_flag("PREFILL_THINK", False)
    enable_thinking = _env_flag("ENABLE_THINKING", True)
    action_list_style = os.environ.get("ACTION_LIST_STYLE", "quoted_lines")
    backend = os.environ.get("BACKEND", "transformers")
    assistant_prefill_text = os.environ.get("ASSISTANT_PREFILL_TEXT")
    use_ocr_history_image = _env_flag("USE_OCR_HISTORY_IMAGE", False)

    processor = AutoProcessor.from_pretrained(model_path)
    prompt_text, action_pool = build_prompt_text(
        with_history=with_history,
        action_list_style=action_list_style,
    )
    image = build_test_image(with_history=with_history, use_ocr_history_image=use_ocr_history_image)
    chat = build_chat(
        prompt_text=prompt_text,
        prefill_think=prefill_think,
        assistant_prefill_text=assistant_prefill_text,
    )
    prompt = build_prompt(
        processor=processor,
        chat=chat,
        prefill_think=prefill_think,
        enable_thinking=enable_thinking,
        assistant_prefill_text=assistant_prefill_text,
    )
    print("MODEL_PATH", model_path)
    print("BACKEND", backend)
    print("WITH_HISTORY", with_history)
    print("PREFILL_THINK", prefill_think)
    print("ENABLE_THINKING", enable_thinking)
    print("ACTION_LIST_STYLE", action_list_style)
    print("ASSISTANT_PREFILL_TEXT", repr(assistant_prefill_text))
    print("USE_OCR_HISTORY_IMAGE", use_ocr_history_image)
    print("PROMPT_START")
    print(prompt[:1500])
    print("PROMPT_END")

    if backend == "vllm":
        text = run_vllm(model_path=model_path, prompt=prompt, image=image)
    elif backend == "transformers":
        text = run_transformers(model_path=model_path, processor=processor, prompt=prompt, image=image)
    else:
        raise ValueError(f"Unsupported BACKEND={backend}")

    normalize_and_report(
        text=text,
        action_pool=action_pool,
        assistant_prefill_text=assistant_prefill_text,
    )


if __name__ == "__main__":
    main()
