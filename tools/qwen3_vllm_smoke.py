#!/usr/bin/env python3
import os
import sys

from PIL import Image


def _sanitize_sys_path(project_root: str) -> None:
    cleaned = []
    delayed = []
    normalized_root = os.path.realpath(project_root)

    for entry in sys.path:
        normalized = os.path.realpath(entry or os.getcwd())
        if normalized == normalized_root:
            delayed.append(entry)
        else:
            cleaned.append(entry)

    if project_root not in cleaned:
        cleaned.append(project_root)
    for entry in delayed:
        if entry not in cleaned:
            cleaned.append(entry)
    sys.path[:] = cleaned


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sanitize_sys_path(project_root)

    import torch
    import transformers
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model_path = os.environ.get(
        "MODEL_PATH", "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct")

    print("torch", torch.__version__, torch.__file__)
    print("transformers", transformers.__version__, transformers.__file__)
    print("vllm", vllm.__version__, vllm.__file__)
    try:
        import flash_attn
        print("flash_attn", flash_attn.__file__)
    except Exception as exc:
        print("flash_attn_import_error", repr(exc))
    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(model_path)
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": "Describe the image in one short sentence."
            },
        ],
    }]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image = Image.new("RGB", (448, 448), color=(255, 255, 255))
    use_image_list = _get_bool_env("USE_IMAGE_LIST", False)

    llm_kwargs = {
        "model": model_path,
        "max_model_len": _get_int_env("MAX_MODEL_LEN", 1024),
        "limit_mm_per_prompt": {"image": 1},
        "trust_remote_code": False,
        "enforce_eager": _get_bool_env("ENFORCE_EAGER", False),
        "gpu_memory_utilization": _get_float_env("GPU_MEMORY_UTILIZATION",
                                                 0.9),
        "max_num_batched_tokens": _get_int_env("MAX_NUM_BATCHED_TOKENS",
                                               1024),
        "max_num_seqs": _get_int_env("MAX_NUM_SEQS", 1),
        "skip_mm_profiling": _get_bool_env("SKIP_MM_PROFILING", False),
    }
    if _get_bool_env("ENABLE_LORA", False):
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = _get_int_env("MAX_LORAS", 1)
        llm_kwargs["max_lora_rank"] = _get_int_env("MAX_LORA_RANK", 8)
    print("llm_kwargs", llm_kwargs)

    llm = LLM(
        **llm_kwargs,
    )
    outputs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": [image] if use_image_list else image
            },
        }],
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=8,
            logprobs=None,
        ),
    )
    for output in outputs:
        print("prompt_tokens", len(output.prompt_token_ids))
        print("text", repr(output.outputs[0].text))


if __name__ == "__main__":
    main()
