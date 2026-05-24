# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import os
import numbers
import re
from PIL import Image
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.environments.prompts.alfworld import ALFWORLD_QWEN3VL_SYSTEM_PROMPT
from agent_system.multi_turn_rollout.utils import (
    process_image,
    process_qwen3_ocr_image,
    get_agentocr_image_metadata,
    to_list_of_dict,
    torch_to_numpy,
    filter_group_data,
    stable_int_seed,
)
from agent_system.multi_turn_rollout.offline_hooks import (
    _extract_prompt_text,
    export_rollout_traces_for_replay,
    run_offline_agentocr_hooks,
)
from agent_system.environments import EnvironmentManagerBase
from agent_system.environments.env_manager import summarize_trust_policy_recovery_metrics
from typing import List, Dict, Any, Optional, Tuple
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.protocol import extract_dataproto_via_active_mask, restore_dataproto_via_active_mask
import time


def _memory_credit_probe_enabled() -> bool:
    return os.environ.get("AGENTOCR_MEMORY_CREDIT_PROBE", "FALSE").upper() in ["TRUE", "1"]


def _trace_export_enabled() -> bool:
    return bool(os.environ.get("AGENTOCR_EXPORT_TRACE_DIR"))


def _looks_like_alfworld_react_prompt(obs_content: str) -> bool:
    if not obs_content:
        return False
    new_markers = [
        "Task:",
        "Current step:",
        "Current textual observation:",
        "Admissible actions:",
        "Rules:",
    ]
    if all(marker in obs_content for marker in new_markers):
        return True

    legacy_react_markers = [
        "Task:",
        "Step:",
        "Previous observations and actions:",
        "Current observation:",
        "Admissible actions:",
        "Instructions:",
    ]
    if all(marker in obs_content for marker in legacy_react_markers):
        return True

    lowered = obs_content.lower()
    return (
        "alfred embodied environment" in lowered
        and (
            "your admissible actions of the current situation are:" in lowered
            or "admissible actions:" in lowered
        )
        and "<action>" in lowered
    )


def _strip_duplicate_react_system_header(obs_content: str) -> str:
    if not obs_content:
        return obs_content
    normalized = str(obs_content).lstrip()
    system_header = ALFWORLD_QWEN3VL_SYSTEM_PROMPT.strip()
    if not normalized.startswith(system_header):
        return obs_content

    remainder = normalized[len(system_header):]
    if remainder.startswith("\n"):
        remainder = remainder[1:]
    return remainder.lstrip()


def _is_qwen3_vl_processor(processor) -> bool:
    if processor is None:
        return False
    class_name = processor.__class__.__name__
    if "Qwen3" in class_name and "Processor" in class_name:
        return True
    candidate_names = [
        getattr(processor, "name_or_path", None),
        getattr(getattr(processor, "tokenizer", None), "name_or_path", None),
    ]
    normalized = " ".join(str(name or "") for name in candidate_names).lower()
    return any(token in normalized for token in ("qwen3-vl", "qwen3vl", "qwen3.5", "qwen3_5"))


def _is_qwen3_vl_thinking_processor(processor, config=None) -> bool:
    if not _is_qwen3_vl_processor(processor):
        return False
    candidate_names = [
        getattr(processor, "name_or_path", None),
        getattr(getattr(processor, "tokenizer", None), "name_or_path", None),
    ]
    if config is not None:
        try:
            candidate_names.append(config.actor_rollout_ref.model.path)
        except Exception:
            pass
    normalized = " ".join(str(name or "") for name in candidate_names).lower()
    return "thinking" in normalized


def _prepare_model_specific_image_input(obs_image, processor):
    processor_kwargs = {}
    if isinstance(obs_image, (list, tuple)):
        obs_images = list(obs_image)
    else:
        obs_images = [obs_image]

    if _is_qwen3_vl_processor(processor):
        image_processor = getattr(processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
        temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2) or 2)
        images = [
            process_qwen3_ocr_image(
                image,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )
            for image in obs_images
        ]
        processor_kwargs["do_resize"] = False
        return images, processor_kwargs

    return [process_image(image) for image in obs_images], processor_kwargs


def _qwen3_mm_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _count_mm_visual_tokens(image_grid_thw, merge_size: int) -> int:
    if image_grid_thw is None:
        return 0
    merge_length = max(1, int(merge_size) ** 2)
    total = 0
    for i in range(len(image_grid_thw)):
        total += int((image_grid_thw[i].prod() // merge_length).item())
    return total


def _build_qwen3_multimodal_model_inputs(
    raw_prompt: str,
    obs_image,
    processor,
    max_prompt_length: int,
) -> Tuple[List[Image.Image], Dict[str, Any], Dict[str, Any], Dict[str, int]]:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
    temporal_patch_size = int(getattr(image_processor, "temporal_patch_size", 2) or 2)
    merge_size = int(getattr(image_processor, "merge_size", 2) or 2)

    requested_max_pixels = _qwen3_mm_env_int("AGENTOCR_QWEN3_OCR_MAX_PIXELS", 131072)
    min_pixels = _qwen3_mm_env_int("AGENTOCR_QWEN3_OCR_MIN_PIXELS", 16384)
    safety_margin = max(0, _qwen3_mm_env_int("AGENTOCR_QWEN3_MM_PROMPT_SAFETY_MARGIN", 256))
    preferred_visual_floor = max(64, _qwen3_mm_env_int("AGENTOCR_QWEN3_MM_MIN_VISUAL_TOKENS", 896))
    hard_visual_floor = max(64, _qwen3_mm_env_int("AGENTOCR_QWEN3_MM_HARD_MIN_VISUAL_TOKENS", 384))
    max_shrink_steps = max(1, _qwen3_mm_env_int("AGENTOCR_QWEN3_MM_MAX_SHRINK_STEPS", 4))
    max_expand_pixels = max(
        requested_max_pixels,
        _qwen3_mm_env_int("AGENTOCR_QWEN3_MM_MAX_EXPAND_PIXELS", requested_max_pixels * 2),
    )
    safe_prompt_len = max(512, int(max_prompt_length) - safety_margin)

    processor_kwargs = {"do_resize": False}
    diagnostics = {
        "safe_prompt_len": int(safe_prompt_len),
        "requested_max_pixels": int(requested_max_pixels),
        "preferred_visual_floor": int(preferred_visual_floor),
        "hard_visual_floor": int(hard_visual_floor),
        "expand_steps": 0,
        "shrink_steps": 0,
    }

    obs_images = list(obs_image) if isinstance(obs_image, (list, tuple)) else [obs_image]
    metadata_entries = [get_agentocr_image_metadata(image) for image in obs_images]
    preferred_floor_overrides = [
        int(meta["agentocr_mm_preferred_visual_floor"])
        for meta in metadata_entries
        if meta.get("agentocr_mm_preferred_visual_floor") is not None
    ]
    hard_floor_overrides = [
        int(meta["agentocr_mm_hard_visual_floor"])
        for meta in metadata_entries
        if meta.get("agentocr_mm_hard_visual_floor") is not None
    ]
    if preferred_floor_overrides:
        preferred_visual_floor = max(64, min(preferred_floor_overrides))
        diagnostics["preferred_visual_floor"] = int(preferred_visual_floor)
    if hard_floor_overrides:
        hard_visual_floor = max(64, min(hard_floor_overrides))
    if hard_visual_floor > preferred_visual_floor:
        hard_visual_floor = preferred_visual_floor
    diagnostics["hard_visual_floor"] = int(hard_visual_floor)
    current_max_pixels = max(min_pixels * max(1, len(obs_images)), requested_max_pixels)
    current_min_pixels = min_pixels

    def _build_processed_images(target_max_pixels: int, target_min_pixels: int) -> List[Image.Image]:
        per_image_max_pixels_local = max(min_pixels, int(target_max_pixels / max(1, len(obs_images))))
        per_image_min_pixels_local = max(min_pixels, int(target_min_pixels / max(1, len(obs_images))))
        per_image_min_pixels_local = min(per_image_min_pixels_local, per_image_max_pixels_local)
        return [
            process_qwen3_ocr_image(
                image,
                max_pixels=per_image_max_pixels_local,
                min_pixels=per_image_min_pixels_local,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
            )
            for image in obs_images
        ]

    def _encode_images(processed_images_local: List[Image.Image]) -> Tuple[Dict[str, Any], int, int, int]:
        model_inputs_local = processor(
            text=[raw_prompt],
            images=processed_images_local,
            return_tensors="pt",
            **processor_kwargs,
        )
        image_grid_thw_local = model_inputs_local.get("image_grid_thw")
        visual_tokens_local = _count_mm_visual_tokens(image_grid_thw_local, merge_size)
        input_len_local = int(model_inputs_local["input_ids"].shape[1])
        text_tokens_local = max(0, input_len_local - visual_tokens_local)
        return model_inputs_local, visual_tokens_local, input_len_local, text_tokens_local

    processed_images = _build_processed_images(current_max_pixels, current_min_pixels)

    final_model_inputs = None
    final_visual_tokens = 0
    final_input_len = 0
    final_text_tokens = 0

    model_inputs, visual_tokens, input_len, text_tokens = _encode_images(processed_images)
    final_model_inputs = model_inputs
    final_visual_tokens = visual_tokens
    final_input_len = input_len
    final_text_tokens = text_tokens

    if input_len < safe_prompt_len and visual_tokens < preferred_visual_floor:
        available_visual_budget = max(hard_visual_floor, safe_prompt_len - text_tokens - 32)
        target_visual_tokens = min(preferred_visual_floor, available_visual_budget)
        if target_visual_tokens > visual_tokens and visual_tokens > 0:
            current_pixels = max(1, sum(image.width * image.height for image in processed_images))
            target_pixels = int(current_pixels * (target_visual_tokens / float(visual_tokens)))
            target_pixels = min(max_expand_pixels, max(current_max_pixels + 1024, target_pixels))
            if target_pixels > current_max_pixels:
                current_max_pixels = target_pixels
                current_min_pixels = target_pixels
                processed_images = _build_processed_images(current_max_pixels, current_min_pixels)
                model_inputs, visual_tokens, input_len, text_tokens = _encode_images(processed_images)
                diagnostics["expand_steps"] = 1
                final_model_inputs = model_inputs
                final_visual_tokens = visual_tokens
                final_input_len = input_len
                final_text_tokens = text_tokens

    for attempt in range(max_shrink_steps):
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
        current_min_pixels = min_pixels
        diagnostics["shrink_steps"] = attempt + 1
        processed_images = _build_processed_images(current_max_pixels, current_min_pixels)
        model_inputs, visual_tokens, input_len, text_tokens = _encode_images(processed_images)
        final_model_inputs = model_inputs
        final_visual_tokens = visual_tokens
        final_input_len = input_len
        final_text_tokens = text_tokens

    diagnostics.update(
        {
            "input_len": int(final_input_len),
            "text_tokens": int(final_text_tokens),
            "visual_tokens": int(final_visual_tokens),
            "final_max_pixels": int(current_max_pixels),
            "final_per_image_max_pixels": int(max(min_pixels, int(current_max_pixels / max(1, len(obs_images))))),
            "image_count": int(len(processed_images)),
            "final_width": int(max(image.width for image in processed_images)),
            "final_height": int(max(image.height for image in processed_images)),
        }
    )
    return processed_images, processor_kwargs, final_model_inputs, diagnostics


_ACTION_BLOCK_RE = re.compile(r"(<action>.*?</action>)", re.IGNORECASE | re.DOTALL)
_SUMMARY_BLOCK_RE = re.compile(r"<summary>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINKING_BLOCK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL)


def _has_closed_reasoning_block(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        ("<think>" in lowered and "</think>" in lowered)
        or ("<thinking>" in lowered and "</thinking>" in lowered)
    )


def _normalize_action_reasoning_format(text: str) -> str:
    lowered = str(text or "").lower()
    if text and "<summary>" in lowered and "<thinking>" not in lowered and "<think>" not in lowered:
        text = _SUMMARY_BLOCK_RE.sub(
            lambda match: f"<thinking>{match.group(1).strip()}</thinking>",
            text,
        )
        lowered = str(text).lower()
    if text and "<think>" in lowered:
        text = _THINK_BLOCK_RE.sub(
            lambda match: f"<thinking>{match.group(1).strip()}</thinking>",
            text,
        )
        lowered = str(text).lower()
    if text and "<thinking>" in lowered:
        text = _THINKING_BLOCK_RE.sub(
            lambda match: f"<thinking>{match.group(1).strip()}</thinking>",
            text,
        )
    if not text or "<action>" not in text.lower():
        return text
    if _has_closed_reasoning_block(text):
        return text

    match = _ACTION_BLOCK_RE.search(text)
    if not match:
        return text

    reasoning_prefix = text[:match.start()].strip()
    action_block = match.group(1).strip()
    action_suffix = text[match.end():].strip()

    if "<think>" in text.lower() and "</think>" not in text.lower():
        normalized_reasoning = text[:match.start()].replace("<think>", "", 1).strip()
        normalized = f"<thinking>{normalized_reasoning}</thinking>\n{action_block}"
        if action_suffix:
            normalized = f"{normalized}\n{action_suffix}"
        return normalized

    if "<thinking>" in text.lower() and "</thinking>" not in text.lower():
        normalized_reasoning = text[:match.start()].replace("<thinking>", "", 1).strip()
        normalized = f"<thinking>{normalized_reasoning}</thinking>\n{action_block}"
        if action_suffix:
            normalized = f"{normalized}\n{action_suffix}"
        return normalized

    if reasoning_prefix:
        normalized = f"<thinking>{reasoning_prefix}</thinking>\n{action_block}"
        if action_suffix:
            normalized = f"{normalized}\n{action_suffix}"
        return normalized

    return f"<thinking></thinking>\n{action_block}"


def _summarize_memory_credit_probe(trajectory_steps: List[Dict]) -> Dict[str, float]:
    active_steps = [step for step in trajectory_steps if bool(step.get("active_masks", True))]
    if not active_steps:
        return {}
    final_step = active_steps[-1]
    summary = run_offline_agentocr_hooks(
        trajectory_steps=active_steps,
        memory_segments=None,
        is_success=bool(final_step.get("is_success", False)),
        current_step=len(active_steps),
        memory_context=_extract_prompt_text(final_step),
    )
    dense_rewards = summary.counterfactual_credit.get("dense_rewards", []) or []
    blame_scores = summary.counterfactual_credit.get("blame_scores", []) or []
    compression_factors = summary.trust_rendering.get("compression_factors", []) or []
    action_counts = summary.trust_rendering.get("action_counts", {}) or {}
    total_actions = max(1, sum(action_counts.values()))
    return {
        "memory_credit/offline_dense_return": float(sum(dense_rewards)) if dense_rewards else 0.0,
        "memory_credit/offline_blame_peak": float(max(blame_scores)) if blame_scores else 0.0,
        "memory_credit/offline_repair_step": float(summary.counterfactual_credit.get("repair_step") or 0.0),
        "memory_credit/trust_compression_mean": float(np.mean(compression_factors)) if compression_factors else 0.0,
        "memory_credit/trust_drop_rate": float(action_counts.get("drop", 0) / total_actions),
        "memory_credit/trust_compress_rate": float(action_counts.get("compress", 0) / total_actions),
        "memory_credit/trust_full_rate": float(action_counts.get("full", 0) / total_actions),
    }


def _fill_missing_sparse_trust_policy_keys(samples: List[Dict]) -> None:
    if not samples:
        return

    exemplar_by_key: Dict[str, object] = {}
    for sample in samples:
        for key, value in sample.items():
            if key.startswith("trust_policy/") and key not in exemplar_by_key:
                exemplar_by_key[key] = value

    if not exemplar_by_key:
        return

    def _default_for(value):
        if isinstance(value, (str, np.str_)):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (numbers.Number, np.number)):
            return 0.0
        if isinstance(value, dict):
            return {}
        if isinstance(value, list):
            return []
        if isinstance(value, tuple):
            return ()
        if isinstance(value, np.ndarray):
            return np.zeros_like(value) if value.dtype != object else np.empty_like(value)
        return 0.0

    default_by_key = {key: _default_for(value) for key, value in exemplar_by_key.items()}
    for sample in samples:
        for key, default_value in default_by_key.items():
            if key not in sample:
                sample[key] = default_value

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None
        obs_image_list = list(obs_image) if isinstance(obs_image, (list, tuple)) else ([obs_image] if obs_image is not None else [])

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        if is_multi_modal:
            obs_content = obs_content.replace("<image>\n", "", 1)
            obs_content = obs_content.replace("<image>", "", 1)

        
        if is_multi_modal and self.processor is not None:
            template_kwargs = dict(apply_chat_template_kwargs)
            add_generation_prompt = True
            response_prefix_text = ""
            if _looks_like_alfworld_react_prompt(obs_content):
                obs_content = _strip_duplicate_react_system_header(obs_content)
                chat = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": ALFWORLD_QWEN3VL_SYSTEM_PROMPT,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            *([{"type": "image"}] * max(1, len(obs_image_list))),
                            {"type": "text", "text": obs_content},
                        ],
                    },
                ]
                if _is_qwen3_vl_processor(self.processor):
                    if _is_qwen3_vl_thinking_processor(self.processor, self.config):
                        # The Thinking template starts with its own `<think>`
                        # block regardless of `enable_thinking=False`, so keep
                        # the official template untouched.
                        template_kwargs.pop("enable_thinking", None)
                        template_kwargs.pop("continue_final_message", None)
                    else:
                        # For Qwen3-VL Instruct, a plain assistant generation
                        # prompt preserves action quality better than forcing a
                        # partially filled `<think>`/`<action>` prefix. The
                        # downstream normalization path can still backfill a
                        # missing think block when the model emits only
                        # `<action>...</action>`.
                        template_kwargs.pop("continue_final_message", None)
            else:
                chat = [{
                    "role": "user",
                    "content": [
                        *([{"type": "image"}] * max(1, len(obs_image_list))),
                        {"type": "text", "text": obs_content},
                    ],
                }]
            prompt_with_chat_template = self.processor.apply_chat_template(
                chat,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **template_kwargs,
            )
        else:
            chat = [{
                "content": obs_content,
                "role": "user",
            }]
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
        
        # Initialize return dict
        row_dict = {}
        image_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            image_heights = [float(image.shape[0]) for image in obs_image_list]
            image_widths = [float(image.shape[1]) for image in obs_image_list]
            image_height = np.float32(max(image_heights))
            image_width = np.float32(max(image_widths))
            image_size = np.float32(sum(h * w for h, w in zip(image_heights, image_widths)))
            image_dict = {
                'image_height': image_height,
                'image_width': image_width,
                'image_size': image_size,
                'image_page_count': np.float32(len(obs_image_list)),
            }
            raw_prompt = prompt_with_chat_template
            if _is_qwen3_vl_processor(self.processor):
                processed_images, image_processor_kwargs, model_inputs, mm_budget_diagnostics = _build_qwen3_multimodal_model_inputs(
                    raw_prompt=raw_prompt,
                    obs_image=obs_image_list,
                    processor=self.processor,
                    max_prompt_length=self.config.data.max_prompt_length,
                )
                image_dict["mm_prompt_token_count"] = np.float32(mm_budget_diagnostics["input_len"])
                image_dict["mm_text_token_count"] = np.float32(mm_budget_diagnostics["text_tokens"])
                image_dict["mm_preferred_visual_floor"] = np.float32(mm_budget_diagnostics["preferred_visual_floor"])
                image_dict["mm_hard_visual_floor"] = np.float32(mm_budget_diagnostics["hard_visual_floor"])
                image_dict["mm_budget_expand_steps"] = np.float32(mm_budget_diagnostics["expand_steps"])
                image_dict["mm_budget_shrink_steps"] = np.float32(mm_budget_diagnostics["shrink_steps"])
            else:
                processed_images, image_processor_kwargs = _prepare_model_specific_image_input(obs_image_list, self.processor)
                model_inputs = self.processor(
                    text=[raw_prompt],
                    images=processed_images,
                    return_tensors='pt',
                    **image_processor_kwargs,
                )
            images = list(processed_images)
            row_dict['multi_modal_data'] = {'image': images}
            row_dict['response_prefix_text'] = response_prefix_text
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            row_dict['multi_modal_inputs'] = dict(model_inputs)
            image_grid_thw = row_dict['multi_modal_inputs'].get('image_grid_thw')
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                memory_visual_token_count = sum(
                    (image_grid_thw[i].prod() // merge_length).item()
                    for i in range(len(image_grid_thw))
                )
                image_dict['memory_visual_token_count'] = np.float32(memory_visual_token_count)

        else:
            raw_prompt = prompt_with_chat_template
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                             tokenizer=self.tokenizer,
                                                                             max_length=self.config.data.max_prompt_length,
                                                                             pad_token_id=self.tokenizer.pad_token_id,
                                                                             left_pad=True,
                                                                             truncation=self.config.data.truncation,)

        if is_multi_modal:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
        
        

        if is_multi_modal:

            if _is_qwen3_vl_processor(self.processor):
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'raw_prompt_text': raw_prompt,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source,
            **image_dict,
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        traj_probe_metrics: List[Dict[str, float]] = [{} for _ in range(batch_size)]
        traj_mechanism_metrics: List[Dict[str, float]] = [{} for _ in range(batch_size)]
        if _memory_credit_probe_enabled():
            for bs in range(batch_size):
                trajectory = total_batch_list[bs]
                if not trajectory:
                    continue
                probe_ready_steps = [dict(step) for step in trajectory]
                success_value = bool(success.get('success_rate', np.zeros(batch_size))[bs] > 0)
                if probe_ready_steps:
                    probe_ready_steps[-1]["is_success"] = success_value
                traj_probe_metrics[bs] = _summarize_memory_credit_probe(probe_ready_steps)
        for bs in range(batch_size):
            trajectory = total_batch_list[bs]
            if not trajectory:
                continue
            traj_mechanism_metrics[bs] = summarize_trust_policy_recovery_metrics(trajectory)

        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    # is_success
                    if success['success_rate'][bs] > 0:
                        data['is_success'] = True
                    else:
                        data['is_success'] = False
                    if traj_mechanism_metrics[bs]:
                        data.update(traj_mechanism_metrics[bs])
                    if traj_probe_metrics[bs]:
                        data.update(traj_probe_metrics[bs])
                    # data['is_success'] = bool(success['success_rate'][bs])
                    effective_batch.append(data)
            
        _fill_missing_sparse_trust_policy_keys(effective_batch)

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        export_dir = os.environ.get("AGENTOCR_EXPORT_TRACE_DIR")
        if export_dir:
            max_trajectories = int(os.environ.get("AGENTOCR_EXPORT_MAX_TRAJ", "2"))
            exported_paths = export_rollout_traces_for_replay(
                total_batch_list=total_batch_list,
                output_dir=export_dir,
                max_trajectories=max_trajectories,
            )
            if exported_paths:
                print(f"[AgentOCR Debug] Exported replay traces: {exported_paths}")
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt_text" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt_text")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            if (
                batch_input.meta_info.get("do_sample", True)
                and not batch_input.meta_info.get("validate", False)
            ):
                rollout_seed_base = int(
                    self.config.actor_rollout_ref.rollout.get("seed", self.config.env.seed) or 0
                )
                sampling_seed = stable_int_seed(
                    rollout_seed_base,
                    bool(getattr(envs, "_agentocr_rollout_is_train", True)),
                    tuple(gen_batch.non_tensor_batch.get("data_source", [])),
                    _step,
                )
                batch_input.meta_info["sampling_seed"] = int(sampling_seed)

            # extract activate data
            batch_input_extracted = extract_dataproto_via_active_mask(batch_input, active_masks)
            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input_extracted, actor_rollout_wg.world_size)
            start_time = time.time()
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            end_time = time.time()
            self.llm_forward_time += end_time - start_time
            # unpad
            batch_output_extracted = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            # restorr
            batch_output = restore_dataproto_via_active_mask(batch_output_extracted, active_masks)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            response_prefix_texts = batch.non_tensor_batch.get('response_prefix_text')
            if response_prefix_texts is not None:
                text_actions = [
                    f"{prefix}{text}" if prefix else text
                    for prefix, text in zip(response_prefix_texts, text_actions)
                ]
            if _is_qwen3_vl_processor(self.processor):
                text_actions = [_normalize_action_reasoning_format(text) for text in text_actions]
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if any('is_action_valid' in info for info in infos):
                batch.non_tensor_batch['is_action_valid'] = np.array(
                    [info.get('is_action_valid', True) for info in infos],
                    dtype=bool,
                )
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if any('compression_factor' in info for info in infos):
                batch.non_tensor_batch['compression_factor'] = np.array(
                    [info.get('compression_factor', 1.0) for info in infos],
                    dtype=np.float32,
                )

            if _trace_export_enabled():
                anchor_obs = next_obs.get('anchor') if isinstance(next_obs, dict) else None
                batch.non_tensor_batch['debug_model_response_text'] = np.array(text_actions, dtype=object)
                batch.non_tensor_batch['debug_env_observation_text'] = np.array(
                    [
                        anchor_obs[i]
                        if anchor_obs is not None and i < len(anchor_obs)
                        else info.get('observation_text', '')
                        for i, info in enumerate(infos)
                    ],
                    dtype=object,
                )
                batch.non_tensor_batch['debug_env_admissible_actions'] = np.array(
                    [list(info.get('admissible_commands', [])) for info in infos],
                    dtype=object,
                )
                batch.non_tensor_batch['debug_env_gamefile'] = np.array(
                    [str(info.get('extra.gamefile', '')) for info in infos],
                    dtype=object,
                )
                batch.non_tensor_batch['debug_env_won'] = np.array(
                    [bool(info.get('won', False)) for info in infos],
                    dtype=object,
                )

            batch.non_tensor_batch['debug_env_postprocessed_action'] = np.array(
                [str(info.get('postprocessed_action', '')) for info in infos],
                dtype=object,
            )
            batch.non_tensor_batch['debug_env_tool_input'] = np.array(
                [
                    tuple(info.get('tool_input', [])) if info.get('tool_input') is not None else tuple()
                    for info in infos
                ],
                dtype=object,
            )
            batch.non_tensor_batch['debug_env_idea_adapter_reasons'] = np.array(
                [str(info.get('idea_adapter_reasons', '')) for info in infos],
                dtype=object,
            )

            # Trust-policy metrics contain dynamic sparse keys such as
            # family/phase-specific indicators. Collect the union across all
            # env infos at this step and fill missing entries with 0.0 so that
            # every rollout sample carries a consistent key set for collation.
            trust_metric_keys = sorted(
                {
                    key
                    for info in infos
                    for key in info.keys()
                    if key.startswith('trust_policy/')
                }
            )
            for key in trust_metric_keys:
                batch.non_tensor_batch[key] = np.array(
                    [float(info.get(key, 0.0)) for info in infos],
                    dtype=np.float32,
                )

            if any('tool_calling' in info for info in infos):
                tool_callings[active_masks] += np.array(
                    [info.get('tool_calling', 0.0) for info in infos],
                    dtype=np.float32,
                )[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        merged_success = {}
        all_success_keys = sorted({key for success in total_success for key in success.keys()})
        for key in all_success_keys:
            chunks = [success[key] for success in total_success if key in success]
            if chunks:
                merged_success[key] = np.concatenate(chunks, axis=0)
        total_success = merged_success
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        self.llm_forward_time = 0
        setattr(envs, "_agentocr_rollout_is_train", bool(is_train))
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
