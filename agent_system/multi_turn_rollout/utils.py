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

import hashlib
import torch
import numpy as np
import random
from typing import Any, List, Tuple, Dict
import math
import os
from PIL import Image, ImageOps
from verl import DataProto


class AgentOCRImageArray(np.ndarray):
    """NumPy array wrapper that can carry lightweight render metadata."""


def attach_agentocr_image_metadata(image: np.ndarray, **metadata: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        return image
    wrapped = image if isinstance(image, AgentOCRImageArray) else image.view(AgentOCRImageArray)
    existing = dict(getattr(wrapped, "_agentocr_metadata", {}) or {})
    for key, value in metadata.items():
        if value is None:
            continue
        existing[str(key)] = value
    wrapped._agentocr_metadata = existing
    return wrapped


def get_agentocr_image_metadata(image: Any) -> Dict[str, Any]:
    return dict(getattr(image, "_agentocr_metadata", {}) or {})

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 28 * 28):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        print(f"Warning: Image too large, resizing from {image.width}x{image.height} to {max_pixels}")
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def process_qwen3_ocr_image(
    image,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
):
    """Prepare OCR history images for Qwen3-VL using an explicit pixel budget."""
    def _get_env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    def _pad_image_for_aspect(
        image_obj: Image.Image,
        min_aspect: float,
        max_aspect: float,
        max_padding_area_ratio: float,
        alignment_factor: int,
    ) -> Image.Image:
        def _round_up(value: int, factor: int) -> int:
            factor = max(1, int(factor))
            return int(math.ceil(max(1, value) / float(factor)) * factor)

        width = max(1, int(image_obj.width))
        height = max(1, int(image_obj.height))
        original_area = max(1, width * height)
        aspect = width / float(height)

        if min_aspect > 0.0 and aspect < min_aspect:
            target_width = _round_up(max(width, int(math.ceil(height * min_aspect))), alignment_factor)
            delta = max(0, target_width - width)
            padding_area = delta * height
            if delta > 0 and (padding_area / float(original_area)) <= max_padding_area_ratio:
                left = delta // 2
                right = delta - left
                image_obj = ImageOps.expand(image_obj, border=(left, 0, right, 0), fill=(255, 255, 255))
                width = image_obj.width
                height = image_obj.height
                aspect = width / float(height)
                original_area = max(1, width * height)

        if max_aspect > 0.0 and aspect > max_aspect:
            target_height = _round_up(max(height, int(math.ceil(width / max_aspect))), alignment_factor)
            delta = max(0, target_height - height)
            padding_area = delta * width
            if delta > 0 and (padding_area / float(original_area)) <= max_padding_area_ratio:
                top = delta // 2
                bottom = delta - top
                image_obj = ImageOps.expand(image_obj, border=(0, top, 0, bottom), fill=(255, 255, 255))

        # Keep externally-resized OCR pages aligned with Qwen3-VL patch packing.
        width = max(1, int(image_obj.width))
        height = max(1, int(image_obj.height))
        aligned_width = _round_up(width, alignment_factor)
        aligned_height = _round_up(height, alignment_factor)
        if aligned_width != width or aligned_height != height:
            left = max(0, aligned_width - width) // 2
            right = max(0, aligned_width - width) - left
            top = max(0, aligned_height - height) // 2
            bottom = max(0, aligned_height - height) - top
            image_obj = ImageOps.expand(image_obj, border=(left, top, right, bottom), fill=(255, 255, 255))
        return image_obj

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
    else:
        if isinstance(image, torch.Tensor):
            image = torch_to_numpy(image)
        if image.max() < 1:
            image = image * 255.0
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

    if max_pixels is None:
        max_pixels = int(os.environ.get("AGENTOCR_QWEN3_OCR_MAX_PIXELS", "131072"))
    if min_pixels is None:
        min_pixels = int(os.environ.get("AGENTOCR_QWEN3_OCR_MIN_PIXELS", "16384"))
    min_aspect = _get_env_float("AGENTOCR_QWEN3_OCR_MIN_ASPECT", 0.75)
    max_aspect = _get_env_float("AGENTOCR_QWEN3_OCR_MAX_ASPECT", 2.5)
    max_padding_area_ratio = _get_env_float("AGENTOCR_QWEN3_OCR_MAX_PADDING_AREA_RATIO", 0.35)

    # Qwen3-VL expects externally resized images to align with its patch packing.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    resize_factor = max(1, int(patch_size) * max(1, int(temporal_patch_size)))
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=resize_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (resized_width, resized_height) != image.size:
        # OCR history images are mostly rendered text. Preserve edge sharpness when
        # shrinking while still allowing smooth upsampling for short histories.
        is_downscale = resized_width < image.width or resized_height < image.height
        resample = Image.Resampling.LANCZOS if is_downscale else Image.Resampling.BICUBIC
        image = image.resize((resized_width, resized_height), resample)

    # Apply only limited post-resize padding so we do not spend most of the
    # visual budget on white margins when the history becomes very tall.
    image = _pad_image_for_aspect(
        image,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        max_padding_area_ratio=max_padding_area_ratio,
        alignment_factor=resize_factor,
    )

    return image


def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    else:
        size_divisor_ref = size_divisor_rollout
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        return data
    
    if mode == "delete":
        remove_indices = np.arange(bs - remainder, bs, dtype=np.int64)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.arange(to_add, dtype=np.int64) % bs
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return adjusted_batch


def stable_int_seed(*parts: Any, modulo: int = 2**31 - 1) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
