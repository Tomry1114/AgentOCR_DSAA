# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 AgentOCR Team
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

from verl import DataProto
import torch
import numpy as np


def _debug_non_tensor_value(values, idx: int):
    if values is None:
        return None
    try:
        value = values[idx]
    except Exception:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class EpisodeRewardManager_Compression:
    """The reward manager with compression-aware reward.

    The compression related behavior is controlled via config parameters passed
    in from Hydra. All arguments have sensible defaults so that existing
    experiments remain backward compatible.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        normalize_by_length: bool = False,
        compression_factor_max: float = 3.0,
        compression_reward_coef: float = 0.01,
        compression_failure_penalty_coef: float = 0.0,
        compression_reward_every_n_steps: int = 1,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        self._print_rng = np.random.RandomState(0)

        # Compression related configs
        # self.compression_factor_max = compression_factor_max
        self.compression_reward_coef = compression_reward_coef
        self.compression_failure_penalty_coef = compression_failure_penalty_coef
        self.compression_reward_every_n_steps = compression_reward_every_n_steps
        # assert self.compression_factor_max > 1.0, "compression_factor_max must be greater than 1.0"
        assert self.compression_reward_coef >= 0.0, "compression_reward_coef must be non-negative"
        assert self.compression_failure_penalty_coef >= 0.0, "compression_failure_penalty_coef must be non-negative"
        assert self.compression_reward_every_n_steps >= 1, "compression_reward_every_n_steps must be at least 1"

        # Track global steps for compression reward activation
        self.global_steps = 0

        print(f"Using EpisodeRewardManager_Compression with compression_reward_coef: {self.compression_reward_coef}, compression_failure_penalty_coef: {self.compression_failure_penalty_coef}, compression_reward_every_n_steps: {self.compression_reward_every_n_steps}")
        
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return a dictionary with additional info
        """
        # Check if compression reward should be applied (every n steps)
        self.global_steps += 1
        apply_compression_reward = self.global_steps % self.compression_reward_every_n_steps == 0

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        compression_rewards = []  # Collect all compression_reward values for statistics

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            response_prefix_text = data_item.non_tensor_batch.get('response_prefix_text', "")
            if response_prefix_text:
                response_str = f"{response_prefix_text}{response_str}"

            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            is_success = data_item.non_tensor_batch['is_success']
            compression_factor = data_item.non_tensor_batch['compression_factor']

            # Only apply compression reward every n steps
            if apply_compression_reward:
                if is_success:
                    # compression_reward = (compression_factor - 1.0) * self.compression_reward_coef if compression_factor < self.compression_factor_max else 0.0
                    compression_reward = np.log(compression_factor) * self.compression_reward_coef
                else:
                    # Failed trajectories: optional compression-based penalty.
                    compression_reward = - np.log(compression_factor) * self.compression_failure_penalty_coef
            else:
                # Not at the n-th step, compression reward is 0
                compression_reward = 0.0

            # Collect compression_reward for statistics
            compression_rewards.append(compression_reward)

            # if is_success and episode_rewards <= 0:
            #     print("Miss Match C1")
            # if not is_success and episode_rewards > 0:
            #     print("Miss Match C2")

            if self.normalize_by_length:
                score = (episode_rewards + compression_reward) / episode_lengths
            else:
                score = episode_rewards + compression_reward
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and self._print_rng.random_sample() < 0.1:
                already_print_data_sources[data_source] += 1
                dataset_index = data_item.non_tensor_batch.get('index', None)
                raw_action = data_item.non_tensor_batch.get('debug_model_response_text', None)
                env_action = data_item.non_tensor_batch.get('debug_env_postprocessed_action', None)
                tool_input = data_item.non_tensor_batch.get('debug_env_tool_input', None)
                adapter_reasons = data_item.non_tensor_batch.get('debug_env_idea_adapter_reasons', None)
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                if dataset_index is not None:
                    print(f"[{data_source}][dataset_index]", dataset_index)
                if raw_action is not None:
                    print(f"[{data_source}][debug_model_response_text]", _debug_non_tensor_value(raw_action, 0))
                if env_action is not None:
                    print(f"[{data_source}][debug_env_postprocessed_action]", _debug_non_tensor_value(env_action, 0))
                if tool_input is not None:
                    print(f"[{data_source}][debug_env_tool_input]", _debug_non_tensor_value(tool_input, 0))
                if adapter_reasons is not None:
                    print(f"[{data_source}][debug_env_idea_adapter_reasons]", _debug_non_tensor_value(adapter_reasons, 0))
                print(f"[{data_source}][score]", score)

        # Calculate compression_reward statistics (only when compression reward is applied)
        if compression_rewards:
            compression_rewards_array = np.array(compression_rewards)
            total_count = len(compression_rewards_array)
            positive_count = np.sum(compression_rewards_array > 0)
            negative_count = np.sum(compression_rewards_array < 0)
            zero_count = np.sum(compression_rewards_array == 0)
            
            print(f"compression_reward_positive_ratio: {(positive_count / total_count).item()}, compression_reward_negative_ratio: {(negative_count / total_count).item()}, compression_reward_zero_ratio: {(zero_count / total_count).item()}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
