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


class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        print(f"Using EpisodeRewardManager")
        self._print_rng = np.random.RandomState(0)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

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

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
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

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
