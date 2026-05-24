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

from typing import List, Tuple
import re
import math


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINKING_BLOCK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL)
_THOUGHT_BLOCK_RE = re.compile(r"<thought>(.*?)</thought>", re.IGNORECASE | re.DOTALL)
_SUMMARY_BLOCK_RE = re.compile(r"<summary>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
_ACTION_BLOCK_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_ACTION_LINE_FALLBACK_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+action|action)\s*:\s*(.+?)(?:\s*</action>|\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_reasoning_block(original_str: str) -> str:
    """Extract a reasoning block, accepting Qwen3's `<thought>` alias and plain text prefixes."""
    think_match = _THINK_BLOCK_RE.search(original_str)
    if think_match:
        return think_match.group(1).strip()

    thinking_match = _THINKING_BLOCK_RE.search(original_str)
    if thinking_match:
        return thinking_match.group(1).strip()

    thought_match = _THOUGHT_BLOCK_RE.search(original_str)
    if thought_match:
        return thought_match.group(1).strip()

    summary_match = _SUMMARY_BLOCK_RE.search(original_str)
    if summary_match:
        return summary_match.group(1).strip()

    action_match = _ACTION_BLOCK_RE.search(original_str)
    if not action_match:
        return ""

    prefix = original_str[:action_match.start()].strip()
    prefix = re.sub(r"<\|im_(start|end)\|>.*", "", prefix, flags=re.IGNORECASE).strip()
    prefix = re.sub(r"^assistant\s*", "", prefix, flags=re.IGNORECASE).strip()
    return prefix


def _extract_last_tagged_action(original_str: str) -> str:
    matches = list(_ACTION_BLOCK_RE.finditer(original_str))
    if not matches:
        return ""
    return matches[-1].group(1).strip().lower()


def _extract_last_admissible_action(original_str: str, action_pool: List[str]) -> str:
    if not original_str or not action_pool:
        return ""

    lowered = original_str.lower()
    best_idx = -1
    best_action = ""
    for candidate in action_pool:
        lowered_candidate = candidate.lower()
        idx = lowered.rfind(lowered_candidate)
        if idx > best_idx or (idx == best_idx and len(lowered_candidate) > len(best_action)):
            best_idx = idx
            best_action = lowered_candidate
    if best_idx < 0:
        return ""
    return best_action


def _extract_compression_factor(original_str: str) -> float:
    comp_matches = list(
        re.finditer(r"<compression>(.*?)</compression>", original_str, re.IGNORECASE | re.DOTALL)
    )
    if not comp_matches:
        return 1.0

    raw_value = comp_matches[-1].group(1).strip()
    try:
        compression_value = float(raw_value)
    except Exception:
        return 1.0

    if math.isnan(compression_value) or not math.isfinite(compression_value):
        return 1.0
    if compression_value < 1.0:
        return 1.0
    if compression_value > 5.0:
        return 5.0
    return compression_value


def alfworld_projection(actions: List[str], action_pools: List[List[str]], check_compression_tag: bool = False) -> Tuple[List[str], List[int], List[float]]:
    """
    An function to process the actions and compression factors
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    check_compression_tag: whether to check the compression tag, default is False.

    Returns:
        actions: List of extracted actions
        valids: List of validity flags (0 or 1)
        compression_factors: List of compression factors (default 1.0 if not specified)
    """

    valids = [0] * len(actions)
    if check_compression_tag:
        compression_factors = [1.0] * len(actions)  # default compression factor

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        candidate_pool = action_pools[i] if action_pools and i < len(action_pools) else []
        extracted_action = _extract_last_tagged_action(original_str)
        if not extracted_action:
            fallback_match = _ACTION_LINE_FALLBACK_RE.search(original_str)
            if fallback_match:
                extracted_action = fallback_match.group(1).strip().lower()
                extracted_action = extracted_action.splitlines()[0].strip()
        if not extracted_action:
            extracted_action = _extract_last_admissible_action(original_str, candidate_pool)

        actions[i] = extracted_action
        valids[i] = 1 if extracted_action else 0

        # Extract compression factor from <compression>...</compression>
        if check_compression_tag:
            compression_factors[i] = _extract_compression_factor(original_str)

        # Validate against the admissible action pool when available.
        if valids[i] and candidate_pool:
            if actions[i] not in {candidate.lower() for candidate in candidate_pool}:
                valids[i] = 0

    if check_compression_tag:
        return actions, valids, compression_factors
    else:
        return actions, valids
