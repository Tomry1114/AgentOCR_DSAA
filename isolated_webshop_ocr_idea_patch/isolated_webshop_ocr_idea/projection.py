from __future__ import annotations

from typing import List, Tuple, Union
import math
import re


_ACTION_BLOCK_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_ACTION_LINE_FALLBACK_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+action|action)\s*:\s*(.+?)(?:\s*</action>|\s*$)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_last_tagged_action(original_str: str) -> str:
    matches = list(_ACTION_BLOCK_RE.finditer(original_str))
    if not matches:
        return ""
    return matches[-1].group(1).strip().lower()


def _extract_last_available_action(original_str: str, action_pool: List[str]) -> str:
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
    matches = list(
        re.finditer(r"<compression>(.*?)</compression>", original_str, re.IGNORECASE | re.DOTALL)
    )
    if not matches:
        return 1.0

    raw_value = matches[-1].group(1).strip()
    try:
        value = float(raw_value)
    except Exception:
        return 1.0

    if math.isnan(value) or not math.isfinite(value):
        return 1.0
    if value < 1.0:
        return 1.0
    if value > 5.0:
        return 5.0
    return value


def webshop_ocr_projection(
    actions: List[str],
    action_pools: List[List[str]],
    check_compression_tag: bool = False,
) -> Union[Tuple[List[str], List[int]], Tuple[List[str], List[int], List[float]]]:
    valids = [0] * len(actions)
    if check_compression_tag:
        compression_factors = [1.0] * len(actions)

    normalized_actions: List[str] = [""] * len(actions)
    for idx, original_str in enumerate(actions):
        candidate_pool = action_pools[idx] if action_pools and idx < len(action_pools) else []
        extracted_action = _extract_last_tagged_action(original_str)
        if not extracted_action:
            fallback_match = _ACTION_LINE_FALLBACK_RE.search(original_str)
            if fallback_match:
                extracted_action = fallback_match.group(1).strip().lower()
                extracted_action = extracted_action.splitlines()[0].strip()
        if not extracted_action:
            extracted_action = _extract_last_available_action(original_str, candidate_pool)

        normalized_actions[idx] = extracted_action
        valids[idx] = 1 if extracted_action else 0

        if valids[idx] and candidate_pool:
            lowered_pool = {candidate.lower() for candidate in candidate_pool}
            is_search_action = extracted_action.startswith("search[") and extracted_action.endswith("]")
            search_allowed = "search[<your query>]" in lowered_pool
            if extracted_action not in lowered_pool and not (is_search_action and search_allowed):
                valids[idx] = 0

        if check_compression_tag:
            compression_factors[idx] = _extract_compression_factor(original_str)

    if check_compression_tag:
        return normalized_actions, valids, compression_factors
    return normalized_actions, valids
