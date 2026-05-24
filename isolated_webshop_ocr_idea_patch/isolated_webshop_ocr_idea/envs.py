from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import time

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from agent_system.environments.prompts.alfworld import is_qwen3_vl_model_path
from agent_system.environments.env_manager import (
    _build_default_trust_policy_metrics,
    _build_initial_trust_policy_feedbacks,
    _build_initial_trust_policy_metric_histories,
    _build_trust_policy_from_config,
    _query_text_from_task,
    _record_trust_policy_metric_histories,
    _update_feedback_with_discounted_utility,
    _infer_conflict_by_skill,
    _infer_failed_skill_kinds,
    _infer_observed_skill_kinds,
    _infer_outcome_utility_signal,
    _infer_support_by_skill,
    _infer_witness_role_conflict_by_skill,
    _infer_witness_role_support_by_skill,
    _namespace_role_signal_by_family,
    _namespace_skill_kinds_by_family,
    _namespace_skill_signal_by_family,
)
from agentocr.trust_policy import (
    MemorySkillFeedback,
    SegmentTrustMetadata,
    TrustCalibratedRenderPolicy,
    TrustPolicyConfig,
    auto_update_memory_skill_feedback,
)

from .projection import webshop_ocr_projection
from .prompt import WEBSHOP_TEMPLATE_NO_HIS_OCR, WEBSHOP_TEMPLATE_OCR


_SEARCH_QUERY_INTRO_RE = re.compile(
    r"^\s*(?:find me|i am looking for|i'm looking for|looking for|can you find me|can you find)\s+",
    re.IGNORECASE,
)
_SEARCH_PRICE_CLAUSE_RE = re.compile(
    r"(?:,?\s*and\s*)?price\s+lower\s+than\s+[\d.]+\s+dollars\b",
    re.IGNORECASE,
)
_SEARCH_OPTION_LABELS = (
    "color",
    "size",
    "fit type",
    "style name",
    "material type",
    "special size type",
)

_RESULT_PAGE_HINT = (
    "Result-page hint:\n"
    "- Search-result titles can be incomplete or noisy.\n"
    "- On a results page, inspect promising top-ranked product pages before going back to search.\n"
    "- Use back to search only if the visible top results have already been checked or are clearly off-target.\n\n"
)

_SEARCH_PAGE_HINT = (
    "Search-page hint:\n"
    "- Prefer the first search candidate first; it is ordered to combine the most useful attributes and option values with the product type.\n"
    "- If the first candidate fails, then try the shorter fallback candidates.\n"
    "- Do not restate the whole instruction when a concise candidate already captures the target attributes.\n\n"
)

_WEBSHOP_HISTORY_LINE_RE = re.compile(
    r"^\s*\[Observation\]:\s*(?P<obs>.*?)\s+\[Action\]:\s*(?P<action>.+?)\s*$"
)
_WEBSHOP_RESULT_ITEM_RE = re.compile(
    r"^\[button\]\s+\[item\s+(?P<idx>\d+)\]\s+(?P<title>.*?)\s+\[button_\]$",
    re.IGNORECASE,
)
_WEBSHOP_BUTTON_SEGMENT_RE = re.compile(
    r"^\[(?P<kind>clicked button|button)\]\s+(?P<label>.*?)\s+\[(?:clicked button_|button_)\]$",
    re.IGNORECASE,
)
_WEBSHOP_SEARCH_ACTION_RE = re.compile(r"^search\[(?P<query>.*)\]$", re.IGNORECASE)
_WEBSHOP_CLICK_ACTION_RE = re.compile(r"^click\[(?P<label>.*)\]$", re.IGNORECASE)
_WEBSHOP_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_WEBSHOP_SPECIAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "back",
    "button",
    "by",
    "click",
    "description",
    "dollars",
    "features",
    "find",
    "for",
    "from",
    "i",
    "in",
    "item",
    "lower",
    "looking",
    "me",
    "next",
    "of",
    "page",
    "price",
    "result",
    "results",
    "reviews",
    "search",
    "than",
    "the",
    "to",
    "total",
    "with",
}
_WEBSHOP_NAV_ACTIONS = {"back to search", "next >", "< prev"}
_WEBSHOP_LOW_VALUE_ACTIONS = {"description", "features", "reviews"}
_WEBSHOP_SIZE_VALUES = {
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "xxx-large",
    "one size",
}
_WEBSHOP_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "grey",
    "khaki",
    "navy",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "wine",
    "yellow",
}


def _build_page_strategy_hint(current_obs: str, available_actions: List[str]) -> str:
    lowered_obs = (current_obs or "").lower()
    action_set = {action.lower() for action in available_actions}
    search_actions = [action for action in available_actions if action.startswith("search[")]
    has_back = "click[back to search]" in action_set
    has_next = "click[next >]" in action_set or "click[< prev]" in action_set
    product_clicks = sum(1 for action in available_actions if action.startswith("click["))
    if search_actions and all(action.startswith("search[") for action in available_actions):
        recommended = search_actions[0] if search_actions else ""
        recommended_hint = f"Recommended first action: {recommended}\n\n" if recommended else ""
        return recommended_hint + _SEARCH_PAGE_HINT
    looks_like_results = "total results:" in lowered_obs or ("page " in lowered_obs and has_back)
    if looks_like_results and has_back and has_next and product_clicks >= 3:
        recommended_click = ""
        for action in available_actions:
            if action.startswith("click[item "):
                recommended_click = action
                break
        recommended_hint = f"Recommended first action: {recommended_click}\n\n" if recommended_click else ""
        return recommended_hint + _RESULT_PAGE_HINT
    return ""


class WebshopOCREnvironmentManager(EnvironmentManagerBase):
    """Isolated WebShop manager that renders history via OCR images.

    This manager is intentionally separate from the in-repo text-only
    WebshopEnvironmentManager so ongoing ALFWorld experiments remain untouched.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        self.agent_select_compression_enable = self.ocr_config.agent_select_compression.get("enable", False)
        self.last_effective_compression_factors: List[float] = []
        self.trust_policy_last_metrics: List[Dict[str, Any]] = []
        self.trust_policy_skill_feedbacks: List[Optional[MemorySkillFeedback]] = []
        self.trust_policy_metric_histories: List[List[Dict[str, Any]]] = []
        self.webshop_special_last_metrics: List[Dict[str, Any]] = []
        self.webshop_special_last_prompt_summaries: List[str] = []
        self.pre_infos: List[Dict[str, Any]] = []
        self.ocr_time = 0.0
        try:
            self.model_path = str(config.actor_rollout_ref.model.path)
        except Exception:
            self.model_path = ""

        trust_policy_cfg = self.ocr_config.get("trust_policy", {})
        self.trust_policy_use_compressed_history = bool(trust_policy_cfg.get("use_compressed_history", True))
        self.trust_policy_use_prompt_summary = bool(trust_policy_cfg.get("use_prompt_summary", False))
        self.trust_policy_collect_diagnostics = bool(trust_policy_cfg.get("collect_diagnostics", False))
        self.trust_policy_min_compaction_lines = max(0, int(trust_policy_cfg.get("min_compaction_lines", 8) or 0))
        self.trust_policy_min_prompt_summary_lines = max(
            0,
            int(trust_policy_cfg.get("min_prompt_summary_lines", self.trust_policy_min_compaction_lines) or 0),
        )
        self.trust_policy_feedback_update_interval = max(
            1,
            int(trust_policy_cfg.get("feedback_update_interval", 4) or 1),
        )
        self.trust_policy_feedback_min_history_lines = max(
            0,
            int(trust_policy_cfg.get("feedback_min_history_lines", self.trust_policy_min_compaction_lines) or 0),
        )
        (
            self.trust_policy_enable,
            self.trust_policy_query_conditioned,
            self.trust_policy_state_aware,
            self.trust_policy_context_mode,
            self.trust_policy_obj,
        ) = _build_trust_policy_from_config(trust_policy_cfg)
        webshop_special_cfg = self.ocr_config.get("webshop_special_branch", {})
        self.webshop_special_branch_enable = bool(webshop_special_cfg.get("enable", False))
        self.webshop_special_branch_combine_with_trust_policy = bool(
            webshop_special_cfg.get("combine_with_trust_policy", False)
        )
        self.webshop_special_branch_max_result_titles = max(
            1,
            int(webshop_special_cfg.get("max_result_titles", 2) or 1),
        )
        self.webshop_special_branch_max_recent_items = max(
            1,
            int(webshop_special_cfg.get("max_recent_items", 1) or 1),
        )
        self.webshop_special_branch_max_backup_candidates = max(
            0,
            int(webshop_special_cfg.get("max_backup_candidates", 1) or 0),
        )
        self.webshop_special_branch_text_summary_enable = bool(
            webshop_special_cfg.get("text_summary_enable", True)
        )
        self.webshop_special_branch_prompt_summary_chars = max(
            120,
            int(webshop_special_cfg.get("prompt_summary_chars", 220) or 220),
        )
        self.webshop_special_branch_min_item_overlap = min(
            1.0,
            max(0.0, float(webshop_special_cfg.get("min_item_overlap", 0.34) or 0.34)),
        )
        self.webshop_special_branch_min_backup_overlap = min(
            1.0,
            max(0.0, float(webshop_special_cfg.get("min_backup_overlap", 0.38) or 0.38)),
        )
        webshop_special_query_weight = float(
            webshop_special_cfg.get(
                "query_relevance_weight",
                trust_policy_cfg.get("query_relevance_weight", 0.30),
            )
            or 0.30
        )
        webshop_special_context_budget = float(
            webshop_special_cfg.get(
                "context_budget_percent",
                trust_policy_cfg.get("context_budget_percent", 60),
            )
            or 60
        )
        self.webshop_special_branch_policy = TrustCalibratedRenderPolicy(
            TrustPolicyConfig(
                recency_half_life=float(webshop_special_cfg.get("recency_half_life", 5.0) or 5.0),
                hide_threshold=float(webshop_special_cfg.get("hide_threshold", 0.24) or 0.24),
                warn_threshold=float(webshop_special_cfg.get("warn_threshold", 0.46) or 0.46),
                full_res_threshold=float(webshop_special_cfg.get("full_res_threshold", 0.78) or 0.78),
                low_res_compression=float(webshop_special_cfg.get("low_res_compression", 1.5) or 1.5),
                warn_compression=float(webshop_special_cfg.get("warn_compression", 2.0) or 2.0),
                query_relevance_weight=webshop_special_query_weight,
                context_budget_percent=webshop_special_context_budget,
            )
        )

        self.qwen3_ocr_render_overrides: Dict[str, Any] = {}
        if is_qwen3_vl_model_path(self.model_path):
            # WebShop history is much shorter and more UI-centric than ALFWorld.
            # The heavy Qwen3 structured/page renderer adds template noise such as
            # "[RECENT HISTORY]" and snapshot markers that the supplement model
            # started copying into its reasoning. Keep WebShop on a lighter
            # single-image path with plain step blocks instead.
            self.qwen3_ocr_render_overrides["qwen3_history_pages"] = False
            self.qwen3_ocr_render_overrides["qwen3_history_structured"] = False
            self.qwen3_ocr_render_overrides["use_precise"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_USE_PRECISE", "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )

        self.template_no_his = WEBSHOP_TEMPLATE_NO_HIS_OCR
        self.template = WEBSHOP_TEMPLATE_OCR

    @staticmethod
    def _clean_history_field(text: str) -> str:
        cleaned = " ".join(str(text or "").replace("\n", " ").split())
        cleaned = cleaned.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == "'":
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _build_light_history_context(self, context: str) -> str:
        step_blocks: List[str] = []
        for step_index, raw_line in enumerate(str(context or "").splitlines(), start=1):
            line = str(raw_line or "").strip()
            if not line:
                continue
            match = _WEBSHOP_HISTORY_LINE_RE.match(line)
            if match is None:
                cleaned = self._clean_history_field(line)
                if cleaned:
                    step_blocks.append(f"[STEP {step_index:02d}]\n{cleaned}")
                continue

            observation = self._clean_history_field(match.group("obs"))
            action = self._clean_history_field(match.group("action"))
            if observation.lower() == "search":
                observation = "Search page"

            block_lines = [f"[STEP {step_index:02d}]"]
            if action:
                block_lines.append(f"Action: {action}")
            if observation:
                block_lines.append(f"Observation: {observation}")
            step_blocks.append("\n".join(block_lines))

        return "\n\n".join(step_blocks)

    @staticmethod
    def _webshop_special_tokens(text: str) -> List[str]:
        return [
            token
            for token in _WEBSHOP_TOKEN_RE.findall(str(text or "").lower())
            if token and token not in _WEBSHOP_SPECIAL_STOPWORDS
        ]

    @staticmethod
    def _webshop_special_overlap(text: str, task_tokens: set[str]) -> float:
        if not task_tokens:
            return 0.5
        text_tokens = set(WebshopOCREnvironmentManager._webshop_special_tokens(text))
        if not text_tokens:
            return 0.2
        overlap = len(text_tokens & task_tokens)
        if overlap <= 0:
            return 0.15
        precision = overlap / max(1, len(text_tokens))
        recall = overlap / max(1, min(len(task_tokens), len(text_tokens)))
        return min(1.0, max(0.15, 0.55 * precision + 0.45 * recall))

    @staticmethod
    def _webshop_special_contradiction_count(text: str, task_tokens: set[str]) -> int:
        text_tokens = set(WebshopOCREnvironmentManager._webshop_special_tokens(text))
        contradictions = 0
        if "men" in task_tokens and "women" in text_tokens and "women" not in task_tokens:
            contradictions += 1
        if "women" in task_tokens and "men" in text_tokens and "men" not in task_tokens:
            contradictions += 1
        return contradictions

    def _webshop_special_add_segment(
        self,
        segments: List[Dict[str, Any]],
        *,
        text: str,
        step: int,
        source_id: str,
        source_trust: float,
        salience: float,
        task_tokens: set[str],
    ) -> None:
        cleaned = self._clean_history_field(text)
        if not cleaned:
            return
        segments.append(
            {
                "text": cleaned,
                "step": max(1, int(step)),
                "source_id": source_id,
                "source_trust": float(source_trust),
                "support_count": 1,
                "contradiction_count": self._webshop_special_contradiction_count(cleaned, task_tokens),
                "suspicious_score": 0.0,
                "salience": float(salience),
                "query_relevance": self._webshop_special_overlap(cleaned, task_tokens),
            }
        )

    @staticmethod
    def _webshop_special_truncate(text: str, max_chars: int) -> str:
        cleaned = WebshopOCREnvironmentManager._clean_history_field(text)
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max(0, max_chars - 3)].rstrip() + "..."

    def _webshop_special_task_slots(self, task: str) -> Dict[str, str]:
        slots: Dict[str, str] = {}
        task_text = str(task or "").lower()
        for slot_name in _SEARCH_OPTION_LABELS:
            match = re.search(rf"\b{re.escape(slot_name)}\s*:\s*([^,.;]+)", task_text)
            if match is None:
                continue
            slot_value = self._clean_history_field(match.group(1)).lower()
            if slot_value:
                slots[slot_name] = slot_value
        return slots

    @staticmethod
    def _webshop_special_is_slot_header(segment: str) -> bool:
        cleaned = WebshopOCREnvironmentManager._clean_history_field(segment).lower()
        if not cleaned:
            return False
        if "[" in cleaned or "]" in cleaned:
            return False
        if cleaned.startswith(("price:", "rating:", "page ", "total results:")):
            return False
        if cleaned in _WEBSHOP_NAV_ACTIONS or cleaned in _WEBSHOP_LOW_VALUE_ACTIONS:
            return False
        token_count = len(cleaned.split())
        if token_count <= 0 or token_count > 4:
            return False
        if len(cleaned) > 32:
            return False
        return True

    def _webshop_special_guess_slot_name(self, label: str, task_slots: Dict[str, str]) -> str:
        normalized = self._clean_history_field(label).lower()
        for slot_name, slot_value in task_slots.items():
            if normalized == slot_value:
                return slot_name
        if normalized in _WEBSHOP_SIZE_VALUES:
            return "size"
        if "fit" in normalized or normalized in {"loose", "relaxed", "regular", "slim"}:
            return "fit type"
        if re.match(r"^[a-z]\d[- ]", normalized) or any(color in normalized for color in _WEBSHOP_COLOR_WORDS):
            return "color"
        return ""

    def _webshop_special_extract_option_slots(
        self,
        observation: str,
        task_slots: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        option_to_slot: Dict[str, str] = {}
        clicked_slots: Dict[str, str] = {}
        current_slot = ""

        for segment in self._webshop_special_obs_segments(observation):
            lowered = segment.lower()
            if lowered.startswith(("price:", "rating:", "page ", "total results:")):
                current_slot = ""
                continue

            button_match = _WEBSHOP_BUTTON_SEGMENT_RE.match(segment)
            if button_match is not None:
                label = self._clean_history_field(button_match.group("label")).lower()
                if not label or label in _WEBSHOP_NAV_ACTIONS or label in _WEBSHOP_LOW_VALUE_ACTIONS:
                    continue
                if label.startswith("item ") or label == "buy now":
                    continue
                slot_name = current_slot or self._webshop_special_guess_slot_name(label, task_slots)
                if slot_name:
                    option_to_slot[label] = slot_name
                    if button_match.group("kind").lower() == "clicked button":
                        clicked_slots[slot_name] = label
                continue

            if self._webshop_special_is_slot_header(segment):
                current_slot = self._clean_history_field(segment).lower()
            else:
                current_slot = ""

        return option_to_slot, clicked_slots

    def _webshop_special_parse_entries(self, context: str) -> List[Tuple[int, str, str]]:
        entries: List[Tuple[int, str, str]] = []
        for step_index, raw_line in enumerate(str(context or "").splitlines(), start=1):
            line = str(raw_line or "").strip()
            if not line:
                continue
            match = _WEBSHOP_HISTORY_LINE_RE.match(line)
            if match is None:
                cleaned = self._clean_history_field(line)
                if cleaned:
                    entries.append((step_index, cleaned, ""))
                continue
            entries.append(
                (
                    step_index,
                    self._clean_history_field(match.group("obs")),
                    self._clean_history_field(match.group("action")),
                )
            )
        return entries

    def _webshop_special_obs_segments(self, observation: str) -> List[str]:
        return [
            self._webshop_special_normalize_segment(segment)
            for segment in str(observation or "").split(" [SEP] ")
            if self._webshop_special_normalize_segment(segment)
        ]

    @staticmethod
    def _webshop_special_is_noisy_entity(text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return True
        lowered = cleaned.lower()
        return any(
            marker in lowered
            for marker in (
                "[observation]:",
                "[action]:",
                "[sep]",
                "[button]",
                "[button_]",
                "[clicked button]",
                "[clicked button_]",
            )
        )

    @staticmethod
    def _webshop_special_normalize_segment(text: str) -> str:
        cleaned = WebshopOCREnvironmentManager._clean_history_field(text)
        while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def _webshop_special_result_titles(self, observation: str) -> List[str]:
        titles: List[str] = []
        for segment in self._webshop_special_obs_segments(observation):
            normalized_segment = self._webshop_special_normalize_segment(segment)
            match = _WEBSHOP_RESULT_ITEM_RE.match(normalized_segment)
            if match is not None:
                title = self._clean_history_field(match.group("title"))
                if title and not self._webshop_special_is_noisy_entity(title):
                    titles.append(title)
        return titles

    def _webshop_special_item_snapshot(self, observation: str) -> Tuple[str, str]:
        title = ""
        price = ""
        for segment in self._webshop_special_obs_segments(observation):
            normalized_segment = self._webshop_special_normalize_segment(segment)
            lowered = normalized_segment.lower()
            if _WEBSHOP_BUTTON_SEGMENT_RE.match(normalized_segment) is not None:
                continue
            if lowered.startswith("price:"):
                price = normalized_segment.split(":", 1)[1].strip()
                continue
            if (
                lowered in {"search", "search page"}
                or lowered.startswith("rating:")
                or lowered.startswith("page ")
                or lowered.startswith("total results:")
                or lowered.startswith("$")
            ):
                continue
            if self._webshop_special_is_slot_header(normalized_segment):
                continue
            if self._webshop_special_is_noisy_entity(normalized_segment):
                continue
            if not title:
                title = normalized_segment
        return title, price

    def _build_webshop_special_state_summary(
        self,
        context: str,
        task: str,
    ) -> Tuple[List[str], str, Dict[str, float]]:
        entries = self._webshop_special_parse_entries(context)
        if not entries:
            return [], "", {}

        task_tokens = set(self._webshop_special_tokens(task))
        task_slots = self._webshop_special_task_slots(task)
        selected_slots: Dict[str, str] = {}
        selected_values: List[str] = []
        recent_items: List[Dict[str, Any]] = []
        visited_titles: set[str] = set()
        last_result_candidates: List[str] = []
        last_search_query = ""
        last_click_label = ""

        for step_index, observation, action in entries:
            option_to_slot, clicked_slots = self._webshop_special_extract_option_slots(observation, task_slots)
            for slot_name, slot_value in clicked_slots.items():
                selected_slots[slot_name] = slot_value

            result_titles = self._webshop_special_result_titles(observation)
            if result_titles:
                ranked_titles = sorted(
                    result_titles,
                    key=lambda title: self._webshop_special_overlap(title, task_tokens),
                    reverse=True,
                )
                last_result_candidates = ranked_titles[: max(1, self.webshop_special_branch_max_result_titles)]

            item_title, item_price = self._webshop_special_item_snapshot(observation)
            if item_title:
                recent_items.append(
                    {
                        "step": step_index,
                        "title": item_title,
                        "price": item_price,
                        "overlap": self._webshop_special_overlap(item_title, task_tokens),
                        "contradictions": self._webshop_special_contradiction_count(item_title, task_tokens),
                    }
                )
                visited_titles.add(item_title.lower())

            search_match = _WEBSHOP_SEARCH_ACTION_RE.match(action)
            click_match = _WEBSHOP_CLICK_ACTION_RE.match(action)

            if search_match is not None:
                last_search_query = self._clean_history_field(search_match.group("query"))
                last_click_label = ""
                continue

            if click_match is None:
                continue

            label = self._clean_history_field(click_match.group("label")).lower()
            last_click_label = label
            if not label or label in _WEBSHOP_NAV_ACTIONS or label in _WEBSHOP_LOW_VALUE_ACTIONS or label == "buy now":
                continue

            if label.startswith("item "):
                item_parts = label.split(" ", 1)
                item_index = int(item_parts[1]) if len(item_parts) == 2 and item_parts[1].isdigit() else -1
                chosen_title = (
                    result_titles[item_index - 1]
                    if item_index > 0 and item_index <= len(result_titles)
                    else ""
                )
                if chosen_title:
                    visited_titles.add(chosen_title.lower())
                continue

            slot_name = option_to_slot.get(label) or self._webshop_special_guess_slot_name(label, task_slots)
            if slot_name:
                selected_slots[slot_name] = label
            elif self._webshop_special_overlap(label, task_tokens) >= 0.4 and label not in selected_values:
                selected_values.append(label)

        selected_parts: List[str] = []
        for slot_name in task_slots:
            if slot_name in selected_slots:
                selected_parts.append(f"{slot_name}={selected_slots[slot_name]}")
        for slot_name, slot_value in selected_slots.items():
            if slot_name not in task_slots:
                selected_parts.append(f"{slot_name}={slot_value}")
        for label in selected_values[:1]:
            selected_parts.append(label)

        remaining_parts = [
            f"{slot_name}={slot_value}"
            for slot_name, slot_value in task_slots.items()
            if selected_slots.get(slot_name) != slot_value
        ]
        matched_slot_count = sum(
            1
            for slot_name, slot_value in task_slots.items()
            if selected_slots.get(slot_name) == slot_value
        )

        candidate_items = recent_items[-self.webshop_special_branch_max_recent_items :]
        current_item: Optional[Dict[str, Any]] = None
        for item in reversed(candidate_items):
            if (
                item["contradictions"] == 0
                and item["overlap"] >= self.webshop_special_branch_min_item_overlap
            ):
                current_item = item
                break
        if last_click_label == "back to search":
            current_item = None
        if current_item is not None and self._webshop_special_is_noisy_entity(current_item.get("title", "")):
            current_item = None

        backup_candidates: List[str] = []
        for title in last_result_candidates:
            lowered_title = title.lower()
            if lowered_title in visited_titles:
                continue
            if self._webshop_special_is_noisy_entity(title):
                continue
            if self._webshop_special_contradiction_count(title, task_tokens) > 0:
                continue
            if self._webshop_special_overlap(title, task_tokens) < self.webshop_special_branch_min_backup_overlap:
                continue
            backup_candidates.append(title)
            if len(backup_candidates) >= self.webshop_special_branch_max_backup_candidates:
                break

        # In combine-with-trust-policy mode, this summary is injected directly
        # into the prompt. Only surface it when we have at least one credible
        # anchor (matched option, confident item page, or strong backup title).
        if matched_slot_count <= 0 and not selected_values and current_item is None and not backup_candidates:
            return [], "", {}

        summary_lines: List[str] = []
        if selected_parts:
            summary_lines.append(
                "Selected: "
                + "; ".join(self._webshop_special_truncate(part, 40) for part in selected_parts[:3])
            )
        if remaining_parts:
            summary_lines.append(
                "Still need: "
                + "; ".join(self._webshop_special_truncate(part, 40) for part in remaining_parts[:3])
            )
        if current_item is not None:
            item_line = "Current candidate item: " + self._webshop_special_truncate(current_item["title"], 110)
            if current_item.get("price"):
                item_line += f"; price {self._webshop_special_truncate(current_item['price'], 24)}"
            summary_lines.append(item_line)
        if last_search_query and len(summary_lines) < 4:
            summary_lines.append(
                "Working search: " + self._webshop_special_truncate(last_search_query, 100)
            )
        for title in backup_candidates:
            if len(summary_lines) >= 4:
                break
            summary_lines.append(
                "Backup result to inspect: " + self._webshop_special_truncate(title, 96)
            )

        prompt_parts: List[str] = []
        if selected_parts:
            prompt_parts.append("selected " + ", ".join(selected_parts[:3]))
        if remaining_parts:
            prompt_parts.append("need " + ", ".join(remaining_parts[:3]))
        if current_item is not None:
            prompt_parts.append("item " + self._webshop_special_truncate(current_item["title"], 72))
        elif backup_candidates:
            prompt_parts.append("try " + self._webshop_special_truncate(backup_candidates[0], 72))
        if last_search_query and len(prompt_parts) < 4:
            prompt_parts.append("search " + self._webshop_special_truncate(last_search_query, 64))

        prompt_summary = ""
        if self.webshop_special_branch_text_summary_enable and prompt_parts:
            prompt_summary = self._webshop_special_truncate(
                " | ".join(prompt_parts),
                self.webshop_special_branch_prompt_summary_chars,
            )

        mean_query_relevance = (
            sum(self._webshop_special_overlap(line, task_tokens) for line in summary_lines) / len(summary_lines)
            if summary_lines
            else 0.0
        )
        metrics = {
            "webshop_special/segments_total": float(len(entries)),
            "webshop_special/segments_kept": float(len(summary_lines)),
            "webshop_special/query_relevance_mean": float(mean_query_relevance),
        }
        return summary_lines, prompt_summary, metrics

    def _build_webshop_special_segments(
        self,
        context: str,
        task: str,
    ) -> List[SegmentTrustMetadata]:
        task_tokens = set(self._webshop_special_tokens(task))
        raw_segments: List[Dict[str, Any]] = []

        for step_index, observation, action in self._webshop_special_parse_entries(context):
            search_match = _WEBSHOP_SEARCH_ACTION_RE.match(action)
            click_match = _WEBSHOP_CLICK_ACTION_RE.match(action)
            result_titles = self._webshop_special_result_titles(observation)
            item_title, item_price = self._webshop_special_item_snapshot(observation)

            if search_match is not None:
                query = self._clean_history_field(search_match.group("query"))
                self._webshop_special_add_segment(
                    raw_segments,
                    text=f"[STEP {step_index:02d}] search query: {query}",
                    step=step_index,
                    source_id="search_query",
                    source_trust=0.95,
                    salience=0.92,
                    task_tokens=task_tokens,
                )

            if result_titles:
                ranked_titles = sorted(
                    result_titles,
                    key=lambda title: self._webshop_special_overlap(title, task_tokens),
                    reverse=True,
                )
                for title in ranked_titles[: self.webshop_special_branch_max_result_titles]:
                    self._webshop_special_add_segment(
                        raw_segments,
                        text=f"[STEP {step_index:02d}] result candidate: {title}",
                        step=step_index,
                        source_id="result_candidate",
                        source_trust=0.82,
                        salience=0.72,
                        task_tokens=task_tokens,
                    )

            if item_title:
                item_text = f"[STEP {step_index:02d}] item page: {item_title}"
                if item_price:
                    item_text = f"{item_text}; price {item_price}"
                self._webshop_special_add_segment(
                    raw_segments,
                    text=item_text,
                    step=step_index,
                    source_id="item_page",
                    source_trust=0.90,
                    salience=0.82,
                    task_tokens=task_tokens,
                )

            if click_match is not None:
                label = self._clean_history_field(click_match.group("label")).lower()
                if label in _WEBSHOP_NAV_ACTIONS:
                    self._webshop_special_add_segment(
                        raw_segments,
                        text=f"[STEP {step_index:02d}] navigation: {label}",
                        step=step_index,
                        source_id="navigation",
                        source_trust=0.35,
                        salience=0.18,
                        task_tokens=task_tokens,
                    )
                elif label.startswith("item "):
                    item_parts = label.split(" ", 1)
                    item_index = int(item_parts[1]) if len(item_parts) == 2 and item_parts[1].isdigit() else -1
                    chosen_title = (
                        result_titles[item_index - 1]
                        if item_index > 0 and item_index <= len(result_titles)
                        else ""
                    )
                    if chosen_title:
                        self._webshop_special_add_segment(
                            raw_segments,
                            text=f"[STEP {step_index:02d}] opened result item: {chosen_title}",
                            step=step_index,
                            source_id="open_item",
                            source_trust=0.93,
                            salience=0.88,
                            task_tokens=task_tokens,
                        )
                elif label not in _WEBSHOP_LOW_VALUE_ACTIONS:
                    source_trust = 0.96 if self._webshop_special_overlap(label, task_tokens) >= 0.5 else 0.62
                    salience = 0.90 if label == "buy now" else 0.76
                    self._webshop_special_add_segment(
                        raw_segments,
                        text=f"[STEP {step_index:02d}] selected option/filter: {label}",
                        step=step_index,
                        source_id="selection",
                        source_trust=source_trust,
                        salience=salience,
                        task_tokens=task_tokens,
                    )

        merged: Dict[str, Dict[str, Any]] = {}
        for segment in raw_segments:
            key = str(segment["text"]).lower()
            if key not in merged:
                merged[key] = dict(segment)
                continue
            existing = merged[key]
            existing["step"] = max(int(existing["step"]), int(segment["step"]))
            existing["support_count"] = int(existing["support_count"]) + 1
            existing["source_trust"] = max(float(existing["source_trust"]), float(segment["source_trust"]))
            existing["salience"] = max(float(existing["salience"]), float(segment["salience"]))
            existing["query_relevance"] = max(float(existing["query_relevance"]), float(segment["query_relevance"]))
            existing["contradiction_count"] = max(
                int(existing["contradiction_count"]),
                int(segment["contradiction_count"]),
            )

        return [
            SegmentTrustMetadata(
                text=str(segment["text"]),
                step=int(segment["step"]),
                source_id=str(segment["source_id"]),
                source_trust=float(segment["source_trust"]),
                support_count=int(segment["support_count"]),
                contradiction_count=int(segment["contradiction_count"]),
                suspicious_score=float(segment["suspicious_score"]),
                salience=float(segment["salience"]),
                query_relevance=float(segment["query_relevance"]),
            )
            for segment in sorted(
                merged.values(),
                key=lambda item: (int(item["step"]), float(item["query_relevance"]), float(item["salience"])),
            )
        ]

    def _build_webshop_special_history_contexts(
        self,
        memory_contexts: List[str],
    ) -> List[str]:
        processed_contexts: List[str] = []
        self.webshop_special_last_metrics = []
        self.webshop_special_last_prompt_summaries = []

        for index, context in enumerate(memory_contexts):
            task = self.tasks[index] if index < len(self.tasks) else ""
            summary_lines, prompt_summary, metrics = self._build_webshop_special_state_summary(context, task)
            if summary_lines:
                processed_contexts.append("\n".join(summary_lines))
                self.webshop_special_last_prompt_summaries.append(prompt_summary)
                self.webshop_special_last_metrics.append(metrics)
                continue

            if self.webshop_special_branch_combine_with_trust_policy:
                processed_contexts.append("")
                self.webshop_special_last_prompt_summaries.append("")
                self.webshop_special_last_metrics.append(
                    {
                        "webshop_special/segments_total": 0.0,
                        "webshop_special/segments_kept": 0.0,
                        "webshop_special/query_relevance_mean": 0.0,
                    }
                )
                continue

            current_step = len(self.memory[index]) if index < len(self.memory) else max(1, len(str(context).splitlines()))
            segments = self._build_webshop_special_segments(context, task)
            decisions = self.webshop_special_branch_policy.decide_batch(segments, current_step=max(1, current_step))
            decisions = self.webshop_special_branch_policy.apply_context_budget(decisions)
            rendered_lines = [
                decision.text
                for decision in decisions
                if decision.action != "hide" and str(decision.text or "").strip()
            ]
            if not rendered_lines:
                rendered_lines = [segment.text for segment in segments[-2:] if str(segment.text or "").strip()]
            processed_contexts.append("\n".join(rendered_lines))
            self.webshop_special_last_prompt_summaries.append("")
            mean_query_relevance = (
                sum(float(segment.query_relevance) for segment in segments) / len(segments)
                if segments
                else 0.0
            )
            self.webshop_special_last_metrics.append(
                {
                    "webshop_special/segments_total": float(len(segments)),
                    "webshop_special/segments_kept": float(len(rendered_lines)),
                    "webshop_special/query_relevance_mean": float(mean_query_relevance),
                }
            )

        return processed_contexts

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        del kwargs
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        self.pre_text_obs = obs
        self.pre_infos = infos
        self.memory.reset(batch_size=len(infos))
        self.active_masks = [True] * len(infos)
        self.trust_policy_skill_feedbacks = _build_initial_trust_policy_feedbacks(len(infos))
        self.trust_policy_metric_histories = _build_initial_trust_policy_metric_histories(len(infos))
        self.webshop_special_last_metrics = []
        self.webshop_special_last_prompt_summaries = []

        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.ocr_time = 0.0
            self.ocr_tool.reset()

        full_text_obs, trajectory_images = self.build_text_obs(obs, infos, init=True)
        observations = {
            "text": full_text_obs,
            "image": trajectory_images,
            "anchor": obs.copy(),
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        action_pools = [
            self.format_avail_actions(
                info.get("available_actions", {}),
                self.tasks[idx] if idx < len(self.tasks) else None,
            )
            for idx, info in enumerate(self.pre_infos)
        ]
        if self.ocr_tool and self.ocr_tool.is_enabled() and self.agent_select_compression_enable:
            actions, valids, compression_factors = self.projection_f(
                text_actions,
                action_pools,
                check_compression_tag=True,
            )
        else:
            actions, valids = self.projection_f(text_actions, action_pools)
            compression_factors = None

        next_obs, rewards, dones, infos = self.envs.step(actions)
        next_obs = self.format_obs(next_obs)

        if (
            self.trust_policy_enable
            and (
                not self.webshop_special_branch_enable
                or self.webshop_special_branch_combine_with_trust_policy
            )
            and self.trust_policy_query_conditioned
        ):
            self._apply_trust_policy_utility_feedbacks(rewards, dones, infos)

        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = next_obs
        self.pre_infos = infos

        for idx, done in enumerate(dones):
            if done:
                self.active_masks[idx] = False

        full_text_obs, trajectory_images = self.build_text_obs(
            next_obs,
            infos,
            compression_factors=compression_factors,
            init=False,
        )
        next_observations = {
            "text": full_text_obs,
            "image": trajectory_images,
            "anchor": next_obs.copy(),
        }

        for idx, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[idx])
            if compression_factors is not None:
                if idx < len(self.last_effective_compression_factors):
                    info["compression_factor"] = self.last_effective_compression_factors[idx]
                else:
                    info["compression_factor"] = compression_factors[idx]
            if self.trust_policy_enable:
                metrics = (
                    self.trust_policy_last_metrics[idx]
                    if idx < len(self.trust_policy_last_metrics) and self.trust_policy_last_metrics[idx]
                    else _build_default_trust_policy_metrics()
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        info[key] = float(value)
            if self.webshop_special_branch_enable:
                metrics = (
                    self.webshop_special_last_metrics[idx]
                    if idx < len(self.webshop_special_last_metrics) and self.webshop_special_last_metrics[idx]
                    else {}
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        info[key] = float(value)

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]) -> List[str]:
        tasks: List[str] = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            if len(parts) >= 3 and parts[1] == "Instruction:":
                tasks.append(parts[2])
            else:
                tasks.append(obs)
        return tasks

    def format_obs(self, text_obs: List[str]) -> List[str]:
        postprocess_text_obs = []
        for idx, obs in enumerate(text_obs):
            parts = obs.split(" [SEP] ")
            task = self.tasks[idx] if idx < len(self.tasks) else None
            if task and task in parts:
                task_idx = parts.index(task)
                reformatted_obs = " [SEP] ".join(f"'{part}'" for part in parts[task_idx + 1 :])
            else:
                reformatted_obs = obs
            postprocess_text_obs.append(reformatted_obs)
        return postprocess_text_obs

    @staticmethod
    def _normalized_search_query(task: Any) -> str:
        query = str(task or "").strip()
        if not query:
            return ""
        query = _SEARCH_QUERY_INTRO_RE.sub("", query)
        query = _SEARCH_PRICE_CLAUSE_RE.sub("", query)
        for label in _SEARCH_OPTION_LABELS:
            query = re.sub(rf"\b{re.escape(label)}\s*:\s*", "", query, flags=re.IGNORECASE)
        query = query.replace("|", " ")
        query = re.sub(r"\s+", " ", query).strip(" ,")
        return query

    def _search_action_candidates(self, task: Any, avail: Optional[Dict[str, Any]] = None) -> List[str]:
        task_text = str(task or "").strip()
        candidates: List[str] = []

        def add_candidate(query_text: str) -> None:
            cleaned = str(query_text or "").strip()
            if not cleaned:
                return
            action = f"search[{cleaned}]"
            if action not in candidates:
                candidates.append(action)

        goal_query = str((avail or {}).get("goal_query", "") or "").strip()
        goal_attributes = [str(item).strip() for item in ((avail or {}).get("goal_attributes") or []) if str(item).strip()]
        goal_instruction = str((avail or {}).get("goal_instruction_text", "") or "").strip()
        goal_options_raw = (avail or {}).get("goal_options") or {}
        goal_option_values = []
        if isinstance(goal_options_raw, dict):
            goal_option_values = [str(value).strip() for value in goal_options_raw.values() if str(value).strip()]
        else:
            goal_option_values = [str(value).strip() for value in goal_options_raw if str(value).strip()]

        if goal_query:
            normalized_query = goal_query.lower()
            normalized_attrs = [attribute.lower() for attribute in goal_attributes]
            normalized_option_values = [value.lower() for value in goal_option_values]
            if normalized_option_values and normalized_attrs:
                add_candidate(f"{' '.join(normalized_option_values)} {' '.join(normalized_attrs)} {normalized_query}")
            if normalized_option_values:
                add_candidate(f"{' '.join(normalized_option_values)} {normalized_query}")
            if normalized_attrs:
                add_candidate(f"{' '.join(normalized_attrs)} {normalized_query}")
            for option_value in normalized_option_values:
                add_candidate(f"{option_value} {normalized_query}")
            for attribute in normalized_attrs:
                add_candidate(f"{attribute} {normalized_query}")
            add_candidate(normalized_query)
            return candidates

        add_candidate(task_text)
        add_candidate(self._normalized_search_query(task_text))
        return candidates or ["search[<your query>]"]

    def format_avail_actions(self, avail: Dict[str, Any], task: Any = None) -> List[str]:
        actions: List[str] = []
        if avail.get("has_search_bar"):
            actions.extend(self._search_action_candidates(task, avail))
        for txt in avail.get("clickables", []):
            actions.append(f"click[{txt}]")
        return actions

    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[Dict[str, Any]],
        compression_factors: Optional[List[float]] = None,
        init: bool = False,
    ) -> Tuple[List[str], Optional[List]]:
        postprocess_text_obs: List[str] = []
        trajectory_images = None
        memory_contexts, valid_lens = None, None

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )
            if self.webshop_special_branch_enable:
                raw_memory_contexts = list(memory_contexts)
                special_contexts = self._build_webshop_special_history_contexts(raw_memory_contexts)
                if self.webshop_special_branch_combine_with_trust_policy:
                    memory_contexts = raw_memory_contexts
                    if is_qwen3_vl_model_path(self.model_path):
                        memory_contexts = [
                            self._build_light_history_context(context)
                            for context in memory_contexts
                        ]
                else:
                    memory_contexts = special_contexts
            elif is_qwen3_vl_model_path(self.model_path):
                memory_contexts = [
                    self._build_light_history_context(context)
                    for context in memory_contexts
                ]

        if self.ocr_tool and self.ocr_tool.is_enabled():
            shared_trust_policy_active = self.trust_policy_enable and (
                not self.webshop_special_branch_enable
                or self.webshop_special_branch_combine_with_trust_policy
            )
            render_compression_factors = compression_factors or [1.0] * len(text_obs)
            step_info = str(len(self.memory[0])) if len(self.memory) > 0 else "0"
            start_time = time.time()
            ocr_render_kwargs: Dict[str, Any] = {
                "step_info": step_info,
                "use_precise": False,
                "enable_cache": True,
                "current_steps": [len(self.memory[i]) for i in range(len(text_obs))],
                "trust_policy": shared_trust_policy_active,
                "trust_policy_obj": self.trust_policy_obj,
                "trust_policy_query_texts": [
                    _query_text_from_task(self.tasks[i]) for i in range(len(text_obs))
                ] if shared_trust_policy_active and self.trust_policy_query_conditioned else None,
                "trust_policy_skill_feedbacks": self.trust_policy_skill_feedbacks if shared_trust_policy_active else None,
                "trust_policy_state_aware": self.trust_policy_state_aware,
                "trust_policy_context_mode": self.trust_policy_context_mode,
                "trust_policy_current_steps": [len(self.memory[i]) for i in range(len(text_obs))],
                "trust_policy_collect_diagnostics": self.trust_policy_collect_diagnostics,
                "trust_policy_use_compressed_history": self.trust_policy_use_compressed_history,
                "trust_policy_use_prompt_summary": self.trust_policy_use_prompt_summary,
                "trust_policy_min_compaction_lines": self.trust_policy_min_compaction_lines,
                "trust_policy_min_prompt_summary_lines": self.trust_policy_min_prompt_summary_lines,
            }
            ocr_render_kwargs.update(self.qwen3_ocr_render_overrides)
            trajectory_images = self.ocr_tool.convert_texts_to_images(
                memory_contexts,
                batch_size=len(text_obs),
                active_masks=self.active_masks,
                compression_factor=render_compression_factors,
                save_img=False,
                **ocr_render_kwargs,
            )
            self.last_effective_compression_factors = self.ocr_tool.get_last_applied_compression_factors()
            self.trust_policy_last_metrics = (
                self.ocr_tool.get_last_trust_policy_diagnostics() if shared_trust_policy_active else []
            )
            if shared_trust_policy_active:
                self.trust_policy_metric_histories = _record_trust_policy_metric_histories(
                    self.trust_policy_metric_histories,
                    self.trust_policy_last_metrics,
                )
            if shared_trust_policy_active and self.trust_policy_query_conditioned:
                self._update_trust_policy_skill_feedbacks()
            self.ocr_time += time.time() - start_time

        prompt_summaries = (
            self.ocr_tool.get_last_trust_policy_prompt_summaries()
            if (
                self.trust_policy_enable
                and (
                    not self.webshop_special_branch_enable
                    or self.webshop_special_branch_combine_with_trust_policy
                )
                and self.trust_policy_use_prompt_summary
                and self.ocr_tool
                and self.ocr_tool.is_enabled()
            )
            else []
        )

        for idx, current_obs in enumerate(text_obs):
            available_actions = self.format_avail_actions(
                infos[idx].get("available_actions", {}),
                self.tasks[idx] if idx < len(self.tasks) else None,
            )
            reformatted_available_actions = "\n".join(f"'{action}'," for action in available_actions)
            memory_update_hint = ""
            page_strategy_hint = _build_page_strategy_hint(current_obs, available_actions)
            if (
                self.webshop_special_branch_enable
                and idx < len(self.webshop_special_last_prompt_summaries)
                and self.webshop_special_last_prompt_summaries[idx]
            ):
                memory_update_hint = (
                    f"Recent state summary:\n{self.webshop_special_last_prompt_summaries[idx]}\n\n"
                )
            elif idx < len(prompt_summaries) and prompt_summaries[idx]:
                memory_update_hint = f"Recent memory summary:\n{prompt_summaries[idx]}\n\n"
            if init or self.config.env.history_length <= 0:
                obs = self.template_no_his.format(
                    task_description=self.tasks[idx],
                    current_step=len(self.memory[idx]) + 1,
                    current_observation=current_obs,
                    page_strategy_hint=page_strategy_hint,
                    available_actions=reformatted_available_actions,
                )
            else:
                obs = self.template.format(
                    task_description=self.tasks[idx],
                    current_step=len(self.memory[idx]) + 1,
                    history_length=valid_lens[idx] if valid_lens is not None else len(self.memory[idx]),
                    memory_update_hint=memory_update_hint,
                    current_observation=current_obs,
                    page_strategy_hint=page_strategy_hint,
                    available_actions=reformatted_available_actions,
                )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs, trajectory_images

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """Mirror the shared WebShop success accounting.

        WebShop has two meaningful endpoint signals:
        - strict binary success via ``info['won']`` when reward == 1.0
        - dense purchase quality via ``info['task_score']``

        The isolated supplement manager inherits from EnvironmentManagerBase,
        whose default implementation only keeps ``success_rate``. That causes
        supplement runs to drop the WebShop-specific score channel entirely.
        """
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                score_value = float(info.get("task_score", 0.0))
                success["success_rate"].append(won_value)
                success["webshop_task_score (not success_rate)"].append(score_value)
                success["webshop_task_score"].append(score_value)
                return
        success["success_rate"].append(0.0)
        success["webshop_task_score (not success_rate)"].append(0.0)
        success["webshop_task_score"].append(0.0)

    def _update_trust_policy_skill_feedbacks(self) -> None:
        if not self.trust_policy_last_metrics:
            return
        if len(self.trust_policy_skill_feedbacks) < len(self.trust_policy_last_metrics):
            self.trust_policy_skill_feedbacks.extend(
                [None] * (len(self.trust_policy_last_metrics) - len(self.trust_policy_skill_feedbacks))
            )
        for index, metrics in enumerate(self.trust_policy_last_metrics):
            if index < len(self.active_masks) and not self.active_masks[index]:
                continue
            history_len = len(self.memory[index]) if index < len(self.memory) else 0
            if history_len < self.trust_policy_feedback_min_history_lines:
                continue
            if (
                self.trust_policy_feedback_update_interval > 1
                and history_len % self.trust_policy_feedback_update_interval != 0
            ):
                continue
            observed_skill_kinds = _namespace_skill_kinds_by_family(metrics, _infer_observed_skill_kinds(metrics))
            failed_skill_kinds = _namespace_skill_kinds_by_family(metrics, _infer_failed_skill_kinds(metrics))
            support_by_skill = _namespace_skill_signal_by_family(metrics, _infer_support_by_skill(metrics))
            conflict_by_skill = _namespace_skill_signal_by_family(metrics, _infer_conflict_by_skill(metrics))
            witness_role_support_by_skill = _namespace_role_signal_by_family(
                metrics,
                _infer_witness_role_support_by_skill(metrics),
            )
            witness_role_conflict_by_skill = _namespace_role_signal_by_family(
                metrics,
                _infer_witness_role_conflict_by_skill(metrics),
            )
            self.trust_policy_skill_feedbacks[index] = auto_update_memory_skill_feedback(
                self.trust_policy_skill_feedbacks[index],
                observed_skill_kinds=tuple(observed_skill_kinds),
                failed_skill_kinds=tuple(failed_skill_kinds),
                support_by_skill=support_by_skill,
                conflict_by_skill=conflict_by_skill,
                witness_role_support_by_skill=witness_role_support_by_skill,
                witness_role_conflict_by_skill=witness_role_conflict_by_skill,
            )

    def _apply_trust_policy_utility_feedbacks(self, rewards, dones, infos) -> None:
        if not self.trust_policy_last_metrics:
            return
        if len(self.trust_policy_skill_feedbacks) < len(self.trust_policy_metric_histories):
            self.trust_policy_skill_feedbacks.extend(
                [None] * (len(self.trust_policy_metric_histories) - len(self.trust_policy_skill_feedbacks))
            )
        for index, metric_history in enumerate(self.trust_policy_metric_histories):
            if index >= len(rewards) or index >= len(infos):
                continue
            history_len = len(self.memory[index]) if index < len(self.memory) else 0
            if not bool(dones[index]):
                if history_len < self.trust_policy_feedback_min_history_lines:
                    continue
                if (
                    self.trust_policy_feedback_update_interval > 1
                    and history_len % self.trust_policy_feedback_update_interval != 0
                ):
                    continue
            outcome_utility = _infer_outcome_utility_signal(rewards[index], dones[index], infos[index])
            self.trust_policy_skill_feedbacks[index] = _update_feedback_with_discounted_utility(
                self.trust_policy_skill_feedbacks[index],
                metric_history,
                outcome_utility=outcome_utility,
            )


def is_webshop_ocr_env(config) -> bool:
    env_name = str(getattr(config.env, "env_name", "") or "").strip().lower()
    return env_name in {"webshopocr", "webshopocridea", "webshop_ocr", "webshop_ocr_idea"}


def make_webshop_ocr_envs(config):
    from agent_system.environments.env_package.webshop import build_webshop_envs

    group_n = int(config.env.rollout.n)
    resources_per_worker = {"num_cpus": float(config.env.resources_per_worker.num_cpus)}
    default_data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "agent_system",
            "environments",
            "env_package",
            "webshop",
            "webshop",
            "data",
        )
    )
    base_dir = os.path.abspath(os.environ.get("WEBSHOP_DATA_ROOT", default_data_root))
    if config.env.webshop.use_small:
        file_path = os.path.abspath(os.path.join(base_dir, "items_shuffle_1000.json"))
        attr_path = os.path.abspath(os.path.join(base_dir, "items_ins_v2_1000.json"))
    else:
        file_path = os.path.abspath(os.path.join(base_dir, "items_shuffle.json"))
        attr_path = os.path.abspath(os.path.join(base_dir, "items_ins_v2.json"))

    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": config.env.webshop.human_goals,
        "file_path": file_path,
        "attr_path": attr_path,
    }
    train_envs = build_webshop_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    val_envs = build_webshop_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
    projection_f = partial(webshop_ocr_projection)
    envs = WebshopOCREnvironmentManager(train_envs, projection_f, config)
    val_envs = WebshopOCREnvironmentManager(val_envs, projection_f, config)
    time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)
    return envs, val_envs
