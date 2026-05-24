# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from math import ceil, exp
import os
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from agentocr.memory_egrc import skill_signal_targets as _component_skill_signal_targets
from agentocr.memory_ema import (
    best_schema_representative as _component_best_schema_representative,
    evolve_schema_witnesses as _component_evolve_schema_witnesses,
    refresh_schema_witnesses as _component_refresh_schema_witnesses,
    schema_component_tags as _component_schema_component_tags,
    schema_lifecycle_stage as _component_schema_lifecycle_stage,
    schema_primary_subject as _component_schema_primary_subject,
    schema_split_lines as _component_schema_split_lines,
    schema_template_signature as _component_schema_template_signature,
    schema_uncertainty as _component_schema_uncertainty,
    schema_verification_signal as _component_schema_verification_signal,
)
from agentocr.memory_fmr import (
    beta_uncertainty as _component_beta_uncertainty,
    update_uncertainty_state as _component_update_uncertainty_state,
)
from agentocr.memory_framework import (
    DEFAULT_MEMORY_FRAMEWORK_CONFIG,
    EGRC_COMPONENT,
    EMA_COMPONENT,
    FMR_COMPONENT,
    MemoryFrameworkConfig,
    resolve_memory_framework_config,
)


RenderAction = Literal["full_res", "low_res", "warn_low_res", "hide"]
TrustContextMode = Literal["query", "state", "slot", "auto"]
MEMORY_FRAMEWORK_COMPONENTS = (EMA_COMPONENT, EGRC_COMPONENT, FMR_COMPONENT)
SUSPICIOUS_MARKERS = (
    "ignore previous instructions",
    "system prompt",
    "do not follow",
    "jailbreak",
)
CONTRADICTION_MARKERS = ("contradict", "outdated", "obsolete", "deprecated", "low-trust")
SALIENT_MARKERS = ("[action]", "<search>", "<answer>", "[skill]", "tool", "final answer")
STATE_TASK_MARKERS: Dict[str, Tuple[str, ...]] = {
    "look_at_obj_in_light": ("light", "lamp", "desklamp", "switch", "bright", "dark"),
    "pick_heat_then_place_in_recep": ("heat", "hot", "microwave", "warm"),
    "pick_cool_then_place_in_recep": ("cool", "cold", "fridge", "freezer", "ice"),
    "pick_clean_then_place_in_recep": ("clean", "wash", "rinse", "faucet", "sink"),
}
STATE_LINE_MARKERS: Dict[str, Tuple[str, ...]] = {
    "look_at_obj_in_light": (
        "switch is on",
        "lamp is on",
        "you turn on the",
        "you switch on the",
        "illuminated",
        "bright enough",
        "visible",
    ),
    "pick_heat_then_place_in_recep": ("steaming", "heated", "hot", "warming", "you heat the"),
    "pick_cool_then_place_in_recep": ("chilled", "cooled", "cold", "icy", "you cool the"),
    "pick_clean_then_place_in_recep": ("rinsed", "clean", "spotless", "washed", "you clean the"),
}
DESIRED_STATE_BY_FAMILY: Dict[str, str] = {
    "look_at_obj_in_light": "light_on",
    "pick_heat_then_place_in_recep": "heated",
    "pick_cool_then_place_in_recep": "cooled",
    "pick_clean_then_place_in_recep": "cleaned",
}
STATE_DEVICE_HINTS_BY_FAMILY: Dict[str, Tuple[str, ...]] = {
    "look_at_obj_in_light": ("desklamp", "lamp", "switch"),
    "pick_heat_then_place_in_recep": ("microwave",),
    "pick_cool_then_place_in_recep": ("fridge", "freezer"),
    "pick_clean_then_place_in_recep": ("sink", "sinkbasin", "faucet"),
}
STATE_ACTION_VERB_BY_STATE: Dict[str, str] = {
    "light_on": "turn on",
    "heated": "heat",
    "cooled": "cool",
    "cleaned": "clean",
}
GENERIC_STATE_QUERY_MARKERS = (
    "then",
    "after",
    "before",
    "while",
    "until",
    "state",
    "status",
    "turn on",
    "turn off",
    "switch",
    "light",
    "dark",
    "heat",
    "cool",
    "clean",
    "wash",
    "rinse",
    "dry",
    "hold",
    "holding",
    "inventory",
    "selected",
    "applied",
    "filter",
    "logged in",
    "opened",
    "closed",
    "tab",
    "page",
)
GENERIC_STATE_LINE_MARKERS = (
    " is on",
    " is off",
    " illuminated",
    " visible",
    " too dark",
    " steaming",
    " heated",
    " hot",
    " chilled",
    " cooled",
    " cold",
    " rinsed",
    " washed",
    " clean",
    " dirty",
    " open",
    " closed",
    " unlocked",
    " locked",
    " selected",
    " applied",
    " logged in",
    " enabled",
    " disabled",
    " holding",
    " carrying",
)
GENERIC_TRANSITION_MARKERS = (
    "after waiting",
    "a moment later",
    "now",
    "already",
    "becomes",
    "became",
    "turns",
    "turned",
    "until it is",
)
DISTRACTOR_MARKERS = ("another", "second", "different", "other")
CANONICAL_STATE_VALUES = {
    "open",
    "closed",
    "heated",
    "cooled",
    "cleaned",
    "light_on",
    "light_off",
}
SLOT_QUERY_MARKERS = (
    " put ",
    " place ",
    " into ",
    " inside ",
    " onto ",
    " open ",
    " close ",
    " carrying ",
    " holding ",
)
QUERY_ACTION_VERBS = {
    "put",
    "place",
    "find",
    "take",
    "get",
    "bring",
    "move",
    "collect",
    "fetch",
    "store",
    "carry",
    "hold",
    "open",
    "close",
    "cool",
    "heat",
    "clean",
    "wash",
    "rinse",
    "look",
    "examine",
}
ENTITY_STOPWORDS = {
    "where",
    "what",
    "which",
    "who",
    "when",
    "you",
    "should",
    "could",
    "would",
    "can",
    "them",
    "it",
    "the",
    "a",
    "an",
    "to",
    "then",
    "after",
    "before",
    "into",
    "inside",
    "onto",
    "in",
    "on",
    "and",
    "with",
    "under",
    "from",
    "at",
    "of",
    "your",
    "task",
    "is",
    "are",
    "was",
    "were",
}
GOAL_STATE_WORDS = {
    "clean",
    "cleaned",
    "cool",
    "cooled",
    "cold",
    "hot",
    "heat",
    "heated",
    "warm",
    "warmed",
    "chilled",
}
GOAL_QUANTITY_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "some",
    "pair",
    "pairs",
    "double",
}
SEARCH_GENERIC_QUERY_PREFIXES = (
    "who ",
    "what ",
    "when ",
    "where ",
    "why ",
    "which ",
    "how ",
    "do ",
    "does ",
    "did ",
    "is ",
    "are ",
    "was ",
    "were ",
    "can ",
    "could ",
    "should ",
)
SEARCH_QUERY_ANCHOR_STOPWORDS = ENTITY_STOPWORDS | {
    "for",
    "how",
    "many",
    "much",
    "latest",
    "current",
    "currently",
    "main",
    "name",
    "named",
    "number",
    "total",
    "version",
    "versions",
    "season",
    "seasons",
    "episode",
    "episodes",
    "series",
    "singer",
    "sings",
    "sang",
    "song",
    "played",
    "play",
    "cross",
    "come",
    "out",
    "called",
    "line",
}
FACT_PATTERN = re.compile(
    r"\b(?:the\s+)?(?P<subject>[a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:is|are)\s+"
    r"(?:in|on|at)\s+(?:the\s+)?(?P<value>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
EMERGENT_LOCATION_PATTERN = re.compile(
    r"\b(?:the\s+)?(?P<subject>you|[a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:is|are)\s+"
    r"(?P<relation>under|behind|beside|near|next to|left of|right of|between)\s+"
    r"(?:the\s+)?(?P<value>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
EMERGENT_PROGRESS_PATTERN = re.compile(
    r"\b(?:you\s+)?(?P<verb>stashed|stored|delivered|inserted|equipped|selected|submitted|activated)\s+"
    r"(?:the\s+)?(?P<object>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
STATE_DESCRIPTOR_PATTERN = re.compile(
    r"\b(?:the\s+)?(?P<subject>[a-z0-9][a-z0-9 _-]{0,40}?)\s+(?:is|are)\s+"
    r"(?P<value>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
STATE_ACTION_FEEDBACK_PATTERN = re.compile(
    r"\byou\s+(?P<verb>heat|clean|cool)\s+(?:the\s+)?(?P<object>[a-z0-9][a-z0-9 _-]{0,40}?)"
    r"(?:\s+using\s+(?:the\s+)?(?P<tool>[a-z0-9][a-z0-9 _-]{0,40}?))?(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
LIGHT_TOGGLE_FEEDBACK_PATTERN = re.compile(
    r"\byou\s+(?P<verb>turn on|switch on)\s+(?:the\s+)?(?P<object>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
PICKUP_PATTERN = re.compile(
    r"\b(?:you\s+)?(?:pick up|take|grab|hold|carry|are carrying)\s*:?\s*(?:a\s+|an\s+|the\s+)?"
    r"(?P<object>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$|\s+(?:from|with|using|in|into|inside|on|onto)\b)",
    flags=re.IGNORECASE,
)
PLACE_PATTERN = re.compile(
    r"\b(?:you\s+)?put\s+(?:the\s+)?(?P<object>[a-z0-9][a-z0-9 _-]{0,40}?)\s+"
    r"(?:in|into|inside|on|onto)\s+(?:the\s+)?(?P<receptacle>[a-z0-9][a-z0-9 _-]{0,40}?)\b",
    flags=re.IGNORECASE,
)
RECEPTACLE_STATE_PATTERN = re.compile(
    r"\b(?:the\s+)?(?P<receptacle>[a-z0-9][a-z0-9 _-]{0,40}?)\s+is\s+(?P<state>open|closed|unlocked|locked)\b",
    flags=re.IGNORECASE,
)
ARRIVE_LOCATION_PATTERN = re.compile(
    r"\byou\s+(?:arrive\s+at|walk\s+to|go\s+to|are\s+in)\s+(?:the\s+)?"
    r"(?P<location>[a-z0-9][a-z0-9 _-]{0,40}?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
ANCHOR_OBSERVATION_PATTERN = re.compile(
    r"\b(?:on|in|inside)\s+(?:the\s+)?(?P<anchor>[a-z0-9][a-z0-9 _-]{0,40}?)\s*,\s*you\s+see\s+(?P<objects>.+?)(?=[\.,]|$)",
    flags=re.IGNORECASE,
)
SKILL_LINE_PATTERN = re.compile(
    r"^\[skill\]\[(?P<kind>goal|location|progress|state)\]\s*(?P<body>.*)$",
    flags=re.IGNORECASE,
)
SEARCH_TAG_PATTERN = re.compile(
    r"<search>(.*?)</search>",
    flags=re.IGNORECASE | re.DOTALL,
)
INFORMATION_TAG_PATTERN = re.compile(
    r"<information>(.*?)</information>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class GoalSlots:
    target_objects: Tuple[str, ...] = ()
    target_receptacle: Optional[str] = None
    target_object_count: int = 0


@dataclass
class RuntimeSlots:
    current_location: Optional[str] = None
    inventory_object: Optional[str] = None
    object_progress: Dict[str, str] = field(default_factory=dict)
    object_locations: Dict[str, str] = field(default_factory=dict)
    receptacle_open: Dict[str, bool] = field(default_factory=dict)
    receptacle_contents: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SchemaWitness:
    lines: Tuple[str, ...]
    role: str
    confidence: float = 1.0
    support: float = 0.0
    conflict: float = 0.0
    freshness: float = 1.0
    eviction_pressure: float = 0.0
    age: int = 0
    hit_count: int = 0
    miss_count: int = 0
    coverage: float = 0.0
    uncertainty: float = 1.0
    lifecycle_stage: str = "proto"
    relation_family: str = ""
    subject: str = ""
    current_value: str = ""
    pointer_lines: Tuple[str, ...] = ()
    support_witness_lines: Tuple[str, ...] = ()
    support_witness_roles: Tuple[str, ...] = ()


@dataclass
class MemorySkills:
    goal_targets: Tuple[str, ...] = ()
    goal_target_count: int = 0
    goal_receptacle: Optional[str] = None
    search_anchor: Optional[str] = None
    search_target_hint: Optional[str] = None
    search_bridge_hint: Optional[str] = None
    search_compare_hint: Optional[str] = None
    search_scope_hint: Optional[str] = None
    search_bridge_entity: Optional[str] = None
    search_answer_candidate: Optional[str] = None
    current_location: Optional[str] = None
    visible_anchor: Optional[str] = None
    target_locations: Dict[str, str] = field(default_factory=dict)
    object_progress: Dict[str, str] = field(default_factory=dict)
    state_flags: Dict[str, str] = field(default_factory=dict)
    searched_without_target: Tuple[str, ...] = ()
    confidence_by_skill: Dict[str, float] = field(default_factory=dict)
    witness_lines_by_skill: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    witness_roles_by_skill: Dict[str, Dict[str, str]] = field(default_factory=dict)
    schema_witnesses_by_skill: Dict[str, Tuple[SchemaWitness, ...]] = field(default_factory=dict)


@dataclass
class MemorySkillFeedback:
    failure_counts: Dict[str, int] = field(default_factory=dict)
    fallback_bias_by_skill: Dict[str, float] = field(default_factory=dict)
    recent_failure_ema_by_skill: Dict[str, float] = field(default_factory=dict)
    reliability_by_skill: Dict[str, float] = field(default_factory=dict)
    support_by_skill: Dict[str, float] = field(default_factory=dict)
    conflict_by_skill: Dict[str, float] = field(default_factory=dict)
    rescue_budget_by_skill: Dict[str, float] = field(default_factory=dict)
    utility_ema_by_skill: Dict[str, float] = field(default_factory=dict)
    evidence_alpha_by_skill: Dict[str, float] = field(default_factory=dict)
    evidence_beta_by_skill: Dict[str, float] = field(default_factory=dict)
    uncertainty_by_skill: Dict[str, float] = field(default_factory=dict)
    witness_utility_by_line: Dict[str, float] = field(default_factory=dict)
    witness_role_support_by_skill: Dict[str, Dict[str, float]] = field(default_factory=dict)
    witness_role_conflict_by_skill: Dict[str, Dict[str, float]] = field(default_factory=dict)
    witness_role_utility_by_skill: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedTrustLine:
    current_location: Optional[str] = None
    visible_anchor: Optional[str] = None
    emergent_location: Optional[Tuple[str, str, str, str]] = None
    emergent_progress: Optional[Tuple[str, str, str]] = None
    line_slot_kinds: Tuple[str, ...] = ()
    pickup_object: Optional[str] = None
    placement: Optional[Tuple[str, str]] = None
    has_fact_signature: bool = False
    mentioned_goal_objects: Tuple[str, ...] = ()
    receptacle_state: Optional[Tuple[str, bool]] = None
    explicit_state: Optional[Tuple[str, str]] = None
    known_state: Optional[Tuple[str, str, str]] = None
    emergent_state: Optional[Tuple[str, str, str]] = None


@dataclass(frozen=True)
class EvidenceUnit:
    family: str
    skill: str
    subject: str
    value: str
    role: str
    line: str


@dataclass(frozen=True)
class PreparedTrustContext:
    query_text: str
    raw_lines: Tuple[str, ...]
    resolved_mode: TrustContextMode
    feedback_family: str
    feedback_phase: str
    goal_slots: GoalSlots
    skills: MemorySkills


@dataclass(frozen=True)
class SegmentTrustMetadata:
    """Metadata used to decide how a history segment should be rendered."""

    text: str
    step: int
    source_id: str = "env"
    source_trust: float = 1.0
    support_count: int = 1
    contradiction_count: int = 0
    suspicious_score: float = 0.0
    salience: float = 0.5
    query_relevance: float = 0.5


@dataclass(frozen=True)
class TrustPolicyConfig:
    """Configuration for trust-calibrated optical memory allocation."""

    recency_half_life: float = 8.0
    hide_threshold: float = 0.20
    warn_threshold: float = 0.45
    full_res_threshold: float = 0.78
    low_res_compression: float = 2.0
    warn_compression: float = 3.0
    hidden_placeholder: str = "[hidden low-trust memory segment]"
    warning_prefix: str = "[low-trust]"
    query_relevance_weight: float = 0.0
    context_budget_percent: float = 100.0


@dataclass(frozen=True)
class RenderDecision:
    """Rendering decision for one segment."""

    text: str
    score: float
    action: RenderAction
    compression_factor: float
    reason: str


class TrustCalibratedRenderPolicy:
    """Score memory segments and map them to optical rendering decisions.

    This is a deliberately small research prototype for the "Trust-Calibrated
    Hierarchical Optical Memory" idea. It is side-effect free and can be wired
    into `OCRTool` later without changing the current training path.
    """

    def __init__(self, config: Optional[TrustPolicyConfig] = None):
        self.config = config or TrustPolicyConfig()

    def score_segment(self, metadata: SegmentTrustMetadata, current_step: int) -> float:
        age = max(0, current_step - metadata.step)
        recency = 0.5 ** (age / max(self.config.recency_half_life, 1e-6))
        support = 1.0 - exp(-max(0, metadata.support_count))
        contradiction_penalty = min(1.0, 0.35 * max(0, metadata.contradiction_count))

        score = (
            0.30 * self._clamp(metadata.source_trust)
            + 0.20 * recency
            + 0.20 * support
            + 0.20 * self._clamp(metadata.salience)
            + 0.10 * (1.0 - self._clamp(metadata.suspicious_score))
            + self.config.query_relevance_weight * (2.0 * self._clamp(metadata.query_relevance) - 1.0)
            - contradiction_penalty
        )
        return self._clamp(score)

    def decide(self, metadata: SegmentTrustMetadata, current_step: int) -> RenderDecision:
        score = self.score_segment(metadata, current_step)

        if score < self.config.hide_threshold:
            return RenderDecision(
                text=self.config.hidden_placeholder,
                score=score,
                action="hide",
                compression_factor=self.config.warn_compression,
                reason="trust score below hide threshold",
            )

        if score < self.config.warn_threshold:
            return RenderDecision(
                text=f"{self.config.warning_prefix} {metadata.text}",
                score=score,
                action="warn_low_res",
                compression_factor=self.config.warn_compression,
                reason="low trust; render with warning and stronger compression",
            )

        if score >= self.config.full_res_threshold:
            return RenderDecision(
                text=metadata.text,
                score=score,
                action="full_res",
                compression_factor=1.0,
                reason="trusted and salient enough for full-resolution memory",
            )

        return RenderDecision(
            text=metadata.text,
            score=score,
            action="low_res",
            compression_factor=self.config.low_res_compression,
            reason="usable but not important enough for full resolution",
        )

    def decide_batch(
        self,
        segments: Iterable[SegmentTrustMetadata],
        current_step: int,
    ) -> List[RenderDecision]:
        return [self.decide(segment, current_step) for segment in segments]

    def apply_context_budget(self, decisions: Sequence[RenderDecision]) -> List[RenderDecision]:
        budget_percent = float(self.config.context_budget_percent)
        if budget_percent >= 100.0:
            return list(decisions)
        budget_percent = max(1.0, budget_percent)

        visible = [(index, decision) for index, decision in enumerate(decisions) if decision.action != "hide"]
        if not visible:
            return list(decisions)

        keep_count = max(1, ceil(len(visible) * budget_percent / 100.0))
        ranked_visible = sorted(visible, key=lambda item: (item[1].score, -item[0]), reverse=True)
        keep_indices = {index for index, _ in ranked_visible[:keep_count]}

        budgeted_decisions: List[RenderDecision] = []
        for index, decision in enumerate(decisions):
            if decision.action == "hide" or index in keep_indices:
                budgeted_decisions.append(decision)
                continue
            budgeted_decisions.append(
                replace(
                    decision,
                    text=self.config.hidden_placeholder,
                    action="hide",
                    reason=f"{decision.reason}; dropped by context budget",
                )
            )
        return budgeted_decisions

    def renderable_text(self, decisions: Iterable[RenderDecision]) -> str:
        return "\n".join(decision.text for decision in decisions if decision.action != "hide")

    def compression_factors(self, decisions: Iterable[RenderDecision]) -> List[float]:
        return [decision.compression_factor for decision in decisions]

    @staticmethod
    def _clamp(value: float, bounds: Tuple[float, float] = (0.0, 1.0)) -> float:
        lo, hi = bounds
        return min(hi, max(lo, float(value)))


def _strip_xml_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _strip_embedded_task_text(text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"(?i)\byour task is to:\s*.*$", "", stripped).strip()
        stripped = re.sub(r"(?i)\btask:\s*.*$", "", stripped).strip()
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def _split_memory_line_channels(line_text: str) -> Tuple[str, str]:
    raw_text = _strip_xml_tags(str(line_text or "")).strip()
    if not raw_text:
        return "", ""
    observation_text = raw_text
    action_text = ""
    parts = re.split(r"\[\s*action\s*\]\s*:\s*", raw_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        observation_text, action_text = parts[0], parts[1]
    observation_text = re.sub(r"^\s*\[\s*observation\s*\]\s*:\s*", "", observation_text, flags=re.IGNORECASE).strip()
    action_text = re.sub(r"^\s*\[\s*action\s*\]\s*:\s*", "", action_text, flags=re.IGNORECASE).strip()
    return observation_text, action_text


def _observation_evidence_text(line_text: str) -> str:
    observation_text, _ = _split_memory_line_channels(line_text)
    cleaned = _strip_embedded_task_text(observation_text)
    if cleaned:
        return cleaned
    return _strip_embedded_task_text(_strip_xml_tags(str(line_text or "")))


def _normalize_fact_token(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def _entity_tokens(text: str) -> Set[str]:
    return {
        token
        for token in _normalize_fact_token(text).split()
        if token and token not in ENTITY_STOPWORDS and token not in QUERY_ACTION_VERBS
    }


def _entity_head(text: str) -> str:
    tokens = [token for token in _normalize_fact_token(text).split() if token not in ENTITY_STOPWORDS]
    for token in reversed(tokens):
        if any(char.isalpha() for char in token):
            return token
    return tokens[-1] if tokens else ""


def _entities_match(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return False
    left_norm = _normalize_fact_token(left)
    right_norm = _normalize_fact_token(right)
    if left_norm == right_norm:
        return True
    left_tokens = _entity_tokens(left_norm)
    right_tokens = _entity_tokens(right_norm)
    if not left_tokens or not right_tokens:
        return False
    return left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)


def _line_mentions_entity(line_text: str, entity: str) -> bool:
    line_norm = _normalize_fact_token(line_text)
    entity_norm = _normalize_fact_token(entity)
    if entity_norm and entity_norm in line_norm:
        return True
    entity_surface = re.sub(r"\b\d+\b$", "", entity_norm).strip()
    if entity_surface and entity_surface in line_norm:
        return True
    line_tokens = _entity_tokens(line_text)
    entity_tokens = _entity_tokens(entity)
    if not entity_tokens:
        return False
    head = _entity_head(entity)
    return entity_tokens.issubset(line_tokens) or bool(head and head in line_tokens)


def _goal_binding_is_distractor(
    line_text: str,
    entity: Optional[str],
    goal_slots: GoalSlots,
) -> bool:
    if entity is None or not _same_class_distractor(line_text, goal_slots):
        return False
    return _goal_object_for_entity(entity, goal_slots) is not None


def _clean_slot_entity(text: str) -> str:
    normalized = _normalize_fact_token(text)
    tokens = [
        token
        for token in normalized.split()
        if token not in ENTITY_STOPWORDS and token not in QUERY_ACTION_VERBS
    ]
    return " ".join(tokens).strip()


def _extract_explicit_goal_quantity(text: str) -> int:
    normalized = _normalize_fact_token(text)
    if not normalized:
        return 0
    tokens = normalized.split()
    if not tokens:
        return 0
    quantity_map = {
        "1": 1,
        "one": 1,
        "2": 2,
        "two": 2,
        "3": 3,
        "three": 3,
        "4": 4,
        "four": 4,
        "5": 5,
        "five": 5,
    }
    return int(quantity_map.get(tokens[0], 0))


def _singularize_goal_token(token: str) -> str:
    token = str(token or "").strip()
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("oes", "sses", "shes", "ches", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _canonicalize_goal_entity(text: str, *, singularize_plural: bool = False) -> str:
    normalized = _normalize_fact_token(text)
    tokens = []
    for token in normalized.split():
        if (
            token in ENTITY_STOPWORDS
            or token in QUERY_ACTION_VERBS
            or token in GOAL_STATE_WORDS
            or token in GOAL_QUANTITY_WORDS
        ):
            continue
        if token.isdigit():
            continue
        tokens.append(_singularize_goal_token(token) if singularize_plural else token)
    return " ".join(tokens).strip()


def _goal_target_count(goal_slots: GoalSlots) -> int:
    return max(int(goal_slots.target_object_count or 0), len(goal_slots.target_objects))


def _goal_uses_instance_tracking(goal_slots: GoalSlots) -> bool:
    return _goal_target_count(goal_slots) > len(goal_slots.target_objects)


def _goal_progress_subject_for_entity(entity: Optional[str], goal_slots: GoalSlots) -> Optional[str]:
    cleaned_entity = _clean_slot_entity(entity or "")
    if not cleaned_entity:
        return None
    goal_object = _goal_object_for_entity(cleaned_entity, goal_slots)
    if goal_object is None:
        return None
    if _goal_uses_instance_tracking(goal_slots):
        return cleaned_entity
    return goal_object


def _goal_progress_subject_keys(
    progress_map: Dict[str, str],
    goal_slots: GoalSlots,
) -> Tuple[str, ...]:
    goal_keys = [goal_object for goal_object in goal_slots.target_objects if goal_object]
    if not progress_map:
        return tuple(goal_keys)

    matched_instance_keys: List[str] = []
    matched_canonical_keys: List[str] = []
    canonical_goal_keys = {_clean_slot_entity(goal_object) for goal_object in goal_keys if _clean_slot_entity(goal_object)}
    for subject in progress_map.keys():
        cleaned_subject = _clean_slot_entity(subject)
        if not cleaned_subject or _goal_object_for_entity(cleaned_subject, goal_slots) is None:
            continue
        if cleaned_subject in canonical_goal_keys:
            matched_canonical_keys.append(cleaned_subject)
        else:
            matched_instance_keys.append(cleaned_subject)

    ordered: List[str] = []
    seen: Set[str] = set()
    source = matched_instance_keys if matched_instance_keys else matched_canonical_keys
    for subject in source:
        if subject in seen:
            continue
        seen.add(subject)
        ordered.append(subject)
    if ordered:
        return tuple(ordered)
    return tuple(goal_keys)


def _goal_progress_state_counts(
    progress_map: Dict[str, str],
    goal_slots: GoalSlots,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for subject in _goal_progress_subject_keys(progress_map, goal_slots):
        state = str(progress_map.get(subject, "missing") or "missing").strip() or "missing"
        counts[state] = counts.get(state, 0) + 1
    return counts


def _goal_progress_state_count(
    progress_map: Dict[str, str],
    goal_slots: GoalSlots,
    *states: str,
) -> int:
    counts = _goal_progress_state_counts(progress_map, goal_slots)
    return sum(int(counts.get(state, 0)) for state in states)


def _goal_progress_state_for_entity(
    entity: Optional[str],
    progress_map: Dict[str, str],
    goal_slots: GoalSlots,
) -> Optional[str]:
    subject = _goal_progress_subject_for_entity(entity, goal_slots)
    if subject and subject in progress_map:
        return progress_map.get(subject)
    goal_object = _goal_object_for_entity(entity, goal_slots)
    if goal_object and goal_object in progress_map:
        return progress_map.get(goal_object)
    return None


def _required_state_for_family(family: Optional[str]) -> str:
    return str(DESIRED_STATE_BY_FAMILY.get(str(family or "").strip(), "") or "").strip()


def _state_device_hint_for_family(family: Optional[str], query_text: Optional[str] = None) -> str:
    family_key = str(family or "").strip()
    normalized_query = _normalize_fact_token(query_text or "")
    if normalized_query and family_key == "look_at_obj_in_light":
        device_match = re.search(
            r"\b(?:under|below|beneath|by|near|next to|beside|with|using)\s+"
            r"(?P<device>[a-z0-9][a-z0-9 _-]{0,40})\b",
            normalized_query,
        )
        if device_match:
            device = _clean_slot_entity(device_match.group("device"))
            if device and any(hint in device for hint in STATE_DEVICE_HINTS_BY_FAMILY.get(family_key, ())):
                return device

    hints = STATE_DEVICE_HINTS_BY_FAMILY.get(family_key, ())
    return str(hints[0] if hints else "").strip()


def _normalize_query_tokens(text: str) -> List[str]:
    stopwords = {
        "where",
        "what",
        "which",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "of",
        "to",
        "for",
        "does",
        "do",
        "did",
        "should",
        "be",
        "find",
        "located",
    }
    tokens = _normalize_fact_token(text).split()
    return [token for token in tokens if token and token not in stopwords]


def _is_simple_search_location_question(query_text: str) -> bool:
    tokens = _normalize_fact_token(query_text).split()
    if len(tokens) < 3 or len(tokens) > 6:
        return False
    if tokens[0] != "where" or tokens[1] not in {"is", "are"}:
        return False
    return "do" not in tokens[:3]


def _is_open_domain_search_query(query_text: str) -> bool:
    normalized = _normalize_fact_token(query_text)
    if not normalized or _infer_state_task_family(query_text) is not None:
        return False
    if _is_simple_search_location_question(query_text):
        return False
    if any(
        normalized.startswith(prefix.strip())
        for prefix in ("put", "place", "find", "take", "bring", "move", "collect", "fetch", "store", "carry", "hold")
    ):
        return False
    if not (
        str(query_text or "").strip().endswith("?")
        or any(normalized.startswith(prefix.strip()) for prefix in SEARCH_GENERIC_QUERY_PREFIXES)
    ):
        return False
    return len(normalized.split()) > 5


def _extract_search_query(line_text: str) -> Optional[str]:
    matches = SEARCH_TAG_PATTERN.findall(str(line_text or ""))
    if not matches:
        return None
    query = " ".join(str(matches[-1]).split()).strip()
    return query or None


def _extract_information_payload(line_text: str) -> str:
    matches = INFORMATION_TAG_PATTERN.findall(str(line_text or ""))
    if matches:
        return " ".join(str(matches[-1]).split()).strip()
    return ""


def _coalesce_open_domain_search_information_lines(lines: Sequence[str]) -> Tuple[str, ...]:
    coalesced: List[str] = []
    info_parts: List[str] = []
    inside_information = False

    for raw_line in lines:
        text = str(raw_line or "").strip()
        if not text:
            continue

        if not inside_information:
            start_match = re.search(r"<information>", text, flags=re.IGNORECASE)
            end_match = re.search(r"</information>", text, flags=re.IGNORECASE)
            if start_match and end_match and start_match.start() < end_match.start():
                coalesced.append(text)
                continue
            if not start_match:
                coalesced.append(text)
                continue
            prefix = text[:start_match.start()].strip()
            if prefix:
                coalesced.append(prefix)
            remainder = text[start_match.end():].strip()
            info_parts = [remainder] if remainder else []
            inside_information = True
            continue

        end_match = re.search(r"</information>", text, flags=re.IGNORECASE)
        if end_match:
            before_end = text[:end_match.start()].strip()
            if before_end:
                info_parts.append(before_end)
            payload = " ".join(part for part in info_parts if part).strip()
            coalesced.append(
                f"<information>{payload}</information>" if payload else "<information></information>"
            )
            suffix = text[end_match.end():].strip()
            if suffix:
                coalesced.append(suffix)
            info_parts = []
            inside_information = False
            continue

        info_parts.append(text)

    if inside_information:
        payload = " ".join(part for part in info_parts if part).strip()
        coalesced.append(
            f"<information>{payload}</information>" if payload else "<information></information>"
        )

    return tuple(coalesced)


def _normalize_search_query_anchor(query_text: str) -> str:
    tokens = [
        token
        for token in _normalize_fact_token(query_text).split()
        if token and token not in SEARCH_QUERY_ANCHOR_STOPWORDS
    ]
    if not tokens:
        tokens = [
            token
            for token in _normalize_fact_token(query_text).split()
            if token and token not in ENTITY_STOPWORDS
        ]
    return " ".join(tokens[:10]).strip()


def _search_task_anchor_hint(query_text: str) -> str:
    question = str(query_text or "").strip()
    if not question:
        return ""

    compare_pair = _search_task_compare_hint(question)
    if compare_pair:
        return " ".join(compare_pair.replace("|", " ").split())

    same_relation, same_source, _ = _search_task_same_relation_components(question)
    if same_relation and same_source:
        source_anchor = _normalize_search_query_anchor(same_source)
        if source_anchor:
            return source_anchor

    direct_anchor, _ = _search_task_direct_components(question)
    if direct_anchor:
        return direct_anchor

    candidate_patterns = (
        r"([A-Z][A-Za-z0-9'&-]*:\s*[A-Z][A-Za-z0-9'&:-]*(?:\s+[A-Z][A-Za-z0-9'&:-]*){0,4})",
        r"^([A-Z][A-Za-z0-9'&:-]*(?:\s+[A-Z][A-Za-z0-9'&:-]*){0,5})\s+"
        r"(?:starred|features|featured|is|was|aired|broadcast|won|had|has|includes|include)\b",
        r"\bas\s+([A-Z][A-Za-z0-9'&:-]*(?:\s+[A-Z][A-Za-z0-9'&:-]*){0,4})\b",
        r"[\"“]([^\"”]{2,80})[\"”]",
    )
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, question):
            candidate = _normalize_fact_token(match.group(1))
            if not candidate:
                continue
            filtered_tokens = [
                token for token in candidate.split()
                if token and token not in ENTITY_STOPWORDS
            ]
            if filtered_tokens:
                return " ".join(filtered_tokens[:8]).strip()
    role_anchor_match = re.search(
        r"\b(chief justice|president|prime minister|governor|mayor|ceo|chief executive)\s+of\s+(.+)$",
        _normalize_fact_token(question),
    )
    if role_anchor_match:
        role = role_anchor_match.group(1).strip()
        entity = role_anchor_match.group(2).strip(" ?.")
        entity_tokens = [
            token for token in entity.split()
            if token and token not in ENTITY_STOPWORDS
        ]
        return " ".join(([role] + entity_tokens)[:8]).strip()
    fallback_tokens = [
        token
        for token in _normalize_fact_token(question).split()
        if token
        and token not in SEARCH_QUERY_ANCHOR_STOPWORDS
        and token not in QUERY_ACTION_VERBS
        and token not in {
            "make",
            "makes",
            "made",
            "up",
            "member",
            "members",
            "composition",
            "consists",
            "consist",
        }
    ]
    if fallback_tokens:
        return " ".join(fallback_tokens[:8]).strip()
    return _normalize_search_query_anchor(question)


def _clean_same_relation_clause(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", _normalize_fact_token(text)).strip(" ?.")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned).strip()
    return cleaned


def _clean_same_relation_target_clause(text: str, relation: str) -> str:
    cleaned = re.sub(r"\s+", " ", _normalize_fact_token(text)).strip(" ?.")
    cleaned = re.sub(r"^(?:what|which|who)\s+", "", cleaned).strip()

    relation_cleanup_patterns = {
        "same network": (
            r"\b(?:is|was|are|were)\s+(?:broadcast|aired|shown|televised)\s+on\s+the\s*$",
            r"\b(?:broadcast|aired|shown|televised)\s+on\s+the\s*$",
        ),
        "same channel": (
            r"\b(?:is|was|are|were)\s+(?:broadcast|aired|shown|televised)\s+on\s+the\s*$",
            r"\b(?:broadcast|aired|shown|televised)\s+on\s+the\s*$",
        ),
        "same country": (
            r"\b(?:is|was|are|were)\s+from\s+the\s*$",
            r"\bfrom\s+the\s*$",
        ),
        "same team": (
            r"\b(?:plays?|played|playing)\s+(?:for|on)\s+the\s*$",
            r"\b(?:is|was|are|were)\s+on\s+the\s*$",
        ),
        "same label": (
            r"\b(?:is|was|are|were)\s+(?:signed|released)\s+(?:to|on)\s+the\s*$",
            r"\b(?:signed|released)\s+(?:to|on)\s+the\s*$",
        ),
    }
    for pattern in relation_cleanup_patterns.get(relation, ()):
        cleaned = re.sub(pattern, "", cleaned).strip()
    cleaned = re.sub(r"\b(?:is|was|are|were)\s*$", "", cleaned).strip()
    return cleaned.strip(" ?.")


def _search_task_same_relation_components(query_text: str) -> Tuple[str, str, str]:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return "", "", ""

    relation_patterns: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        (
            "same network",
            (
                "broadcast on the same network as",
                "aired on the same network as",
                "shown on the same network as",
                "televised on the same network as",
                "same network as",
            ),
        ),
        (
            "same channel",
            (
                "broadcast on the same channel as",
                "aired on the same channel as",
                "shown on the same channel as",
                "televised on the same channel as",
                "same channel as",
            ),
        ),
        (
            "same country",
            (
                "from the same country as",
                "in the same country as",
                "same country as",
            ),
        ),
        (
            "same team",
            (
                "for the same team as",
                "on the same team as",
                "same team as",
            ),
        ),
        (
            "same label",
            (
                "signed to the same label as",
                "released on the same label as",
                "on the same label as",
                "same label as",
            ),
        ),
    )
    for relation, patterns in relation_patterns:
        for pattern in patterns:
            if pattern not in normalized:
                continue
            left, right = normalized.split(pattern, 1)
            source_clause = _clean_same_relation_clause(right)
            target_clause = _clean_same_relation_target_clause(left, relation)
            if source_clause or target_clause:
                return relation, source_clause, target_clause
    return "", "", ""


def _canonicalize_query_fact_value(text: str) -> str:
    cleaned_text = re.sub(r"\bdoc\s+\d+\s*:\s*", " ", str(text or ""), flags=re.IGNORECASE)
    normalized = _normalize_fact_token(cleaned_text)
    if not normalized:
        return ""
    tokens = [
        token for token in normalized.split()
        if token and token not in ENTITY_STOPWORDS
    ]
    return " ".join(tokens[:16]).strip()


def _extract_fact_signature(text: str) -> Optional[Tuple[str, str]]:
    information_matches = re.findall(r"<information>(.*?)</information>", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = information_matches[-1] if information_matches else _observation_evidence_text(text)
    match = FACT_PATTERN.search(cleaned)
    if not match:
        return None
    subject = _normalize_fact_token(match.group("subject"))
    value = _normalize_fact_token(match.group("value"))
    if not subject or not value:
        return None
    return subject, value


def _extract_search_query_fact_signature(
    line_text: str,
    *,
    active_search_anchor: str,
) -> Optional[Tuple[str, str]]:
    if not active_search_anchor:
        return None
    payload = _extract_information_payload(line_text)
    if not payload:
        return None
    value = _canonicalize_query_fact_value(payload)
    if not value:
        return None
    return active_search_anchor, value


def _search_task_target_hint(query_text: str) -> str:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return ""

    same_relation, _, same_target = _search_task_same_relation_components(query_text)
    if same_relation and same_target:
        return same_target

    if re.match(r"^(?:what|which)\s+(?:is|are)\s+.+\s+called$", normalized):
        return "term"

    _, direct_target = _search_task_direct_components(query_text)
    if direct_target:
        return direct_target

    hint_rules = (
        (("who was born earlier", "who is older"), "birth date comparison"),
        (("more valuable",), "current value comparison"),
        (("who makes up", "made up of", "consists of", "members of", "member of", "composition of"), "members composition"),
        (("same network", "same channel"), "same network"),
        (("same country",), "same country"),
        (("same team",), "same team"),
        (("same label",), "same label"),
        (("how old", "current age", "age of"), "current age"),
        (("ethnicity",), "ethnicity"),
        (("religion",), "religion"),
        (("nationality",), "nationality"),
        (("language",), "language"),
        (("occupation",), "occupation"),
        (("population",), "population"),
        (("distance",), "distance"),
        (("mascot",), "mascot"),
        (("award", "awards"), "award"),
        (("release date", "released", "premiered"), "release date"),
    )
    for triggers, hint in hint_rules:
        if any(trigger in normalized for trigger in triggers):
            return hint

    if (
        normalized.startswith("who is the ")
        and any(
            phrase in normalized
            for phrase in (
                "chief justice of",
                "president of",
                "prime minister of",
                "governor of",
                "mayor of",
                "ceo of",
                "chief executive of",
            )
        )
    ):
        return "current role holder"

    role_year_media_match = re.search(
        r"\broles?\s+in\s+what\s+((?:19|20)\d{2}\s+(?:film|movie|song|album|book|novel|series|television series|tv series|episode|season|show))\b",
        normalized,
    )
    if role_year_media_match:
        return f"roles in {role_year_media_match.group(1).strip()}"

    year_media_match = re.search(
        r"\b((?:19|20)\d{2}\s+(?:film|movie|song|album|book|novel|series|television series|tv series|episode|season|show))\b",
        normalized,
    )
    if year_media_match:
        return year_media_match.group(1)

    typed_target_match = re.search(
        r"\b(?:what|which)\s+((?:[a-z0-9-]+\s+){0,3}"
        r"(?:film|movie|song|album|book|novel|series|television series|tv series|episode|season|show|year|date|age|country|team|label|network))\b",
        normalized,
    )
    if typed_target_match:
        return typed_target_match.group(1).strip()

    filtered_tokens = [
        token
        for token in normalized.split()
        if token
        and token not in ENTITY_STOPWORDS
        and token not in QUERY_ACTION_VERBS
        and token not in {
            "who",
            "what",
            "when",
            "where",
            "which",
            "whose",
            "whom",
            "many",
            "much",
            "name",
        }
    ]
    return " ".join(filtered_tokens[:8]).strip()


def _search_task_compare_hint(query_text: str) -> str:
    question = str(query_text or "").strip()
    if not question:
        return ""

    patterns = (
        r"who\s+was\s+born\s+earlier,\s*(?P<a>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})\s+or\s+(?P<b>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})",
        r"who\s+is\s+older,\s*(?P<a>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})\s+or\s+(?P<b>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})",
        r"which\s+is\s+(?:currently\s+)?more\s+valuable,\s*(?P<a>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5})\s+or\s+(?P<b>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5})",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        left = _normalize_fact_token(match.group("a"))
        right = _normalize_fact_token(match.group("b"))
        if left and right:
            return f"{left} | {right}"
    return ""


def _search_task_direct_components(query_text: str) -> Tuple[str, str]:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return "", ""

    winner_match = re.match(r"^who won (?:the )?(?P<event>.+)$", normalized)
    if winner_match:
        event = winner_match.group("event").strip(" ?.")
        if event:
            return event, "winner"

    currency_prefixes = (
        "what is the money used in ",
        "what money is used in ",
        "what is the currency used in ",
        "what currency is used in ",
        "money used in ",
        "currency used in ",
    )
    for prefix in currency_prefixes:
        if normalized.startswith(prefix):
            place = normalized[len(prefix):].strip(" ?.")
            if place:
                return place, "currency"

    target_of_match = re.match(
        r"^(?:which|what|who)\s+(?P<role>[a-z0-9'&()./-]+(?:\s+[a-z0-9'&()./-]+){0,6})\s+"
        r"was\s+the\s+target\s+of\s+(?P<event>.+)$",
        normalized,
    )
    if target_of_match:
        role = target_of_match.group("role").strip(" ?.")
        event = re.sub(r"^(?:an|a|the)\s+", "", target_of_match.group("event")).strip(" ?.")
        if role and event:
            return event, role

    return "", ""


def _search_task_bridge_hint(query_text: str) -> str:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return ""

    same_relation, same_source, _ = _search_task_same_relation_components(query_text)
    if same_relation and same_source:
        return same_source

    bridge_patterns = (
        r"\b((?:english|american|indian|british|french|german|canadian|australian|male|female)\s+"
        r"(?:actor|actress|singer|author|writer|director|player|politician|governor|president))\b",
        r"\b((?:[a-z0-9-]+\s+){0,2}(?:drama|comedy|romance|crime)\s+(?:television series|tv series))\b",
    )
    for pattern in bridge_patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1).strip()
    return ""


def _search_task_scope_hint(query_text: str) -> str:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return ""

    same_relation, _, _ = _search_task_same_relation_components(query_text)
    target_hint = _search_task_target_hint(query_text)
    target_media_tokens = {
        token
        for token in str(target_hint or "").split()
        if token in {"film", "movie", "series", "episode", "season", "show"}
    }
    scope_phrases: List[str] = []
    if same_relation:
        scope_phrases.append(same_relation)
    scope_phrases.extend((
        "in russia",
        "in india",
        "in china",
        "in the united states",
        "in the uk",
        "in england",
        "in france",
        "in germany",
        "in japan",
        "in canada",
        "in australia",
        "national level",
        "state council",
        "television series",
        "tv series",
        "film",
        "movie",
        "episode",
        "season",
    ))
    matched = [
        phrase
        for phrase in scope_phrases
        if phrase in normalized
        and not (
            phrase in {"film", "movie", "episode", "season"}
            and phrase in target_media_tokens
        )
    ]
    deduped: List[str] = []
    for phrase in matched:
        if phrase not in deduped:
            deduped.append(phrase)
    return " | ".join(deduped[:3]).strip()


def _extract_search_bridge_entity_candidate(query_text: str, info_payload: str) -> str:
    bridge_hint = _search_task_bridge_hint(query_text)
    if not bridge_hint:
        return ""

    bridge_pattern = re.escape(bridge_hint).replace(r"\ ", r"\s+")
    match = re.search(
        rf"\b([A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){{0,3}})\s+"
        rf"(?:is|was)\s+an?\s+{bridge_pattern}\b",
        str(info_payload or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _normalize_fact_token(match.group(1))


def _search_info_cleaned_text(info_payload: str) -> str:
    cleaned = re.sub(r"\bdoc\s+\d+\s*:\s*", " ", str(info_payload or ""), flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _search_info_leading_title(info_payload: str) -> str:
    cleaned = _search_info_cleaned_text(info_payload)
    match = re.match(r'^"(?P<title>[^"]{2,120})"', cleaned)
    return str(match.group("title") or "").strip() if match else ""


def _search_current_role_phrase(query_text: str) -> str:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return ""
    match = re.match(
        r"^(?:who|which|what)\s+is\s+(?:the\s+)?(?P<role>.+?)(?:\s+of\b.+)?$",
        normalized,
    )
    return str(match.group("role") or "").strip() if match else ""


def _extract_search_current_role_holder_candidate(query_text: str, info_payload: str) -> str:
    role_phrase = _search_current_role_phrase(query_text)
    cleaned = _search_info_cleaned_text(info_payload)
    if not role_phrase or not cleaned:
        return ""

    role_pattern = re.escape(role_phrase).replace(r"\ ", r"\s+")
    person_pattern = r"[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}"
    patterns = (
        rf"\bthe\s+current\s+{role_pattern}\b(?:\s+of\b[^.;]{{0,80}})?\s+(?:is|was)\s+(?P<answer>{person_pattern})\b",
        rf"\b(?P<answer>{person_pattern})\b[^.;]{{0,140}}\bcurrent\s+{role_pattern}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _normalize_fact_token(match.group("answer"))
    return ""


def _extract_search_currency_candidate(info_payload: str) -> str:
    cleaned = _search_info_cleaned_text(info_payload)
    if not cleaned:
        return ""

    title = _search_info_leading_title(cleaned)
    if title and re.search(r"\bofficial currency of\b", cleaned, flags=re.IGNORECASE):
        return _normalize_fact_token(title)

    match = re.search(
        r"\b(?:the\s+)?(?P<answer>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3}\s+[a-z][a-z0-9'&.-]*)"
        r"\s*(?:\([^)]*\))?\s+is\s+the\s+official\s+currency\s+of\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return _normalize_fact_token(match.group("answer"))
    return ""


def _extract_search_winner_candidate(info_payload: str) -> str:
    cleaned = _search_info_cleaned_text(info_payload)
    if not cleaned:
        return ""

    person_pattern = r"[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3}"
    patterns = (
        rf"\b(?:With|with)\s+(?:his|her|their)\s+victory\b[^.;]{{0,120}},\s*(?P<answer>{person_pattern})\b",
        rf"\b(?P<answer>{person_pattern})\b\s+(?:won|Won)(?:\s+the)?\s+(?:gold medal|competition|event|title)\b",
        rf"\b(?P<answer>{person_pattern})\b\s+(?:became|Became)\s+the\s+(?:winner|champion)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            return _normalize_fact_token(match.group("answer"))
    return ""


def _extract_search_count_candidate(query_text: str, info_payload: str) -> str:
    question_lower = _normalize_fact_token(query_text)
    cleaned = _search_info_cleaned_text(info_payload)
    if not question_lower or not cleaned:
        return ""

    if "municipalities" in question_lower:
        match = re.search(r"\b(?P<count>\d{1,4}(?:,\d{3})*)\s+municipalities\b", cleaned, flags=re.IGNORECASE)
        if match:
            return _normalize_fact_token(match.group("count"))
    return ""


def _extract_search_role_target_candidate(target_hint: str, info_payload: str) -> str:
    normalized_target = _normalize_fact_token(target_hint)
    cleaned = _search_info_cleaned_text(info_payload)
    if not normalized_target or not cleaned:
        return ""

    person_pattern = r"[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}"
    role_pattern = re.escape(normalized_target).replace(r"\ ", r"\s+")
    patterns = (
        rf"\bthe\s+{role_pattern}\b(?:\s+was|\s+is)\s+(?P<answer>{person_pattern})\b",
        rf"\b(?P<answer>{person_pattern})\b\s+(?:is|was)\s+(?:an?|the)\s+[^.;]{{0,60}}\b{role_pattern}\b",
        rf"\b(?:politician|prime\s+minister|president|governor|mayor|actor|actress|singer|author|writer|director)\s+"
        rf"(?P<answer>{person_pattern})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return _normalize_fact_token(match.group("answer"))
    return ""


def _extract_search_answer_candidate(query_text: str, info_payload: str) -> str:
    question_lower = _normalize_fact_token(query_text)
    if not question_lower or not info_payload:
        return ""

    if re.search(r"\b(member|members|composition|consists? of|make(?:s)? up|made up of)\b", question_lower):
        parenthetical_match = re.search(
            r"\bleaders?\s*\(([^()]{3,80})\)",
            str(info_payload or ""),
            flags=re.IGNORECASE,
        )
        if parenthetical_match:
            return _normalize_fact_token(parenthetical_match.group(1))

        phrase_match = re.search(
            r"\b(?:made up of|consists? of)\s+(?:the\s+)?([^.;]{3,80})",
            str(info_payload or ""),
            flags=re.IGNORECASE,
        )
        if phrase_match:
            candidate = re.sub(r"\s+of\b.*$", "", phrase_match.group(1), flags=re.IGNORECASE)
            return _normalize_fact_token(candidate)

    target_hint = _search_task_target_hint(query_text)
    if target_hint == "current role holder":
        candidate = _extract_search_current_role_holder_candidate(query_text, info_payload)
        if candidate:
            return candidate

    if target_hint == "currency":
        candidate = _extract_search_currency_candidate(info_payload)
        if candidate:
            return candidate

    if target_hint == "winner":
        candidate = _extract_search_winner_candidate(info_payload)
        if candidate:
            return candidate

    count_candidate = _extract_search_count_candidate(query_text, info_payload)
    if count_candidate:
        return count_candidate

    if target_hint and target_hint not in {"current role holder", "currency", "winner"}:
        candidate = _extract_search_role_target_candidate(target_hint, info_payload)
        if candidate:
            return candidate
    return ""


def _latest_search_information_line(lines: Sequence[str]) -> str:
    search_lines = _coalesce_open_domain_search_information_lines(lines)
    for line in reversed(search_lines):
        if _extract_information_payload(line):
            return str(line or "").strip()
    return ""


def _latest_search_query_line(lines: Sequence[str]) -> str:
    for line in reversed(tuple(lines)):
        if _extract_search_query(line):
            return str(line or "").strip()
    return ""


def _summarize_search_information_line(
    query_text: str,
    info_line: str,
    skills: MemorySkills,
) -> str:
    info_payload = _extract_information_payload(info_line)
    if not info_payload:
        return ""

    question_lower = _normalize_fact_token(query_text)
    answer_candidate = _normalize_fact_token(skills.search_answer_candidate or "")
    bridge_entity = _normalize_fact_token(skills.search_bridge_entity or "")
    target_hint = _normalize_fact_token(_search_task_target_hint(query_text))
    anchor_hint = _normalize_fact_token(skills.search_anchor or _search_task_anchor_hint(query_text))
    role_phrase = _normalize_fact_token(_search_current_role_phrase(query_text))

    summary_body = ""
    if answer_candidate:
        if target_hint == "current role holder" and role_phrase:
            summary_body = f"Current {role_phrase} is {answer_candidate}."
        elif target_hint == "currency" and anchor_hint:
            summary_body = f"Currency used in {anchor_hint} is {answer_candidate}."
        elif target_hint == "winner":
            summary_body = f"Winner is {answer_candidate}."
        elif re.search(r"\bhow many\b", question_lower):
            summary_body = f"Requested count is {answer_candidate}."
        elif target_hint in {"british prime minister", "prime minister", "president", "governor", "mayor"}:
            summary_body = f"The {target_hint} is {answer_candidate}."
        elif target_hint:
            summary_body = f"{target_hint} is {answer_candidate}."
        else:
            summary_body = f"Answer candidate is {answer_candidate}."
    elif bridge_entity:
        summary_body = f"Relevant bridge entity is {bridge_entity}."

    if not summary_body:
        return ""

    summary_line = f"<information>{summary_body}</information>"
    if len(summary_line) >= len(str(info_line or "").strip()):
        return ""
    return summary_line


def build_search_task_card_context(query_text: str) -> str:
    question = str(query_text or "").strip()
    if not question:
        return "[QUESTION]"

    lines = ["[QUESTION]", question]
    if not _is_open_domain_search_query(question):
        return "\n".join(lines)

    anchor = _search_task_anchor_hint(question)
    compare_hint = _search_task_compare_hint(question)
    bridge_hint = _search_task_bridge_hint(question)
    target_hint = _search_task_target_hint(question)
    scope_hint = _search_task_scope_hint(question)

    if anchor:
        lines.append(f"[SEARCH_ANCHOR] {anchor}")
    if compare_hint:
        lines.append(f"[SEARCH_COMPARE] {compare_hint}")
    if bridge_hint:
        lines.append(f"[SEARCH_BRIDGE] {bridge_hint}")
    if target_hint:
        lines.append(f"[SEARCH_TARGET] {target_hint}")
    if scope_hint:
        lines.append(f"[SEARCH_SCOPE] {scope_hint}")
    return "\n".join(lines)


def parse_goal_slots(query_text: str) -> GoalSlots:
    normalized = _normalize_fact_token(query_text)
    if not normalized:
        return GoalSlots()
    if _is_open_domain_search_query(query_text):
        return GoalSlots()

    state_family = _infer_state_task_family(query_text)
    receptacle_match = None
    for match in re.finditer(
        r"\b(?:in|into|inside|on|onto)\s+(?:the\s+)?(?P<receptacle>[a-z0-9][a-z0-9 _-]{0,40})\b",
        normalized,
    ):
        receptacle_match = match

    target_receptacle = None
    object_phrase = normalized
    if receptacle_match is not None:
        target_receptacle = _clean_slot_entity(receptacle_match.group("receptacle"))
        object_phrase = normalized[: receptacle_match.start()]

    if state_family == "look_at_obj_in_light":
        object_phrase = re.sub(r"^\s*(?:look|examine|inspect)(?:\s+at)?\s+", "", object_phrase).strip()
        object_phrase = re.split(
            r"\s+\b(?:under|below|beneath|by|near|next to|beside|with|using)\b\s+",
            object_phrase,
            maxsplit=1,
        )[0].strip()

    object_phrase = re.sub(r"\band\s+(?:put|place|store|bring|move)\b.*$", "", object_phrase).strip()
    object_phrase = re.sub(
        r"\b(?:put|place|find|get|take|fetch|locate|collect|move|bring|store|carry|hold)\b",
        " ",
        object_phrase,
    )
    object_phrase = re.sub(r"\b(?:your task is to|task is to)\b", " ", object_phrase)

    explicit_quantity = _extract_explicit_goal_quantity(object_phrase)
    singularize_plural = explicit_quantity > 1
    candidates = [
        re.sub(
            r"^(?:some|any|one|two|three|four|five|\d+)\s+",
            "",
            _canonicalize_goal_entity(part, singularize_plural=singularize_plural),
        ).strip()
        for part in re.split(r"\s+(?:and|plus)\s+|,\s*", object_phrase)
    ]
    target_objects = tuple(dict.fromkeys(candidate for candidate in candidates if candidate))
    return GoalSlots(
        target_objects=target_objects,
        target_receptacle=target_receptacle or None,
        target_object_count=max(explicit_quantity, len(target_objects)),
    )


def _query_requests_slot_tracking(query_text: str) -> bool:
    normalized = f" {_normalize_fact_token(query_text)} "
    goal_slots = parse_goal_slots(query_text)
    if _goal_uses_instance_tracking(goal_slots):
        return True
    if (
        goal_slots.target_objects
        and goal_slots.target_receptacle
        and any(marker in normalized for marker in SLOT_QUERY_MARKERS)
    ):
        return True
    if any(marker in normalized for marker in (" and put ", " then put ", " before putting ", " after putting ")):
        return True
    return False


def _is_low_value_navigation_location_line(
    line_text: str,
    *,
    goal_slots: GoalSlots,
    parsed_line: Optional[ParsedTrustLine] = None,
) -> bool:
    text = str(line_text or "").strip()
    if not text:
        return False

    parsed = parsed_line or _analyze_trust_line_cached(
        text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    if parsed.current_location is None:
        return False

    line_slot_kinds = set(parsed.line_slot_kinds)
    mentions_goal_object = bool(parsed.mentioned_goal_objects)
    mentions_receptacle = bool(
        goal_slots.target_receptacle and _line_mentions_entity(text, goal_slots.target_receptacle)
    )
    if (
        parsed.visible_anchor is not None
        or parsed.has_fact_signature
        or parsed.receptacle_state is not None
        or parsed.explicit_state is not None
        or parsed.known_state is not None
        or parsed.emergent_state is not None
        or parsed.pickup_object is not None
        or parsed.placement is not None
        or parsed.emergent_location is not None
        or parsed.emergent_progress is not None
        or line_slot_kinds.intersection({"inventory", "progress"})
        or mentions_goal_object
        or mentions_receptacle
    ):
        return False

    lowered = _normalize_fact_token(_observation_evidence_text(text))
    return any(
        marker in lowered
        for marker in (
            "you arrive at",
            "you are at",
            "[action]: go to",
            " go to ",
        )
    )


def _encode_skill_entity(entity: str) -> str:
    return _clean_slot_entity(entity).replace(" ", "_")


def _encode_search_phrase(text: str) -> str:
    return "_".join(token for token in _normalize_fact_token(text).split() if token)


def _decode_skill_entity(entity: str) -> str:
    return _clean_slot_entity(entity.replace("_", " "))


def _recent_unique_entities(entities: Sequence[str], limit: int = 4) -> Tuple[str, ...]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for entity in reversed([_clean_slot_entity(item) for item in entities if _clean_slot_entity(item)]):
        if entity in seen:
            continue
        seen.add(entity)
        ordered.append(entity)
        if len(ordered) >= limit:
            break
    return tuple(reversed(ordered))


def _extract_searched_without_target_location(
    line_text: str,
    goal_slots: GoalSlots,
    *,
    parsed_line: Optional[ParsedTrustLine] = None,
) -> Optional[str]:
    if not goal_slots.target_objects:
        return None

    text = str(line_text or "").strip()
    if not text:
        return None

    parsed = parsed_line or _analyze_trust_line_cached(
        text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    if parsed.mentioned_goal_objects:
        return None
    if parsed.pickup_object and _goal_object_for_entity(parsed.pickup_object, goal_slots):
        return None
    if parsed.placement is not None:
        placed_entity, _ = parsed.placement
        if _goal_object_for_entity(placed_entity, goal_slots):
            return None

    location = _clean_slot_entity(parsed.visible_anchor or parsed.current_location or "")
    if not location:
        return None
    if goal_slots.target_receptacle and _entities_match(location, goal_slots.target_receptacle):
        return None

    lowered = _normalize_fact_token(_observation_evidence_text(text))
    if not (
        "you see" in lowered
        or "is empty" in lowered
        or "looks empty" in lowered
        or "you see nothing" in lowered
    ):
        return None
    return location


def _prune_low_value_navigation_location_witnesses(
    skills: MemorySkills,
    goal_slots: GoalSlots,
) -> None:
    location_witnesses = list(skills.witness_lines_by_skill.get("location", ()))
    if len(location_witnesses) <= 1:
        return

    has_concrete_target_progress = any(
        state in {"located", "holding", "placed"}
        for state in _goal_progress_state_counts(skills.object_progress, goal_slots).keys()
    )
    has_late_phase_target_progress = any(
        state in {"holding", "placed"}
        for state in _goal_progress_state_counts(skills.object_progress, goal_slots).keys()
    )
    has_concrete_target_location = any(
        _goal_object_for_entity(goal_object, goal_slots) is not None and str(location or "").strip()
        for goal_object, location in skills.target_locations.items()
    )
    if not (has_concrete_target_progress or has_concrete_target_location):
        return

    filtered_witnesses = [
        line
        for line in location_witnesses
        if not _is_low_value_navigation_location_line(line, goal_slots=goal_slots)
    ]
    if not filtered_witnesses or len(filtered_witnesses) == len(location_witnesses):
        return

    kept_keys = {_normalized_line_key(line) for line in filtered_witnesses}
    role_mapping = dict(skills.witness_roles_by_skill.get("location", {}))
    has_agent_location = any(
        role_mapping.get(_normalized_line_key(line)) == "agent_location"
        for line in location_witnesses
    )
    if (
        has_agent_location
        and not has_late_phase_target_progress
        and not any(
            role_mapping.get(_normalized_line_key(line)) == "agent_location"
            for line in filtered_witnesses
        )
    ):
        for line in reversed(location_witnesses):
            normalized_key = _normalized_line_key(line)
            if role_mapping.get(normalized_key) != "agent_location":
                continue
            kept_keys.add(normalized_key)
            filtered_witnesses = [
                existing
                for existing in location_witnesses
                if _normalized_line_key(existing) in kept_keys
            ]
            break
    skills.witness_lines_by_skill["location"] = tuple(filtered_witnesses)
    skills.witness_roles_by_skill["location"] = {
        key: value for key, value in role_mapping.items() if key in kept_keys
    }


def _extract_pickup_object(line_text: str) -> Optional[str]:
    cleaned = _normalize_fact_token(_observation_evidence_text(line_text))
    match = PICKUP_PATTERN.search(cleaned)
    return _clean_slot_entity(match.group("object")) if match else None


def _extract_placement(line_text: str) -> Optional[Tuple[str, str]]:
    cleaned = _normalize_fact_token(_observation_evidence_text(line_text))
    match = PLACE_PATTERN.search(cleaned)
    if not match:
        return None
    return _clean_slot_entity(match.group("object")), _clean_slot_entity(match.group("receptacle"))


def _extract_receptacle_state(line_text: str) -> Optional[Tuple[str, bool]]:
    observation_text = _observation_evidence_text(line_text)
    for fragment in reversed(re.split(r"[.!?;\n]+", observation_text)):
        cleaned = _normalize_fact_token(fragment)
        if not cleaned:
            continue
        match = RECEPTACLE_STATE_PATTERN.search(cleaned)
        if not match:
            continue
        state = match.group("state").lower()
        return _clean_slot_entity(match.group("receptacle")), state in {"open", "unlocked"}
    return None


def _extract_current_location(line_text: str) -> Optional[str]:
    cleaned = " ".join(_observation_evidence_text(line_text).lower().split())
    match = ARRIVE_LOCATION_PATTERN.search(cleaned)
    return _clean_slot_entity(match.group("location")) if match else None


def _extract_visible_anchor(line_text: str) -> Optional[str]:
    cleaned = " ".join(_observation_evidence_text(line_text).lower().split())
    match = ANCHOR_OBSERVATION_PATTERN.search(cleaned)
    if not match:
        return None
    raw_anchor = _normalize_fact_token(match.group("anchor"))
    if raw_anchor in {"it", "there", "here"}:
        current_location = _extract_current_location(line_text)
        return current_location
    return _clean_slot_entity(match.group("anchor"))


def _extract_anchor_object_locations(
    line_text: str,
    goal_slots: GoalSlots,
) -> Tuple[Tuple[str, str], ...]:
    cleaned = " ".join(_observation_evidence_text(line_text).lower().split())
    match = ANCHOR_OBSERVATION_PATTERN.search(cleaned)
    if not match:
        return ()
    anchor = _extract_visible_anchor(line_text)
    if not anchor:
        return ()
    objects_text = _normalize_fact_token(match.group("objects"))
    if not objects_text or "nothing" in objects_text:
        return ()

    assignments: List[Tuple[str, str]] = []
    seen_subjects: Set[str] = set()
    for goal_object in goal_slots.target_objects:
        subject = _goal_progress_subject_for_entity(goal_object, goal_slots) or goal_object
        if not subject or subject in seen_subjects:
            continue
        if _goal_binding_is_distractor(line_text, goal_object, goal_slots):
            continue
        if _line_mentions_entity(objects_text, goal_object):
            seen_subjects.add(subject)
            assignments.append((subject, anchor))
    return tuple(assignments)


def _normalize_emergent_role_suffix(text: str) -> str:
    cleaned = _clean_slot_entity(text).replace(" ", "_")
    return cleaned[:48] if cleaned else ""


def _normalize_emergent_relation_suffix(text: str) -> str:
    cleaned = "_".join(token for token in _normalize_fact_token(text).split() if token)
    return cleaned[:48] if cleaned else ""


def _extract_known_light_state(
    line_text: str,
    *,
    goal_objects: Sequence[str],
) -> Optional[Tuple[str, str, str]]:
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))
    signature = _extract_fact_signature(line_text)
    if signature is not None:
        subject, value = signature
        subject_norm = _normalize_fact_token(subject)
        value_norm = _normalize_fact_token(value)
        if any(hint in subject_norm for hint in ("lamp", "desklamp", "switch", "light")):
            if value_norm in {"on", "illuminated", "bright"}:
                return subject, "light_on", "light_state"
            if value_norm in {"off", "dark"}:
                return subject, "light_off", "light_state"

    descriptor_match = STATE_DESCRIPTOR_PATTERN.search(lowered)
    if descriptor_match:
        subject = _clean_slot_entity(descriptor_match.group("subject"))
        value = _normalize_fact_token(descriptor_match.group("value")).strip()
        if any(hint in subject for hint in ("lamp", "desklamp", "switch", "light")):
            if value in {"on", "illuminated", "bright"}:
                return subject, "light_on", "light_state"
            if value in {"off", "dark"}:
                return subject, "light_off", "light_state"
        if subject in {"room", "area", "environment"}:
            if value in {"illuminated", "bright"}:
                return "environment", "light_on", "light_state"
            if value in {"off", "dark"}:
                return "environment", "light_off", "light_state"

    if "too dark" in lowered:
        return "environment", "light_off", "light_state"
    if "illuminated" in lowered or "bright enough" in lowered:
        return next(iter(goal_objects), "environment"), "light_on", "light_state"
    return None


def _extract_known_object_state(
    line_text: str,
    *,
    goal_objects: Sequence[str],
) -> Optional[Tuple[str, str, str]]:
    if _extract_receptacle_state(line_text) is not None:
        return None
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))
    lowered = re.sub(
        r"^(?:a\s+moment\s+later|moment\s+later|after\s+waiting|afterward|afterwards|now|already|then)\s+",
        "",
        lowered,
    )
    descriptor_match = STATE_DESCRIPTOR_PATTERN.search(lowered)
    if not descriptor_match:
        return None
    subject = _clean_slot_entity(descriptor_match.group("subject"))
    value = _normalize_fact_token(descriptor_match.group("value")).strip()
    state_name = {
        "steaming": "heated",
        "heated": "heated",
        "hot": "heated",
        "warm": "heated",
        "chilled": "cooled",
        "cooled": "cooled",
        "cold": "cooled",
        "icy": "cooled",
        "rinsed": "cleaned",
        "washed": "cleaned",
        "clean": "cleaned",
        "cleaned": "cleaned",
        "spotless": "cleaned",
        "dirty": "dirty",
    }.get(value)
    if state_name is None:
        return None
    normalized_subject = subject or next(iter(goal_objects), "environment")
    return normalized_subject, state_name, "object_state"


@lru_cache(maxsize=65536)
def _analyze_trust_line_cached(
    line_text: str,
    goal_objects: Tuple[str, ...],
    goal_receptacle: Optional[str],
) -> ParsedTrustLine:
    goal_slots = GoalSlots(target_objects=goal_objects, target_receptacle=goal_receptacle)
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))
    explicit_state = _extract_explicit_state_action(line_text)

    known_state: Optional[Tuple[str, str, str]] = None
    if explicit_state is None:
        known_state = _extract_known_light_state(line_text, goal_objects=goal_objects)
    if explicit_state is None and known_state is None:
        known_state = _extract_known_object_state(line_text, goal_objects=goal_objects)

    mentioned_goal_objects = tuple(
        goal_object
        for goal_object in goal_objects
        if _line_mentions_entity(line_text, goal_object)
    )

    return ParsedTrustLine(
        current_location=_extract_current_location(line_text),
        visible_anchor=_extract_visible_anchor(line_text),
        emergent_location=_extract_emergent_location_role(line_text, goal_slots),
        emergent_progress=_extract_emergent_progress_role(line_text, goal_slots),
        line_slot_kinds=tuple(sorted(classify_slot_kinds(line_text, goal_slots))),
        pickup_object=_extract_pickup_object(line_text),
        placement=_extract_placement(line_text),
        has_fact_signature=_extract_fact_signature(line_text) is not None,
        mentioned_goal_objects=mentioned_goal_objects,
        receptacle_state=_extract_receptacle_state(line_text),
        explicit_state=explicit_state,
        known_state=known_state,
        emergent_state=None if explicit_state is not None or known_state is not None else _extract_emergent_state_role(line_text, goal_slots),
    )


def _extract_emergent_state_role(
    line_text: str,
    goal_slots: GoalSlots,
) -> Optional[Tuple[str, str, str]]:
    cleaned_line = f" {_normalize_fact_token(_observation_evidence_text(line_text))} "
    if any(marker in cleaned_line for marker in (" is in ", " is on ", " is inside ", " is at ", " is into ", " is onto ")):
        return None
    if EMERGENT_LOCATION_PATTERN.search(cleaned_line):
        return None
    signature = _extract_fact_signature(line_text)
    if signature is not None:
        subject, value = signature
    else:
        cleaned = _normalize_fact_token(_observation_evidence_text(line_text))
        match = STATE_DESCRIPTOR_PATTERN.search(cleaned)
        if not match:
            return None
        subject = _clean_slot_entity(match.group("subject"))
        value = _normalize_fact_token(match.group("value")).strip()
    descriptor = _normalize_fact_token(value).strip()
    if not descriptor:
        return None
    if descriptor in CANONICAL_STATE_VALUES:
        return None
    role_suffix = _normalize_emergent_role_suffix(descriptor)
    if not role_suffix:
        return None
    if _goal_object_for_entity(subject, goal_slots):
        return subject, descriptor, f"emergent_object_state_{role_suffix}"
    if goal_slots.target_receptacle and _entities_match(subject, goal_slots.target_receptacle):
        return subject, descriptor, f"emergent_receptacle_state_{role_suffix}"
    return None


def _extract_explicit_state_action(line_text: str) -> Optional[Tuple[str, str]]:
    cleaned = _normalize_fact_token(_observation_evidence_text(line_text))
    match = STATE_ACTION_FEEDBACK_PATTERN.search(cleaned)
    if match:
        object_name = _clean_slot_entity(match.group("object"))
        state_name = {
            "heat": "heated",
            "clean": "cleaned",
            "cool": "cooled",
        }.get(match.group("verb").lower())
        if state_name is not None:
            return object_name or "environment", state_name

    toggle_match = LIGHT_TOGGLE_FEEDBACK_PATTERN.search(cleaned)
    if toggle_match:
        object_name = _clean_slot_entity(toggle_match.group("object"))
        return object_name or "environment", "light_on"

    return None


def _extract_emergent_location_role(
    line_text: str,
    goal_slots: GoalSlots,
) -> Optional[Tuple[str, str, str, str]]:
    cleaned = _normalize_fact_token(_observation_evidence_text(line_text))
    match = EMERGENT_LOCATION_PATTERN.search(cleaned)
    if not match:
        return None
    subject = _clean_slot_entity(match.group("subject"))
    relation = " ".join(match.group("relation").lower().split())
    value = _clean_slot_entity(match.group("value"))
    if not relation or not value:
        return None
    role_suffix = _normalize_emergent_relation_suffix(relation)
    if not role_suffix:
        return None
    goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
    if goal_subject:
        return goal_subject, relation, value, f"emergent_target_relation_{role_suffix}"
    if goal_slots.target_receptacle and _entities_match(subject, goal_slots.target_receptacle):
        return subject, relation, value, f"emergent_receptacle_relation_{role_suffix}"
    if subject == "you":
        return subject, relation, value, f"emergent_agent_relation_{role_suffix}"
    return None


def _extract_emergent_progress_role(
    line_text: str,
    goal_slots: GoalSlots,
) -> Optional[Tuple[str, str, str]]:
    cleaned = _normalize_fact_token(_observation_evidence_text(line_text))
    match = EMERGENT_PROGRESS_PATTERN.search(cleaned)
    if not match:
        return None
    progress_object = _clean_slot_entity(match.group("object"))
    progress_verb = _clean_slot_entity(match.group("verb"))
    if not progress_object or not progress_verb:
        return None
    goal_subject = _goal_progress_subject_for_entity(progress_object, goal_slots)
    if not goal_subject:
        return None
    role_suffix = _normalize_emergent_relation_suffix(progress_verb)
    if not role_suffix:
        return None
    return goal_subject, progress_verb, f"emergent_progress_action_{role_suffix}"


def _parse_skill_fields(body: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)=([^=]+?)(?=\s+[A-Za-z0-9_]+=|$)", str(body or "").strip()):
        key = str(match.group(1) or "").strip().lower()
        value = str(match.group(2) or "").strip()
        if key and value:
            fields[key] = value
    return fields


def _apply_existing_skill_line(line_text: str, skills: MemorySkills) -> bool:
    match = SKILL_LINE_PATTERN.match(line_text.strip())
    if not match:
        return False

    kind = match.group("kind").lower()
    body = match.group("body").strip()
    fields = _parse_skill_fields(body)

    if kind == "goal":
        targets_value = fields.get("targets")
        if targets_value and targets_value != "none":
            skills.goal_targets = tuple(
                target.strip() for target in targets_value.split(",") if target.strip()
            )
        target_count_value = fields.get("target_count")
        if target_count_value and str(target_count_value).strip().isdigit():
            skills.goal_target_count = int(str(target_count_value).strip())
        receptacle_value = fields.get("receptacle")
        if receptacle_value and receptacle_value != "none":
            skills.goal_receptacle = receptacle_value
        skills.confidence_by_skill["goal"] = max(skills.confidence_by_skill.get("goal", 0.0), 0.95)
        return True

    if kind == "location":
        if "agent" in fields and fields["agent"] != "unknown":
            skills.current_location = fields["agent"]
        if "anchor" in fields and fields["anchor"] != "unknown":
            skills.visible_anchor = fields["anchor"]
        for key, value in fields.items():
            if key in {"agent", "anchor", "conf"}:
                continue
            skills.target_locations[_decode_skill_entity(key)] = value
        conf_value = float(fields.get("conf", skills.confidence_by_skill.get("location", 0.9)))
        skills.confidence_by_skill["location"] = max(skills.confidence_by_skill.get("location", 0.0), conf_value)
        return True

    if kind == "progress":
        for key, value in fields.items():
            if key == "searched_without_target":
                skills.searched_without_target = tuple(
                    _decode_skill_entity(item)
                    for item in value.split(",")
                    if _decode_skill_entity(item)
                )
                continue
            if key == "bridge_entity":
                skills.search_bridge_entity = _decode_skill_entity(value)
                continue
            if key == "answer_candidate":
                skills.search_answer_candidate = _decode_skill_entity(value)
                continue
            skills.object_progress[_decode_skill_entity(key)] = value
        skills.confidence_by_skill["progress"] = max(skills.confidence_by_skill.get("progress", 0.0), 0.9)
        return True

    if kind == "state":
        for key, value in fields.items():
            skills.state_flags[_decode_skill_entity(key)] = value
        skills.confidence_by_skill["state"] = max(skills.confidence_by_skill.get("state", 0.0), 0.9)
        return True

    return False


def _has_observed_progress(progress: Dict[str, str]) -> bool:
    return any(value != "missing" for value in progress.values())


def _family_phase_scoped_kind(kind: str, family: Optional[str], phase: Optional[str] = None) -> str:
    family_key = str(family or "").strip()
    phase_key = str(phase or "").strip()
    if family_key and phase_key:
        return f"{family_key}::{phase_key}::{kind}"
    if family_key:
        return f"{family_key}::{kind}"
    return kind


def _skill_feedback_value(
    feedback: Optional[MemorySkillFeedback],
    field_name: str,
    kind: str,
    family: Optional[str] = None,
    phase: Optional[str] = None,
) -> float:
    if feedback is None:
        return 0.0
    mapping = getattr(feedback, field_name, None)
    if not isinstance(mapping, dict):
        return 0.0
    for candidate_kind in (
        _family_phase_scoped_kind(kind, family, phase),
        _family_phase_scoped_kind(kind, family, None),
        kind,
    ):
        if candidate_kind in mapping:
            return float(mapping.get(candidate_kind, 0.0))
    return float(mapping.get(kind, 0.0))


def _normalized_line_key(line_text: str) -> str:
    return " ".join(_strip_xml_tags(str(line_text)).lower().split())


def _witness_feedback_value(feedback: Optional[MemorySkillFeedback], line_text: str) -> float:
    if feedback is None:
        return 0.0
    witness_utility = getattr(feedback, "witness_utility_by_line", None)
    if not isinstance(witness_utility, dict):
        return 0.0
    return float(witness_utility.get(_normalized_line_key(line_text), 0.0))


def _skill_role_feedback_value(
    feedback: Optional[MemorySkillFeedback],
    field_name: str,
    kind: str,
    role: str,
    family: Optional[str] = None,
    phase: Optional[str] = None,
) -> float:
    if feedback is None or not field_name or not kind or not role:
        return 0.0
    role_mapping = getattr(feedback, field_name, None)
    if not isinstance(role_mapping, dict):
        return 0.0
    for candidate_kind in (
        _family_phase_scoped_kind(kind, family, phase),
        _family_phase_scoped_kind(kind, family, None),
        kind,
    ):
        kind_mapping = role_mapping.get(candidate_kind)
        if isinstance(kind_mapping, dict) and role in kind_mapping:
            return float(kind_mapping.get(role, 0.0))
    return 0.0


def _skill_role_feedback_score(
    feedback: Optional[MemorySkillFeedback],
    kind: str,
    role: str,
    family: Optional[str] = None,
    phase: Optional[str] = None,
) -> float:
    role_utility = _skill_role_feedback_value(feedback, "witness_role_utility_by_skill", kind, role, family, phase)
    role_support = _skill_role_feedback_value(feedback, "witness_role_support_by_skill", kind, role, family, phase)
    role_conflict = _skill_role_feedback_value(feedback, "witness_role_conflict_by_skill", kind, role, family, phase)
    return 1.0 * role_utility + 0.55 * role_support - 0.75 * role_conflict


def _decoupled_skill_feedback_bundle(
    feedback: Optional[MemorySkillFeedback],
    kind: str,
    role: str,
    family: Optional[str],
    phase: Optional[str],
    *,
    decouple_role_signals: bool,
) -> Dict[str, float]:
    bundle = {
        "bias": _skill_feedback_value(feedback, "fallback_bias_by_skill", kind, family, phase),
        "reliability": _skill_feedback_value(feedback, "reliability_by_skill", kind, family, phase),
        "support": _skill_feedback_value(feedback, "support_by_skill", kind, family, phase),
        "conflict": _skill_feedback_value(feedback, "conflict_by_skill", kind, family, phase),
        "rescue": _skill_feedback_value(feedback, "rescue_budget_by_skill", kind, family, phase),
        "utility": _skill_feedback_value(feedback, "utility_ema_by_skill", kind, family, phase),
        "uncertainty": _skill_feedback_value(feedback, "uncertainty_by_skill", kind, family, phase),
        "role_score": _skill_role_feedback_score(feedback, kind, role, family, phase),
    }
    if not decouple_role_signals:
        return bundle

    role_positive = max(0.0, bundle["role_score"])
    role_negative = max(0.0, -bundle["role_score"])
    bundle["support"] = min(1.0, bundle["support"] + 0.12 * role_positive)
    bundle["conflict"] = bundle["conflict"] * max(0.10, 1.0 - 0.85 * role_positive)
    positive_utility = max(0.0, bundle["utility"]) + 0.25 * role_positive
    negative_utility = min(0.0, bundle["utility"]) * max(0.10, 1.0 - 0.85 * role_positive)
    bundle["utility"] = max(-1.0, min(1.0, positive_utility + negative_utility))
    bundle["rescue"] = min(2.5, bundle["rescue"] + 0.35 * role_positive)
    bundle["uncertainty"] = bundle["uncertainty"] * max(0.20, 1.0 - 0.55 * role_positive)
    bundle["bias"] = bundle["bias"] * max(0.25, 1.0 - 0.65 * role_negative)
    bundle["support"] = bundle["support"] * max(0.20, 1.0 - 0.55 * role_negative)
    bundle["rescue"] = bundle["rescue"] * max(0.20, 1.0 - 0.65 * role_negative)
    bundle["uncertainty"] = min(1.0, bundle["uncertainty"] + 0.20 * role_negative)
    return bundle


def _clamp_signed_unit(value: float) -> float:
    return min(1.0, max(-1.0, float(value)))


def _adaptive_ema_momentum(
    base_momentum: float,
    previous_value: float,
    signal_value: float,
) -> float:
    base = min(0.95, max(1e-6, float(base_momentum)))
    previous = float(previous_value)
    signal = float(signal_value)
    if abs(previous) <= 1e-8 or abs(signal) <= 1e-8:
        return base
    if previous * signal > 0.0:
        stability = 1.0 - min(1.0, abs(previous - signal))
        scale = 1.0 + 0.75 * stability * max(abs(previous), abs(signal))
    else:
        scale = 0.55
    return min(0.95, max(base * 0.55, min(base * 1.75, base * scale)))


def _adaptive_rescue_budget(
    *,
    support: float,
    conflict: float,
    reliability: float,
    utility: float,
    bias: float,
    failure_ema: float,
) -> float:
    support = min(1.0, max(0.0, float(support)))
    conflict = min(1.0, max(0.0, float(conflict)))
    reliability = min(1.0, max(0.0, float(reliability)))
    utility = _clamp_signed_unit(utility)
    bias = min(3.0, max(0.0, float(bias)))
    failure_ema = min(1.0, max(0.0, float(failure_ema)))

    positive_utility = max(0.0, utility)
    negative_utility = max(0.0, -utility)
    evidence_value = min(1.0, 0.55 * support + 0.45 * positive_utility)
    risk_pressure = min(1.0, max(conflict, 1.0 - reliability, failure_ema))
    support_weight = 0.30 + 0.60 * evidence_value
    conflict_weight = 0.50 + 0.70 * risk_pressure * (0.40 + 0.60 * evidence_value)
    reliability_weight = 0.20 + 0.55 * risk_pressure * (0.35 + 0.65 * evidence_value)
    utility_weight = 0.25 + 0.75 * evidence_value
    bias_weight = 0.08 + 0.22 * failure_ema
    negative_penalty = 0.45 + 0.75 * (1.0 - evidence_value)

    rescue_budget = (
        support_weight * support
        + conflict_weight * conflict
        + reliability_weight * (1.0 - reliability)
        + utility_weight * positive_utility
        + bias_weight * bias
        - negative_penalty * negative_utility
    )
    return min(2.5, max(0.0, rescue_budget))


def _beta_uncertainty(alpha: float, beta: float) -> float:
    return _component_beta_uncertainty(alpha, beta)


def _schema_uncertainty(
    *,
    confidence: float,
    support: float,
    conflict: float,
    freshness: float,
    coverage: float,
    hit_count: int,
    miss_count: int,
) -> float:
    return _component_schema_uncertainty(
        confidence=confidence,
        support=support,
        conflict=conflict,
        freshness=freshness,
        coverage=coverage,
        hit_count=hit_count,
        miss_count=miss_count,
    )


def _schema_lifecycle_stage(
    *,
    support: float,
    conflict: float,
    freshness: float,
    eviction_pressure: float,
    uncertainty: float,
    age: int,
    hit_count: int,
    miss_count: int,
) -> str:
    return _component_schema_lifecycle_stage(
        support=support,
        conflict=conflict,
        freshness=freshness,
        eviction_pressure=eviction_pressure,
        uncertainty=uncertainty,
        age=age,
        hit_count=hit_count,
        miss_count=miss_count,
    )


def _schema_verification_signal(
    line_text: str,
    schema_witnesses: Sequence[SchemaWitness],
) -> float:
    return _component_schema_verification_signal(
        line_text,
        schema_witnesses,
        normalized_line_key=_normalized_line_key,
    )


def _skill_bias_targets(bias: float) -> Dict[str, float]:
    clipped_bias = min(3.0, max(0.0, float(bias)))
    return {
        "query_relevance": min(1.0, 0.55 + 0.10 * clipped_bias),
        "salience": min(1.0, 0.60 + 0.10 * clipped_bias),
        "source_trust": min(1.0, 0.60 + 0.08 * clipped_bias),
    }


def _skill_signal_targets(
    *,
    skill_conf: float,
    reliability: float,
    support: float,
    rescue: float,
    utility: float,
    uncertainty: float = 0.0,
    role_utility: float = 0.0,
    witness_utility: float = 0.0,
    witness_mode: bool,
    adaptive_skill_mix: bool,
) -> Dict[str, float]:
    return _component_skill_signal_targets(
        clamp_signed_unit=_clamp_signed_unit,
        skill_conf=skill_conf,
        reliability=reliability,
        support=support,
        rescue=rescue,
        utility=utility,
        uncertainty=uncertainty,
        role_utility=role_utility,
        witness_utility=witness_utility,
        witness_mode=witness_mode,
        adaptive_skill_mix=adaptive_skill_mix,
    )


def _append_skill_witness(
    skills: MemorySkills,
    kind: str,
    line_text: str,
    *,
    feedback: Optional[MemorySkillFeedback] = None,
    family: Optional[str] = None,
    phase: Optional[str] = None,
    role: str = "generic",
    limit: int = 3,
) -> None:
    normalized_line = " ".join(str(line_text).split()).strip()
    if not normalized_line:
        return
    current = list(skills.witness_lines_by_skill.get(kind, ()))
    role_mapping = dict(skills.witness_roles_by_skill.get(kind, {}))
    normalized_key = _normalized_line_key(normalized_line)
    if any(_normalized_line_key(existing) == normalized_key for existing in current):
        previous_role = role_mapping.get(normalized_key)
        if previous_role is None or _skill_witness_role_priority(kind, role) > _skill_witness_role_priority(kind, previous_role):
            role_mapping[normalized_key] = role
            skills.witness_roles_by_skill[kind] = role_mapping
        return

    candidate_lines = current + [normalized_line]
    role_mapping[normalized_key] = role
    selected_lines = _select_diversified_skill_witnesses(
        candidate_lines,
        kind,
        role_mapping,
        feedback=feedback,
        family=family,
        phase=phase,
        limit=limit,
    )
    selected_keys = {_normalized_line_key(line) for line in selected_lines}
    skills.witness_lines_by_skill[kind] = tuple(selected_lines)
    skills.witness_roles_by_skill[kind] = {
        key: value for key, value in role_mapping.items() if key in selected_keys
    }


def _skill_witness_role_priority(kind: str, role: str) -> int:
    if kind == "location":
        if role.startswith("emergent_target_relation_"):
            return 5
        if role.startswith("emergent_agent_relation_"):
            return 4
        if role.startswith("emergent_receptacle_relation_"):
            return 3
    if kind == "progress":
        if role.startswith("emergent_progress_action_"):
            return 5
    if kind == "state":
        if role.startswith("emergent_receptacle_state_"):
            return 3
        if role.startswith("emergent_object_state_"):
            return 2
    priorities = {
        "location": {
            "query_fact": 4,
            "target_fact": 4,
            "state_device_fact": 4,
            "agent_location": 3,
            "anchor": 2,
            "context_location": 1,
            "generic": 0,
        },
        "progress": {
            "placed": 4,
            "holding": 3,
            "located": 2,
            "inventory": 1,
            "generic": 0,
        },
        "state": {
            "receptacle_state": 3,
            "object_state": 2,
            "light_state": 4,
            "generic": 0,
        },
    }
    return int(priorities.get(kind, {}).get(role, priorities.get(kind, {}).get("generic", 0)))


def _select_diversified_skill_witnesses(
    candidate_lines: Sequence[str],
    kind: str,
    role_mapping: Dict[str, str],
    *,
    feedback: Optional[MemorySkillFeedback] = None,
    family: Optional[str] = None,
    phase: Optional[str] = None,
    limit: int,
) -> List[str]:
    def _chain_signature(line_text: str) -> Optional[Tuple[str, str]]:
        lowered = _normalize_fact_token(_observation_evidence_text(line_text))
        if kind == "progress":
            placement = _extract_placement(line_text)
            if placement is not None:
                placed_object, _ = placement
                return placed_object, "placed"
            pickup_object = _extract_pickup_object(line_text)
            if pickup_object:
                if "are carrying" in lowered or "carrying" in lowered or "hold" in lowered:
                    return pickup_object, "carry"
                if "pick up" in lowered or "take" in lowered or "grab" in lowered:
                    return pickup_object, "pickup"
                return pickup_object, "holding"
            signature = _extract_fact_signature(line_text)
            if signature is not None:
                subject, _ = signature
                return subject, "located"
            return None
        if kind == "location":
            current_location = _extract_current_location(line_text)
            if current_location is not None:
                return current_location, "agent_location"
            anchor = _extract_visible_anchor(line_text)
            if anchor is not None:
                return anchor, "anchor"
            signature = _extract_fact_signature(line_text)
            if signature is not None:
                _, value = signature
                return value, "target_fact"
            return None
        if kind == "state":
            receptacle_state = _extract_receptacle_state(line_text)
            if receptacle_state is not None:
                receptacle, _ = receptacle_state
                return receptacle, "receptacle_state"
            explicit_state = _extract_explicit_state_action(line_text)
            if explicit_state is not None:
                object_name, _ = explicit_state
                return object_name, "object_state"
            light_state = _extract_known_light_state(line_text, goal_objects=())
            if light_state is not None:
                subject, _, role = light_state
                return subject, role
            for token, state_name in (
                ("steaming", "object_state"),
                ("heated", "object_state"),
                ("chilled", "object_state"),
                ("cooled", "object_state"),
                ("rinsed", "object_state"),
                ("washed", "object_state"),
                ("clean", "object_state"),
                ("dirty", "object_state"),
            ):
                if token in lowered:
                    return "environment", state_name
            return None
        return None

    def _pair_chain_bonus(left_stage: str, right_stage: str) -> float:
        if left_stage == right_stage:
            return 0.0
        pair = {left_stage, right_stage}
        if kind == "progress":
            if pair == {"pickup", "carry"}:
                return 1.0
            if pair == {"carry", "placed"}:
                return 1.0
            if pair == {"pickup", "placed"}:
                return 0.75
            if "located" in pair and len(pair) == 2:
                return 0.55
            return 0.35
        if kind == "location":
            return 0.45
        if kind == "state":
            return 0.25
        return 0.0

    def _chain_support_score(line_text: str) -> float:
        signature = _chain_signature(line_text)
        if signature is None:
            return 0.0
        subject, stage = signature
        support = 0.0
        for peer_line in candidate_lines:
            if peer_line == line_text:
                continue
            peer_signature = _chain_signature(peer_line)
            if peer_signature is None:
                continue
            peer_subject, peer_stage = peer_signature
            if _entities_match(subject, peer_subject):
                support += _pair_chain_bonus(stage, peer_stage)
        return min(1.5, support)

    def _chain_increment_score(line_text: str, selected_lines: Sequence[str]) -> float:
        signature = _chain_signature(line_text)
        if signature is None:
            return 0.0
        subject, stage = signature
        increment = 0.0
        for selected_line in selected_lines:
            selected_signature = _chain_signature(selected_line)
            if selected_signature is None:
                continue
            selected_subject, selected_stage = selected_signature
            if _entities_match(subject, selected_subject):
                increment = max(increment, _pair_chain_bonus(stage, selected_stage))
        return increment

    selected: List[str] = []
    seen_roles: Set[str] = set()
    ordered_candidates = sorted(
        candidate_lines,
        key=lambda line: (
            -(
                1.5 * _skill_role_feedback_score(
                    feedback,
                    kind,
                    role_mapping.get(_normalized_line_key(line), "generic"),
                    family,
                    phase,
                )
                + 0.75 * _witness_feedback_value(feedback, line)
                + 0.90 * _chain_support_score(line)
            ),
            -_skill_witness_role_priority(kind, role_mapping.get(_normalized_line_key(line), "generic")),
            -candidate_lines.index(line),
        ),
    )
    for line in ordered_candidates:
        line_key = _normalized_line_key(line)
        role = role_mapping.get(line_key, "generic")
        role_score = _skill_role_feedback_score(feedback, kind, role, family, phase)
        chain_increment = _chain_increment_score(line, selected)
        if role in seen_roles and chain_increment < 0.75:
            continue
        if role_score <= -0.15 and len(candidate_lines) > limit and chain_increment < 0.50:
            continue
        selected.append(line)
        seen_roles.add(role)
        if len(selected) >= limit:
            return selected

    for line in ordered_candidates:
        if len(selected) >= limit:
            break
        if line not in selected:
            selected.append(line)
    return selected[:limit]


def _adaptive_skill_witness_limit(
    feedback: Optional[MemorySkillFeedback],
    kind: str,
    family: Optional[str] = None,
    phase: Optional[str] = None,
    *,
    default_limit: int = 3,
    min_limit: int = 2,
    max_limit: int = 4,
) -> int:
    if feedback is None:
        return default_limit

    support = _skill_feedback_value(feedback, "support_by_skill", kind, family, phase)
    conflict = _skill_feedback_value(feedback, "conflict_by_skill", kind, family, phase)
    utility = _skill_feedback_value(feedback, "utility_ema_by_skill", kind, family, phase)
    rescue = _skill_feedback_value(feedback, "rescue_budget_by_skill", kind, family, phase)

    positive_signal = max(support, conflict, max(0.0, utility), rescue / 2.0)
    if positive_signal >= 0.45:
        return max_limit
    if utility <= -0.20 and support < 0.25 and rescue < 0.60:
        return min_limit
    return default_limit


def _line_matches_skill_witness(line_text: str, witness_lines: Sequence[str]) -> bool:
    candidate_key = _normalized_line_key(line_text)
    return any(_normalized_line_key(witness_line) == candidate_key for witness_line in witness_lines)


def _schema_line_key(lines: Sequence[str]) -> str:
    return "||".join(_normalized_line_key(line) for line in lines if line and str(line).strip())


def _schema_last_line_index(schema: SchemaWitness, line_index_by_key: Dict[str, int]) -> int:
    indices = [line_index_by_key.get(_normalized_line_key(line), -1) for line in schema.lines]
    return max(indices) if indices else -1


def _schema_fact_signature(schema: SchemaWitness, goal_slots: GoalSlots) -> Optional[Tuple[str, str]]:
    if schema.subject and schema.current_value and schema.relation_family in {
        "agent_room",
        "object_location",
        "device_location",
    }:
        return schema.subject, schema.current_value
    for line in reversed(schema.lines):
        placement = _extract_placement(line)
        if placement is not None:
            placed_object, receptacle = placement
            goal_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
            if goal_subject:
                return goal_subject, receptacle
        for subject, value in _extract_anchor_object_locations(line, goal_slots):
            if subject:
                return subject, value
        signature = _extract_fact_signature(line)
        if signature is None:
            pickup_object = _extract_pickup_object(line)
            if pickup_object:
                goal_subject = _goal_progress_subject_for_entity(pickup_object, goal_slots)
                if goal_subject:
                    return goal_subject, "inventory"
            continue
        subject, value = signature
        if _goal_binding_is_distractor(line, subject, goal_slots):
            continue
        goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
        if goal_subject:
            return goal_subject, value
    return None


def _schema_state_signature(schema: SchemaWitness, goal_slots: GoalSlots) -> Optional[Tuple[str, str]]:
    if schema.subject and schema.current_value and schema.relation_family in {
        "receptacle_state",
        "object_state",
        "light_state",
        "emergent_state",
    }:
        return schema.subject, schema.current_value
    for line in schema.lines:
        receptacle_state = _extract_receptacle_state(line)
        if receptacle_state is not None:
            receptacle, is_open = receptacle_state
            return receptacle, "open" if is_open else "closed"
        explicit_state = _extract_explicit_state_action(line)
        if explicit_state is not None:
            object_name, state_name = explicit_state
            subject = _goal_progress_subject_for_entity(object_name, goal_slots) or object_name or next(iter(goal_slots.target_objects), "environment")
            return subject, state_name
        light_state = _extract_known_light_state(line, goal_objects=goal_slots.target_objects)
        if light_state is not None:
            subject, state_name, _ = light_state
            return subject, state_name
        object_state = _extract_known_object_state(line, goal_objects=goal_slots.target_objects)
        if object_state is not None:
            subject, state_name, _ = object_state
            subject = _goal_progress_subject_for_entity(subject, goal_slots) or subject
            return subject, state_name
    return None


def _schema_progress_signature(schema: SchemaWitness, goal_slots: GoalSlots) -> Optional[Tuple[str, str]]:
    if schema.subject and schema.current_value and schema.relation_family in {
        "task_progress",
        "inventory",
    }:
        return schema.subject, schema.current_value
    for line in schema.lines:
        placement = _extract_placement(line)
        if placement is not None:
            placed_object, receptacle = placement
            goal_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
            if goal_subject:
                return goal_subject, f"placed:{receptacle}"
        pickup_object = _extract_pickup_object(line)
        if pickup_object:
            goal_subject = _goal_progress_subject_for_entity(pickup_object, goal_slots)
            if goal_subject:
                return goal_subject, "holding"
    return None


def _refresh_schema_witnesses(
    skills: MemorySkills,
    lines: Sequence[str],
    goal_slots: GoalSlots,
) -> None:
    _component_refresh_schema_witnesses(
        skills,
        lines,
        goal_slots,
        schema_last_line_index=_schema_last_line_index,
        schema_fact_signature=_schema_fact_signature,
        schema_state_signature=_schema_state_signature,
        schema_progress_signature=_schema_progress_signature,
        goal_object_for_entity=_goal_object_for_entity,
        extract_fact_signature=_extract_fact_signature,
        extract_anchor_object_locations=_extract_anchor_object_locations,
        extract_placement=_extract_placement,
        extract_pickup_object=_extract_pickup_object,
    )


def _append_schema_witness(
    skills: MemorySkills,
    kind: str,
    lines: Sequence[str],
    *,
    role: str,
    confidence: float = 1.0,
    relation_family: str = "",
    subject: str = "",
    current_value: str = "",
    pointer_lines: Sequence[str] = (),
    support_witness_lines: Sequence[str] = (),
    support_witness_roles: Sequence[str] = (),
    min_lines: int = 2,
) -> None:
    normalized_lines = tuple(line for line in lines if line and str(line).strip())
    if len(normalized_lines) < max(1, int(min_lines)):
        return
    current = list(skills.schema_witnesses_by_skill.get(kind, ()))
    candidate_key = _schema_line_key(normalized_lines)
    if not candidate_key:
        return
    normalized_pointer_lines = tuple(line for line in pointer_lines if line and str(line).strip())
    normalized_support_lines = tuple(line for line in support_witness_lines if line and str(line).strip())
    normalized_support_roles = tuple(
        str(role_name)
        for role_name in support_witness_roles
        if role_name and str(role_name).strip()
    )
    if any(
        _schema_line_key(existing.lines) == candidate_key
        and str(existing.role) == str(role)
        and str(existing.relation_family or "") == str(relation_family or "")
        and str(existing.subject or "") == str(subject or "")
        and str(existing.current_value or "") == str(current_value or "")
        for existing in current
    ):
        return
    current.append(
        SchemaWitness(
            lines=normalized_lines,
            role=str(role),
            confidence=min(1.0, max(0.0, float(confidence))),
            hit_count=1,
            coverage=1.0,
            uncertainty=max(0.20, 0.82 - 0.45 * min(1.0, max(0.0, float(confidence)))),
            relation_family=str(relation_family or ""),
            subject=str(subject or ""),
            current_value=str(current_value or ""),
            pointer_lines=normalized_pointer_lines,
            support_witness_lines=normalized_support_lines,
            support_witness_roles=normalized_support_roles,
        )
    )
    skills.schema_witnesses_by_skill[kind] = tuple(current)


def _ordered_unique_lines(lines: Sequence[str]) -> Tuple[str, ...]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        key = _normalized_line_key(text)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return tuple(ordered)


def _typed_schema_confidence(
    *,
    base: float,
    support_count: int,
) -> float:
    return min(0.98, max(0.50, float(base) + 0.04 * max(0, int(support_count))))


def _inject_typed_alfworld_schema_witnesses(
    skills: MemorySkills,
    lines: Sequence[str],
    goal_slots: GoalSlots,
    *,
    state_device_hint: Optional[str] = None,
) -> None:
    if not lines:
        return

    runtime_slots = RuntimeSlots(
        object_progress={goal_object: "missing" for goal_object in goal_slots.target_objects if goal_object}
    )
    latest_arrival_line_by_location: Dict[str, str] = {}
    latest_anchor_line_by_location: Dict[str, str] = {}
    latest_open_line_by_receptacle: Dict[str, str] = {}
    latest_pickup_line_by_subject: Dict[str, str] = {}
    latest_state_device_line: Dict[str, str] = {}
    saw_typed_signal = False

    def _append_typed_schema(
        kind: str,
        *,
        pointer_line: str,
        support_lines: Sequence[str],
        support_roles: Sequence[str],
        relation_family: str,
        subject: str,
        current_value: str,
        role: str,
        base_confidence: float,
    ) -> None:
        ordered_pointer_lines = _ordered_unique_lines((pointer_line,))
        ordered_support_lines = _ordered_unique_lines(tuple(support_lines))
        combined_lines = _ordered_unique_lines(ordered_pointer_lines + ordered_support_lines)
        effective_support_lines = ordered_support_lines if ordered_support_lines else ordered_pointer_lines
        if not combined_lines:
            return
        _append_schema_witness(
            skills,
            kind,
            combined_lines,
            role=role,
            confidence=_typed_schema_confidence(
                base=base_confidence,
                support_count=len(effective_support_lines),
            ),
            relation_family=relation_family,
            subject=subject,
            current_value=current_value,
            pointer_lines=ordered_pointer_lines,
            support_witness_lines=effective_support_lines,
            support_witness_roles=tuple(str(item) for item in support_roles if item and str(item).strip()),
            min_lines=1,
        )

    for line_text in lines:
        text = str(line_text or "").strip()
        if not text:
            continue
        parsed_line = _analyze_trust_line_cached(
            text,
            goal_slots.target_objects,
            goal_slots.target_receptacle,
        )
        location_hint = runtime_slots.current_location

        if parsed_line.current_location:
            saw_typed_signal = True
            latest_arrival_line_by_location[parsed_line.current_location] = text
            _append_typed_schema(
                "location",
                pointer_line=text,
                support_lines=(),
                support_roles=("agent_location",),
                relation_family="agent_room",
                subject="agent",
                current_value=parsed_line.current_location,
                role="typed_agent_location",
                base_confidence=0.84,
            )

        if parsed_line.visible_anchor:
            saw_typed_signal = True
            latest_anchor_line_by_location[parsed_line.visible_anchor] = text

        for goal_subject, anchor_value in _extract_anchor_object_locations(text, goal_slots):
            saw_typed_signal = True
            support_lines: List[str] = []
            support_roles: List[str] = []
            arrival_line = latest_arrival_line_by_location.get(anchor_value)
            if arrival_line and _normalized_line_key(arrival_line) != _normalized_line_key(text):
                support_lines.append(arrival_line)
                support_roles.append("agent_location")
            _append_typed_schema(
                "location",
                pointer_line=text,
                support_lines=support_lines,
                support_roles=support_roles or ("anchor",),
                relation_family="object_location",
                subject=goal_subject,
                current_value=anchor_value,
                role="typed_object_location_anchor",
                base_confidence=0.90,
            )

        if parsed_line.has_fact_signature:
            signature = _extract_fact_signature(text)
            if signature is not None:
                subject, value = signature
                if not _goal_binding_is_distractor(text, subject, goal_slots):
                    goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
                    if goal_subject:
                        saw_typed_signal = True
                        support_lines = []
                        support_roles = []
                        arrival_line = latest_arrival_line_by_location.get(value)
                        if arrival_line and _normalized_line_key(arrival_line) != _normalized_line_key(text):
                            support_lines.append(arrival_line)
                            support_roles.append("agent_location")
                        anchor_line = latest_anchor_line_by_location.get(value)
                        if anchor_line and _normalized_line_key(anchor_line) != _normalized_line_key(text):
                            support_lines.append(anchor_line)
                            support_roles.append("anchor")
                        _append_typed_schema(
                            "location",
                            pointer_line=text,
                            support_lines=support_lines,
                            support_roles=support_roles or ("target_fact",),
                            relation_family="object_location",
                            subject=goal_subject,
                            current_value=value,
                            role="typed_object_location_fact",
                            base_confidence=0.88,
                        )
                if state_device_hint and _entities_match(subject, state_device_hint):
                    saw_typed_signal = True
                    latest_state_device_line[state_device_hint] = text
                    _append_typed_schema(
                        "location",
                        pointer_line=text,
                        support_lines=(),
                        support_roles=("state_device_fact",),
                        relation_family="device_location",
                        subject=state_device_hint,
                        current_value=value,
                        role="typed_device_location",
                        base_confidence=0.86,
                    )

        if parsed_line.receptacle_state is not None:
            saw_typed_signal = True
            receptacle, is_open = parsed_line.receptacle_state
            if is_open:
                latest_open_line_by_receptacle[receptacle] = text
            _append_typed_schema(
                "state",
                pointer_line=text,
                support_lines=(),
                support_roles=("receptacle_state",),
                relation_family="receptacle_state",
                subject=receptacle,
                current_value="open" if is_open else "closed",
                role="typed_receptacle_state",
                base_confidence=0.88,
            )

        if parsed_line.pickup_object:
            goal_subject = _goal_progress_subject_for_entity(parsed_line.pickup_object, goal_slots)
            if goal_subject:
                saw_typed_signal = True
                latest_pickup_line_by_subject[goal_subject] = text
                support_lines = []
                support_roles = []
                if location_hint:
                    arrival_line = latest_arrival_line_by_location.get(location_hint)
                    if arrival_line and _normalized_line_key(arrival_line) != _normalized_line_key(text):
                        support_lines.append(arrival_line)
                        support_roles.append("agent_location")
                _append_typed_schema(
                    "location",
                    pointer_line=text,
                    support_lines=support_lines,
                    support_roles=support_roles or ("holding",),
                    relation_family="object_location",
                    subject=goal_subject,
                    current_value="inventory",
                    role="typed_inventory_location",
                    base_confidence=0.90,
                )
                _append_typed_schema(
                    "progress",
                    pointer_line=text,
                    support_lines=support_lines,
                    support_roles=support_roles or ("holding",),
                    relation_family="task_progress",
                    subject=goal_subject,
                    current_value="holding",
                    role="typed_progress_holding",
                    base_confidence=0.90,
                )
                _append_typed_schema(
                    "progress",
                    pointer_line=text,
                    support_lines=(),
                    support_roles=("holding",),
                    relation_family="inventory",
                    subject="agent",
                    current_value=goal_subject,
                    role="typed_inventory_pointer",
                    base_confidence=0.90,
                )

        if parsed_line.placement is not None:
            placed_object, receptacle = parsed_line.placement
            goal_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
            if goal_subject:
                saw_typed_signal = True
                support_lines = []
                support_roles = []
                pickup_line = latest_pickup_line_by_subject.get(goal_subject)
                if pickup_line and _normalized_line_key(pickup_line) != _normalized_line_key(text):
                    support_lines.append(pickup_line)
                    support_roles.append("holding")
                open_line = latest_open_line_by_receptacle.get(receptacle)
                if open_line and _normalized_line_key(open_line) != _normalized_line_key(text):
                    support_lines.append(open_line)
                    support_roles.append("receptacle_state")
                _append_typed_schema(
                    "location",
                    pointer_line=text,
                    support_lines=support_lines,
                    support_roles=support_roles or ("placed",),
                    relation_family="object_location",
                    subject=goal_subject,
                    current_value=receptacle,
                    role="typed_object_location_place",
                    base_confidence=0.93,
                )
                _append_typed_schema(
                    "progress",
                    pointer_line=text,
                    support_lines=support_lines,
                    support_roles=support_roles or ("placed",),
                    relation_family="task_progress",
                    subject=goal_subject,
                    current_value=f"placed:{receptacle}",
                    role="typed_progress_place",
                    base_confidence=0.94,
                )
                _append_typed_schema(
                    "progress",
                    pointer_line=text,
                    support_lines=(),
                    support_roles=("placed",),
                    relation_family="inventory",
                    subject="agent",
                    current_value="empty",
                    role="typed_inventory_empty",
                    base_confidence=0.82,
                )

        if parsed_line.emergent_progress is not None:
            progress_object, progress_value, progress_role = parsed_line.emergent_progress
            saw_typed_signal = True
            _append_typed_schema(
                "progress",
                pointer_line=text,
                support_lines=(),
                support_roles=(progress_role,),
                relation_family="task_progress",
                subject=progress_object,
                current_value=progress_value,
                role="typed_progress_emergent",
                base_confidence=0.78,
            )

        explicit_state = parsed_line.explicit_state
        if explicit_state is not None:
            object_name, state_name = explicit_state
            subject = _goal_progress_subject_for_entity(object_name, goal_slots) or object_name or next(iter(goal_slots.target_objects), "environment")
            saw_typed_signal = True
            _append_typed_schema(
                "state",
                pointer_line=text,
                support_lines=(),
                support_roles=("object_state",),
                relation_family="object_state",
                subject=subject,
                current_value=state_name,
                role="typed_object_state",
                base_confidence=0.92,
            )
        elif parsed_line.known_state is not None:
            subject, state_name, state_role = parsed_line.known_state
            if state_role != "light_state":
                subject = _goal_progress_subject_for_entity(subject, goal_slots) or subject
            support_lines = []
            support_roles = []
            if state_role == "light_state" and state_device_hint:
                device_line = latest_state_device_line.get(state_device_hint)
                if device_line and _normalized_line_key(device_line) != _normalized_line_key(text):
                    support_lines.append(device_line)
                    support_roles.append("state_device_fact")
            saw_typed_signal = True
            _append_typed_schema(
                "state",
                pointer_line=text,
                support_lines=support_lines,
                support_roles=support_roles or (state_role,),
                relation_family=state_role,
                subject=subject,
                current_value=state_name,
                role=f"typed_{state_role}",
                base_confidence=0.88 if state_role == "light_state" else 0.90,
            )
        elif parsed_line.emergent_state is not None:
            subject, state_name, emergent_role = parsed_line.emergent_state
            saw_typed_signal = True
            _append_typed_schema(
                "state",
                pointer_line=text,
                support_lines=(),
                support_roles=(emergent_role,),
                relation_family="emergent_state",
                subject=subject,
                current_value=state_name,
                role="typed_emergent_state",
                base_confidence=0.74,
            )

        _apply_slot_updates(text, goal_slots, runtime_slots)

    if not saw_typed_signal:
        return


def _schema_component_tags(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
) -> Tuple[str, ...]:
    return _component_schema_component_tags(
        schema,
        kind,
        goal_slots,
        normalize_fact_token=_normalize_fact_token,
        strip_xml_tags=_strip_xml_tags,
        extract_current_location=_extract_current_location,
        extract_fact_signature=_extract_fact_signature,
        goal_object_for_entity=_goal_object_for_entity,
        extract_visible_anchor=_extract_visible_anchor,
        extract_anchor_object_locations=_extract_anchor_object_locations,
        extract_pickup_object=_extract_pickup_object,
        extract_placement=_extract_placement,
        extract_receptacle_state=_extract_receptacle_state,
    )


def _schema_template_signature(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
) -> Optional[Tuple[str, ...]]:
    return _component_schema_template_signature(
        schema,
        kind,
        goal_slots,
        normalize_fact_token=_normalize_fact_token,
        strip_xml_tags=_strip_xml_tags,
        extract_current_location=_extract_current_location,
        extract_fact_signature=_extract_fact_signature,
        goal_object_for_entity=_goal_object_for_entity,
        extract_visible_anchor=_extract_visible_anchor,
        extract_anchor_object_locations=_extract_anchor_object_locations,
        extract_pickup_object=_extract_pickup_object,
        extract_placement=_extract_placement,
        extract_receptacle_state=_extract_receptacle_state,
    )


def _schema_primary_subject(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
) -> str:
    return _component_schema_primary_subject(
        schema,
        kind,
        goal_slots,
        schema_fact_signature=_schema_fact_signature,
        schema_progress_signature=_schema_progress_signature,
        schema_state_signature=_schema_state_signature,
        goal_object_for_entity=_goal_object_for_entity,
        normalize_fact_token=_normalize_fact_token,
    )


def _schema_split_lines(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
    line_key_set: Set[str],
) -> Tuple[str, ...]:
    return _component_schema_split_lines(
        schema,
        kind,
        goal_slots,
        line_key_set,
        normalized_line_key=_normalized_line_key,
        extract_fact_signature=_extract_fact_signature,
        goal_object_for_entity=_goal_object_for_entity,
        extract_visible_anchor=_extract_visible_anchor,
        extract_anchor_object_locations=_extract_anchor_object_locations,
        extract_current_location=_extract_current_location,
        extract_pickup_object=_extract_pickup_object,
        extract_placement=_extract_placement,
        extract_receptacle_state=_extract_receptacle_state,
        normalize_fact_token=_normalize_fact_token,
        strip_xml_tags=_strip_xml_tags,
    )


def _best_schema_representative(
    schemas: Sequence[SchemaWitness],
) -> SchemaWitness:
    return _component_best_schema_representative(schemas)


def _evolve_schema_witnesses(
    skills: MemorySkills,
    lines: Sequence[str],
    goal_slots: GoalSlots,
) -> None:
    _component_evolve_schema_witnesses(
        skills,
        lines,
        goal_slots,
        schema_cls=SchemaWitness,
        normalized_line_key=_normalized_line_key,
        schema_line_key=_schema_line_key,
        normalize_emergent_role_suffix=_normalize_emergent_role_suffix,
        schema_template_signature_fn=_schema_template_signature,
        schema_primary_subject_fn=_schema_primary_subject,
        best_schema_representative_fn=_best_schema_representative,
        schema_split_lines_fn=_schema_split_lines,
    )


def _inject_subject_merge_schemas(
    skills: MemorySkills,
    goal_slots: GoalSlots,
) -> None:
    location_schemas = list(skills.schema_witnesses_by_skill.get("location", ()))
    if not location_schemas:
        return

    grouped: Dict[Tuple[str, str], List[SchemaWitness]] = {}
    for schema in location_schemas:
        relation_family, subject, current_value = _schema_relation_subject_value(schema, "location", goal_slots)
        if relation_family not in {"object_location", "device_location"}:
            continue
        if not subject or not current_value:
            continue
        grouped.setdefault((relation_family, subject), []).append(schema)

    for (relation_family, subject), members in grouped.items():
        if len(members) < 2:
            continue
        merged_role_prefix = f"merged_location_{_normalize_emergent_role_suffix(subject)}"
        if any(str(schema.role).startswith(merged_role_prefix) for schema in location_schemas):
            continue
        representative = _best_schema_representative(members)
        _append_schema_witness(
            skills,
            "location",
            representative.lines,
            role=merged_role_prefix,
            confidence=max(0.72, float(representative.confidence)),
            relation_family=relation_family,
            subject=subject,
            current_value=representative.current_value or _schema_relation_subject_value(representative, "location", goal_slots)[2],
            pointer_lines=representative.pointer_lines,
            support_witness_lines=representative.support_witness_lines,
            support_witness_roles=representative.support_witness_roles,
            min_lines=1,
        )


def _goal_object_for_entity(entity: Optional[str], goal_slots: GoalSlots) -> Optional[str]:
    for goal_object in goal_slots.target_objects:
        if _entities_match(entity, goal_object):
            return goal_object
    return None


def _target_receptacle_matches(receptacle: Optional[str], goal_slots: GoalSlots) -> bool:
    return bool(goal_slots.target_receptacle and _entities_match(receptacle, goal_slots.target_receptacle))


def _all_goal_objects_placed(runtime_slots: RuntimeSlots, goal_slots: GoalSlots) -> bool:
    total_targets = _goal_target_count(goal_slots)
    if total_targets <= 0:
        return False
    return _goal_progress_state_count(runtime_slots.object_progress, goal_slots, "placed") >= total_targets


def _same_class_distractor(line_text: str, goal_slots: GoalSlots) -> bool:
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))
    if not any(marker in lowered for marker in DISTRACTOR_MARKERS):
        return False
    line_tokens = _entity_tokens(lowered)
    return any(_entity_head(goal_object) in line_tokens for goal_object in goal_slots.target_objects)


def _slot_line_score(line_text: str, goal_slots: GoalSlots, runtime_slots: RuntimeSlots) -> float:
    if not goal_slots.target_objects and not goal_slots.target_receptacle:
        return 0.0

    score = 0.0
    placement = _extract_placement(line_text)
    pickup_object = _extract_pickup_object(line_text)
    receptacle_state = _extract_receptacle_state(line_text)
    signature = _extract_fact_signature(line_text)

    if placement:
        placed_entity, receptacle = placement
        goal_subject = _goal_progress_subject_for_entity(placed_entity, goal_slots)
        if goal_subject and runtime_slots.object_progress.get(goal_subject, "missing") != "placed":
            score = max(score, 1.0 if _target_receptacle_matches(receptacle, goal_slots) else 0.65)

    if pickup_object:
        goal_subject = _goal_progress_subject_for_entity(pickup_object, goal_slots)
        if goal_subject and runtime_slots.object_progress.get(goal_subject, "missing") in {"missing", "located"}:
            score = max(score, 0.9)

    if signature:
        subject, value = signature
        goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
        if goal_subject and runtime_slots.object_progress.get(goal_subject, "missing") == "missing":
            score = max(score, 0.78)
        if goal_slots.target_receptacle and _entities_match(subject, goal_slots.target_receptacle):
            score = max(score, 0.65 if not _all_goal_objects_placed(runtime_slots, goal_slots) else 0.25)
        if goal_slots.target_receptacle and _entities_match(value, goal_slots.target_receptacle) and goal_subject:
            score = max(score, 0.85)

    if receptacle_state:
        receptacle, is_open = receptacle_state
        if _target_receptacle_matches(receptacle, goal_slots) and is_open and not _all_goal_objects_placed(runtime_slots, goal_slots):
            score = max(score, 0.85)

    for goal_subject in _goal_progress_subject_keys(runtime_slots.object_progress, goal_slots):
        if _line_mentions_entity(line_text, goal_subject) and runtime_slots.object_progress.get(goal_subject) == "placed":
            score = min(score, 0.2) if score > 0 else 0.15

    if _same_class_distractor(line_text, goal_slots):
        score = min(score, 0.1) if score > 0 else 0.05

    return score


def _apply_slot_updates(line_text: str, goal_slots: GoalSlots, runtime_slots: RuntimeSlots) -> None:
    placement = _extract_placement(line_text)
    pickup_object = _extract_pickup_object(line_text)
    receptacle_state = _extract_receptacle_state(line_text)
    signature = _extract_fact_signature(line_text)
    current_location = _extract_current_location(line_text)
    anchor_locations = _extract_anchor_object_locations(line_text, goal_slots)

    if current_location:
        runtime_slots.current_location = current_location

    if pickup_object:
        goal_subject = _goal_progress_subject_for_entity(pickup_object, goal_slots)
        runtime_slots.inventory_object = goal_subject or pickup_object
        if goal_subject:
            runtime_slots.object_progress[goal_subject] = "holding"
            runtime_slots.object_locations[goal_subject] = "inventory"
            for contents in runtime_slots.receptacle_contents.values():
                contents.discard(goal_subject)

    if signature:
        subject, value = signature
        if _goal_binding_is_distractor(line_text, subject, goal_slots):
            subject = None
        goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
        if goal_subject:
            if runtime_slots.object_progress.get(goal_subject, "missing") == "missing":
                runtime_slots.object_progress[goal_subject] = "located"
            runtime_slots.object_locations[goal_subject] = value

    for subject, value in anchor_locations:
        if runtime_slots.object_progress.get(subject, "missing") == "missing":
            runtime_slots.object_progress[subject] = "located"
        runtime_slots.object_locations[subject] = value

    if receptacle_state:
        receptacle, is_open = receptacle_state
        runtime_slots.receptacle_open[receptacle] = is_open

    if placement:
        placed_entity, receptacle = placement
        goal_subject = _goal_progress_subject_for_entity(placed_entity, goal_slots)
        if goal_subject:
            runtime_slots.object_progress[goal_subject] = "placed"
            runtime_slots.inventory_object = None
            runtime_slots.object_locations[goal_subject] = receptacle
            for contents in runtime_slots.receptacle_contents.values():
                contents.discard(goal_subject)
            runtime_slots.receptacle_contents.setdefault(receptacle, set()).add(goal_subject)


def _infer_schema_witnesses(
    skills: MemorySkills,
    lines: Sequence[str],
    goal_slots: GoalSlots,
) -> None:
    if not lines:
        return

    location_line_by_value: Dict[str, str] = {}
    target_fact_line_by_value: Dict[str, str] = {}
    anchor_target_line_by_value: Dict[str, str] = {}
    anchor_lines: List[Tuple[str, str]] = []
    pickup_line_by_object: Dict[str, str] = {}
    placement_line_by_object: Dict[str, str] = {}
    receptacle_open_lines: Dict[str, str] = {}
    latest_search_query_line_by_anchor: Dict[str, str] = {}
    latest_search_anchor = ""
    query_result_lines_by_anchor: Dict[str, str] = {}

    for line_text in lines:
        search_query = _extract_search_query(line_text)
        if search_query:
            latest_search_anchor = _normalize_search_query_anchor(search_query)
            if latest_search_anchor:
                latest_search_query_line_by_anchor[latest_search_anchor] = line_text
        current_location = _extract_current_location(line_text)
        if current_location and current_location not in location_line_by_value:
            location_line_by_value[current_location] = line_text

        signature = _extract_fact_signature(line_text)
        if signature is not None:
            subject, value = signature
            if _goal_binding_is_distractor(line_text, subject, goal_slots):
                subject = None
            goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
            if goal_subject and value not in target_fact_line_by_value:
                target_fact_line_by_value[value] = line_text
        elif latest_search_anchor and _extract_information_payload(line_text):
            query_result_lines_by_anchor[latest_search_anchor] = line_text

        visible_anchor = _extract_visible_anchor(line_text)
        if visible_anchor:
            anchor_lines.append((visible_anchor, line_text))
            for _, anchor_value in _extract_anchor_object_locations(line_text, goal_slots):
                if anchor_value not in anchor_target_line_by_value:
                    anchor_target_line_by_value[anchor_value] = line_text

        pickup_object = _extract_pickup_object(line_text)
        if pickup_object:
            goal_subject = _goal_progress_subject_for_entity(pickup_object, goal_slots)
            if goal_subject and goal_subject not in pickup_line_by_object:
                pickup_line_by_object[goal_subject] = line_text

        placement = _extract_placement(line_text)
        if placement is not None:
            placed_object, receptacle = placement
            goal_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
            if goal_subject and _target_receptacle_matches(receptacle, goal_slots) and goal_subject not in placement_line_by_object:
                placement_line_by_object[goal_subject] = line_text

        receptacle_state = _extract_receptacle_state(line_text)
        if receptacle_state is not None:
            receptacle, is_open = receptacle_state
            if is_open and _target_receptacle_matches(receptacle, goal_slots) and receptacle not in receptacle_open_lines:
                receptacle_open_lines[receptacle] = line_text

    for location_value, current_line in location_line_by_value.items():
        target_fact_line = target_fact_line_by_value.get(location_value) or anchor_target_line_by_value.get(location_value)
        if target_fact_line is None:
            continue
        bundle_lines = [current_line, target_fact_line]
        anchor_line = next(
            (
                line_text
                for anchor_value, line_text in anchor_lines
                if _entities_match(anchor_value, location_value) or _line_mentions_entity(line_text, location_value)
            ),
            None,
        )
        if anchor_line is not None:
            bundle_lines.append(anchor_line)
            _append_schema_witness(
                skills,
                "location",
                bundle_lines,
                role="schema_location_bundle",
                confidence=0.85 if anchor_line is not None else 0.70,
            )

    for goal_object in goal_slots.target_objects:
        pickup_line = pickup_line_by_object.get(goal_object)
        placement_line = placement_line_by_object.get(goal_object)
        if pickup_line is not None and placement_line is not None:
            _append_schema_witness(
                skills,
                "location",
                [pickup_line, placement_line],
                role="schema_location_move_chain",
                confidence=0.88,
            )
        if pickup_line is not None and skills.current_location:
            current_line = location_line_by_value.get(skills.current_location)
            if current_line is not None:
                _append_schema_witness(
                    skills,
                    "location",
                    [current_line, pickup_line],
                    role="schema_inventory_pointer",
                    confidence=0.76,
                )

    if goal_slots.target_receptacle:
        receptacle_open_line = next(iter(receptacle_open_lines.values()), None)
        for goal_object in goal_slots.target_objects:
            pickup_line = pickup_line_by_object.get(goal_object)
            placement_line = placement_line_by_object.get(goal_object)
            if pickup_line is None or placement_line is None:
                continue
            chain_lines = [pickup_line]
            if receptacle_open_line is not None:
                chain_lines.append(receptacle_open_line)
            chain_lines.append(placement_line)
            _append_schema_witness(
                skills,
                "progress",
                chain_lines,
                role="schema_placement_chain",
                confidence=0.90 if receptacle_open_line is not None else 0.75,
            )

    if not goal_slots.target_objects and not goal_slots.target_receptacle:
        for anchor, result_line in query_result_lines_by_anchor.items():
            search_query_line = latest_search_query_line_by_anchor.get(anchor, "")
            if not search_query_line:
                continue
            current_value = _canonicalize_query_fact_value(_extract_information_payload(result_line))
            if not current_value:
                continue
            _append_schema_witness(
                skills,
                "location",
                [search_query_line, result_line],
                role="schema_query_result_bundle",
                confidence=0.72,
                relation_family="query_result",
                subject=anchor,
                current_value=current_value,
                pointer_lines=(result_line,),
                support_witness_lines=(search_query_line,),
                support_witness_roles=("query_anchor",),
                min_lines=1,
            )


def _schema_line_signal(
    line_text: str,
    schema_witnesses: Sequence[SchemaWitness],
) -> float:
    line_key = _normalized_line_key(line_text)
    best_positive = 0.0
    strongest_negative = 0.0
    for schema in schema_witnesses:
        schema_keys = {_normalized_line_key(line) for line in schema.lines}
        if line_key not in schema_keys:
            continue
        stage_positive = {
            "proto": 0.92,
            "stable": 1.08,
            "stale": 0.72,
            "retired": 0.20,
        }.get(schema.lifecycle_stage, 1.0)
        stage_negative = {
            "proto": 0.90,
            "stable": 0.75,
            "stale": 1.15,
            "retired": 1.35,
        }.get(schema.lifecycle_stage, 1.0)
        size_bonus = max(0.0, min(0.20, 0.10 * (len(schema.lines) - 2)))
        positive_signal = min(
            1.0,
            (
                0.10
                + size_bonus
                + 0.18 * float(schema.confidence)
                + 0.26 * float(schema.support)
                + 0.16 * float(schema.freshness)
                + 0.12 * float(schema.coverage)
                - 0.12 * float(schema.uncertainty)
            ) * stage_positive,
        )
        negative_signal = min(
            1.0,
            (
                0.55 * float(schema.conflict)
                + 0.50 * float(schema.eviction_pressure)
                + 0.18 * float(schema.uncertainty)
            ) * stage_negative,
        )
        if positive_signal >= negative_signal:
            best_positive = max(best_positive, positive_signal - 0.40 * negative_signal)
        else:
            strongest_negative = max(strongest_negative, negative_signal - 0.35 * positive_signal)
    return max(-1.0, min(1.0, best_positive - strongest_negative))


def _schema_pivot_entities(skills: MemorySkills, kind: str, goal_slots: GoalSlots) -> Tuple[str, ...]:
    pivots: List[str] = []
    if kind in {"location", "progress", "state"}:
        pivots.extend(goal_slots.target_objects)
    if goal_slots.target_receptacle and kind in {"location", "progress", "state"}:
        pivots.append(goal_slots.target_receptacle)
    if kind == "location":
        if skills.current_location:
            pivots.append(skills.current_location)
        if skills.visible_anchor:
            pivots.append(skills.visible_anchor)
        pivots.extend(skills.target_locations.values())
    elif kind == "progress":
        pivots.extend(skills.object_progress.keys())
    elif kind == "state":
        pivots.extend(skills.state_flags.keys())
    ordered: List[str] = []
    seen: Set[str] = set()
    for pivot in pivots:
        normalized = _normalize_fact_token(pivot)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(pivot)
    return tuple(ordered)


def _infer_latent_entity_schema_witnesses(
    skills: MemorySkills,
    goal_slots: GoalSlots,
) -> None:
    for kind in ("location", "progress", "state"):
        witness_lines = skills.witness_lines_by_skill.get(kind, ())
        if len(witness_lines) < 2:
            continue
        roles_by_line = skills.witness_roles_by_skill.get(kind, {})
        for pivot in _schema_pivot_entities(skills, kind, goal_slots):
            member_lines = [line for line in witness_lines if _line_mentions_entity(line, pivot)]
            if len(member_lines) < 2:
                continue
            unique_roles = {
                roles_by_line.get(_normalized_line_key(line), "generic")
                for line in member_lines
            }
            if len(unique_roles) < 2 and len(member_lines) < 3:
                continue
            size_bonus = min(0.12, 0.04 * max(0, len(member_lines) - 2))
            diversity_bonus = min(0.10, 0.05 * max(0, len(unique_roles) - 1))
            confidence = min(0.92, 0.62 + size_bonus + diversity_bonus)
            pivot_suffix = _normalize_emergent_role_suffix(_entity_head(pivot) or pivot)
            _append_schema_witness(
                skills,
                kind,
                member_lines,
                role=f"latent_{kind}_entity_bundle_{pivot_suffix}",
                confidence=confidence,
            )


def _compile_open_domain_search_skills_from_lines(
    lines: Sequence[str],
    query_text: str,
    *,
    feedback: Optional[MemorySkillFeedback] = None,
    adaptive_witness_budget: bool = True,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
) -> MemorySkills:
    lines = _coalesce_open_domain_search_information_lines(lines)
    framework = resolve_memory_framework_config(memory_framework_config)
    feedback_family = infer_monitor_family(query_text)
    feedback_phase = infer_monitor_phase(query_text, lines)
    search_unresolved_history = bool(
        not any(_extract_information_payload(line) for line in lines)
    )
    skills = MemorySkills(
        goal_targets=(),
        goal_target_count=0,
        goal_receptacle=None,
        object_progress={},
        confidence_by_skill={
            "goal": 0.0,
            "location": 0.0,
            "progress": 0.0,
            "state": 0.0,
        },
    )
    if search_unresolved_history:
        skills.search_anchor = _search_task_anchor_hint(query_text)
        skills.search_compare_hint = _search_task_compare_hint(query_text)
        skills.search_target_hint = _search_task_target_hint(query_text)
        skills.search_bridge_hint = _search_task_bridge_hint(query_text)
        skills.search_scope_hint = _search_task_scope_hint(query_text)
    witness_limit_by_kind = {
        kind: (
            _adaptive_skill_witness_limit(feedback, kind, feedback_family, feedback_phase)
            if adaptive_witness_budget
            else 3
        )
        for kind in ("location", "progress", "state")
    }
    active_search_anchor = _search_task_anchor_hint(query_text) or _normalize_search_query_anchor(query_text)
    active_search_line = ""

    for line_text in lines:
        search_query = _extract_search_query(line_text)
        if search_query:
            active_search_anchor = _normalize_search_query_anchor(search_query) or active_search_anchor
            active_search_line = str(line_text or "").strip()

        if _apply_existing_skill_line(line_text, skills):
            continue

        info_payload = _extract_information_payload(line_text)
        if not info_payload or not active_search_anchor:
            continue

        skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.58)
        bridge_entity = _extract_search_bridge_entity_candidate(query_text, info_payload)
        if bridge_entity:
            skills.search_bridge_entity = bridge_entity
            skills.confidence_by_skill["progress"] = max(skills.confidence_by_skill["progress"], 0.58)
            _append_skill_witness(
                skills,
                "progress",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="search_bridge_entity",
                limit=witness_limit_by_kind["progress"],
            )
        answer_candidate = _extract_search_answer_candidate(query_text, info_payload)
        if answer_candidate:
            skills.search_answer_candidate = answer_candidate
            skills.confidence_by_skill["progress"] = max(skills.confidence_by_skill["progress"], 0.68)
            _append_skill_witness(
                skills,
                "progress",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="search_answer_candidate",
                limit=witness_limit_by_kind["progress"],
            )
        _append_skill_witness(
            skills,
            "location",
            line_text,
            feedback=feedback,
            family=feedback_family,
            phase=feedback_phase,
            role="query_fact",
            limit=witness_limit_by_kind["location"],
        )
        fact_value = _canonicalize_query_fact_value(info_payload)
        if active_search_line and fact_value:
            _append_schema_witness(
                skills,
                "location",
                [active_search_line, line_text],
                role="schema_query_result_bundle",
                confidence=0.72,
                relation_family="query_result",
                subject=active_search_anchor,
                current_value=fact_value,
                pointer_lines=(str(line_text or "").strip(),),
                support_witness_lines=(active_search_line,),
                support_witness_roles=("query_anchor",),
                min_lines=1,
            )

    if framework.enable_ema:
        _prune_low_value_navigation_location_witnesses(skills, GoalSlots())
    return skills


def _slot_line_scores(lines: Sequence[str], goal_slots: GoalSlots) -> List[float]:
    runtime_slots = RuntimeSlots(
        object_progress={goal_object: "missing" for goal_object in goal_slots.target_objects}
    )
    scores: List[float] = []
    for line_text in lines:
        scores.append(_slot_line_score(line_text, goal_slots, runtime_slots))
        _apply_slot_updates(line_text, goal_slots, runtime_slots)
    return scores


def compile_memory_skills_from_lines(
    lines: Sequence[str],
    query_text: str,
    feedback: Optional[MemorySkillFeedback] = None,
    adaptive_witness_budget: bool = True,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
) -> MemorySkills:
    framework = resolve_memory_framework_config(memory_framework_config)
    feedback_family = infer_monitor_family(query_text)
    feedback_phase = infer_monitor_phase(query_text, lines)
    generic_search_query = _is_open_domain_search_query(query_text)
    if generic_search_query:
        return _compile_open_domain_search_skills_from_lines(
            lines,
            query_text,
            feedback=feedback,
            adaptive_witness_budget=adaptive_witness_budget,
            memory_framework_config=memory_framework_config,
        )
    goal_slots = GoalSlots() if generic_search_query else parse_goal_slots(query_text)
    state_device_hint = _state_device_hint_for_family(feedback_family, query_text=query_text)
    runtime_slots = RuntimeSlots(
        object_progress={goal_object: "missing" for goal_object in goal_slots.target_objects if goal_object}
    )
    search_unresolved_history = bool(
        generic_search_query
        and not any(_extract_information_payload(line) for line in lines)
    )
    search_task_card_only = bool(
        search_unresolved_history
        and any(str(line or "").strip().startswith("[SEARCH_") for line in lines)
        and not any(_extract_search_query(line) for line in lines)
    )
    skills = MemorySkills(
        goal_targets=goal_slots.target_objects,
        goal_target_count=_goal_target_count(goal_slots),
        goal_receptacle=goal_slots.target_receptacle,
        object_progress=dict(runtime_slots.object_progress),
        confidence_by_skill={
            "goal": 0.95 if goal_slots.target_objects or goal_slots.target_receptacle else 0.0,
            "location": 0.0,
            "progress": 0.0,
            "state": 0.0,
        },
    )
    if search_unresolved_history:
        skills.search_anchor = _search_task_anchor_hint(query_text)
        skills.search_compare_hint = _search_task_compare_hint(query_text)
        skills.search_target_hint = _search_task_target_hint(query_text)
        skills.search_bridge_hint = _search_task_bridge_hint(query_text)
        skills.search_scope_hint = _search_task_scope_hint(query_text)
    witness_limit_by_kind = {
        kind: (
            _adaptive_skill_witness_limit(feedback, kind, feedback_family, feedback_phase)
            if adaptive_witness_budget
            else 3
        )
        for kind in ("location", "progress", "state")
    }
    searched_without_target: List[str] = []
    active_search_anchor = (
        _search_task_anchor_hint(query_text) or _normalize_search_query_anchor(query_text)
    ) if generic_search_query else ""
    active_search_line = ""

    for line_text in lines:
        search_query = _extract_search_query(line_text)
        if search_query:
            active_search_anchor = _normalize_search_query_anchor(search_query) or active_search_anchor
            active_search_line = str(line_text or "").strip()

        if _apply_existing_skill_line(line_text, skills):
            continue

        parsed_line = _analyze_trust_line_cached(
            str(line_text),
            goal_slots.target_objects,
            goal_slots.target_receptacle,
        )
        line_slot_kinds = set(parsed_line.line_slot_kinds)
        searched_location = _extract_searched_without_target_location(
            line_text,
            goal_slots,
            parsed_line=parsed_line,
        )
        if searched_location is not None:
            searched_without_target.append(searched_location)
            skills.searched_without_target = _recent_unique_entities(searched_without_target, limit=4)
            skills.confidence_by_skill["progress"] = max(skills.confidence_by_skill["progress"], 0.35)

        if parsed_line.current_location:
            skills.current_location = parsed_line.current_location
            skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.85)
            _append_skill_witness(
                skills,
                "location",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="agent_location",
                limit=witness_limit_by_kind["location"],
            )

        if parsed_line.visible_anchor:
            skills.visible_anchor = parsed_line.visible_anchor
            skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.75)
            _append_skill_witness(
                skills,
                "location",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="anchor",
                limit=witness_limit_by_kind["location"],
            )
            if state_device_hint and _line_mentions_entity(line_text, state_device_hint):
                skills.target_locations[state_device_hint] = parsed_line.visible_anchor
                skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.82)
                _append_skill_witness(
                    skills,
                    "location",
                    line_text,
                    feedback=feedback,
                    family=feedback_family,
                    phase=feedback_phase,
                    role="state_device_fact",
                    limit=witness_limit_by_kind["location"],
                )

        for goal_subject, anchor_value in _extract_anchor_object_locations(line_text, goal_slots):
            skills.target_locations[goal_subject] = anchor_value
            skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.88)
            _append_skill_witness(
                skills,
                "location",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="target_fact",
                limit=witness_limit_by_kind["location"],
            )

        if parsed_line.emergent_location is not None:
            subject, relation, value, emergent_role = parsed_line.emergent_location
            if subject != "you":
                skills.target_locations[subject] = f"{relation}:{value}"
            skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.78)
            _append_skill_witness(
                skills,
                "location",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role=emergent_role,
                limit=witness_limit_by_kind["location"],
            )

        _apply_slot_updates(line_text, goal_slots, runtime_slots)
        signature = _extract_fact_signature(line_text)
        if signature is not None and state_device_hint:
            subject, value = signature
            if _entities_match(subject, state_device_hint):
                skills.target_locations[subject] = value
                skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.82)
                _append_skill_witness(
                    skills,
                    "location",
                    line_text,
                    feedback=feedback,
                    family=feedback_family,
                    phase=feedback_phase,
                    role="state_device_fact",
                    limit=witness_limit_by_kind["location"],
                )
        if parsed_line.emergent_progress is not None:
            progress_object, progress_value, _ = parsed_line.emergent_progress
            runtime_slots.object_progress[progress_object] = progress_value
        skills.object_progress = dict(runtime_slots.object_progress)
        if runtime_slots.object_progress:
            skills.confidence_by_skill["progress"] = max(
                skills.confidence_by_skill["progress"],
                0.8 if any(value != "missing" for value in runtime_slots.object_progress.values()) else 0.2,
            )
        if line_slot_kinds.intersection({"inventory", "progress"}):
            progress_role = "inventory"
            if parsed_line.pickup_object:
                progress_role = "holding"
            elif parsed_line.placement:
                progress_role = "placed"
            elif parsed_line.has_fact_signature:
                progress_role = "located"
            elif parsed_line.emergent_progress is not None:
                _, _, progress_role = parsed_line.emergent_progress
            _append_skill_witness(
                skills,
                "progress",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role=progress_role,
                limit=witness_limit_by_kind["progress"],
            )

        for goal_object, value in runtime_slots.object_locations.items():
            skills.target_locations[goal_object] = value
            skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.9)
            if _line_mentions_entity(line_text, goal_object) and not _goal_binding_is_distractor(line_text, goal_object, goal_slots):
                _append_skill_witness(
                    skills,
                    "location",
                    line_text,
                    feedback=feedback,
                    family=feedback_family,
                    phase=feedback_phase,
                    role="target_fact",
                    limit=witness_limit_by_kind["location"],
                )

        if parsed_line.receptacle_state:
            receptacle, is_open = parsed_line.receptacle_state
            skills.state_flags[receptacle] = "open" if is_open else "closed"
            skills.confidence_by_skill["state"] = max(skills.confidence_by_skill["state"], 0.8)
            _append_skill_witness(
                skills,
                "state",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="receptacle_state",
                limit=witness_limit_by_kind["state"],
            )

        matched_known_state_role = False
        if parsed_line.explicit_state is not None:
            matched_known_state_role = True
            object_name, state_name = parsed_line.explicit_state
            subject = _goal_progress_subject_for_entity(object_name, goal_slots) or object_name or next(iter(goal_slots.target_objects), "environment")
            skills.state_flags[subject] = state_name
            skills.confidence_by_skill["state"] = max(skills.confidence_by_skill["state"], 0.9)
            _append_skill_witness(
                skills,
                "state",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="object_state",
                limit=witness_limit_by_kind["state"],
            )
        elif parsed_line.known_state is not None:
            matched_known_state_role = True
            subject, state_name, state_role = parsed_line.known_state
            if state_role != "light_state":
                subject = _goal_progress_subject_for_entity(subject, goal_slots) or subject
            skills.state_flags[subject] = state_name
            skills.confidence_by_skill["state"] = max(skills.confidence_by_skill["state"], 0.9)
            _append_skill_witness(
                skills,
                "state",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role=state_role,
                limit=witness_limit_by_kind["state"],
            )
        if not matched_known_state_role:
            if parsed_line.emergent_state is not None:
                subject, state_name, emergent_role = parsed_line.emergent_state
                skills.state_flags[subject] = state_name
                skills.confidence_by_skill["state"] = max(skills.confidence_by_skill["state"], 0.75)
                _append_skill_witness(
                    skills,
                    "state",
                    line_text,
                    feedback=feedback,
                    family=feedback_family,
                    phase=feedback_phase,
                    role=emergent_role,
                    limit=witness_limit_by_kind["state"],
                )
        if "location" in line_slot_kinds:
            _append_skill_witness(
                skills,
                "location",
                line_text,
                feedback=feedback,
                family=feedback_family,
                phase=feedback_phase,
                role="context_location",
                limit=witness_limit_by_kind["location"],
            )

        if generic_search_query and active_search_anchor:
            info_payload = _extract_information_payload(line_text)
            if info_payload:
                skills.confidence_by_skill["location"] = max(skills.confidence_by_skill["location"], 0.58)
                bridge_entity = _extract_search_bridge_entity_candidate(query_text, info_payload)
                if bridge_entity:
                    skills.search_bridge_entity = bridge_entity
                    skills.confidence_by_skill["progress"] = max(skills.confidence_by_skill["progress"], 0.58)
                    _append_skill_witness(
                        skills,
                        "progress",
                        line_text,
                        feedback=feedback,
                        family=feedback_family,
                        phase=feedback_phase,
                        role="search_bridge_entity",
                        limit=witness_limit_by_kind["progress"],
                    )
                answer_candidate = _extract_search_answer_candidate(query_text, info_payload)
                if answer_candidate:
                    skills.search_answer_candidate = answer_candidate
                    skills.confidence_by_skill["progress"] = max(skills.confidence_by_skill["progress"], 0.68)
                    _append_skill_witness(
                        skills,
                        "progress",
                        line_text,
                        feedback=feedback,
                        family=feedback_family,
                        phase=feedback_phase,
                        role="search_answer_candidate",
                        limit=witness_limit_by_kind["progress"],
                    )
                _append_skill_witness(
                    skills,
                    "location",
                    line_text,
                    feedback=feedback,
                    family=feedback_family,
                    phase=feedback_phase,
                    role="query_fact",
                    limit=witness_limit_by_kind["location"],
                )
                fact_value = _canonicalize_query_fact_value(info_payload)
                if active_search_line and fact_value:
                    _append_schema_witness(
                        skills,
                        "location",
                        [active_search_line, line_text],
                        role="schema_query_result_bundle",
                        confidence=0.72,
                        relation_family="query_result",
                        subject=active_search_anchor,
                        current_value=fact_value,
                        pointer_lines=(str(line_text or "").strip(),),
                        support_witness_lines=(active_search_line,),
                        support_witness_roles=("query_anchor",),
                        min_lines=1,
                    )

    _prune_low_value_navigation_location_witnesses(skills, goal_slots)

    if feedback is not None:
        for kind in ("goal", "location", "progress", "state"):
            bias = _skill_feedback_value(feedback, "fallback_bias_by_skill", kind, feedback_family, feedback_phase)
            reliability = _skill_feedback_value(feedback, "reliability_by_skill", kind, feedback_family, feedback_phase)
            support = _skill_feedback_value(feedback, "support_by_skill", kind, feedback_family, feedback_phase)
            conflict = _skill_feedback_value(feedback, "conflict_by_skill", kind, feedback_family, feedback_phase)
            utility = _skill_feedback_value(feedback, "utility_ema_by_skill", kind, feedback_family, feedback_phase)
            if bias > 0.0:
                skills.confidence_by_skill[kind] = min(
                    1.0,
                    skills.confidence_by_skill.get(kind, 0.0) + 0.1 * bias,
                )
            if reliability > 0.0 or support > 0.0 or conflict > 0.0 or utility > 0.0:
                skills.confidence_by_skill[kind] = min(
                    1.0,
                    max(
                        0.0,
                        skills.confidence_by_skill.get(kind, 0.0)
                        + 0.08 * reliability
                        + 0.05 * support
                        + 0.10 * utility
                        - 0.10 * conflict,
                    ),
                )

    if framework.enable_ema:
        _infer_schema_witnesses(skills, lines, goal_slots)
        _infer_latent_entity_schema_witnesses(skills, goal_slots)
        _inject_typed_alfworld_schema_witnesses(
            skills,
            lines,
            goal_slots,
            state_device_hint=state_device_hint,
        )
        _refresh_schema_witnesses(skills, lines, goal_slots)
        _evolve_schema_witnesses(skills, lines, goal_slots)
        _inject_subject_merge_schemas(skills, goal_slots)
    return skills


def synthesize_skill_lines(
    skills: MemorySkills,
    *,
    max_lines: int = 4,
    include_low_confidence: bool = False,
    family: Optional[str] = None,
    query_text: Optional[str] = None,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
) -> List[str]:
    framework = resolve_memory_framework_config(memory_framework_config)
    if not framework.enable_ema:
        return []
    lines: List[str] = []
    goal_fields: List[str] = []
    if skills.goal_targets or skills.goal_receptacle:
        goal_targets = ",".join(skills.goal_targets) if skills.goal_targets else "none"
        receptacle = skills.goal_receptacle or "none"
        goal_fields.extend([f"targets={goal_targets}", f"receptacle={receptacle}"])
        goal_target_count = max(int(skills.goal_target_count or 0), len(skills.goal_targets))
        if goal_target_count > 1:
            goal_fields.append(f"target_count={goal_target_count}")
        required_state = _required_state_for_family(family)
        if required_state:
            goal_fields.append(f"required_state={required_state}")
        state_device = _state_device_hint_for_family(family, query_text=query_text)
        if state_device:
            goal_fields.append(f"state_device={_encode_skill_entity(state_device)}")
    if skills.search_anchor:
        goal_fields.append(f"search_anchor={_encode_skill_entity(skills.search_anchor)}")
    if skills.search_bridge_hint:
        goal_fields.append(f"search_bridge={_encode_skill_entity(skills.search_bridge_hint)}")
    if skills.search_compare_hint:
        goal_fields.append(f"search_compare={_encode_search_phrase(skills.search_compare_hint)}")
    if skills.search_target_hint:
        goal_fields.append(f"search_target={_encode_skill_entity(skills.search_target_hint)}")
    if skills.search_scope_hint:
        goal_fields.append(f"search_scope={_encode_skill_entity(skills.search_scope_hint)}")
    if goal_fields:
        lines.append(f"[SKILL][goal] {' '.join(goal_fields)}")

    location_conf = float(skills.confidence_by_skill.get("location", 0.0))
    if include_low_confidence or location_conf >= 0.25:
        location_fields: List[str] = []
        location_value_fields: List[str] = []
        if skills.current_location:
            location_fields.append(f"agent={skills.current_location}")
        if skills.visible_anchor:
            location_fields.append(f"anchor={skills.visible_anchor}")
        if skills.target_locations:
            for goal_object, value in sorted(skills.target_locations.items()):
                progress_state = skills.object_progress.get(goal_object)
                if progress_state == "placed":
                    continue
                if progress_state == "holding" and value != "inventory":
                    continue
                mapped_value = (
                    skills.current_location
                    if value == "inventory"
                    and progress_state == "holding"
                    and skills.current_location
                    else value
                )
                if not str(mapped_value or "").strip():
                    continue
                location_value_fields.append(
                    f"{_encode_skill_entity(goal_object)}={mapped_value}"
                )
        location_fields.extend(location_value_fields)
        if location_fields:
            if skills.visible_anchor or location_value_fields:
                location_fields.append(
                    f"conf={min(1.0, max(location_conf, 1.0 if (skills.current_location and location_value_fields) else location_conf)):.2f}"
                )
            lines.append(f"[SKILL][location] {' '.join(location_fields)}")

    progress_conf = float(skills.confidence_by_skill.get("progress", 0.0))
    progress_tokens: List[str] = []
    if skills.object_progress:
        goal_slots = GoalSlots(
            target_objects=skills.goal_targets,
            target_receptacle=skills.goal_receptacle,
            target_object_count=max(int(skills.goal_target_count or 0), len(skills.goal_targets)),
        )
        goal_progress_subjects = set(_goal_progress_subject_keys(skills.object_progress, goal_slots))
        progress_tokens.extend(
            f"{_encode_skill_entity(goal_object)}={state}"
            for goal_object, state in sorted(skills.object_progress.items())
            if not goal_progress_subjects or goal_object in goal_progress_subjects or state != "missing"
        )
    if skills.searched_without_target:
        progress_tokens.append(
            "searched_without_target="
            + ",".join(_encode_skill_entity(entity) for entity in skills.searched_without_target)
        )
    if skills.search_bridge_entity:
        progress_tokens.append(f"bridge_entity={_encode_search_phrase(skills.search_bridge_entity)}")
    if skills.search_answer_candidate:
        if _normalize_fact_token(skills.search_answer_candidate) != _normalize_fact_token(skills.search_bridge_entity or ""):
            progress_tokens.append(f"answer_candidate={_encode_search_phrase(skills.search_answer_candidate)}")
    if progress_tokens and (include_low_confidence or progress_conf >= 0.25):
        lines.append(f"[SKILL][progress] {' '.join(progress_tokens)}")

    canonical_state_items = [
        (subject, state)
        for subject, state in sorted(skills.state_flags.items())
        if str(state or "").strip() in CANONICAL_STATE_VALUES
    ]
    state_conf = float(skills.confidence_by_skill.get("state", 0.0))
    if canonical_state_items and (include_low_confidence or state_conf >= 0.25):
        state_fields = " ".join(
            f"{_encode_skill_entity(subject)}={state}"
            for subject, state in canonical_state_items
        )
        lines.append(f"[SKILL][state] {state_fields}")

    return lines[:max_lines]


def _compact_context_limits(context_budget_percent: float) -> Tuple[int, int, int]:
    budget_ratio = max(0.05, min(1.0, float(context_budget_percent) / 100.0))
    if budget_ratio > 0.90:
        return 8, 680, 2
    if budget_ratio <= 0.35:
        return 5, 360, 1
    if budget_ratio <= 0.60:
        return 6, 520, 2
    if budget_ratio <= 0.80:
        return 7, 600, 2
    return 7, 560, 2


def _phase_aware_compact_context_limits(
    max_total_lines: int,
    max_total_chars: int,
    max_recent_lines: int,
    *,
    family: Optional[str],
    phase: Optional[str],
) -> Tuple[int, int, int]:
    phase_key = str(phase or "").strip()

    def _bump(lines: int = 0, chars: int = 0, recent: int = 0) -> Tuple[int, int, int]:
        return (
            max_total_lines + max(0, int(lines)),
            max_total_chars + max(0, int(chars)),
            max_recent_lines + max(0, int(recent)),
        )

    if family == "pick_and_place":
        if phase_key in {"search", "locate"}:
            return _bump(lines=1, chars=140, recent=1)
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return _bump(lines=1, chars=120, recent=1)

    if family == "pick_two_obj_and_place":
        if phase_key in {"search", "locate"}:
            return _bump(lines=2, chars=220, recent=1)
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return _bump(lines=2, chars=200, recent=1)

    if family in DESIRED_STATE_BY_FAMILY:
        if phase_key == "state_pending":
            return _bump(lines=2, chars=260, recent=1)
        if phase_key in {"state_ready", "post_state_action"}:
            return _bump(lines=3, chars=320, recent=1)

    return max_total_lines, max_total_chars, max_recent_lines


def _phase_aware_context_budget_percent(
    base_budget_percent: float,
    *,
    family: Optional[str],
    phase: Optional[str],
    raw_line_count: int,
) -> float:
    budget = max(5.0, min(100.0, float(base_budget_percent)))
    phase_key = str(phase or "").strip()
    raw_count = max(0, int(raw_line_count))

    def _scale(factor: float) -> float:
        return min(100.0, max(budget, budget * factor))

    if phase_key in {"search", "locate"}:
        if family == "pick_two_obj_and_place":
            return _scale(1.20)
        if family == "pick_and_place":
            return _scale(1.12)
        if raw_count <= 16:
            return budget
        return _scale(1.08)

    if family == "pick_two_obj_and_place":
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return _scale(1.12)
        return _scale(1.05)

    if family == "pick_and_place":
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return _scale(1.04)
        return _scale(1.08)

    if family in DESIRED_STATE_BY_FAMILY and phase_key == "state_pending":
        return _scale(1.08)

    if family in DESIRED_STATE_BY_FAMILY and phase_key in {"state_ready", "post_state_action"}:
        return _scale(1.12)

    if phase_key in {"carry_place", "partial_place", "post_place"}:
        return max(5.0, budget * 0.90)

    return budget


def _phase_aware_min_compaction_lines(
    base_min_compaction_lines: int,
    *,
    family: Optional[str],
    phase: Optional[str],
) -> int:
    min_lines = max(0, int(base_min_compaction_lines))
    if min_lines <= 0:
        return 0
    phase_key = str(phase or "").strip()

    if phase_key in {"search", "locate"}:
        if family == "pick_two_obj_and_place":
            return max(min_lines, 14)
        if family == "pick_and_place":
            return max(min_lines, 12)
        return max(min_lines, 10)

    if family == "pick_two_obj_and_place":
        if phase_key == "carry_place":
            return max(min_lines, 12)
        if phase_key in {"partial_place", "post_place"}:
            return max(min_lines, 8)
        return max(min_lines, 12)

    if family == "pick_and_place":
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return max(min_lines, 8)
        return max(min_lines, 8)

    if family in DESIRED_STATE_BY_FAMILY and phase_key == "state_pending":
        return max(min_lines, 10)

    if family in DESIRED_STATE_BY_FAMILY and phase_key in {"state_ready", "post_state_action"}:
        return max(min_lines, 12)

    return min_lines


def _compact_context_skill_priority(resolved_mode: TrustContextMode) -> Tuple[str, ...]:
    if resolved_mode == "state":
        return ("state", "progress", "location")
    if resolved_mode == "slot":
        return ("location", "progress", "state")
    return ("location", "progress", "state")


def _compact_context_family_priority(
    resolved_mode: TrustContextMode,
    family: Optional[str],
    phase: Optional[str],
) -> Tuple[str, ...]:
    state_family = family in DESIRED_STATE_BY_FAMILY
    phase_key = str(phase or "").strip()
    if family == "pick_two_obj_and_place":
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return ("progress", "location", "state")
        return ("location", "progress", "state")
    if state_family and phase_key in {"state_ready", "post_state_action"}:
        return ("state", "progress", "location")
    if state_family and phase_key == "state_pending":
        if family == "look_at_obj_in_light":
            return ("state", "location", "progress")
        return ("state", "progress", "location")
    return _compact_context_skill_priority(resolved_mode)


def _compact_context_skill_line_cap(
    max_total_lines: int,
    family: Optional[str],
    phase: Optional[str],
) -> int:
    phase_key = str(phase or "").strip()
    if family == "pick_and_place" and phase_key in {"search", "locate"}:
        return 3
    if family == "pick_and_place" and phase_key in {"carry_place", "partial_place", "post_place"}:
        return 2 if max_total_lines <= 7 else 3
    if phase_key in {"search", "locate"} and max_total_lines <= 5:
        return 3
    if family == "pick_two_obj_and_place":
        if phase_key in {"search", "locate", "carry_place", "partial_place"}:
            return 3
        if max_total_lines <= 5:
            return 2
        return 3
    if family in DESIRED_STATE_BY_FAMILY and phase_key in {"state_ready", "post_state_action"}:
        return 3 if max_total_lines <= 8 else 4
    if max_total_lines <= 5:
        return 3
    if max_total_lines <= 6:
        return 3
    return 4


def _compact_context_skill_line_priority(
    skill_kind: str,
    resolved_mode: TrustContextMode,
    family: Optional[str],
    phase: Optional[str],
) -> int:
    phase_key = str(phase or "").strip()
    if family in DESIRED_STATE_BY_FAMILY:
        if phase_key == "post_state_action":
            priorities = {"goal": 100, "state": 94, "progress": 90, "location": 82}
        elif phase_key == "state_ready":
            priorities = {"goal": 100, "state": 92, "progress": 90, "location": 82}
        elif family == "look_at_obj_in_light":
            priorities = {"goal": 100, "state": 94, "location": 90, "progress": 84}
        else:
            priorities = {"goal": 100, "state": 92, "location": 88, "progress": 84}
        return priorities.get(skill_kind, 0)
    if phase_key in {"search", "locate"}:
        priorities = {"goal": 100, "location": 94, "progress": 88, "state": 78}
        return priorities.get(skill_kind, 0)
    if phase_key in {"carry_place", "partial_place", "post_place"}:
        priorities = {"goal": 100, "progress": 94, "location": 88, "state": 78}
        return priorities.get(skill_kind, 0)
    if resolved_mode == "state":
        priorities = {"goal": 100, "state": 92, "progress": 86, "location": 78}
    elif resolved_mode == "slot":
        priorities = {"goal": 100, "progress": 92, "location": 88, "state": 80}
    else:
        priorities = {"goal": 100, "location": 90, "progress": 82, "state": 74}
    return priorities.get(skill_kind, 0)


def _select_compact_skill_lines(
    skill_lines: Sequence[str],
    *,
    max_total_lines: int,
    resolved_mode: TrustContextMode,
    family: Optional[str],
    phase: Optional[str],
) -> List[str]:
    if not skill_lines:
        return []
    phase_key = str(phase or "").strip()
    filtered_skill_lines: List[str] = []
    for line in skill_lines:
        match = SKILL_LINE_PATTERN.match(line.strip())
        kind = match.group("kind").lower() if match else "generic"
        if (
            kind == "state"
            and family in {"pick_and_place", "pick_two_obj_and_place"}
            and phase_key in {"search", "locate", "carry_place", "partial_place", "post_place"}
        ):
            continue
        filtered_skill_lines.append(line)
    if not filtered_skill_lines:
        return []
    cap = max(1, _compact_context_skill_line_cap(max_total_lines, family, phase))
    ranked = sorted(
        enumerate(filtered_skill_lines),
        key=lambda item: (
            -_compact_context_skill_line_priority(
                (
                    (match.group("kind").lower() if match else "generic")
                    if (match := SKILL_LINE_PATTERN.match(item[1].strip()))
                    else "generic"
                ),
                resolved_mode,
                family,
                phase,
            ),
            item[0],
        ),
    )
    selected = [line for _, line in ranked[:cap]]
    selected_keys = {_normalized_line_key(line) for line in selected}
    return [line for line in filtered_skill_lines if _normalized_line_key(line) in selected_keys]


def _line_has_desired_state_completion(
    line_text: str,
    *,
    family: Optional[str],
    goal_slots: GoalSlots,
) -> bool:
    family_key = str(family or "").strip()
    desired_state = _required_state_for_family(family)
    if not desired_state:
        return False
    state_device_hint = _state_device_hint_for_family(family_key)
    parsed = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    explicit_state = parsed.explicit_state
    if explicit_state is not None:
        object_name, state_name = explicit_state
        if (
            state_name == desired_state
            and (
                not goal_slots.target_objects
                or _goal_object_for_entity(object_name, goal_slots) is not None
            )
        ):
            return True
    for state_triplet in (parsed.known_state, parsed.emergent_state):
        if state_triplet is None:
            continue
        subject, state_name, _ = state_triplet
        subject_matches_completion = (
            not goal_slots.target_objects
            or _goal_object_for_entity(subject, goal_slots) is not None
        )
        if (
            family_key == "look_at_obj_in_light"
            and state_device_hint
            and _entities_match(subject, state_device_hint)
        ):
            subject_matches_completion = True
        if (
            state_name == desired_state
            and subject_matches_completion
        ):
            return True
    return False


def _line_mentions_state_device(
    line_text: str,
    *,
    family: Optional[str],
) -> bool:
    hints = STATE_DEVICE_HINTS_BY_FAMILY.get(str(family or "").strip(), ())
    if not hints:
        return False
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))
    return any(hint in lowered for hint in hints)


def _line_mentions_goal_receptacle(
    line_text: str,
    *,
    goal_slots: GoalSlots,
) -> bool:
    return bool(
        goal_slots.target_receptacle
        and _line_mentions_entity(line_text, goal_slots.target_receptacle)
    )


def _line_supports_multi_object_anchor(
    line_text: str,
    *,
    goal_object: str,
    goal_slots: GoalSlots,
) -> bool:
    if not _line_mentions_entity(line_text, goal_object):
        return False
    parsed = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    if parsed.pickup_object is not None or parsed.placement is not None:
        return True
    if parsed.explicit_state is not None or parsed.known_state is not None or parsed.emergent_state is not None:
        return True
    if parsed.has_fact_signature or parsed.emergent_location is not None or parsed.emergent_progress is not None:
        return True
    if "inventory" in parsed.line_slot_kinds or "progress" in parsed.line_slot_kinds:
        return True
    return False


def _select_multi_object_anchor_lines(
    raw_lines: Sequence[str],
    *,
    goal_slots: GoalSlots,
    phase: Optional[str],
) -> List[str]:
    if _goal_target_count(goal_slots) < 2 or not goal_slots.target_receptacle:
        return []

    cleaned_lines = [str(line).strip() for line in raw_lines if str(line or "").strip()]
    anchors: List[str] = []
    seen_line_keys: Set[str] = set()

    if len(goal_slots.target_objects) == 1:
        goal_object = goal_slots.target_objects[0]
        for line in reversed(cleaned_lines):
            if not _line_supports_multi_object_anchor(
                line,
                goal_object=goal_object,
                goal_slots=goal_slots,
            ):
                continue
            line_key = _normalized_line_key(line)
            if line_key in seen_line_keys:
                continue
            anchors.append(line)
            seen_line_keys.add(line_key)
            if len(anchors) >= min(2, _goal_target_count(goal_slots)):
                break
    else:
        for goal_object in goal_slots.target_objects:
            for line in reversed(cleaned_lines):
                if not _line_supports_multi_object_anchor(
                    line,
                    goal_object=goal_object,
                    goal_slots=goal_slots,
                ):
                    continue
                line_key = _normalized_line_key(line)
                if line_key in seen_line_keys:
                    break
                anchors.append(line)
                seen_line_keys.add(line_key)
                break

    phase_key = str(phase or "").strip()
    if phase_key in {"carry_place", "partial_place", "post_place"}:
        for line in reversed(cleaned_lines):
            if not _line_mentions_goal_receptacle(line, goal_slots=goal_slots):
                continue
            parsed = _analyze_trust_line_cached(
                line,
                goal_slots.target_objects,
                goal_slots.target_receptacle,
            )
            if not (
                parsed.receptacle_state is not None
                or parsed.current_location is not None
                or parsed.visible_anchor is not None
                or parsed.placement is not None
            ):
                continue
            line_key = _normalized_line_key(line)
            if line_key in seen_line_keys:
                break
            anchors.append(line)
            seen_line_keys.add(line_key)
            break

    return anchors[:3]


def _select_single_object_anchor_lines(
    raw_lines: Sequence[str],
    *,
    goal_slots: GoalSlots,
    phase: Optional[str],
) -> List[str]:
    if len(goal_slots.target_objects) != 1 or not goal_slots.target_receptacle:
        return []

    goal_object = goal_slots.target_objects[0]
    cleaned_lines = [str(line).strip() for line in raw_lines if str(line or "").strip()]
    phase_key = str(phase or "").strip()

    goal_progress_line = None
    goal_support_line = None
    receptacle_line = None
    for line in reversed(cleaned_lines):
        parsed = _analyze_trust_line_cached(
            line,
            goal_slots.target_objects,
            goal_slots.target_receptacle,
        )
        if goal_progress_line is None and _line_mentions_entity(line, goal_object):
            if (
                parsed.pickup_object is not None
                or "inventory" in parsed.line_slot_kinds
                or "progress" in parsed.line_slot_kinds
                or parsed.emergent_progress is not None
            ):
                goal_progress_line = line
        if goal_support_line is None and _line_supports_multi_object_anchor(
            line,
            goal_object=goal_object,
            goal_slots=goal_slots,
        ):
            goal_support_line = line
        if receptacle_line is None and _line_mentions_goal_receptacle(line, goal_slots=goal_slots):
            if (
                parsed.receptacle_state is not None
                or parsed.current_location is not None
                or parsed.visible_anchor is not None
                or parsed.placement is not None
            ):
                receptacle_line = line
        if goal_progress_line is not None and goal_support_line is not None and receptacle_line is not None:
            break

    anchors: List[str] = []
    if phase_key in {"carry_place", "partial_place", "post_place"}:
        carry_line_key = _normalized_line_key(goal_progress_line or "")
        has_explicit_inventory_progress = bool(
            goal_progress_line
            and ("inventory" in carry_line_key or "carrying" in carry_line_key)
        )
        for candidate in (
            goal_progress_line,
            goal_support_line if not has_explicit_inventory_progress else None,
            receptacle_line,
        ):
            if candidate is not None and candidate not in anchors:
                anchors.append(candidate)
    else:
        for candidate in (
            goal_support_line,
            goal_progress_line if phase_key == "locate" else None,
            receptacle_line if phase_key == "locate" else None,
        ):
            if candidate is not None and candidate not in anchors:
                anchors.append(candidate)

    return anchors[:3]


def _select_state_task_anchor_lines(
    raw_lines: Sequence[str],
    *,
    family: Optional[str],
    phase: Optional[str],
    goal_slots: GoalSlots,
) -> List[str]:
    family_key = str(family or "").strip()
    if family_key not in DESIRED_STATE_BY_FAMILY:
        return []
    phase_key = str(phase or "").strip()
    desired_state = _required_state_for_family(family_key)
    object_anchor_lines = _select_single_object_anchor_lines(
        raw_lines,
        goal_slots=goal_slots,
        phase=None,
    )
    object_location_anchor = object_anchor_lines[0] if object_anchor_lines else None
    explicit_completion_line = None
    completion_line = None
    post_state_followup_line = None
    carry_line = None
    device_line = None
    receptacle_line = None
    for line in reversed([str(line).strip() for line in raw_lines if str(line or "").strip()]):
        parsed = _analyze_trust_line_cached(
            line,
            goal_slots.target_objects,
            goal_slots.target_receptacle,
        )
        if explicit_completion_line is None and parsed.explicit_state is not None:
            object_name, state_name = parsed.explicit_state
            if (
                state_name == desired_state
                and (
                    not goal_slots.target_objects
                    or _goal_object_for_entity(object_name, goal_slots) is not None
                )
            ):
                explicit_completion_line = line
        if completion_line is None and _line_has_desired_state_completion(
            line,
            family=family_key,
            goal_slots=goal_slots,
        ):
            completion_line = line
        if carry_line is None and any(_line_mentions_entity(line, goal_object) for goal_object in goal_slots.target_objects):
            if (
                parsed.pickup_object is not None
                or "inventory" in parsed.line_slot_kinds
                or "progress" in parsed.line_slot_kinds
                or parsed.emergent_progress is not None
            ):
                carry_line = line
        if post_state_followup_line is None and _line_mentions_state_device(line, family=family_key):
            completion_key = _normalized_line_key(explicit_completion_line or completion_line or "")
            line_key = _normalized_line_key(line)
            if line_key != completion_key:
                post_state_followup_line = line
        if device_line is None and _line_mentions_state_device(line, family=family_key):
            device_line = line
        if receptacle_line is None and _line_mentions_goal_receptacle(line, goal_slots=goal_slots):
            if (
                parsed.receptacle_state is not None
                or parsed.current_location is not None
                or parsed.visible_anchor is not None
                or parsed.placement is not None
            ):
                receptacle_line = line
        if completion_line is not None and device_line is not None and receptacle_line is not None:
            break

    anchors: List[str] = []
    completion_anchor = explicit_completion_line or completion_line
    if phase_key in {"state_ready", "post_state_action"}:
        candidates = (
            carry_line,
            receptacle_line,
            post_state_followup_line,
            completion_anchor,
            device_line,
        )
    elif phase_key == "state_pending":
        candidates = (
            object_location_anchor,
            device_line,
            carry_line,
            receptacle_line,
            completion_anchor,
        )
    else:
        candidates = (
            completion_anchor,
            carry_line,
            device_line,
            receptacle_line,
        )
    for candidate in candidates:
        if candidate is not None and candidate not in anchors:
            anchors.append(candidate)
    return anchors[:4]


def _compact_context_guaranteed_kinds(
    resolved_mode: TrustContextMode,
    family: Optional[str],
    phase: Optional[str],
) -> Tuple[str, ...]:
    phase_key = str(phase or "").strip()
    if family == "pick_two_obj_and_place":
        if phase_key in {"search", "locate"}:
            return ("location", "progress")
        return ("progress", "location")
    if family == "pick_and_place":
        if phase_key in {"search", "locate"}:
            return ("location", "progress")
        if phase_key in {"carry_place", "partial_place", "post_place"}:
            return ("progress", "location")
    if phase_key in {"carry_place", "partial_place", "post_place"}:
        return ("progress",)
    if family in DESIRED_STATE_BY_FAMILY and phase_key in {"state_ready", "post_state_action"}:
        return ("state", "progress")
    if family in DESIRED_STATE_BY_FAMILY and phase_key == "state_pending":
        if family == "look_at_obj_in_light":
            return ("state", "location", "progress")
        return ("state", "location", "progress")
    if resolved_mode == "state":
        return ("state", "progress")
    if resolved_mode == "slot":
        return ("location", "progress")
    return ("location",)


def _skill_line_kind(line_text: str) -> str:
    match = SKILL_LINE_PATTERN.match(str(line_text or "").strip())
    if not match:
        return ""
    return str(match.group("kind") or "").strip().lower()


def _compact_line_char_count(lines: Sequence[str]) -> int:
    total = 0
    for index, line in enumerate(lines):
        text = str(line or "").strip()
        if not text:
            continue
        total += len(text)
        if index > 0:
            total += 1
    return total


def _build_late_state_compact_lines(
    *,
    skill_lines: Sequence[str],
    candidate_source_lines: Sequence[str],
    family: Optional[str],
    phase: Optional[str],
    goal_slots: GoalSlots,
    max_total_lines: int,
) -> List[str]:
    phase_key = str(phase or "").strip()
    family_key = str(family or "").strip()
    if family_key not in DESIRED_STATE_BY_FAMILY:
        return []
    if len(candidate_source_lines) > 5:
        return []
    has_completion_anchor = any(
        (
            _line_has_desired_state_completion(
                line,
                family=family,
                goal_slots=goal_slots,
            )
            or _state_line_score(line, family_key) >= 0.5
        )
        for line in candidate_source_lines
    )
    if not (phase_key in {"state_ready", "post_state_action"} or has_completion_anchor):
        return []

    compact_lines: List[str] = []
    seen_keys: Set[str] = set()

    def add_line(line_text: str) -> bool:
        text = str(line_text or "").strip()
        if not text:
            return False
        line_key = text.lower() if text.startswith("[SKILL]") else _normalized_line_key(text)
        if line_key in seen_keys:
            return False
        next_lines = compact_lines + [text]
        if len(next_lines) > max_total_lines:
            return False
        compact_lines.append(text)
        seen_keys.add(line_key)
        return True

    allowed_skill_kinds = {"goal", "state"}
    if family_key != "look_at_obj_in_light":
        allowed_skill_kinds.add("progress")
    for line in skill_lines:
        if _skill_line_kind(line) not in allowed_skill_kinds:
            continue
        add_line(line)

    for anchor_line in _select_state_task_anchor_lines(
        candidate_source_lines,
        family=family,
        phase=phase,
        goal_slots=goal_slots,
    ):
        if _same_class_distractor(anchor_line, goal_slots):
            continue
        parsed = _analyze_trust_line_cached(
            anchor_line,
            goal_slots.target_objects,
            goal_slots.target_receptacle,
        )
        if (
            parsed.current_location is not None
            and not parsed.mentioned_goal_objects
            and parsed.visible_anchor is None
            and parsed.receptacle_state is None
            and parsed.placement is None
            and parsed.explicit_state is None
            and parsed.known_state is None
            and parsed.emergent_state is None
            and parsed.pickup_object is None
            and parsed.emergent_progress is None
            and not parsed.line_slot_kinds.intersection({"inventory", "progress", "receptacle"})
        ):
            continue
        add_line(anchor_line)

    return compact_lines


def _compact_repair_evict_score(
    line_text: str,
    *,
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
    latest_recent_line_keys: Set[str],
    fact_key_to_line_keys: Dict[str, Set[str]],
    latest_line_key_by_fact: Dict[str, str],
    line_facts_by_key: Dict[str, Set[str]],
) -> int:
    kind = _skill_line_kind(line_text)
    if kind == "goal":
        return 100
    if kind == "state":
        return 80
    if kind == "progress":
        return 25
    if kind == "location":
        return 15
    if _line_has_hard_guard(
        line_text,
        goal_slots=goal_slots,
        skills=skills,
        phase=phase,
        latest_recent_line_keys=set(),
        fact_key_to_line_keys=fact_key_to_line_keys,
        latest_line_key_by_fact=latest_line_key_by_fact,
        line_facts_by_key=line_facts_by_key,
    ):
        return 60
    return 5


def _repair_compact_context_coverage(
    compact_lines: Sequence[str],
    *,
    raw_lines: Sequence[str],
    candidate_lines: Optional[Sequence[str]] = None,
    goal_slots: GoalSlots,
    skills: MemorySkills,
    resolved_mode: TrustContextMode,
    family: Optional[str],
    phase: Optional[str],
    max_total_lines: int,
    max_total_chars: int,
    latest_recent_line_keys: Set[str],
    fact_key_to_line_keys: Dict[str, Set[str]],
    latest_line_key_by_fact: Dict[str, str],
    line_facts_by_key: Dict[str, Set[str]],
) -> List[str]:
    repaired_lines = [str(line).strip() for line in compact_lines if str(line or "").strip()]
    if not repaired_lines:
        return repaired_lines
    repair_source_lines = [str(line).strip() for line in (candidate_lines or raw_lines) if str(line or "").strip()]
    repair_source_keys = {_normalized_line_key(line) for line in repair_source_lines}

    required_candidates: List[str] = []
    required_candidates.extend(
        _select_state_task_anchor_lines(
            repair_source_lines,
            family=family,
            phase=phase,
            goal_slots=goal_slots,
        )
    )
    if family not in DESIRED_STATE_BY_FAMILY:
        required_candidates.extend(
            _select_single_object_anchor_lines(
                repair_source_lines,
                goal_slots=goal_slots,
                phase=phase,
            )
        )
        required_candidates.extend(
            _select_multi_object_anchor_lines(
                repair_source_lines,
                goal_slots=goal_slots,
                phase=phase,
            )
        )
    for kind in _compact_context_guaranteed_kinds(resolved_mode, family, phase):
        for line in skills.witness_lines_by_skill.get(kind, ()):
            if repair_source_keys and _normalized_line_key(line) not in repair_source_keys:
                continue
            if _is_stale_source_location_line_for_late_phase(
                line,
                goal_slots=goal_slots,
                skills=skills,
                phase=phase,
            ):
                continue
            required_candidates.append(line)

    required_lines: List[str] = []
    seen_required: Set[str] = set()
    for candidate in required_candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        if _normalized_line_key(text) == "inventory":
            continue
        if _is_stale_source_location_line_for_late_phase(
            text,
            goal_slots=goal_slots,
            skills=skills,
            phase=phase,
        ):
            continue
        key = _normalized_line_key(text)
        if key in seen_required:
            continue
        seen_required.add(key)
        required_lines.append(text)

    current_keys = {_normalized_line_key(line) for line in repaired_lines}
    protected_required_keys: Set[str] = set()
    for line in repaired_lines:
        line_key = _normalized_line_key(line)
        skill_kind = _skill_line_kind(line)
        if skill_kind == "goal":
            protected_required_keys.add(line_key)
            continue
        if family in DESIRED_STATE_BY_FAMILY and skill_kind == "state":
            protected_required_keys.add(line_key)
            continue
        if (
            family in DESIRED_STATE_BY_FAMILY
            and skill_kind == "location"
            and _line_mentions_state_device(line, family=family)
        ):
            protected_required_keys.add(line_key)

    for required_line in required_lines:
        required_key = _normalized_line_key(required_line)
        if required_key in current_keys:
            protected_required_keys.add(required_key)
            continue

        while repaired_lines:
            projected_line_count = len(repaired_lines) + 1
            projected_char_count = _compact_line_char_count(repaired_lines + [required_line])
            if projected_line_count <= max_total_lines and projected_char_count <= max_total_chars:
                break

            victim_idx = None
            victim_score = None
            for idx in range(len(repaired_lines) - 1, -1, -1):
                candidate_line = repaired_lines[idx]
                candidate_key = _normalized_line_key(candidate_line)
                if candidate_key in protected_required_keys:
                    continue
                score = _compact_repair_evict_score(
                    candidate_line,
                    goal_slots=goal_slots,
                    skills=skills,
                    phase=phase,
                    latest_recent_line_keys=latest_recent_line_keys,
                    fact_key_to_line_keys=fact_key_to_line_keys,
                    latest_line_key_by_fact=latest_line_key_by_fact,
                    line_facts_by_key=line_facts_by_key,
                )
                if victim_score is None or score < victim_score:
                    victim_idx = idx
                    victim_score = score
            if victim_idx is None:
                break
            removed_line = repaired_lines.pop(victim_idx)
            current_keys.discard(_normalized_line_key(removed_line))

        projected_line_count = len(repaired_lines) + 1
        projected_char_count = _compact_line_char_count(repaired_lines + [required_line])
        if projected_line_count <= max_total_lines and projected_char_count <= max_total_chars:
            repaired_lines.append(required_line)
            current_keys.add(required_key)
            protected_required_keys.add(required_key)

    return repaired_lines


def _prune_redundant_late_phase_lines(
    compact_lines: Sequence[str],
    *,
    goal_slots: GoalSlots,
    phase: Optional[str],
) -> List[str]:
    pruned_lines = [str(line).strip() for line in compact_lines if str(line or "").strip()]
    phase_key = str(phase or "").strip()
    if phase_key not in {"carry_place", "partial_place", "post_place", "state_ready", "post_state_action"}:
        return pruned_lines
    max_keep_lines = 6 if phase_key in {"state_ready", "post_state_action"} else 5
    if len(pruned_lines) <= max_keep_lines:
        return pruned_lines

    def _parsed(line_text: str) -> ParsedTrustLine:
        return _analyze_trust_line_cached(
            line_text,
            goal_slots.target_objects,
            goal_slots.target_receptacle,
        )

    has_receptacle_state_line = any(
        (
            (parsed.receptacle_state is not None and _target_receptacle_matches(parsed.receptacle_state[0], goal_slots))
            or (
                parsed.placement is not None
                and _target_receptacle_matches(parsed.placement[1], goal_slots)
            )
        )
        for parsed in (_parsed(line) for line in pruned_lines if not line.startswith("[SKILL]"))
    )
    if has_receptacle_state_line:
        for idx, line in enumerate(pruned_lines):
            if line.startswith("[SKILL]"):
                continue
            parsed = _parsed(line)
            if (
                parsed.current_location is not None
                and _target_receptacle_matches(parsed.current_location, goal_slots)
                and (
                    parsed.visible_anchor is None
                    or _entities_match(parsed.visible_anchor, parsed.current_location)
                )
                and parsed.receptacle_state is None
                and parsed.placement is None
                and not parsed.has_fact_signature
                and not parsed.mentioned_goal_objects
            ):
                pruned_lines.pop(idx)
                break

    has_explicit_carry_line = any(
        (
            not line.startswith("[SKILL]")
            and any(_line_mentions_entity(line, goal_object) for goal_object in goal_slots.target_objects)
            and ("inventory" in _normalized_line_key(line) or "you are carrying" in _normalized_line_key(line))
        )
        for line in pruned_lines
    )
    if has_explicit_carry_line:
        for idx, line in enumerate(pruned_lines):
            if line.startswith("[SKILL]"):
                continue
            parsed = _parsed(line)
            if parsed.pickup_object is None:
                continue
            goal_object = _goal_object_for_entity(parsed.pickup_object, goal_slots)
            if goal_object is None:
                continue
            normalized_line = _normalized_line_key(line)
            if "inventory" in normalized_line or "you are carrying" in normalized_line:
                continue
            pruned_lines.pop(idx)
            break

    return pruned_lines


def _recent_line_incremental_score(
    line_text: str,
    *,
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
) -> float:
    text = str(line_text or "").strip()
    if not text or text.startswith("[SKILL]"):
        return -1e6

    lowered = _normalize_fact_token(_observation_evidence_text(text))
    parsed_line = _analyze_trust_line_cached(
        text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    line_slot_kinds = set(parsed_line.line_slot_kinds)
    mentions_goal_object = bool(parsed_line.mentioned_goal_objects)
    mentions_receptacle = bool(
        goal_slots.target_receptacle and _line_mentions_entity(text, goal_slots.target_receptacle)
    )

    score = 0.25
    if parsed_line.placement is not None:
        score += 5.0
    if parsed_line.explicit_state is not None:
        score += 4.5
    if parsed_line.known_state is not None or parsed_line.emergent_state is not None:
        score += 4.0
    if parsed_line.pickup_object is not None:
        score += 3.5
    if parsed_line.receptacle_state is not None:
        score += 3.25
    if mentions_goal_object:
        score += 3.0
    if mentions_receptacle:
        score += 2.5
    if parsed_line.has_fact_signature:
        score += 2.25
    if line_slot_kinds.intersection({"inventory", "progress"}) or parsed_line.emergent_progress is not None:
        score += 2.0
    if parsed_line.visible_anchor is not None:
        score += 1.5
    if parsed_line.current_location is not None or "location" in line_slot_kinds:
        score += 0.75

    if parsed_line.has_fact_signature and not mentions_goal_object and not mentions_receptacle:
        signature = _extract_fact_signature(text)
        if signature is not None:
            subject, value = signature
            normalized_subject = _goal_progress_subject_for_entity(subject, goal_slots) or _clean_slot_entity(subject)
            normalized_value = _canonical_summary_fact_value(value)
            goal_objects = {
                _clean_slot_entity(obj)
                for obj in goal_slots.target_objects
                if _clean_slot_entity(obj)
            }
            goal_receptacle = _clean_slot_entity(goal_slots.target_receptacle or "")
            if (
                normalized_subject not in goal_objects
                and not (goal_receptacle and (normalized_subject == goal_receptacle or normalized_value == goal_receptacle))
            ):
                score -= 2.5

    if "nothing happens" in lowered:
        score -= 3.0
        if not (
            mentions_goal_object
            or mentions_receptacle
            or parsed_line.explicit_state is not None
            or parsed_line.known_state is not None
            or parsed_line.emergent_state is not None
            or parsed_line.receptacle_state is not None
            or parsed_line.has_fact_signature
            or line_slot_kinds.intersection({"inventory", "progress"})
        ):
            score -= 1.5

    if (
        parsed_line.current_location is not None
        and not mentions_goal_object
        and not mentions_receptacle
        and parsed_line.visible_anchor is None
        and parsed_line.has_fact_signature is False
        and parsed_line.receptacle_state is None
        and parsed_line.explicit_state is None
        and parsed_line.known_state is None
        and parsed_line.emergent_state is None
        and not line_slot_kinds.intersection({"inventory", "progress"})
    ):
        score -= 1.75

    if (
        parsed_line.current_location is not None
        and skills.current_location
        and _entities_match(parsed_line.current_location, skills.current_location)
        and not mentions_goal_object
        and not mentions_receptacle
    ):
        score -= 0.75

    if parsed_line.receptacle_state is not None:
        receptacle, is_open = parsed_line.receptacle_state
        state_name = "open" if is_open else "closed"
        if skills.state_flags.get(receptacle) == state_name and not mentions_goal_object:
            score -= 0.5

    phase_key = str(phase or "").strip()
    if phase_key in {"search", "locate"} and not (
        mentions_goal_object
        or mentions_receptacle
        or parsed_line.has_fact_signature
        or line_slot_kinds.intersection({"inventory", "progress"})
        or parsed_line.receptacle_state is not None
    ):
        score -= 0.5

    return score


def _rank_recent_compact_lines(
    candidate_recent_lines: Sequence[str],
    *,
    max_recent_lines: int,
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
    preferred_line_keys: Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    if max_recent_lines <= 0 or not candidate_recent_lines:
        return []

    recent_window = max(4, max_recent_lines * 4)
    scoped_recent_lines = list(candidate_recent_lines[-recent_window:])
    scored_recent_lines = [
        (
            line,
            _recent_line_incremental_score(
                line,
                goal_slots=goal_slots,
                skills=skills,
                phase=phase,
            )
            + (
                0.35
                if preferred_line_keys and _normalized_line_key(line) in preferred_line_keys
                else 0.0
            ),
            idx,
        )
        for idx, line in enumerate(scoped_recent_lines)
    ]
    scored_recent_lines.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [(line, score) for line, score, _ in scored_recent_lines]


def _summary_dedup_phase_active(phase: Optional[str]) -> bool:
    return str(phase or "").strip() in {
        "carry_place",
        "partial_place",
        "post_place",
        "state_ready",
        "post_state_action",
    }


def _is_redundant_inventory_recent_line(
    line_text: str,
    *,
    compact_lines: Sequence[str],
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
) -> bool:
    phase_key = str(phase or "").strip()
    if phase_key not in {"carry_place", "partial_place", "post_place", "post_state_action"}:
        return False

    parsed_line = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))
    if not (
        "inventory" in parsed_line.line_slot_kinds
        or "progress" in parsed_line.line_slot_kinds
        or "you are carrying" in lowered
    ):
        return False
    if (
        parsed_line.current_location is not None
        or parsed_line.visible_anchor is not None
        or parsed_line.receptacle_state is not None
        or parsed_line.placement is not None
        or parsed_line.explicit_state is not None
        or parsed_line.known_state is not None
        or parsed_line.emergent_state is not None
        or parsed_line.has_fact_signature
        or parsed_line.emergent_location is not None
    ):
        return False

    if not any(state in {"holding", "placed"} for state in skills.object_progress.values()):
        return False

    compact_text = "\n".join(str(line) for line in compact_lines)
    if "[SKILL][progress]" in compact_text:
        return True
    for goal_object in goal_slots.target_objects:
        if goal_object and goal_object in compact_text and any(
            marker in compact_text.lower() for marker in ("pick up", "carrying", "holding")
        ):
            return True
    return False


def _canonical_summary_fact_value(value: Optional[str]) -> str:
    normalized = _clean_slot_entity(str(value or ""))
    if not normalized:
        return ""
    for prefix in ("in:", "on:", "at:"):
        if normalized.startswith(prefix):
            normalized = normalized.split(":", 1)[1].strip()
            break
    return normalized


def _summary_fact_keys_from_text(summary_text: str) -> Set[str]:
    summary_keys: Set[str] = set()
    if not summary_text:
        return summary_keys

    for raw_line in summary_text.splitlines():
        line = str(raw_line or "").strip()
        if not line.startswith("- "):
            continue
        body = line[2:].strip()
        if body.startswith("required state:"):
            value = _clean_slot_entity(body.split(":", 1)[1])
            if value:
                summary_keys.add(f"required_state|goal|{value}")
            continue
        if body.startswith("state device:"):
            value = _canonical_summary_fact_value(body.split(":", 1)[1])
            if value:
                summary_keys.add(f"state_device|goal|{value}")
            continue
        if body.startswith("current location:"):
            value = _canonical_summary_fact_value(body.split(":", 1)[1])
            if value:
                summary_keys.add(f"location|agent|{value}")
            continue
        if body.startswith("visible anchor:"):
            value = _canonical_summary_fact_value(body.split(":", 1)[1])
            if value:
                summary_keys.add(f"location|anchor|{value}")
            continue
        if body.startswith("holding:"):
            for token in body.split(":", 1)[1].split(","):
                obj = _clean_slot_entity(token)
                if obj:
                    summary_keys.add(f"progress|holding|{obj}")
            continue
        if body.startswith("placed:"):
            quantity_match = re.search(r"(\d+)\s*/\s*(\d+)", body)
            if quantity_match:
                summary_keys.add(
                    f"progress|placed_count|{quantity_match.group(1)}/{quantity_match.group(2)}"
                )
            continue
        if body.startswith("known locations:"):
            for token in body.split(":", 1)[1].split(","):
                if "->" not in token:
                    continue
                subject, value = token.split("->", 1)
                subject_name = _clean_slot_entity(subject)
                location_value = _canonical_summary_fact_value(value)
                if subject_name and location_value:
                    summary_keys.add(f"location|{subject_name}|{location_value}")
            continue
        if body.startswith("already searched without target:"):
            for token in body.split(":", 1)[1].split(","):
                location = _clean_slot_entity(token)
                if location:
                    summary_keys.add(f"progress|searched_without_target|{location}")
            continue
        if body.startswith("updated state:"):
            for token in body.split(":", 1)[1].split(","):
                if "->" not in token:
                    continue
                subject, state_name = token.split("->", 1)
                normalized_subject = _clean_slot_entity(subject)
                normalized_state = _clean_slot_entity(state_name)
                if normalized_subject and normalized_state:
                    summary_keys.add(f"state|{normalized_subject}|{normalized_state}")
    return summary_keys


def _line_summary_fact_keys(
    line_text: str,
    *,
    kind: Optional[str],
    goal_slots: GoalSlots,
) -> Set[str]:
    parsed = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    fact_keys: Set[str] = set()

    if kind in {None, "location"}:
        if parsed.current_location:
            fact_keys.add(f"location|agent|{_canonical_summary_fact_value(parsed.current_location)}")
        if parsed.visible_anchor:
            fact_keys.add(f"location|anchor|{_canonical_summary_fact_value(parsed.visible_anchor)}")
        if parsed.emergent_location is not None:
            subject, _, value, _ = parsed.emergent_location
            normalized_subject = _goal_progress_subject_for_entity(subject, goal_slots) or _clean_slot_entity(subject)
            normalized_value = _canonical_summary_fact_value(value)
            if normalized_subject and normalized_value:
                fact_keys.add(f"location|{normalized_subject}|{normalized_value}")
        signature = _extract_fact_signature(line_text) if parsed.has_fact_signature else None
        if signature is not None:
            subject, value = signature
            normalized_subject = _goal_progress_subject_for_entity(subject, goal_slots) or _clean_slot_entity(subject)
            normalized_value = _canonical_summary_fact_value(value)
            if normalized_subject and normalized_value:
                fact_keys.add(f"location|{normalized_subject}|{normalized_value}")

    if kind in {None, "progress"}:
        searched_location = _extract_searched_without_target_location(
            line_text,
            goal_slots,
            parsed_line=parsed,
        )
        if searched_location:
            fact_keys.add(
                f"progress|searched_without_target|{_canonical_summary_fact_value(searched_location)}"
            )
        if parsed.pickup_object:
            normalized_subject = _goal_progress_subject_for_entity(parsed.pickup_object, goal_slots) or _clean_slot_entity(parsed.pickup_object)
            if normalized_subject:
                fact_keys.add(f"progress|holding|{normalized_subject}")
        if parsed.placement is not None:
            placed_entity, _ = parsed.placement
            normalized_subject = _goal_progress_subject_for_entity(placed_entity, goal_slots) or _clean_slot_entity(placed_entity)
            if normalized_subject:
                fact_keys.add(f"progress|placed|{normalized_subject}")
        if parsed.emergent_progress is not None:
            progress_object, progress_value, _ = parsed.emergent_progress
            normalized_subject = _goal_progress_subject_for_entity(progress_object, goal_slots) or _clean_slot_entity(progress_object)
            normalized_value = _clean_slot_entity(progress_value)
            if normalized_subject and normalized_value:
                fact_keys.add(f"progress|{normalized_value}|{normalized_subject}")

    if kind in {None, "state"}:
        if parsed.receptacle_state is not None:
            receptacle, is_open = parsed.receptacle_state
            normalized_subject = _clean_slot_entity(receptacle)
            if normalized_subject:
                fact_keys.add(f"state|{normalized_subject}|{'open' if is_open else 'closed'}")
        if parsed.explicit_state is not None:
            object_name, state_name = parsed.explicit_state
            normalized_subject = _goal_object_for_entity(object_name, goal_slots) or _clean_slot_entity(object_name)
            normalized_state = _clean_slot_entity(state_name)
            if normalized_subject and normalized_state:
                fact_keys.add(f"state|{normalized_subject}|{normalized_state}")

    return {key for key in fact_keys if key and not key.endswith("|")}


def _holds_or_placed_goal_object(
    skills: MemorySkills,
    goal_slots: GoalSlots,
) -> bool:
    return _goal_progress_state_count(skills.object_progress, goal_slots, "holding", "placed") > 0


def _has_explicit_holding_witness(
    skills: MemorySkills,
    goal_slots: GoalSlots,
) -> bool:
    for line in skills.witness_lines_by_skill.get("progress", ()):
        normalized = _normalized_line_key(line)
        if "inventory" not in normalized and "you are carrying" not in normalized:
            continue
        if not goal_slots.target_objects:
            return True
        if any(_line_mentions_entity(line, goal_object) for goal_object in goal_slots.target_objects):
            return True
    return False


def _is_stale_source_location_line_for_late_phase(
    line_text: str,
    *,
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
) -> bool:
    if not _summary_dedup_phase_active(phase):
        return False
    has_explicit_holding = _has_explicit_holding_witness(skills, goal_slots)
    has_placed_object = _goal_progress_state_count(skills.object_progress, goal_slots, "placed") > 0
    if not has_explicit_holding and not has_placed_object:
        return False

    parsed = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    signature = _extract_fact_signature(line_text) if parsed.has_fact_signature else None
    if goal_slots.target_receptacle and _line_mentions_entity(line_text, goal_slots.target_receptacle):
        return False
    if parsed.current_location and not parsed.mentioned_goal_objects and signature is None:
        return False

    if parsed.emergent_location is not None:
        subject, _, _, _ = parsed.emergent_location
        progress_state = _goal_progress_state_for_entity(subject, skills.object_progress, goal_slots)
        if progress_state == "placed" or (progress_state is not None and has_explicit_holding):
            return True
    if signature is not None:
        subject, _ = signature
        progress_state = _goal_progress_state_for_entity(subject, skills.object_progress, goal_slots)
        if progress_state == "placed" or (progress_state is not None and has_explicit_holding):
            return True
    return False


def _is_stale_fact_key_for_phase(
    fact_key: str,
    *,
    skills: MemorySkills,
    phase: Optional[str],
) -> bool:
    if not _summary_dedup_phase_active(phase):
        return False
    parts = [token.strip() for token in str(fact_key).split("|")]
    if len(parts) != 3:
        return False
    kind, subject, value = parts
    progress_state = skills.object_progress.get(subject)
    if kind == "location" and progress_state in {"holding", "placed"}:
        return True
    if kind == "progress" and value == "holding" and progress_state == "placed":
        return True
    return False


def _build_line_fact_index(
    raw_lines: Sequence[str],
    *,
    goal_slots: GoalSlots,
) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, Set[str]]]:
    fact_key_to_line_keys: Dict[str, Set[str]] = {}
    latest_line_key_by_fact: Dict[str, str] = {}
    line_facts_by_key: Dict[str, Set[str]] = {}
    for line in raw_lines:
        normalized_line_key = _normalized_line_key(line)
        fact_keys = _line_summary_fact_keys(line, kind=None, goal_slots=goal_slots)
        line_facts_by_key[normalized_line_key] = set(fact_keys)
        for fact_key in fact_keys:
            fact_key_to_line_keys.setdefault(fact_key, set()).add(normalized_line_key)
            latest_line_key_by_fact[fact_key] = normalized_line_key
    return fact_key_to_line_keys, latest_line_key_by_fact, line_facts_by_key


def _fact_key_is_goal_relevant(
    fact_key: str,
    *,
    goal_slots: GoalSlots,
) -> bool:
    parts = [token.strip() for token in str(fact_key).split("|")]
    if len(parts) != 3:
        return False

    kind, subject, value = parts
    goal_objects = {
        _clean_slot_entity(obj)
        for obj in goal_slots.target_objects
        if _clean_slot_entity(obj)
    }
    goal_receptacle = _clean_slot_entity(goal_slots.target_receptacle or "")

    if kind == "progress":
        if value == "searched_without_target":
            return True
        return subject in goal_objects

    if kind == "location":
        if subject in goal_objects:
            return True
        if goal_receptacle and (subject == goal_receptacle or value == goal_receptacle):
            return True
        if subject in {"agent", "anchor"} and goal_receptacle and value == goal_receptacle:
            return True
        return False

    if kind == "state":
        if subject in goal_objects:
            return True
        if goal_receptacle and subject == goal_receptacle:
            return True
        return False

    return False


def _recent_line_deserves_hard_guard(
    line_text: str,
    *,
    goal_slots: GoalSlots,
) -> bool:
    parsed = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    line_slot_kinds = set(parsed.line_slot_kinds)
    if parsed.placement is not None or parsed.pickup_object is not None:
        return True
    if (
        parsed.explicit_state is not None
        or parsed.known_state is not None
        or parsed.emergent_state is not None
        or parsed.receptacle_state is not None
    ):
        return True
    if parsed.emergent_location is not None or parsed.emergent_progress is not None:
        return True
    if parsed.mentioned_goal_objects:
        return True
    if goal_slots.target_receptacle and _line_mentions_entity(line_text, goal_slots.target_receptacle):
        return True
    if line_slot_kinds.intersection({"inventory", "progress", "receptacle"}):
        return True
    return False


def _line_has_hard_guard(
    line_text: str,
    *,
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
    latest_recent_line_keys: Set[str],
    fact_key_to_line_keys: Dict[str, Set[str]],
    latest_line_key_by_fact: Dict[str, str],
    line_facts_by_key: Dict[str, Set[str]],
) -> bool:
    line_key = _normalized_line_key(line_text)
    if line_key in latest_recent_line_keys:
        return True

    fact_keys = line_facts_by_key.get(line_key, set())
    for fact_key in fact_keys:
        line_keys = fact_key_to_line_keys.get(fact_key, set())
        if _is_stale_fact_key_for_phase(fact_key, skills=skills, phase=phase):
            fact_parts = [token.strip() for token in str(fact_key).split("|")]
            preserve_unique_goal_location = (
                len(line_keys) <= 1
                and len(fact_parts) == 3
                and fact_parts[0] == "location"
                and _clean_slot_entity(fact_parts[1])
                in {
                    _clean_slot_entity(obj)
                    for obj in goal_slots.target_objects
                    if _clean_slot_entity(obj)
                }
            )
            if not preserve_unique_goal_location:
                continue
        if not _fact_key_is_goal_relevant(fact_key, goal_slots=goal_slots):
            continue
        if len(line_keys) <= 1:
            return True
        if latest_line_key_by_fact.get(fact_key) == line_key:
            return True
        if _summary_dedup_phase_active(phase) and fact_key.startswith("progress|searched_without_target|"):
            return True
    return False


def _should_skip_summary_duplicate_raw_line(
    line_text: str,
    *,
    kind: Optional[str],
    goal_slots: GoalSlots,
    skills: MemorySkills,
    phase: Optional[str],
    summary_fact_keys: Set[str],
    latest_recent_line_keys: Set[str],
    fact_key_to_line_keys: Dict[str, Set[str]],
    latest_line_key_by_fact: Dict[str, str],
    line_facts_by_key: Dict[str, Set[str]],
) -> bool:
    if not summary_fact_keys:
        return False
    if kind == "location":
        return False
    if _line_has_hard_guard(
        line_text,
        goal_slots=goal_slots,
        skills=skills,
        phase=phase,
        latest_recent_line_keys=latest_recent_line_keys,
        fact_key_to_line_keys=fact_key_to_line_keys,
        latest_line_key_by_fact=latest_line_key_by_fact,
        line_facts_by_key=line_facts_by_key,
    ):
        return False
    line_fact_keys = _line_summary_fact_keys(line_text, kind=kind, goal_slots=goal_slots)
    if kind is None:
        line_fact_keys = {key for key in line_fact_keys if not key.startswith("location|")}
    return bool(line_fact_keys & summary_fact_keys)


def prepare_trust_context(
    query_text: str,
    raw_lines: Sequence[str],
    *,
    requested_mode: Optional[TrustContextMode] = None,
    state_aware: bool = False,
    feedback: Optional[MemorySkillFeedback] = None,
    adaptive_witness_budget: bool = True,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
) -> Optional[PreparedTrustContext]:
    cleaned_raw_lines = tuple(line for line in raw_lines if line and str(line).strip())
    if not query_text or not cleaned_raw_lines:
        return None

    framework = resolve_memory_framework_config(memory_framework_config)
    resolved_mode = infer_query_context_mode(
        query_text=query_text,
        lines=cleaned_raw_lines,
        requested_mode=requested_mode,
        state_aware=state_aware,
    )
    feedback_family = infer_monitor_family(query_text)
    feedback_phase = infer_monitor_phase(query_text, cleaned_raw_lines)
    goal_slots = parse_goal_slots(query_text)
    skills = compile_memory_skills_from_lines(
        cleaned_raw_lines,
        query_text,
        feedback=feedback,
        adaptive_witness_budget=adaptive_witness_budget,
        memory_framework_config=framework,
    )
    return PreparedTrustContext(
        query_text=query_text,
        raw_lines=cleaned_raw_lines,
        resolved_mode=resolved_mode,
        feedback_family=feedback_family,
        feedback_phase=feedback_phase,
        goal_slots=goal_slots,
        skills=skills,
    )


def build_compact_trust_context(
    query_text: str,
    raw_lines: Sequence[str],
    *,
    kept_lines: Optional[Sequence[str]] = None,
    requested_mode: Optional[TrustContextMode] = None,
    state_aware: bool = False,
    feedback: Optional[MemorySkillFeedback] = None,
    context_budget_percent: float = 50.0,
    prompt_summary_active: bool = False,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
    prepared_context: Optional[PreparedTrustContext] = None,
) -> str:
    cleaned_raw_lines = [line for line in raw_lines if line and str(line).strip()]
    if not query_text or not cleaned_raw_lines:
        fallback_lines = [line for line in (kept_lines or raw_lines) if line and str(line).strip()]
        return "\n".join(fallback_lines)

    cleaned_kept_lines = [line for line in (kept_lines or ()) if line and str(line).strip()]
    kept_line_keys = {_normalized_line_key(line) for line in cleaned_kept_lines}
    generic_search_query = _is_open_domain_search_query(query_text)
    analysis_raw_lines = (
        list(_coalesce_open_domain_search_information_lines(cleaned_raw_lines))
        if generic_search_query
        else cleaned_raw_lines
    )

    framework = resolve_memory_framework_config(memory_framework_config)
    prepared = prepared_context
    if prepared is not None and tuple(cleaned_raw_lines) != prepared.raw_lines:
        prepared = None
    if prepared is None:
        prepared = prepare_trust_context(
            query_text=query_text,
            raw_lines=cleaned_raw_lines,
            requested_mode=requested_mode,
            state_aware=state_aware,
            feedback=feedback,
            memory_framework_config=framework,
        )
    resolved_mode = prepared.resolved_mode if prepared is not None else infer_query_context_mode(
        query_text=query_text,
        lines=cleaned_raw_lines,
        requested_mode=requested_mode,
        state_aware=state_aware,
    )
    skills = prepared.skills if prepared is not None else compile_memory_skills_from_lines(
        cleaned_raw_lines,
        query_text,
        feedback=feedback,
        memory_framework_config=framework,
    )
    feedback_family = prepared.feedback_family if prepared is not None else infer_monitor_family(query_text)
    feedback_phase = prepared.feedback_phase if prepared is not None else infer_monitor_phase(query_text, cleaned_raw_lines)
    goal_slots = prepared.goal_slots if prepared is not None else parse_goal_slots(query_text)
    summary_text = ""
    summary_fact_keys: Set[str] = set()
    if prompt_summary_active:
        summary_text = build_trust_policy_text_summary(
            query_text=query_text,
            raw_lines=cleaned_raw_lines,
            requested_mode=requested_mode,
            state_aware=state_aware,
            feedback=feedback,
            memory_framework_config=framework,
            prepared_context=prepared,
        )
        summary_fact_keys = _summary_fact_keys_from_text(summary_text)
    latest_recent_line_keys = {
        _normalized_line_key(line)
        for line in cleaned_raw_lines[-max(2, min(3, len(cleaned_raw_lines))):]
        if _recent_line_deserves_hard_guard(
            line,
            goal_slots=goal_slots,
        )
    }
    fact_key_to_line_keys, latest_line_key_by_fact, line_facts_by_key = _build_line_fact_index(
        cleaned_raw_lines,
        goal_slots=goal_slots,
    )

    def _allow_compact_candidate(line_text: str) -> bool:
        if not kept_line_keys:
            return True
        line_key = _normalized_line_key(line_text)
        if line_key in kept_line_keys:
            return True
        return _line_has_hard_guard(
            line_text,
            goal_slots=goal_slots,
            skills=skills,
            phase=feedback_phase,
            latest_recent_line_keys=latest_recent_line_keys,
            fact_key_to_line_keys=fact_key_to_line_keys,
            latest_line_key_by_fact=latest_line_key_by_fact,
            line_facts_by_key=line_facts_by_key,
        )

    candidate_source_lines = [
        line for line in cleaned_raw_lines
        if _allow_compact_candidate(line)
    ]
    if not candidate_source_lines:
        candidate_source_lines = cleaned_kept_lines if cleaned_kept_lines else cleaned_raw_lines
    latest_search_information_line = ""
    latest_search_information_source_line = ""
    latest_search_query_line = ""
    search_compact_source_lines = list(candidate_source_lines)
    search_summary_skip_line_keys: Set[str] = set()
    search_unresolved_history = bool(
        generic_search_query
        and not any(_extract_information_payload(line) for line in analysis_raw_lines)
    )
    search_task_card_only = bool(
        search_unresolved_history
        and any(str(line or "").strip().startswith("[SEARCH_") for line in analysis_raw_lines)
        and not any(_extract_search_query(line) for line in analysis_raw_lines)
    )
    if generic_search_query:
        latest_search_information_source_line = _latest_search_information_line(analysis_raw_lines)
        latest_search_query_line = _latest_search_query_line(analysis_raw_lines)
        if latest_search_information_source_line:
            latest_info_key = _normalized_line_key(latest_search_information_source_line)
            candidate_info_keys = {
                _normalized_line_key(line)
                for line in _coalesce_open_domain_search_information_lines(search_compact_source_lines)
            }
            if latest_info_key not in candidate_info_keys:
                search_compact_source_lines.append(latest_search_information_source_line)

    compact_skills = skills
    if tuple(search_compact_source_lines) != tuple(cleaned_raw_lines):
        compact_skills = compile_memory_skills_from_lines(
            search_compact_source_lines,
            query_text,
            feedback=feedback,
            memory_framework_config=framework,
        )
    if generic_search_query and latest_search_information_source_line:
        latest_search_information_line = latest_search_information_source_line
        latest_search_query_text = query_text or _extract_search_query(latest_search_query_line) or ""
        summarized_information_line = _summarize_search_information_line(
            latest_search_query_text,
            latest_search_information_source_line,
            compact_skills,
        )
        if summarized_information_line:
            latest_search_information_line = summarized_information_line
            search_summary_skip_line_keys.add(_normalized_line_key(latest_search_information_source_line))

    skill_lines = synthesize_skill_lines(
        compact_skills,
        family=feedback_family,
        query_text=query_text,
        memory_framework_config=framework,
    )
    if search_unresolved_history:
        task_card_source_lines = (
            cleaned_raw_lines
            if search_task_card_only
            else build_search_task_card_context(query_text).splitlines()
        )
        relation_query = bool(_search_task_same_relation_components(query_text)[0])
        task_card_lines: List[str] = []
        seen_task_card_lines: Set[str] = set()

        def _append_task_card_line(line_text: str) -> None:
            text = str(line_text or "").strip()
            if not text or text in seen_task_card_lines:
                return
            seen_task_card_lines.add(text)
            task_card_lines.append(text)

        for line in task_card_source_lines:
            text = str(line or "").strip()
            if text == "[QUESTION]" or text.startswith("[SKILL]["):
                continue
            if not text.startswith("[SEARCH_"):
                if _normalize_fact_token(text) == _normalize_fact_token(query_text):
                    continue
                _append_task_card_line(text)
                continue
            if text.startswith("[SEARCH_ANCHOR]") and compact_skills.search_anchor:
                _append_task_card_line(text)
                continue
            if text.startswith("[SEARCH_COMPARE]") and compact_skills.search_compare_hint:
                _append_task_card_line(text)
                continue
            if text.startswith("[SEARCH_BRIDGE]") and compact_skills.search_bridge_hint:
                if relation_query and compact_skills.search_scope_hint:
                    continue
                _append_task_card_line(text)
                continue
            if text.startswith("[SEARCH_TARGET]") and compact_skills.search_target_hint and compact_skills.search_target_hint in text:
                _append_task_card_line(text)
                continue
            if text.startswith("[SEARCH_SCOPE]") and compact_skills.search_scope_hint:
                if relation_query or compact_skills.search_compare_hint:
                    _append_task_card_line(text)
                    continue
                _append_task_card_line(text)
                continue
        max_task_card_lines = 4 if (compact_skills.search_compare_hint or compact_skills.search_bridge_hint) else 3
        return "\n".join(task_card_lines[:max_task_card_lines]) if task_card_lines else "\n".join(task_card_source_lines[:max_task_card_lines])

    effective_budget_percent = _phase_aware_context_budget_percent(
        context_budget_percent,
        family=feedback_family,
        phase=feedback_phase,
        raw_line_count=len(cleaned_raw_lines),
    )
    max_total_lines, max_total_chars, max_recent_lines = _compact_context_limits(effective_budget_percent)
    max_total_lines, max_total_chars, max_recent_lines = _phase_aware_compact_context_limits(
        max_total_lines,
        max_total_chars,
        max_recent_lines,
        family=feedback_family,
        phase=feedback_phase,
    )
    if (
        feedback_family in DESIRED_STATE_BY_FAMILY
        and str(feedback_phase or "").strip() in {"state_ready", "post_state_action"}
    ):
        max_recent_lines = 0
    skill_lines = _select_compact_skill_lines(
        skill_lines,
        max_total_lines=max_total_lines,
        resolved_mode=resolved_mode,
        family=feedback_family,
        phase=feedback_phase,
    )
    late_state_compact_lines = _build_late_state_compact_lines(
        skill_lines=skill_lines,
        candidate_source_lines=candidate_source_lines,
        family=feedback_family,
        phase=feedback_phase,
        goal_slots=goal_slots,
        max_total_lines=min(max_total_lines, 6),
    )
    if late_state_compact_lines:
        return "\n".join(late_state_compact_lines)
    compact_lines: List[str] = []
    seen_keys: Set[str] = set()
    char_count = 0

    def add_line(line_text: str) -> bool:
        nonlocal char_count
        text = str(line_text or "").strip()
        if not text:
            return False
        line_key = text.lower() if text.startswith("[SKILL]") else _normalized_line_key(text)
        if line_key in seen_keys:
            return False
        next_char_count = char_count + len(text) + (1 if compact_lines else 0)
        if compact_lines and (len(compact_lines) >= max_total_lines or next_char_count > max_total_chars):
            return False
        compact_lines.append(text)
        seen_keys.add(line_key)
        char_count = next_char_count
        return True

    if latest_search_information_line:
        add_line(latest_search_information_line)
    for line in skill_lines:
        add_line(line)

    witness_budget = max(1, max_total_lines - len(compact_lines) - max_recent_lines)
    witness_added = 0
    for anchor_line in _select_state_task_anchor_lines(
        candidate_source_lines,
        family=feedback_family,
        phase=feedback_phase,
        goal_slots=goal_slots,
    ):
        if witness_added >= witness_budget:
            break
        if add_line(anchor_line):
            witness_added += 1
    for anchor_line in _select_single_object_anchor_lines(
        candidate_source_lines,
        goal_slots=goal_slots,
        phase=feedback_phase,
    ):
        if witness_added >= witness_budget:
            break
        if add_line(anchor_line):
            witness_added += 1
    for anchor_line in _select_multi_object_anchor_lines(
        candidate_source_lines,
        goal_slots=goal_slots,
        phase=feedback_phase,
    ):
        if witness_added >= witness_budget:
            break
        if add_line(anchor_line):
            witness_added += 1
    witness_priority = _compact_context_family_priority(
        resolved_mode,
        feedback_family,
        feedback_phase,
    )
    candidate_witnesses = {
        kind: [
            line
            for line in skills.witness_lines_by_skill.get(kind, ())
            if _normalized_line_key(line) not in search_summary_skip_line_keys
            if _allow_compact_candidate(line)
            and not (
                _is_stale_source_location_line_for_late_phase(
                    line,
                    goal_slots=goal_slots,
                    skills=skills,
                    phase=feedback_phase,
                )
            )
            and not (
                prompt_summary_active
                and _summary_dedup_phase_active(feedback_phase)
                and _should_skip_summary_duplicate_raw_line(
                    line,
                    kind=kind,
                    goal_slots=goal_slots,
                    skills=skills,
                    phase=feedback_phase,
                    summary_fact_keys=summary_fact_keys,
                    latest_recent_line_keys=latest_recent_line_keys,
                    fact_key_to_line_keys=fact_key_to_line_keys,
                    latest_line_key_by_fact=latest_line_key_by_fact,
                    line_facts_by_key=line_facts_by_key,
                )
            )
        ]
        for kind in ("location", "progress", "state")
    }
    witness_offsets = {kind: 0 for kind in candidate_witnesses}
    allow_witness_backfill = not (
        prompt_summary_active and _summary_dedup_phase_active(feedback_phase)
    )
    if (
        feedback_family in DESIRED_STATE_BY_FAMILY
        and str(feedback_phase or "").strip() in {"state_ready", "post_state_action"}
    ):
        allow_witness_backfill = False

    for kind in _compact_context_guaranteed_kinds(resolved_mode, feedback_family, feedback_phase):
        if witness_added >= witness_budget:
            break
        witness_lines = candidate_witnesses.get(kind, ())
        while witness_offsets[kind] < len(witness_lines):
            witness_line = witness_lines[witness_offsets[kind]]
            witness_offsets[kind] += 1
            if add_line(witness_line):
                witness_added += 1
                break

    if allow_witness_backfill:
        made_progress = True
        while witness_added < witness_budget and made_progress:
            made_progress = False
            for kind in witness_priority:
                if witness_added >= witness_budget:
                    break
                witness_lines = candidate_witnesses.get(kind, ())
                while witness_offsets[kind] < len(witness_lines):
                    witness_line = witness_lines[witness_offsets[kind]]
                    witness_offsets[kind] += 1
                    if add_line(witness_line):
                        witness_added += 1
                        made_progress = True
                        break

    recent_added = 0
    candidate_recent_lines = [
        line
        for line in candidate_source_lines
        if _normalized_line_key(line) not in search_summary_skip_line_keys
    ]
    for recent_line, recent_score in _rank_recent_compact_lines(
        candidate_recent_lines,
        max_recent_lines=max_recent_lines,
        goal_slots=goal_slots,
        skills=skills,
        phase=feedback_phase,
        preferred_line_keys=kept_line_keys,
    ):
        if recent_added >= max_recent_lines:
            break
        if recent_score <= 0.0:
            continue
        if (
            _is_stale_source_location_line_for_late_phase(
            recent_line,
            goal_slots=goal_slots,
            skills=skills,
            phase=feedback_phase,
            )
        ):
            continue
        if (
            prompt_summary_active
            and _summary_dedup_phase_active(feedback_phase)
            and _should_skip_summary_duplicate_raw_line(
                recent_line,
                kind=None,
                goal_slots=goal_slots,
                skills=skills,
                phase=feedback_phase,
                summary_fact_keys=summary_fact_keys,
                latest_recent_line_keys=latest_recent_line_keys,
                fact_key_to_line_keys=fact_key_to_line_keys,
                latest_line_key_by_fact=latest_line_key_by_fact,
                line_facts_by_key=line_facts_by_key,
            )
        ):
            continue
        if _is_redundant_inventory_recent_line(
            recent_line,
            compact_lines=compact_lines,
            goal_slots=goal_slots,
            skills=skills,
            phase=feedback_phase,
        ):
            continue
        if add_line(recent_line):
            recent_added += 1

    compact_lines = _repair_compact_context_coverage(
        compact_lines,
        raw_lines=cleaned_raw_lines,
        candidate_lines=candidate_recent_lines,
        goal_slots=goal_slots,
        skills=skills,
        resolved_mode=resolved_mode,
        family=feedback_family,
        phase=feedback_phase,
        max_total_lines=max_total_lines,
        max_total_chars=max_total_chars,
        latest_recent_line_keys=latest_recent_line_keys,
        fact_key_to_line_keys=fact_key_to_line_keys,
        latest_line_key_by_fact=latest_line_key_by_fact,
        line_facts_by_key=line_facts_by_key,
    )
    compact_lines = _prune_redundant_late_phase_lines(
        compact_lines,
        goal_slots=goal_slots,
        phase=feedback_phase,
    )

    if compact_lines:
        return "\n".join(compact_lines)
    fallback_lines = cleaned_kept_lines if cleaned_kept_lines else cleaned_raw_lines
    return "\n".join(fallback_lines)


def _format_memory_phase_hint(phase: str) -> Optional[str]:
    phase_map = {
        "state_pending": "required state change is still pending",
        "state_ready": "required state change is complete",
        "post_state_action": "required state change is complete; continue with placement or follow-up action",
        "search": "still searching for task-relevant evidence",
        "locate": "task-relevant object or location evidence has been found",
        "carry_place": "the target object is already being carried",
        "partial_place": "part of the placement progress is already completed",
        "post_place": "the placement goal is already completed",
    }
    return phase_map.get(str(phase or "").strip())


def build_trust_policy_text_summary(
    query_text: str,
    raw_lines: Sequence[str],
    *,
    requested_mode: Optional[TrustContextMode] = None,
    state_aware: bool = False,
    feedback: Optional[MemorySkillFeedback] = None,
    max_lines: int = 5,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
    prepared_context: Optional[PreparedTrustContext] = None,
) -> str:
    if not query_text:
        return ""
    cleaned_lines = [line for line in raw_lines if line and str(line).strip()]
    if not cleaned_lines:
        return ""

    framework = resolve_memory_framework_config(memory_framework_config)
    prepared = prepared_context
    if prepared is not None and tuple(cleaned_lines) != prepared.raw_lines:
        prepared = None
    if prepared is None:
        prepared = prepare_trust_context(
            query_text=query_text,
            raw_lines=cleaned_lines,
            requested_mode=requested_mode,
            state_aware=state_aware,
            feedback=feedback,
            memory_framework_config=framework,
        )
    goal_slots = prepared.goal_slots if prepared is not None else parse_goal_slots(query_text)
    skills = prepared.skills if prepared is not None else compile_memory_skills_from_lines(
        cleaned_lines,
        query_text,
        feedback=feedback,
        memory_framework_config=framework,
    )

    summary_lines: List[str] = []
    family = prepared.feedback_family if prepared is not None else infer_monitor_family(query_text)
    location_conf = float(skills.confidence_by_skill.get("location", 0.0))
    progress_conf = float(skills.confidence_by_skill.get("progress", 0.0))
    state_conf = float(skills.confidence_by_skill.get("state", 0.0))

    required_state = _required_state_for_family(family)
    if required_state:
        summary_lines.append(f"- required state: {required_state}")

    # Keep this block strictly factual. High-level phase inferences can
    # over-assert progress and destabilize action selection.
    goal_progress_subjects = _goal_progress_subject_keys(skills.object_progress, goal_slots)
    total_targets = max(_goal_target_count(goal_slots), int(skills.goal_target_count or 0))
    placed_count = _goal_progress_state_count(skills.object_progress, goal_slots, "placed")
    holding_objects = sorted(
        goal_object
        for goal_object in goal_progress_subjects
        if skills.object_progress.get(goal_object) == "holding"
    )
    if _goal_uses_instance_tracking(goal_slots) and placed_count > 0 and progress_conf >= 0.60:
        goal_label = next(iter(goal_slots.target_objects), "target object")
        summary_lines.append(f"- placed: {placed_count}/{max(placed_count, total_targets)} {goal_label}")
    if holding_objects and progress_conf >= 0.60:
        summary_lines.append(f"- holding: {', '.join(holding_objects)}")

    known_locations: List[str] = []
    for goal_object, location_value in sorted(skills.target_locations.items()):
        if _goal_object_for_entity(goal_object, goal_slots) is None:
            continue
        progress_state = _goal_progress_state_for_entity(goal_object, skills.object_progress, goal_slots)
        if location_value and progress_state not in {"holding", "placed"}:
            known_locations.append(f"{goal_object} -> {location_value}")
    if known_locations and location_conf >= 0.60:
        summary_lines.append(f"- known locations: {', '.join(known_locations[:2])}")

    if skills.searched_without_target and progress_conf >= 0.55:
        summary_lines.append(
            "- already searched without target: "
            + ", ".join(skills.searched_without_target[:3])
        )

    state_device = _state_device_hint_for_family(family, query_text=query_text)
    if state_device and not any(
        _line_has_desired_state_completion(line, family=family, goal_slots=goal_slots)
        for line in cleaned_lines
    ):
        summary_lines.append(f"- state device: {state_device}")

    state_updates: List[str] = []
    relevant_state_subjects = list(goal_slots.target_objects)
    if goal_slots.target_receptacle:
        relevant_state_subjects.append(goal_slots.target_receptacle)
    if skills.current_location:
        relevant_state_subjects.append(skills.current_location)
    if skills.visible_anchor:
        relevant_state_subjects.append(skills.visible_anchor)

    for subject, state_name in sorted(skills.state_flags.items()):
        if not state_name:
            continue
        display_subject = str(subject)
        normalized_query = _normalize_fact_token(query_text)
        normalized_subject = _normalize_fact_token(subject)
        if normalized_subject and f"some {normalized_subject}" in normalized_query:
            display_subject = f"some {normalized_subject}"
        elif normalized_subject and f"any {normalized_subject}" in normalized_query:
            display_subject = f"any {normalized_subject}"
        normalized_subject = _normalize_fact_token(subject)
        if (
            not normalized_subject
            or normalized_subject.startswith(("you ", "observation ", "action "))
            or any(
                marker in normalized_subject
                for marker in ("you arrive", "go to", "observation", "action")
            )
        ):
            continue
        if relevant_state_subjects and not any(
            _entities_match(subject, candidate) for candidate in relevant_state_subjects if candidate
        ):
            continue
        state_updates.append(f"{display_subject} -> {state_name}")
    if state_updates and state_conf >= 0.60:
        summary_lines.append(f"- updated state: {', '.join(state_updates[:2])}")

    if not summary_lines:
        return ""

    clipped = summary_lines[: max(1, int(max_lines))]
    return "Recent memory facts:\n" + "\n".join(clipped)


def update_memory_skill_feedback(
    feedback: Optional[MemorySkillFeedback],
    *,
    failed_skill_kinds: Sequence[str],
) -> MemorySkillFeedback:
    updated = MemorySkillFeedback(
        failure_counts=dict(feedback.failure_counts) if feedback is not None else {},
        fallback_bias_by_skill=dict(feedback.fallback_bias_by_skill) if feedback is not None else {},
        recent_failure_ema_by_skill=dict(feedback.recent_failure_ema_by_skill) if feedback is not None else {},
        reliability_by_skill=dict(feedback.reliability_by_skill) if feedback is not None else {},
        support_by_skill=dict(feedback.support_by_skill) if feedback is not None else {},
        conflict_by_skill=dict(feedback.conflict_by_skill) if feedback is not None else {},
        rescue_budget_by_skill=dict(feedback.rescue_budget_by_skill) if feedback is not None else {},
        utility_ema_by_skill=dict(feedback.utility_ema_by_skill) if feedback is not None else {},
        evidence_alpha_by_skill=dict(feedback.evidence_alpha_by_skill) if feedback is not None else {},
        evidence_beta_by_skill=dict(feedback.evidence_beta_by_skill) if feedback is not None else {},
        uncertainty_by_skill=dict(feedback.uncertainty_by_skill) if feedback is not None else {},
        witness_utility_by_line=dict(feedback.witness_utility_by_line) if feedback is not None else {},
        witness_role_support_by_skill={
            kind: dict(role_values)
            for kind, role_values in (feedback.witness_role_support_by_skill.items() if feedback is not None else [])
        },
        witness_role_conflict_by_skill={
            kind: dict(role_values)
            for kind, role_values in (feedback.witness_role_conflict_by_skill.items() if feedback is not None else [])
        },
        witness_role_utility_by_skill={
            kind: dict(role_values)
            for kind, role_values in (feedback.witness_role_utility_by_skill.items() if feedback is not None else [])
        },
    )
    for kind in failed_skill_kinds:
        updated.failure_counts[kind] = updated.failure_counts.get(kind, 0) + 1
        # The outer loop can turn repeated failures into a stronger bias
        # for preserving or surfacing this skill in future prompts.
        updated.fallback_bias_by_skill[kind] = min(
            3.0,
            updated.fallback_bias_by_skill.get(kind, 0.0) + 1.0,
        )
    return updated


def decay_memory_skill_feedback(
    feedback: Optional[MemorySkillFeedback],
    *,
    decay: float = 0.25,
) -> Optional[MemorySkillFeedback]:
    if feedback is None:
        return None

    updated = MemorySkillFeedback(
        failure_counts=dict(feedback.failure_counts),
        fallback_bias_by_skill=dict(feedback.fallback_bias_by_skill),
        recent_failure_ema_by_skill=dict(feedback.recent_failure_ema_by_skill),
        reliability_by_skill=dict(feedback.reliability_by_skill),
        support_by_skill=dict(feedback.support_by_skill),
        conflict_by_skill=dict(feedback.conflict_by_skill),
        rescue_budget_by_skill=dict(feedback.rescue_budget_by_skill),
        utility_ema_by_skill=dict(feedback.utility_ema_by_skill),
        evidence_alpha_by_skill=dict(feedback.evidence_alpha_by_skill),
        evidence_beta_by_skill=dict(feedback.evidence_beta_by_skill),
        uncertainty_by_skill=dict(feedback.uncertainty_by_skill),
        witness_utility_by_line=dict(feedback.witness_utility_by_line),
        witness_role_support_by_skill={
            kind: dict(role_values) for kind, role_values in feedback.witness_role_support_by_skill.items()
        },
        witness_role_conflict_by_skill={
            kind: dict(role_values) for kind, role_values in feedback.witness_role_conflict_by_skill.items()
        },
        witness_role_utility_by_skill={
            kind: dict(role_values) for kind, role_values in feedback.witness_role_utility_by_skill.items()
        },
    )
    for kind, bias in list(updated.fallback_bias_by_skill.items()):
        next_bias = max(0.0, float(bias) - float(decay))
        if next_bias <= 1e-8:
            updated.fallback_bias_by_skill.pop(kind, None)
        else:
            updated.fallback_bias_by_skill[kind] = next_bias

    evidence_decay = max(0.0, min(1.0, 1.0 - 0.5 * float(decay)))
    tracked_uncertainty_kinds = (
        set(updated.evidence_alpha_by_skill)
        | set(updated.evidence_beta_by_skill)
        | set(updated.uncertainty_by_skill)
    )
    for kind in tracked_uncertainty_kinds:
        next_alpha = float(updated.evidence_alpha_by_skill.get(kind, 0.0)) * evidence_decay
        next_beta = float(updated.evidence_beta_by_skill.get(kind, 0.0)) * evidence_decay
        if next_alpha <= 1e-8:
            updated.evidence_alpha_by_skill.pop(kind, None)
        else:
            updated.evidence_alpha_by_skill[kind] = next_alpha
        if next_beta <= 1e-8:
            updated.evidence_beta_by_skill.pop(kind, None)
        else:
            updated.evidence_beta_by_skill[kind] = next_beta
        next_uncertainty = _component_beta_uncertainty(1.0 + next_alpha, 1.0 + next_beta)
        if next_uncertainty <= 1e-8:
            updated.uncertainty_by_skill.pop(kind, None)
        else:
            updated.uncertainty_by_skill[kind] = next_uncertainty

    return updated


def auto_update_memory_skill_feedback(
    feedback: Optional[MemorySkillFeedback],
    *,
    observed_skill_kinds: Sequence[str],
    failed_skill_kinds: Sequence[str],
    support_by_skill: Optional[Dict[str, float]] = None,
    conflict_by_skill: Optional[Dict[str, float]] = None,
    utility_by_skill: Optional[Dict[str, float]] = None,
    witness_utility_by_line: Optional[Dict[str, float]] = None,
    witness_role_support_by_skill: Optional[Dict[str, Dict[str, float]]] = None,
    witness_role_conflict_by_skill: Optional[Dict[str, Dict[str, float]]] = None,
    witness_role_utility_by_skill: Optional[Dict[str, Dict[str, float]]] = None,
    ema_momentum: float = 0.2,
    adaptive_momentum: bool = True,
    adaptive_rescue: bool = True,
    decay_min: float = 0.05,
    decay_max: float = 0.35,
    bias_gain: float = 1.0,
    bias_max: float = 3.0,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
) -> MemorySkillFeedback:
    framework = resolve_memory_framework_config(memory_framework_config)
    if not framework.enable_fmr:
        return update_memory_skill_feedback(
            feedback,
            failed_skill_kinds=failed_skill_kinds,
        )
    updated = MemorySkillFeedback(
        failure_counts=dict(feedback.failure_counts) if feedback is not None else {},
        fallback_bias_by_skill=dict(feedback.fallback_bias_by_skill) if feedback is not None else {},
        recent_failure_ema_by_skill=dict(feedback.recent_failure_ema_by_skill) if feedback is not None else {},
        reliability_by_skill=dict(feedback.reliability_by_skill) if feedback is not None else {},
        support_by_skill=dict(feedback.support_by_skill) if feedback is not None else {},
        conflict_by_skill=dict(feedback.conflict_by_skill) if feedback is not None else {},
        rescue_budget_by_skill=dict(feedback.rescue_budget_by_skill) if feedback is not None else {},
        utility_ema_by_skill=dict(feedback.utility_ema_by_skill) if feedback is not None else {},
        evidence_alpha_by_skill=dict(feedback.evidence_alpha_by_skill) if feedback is not None else {},
        evidence_beta_by_skill=dict(feedback.evidence_beta_by_skill) if feedback is not None else {},
        uncertainty_by_skill=dict(feedback.uncertainty_by_skill) if feedback is not None else {},
        witness_utility_by_line=dict(feedback.witness_utility_by_line) if feedback is not None else {},
        witness_role_support_by_skill={
            kind: dict(role_values)
            for kind, role_values in (feedback.witness_role_support_by_skill.items() if feedback is not None else [])
        },
        witness_role_conflict_by_skill={
            kind: dict(role_values)
            for kind, role_values in (feedback.witness_role_conflict_by_skill.items() if feedback is not None else [])
        },
        witness_role_utility_by_skill={
            kind: dict(role_values)
            for kind, role_values in (feedback.witness_role_utility_by_skill.items() if feedback is not None else [])
        },
    )

    observed_set = {kind for kind in observed_skill_kinds if kind}
    failed_set = {kind for kind in failed_skill_kinds if kind}
    support_input = support_by_skill or {}
    conflict_input = conflict_by_skill or {}
    utility_input = utility_by_skill or {}
    witness_utility_input = {
        _normalized_line_key(line_text): _clamp_signed_unit(signal)
        for line_text, signal in (witness_utility_by_line or {}).items()
        if line_text and str(line_text).strip()
    }
    witness_role_support_input = {
        kind: {
            str(role): min(1.0, max(0.0, float(signal)))
            for role, signal in role_values.items()
            if role and str(role).strip()
        }
        for kind, role_values in (witness_role_support_by_skill or {}).items()
        if kind and isinstance(role_values, dict)
    }
    witness_role_conflict_input = {
        kind: {
            str(role): min(1.0, max(0.0, float(signal)))
            for role, signal in role_values.items()
            if role and str(role).strip()
        }
        for kind, role_values in (witness_role_conflict_by_skill or {}).items()
        if kind and isinstance(role_values, dict)
    }
    witness_role_utility_input = {
        kind: {
            str(role): _clamp_signed_unit(signal)
            for role, signal in role_values.items()
            if role and str(role).strip()
        }
        for kind, role_values in (witness_role_utility_by_skill or {}).items()
        if kind and isinstance(role_values, dict)
    }
    tracked_kinds = (
        set(updated.fallback_bias_by_skill)
        | set(updated.recent_failure_ema_by_skill)
        | set(updated.reliability_by_skill)
        | set(updated.support_by_skill)
        | set(updated.conflict_by_skill)
        | set(updated.rescue_budget_by_skill)
        | set(updated.utility_ema_by_skill)
        | set(updated.evidence_alpha_by_skill)
        | set(updated.evidence_beta_by_skill)
        | set(updated.uncertainty_by_skill)
        | set(support_input)
        | set(conflict_input)
        | set(utility_input)
        | observed_set
        | failed_set
    )

    for kind in tracked_kinds:
        previous_ema = float(updated.recent_failure_ema_by_skill.get(kind, 0.0))
        failure_signal = 1.0 if kind in failed_set else 0.0
        failure_momentum = (
            _adaptive_ema_momentum(ema_momentum, previous_ema, failure_signal)
            if adaptive_momentum
            else float(ema_momentum)
        )
        next_ema = (1.0 - failure_momentum) * previous_ema + failure_momentum * failure_signal
        if next_ema <= 1e-8:
            updated.recent_failure_ema_by_skill.pop(kind, None)
        else:
            updated.recent_failure_ema_by_skill[kind] = next_ema

        previous_reliability = float(updated.reliability_by_skill.get(kind, 0.0))
        success_signal = 1.0 if kind in observed_set and kind not in failed_set else 0.0
        reliability_momentum = (
            _adaptive_ema_momentum(ema_momentum, previous_reliability, success_signal)
            if adaptive_momentum
            else float(ema_momentum)
        )
        next_reliability = (1.0 - reliability_momentum) * previous_reliability + reliability_momentum * success_signal
        if next_reliability <= 1e-8:
            updated.reliability_by_skill.pop(kind, None)
        else:
            updated.reliability_by_skill[kind] = next_reliability

        previous_support = float(updated.support_by_skill.get(kind, 0.0))
        support_signal = min(1.0, float(support_input.get(kind, 0.0)))
        support_momentum = (
            _adaptive_ema_momentum(ema_momentum, previous_support, support_signal)
            if adaptive_momentum
            else float(ema_momentum)
        )
        next_support = (1.0 - support_momentum) * previous_support + support_momentum * support_signal
        if next_support <= 1e-8:
            updated.support_by_skill.pop(kind, None)
        else:
            updated.support_by_skill[kind] = next_support

        previous_conflict = float(updated.conflict_by_skill.get(kind, 0.0))
        conflict_signal = min(1.0, float(conflict_input.get(kind, 0.0)))
        conflict_momentum = (
            _adaptive_ema_momentum(ema_momentum, previous_conflict, conflict_signal)
            if adaptive_momentum
            else float(ema_momentum)
        )
        next_conflict = (1.0 - conflict_momentum) * previous_conflict + conflict_momentum * conflict_signal
        if next_conflict <= 1e-8:
            updated.conflict_by_skill.pop(kind, None)
        else:
            updated.conflict_by_skill[kind] = next_conflict

        previous_utility = float(updated.utility_ema_by_skill.get(kind, 0.0))
        utility_signal = _clamp_signed_unit(utility_input.get(kind, 0.0))
        utility_momentum = (
            _adaptive_ema_momentum(ema_momentum, previous_utility, utility_signal)
            if adaptive_momentum
            else float(ema_momentum)
        )
        next_utility = (1.0 - utility_momentum) * previous_utility + utility_momentum * utility_signal
        if abs(next_utility) <= 1e-8:
            updated.utility_ema_by_skill.pop(kind, None)
        else:
            updated.utility_ema_by_skill[kind] = next_utility

        previous_alpha = float(updated.evidence_alpha_by_skill.get(kind, 0.0))
        previous_beta = float(updated.evidence_beta_by_skill.get(kind, 0.0))
        next_alpha, next_beta, next_uncertainty = _component_update_uncertainty_state(
            previous_alpha=previous_alpha,
            previous_beta=previous_beta,
            success_signal=success_signal,
            failure_signal=failure_signal,
            support_signal=support_signal,
            conflict_signal=conflict_signal,
            utility_signal=utility_signal,
            next_reliability=next_reliability,
            next_conflict=next_conflict,
        )
        if next_alpha <= 1e-8:
            updated.evidence_alpha_by_skill.pop(kind, None)
        else:
            updated.evidence_alpha_by_skill[kind] = next_alpha
        if next_beta <= 1e-8:
            updated.evidence_beta_by_skill.pop(kind, None)
        else:
            updated.evidence_beta_by_skill[kind] = next_beta
        if next_uncertainty <= 1e-8:
            updated.uncertainty_by_skill.pop(kind, None)
        else:
            updated.uncertainty_by_skill[kind] = next_uncertainty

        current_bias = float(updated.fallback_bias_by_skill.get(kind, 0.0))
        auto_decay = (
            float(decay_min)
            + (1.0 - next_ema) * float(decay_max - decay_min) * min(1.5, max(0.35, 1.0 - 0.65 * next_utility))
        )
        next_bias = max(0.0, current_bias - auto_decay)

        if kind in failed_set:
            updated.failure_counts[kind] = updated.failure_counts.get(kind, 0) + 1
            next_bias = min(float(bias_max), next_bias + float(bias_gain))

        if next_bias <= 1e-8:
            updated.fallback_bias_by_skill.pop(kind, None)
        else:
            updated.fallback_bias_by_skill[kind] = next_bias

        rescue_budget = (
            _adaptive_rescue_budget(
                support=next_support,
                conflict=next_conflict,
                reliability=next_reliability,
                utility=next_utility,
                bias=next_bias,
                failure_ema=next_ema,
            )
            if adaptive_rescue
            else min(
                2.5,
                max(
                    0.0,
                    0.6 * next_support
                    + 1.1 * next_conflict
                    + 0.8 * (1.0 - next_reliability)
                    + 0.8 * next_utility
                    + 0.2 * next_bias,
                ),
            )
        )
        if rescue_budget <= 1e-8:
            updated.rescue_budget_by_skill.pop(kind, None)
        else:
            updated.rescue_budget_by_skill[kind] = rescue_budget

    tracked_witness_lines = set(updated.witness_utility_by_line) | set(witness_utility_input)
    for witness_key in tracked_witness_lines:
        previous_witness_utility = float(updated.witness_utility_by_line.get(witness_key, 0.0))
        witness_signal = float(witness_utility_input.get(witness_key, 0.0))
        witness_momentum = (
            _adaptive_ema_momentum(ema_momentum, previous_witness_utility, witness_signal)
            if adaptive_momentum
            else float(ema_momentum)
        )
        next_witness_utility = (1.0 - witness_momentum) * previous_witness_utility + witness_momentum * witness_signal
        if abs(next_witness_utility) <= 1e-8:
            updated.witness_utility_by_line.pop(witness_key, None)
        else:
            updated.witness_utility_by_line[witness_key] = next_witness_utility

    tracked_role_kinds = (
        set(updated.witness_role_support_by_skill)
        | set(updated.witness_role_conflict_by_skill)
        | set(updated.witness_role_utility_by_skill)
        | set(witness_role_support_input)
        | set(witness_role_conflict_input)
        | set(witness_role_utility_input)
    )
    for kind in tracked_role_kinds:
        previous_support_mapping = updated.witness_role_support_by_skill.get(kind, {})
        previous_conflict_mapping = updated.witness_role_conflict_by_skill.get(kind, {})
        previous_role_mapping = updated.witness_role_utility_by_skill.get(kind, {})
        next_support_mapping: Dict[str, float] = {}
        next_conflict_mapping: Dict[str, float] = {}
        next_role_mapping: Dict[str, float] = {}
        tracked_roles = (
            set(previous_support_mapping)
            | set(previous_conflict_mapping)
            | set(previous_role_mapping)
            | set(witness_role_support_input.get(kind, {}))
            | set(witness_role_conflict_input.get(kind, {}))
            | set(witness_role_utility_input.get(kind, {}))
        )
        for role in tracked_roles:
            previous_role_support = float(previous_support_mapping.get(role, 0.0))
            role_support_signal = float(witness_role_support_input.get(kind, {}).get(role, 0.0))
            role_support_momentum = (
                _adaptive_ema_momentum(ema_momentum, previous_role_support, role_support_signal)
                if adaptive_momentum
                else float(ema_momentum)
            )
            next_role_support = (1.0 - role_support_momentum) * previous_role_support + role_support_momentum * role_support_signal
            if next_role_support > 1e-8:
                next_support_mapping[role] = next_role_support

            previous_role_conflict = float(previous_conflict_mapping.get(role, 0.0))
            role_conflict_signal = float(witness_role_conflict_input.get(kind, {}).get(role, 0.0))
            role_conflict_momentum = (
                _adaptive_ema_momentum(ema_momentum, previous_role_conflict, role_conflict_signal)
                if adaptive_momentum
                else float(ema_momentum)
            )
            next_role_conflict = (1.0 - role_conflict_momentum) * previous_role_conflict + role_conflict_momentum * role_conflict_signal
            if next_role_conflict > 1e-8:
                next_conflict_mapping[role] = next_role_conflict

            previous_role_utility = float(previous_role_mapping.get(role, 0.0))
            role_signal = float(witness_role_utility_input.get(kind, {}).get(role, 0.0))
            role_utility_momentum = (
                _adaptive_ema_momentum(ema_momentum, previous_role_utility, role_signal)
                if adaptive_momentum
                else float(ema_momentum)
            )
            next_role_utility = (1.0 - role_utility_momentum) * previous_role_utility + role_utility_momentum * role_signal
            if abs(next_role_utility) > 1e-8:
                next_role_mapping[role] = next_role_utility
        if next_support_mapping:
            updated.witness_role_support_by_skill[kind] = next_support_mapping
        else:
            updated.witness_role_support_by_skill.pop(kind, None)
        if next_conflict_mapping:
            updated.witness_role_conflict_by_skill[kind] = next_conflict_mapping
        else:
            updated.witness_role_conflict_by_skill.pop(kind, None)
        if next_role_mapping:
            updated.witness_role_utility_by_skill[kind] = next_role_mapping
        else:
            updated.witness_role_utility_by_skill.pop(kind, None)

    return updated


def infer_monitor_family(query_text: str) -> str:
    if _is_open_domain_search_query(query_text):
        return "generic"
    state_family = _infer_state_task_family(query_text)
    if state_family is not None:
        return state_family

    goal_slots = parse_goal_slots(query_text)
    if _goal_target_count(goal_slots) >= 2 and goal_slots.target_receptacle:
        return "pick_two_obj_and_place"
    if len(goal_slots.target_objects) == 1 and goal_slots.target_receptacle:
        return "pick_and_place"

    lowered = _normalize_fact_token(query_text)
    if any(marker in lowered for marker in ("holding", "inventory", "carry", "carrying", "hold")):
        return "inventory_sensitive"
    return "generic"


def _infer_monitor_phase_from_skills(query_text: str, skills: MemorySkills) -> str:
    state_family = _infer_state_task_family(query_text)
    if state_family is not None:
        desired_state = DESIRED_STATE_BY_FAMILY.get(state_family)
        observed_states = set(skills.state_flags.values())
        if desired_state and desired_state in observed_states:
            if state_family != "look_at_obj_in_light" and _has_observed_progress(skills.object_progress):
                return "post_state_action"
            return "state_ready"
        if observed_states:
            return "state_pending"
        if _has_observed_progress(skills.object_progress):
            if state_family == "look_at_obj_in_light":
                return "state_pending"
            return "post_state_action"
        return "state_pending"

    goal_slots = GoalSlots(
        target_objects=skills.goal_targets,
        target_receptacle=skills.goal_receptacle,
        target_object_count=max(int(skills.goal_target_count or 0), len(skills.goal_targets)),
    )
    total_targets = _goal_target_count(goal_slots)
    placed_count = _goal_progress_state_count(skills.object_progress, goal_slots, "placed")
    holding_count = _goal_progress_state_count(skills.object_progress, goal_slots, "holding")
    located_count = _goal_progress_state_count(skills.object_progress, goal_slots, "located")
    if total_targets > 0 and placed_count >= total_targets:
        return "post_place"
    if placed_count > 0:
        return "partial_place"
    if holding_count > 0:
        return "carry_place"
    if located_count > 0 or skills.current_location or skills.target_locations or skills.visible_anchor:
        return "locate"
    return "search"


def infer_monitor_phase(query_text: str, lines: Sequence[str]) -> str:
    if _is_open_domain_search_query(query_text):
        search_lines = _coalesce_open_domain_search_information_lines(lines)
        return "locate" if any(_extract_information_payload(line) for line in search_lines) else "search"
    state_family = _infer_state_task_family(query_text)
    lowered_lines = [_normalize_fact_token(_observation_evidence_text(line)) for line in lines if line and str(line).strip()]
    if state_family is not None:
        goal_slots = parse_goal_slots(query_text)
        has_state_ready_signal = any(
            _line_has_desired_state_completion(
                line,
                family=state_family,
                goal_slots=goal_slots,
            )
            for line in lines
            if line and str(line).strip()
        )
        has_post_state_progress = any(
            marker in line for line in lowered_lines for marker in ("you pick up", "you are carrying", "you put")
        )
        if state_family != "look_at_obj_in_light" and has_state_ready_signal and has_post_state_progress:
            return "post_state_action"
        if has_state_ready_signal:
            return "state_ready"
        if has_post_state_progress:
            if state_family == "look_at_obj_in_light":
                return "state_pending"
            return "post_state_action"
        if any(marker in line for line in lowered_lines for marker in GENERIC_STATE_LINE_MARKERS):
            return "state_pending"
        return "state_pending"

    goal_slots = parse_goal_slots(query_text)
    placed_entities: Set[str] = set()
    holding_entities: Set[str] = set()
    located_entities: Set[str] = set()
    has_location = False
    for line in lines:
        if not line or not str(line).strip():
            continue
        visible_anchor = _extract_visible_anchor(line)
        current_location = _extract_current_location(line)
        if visible_anchor is not None:
            has_location = True
        elif current_location is not None and goal_slots.target_receptacle and _entities_match(current_location, goal_slots.target_receptacle):
            has_location = True
        pickup_object = _extract_pickup_object(line)
        if pickup_object and _goal_object_for_entity(pickup_object, goal_slots):
            holding_entities.add(_normalize_fact_token(pickup_object))
        placement = _extract_placement(line)
        if placement is not None:
            placed_entity, receptacle = placement
            if _goal_object_for_entity(placed_entity, goal_slots) and _target_receptacle_matches(receptacle, goal_slots):
                placed_entities.add(_normalize_fact_token(placed_entity))
        signature = _extract_fact_signature(line)
        if signature is not None:
            subject, _ = signature
            if _goal_object_for_entity(subject, goal_slots):
                located_entities.add(_normalize_fact_token(subject))
                has_location = True
    placed = len(placed_entities)
    holding = len(holding_entities)
    located = len(located_entities)
    total_targets = _goal_target_count(goal_slots)
    if total_targets > 0 and placed >= total_targets:
        return "post_place"
    if placed > 0:
        return "partial_place"
    if holding > 0:
        return "carry_place"
    if located > 0 or has_location:
        return "locate"
    return "search"


def classify_slot_kinds(line_text: str, goal_slots: GoalSlots) -> Set[str]:
    kinds: Set[str] = set()

    pickup_object = _extract_pickup_object(line_text)
    placement = _extract_placement(line_text)
    receptacle_state = _extract_receptacle_state(line_text)
    current_location = _extract_current_location(line_text)
    visible_anchor = _extract_visible_anchor(line_text)
    signature = _extract_fact_signature(line_text)
    emergent_location = _extract_emergent_location_role(line_text, goal_slots)
    emergent_progress = _extract_emergent_progress_role(line_text, goal_slots)
    lowered = _normalize_fact_token(_observation_evidence_text(line_text))

    if pickup_object or any(marker in lowered for marker in ("you are carrying", "inventory")):
        kinds.add("inventory")
        kinds.add("progress")

    if placement is not None:
        kinds.add("progress")
        kinds.add("receptacle")

    if emergent_progress is not None:
        kinds.add("progress")

    if receptacle_state is not None:
        kinds.add("receptacle")

    if current_location is not None or visible_anchor is not None:
        kinds.add("location")

    if emergent_location is not None:
        kinds.add("location")

    if signature is not None:
        subject, value = signature
        if _goal_object_for_entity(subject, goal_slots):
            kinds.add("location")
        if goal_slots.target_receptacle and (
            _entities_match(subject, goal_slots.target_receptacle) or _entities_match(value, goal_slots.target_receptacle)
        ):
            kinds.add("receptacle")

    return kinds


def _collect_line_evidence_units(
    line_text: str,
    goal_slots: GoalSlots,
    *,
    state_device_hint: Optional[str] = None,
) -> Tuple[EvidenceUnit, ...]:
    parsed_line = _analyze_trust_line_cached(
        line_text,
        goal_slots.target_objects,
        goal_slots.target_receptacle,
    )
    units: List[EvidenceUnit] = []
    seen_keys: Set[Tuple[str, str, str, str, str]] = set()

    def _append_unit(
        family: str,
        skill: str,
        subject: Optional[str],
        value: Optional[str],
        role: str,
    ) -> None:
        subject_key = str(subject or "").strip()
        value_key = str(value or "").strip()
        if not subject_key or not value_key:
            return
        dedup_key = (family, skill, subject_key, value_key, role)
        if dedup_key in seen_keys:
            return
        seen_keys.add(dedup_key)
        units.append(
            EvidenceUnit(
                family=family,
                skill=skill,
                subject=subject_key,
                value=value_key,
                role=role,
                line=str(line_text),
            )
        )

    if parsed_line.current_location:
        _append_unit(
            "agent_location_pointer",
            "location",
            "agent",
            parsed_line.current_location,
            "agent_location",
        )

    if parsed_line.visible_anchor:
        _append_unit(
            "anchor_witness",
            "location",
            "anchor",
            parsed_line.visible_anchor,
            "anchor",
        )

    for goal_subject, anchor_value in _extract_anchor_object_locations(line_text, goal_slots):
        _append_unit(
            "object_location_pointer",
            "location",
            goal_subject,
            anchor_value,
            "target_fact",
        )

    signature = _extract_fact_signature(line_text) if parsed_line.has_fact_signature else None
    if signature is not None:
        subject, value = signature
        if not _goal_binding_is_distractor(line_text, subject, goal_slots):
            goal_subject = _goal_progress_subject_for_entity(subject, goal_slots)
            if goal_subject:
                _append_unit(
                    "object_location_pointer",
                    "location",
                    goal_subject,
                    value,
                    "target_fact",
                )
        if state_device_hint and _entities_match(subject, state_device_hint):
            _append_unit(
                "device_location_pointer",
                "location",
                state_device_hint,
                value,
                "state_device_fact",
            )

    if parsed_line.pickup_object:
        goal_subject = _goal_progress_subject_for_entity(parsed_line.pickup_object, goal_slots)
        if goal_subject:
            _append_unit(
                "inventory_pointer",
                "progress",
                goal_subject,
                "inventory",
                "holding",
            )
            _append_unit(
                "progress_witness",
                "progress",
                goal_subject,
                "holding",
                "holding",
            )

    if parsed_line.placement is not None:
        placed_object, receptacle = parsed_line.placement
        goal_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
        if goal_subject:
            _append_unit(
                "object_location_pointer",
                "location",
                goal_subject,
                receptacle,
                "target_fact",
            )
            _append_unit(
                "progress_witness",
                "progress",
                goal_subject,
                f"placed:{receptacle}",
                "placed",
            )
    elif parsed_line.emergent_progress is not None:
        progress_object, progress_value, progress_role = parsed_line.emergent_progress
        _append_unit(
            "progress_witness",
            "progress",
            progress_object,
            progress_value,
            progress_role,
        )

    if parsed_line.receptacle_state is not None:
        receptacle, is_open = parsed_line.receptacle_state
        _append_unit(
            "state_witness",
            "state",
            receptacle,
            "open" if is_open else "closed",
            "receptacle_state",
        )
    elif parsed_line.explicit_state is not None:
        object_name, state_name = parsed_line.explicit_state
        subject = _goal_progress_subject_for_entity(object_name, goal_slots) or object_name
        _append_unit(
            "state_witness",
            "state",
            subject,
            state_name,
            "object_state",
        )
    elif parsed_line.known_state is not None:
        subject, state_name, state_role = parsed_line.known_state
        mapped_subject = _goal_progress_subject_for_entity(subject, goal_slots) or subject
        _append_unit(
            "state_witness",
            "state",
            mapped_subject,
            state_name,
            state_role,
        )
    elif parsed_line.emergent_state is not None:
        subject, state_name, state_role = parsed_line.emergent_state
        _append_unit(
            "state_witness",
            "state",
            subject,
            state_name,
            state_role,
        )

    return tuple(units)


def _collect_evidence_units(
    lines: Sequence[str],
    goal_slots: GoalSlots,
    *,
    state_device_hint: Optional[str] = None,
) -> List[EvidenceUnit]:
    collected: List[EvidenceUnit] = []
    for line_text in lines:
        collected.extend(
            _collect_line_evidence_units(
                line_text,
                goal_slots,
                state_device_hint=state_device_hint,
            )
        )
    return collected


def _collect_skill_support_and_conflict_counts(
    query_text: str,
    raw_lines: Sequence[str],
    goal_slots: GoalSlots,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    support_counts = {
        "location": 0,
        "progress": 0,
        "state": 0,
    }
    conflict_counts = {
        "location": 0,
        "progress": 0,
        "state": 0,
    }

    if _is_open_domain_search_query(query_text):
        raw_lines = _coalesce_open_domain_search_information_lines(raw_lines)
        active_search_anchor = _search_task_anchor_hint(query_text) or _normalize_search_query_anchor(query_text)
        location_claims: Dict[str, Set[str]] = {}
        for line_text in raw_lines:
            search_query = _extract_search_query(line_text)
            if search_query:
                active_search_anchor = _normalize_search_query_anchor(search_query) or active_search_anchor
            info_payload = _extract_information_payload(line_text)
            if not active_search_anchor or not info_payload:
                continue
            fact_value = _canonicalize_query_fact_value(info_payload)
            if not fact_value:
                continue
            support_counts["location"] += 1
            location_claims.setdefault(active_search_anchor, set()).add(fact_value)
        conflict_counts["location"] = sum(max(0, len(values) - 1) for values in location_claims.values())
        return support_counts, conflict_counts

    location_claims: Dict[str, Set[str]] = {}
    state_claims: Dict[str, Set[str]] = {}
    progress_claims: Dict[str, Set[str]] = {}

    state_device_hint = _state_device_hint_for_family(
        infer_monitor_family(query_text),
        query_text=query_text,
    )
    evidence_units = _collect_evidence_units(
        raw_lines,
        goal_slots,
        state_device_hint=state_device_hint,
    )
    for unit in evidence_units:
        if unit.family == "anchor_witness":
            support_counts["location"] += 1
            continue
        if unit.skill in support_counts:
            support_counts[unit.skill] += 1
        if unit.family in {"agent_location_pointer", "object_location_pointer", "device_location_pointer"}:
            location_claims.setdefault(unit.subject, set()).add(unit.value)
        elif unit.family in {"inventory_pointer", "progress_witness"}:
            progress_claims.setdefault(unit.subject, set()).add(unit.value)
        elif unit.family == "state_witness":
            state_claims.setdefault(unit.subject, set()).add(unit.value)

    conflict_counts["location"] = sum(max(0, len(values) - 1) for values in location_claims.values())
    conflict_counts["state"] = sum(max(0, len(values) - 1) for values in state_claims.values())
    conflict_counts["progress"] = sum(max(0, len(values) - 1) for values in progress_claims.values())
    return support_counts, conflict_counts


def _schema_relation_subject_value(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
) -> Tuple[str, str, str]:
    relation_family = str(schema.relation_family or "").strip()
    subject = str(schema.subject or "").strip()
    current_value = str(schema.current_value or "").strip()
    if relation_family and subject and current_value:
        return relation_family, subject, current_value

    if kind == "location":
        signature = _schema_fact_signature(schema, goal_slots)
        if signature is not None:
            subject, current_value = signature
            return relation_family or "location", str(subject), str(current_value)
    elif kind == "progress":
        signature = _schema_progress_signature(schema, goal_slots)
        if signature is not None:
            subject, current_value = signature
            return relation_family or "progress", str(subject), str(current_value)
    elif kind == "state":
        signature = _schema_state_signature(schema, goal_slots)
        if signature is not None:
            subject, current_value = signature
            return relation_family or "state", str(subject), str(current_value)
    return relation_family or kind, subject or "generic", current_value


def _schema_pointer_lines(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
) -> Tuple[str, ...]:
    explicit_pointer_lines = _ordered_unique_lines(schema.pointer_lines)
    if explicit_pointer_lines:
        return explicit_pointer_lines

    relation_family, subject, current_value = _schema_relation_subject_value(schema, kind, goal_slots)
    if subject and current_value:
        pointer_candidates: List[str] = []
        for line in reversed(schema.lines):
            parsed_line = _analyze_trust_line_cached(
                line,
                goal_slots.target_objects,
                goal_slots.target_receptacle,
            )
            if relation_family == "agent_room" and parsed_line.current_location and _entities_match(parsed_line.current_location, current_value):
                pointer_candidates.append(line)
                continue
            if kind == "location":
                placement = parsed_line.placement
                if placement is not None:
                    placed_object, receptacle = placement
                    mapped_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
                    if mapped_subject == subject and _entities_match(receptacle, current_value):
                        pointer_candidates.append(line)
                        continue
                if current_value == "inventory" and parsed_line.pickup_object:
                    mapped_subject = _goal_progress_subject_for_entity(parsed_line.pickup_object, goal_slots)
                    if mapped_subject == subject:
                        pointer_candidates.append(line)
                        continue
                for anchor_subject, anchor_value in _extract_anchor_object_locations(line, goal_slots):
                    if anchor_subject == subject and _entities_match(anchor_value, current_value):
                        pointer_candidates.append(line)
                        break
                if pointer_candidates and pointer_candidates[-1] == line:
                    continue
                signature = _extract_fact_signature(line)
                if signature is not None:
                    fact_subject, fact_value = signature
                    mapped_subject = _goal_progress_subject_for_entity(fact_subject, goal_slots) or fact_subject
                    if mapped_subject == subject and _entities_match(fact_value, current_value):
                        pointer_candidates.append(line)
                        continue
            elif kind == "progress":
                if relation_family == "inventory" and parsed_line.pickup_object:
                    mapped_subject = _goal_progress_subject_for_entity(parsed_line.pickup_object, goal_slots)
                    if subject == "agent" and mapped_subject and mapped_subject == current_value:
                        pointer_candidates.append(line)
                        continue
                placement = parsed_line.placement
                if placement is not None:
                    placed_object, receptacle = placement
                    mapped_subject = _goal_progress_subject_for_entity(placed_object, goal_slots)
                    if mapped_subject == subject and current_value == f"placed:{receptacle}":
                        pointer_candidates.append(line)
                        continue
                if parsed_line.pickup_object:
                    mapped_subject = _goal_progress_subject_for_entity(parsed_line.pickup_object, goal_slots)
                    if mapped_subject == subject and current_value == "holding":
                        pointer_candidates.append(line)
                        continue
            elif kind == "state":
                explicit_state = parsed_line.explicit_state
                if explicit_state is not None:
                    object_name, state_name = explicit_state
                    mapped_subject = _goal_progress_subject_for_entity(object_name, goal_slots) or object_name
                    if mapped_subject == subject and state_name == current_value:
                        pointer_candidates.append(line)
                        continue
                if parsed_line.known_state is not None:
                    state_subject, state_name, _ = parsed_line.known_state
                    mapped_subject = _goal_progress_subject_for_entity(state_subject, goal_slots) or state_subject
                    if mapped_subject == subject and state_name == current_value:
                        pointer_candidates.append(line)
                        continue
                if parsed_line.receptacle_state is not None:
                    receptacle, is_open = parsed_line.receptacle_state
                    if receptacle == subject and ("open" if is_open else "closed") == current_value:
                        pointer_candidates.append(line)
                        continue
        pointer_lines = _ordered_unique_lines(pointer_candidates)
        if pointer_lines:
            return pointer_lines
    return _ordered_unique_lines(schema.lines[-1:]) or _ordered_unique_lines(schema.lines[:1])


def _schema_support_lines(
    schema: SchemaWitness,
    *,
    pointer_lines: Sequence[str],
) -> Tuple[str, ...]:
    explicit_support_lines = _ordered_unique_lines(schema.support_witness_lines)
    if explicit_support_lines:
        return explicit_support_lines
    pointer_keys = {_normalized_line_key(line) for line in pointer_lines}
    remaining_lines = [
        line
        for line in schema.lines
        if _normalized_line_key(line) not in pointer_keys
    ]
    fallback_support = _ordered_unique_lines(remaining_lines)
    if fallback_support:
        return fallback_support
    return _ordered_unique_lines(pointer_lines)


def _schema_keep_stats(
    schema: SchemaWitness,
    kind: str,
    goal_slots: GoalSlots,
    kept_line_keys: Set[str],
) -> Tuple[float, float, float, float]:
    pointer_lines = _schema_pointer_lines(schema, kind, goal_slots)
    support_lines = _schema_support_lines(schema, pointer_lines=pointer_lines)
    pointer_kept = float(any(_normalized_line_key(line) in kept_line_keys for line in pointer_lines))
    support_kept = float(any(_normalized_line_key(line) in kept_line_keys for line in support_lines))
    relevant_lines = _ordered_unique_lines(pointer_lines + support_lines)
    keep_fraction = (
        float(sum(1 for line in relevant_lines if _normalized_line_key(line) in kept_line_keys) / len(relevant_lines))
        if relevant_lines
        else 0.0
    )
    grounded_success = float(pointer_kept > 0.0 and support_kept > 0.0)
    return keep_fraction, grounded_success, pointer_kept, support_kept


def collect_trust_policy_monitor(
    query_text: Optional[str],
    raw_lines: Sequence[str],
    decisions: Optional[Sequence[RenderDecision]] = None,
    rendered_lines: Optional[Sequence[str]] = None,
    prompt_summary_text: Optional[str] = None,
    requested_mode: Optional[TrustContextMode] = None,
    state_aware: bool = False,
    detailed: bool = True,
    prepared_context: Optional[PreparedTrustContext] = None,
) -> Dict[str, Any]:
    if not query_text:
        return {}
    generic_search_query = _is_open_domain_search_query(query_text)

    cleaned_raw_lines = [line for line in raw_lines if line and str(line).strip()]
    analysis_raw_lines = (
        list(_coalesce_open_domain_search_information_lines(cleaned_raw_lines))
        if generic_search_query
        else cleaned_raw_lines
    )
    prepared = prepared_context
    if prepared is not None and tuple(cleaned_raw_lines) != prepared.raw_lines:
        prepared = None
    if prepared is None:
        prepared = prepare_trust_context(
            query_text=query_text,
            raw_lines=cleaned_raw_lines,
            requested_mode=requested_mode,
            state_aware=state_aware,
        )
    resolved_mode = (
        prepared.resolved_mode
        if prepared is not None
        else infer_query_context_mode(
            query_text=query_text,
            lines=cleaned_raw_lines,
            requested_mode=requested_mode,
            state_aware=state_aware,
        )
    )
    family = prepared.feedback_family if prepared is not None else infer_monitor_family(query_text)
    goal_slots = prepared.goal_slots if prepared is not None else parse_goal_slots(query_text)
    goal_objects = tuple(goal_slots.target_objects)
    goal_receptacle = goal_slots.target_receptacle
    state_device_hint = _state_device_hint_for_family(family, query_text=query_text)
    raw_skills = prepared.skills if prepared is not None else compile_memory_skills_from_lines(cleaned_raw_lines, query_text)
    phase = prepared.feedback_phase if prepared is not None else infer_monitor_phase(query_text, cleaned_raw_lines)
    skill_support_counts, skill_conflict_counts = _collect_skill_support_and_conflict_counts(
        query_text,
        analysis_raw_lines,
        goal_slots,
    )
    if rendered_lines is not None:
        kept_lines = [line for line in rendered_lines if line and str(line).strip()]
    elif decisions is not None:
        kept_lines = [
            decision.text
            for decision in decisions
            if decision.action != "hide" and "[hidden low-trust memory segment]" not in decision.text
        ]
    else:
        kept_lines = cleaned_raw_lines
    analysis_kept_lines = (
        list(_coalesce_open_domain_search_information_lines(kept_lines))
        if generic_search_query
        else kept_lines
    )
    raw_evidence_units = [] if generic_search_query else _collect_evidence_units(
        cleaned_raw_lines,
        goal_slots,
        state_device_hint=state_device_hint,
    )
    kept_evidence_units = [] if generic_search_query else _collect_evidence_units(
        kept_lines,
        goal_slots,
        state_device_hint=state_device_hint,
    )
    skills = compile_memory_skills_from_lines(analysis_kept_lines, query_text) if detailed else raw_skills
    kept_line_keys = {_normalized_line_key(line) for line in analysis_kept_lines}
    witness_lines_by_skill = {
        kind: list(raw_skills.witness_lines_by_skill.get(kind, ()))
        for kind in ("location", "progress", "state")
    }
    witness_roles_by_skill = {
        kind: {
            line: raw_skills.witness_roles_by_skill.get(kind, {}).get(_normalized_line_key(line), "generic")
            for line in witness_lines
        }
        for kind, witness_lines in witness_lines_by_skill.items()
    }
    kept_witness_lines_by_skill = {
        kind: [
            line
            for line in witness_lines
            if _normalized_line_key(line) in kept_line_keys
        ]
        for kind, witness_lines in witness_lines_by_skill.items()
    }
    kept_witness_roles_by_skill = {
        kind: {
            line: witness_roles_by_skill[kind].get(line, "generic")
            for line in kept_witness_lines
        }
        for kind, kept_witness_lines in kept_witness_lines_by_skill.items()
    }
    witness_role_counts_by_skill = {
        kind: {
            role: sum(1 for mapped_role in witness_roles.values() if mapped_role == role)
            for role in sorted(set(witness_roles.values()))
        }
        for kind, witness_roles in witness_roles_by_skill.items()
    }
    kept_witness_role_counts_by_skill = {
        kind: {
            role: sum(1 for mapped_role in kept_witness_roles.values() if mapped_role == role)
            for role in sorted(set(kept_witness_roles.values()))
        }
        for kind, kept_witness_roles in kept_witness_roles_by_skill.items()
    }
    summary_fact_keys = _summary_fact_keys_from_text(str(prompt_summary_text or ""))
    summary_duplicate_lines_by_skill = {
        kind: [
            line
            for line in kept_witness_lines_by_skill[kind]
            if _line_summary_fact_keys(line, kind=kind, goal_slots=goal_slots) & summary_fact_keys
        ]
        for kind in ("location", "progress", "state")
    }
    metrics: Dict[str, Any] = {
        "trust_policy/query_mode": float(resolved_mode == "query"),
        "trust_policy/state_mode": float(resolved_mode == "state"),
        "trust_policy/slot_mode": float(resolved_mode == "slot"),
        "trust_policy/family": family,
        "trust_policy/family_target": family,
        "trust_policy/phase": phase,
        "trust_policy/phase_target": phase,
        "trust_policy/family_phase_target": f"{family}::{phase}",
        f"trust_policy/family/{family}": 1.0,
        f"trust_policy/family/{family}/phase/{phase}": 1.0,
        f"trust_policy/family/{family}/query_mode": float(resolved_mode == "query"),
        f"trust_policy/family/{family}/state_mode": float(resolved_mode == "state"),
        f"trust_policy/family/{family}/slot_mode": float(resolved_mode == "slot"),
        "trust_policy/target_object_count": float(_goal_target_count(goal_slots)),
        "trust_policy/has_target_receptacle": float(goal_slots.target_receptacle is not None),
        "trust_policy/skill/location_present": float(bool(skills.current_location or skills.target_locations or skills.visible_anchor)),
        "trust_policy/skill/location_conf": float(skills.confidence_by_skill.get("location", 0.0)),
        "trust_policy/skill/progress_present": float(
            bool(skills.search_bridge_entity or skills.search_answer_candidate or skills.searched_without_target)
            if generic_search_query
            else _has_observed_progress(skills.object_progress)
        ),
        "trust_policy/pick_two/placed_count": float(_goal_progress_state_count(skills.object_progress, goal_slots, "placed")),
        "trust_policy/pick_two/holding_count": float(_goal_progress_state_count(skills.object_progress, goal_slots, "holding")),
        "trust_policy/pick_two/remaining_count": float(
            max(
                0,
                _goal_target_count(goal_slots)
                - _goal_progress_state_count(skills.object_progress, goal_slots, "placed"),
            )
        ),
        "trust_policy/skill/state_present": float(bool(skills.state_flags)),
        "trust_policy/skill/location_support_count": float(skill_support_counts["location"]),
        "trust_policy/skill/progress_support_count": float(skill_support_counts["progress"]),
        "trust_policy/skill/state_support_count": float(skill_support_counts["state"]),
        "trust_policy/skill/location_conflict_count": float(skill_conflict_counts["location"]),
        "trust_policy/skill/progress_conflict_count": float(skill_conflict_counts["progress"]),
        "trust_policy/skill/state_conflict_count": float(skill_conflict_counts["state"]),
        "trust_policy/witness/location_lines": witness_lines_by_skill["location"],
        "trust_policy/witness/progress_lines": witness_lines_by_skill["progress"],
        "trust_policy/witness/state_lines": witness_lines_by_skill["state"],
        "trust_policy/witness/location_roles": witness_roles_by_skill["location"],
        "trust_policy/witness/progress_roles": witness_roles_by_skill["progress"],
        "trust_policy/witness/state_roles": witness_roles_by_skill["state"],
        "trust_policy/witness/location_role_counts": witness_role_counts_by_skill["location"],
        "trust_policy/witness/progress_role_counts": witness_role_counts_by_skill["progress"],
        "trust_policy/witness/state_role_counts": witness_role_counts_by_skill["state"],
        "trust_policy/witness/location_kept_lines": kept_witness_lines_by_skill["location"],
        "trust_policy/witness/progress_kept_lines": kept_witness_lines_by_skill["progress"],
        "trust_policy/witness/state_kept_lines": kept_witness_lines_by_skill["state"],
        "trust_policy/witness/location_kept_roles": kept_witness_roles_by_skill["location"],
        "trust_policy/witness/progress_kept_roles": kept_witness_roles_by_skill["progress"],
        "trust_policy/witness/state_kept_roles": kept_witness_roles_by_skill["state"],
        "trust_policy/witness/location_kept_role_counts": kept_witness_role_counts_by_skill["location"],
        "trust_policy/witness/progress_kept_role_counts": kept_witness_role_counts_by_skill["progress"],
        "trust_policy/witness/state_kept_role_counts": kept_witness_role_counts_by_skill["state"],
        "trust_policy/witness/location_count": float(len(witness_lines_by_skill["location"])),
        "trust_policy/witness/progress_count": float(len(witness_lines_by_skill["progress"])),
        "trust_policy/witness/state_count": float(len(witness_lines_by_skill["state"])),
        "trust_policy/witness/location_kept_count": float(len(kept_witness_lines_by_skill["location"])),
        "trust_policy/witness/progress_kept_count": float(len(kept_witness_lines_by_skill["progress"])),
        "trust_policy/witness/state_kept_count": float(len(kept_witness_lines_by_skill["state"])),
        "trust_policy/prompt_summary_active": float(bool(summary_fact_keys)),
        "trust_policy/raw_line_count": float(len(cleaned_raw_lines)),
        "trust_policy/rendered_line_count": float(len(kept_lines)),
        "trust_policy/raw_char_count": float(sum(len(str(line)) for line in cleaned_raw_lines)),
        "trust_policy/rendered_char_count": float(sum(len(str(line)) for line in kept_lines)),
    }
    evidence_families = (
        "agent_location_pointer",
        "object_location_pointer",
        "device_location_pointer",
        "inventory_pointer",
        "progress_witness",
        "state_witness",
        "anchor_witness",
    )
    raw_evidence_counts = {
        family_name: sum(1 for unit in raw_evidence_units if unit.family == family_name)
        for family_name in evidence_families
    }
    kept_evidence_counts = {
        family_name: sum(1 for unit in kept_evidence_units if unit.family == family_name)
        for family_name in evidence_families
    }
    for family_name in evidence_families:
        metrics[f"trust_policy/evidence/{family_name}_evidence_count"] = float(raw_evidence_counts[family_name])
        metrics[f"trust_policy/evidence/{family_name}_kept_count"] = float(kept_evidence_counts[family_name])
        metrics[f"trust_policy/evidence/{family_name}_evidence"] = float(raw_evidence_counts[family_name] > 0)
        metrics[f"trust_policy/evidence/{family_name}_kept"] = float(kept_evidence_counts[family_name] > 0)
    location_pointer_evidence = (
        raw_evidence_counts["agent_location_pointer"]
        + raw_evidence_counts["object_location_pointer"]
        + raw_evidence_counts["device_location_pointer"]
    )
    location_pointer_kept = (
        kept_evidence_counts["agent_location_pointer"]
        + kept_evidence_counts["object_location_pointer"]
        + kept_evidence_counts["device_location_pointer"]
    )
    metrics["trust_policy/evidence/location_pointer_evidence_count"] = float(location_pointer_evidence)
    metrics["trust_policy/evidence/location_pointer_kept_count"] = float(location_pointer_kept)
    metrics["trust_policy/evidence/location_pointer_evidence"] = float(location_pointer_evidence > 0)
    metrics["trust_policy/evidence/location_pointer_kept"] = float(location_pointer_kept > 0)
    state_family = family if family in DESIRED_STATE_BY_FAMILY else None
    if state_family is not None:
        required_state = _required_state_for_family(state_family)
        raw_achieved_state_lines = [
            line for line in cleaned_raw_lines
            if _line_has_desired_state_completion(line, family=state_family, goal_slots=goal_slots)
        ]
        kept_achieved_state_lines = [
            line for line in kept_lines
            if _line_has_desired_state_completion(line, family=state_family, goal_slots=goal_slots)
        ]
        raw_state_transition_lines = [
            line
            for line in cleaned_raw_lines
            if (
                (explicit_state := _extract_explicit_state_action(line)) is not None
                and explicit_state[1] == required_state
            )
        ]
        kept_state_transition_lines = [
            line
            for line in kept_lines
            if (
                (explicit_state := _extract_explicit_state_action(line)) is not None
                and explicit_state[1] == required_state
            )
        ]
        raw_state_device_lines = [
            line for line in cleaned_raw_lines if _line_mentions_state_device(line, family=state_family)
        ]
        kept_state_device_lines = [
            line for line in kept_lines if _line_mentions_state_device(line, family=state_family)
        ]
        final_receptacle_kept = float(
            bool(goal_receptacle)
            and any(
                _line_mentions_entity(line, goal_receptacle)
                or f"receptacle={goal_receptacle}" in _normalize_fact_token(_strip_xml_tags(line))
                for line in kept_lines
            )
        )
        metrics["trust_policy/state/required_state_kept"] = float(
            bool(required_state)
            and (
                any(f"required_state={required_state}" in str(line) for line in kept_lines)
                or f"required state: {required_state}" in str(prompt_summary_text or "").lower()
            )
        )
        metrics["trust_policy/state/achieved_state_evidence"] = float(bool(raw_achieved_state_lines))
        metrics["trust_policy/state/achieved_state_kept"] = float(bool(kept_achieved_state_lines))
        metrics["trust_policy/state/state_transition_raw_witness_evidence"] = float(bool(raw_state_transition_lines))
        metrics["trust_policy/state/state_transition_raw_witness_kept"] = float(bool(kept_state_transition_lines))
        metrics["trust_policy/state/state_device_evidence"] = float(bool(raw_state_device_lines))
        metrics["trust_policy/state/state_device_kept"] = float(bool(kept_state_device_lines))
        metrics["trust_policy/state/final_receptacle_kept"] = final_receptacle_kept
        metrics["trust_policy/state/state_only_witness_drop"] = float(
            (
                len(raw_achieved_state_lines) == 1
                and len(kept_achieved_state_lines) == 0
            )
            or (
                len(raw_state_transition_lines) == 1
                and len(kept_state_transition_lines) == 0
            )
        )
    raw_line_total = float(metrics["trust_policy/raw_line_count"])
    rendered_line_total = float(metrics["trust_policy/rendered_line_count"])
    raw_char_total = float(metrics["trust_policy/raw_char_count"])
    rendered_char_total = float(metrics["trust_policy/rendered_char_count"])
    metrics["trust_policy/render_line_keep_ratio"] = (
        rendered_line_total / raw_line_total if raw_line_total > 0.0 else 0.0
    )
    metrics["trust_policy/render_char_keep_ratio"] = (
        rendered_char_total / raw_char_total if raw_char_total > 0.0 else 0.0
    )
    metrics["trust_policy/render_compacted"] = float(
        rendered_line_total + 1e-8 < raw_line_total or rendered_char_total + 1e-8 < raw_char_total
    )

    slot_kinds = ("inventory", "receptacle", "progress", "location")
    total_kind_counts = {kind: 0 for kind in slot_kinds}
    kept_kind_counts = {kind: 0 for kind in slot_kinds}

    if not generic_search_query:
        for line in raw_lines:
            kinds = _analyze_trust_line_cached(line, goal_objects, goal_receptacle).line_slot_kinds
            for kind in kinds:
                if kind in total_kind_counts:
                    total_kind_counts[kind] += 1

        for line in kept_lines:
            kinds = _analyze_trust_line_cached(line, goal_objects, goal_receptacle).line_slot_kinds
            for kind in kinds:
                if kind in kept_kind_counts:
                    kept_kind_counts[kind] += 1

    for kind in slot_kinds:
        metrics[f"trust_policy/slot/{kind}_evidence"] = float(total_kind_counts[kind] > 0)
        metrics[f"trust_policy/slot/{kind}_kept"] = float(kept_kind_counts[kind] > 0)
        metrics[f"trust_policy/slot/{kind}_evidence_count"] = float(total_kind_counts[kind])
        metrics[f"trust_policy/slot/{kind}_kept_count"] = float(kept_kind_counts[kind])
        metrics[f"trust_policy/slot/{kind}_kept_fraction"] = (
            float(kept_kind_counts[kind] / total_kind_counts[kind]) if total_kind_counts[kind] > 0 else 0.0
        )

    witness_keep_fractions: List[float] = []
    anchor_total = float(witness_role_counts_by_skill["location"].get("anchor", 0))
    anchor_kept = float(kept_witness_role_counts_by_skill["location"].get("anchor", 0))
    for kind in ("location", "progress", "state"):
        witness_total = float(len(witness_lines_by_skill[kind]))
        witness_kept = float(len(kept_witness_lines_by_skill[kind]))
        witness_keep_fraction = (witness_kept / witness_total) if witness_total > 0.0 else 0.0
        metrics[f"trust_policy/witness/{kind}_kept_fraction"] = witness_keep_fraction
        summary_duplicate_count = float(len(summary_duplicate_lines_by_skill[kind]))
        metrics[f"trust_policy/witness/{kind}_summary_duplicate_lines"] = summary_duplicate_lines_by_skill[kind]
        metrics[f"trust_policy/witness/{kind}_summary_duplicate_count"] = summary_duplicate_count
        metrics[f"trust_policy/witness/{kind}_summary_duplicate_rate"] = (
            summary_duplicate_count / witness_kept if witness_kept > 0.0 else 0.0
        )
        if witness_total > 0.0:
            witness_keep_fractions.append(witness_keep_fraction)

    slot_keep_fractions = [
        float(metrics[f"trust_policy/slot/{kind}_kept_fraction"])
        for kind in slot_kinds
        if float(metrics[f"trust_policy/slot/{kind}_evidence_count"]) > 0.0
    ]
    summary_duplicate_total = float(
        sum(len(summary_duplicate_lines_by_skill[kind]) for kind in ("location", "progress", "state"))
    )
    kept_witness_total = float(sum(len(kept_witness_lines_by_skill[kind]) for kind in ("location", "progress", "state")))
    metrics["trust_policy/mechanism/key_evidence_keep_rate"] = (
        float(sum(slot_keep_fractions) / len(slot_keep_fractions)) if slot_keep_fractions else 0.0
    )
    metrics["trust_policy/mechanism/witness_keep_rate"] = (
        float(sum(witness_keep_fractions) / len(witness_keep_fractions)) if witness_keep_fractions else 0.0
    )
    metrics["trust_policy/mechanism/summary_duplicate_reinjection_count"] = summary_duplicate_total
    metrics["trust_policy/mechanism/summary_duplicate_reinjection_rate"] = (
        summary_duplicate_total / kept_witness_total if kept_witness_total > 0.0 else 0.0
    )
    metrics["trust_policy/mechanism/anchor_keep_rate"] = (anchor_kept / anchor_total) if anchor_total > 0.0 else 0.0
    metrics["trust_policy/mechanism/evidence_retention_precision_rate"] = (
        0.60 * float(metrics["trust_policy/mechanism/key_evidence_keep_rate"])
        + 0.25 * float(metrics["trust_policy/mechanism/witness_keep_rate"])
        + 0.15 * float(metrics["trust_policy/mechanism/anchor_keep_rate"])
    )

    if not detailed:
        return metrics

    schema_witnesses_by_skill = {
        kind: list(raw_skills.schema_witnesses_by_skill.get(kind, ()))
        for kind in ("location", "progress", "state")
    }
    grounded_evidence_counts = {
        "location_pointer": 0,
        "inventory_pointer": 0,
        "progress_witness": 0,
        "state_witness": 0,
    }
    grounded_kept_counts = {
        "location_pointer": 0,
        "inventory_pointer": 0,
        "progress_witness": 0,
        "state_witness": 0,
    }

    schema_refresh_applicable_flags: List[float] = []
    schema_refresh_success_flags: List[float] = []
    schema_target_keep_fractions: List[float] = []
    fresh_schema_selection_flags: List[float] = []
    stale_schema_activation_flags: List[float] = []
    for kind in ("location", "progress", "state"):
        schemas = schema_witnesses_by_skill[kind]
        metrics[f"trust_policy/schema/{kind}_count"] = float(len(schemas))
        metrics[f"trust_policy/schema/{kind}_stale_count"] = float(
            sum(1 for schema in schemas if schema.lifecycle_stage in {"stale", "retired"})
        )
        metrics[f"trust_policy/schema/{kind}_freshness_mean"] = (
            float(sum(schema.freshness for schema in schemas) / len(schemas)) if schemas else 0.0
        )
        metrics[f"trust_policy/schema/{kind}_uncertainty_mean"] = (
            float(sum(schema.uncertainty for schema in schemas) / len(schemas)) if schemas else 0.0
        )
        if not schemas:
            metrics[f"trust_policy/schema/{kind}_target_keep_fraction"] = 0.0
            metrics[f"trust_policy/schema/{kind}_grounded_keep_rate"] = 0.0
            metrics[f"trust_policy/schema/{kind}_fresh_selection"] = 0.0
            metrics[f"trust_policy/schema/{kind}_stale_activation"] = 0.0
            metrics[f"trust_policy/schema/{kind}_refresh_applicable"] = 0.0
            metrics[f"trust_policy/schema/{kind}_refresh_success"] = 0.0
            continue
        grouped_schemas: Dict[Tuple[str, str], List[SchemaWitness]] = {}
        for schema in schemas:
            relation_family, subject, _ = _schema_relation_subject_value(schema, kind, goal_slots)
            group_key = (
                relation_family or kind,
                subject or _schema_primary_subject(schema, kind, goal_slots),
            )
            grouped_schemas.setdefault(group_key, []).append(schema)

        applicable_group_flags: List[float] = []
        refresh_success_values: List[float] = []
        target_keep_values: List[float] = []
        grounded_keep_values: List[float] = []
        fresh_selection_values: List[float] = []
        stale_activation_values: List[float] = []

        for group_schemas in grouped_schemas.values():
            target_schema = _best_schema_representative(group_schemas)
            relation_family, _, current_value = _schema_relation_subject_value(target_schema, kind, goal_slots)
            target_keep_fraction, grounded_success, _, _ = _schema_keep_stats(
                target_schema,
                kind,
                goal_slots,
                kept_line_keys,
            )
            target_keep_values.append(target_keep_fraction)
            grounded_keep_values.append(grounded_success)
            fresh_selection_values.append(float(target_schema.lifecycle_stage not in {"stale", "retired"}))

            if kind == "location" and relation_family in {"agent_room", "object_location", "device_location"} and current_value != "inventory":
                grounded_evidence_counts["location_pointer"] += 1
                grounded_kept_counts["location_pointer"] += int(grounded_success > 0.0)
            elif kind == "progress" and relation_family == "inventory" and current_value != "empty":
                grounded_evidence_counts["inventory_pointer"] += 1
                grounded_kept_counts["inventory_pointer"] += int(grounded_success > 0.0)
            elif kind == "progress" and relation_family == "task_progress":
                grounded_evidence_counts["progress_witness"] += 1
                grounded_kept_counts["progress_witness"] += int(grounded_success > 0.0)
            elif kind == "state":
                grounded_evidence_counts["state_witness"] += 1
                grounded_kept_counts["state_witness"] += int(grounded_success > 0.0)

            unique_values = {
                value or _schema_line_key(schema.lines)
                for _, _, value in (_schema_relation_subject_value(schema, kind, goal_slots) for schema in group_schemas)
            }
            stale_schemas = [
                schema
                for schema in group_schemas
                if schema is not target_schema and schema.lifecycle_stage in {"stale", "retired"}
            ]
            applicable = float(len(unique_values) > 1 or bool(stale_schemas))
            if applicable > 0.0:
                applicable_group_flags.append(applicable)
                refresh_success_values.append(
                    float(
                        grounded_success > 0.0
                        and target_schema.lifecycle_stage not in {"stale", "retired"}
                    )
                )
            if stale_schemas:
                stale_activation_values.append(
                    max(
                        _schema_keep_stats(schema, kind, goal_slots, kept_line_keys)[0]
                        for schema in stale_schemas
                    )
                )

        applicable = float(bool(applicable_group_flags))
        refresh_success = (
            float(sum(refresh_success_values) / len(refresh_success_values))
            if refresh_success_values
            else 0.0
        )
        target_keep_fraction = (
            float(sum(target_keep_values) / len(target_keep_values))
            if target_keep_values
            else 0.0
        )
        grounded_keep_rate = (
            float(sum(grounded_keep_values) / len(grounded_keep_values))
            if grounded_keep_values
            else 0.0
        )
        fresh_selection = (
            float(sum(fresh_selection_values) / len(fresh_selection_values))
            if fresh_selection_values
            else 0.0
        )
        stale_activation = (
            float(sum(stale_activation_values) / len(stale_activation_values))
            if stale_activation_values
            else 0.0
        )
        metrics[f"trust_policy/schema/{kind}_refresh_applicable"] = applicable
        metrics[f"trust_policy/schema/{kind}_target_keep_fraction"] = target_keep_fraction
        metrics[f"trust_policy/schema/{kind}_grounded_keep_rate"] = grounded_keep_rate
        metrics[f"trust_policy/schema/{kind}_refresh_success"] = refresh_success
        metrics[f"trust_policy/schema/{kind}_fresh_selection"] = fresh_selection
        metrics[f"trust_policy/schema/{kind}_stale_activation"] = stale_activation
        if applicable > 0.0:
            schema_refresh_applicable_flags.append(applicable)
            schema_refresh_success_flags.append(refresh_success)
            schema_target_keep_fractions.append(target_keep_fraction)
            fresh_schema_selection_flags.append(fresh_selection)
        if stale_activation_values:
            stale_schema_activation_flags.append(stale_activation)

    for family_name in ("location_pointer", "inventory_pointer", "progress_witness", "state_witness"):
        grounded_evidence_count = float(grounded_evidence_counts[family_name])
        grounded_kept_count = float(grounded_kept_counts[family_name])
        grounded_miss_count = max(0.0, grounded_evidence_count - grounded_kept_count)
        metrics[f"trust_policy/evidence/{family_name}_grounded_evidence_count"] = grounded_evidence_count
        metrics[f"trust_policy/evidence/{family_name}_grounded_kept_count"] = grounded_kept_count
        metrics[f"trust_policy/evidence/{family_name}_grounded_miss_count"] = grounded_miss_count
        metrics[f"trust_policy/evidence/{family_name}_grounded_keep_rate"] = (
            grounded_kept_count / grounded_evidence_count if grounded_evidence_count > 0.0 else 0.0
        )

    metrics["trust_policy/mechanism/schema_refresh_applicable_rate"] = (
        float(sum(schema_refresh_applicable_flags) / len(schema_refresh_applicable_flags))
        if schema_refresh_applicable_flags
        else 0.0
    )
    metrics["trust_policy/mechanism/schema_refresh_accuracy_rate"] = (
        float(sum(schema_refresh_success_flags) / len(schema_refresh_success_flags))
        if schema_refresh_success_flags
        else 0.0
    )
    metrics["trust_policy/mechanism/schema_target_keep_rate"] = (
        float(sum(schema_target_keep_fractions) / len(schema_target_keep_fractions))
        if schema_target_keep_fractions
        else 0.0
    )
    metrics["trust_policy/mechanism/fresh_schema_selection_rate"] = (
        float(sum(fresh_schema_selection_flags) / len(fresh_schema_selection_flags))
        if fresh_schema_selection_flags
        else 0.0
    )
    metrics["trust_policy/mechanism/stale_schema_activation_rate"] = (
        float(sum(stale_schema_activation_flags) / len(stale_schema_activation_flags))
        if stale_schema_activation_flags
        else 0.0
    )

    return metrics


def _infer_state_task_family(query_text: str) -> Optional[str]:
    normalized = _normalize_fact_token(query_text)
    tokens = set(normalized.split())

    if (
        any(word in tokens for word in {"look", "examine", "inspect"})
        and any(word in tokens for word in {"light", "lamp", "desklamp", "switch", "bright", "dark"})
    ):
        return "look_at_obj_in_light"

    if any(word in tokens for word in {"heat", "hot", "warm", "microwave"}):
        return "pick_heat_then_place_in_recep"

    if any(word in tokens for word in {"cool", "cold", "chill", "chilled", "freezer", "ice"}):
        return "pick_cool_then_place_in_recep"

    if any(word in tokens for word in {"clean", "wash", "rinse", "rinsed"}):
        return "pick_clean_then_place_in_recep"

    # Backward-compatible fallback for very close phrasing.
    for task_family, markers in STATE_TASK_MARKERS.items():
        if sum(marker in normalized for marker in markers) >= 2:
            return task_family
    return None


def _query_requests_state_tracking(query_text: str) -> bool:
    lowered = _normalize_fact_token(query_text)
    tokens = set(lowered.split())
    for marker in GENERIC_STATE_QUERY_MARKERS:
        normalized_marker = _normalize_fact_token(marker)
        if not normalized_marker:
            continue
        if " " in normalized_marker:
            if normalized_marker in lowered:
                return True
            continue
        if normalized_marker in tokens:
            return True
    return _infer_state_task_family(query_text) is not None


def _generic_state_line_score(text: str) -> float:
    lowered = f" {_normalize_fact_token(_observation_evidence_text(text))} "
    score = 0.0
    fact_signature = _extract_fact_signature(text)

    generic_markers_wo_on_off = tuple(
        marker for marker in GENERIC_STATE_LINE_MARKERS if marker not in {" is on", " is off"}
    )
    if any(marker in lowered for marker in generic_markers_wo_on_off):
        score = max(score, 0.85)
    if any(marker in lowered for marker in (" is on", " is off")):
        on_off_is_state = (
            fact_signature is None
            or _extract_receptacle_state(text) is not None
            or any(token in lowered for token in (" light ", " lamp ", " switch ", " desklamp "))
        )
        if on_off_is_state:
            score = max(score, 0.85)
    if _extract_explicit_state_action(text) is not None:
        score = max(score, 0.85)
    if any(marker in lowered for marker in GENERIC_TRANSITION_MARKERS):
        score = max(score, 0.7)
    if any(marker in lowered for marker in ("you pick up", "you take", "you are carrying", "you hold", "inventory")):
        score = max(score, 0.6)
    if any(marker in lowered for marker in DISTRACTOR_MARKERS):
        score = min(score, 0.1) if score > 0.0 else 0.05

    return score


def _state_line_score(text: str, task_family: Optional[str]) -> float:
    generic_score = _generic_state_line_score(text)
    if not task_family:
        return generic_score

    lowered = _normalize_fact_token(_observation_evidence_text(text))
    explicit_state = _extract_explicit_state_action(text)
    if task_family == "look_at_obj_in_light" and "too dark" in lowered:
        return max(generic_score, 0.0)
    if explicit_state is not None and explicit_state[1] == DESIRED_STATE_BY_FAMILY.get(task_family):
        return max(generic_score, 1.0)
    if any(marker in lowered for marker in STATE_LINE_MARKERS[task_family]):
        return max(generic_score, 1.0)

    if any(marker in lowered for marker in ("you pick up", "you take", "you are carrying", "you hold")):
        return max(generic_score, 0.6)

    if _extract_fact_signature(text) is not None:
        return max(generic_score, 0.1)

    return generic_score


def infer_query_context_mode(
    query_text: str,
    lines: Sequence[str],
    requested_mode: Optional[TrustContextMode] = None,
    state_aware: bool = False,
) -> TrustContextMode:
    if requested_mode in {"query", "state", "slot"}:
        return requested_mode
    if requested_mode != "auto":
        return "state" if state_aware else "query"

    state_family = _infer_state_task_family(query_text)
    if state_family is not None:
        return "state"

    if _query_requests_state_tracking(query_text):
        return "state"

    goal_slots = parse_goal_slots(query_text)
    phase = infer_monitor_phase(query_text, lines)
    slot_requested = _query_requests_slot_tracking(query_text)
    slot_scores = _slot_line_scores(lines, goal_slots) if goal_slots.target_objects or goal_slots.target_receptacle else []
    strong_slot_lines = sum(1 for score in slot_scores if score >= 0.7)
    has_high_precision_slot_signal = any(score >= 0.75 for score in slot_scores)

    if slot_requested:
        if len(goal_slots.target_objects) > 1 and phase == "carry_place":
            return "query"
        if _goal_uses_instance_tracking(goal_slots):
            return "slot"
        if strong_slot_lines >= 2:
            return "slot"
        if has_high_precision_slot_signal and phase != "search":
            return "slot"
        if phase in {"search", "locate", "carry_place"}:
            return "query"
        return "slot"

    if len(goal_slots.target_objects) > 1 and not _goal_uses_instance_tracking(goal_slots):
        if phase == "carry_place":
            return "query"
    if strong_slot_lines >= 2:
        return "slot"
    # In auto mode, state-aware routing should only activate when the query
    # itself explicitly asks for state tracking (or matches a known state
    # task family). Generic ALFWorld memories frequently contain incidental
    # lines such as "the fridge is closed" or "you are carrying ..."; using
    # those lines alone to promote a task into the state branch collapses
    # query-mode and prevents the query-conditioned path from activating.
    return "query"


def build_trust_segments_from_lines(
    lines: Iterable[str],
    current_step: int,
    default_source_id: str = "env",
) -> List[SegmentTrustMetadata]:
    raw_lines = [line.strip() for line in lines if line and line.strip()]
    if not raw_lines:
        return []

    signatures: List[Optional[Tuple[str, str]]] = []
    active_search_anchor = ""
    for line in raw_lines:
        search_query = _extract_search_query(line)
        if search_query:
            active_search_anchor = _normalize_search_query_anchor(search_query)
        signature = _extract_fact_signature(line)
        if signature is None:
            signature = _extract_search_query_fact_signature(
                line,
                active_search_anchor=active_search_anchor,
            )
        signatures.append(signature)
    support_counts = [1] * len(raw_lines)
    contradiction_counts = [0] * len(raw_lines)
    grouped_claims: Dict[str, List[Tuple[int, str]]] = {}

    for index, signature in enumerate(signatures):
        if signature is None:
            continue
        subject, value = signature
        grouped_claims.setdefault(subject, []).append((index, value))

    for claims in grouped_claims.values():
        value_counts: Dict[str, int] = {}
        for _, value in claims:
            value_counts[value] = value_counts.get(value, 0) + 1
        for index, value in claims:
            support_counts[index] = max(support_counts[index], value_counts[value])
            contradiction_counts[index] += sum(1 for _, other_value in claims if other_value != value)

    total_lines = len(raw_lines)
    segments: List[SegmentTrustMetadata] = []
    for line_idx, line in enumerate(raw_lines, start=1):
        lower = line.lower()
        is_skill_line = lower.startswith("[skill]")
        suspicious_score = 0.9 if any(marker in lower for marker in SUSPICIOUS_MARKERS) else 0.0
        contradiction_hint = 2 if suspicious_score > 0 else 0
        if contradiction_hint == 0 and any(marker in lower for marker in CONTRADICTION_MARKERS):
            contradiction_hint = 1
        salience = 0.8 if any(marker in lower for marker in SALIENT_MARKERS) else 0.5

        if signatures[line_idx - 1] is not None and support_counts[line_idx - 1] > 1:
            salience = max(salience, 0.7)

        if is_skill_line:
            source_trust = 0.95
            salience = max(salience, 0.95)
        elif suspicious_score > 0:
            source_trust = 0.2
        elif contradiction_hint > 0:
            source_trust = 0.4
        else:
            source_trust = 0.8

        segments.append(
            SegmentTrustMetadata(
                text=line,
                step=max(1, current_step - total_lines + line_idx),
                source_id=default_source_id,
                source_trust=source_trust,
                support_count=support_counts[line_idx - 1],
                contradiction_count=max(contradiction_counts[line_idx - 1], contradiction_hint),
                suspicious_score=suspicious_score,
                salience=salience,
                query_relevance=0.5,
            )
        )
    return segments


def build_query_conditioned_segments_from_lines(
    lines: Iterable[str],
    current_step: int,
    query_text: str,
    default_source_id: str = "env",
    state_aware: bool = False,
    context_mode: Optional[TrustContextMode] = None,
    skill_feedback: Optional[MemorySkillFeedback] = None,
    preserve_skill_witnesses: bool = True,
    preserve_schema_witnesses: bool = True,
    adaptive_witness_budget: bool = True,
    decouple_role_signals: bool = True,
    adaptive_skill_mix: bool = True,
    memory_framework_config: Optional[MemoryFrameworkConfig] = None,
    prepared_context: Optional[PreparedTrustContext] = None,
) -> List[SegmentTrustMetadata]:
    framework = resolve_memory_framework_config(memory_framework_config)
    raw_lines = [line for line in lines if line and line.strip()]
    disable_skill_summary = os.getenv("AGENTOCR_DISABLE_TRUST_SKILL_SUMMARY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    prepared = prepared_context
    if prepared is not None and tuple(raw_lines) != prepared.raw_lines:
        prepared = None
    if prepared is None:
        prepared = prepare_trust_context(
            query_text=query_text,
            raw_lines=raw_lines,
            requested_mode=context_mode,
            state_aware=state_aware,
            feedback=skill_feedback,
            adaptive_witness_budget=adaptive_witness_budget,
            memory_framework_config=framework,
        )
    feedback_family = prepared.feedback_family if prepared is not None else infer_monitor_family(query_text)
    resolved_mode = (
        prepared.resolved_mode
        if prepared is not None
        else infer_query_context_mode(
            query_text=query_text,
            lines=raw_lines,
            requested_mode=context_mode,
            state_aware=state_aware,
        )
    )
    skills = prepared.skills if prepared is not None else compile_memory_skills_from_lines(
        raw_lines,
        query_text,
        feedback=skill_feedback,
        adaptive_witness_budget=adaptive_witness_budget,
        memory_framework_config=framework,
    )
    feedback_family = prepared.feedback_family if prepared is not None else infer_monitor_family(query_text)
    feedback_phase = prepared.feedback_phase if prepared is not None else infer_monitor_phase(query_text, raw_lines)
    skill_lines = (
        synthesize_skill_lines(
            skills,
            family=feedback_family,
            query_text=query_text,
            memory_framework_config=framework,
        )
        if not disable_skill_summary
        else []
    )
    augmented_lines = skill_lines + raw_lines
    base_segments = build_trust_segments_from_lines(
        augmented_lines,
        current_step=current_step,
        default_source_id=default_source_id,
    )
    query_tokens = _normalize_query_tokens(query_text)
    if not query_tokens:
        return base_segments
    state_task_family = _infer_state_task_family(query_text) if resolved_mode == "state" else None
    goal_slots = (
        prepared.goal_slots
        if (prepared is not None and resolved_mode == "slot")
        else (parse_goal_slots(query_text) if resolved_mode == "slot" else GoalSlots())
    )
    slot_scores = (
        _slot_line_scores(raw_lines, goal_slots)
        if resolved_mode == "slot" and (goal_slots.target_objects or goal_slots.target_receptacle)
        else [0.0] * len(raw_lines)
    )
    location_witnesses = skills.witness_lines_by_skill.get("location", ()) if framework.enable_egrc else ()
    progress_witnesses = skills.witness_lines_by_skill.get("progress", ()) if framework.enable_egrc else ()
    state_witnesses = skills.witness_lines_by_skill.get("state", ()) if framework.enable_egrc else ()
    location_schemas = skills.schema_witnesses_by_skill.get("location", ()) if (framework.enable_ema and framework.enable_egrc) else ()
    progress_schemas = skills.schema_witnesses_by_skill.get("progress", ()) if (framework.enable_ema and framework.enable_egrc) else ()
    state_schemas = skills.schema_witnesses_by_skill.get("state", ()) if (framework.enable_ema and framework.enable_egrc) else ()
    location_roles = skills.witness_roles_by_skill.get("location", {}) if framework.enable_egrc else {}
    progress_roles = skills.witness_roles_by_skill.get("progress", {}) if framework.enable_egrc else {}
    state_roles = skills.witness_roles_by_skill.get("state", {}) if framework.enable_egrc else {}

    conditioned_segments: List[SegmentTrustMetadata] = []
    for index, segment in enumerate(base_segments):
        if index < len(skill_lines):
            skill_match = SKILL_LINE_PATTERN.match(segment.text.strip())
            skill_kind = skill_match.group("kind").lower() if skill_match else None
            skill_reliability = _skill_feedback_value(skill_feedback, "reliability_by_skill", skill_kind, feedback_family, feedback_phase) if skill_kind else 0.0
            skill_support = _skill_feedback_value(skill_feedback, "support_by_skill", skill_kind, feedback_family, feedback_phase) if skill_kind else 0.0
            skill_conflict = _skill_feedback_value(skill_feedback, "conflict_by_skill", skill_kind, feedback_family, feedback_phase) if skill_kind else 0.0
            skill_utility = _skill_feedback_value(skill_feedback, "utility_ema_by_skill", skill_kind, feedback_family, feedback_phase) if skill_kind else 0.0
            skill_uncertainty = _skill_feedback_value(skill_feedback, "uncertainty_by_skill", skill_kind, feedback_family, feedback_phase) if skill_kind else 0.0
            conditioned_segments.append(
                SegmentTrustMetadata(
                    text=segment.text,
                    step=segment.step,
                    source_id=segment.source_id,
                    source_trust=max(
                        segment.source_trust,
                        min(
                            1.0,
                            0.88 + 0.04 * framework.enable_ema
                            + (
                                0.08 * skill_reliability + 0.04 * skill_support + 0.08 * skill_utility - 0.14 * skill_uncertainty
                                if framework.enable_egrc
                                else 0.0
                            ),
                        ),
                    ),
                    support_count=max(segment.support_count, 2 + int(round(skill_support))),
                    contradiction_count=segment.contradiction_count,
                    suspicious_score=segment.suspicious_score,
                    salience=max(
                        segment.salience,
                        min(
                            1.0,
                            0.92
                            + (
                                0.05 * skill_reliability + 0.03 * skill_support + 0.05 * skill_utility - 0.10 * skill_uncertainty
                                if framework.enable_egrc
                                else 0.0
                            ),
                        ),
                    ),
                    query_relevance=1.0,
                )
            )
            continue
        signature = _extract_fact_signature(segment.text)
        lowered = _normalize_fact_token(_strip_xml_tags(segment.text))
        overlap = sum(token in lowered.split() for token in query_tokens)
        query_relevance = min(1.0, overlap / max(1, len(query_tokens)))
        salience = segment.salience
        source_trust = segment.source_trust
        state_score = _state_line_score(segment.text, state_task_family)
        raw_index = index - len(skill_lines)
        slot_score = slot_scores[raw_index] if raw_index < len(slot_scores) else 0.0
        line_slot_kinds = classify_slot_kinds(segment.text, goal_slots) if (goal_slots.target_objects or goal_slots.target_receptacle) else set()

        if signature is not None:
            subject, _ = signature
            subject_tokens = subject.split()
            if any(token in query_tokens for token in subject_tokens):
                query_relevance = max(query_relevance, 1.0)
            elif query_relevance == 0.0:
                query_relevance = 0.0
        elif query_relevance == 0.0:
            query_relevance = 0.2

        location_skill_conf = float(skills.confidence_by_skill.get("location", 0.0))
        progress_skill_conf = float(skills.confidence_by_skill.get("progress", 0.0))
        state_skill_conf = float(skills.confidence_by_skill.get("state", 0.0))
        is_location_witness = preserve_skill_witnesses and _line_matches_skill_witness(segment.text, location_witnesses)
        is_progress_witness = preserve_skill_witnesses and _line_matches_skill_witness(segment.text, progress_witnesses)
        is_state_witness = preserve_skill_witnesses and _line_matches_skill_witness(segment.text, state_witnesses)
        location_schema_signal = _schema_line_signal(segment.text, location_schemas) if preserve_schema_witnesses else 0.0
        progress_schema_signal = _schema_line_signal(segment.text, progress_schemas) if preserve_schema_witnesses else 0.0
        state_schema_signal = _schema_line_signal(segment.text, state_schemas) if preserve_schema_witnesses else 0.0
        location_schema_verify = _schema_verification_signal(segment.text, location_schemas) if preserve_schema_witnesses else 0.0
        progress_schema_verify = _schema_verification_signal(segment.text, progress_schemas) if preserve_schema_witnesses else 0.0
        state_schema_verify = _schema_verification_signal(segment.text, state_schemas) if preserve_schema_witnesses else 0.0
        witness_utility = _witness_feedback_value(skill_feedback, segment.text)
        location_role = location_roles.get(_normalized_line_key(segment.text), "generic")
        progress_role = progress_roles.get(_normalized_line_key(segment.text), "generic")
        state_role = state_roles.get(_normalized_line_key(segment.text), "generic")
        location_bundle = _decoupled_skill_feedback_bundle(
            skill_feedback,
            "location",
            location_role,
            feedback_family,
            feedback_phase,
            decouple_role_signals=decouple_role_signals,
        )
        progress_bundle = _decoupled_skill_feedback_bundle(
            skill_feedback,
            "progress",
            progress_role,
            feedback_family,
            feedback_phase,
            decouple_role_signals=decouple_role_signals,
        )
        state_bundle = _decoupled_skill_feedback_bundle(
            skill_feedback,
            "state",
            state_role,
            feedback_family,
            feedback_phase,
            decouple_role_signals=decouple_role_signals,
        )
        location_bias = location_bundle["bias"]
        progress_bias = progress_bundle["bias"]
        state_bias = state_bundle["bias"]
        location_reliability = location_bundle["reliability"]
        progress_reliability = progress_bundle["reliability"]
        state_reliability = state_bundle["reliability"]
        location_support = location_bundle["support"]
        progress_support = progress_bundle["support"]
        state_support = state_bundle["support"]
        location_conflict = location_bundle["conflict"]
        progress_conflict = progress_bundle["conflict"]
        state_conflict = state_bundle["conflict"]
        location_rescue = location_bundle["rescue"]
        progress_rescue = progress_bundle["rescue"]
        state_rescue = state_bundle["rescue"]
        location_utility = location_bundle["utility"]
        progress_utility = progress_bundle["utility"]
        state_utility = state_bundle["utility"]
        location_uncertainty = location_bundle["uncertainty"]
        progress_uncertainty = progress_bundle["uncertainty"]
        state_uncertainty = state_bundle["uncertainty"]
        location_role_utility = location_bundle["role_score"]
        progress_role_utility = progress_bundle["role_score"]
        state_role_utility = state_bundle["role_score"]
        adjusted_support_count = segment.support_count

        location_signal_applicable = bool(
            slot_score > 0.0
            or is_location_witness
            or location_schema_signal > 0.0
            or location_schema_verify > 0.0
        )
        if framework.enable_egrc and location_bias > 0.0 and location_signal_applicable:
            location_bias_targets = _skill_bias_targets(location_bias)
            query_relevance = max(query_relevance, location_bias_targets["query_relevance"])
            salience = max(salience, location_bias_targets["salience"])
            source_trust = max(source_trust, location_bias_targets["source_trust"])
        if framework.enable_egrc and location_signal_applicable:
            location_targets = _skill_signal_targets(
                skill_conf=location_skill_conf,
                reliability=location_reliability,
                support=location_support,
                rescue=location_rescue,
                utility=location_utility,
                uncertainty=location_uncertainty,
                witness_mode=False,
                adaptive_skill_mix=adaptive_skill_mix,
            )
            query_relevance = max(query_relevance, location_targets["query_relevance"])
            salience = max(salience, location_targets["salience"])
            source_trust = max(source_trust, location_targets["source_trust"])
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(location_targets["support_bonus"])),
            )
        if framework.enable_egrc and is_location_witness:
            location_witness_targets = _skill_signal_targets(
                skill_conf=location_skill_conf,
                reliability=location_reliability,
                support=location_support,
                rescue=location_rescue,
                utility=location_utility,
                uncertainty=location_uncertainty,
                role_utility=location_role_utility,
                witness_utility=witness_utility,
                witness_mode=True,
                adaptive_skill_mix=adaptive_skill_mix,
            )
            query_relevance = max(
                query_relevance,
                location_witness_targets["query_relevance"],
            )
            salience = max(
                salience,
                location_witness_targets["salience"],
            )
            source_trust = max(
                source_trust,
                location_witness_targets["source_trust"],
            )
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + 1 + int(round(location_witness_targets["support_bonus"])),
            )
        if framework.enable_egrc and location_schema_signal > 0.0:
            query_relevance = min(
                1.0,
                query_relevance
                + 0.04
                + 0.10 * location_schema_signal
                + 0.03 * location_support
                + 0.03 * max(0.0, location_utility),
            )
            salience = min(
                1.0,
                salience + 0.04 + 0.09 * location_schema_signal + 0.03 * location_support,
            )
            source_trust = min(
                1.0,
                source_trust + 0.04 + 0.08 * location_schema_signal + 0.03 * location_reliability,
            )
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(1.0 + 1.4 * location_schema_signal)),
            )
        elif framework.enable_egrc and location_schema_signal < 0.0:
            schema_penalty = abs(location_schema_signal)
            query_relevance = max(0.0, query_relevance - (0.04 + 0.10 * schema_penalty))
            salience = max(0.0, salience - (0.04 + 0.08 * schema_penalty))
            source_trust = max(0.0, source_trust - (0.04 + 0.08 * schema_penalty))
            adjusted_support_count = max(
                1,
                adjusted_support_count - int(round(1.0 + 1.2 * schema_penalty)),
            )
        if framework.enable_egrc and location_schema_verify > 0.0:
            query_relevance = max(query_relevance, min(0.50 + 0.24 * location_schema_verify, 0.86))
            salience = max(salience, min(0.44 + 0.18 * location_schema_verify, 0.80))
            source_trust = max(source_trust, min(0.46 + 0.08 * location_schema_verify, 0.72))
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(location_schema_verify)),
            )

        if framework.enable_egrc and progress_bias > 0.0 and bool(line_slot_kinds.intersection({"inventory", "progress"})):
            progress_bias_targets = _skill_bias_targets(progress_bias)
            query_relevance = max(query_relevance, progress_bias_targets["query_relevance"])
            salience = max(salience, progress_bias_targets["salience"])
            source_trust = max(source_trust, progress_bias_targets["source_trust"])
        if framework.enable_egrc and bool(line_slot_kinds.intersection({"inventory", "progress"})):
            progress_targets = _skill_signal_targets(
                skill_conf=progress_skill_conf,
                reliability=progress_reliability,
                support=progress_support,
                rescue=progress_rescue,
                utility=progress_utility,
                uncertainty=progress_uncertainty,
                witness_mode=False,
                adaptive_skill_mix=adaptive_skill_mix,
            )
            query_relevance = max(query_relevance, progress_targets["query_relevance"])
            salience = max(salience, progress_targets["salience"])
            source_trust = max(source_trust, progress_targets["source_trust"])
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(progress_targets["support_bonus"])),
            )
        if framework.enable_egrc and is_progress_witness:
            progress_witness_targets = _skill_signal_targets(
                skill_conf=progress_skill_conf,
                reliability=progress_reliability,
                support=progress_support,
                rescue=progress_rescue,
                utility=progress_utility,
                uncertainty=progress_uncertainty,
                role_utility=progress_role_utility,
                witness_utility=witness_utility,
                witness_mode=True,
                adaptive_skill_mix=adaptive_skill_mix,
            )
            query_relevance = max(
                query_relevance,
                progress_witness_targets["query_relevance"],
            )
            salience = max(
                salience,
                progress_witness_targets["salience"],
            )
            source_trust = max(
                source_trust,
                progress_witness_targets["source_trust"],
            )
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + 1 + int(round(progress_witness_targets["support_bonus"])),
            )
        if framework.enable_egrc and progress_schema_signal > 0.0:
            query_relevance = min(
                1.0,
                query_relevance
                + 0.05
                + 0.10 * progress_schema_signal
                + 0.03 * progress_support
                + 0.03 * max(0.0, progress_utility),
            )
            salience = min(
                1.0,
                salience + 0.04 + 0.09 * progress_schema_signal + 0.03 * progress_support,
            )
            source_trust = min(
                1.0,
                source_trust + 0.04 + 0.08 * progress_schema_signal + 0.03 * progress_reliability,
            )
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(1.0 + 1.4 * progress_schema_signal)),
            )
        elif framework.enable_egrc and progress_schema_signal < 0.0:
            schema_penalty = abs(progress_schema_signal)
            query_relevance = max(0.0, query_relevance - (0.05 + 0.10 * schema_penalty))
            salience = max(0.0, salience - (0.04 + 0.08 * schema_penalty))
            source_trust = max(0.0, source_trust - (0.04 + 0.08 * schema_penalty))
            adjusted_support_count = max(
                1,
                adjusted_support_count - int(round(1.0 + 1.2 * schema_penalty)),
            )
        if framework.enable_egrc and progress_schema_verify > 0.0:
            query_relevance = max(query_relevance, min(0.50 + 0.24 * progress_schema_verify, 0.86))
            salience = max(salience, min(0.44 + 0.18 * progress_schema_verify, 0.80))
            source_trust = max(source_trust, min(0.46 + 0.08 * progress_schema_verify, 0.72))
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(progress_schema_verify)),
            )

        if framework.enable_egrc and state_bias > 0.0 and (state_score > 0.0 or "receptacle" in line_slot_kinds):
            state_bias_targets = _skill_bias_targets(state_bias)
            query_relevance = max(query_relevance, state_bias_targets["query_relevance"])
            salience = max(salience, state_bias_targets["salience"])
            source_trust = max(source_trust, state_bias_targets["source_trust"])
        if framework.enable_egrc and (state_score > 0.0 or "receptacle" in line_slot_kinds):
            state_targets = _skill_signal_targets(
                skill_conf=state_skill_conf,
                reliability=state_reliability,
                support=state_support,
                rescue=state_rescue,
                utility=state_utility,
                uncertainty=state_uncertainty,
                witness_mode=False,
                adaptive_skill_mix=adaptive_skill_mix,
            )
            query_relevance = max(query_relevance, state_targets["query_relevance"])
            salience = max(salience, state_targets["salience"])
            source_trust = max(source_trust, state_targets["source_trust"])
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(state_targets["support_bonus"])),
            )
        if framework.enable_egrc and is_state_witness:
            state_witness_targets = _skill_signal_targets(
                skill_conf=state_skill_conf,
                reliability=state_reliability,
                support=state_support,
                rescue=state_rescue,
                utility=state_utility,
                uncertainty=state_uncertainty,
                role_utility=state_role_utility,
                witness_utility=witness_utility,
                witness_mode=True,
                adaptive_skill_mix=adaptive_skill_mix,
            )
            query_relevance = max(
                query_relevance,
                state_witness_targets["query_relevance"],
            )
            salience = max(
                salience,
                state_witness_targets["salience"],
            )
            source_trust = max(
                source_trust,
                state_witness_targets["source_trust"],
            )
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + 1 + int(round(state_witness_targets["support_bonus"])),
            )
        if framework.enable_egrc and state_schema_signal > 0.0:
            query_relevance = min(
                1.0,
                query_relevance
                + 0.04
                + 0.09 * state_schema_signal
                + 0.03 * state_support
                + 0.03 * max(0.0, state_utility),
            )
            salience = min(
                1.0,
                salience + 0.04 + 0.08 * state_schema_signal + 0.03 * state_support,
            )
            source_trust = min(
                1.0,
                source_trust + 0.04 + 0.08 * state_schema_signal + 0.03 * state_reliability,
            )
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(1.0 + 1.4 * state_schema_signal)),
            )
        elif framework.enable_egrc and state_schema_signal < 0.0:
            schema_penalty = abs(state_schema_signal)
            query_relevance = max(0.0, query_relevance - (0.04 + 0.09 * schema_penalty))
            salience = max(0.0, salience - (0.04 + 0.08 * schema_penalty))
            source_trust = max(0.0, source_trust - (0.04 + 0.08 * schema_penalty))
            adjusted_support_count = max(
                1,
                adjusted_support_count - int(round(1.0 + 1.2 * schema_penalty)),
            )
        if framework.enable_egrc and state_schema_verify > 0.0:
            query_relevance = max(query_relevance, min(0.50 + 0.24 * state_schema_verify, 0.84))
            salience = max(salience, min(0.44 + 0.18 * state_schema_verify, 0.78))
            source_trust = max(source_trust, min(0.46 + 0.08 * state_schema_verify, 0.70))
            adjusted_support_count = max(
                adjusted_support_count,
                segment.support_count + int(round(state_schema_verify)),
            )

        if resolved_mode == "state":
            if state_score >= 0.8:
                query_relevance = max(query_relevance, 0.95)
                salience = max(salience, 0.9)
                source_trust = max(source_trust, 0.9)
            elif state_score >= 0.5:
                query_relevance = max(query_relevance, 0.75)
                salience = max(salience, 0.75)
                source_trust = max(source_trust, 0.75)
            elif signature is not None or any(token in lowered for token in DISTRACTOR_MARKERS):
                if state_conflict > 0.0 or state_rescue > 0.0:
                    query_relevance = max(query_relevance, min(0.40 + 0.10 * state_rescue, 0.65))
                    salience = max(salience, min(0.35 + 0.10 * state_rescue, 0.60))
                    source_trust = max(source_trust, min(0.40 + 0.08 * state_rescue, 0.62))
                else:
                    query_relevance = min(query_relevance, 0.05)
                    salience = min(salience, 0.1)
                    source_trust = min(source_trust, 0.35)

        if resolved_mode == "slot":
            combined_slot_score = max(slot_score, state_score)
            if combined_slot_score >= 0.9:
                query_relevance = max(query_relevance, 0.98)
                salience = max(salience, 0.95)
                source_trust = max(source_trust, 0.92)
            elif combined_slot_score >= 0.7:
                query_relevance = max(query_relevance, 0.85)
                salience = max(salience, 0.85)
                source_trust = max(source_trust, 0.85)
            elif combined_slot_score >= 0.5:
                query_relevance = max(query_relevance, 0.65)
                salience = max(salience, 0.65)
                source_trust = max(source_trust, 0.75)
            elif signature is not None or any(token in lowered for token in DISTRACTOR_MARKERS):
                irrelevant_slot_fact = False
                if signature is not None:
                    subject, value = signature
                    irrelevant_slot_fact = (
                        _goal_object_for_entity(subject, goal_slots) is None
                        and not (
                            goal_slots.target_receptacle
                            and (
                                _entities_match(subject, goal_slots.target_receptacle)
                                or _entities_match(value, goal_slots.target_receptacle)
                            )
                        )
                    )
                if irrelevant_slot_fact and not line_slot_kinds.intersection({"inventory", "progress"}):
                    query_relevance = min(query_relevance, 0.05)
                    salience = min(salience, 0.1)
                    source_trust = min(source_trust, 0.35)
                elif location_conflict > 0.0 or progress_conflict > 0.0 or state_conflict > 0.0 or location_rescue > 0.0:
                    rescue = max(location_rescue, progress_rescue, state_rescue)
                    query_relevance = max(query_relevance, min(0.38 + 0.12 * rescue, 0.68))
                    salience = max(salience, min(0.35 + 0.10 * rescue, 0.60))
                    source_trust = max(source_trust, min(0.38 + 0.08 * rescue, 0.60))
                else:
                    query_relevance = min(query_relevance, 0.05)
                    salience = min(salience, 0.1)
                    source_trust = min(source_trust, 0.35)

        conditioned_segments.append(
            SegmentTrustMetadata(
                text=segment.text,
                step=segment.step,
                source_id=segment.source_id,
                source_trust=source_trust,
                support_count=adjusted_support_count,
                contradiction_count=segment.contradiction_count,
                suspicious_score=segment.suspicious_score,
                salience=max(salience, 0.7) if query_relevance >= 0.8 else min(salience, 0.3),
                query_relevance=query_relevance,
            )
        )
    return conditioned_segments
