# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
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

from typing import List, Tuple, Dict, Union, Any, Optional
from collections import defaultdict
import math
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from agentocr.trust_policy import (
    auto_update_memory_skill_feedback,
    build_search_task_card_context,
    MemorySkillFeedback,
    TrustCalibratedRenderPolicy,
    TrustPolicyConfig,
)
from omegaconf import OmegaConf
import time

def _agentocr_boot_debug(message: str) -> None:
    if os.environ.get("AGENTOCR_BOOT_DEBUG") == "1":
        print(f"[AGENTOCR_BOOT_DEBUG] {message}", flush=True)


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


def _query_text_from_task(task: Any) -> str:
    if isinstance(task, dict):
        for key in ("question", "task_description", "prompt", "goal"):
            if key in task and task[key]:
                return str(task[key])
    return str(task)


def _build_default_trust_policy_metrics() -> Dict[str, float]:
    return {
        "trust_policy/query_mode": 0.0,
        "trust_policy/state_mode": 0.0,
        "trust_policy/slot_mode": 0.0,
    }


def _build_initial_trust_policy_feedbacks(batch_size: int) -> List[Optional[MemorySkillFeedback]]:
    return [None for _ in range(batch_size)]


def _build_initial_trust_policy_metric_histories(batch_size: int) -> List[List[Dict[str, Any]]]:
    return [[] for _ in range(batch_size)]


def _strip_embedded_task_text_for_memory(text: Any) -> str:
    cleaned_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if "your task is to:" in lowered:
            stripped = stripped[:lowered.index("your task is to:")].strip()
            lowered = stripped.lower()
        if "task:" in lowered:
            stripped = stripped[:lowered.index("task:")].strip()
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def _trust_metric_value(metrics: Dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _typed_evidence_counts(metrics: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    location_evidence = _trust_metric_value(metrics, "trust_policy/evidence/location_pointer_evidence_count")
    location_kept = _trust_metric_value(metrics, "trust_policy/evidence/location_pointer_kept_count")
    if location_evidence <= 0.0 and location_kept <= 0.0:
        location_evidence = (
            _trust_metric_value(metrics, "trust_policy/slot/location_evidence_count")
            + _trust_metric_value(metrics, "trust_policy/slot/receptacle_evidence_count")
        )
        location_kept = (
            _trust_metric_value(metrics, "trust_policy/slot/location_kept_count")
            + _trust_metric_value(metrics, "trust_policy/slot/receptacle_kept_count")
        )

    inventory_evidence = _trust_metric_value(metrics, "trust_policy/evidence/inventory_pointer_evidence_count")
    inventory_kept = _trust_metric_value(metrics, "trust_policy/evidence/inventory_pointer_kept_count")
    if inventory_evidence <= 0.0 and inventory_kept <= 0.0:
        inventory_evidence = _trust_metric_value(metrics, "trust_policy/slot/inventory_evidence_count")
        inventory_kept = _trust_metric_value(metrics, "trust_policy/slot/inventory_kept_count")

    progress_evidence = _trust_metric_value(metrics, "trust_policy/evidence/progress_witness_evidence_count")
    progress_kept = _trust_metric_value(metrics, "trust_policy/evidence/progress_witness_kept_count")
    if progress_evidence <= 0.0 and progress_kept <= 0.0:
        progress_evidence = _trust_metric_value(metrics, "trust_policy/slot/progress_evidence_count")
        progress_kept = _trust_metric_value(metrics, "trust_policy/slot/progress_kept_count")

    state_evidence = _trust_metric_value(metrics, "trust_policy/evidence/state_witness_evidence_count")
    state_kept = _trust_metric_value(metrics, "trust_policy/evidence/state_witness_kept_count")
    if state_evidence <= 0.0 and state_kept <= 0.0:
        state_evidence = max(
            _trust_metric_value(metrics, "trust_policy/state/achieved_state_evidence"),
            _trust_metric_value(metrics, "trust_policy/state/state_transition_raw_witness_evidence"),
            _trust_metric_value(metrics, "trust_policy/state/state_device_evidence"),
            _trust_metric_value(metrics, "trust_policy/skill/state_present"),
        )
        state_kept = max(
            _trust_metric_value(metrics, "trust_policy/state/achieved_state_kept"),
            _trust_metric_value(metrics, "trust_policy/state/state_transition_raw_witness_kept"),
            _trust_metric_value(metrics, "trust_policy/state/state_device_kept"),
        )

    return {
        "location_pointer": (location_evidence, location_kept),
        "inventory_pointer": (inventory_evidence, inventory_kept),
        "progress_witness": (progress_evidence, progress_kept),
        "state_witness": (state_evidence, state_kept),
    }


def _typed_evidence_missing_counts(metrics: Dict[str, Any]) -> Dict[str, float]:
    typed_counts = _typed_evidence_counts(metrics)
    missing_counts: Dict[str, float] = {}
    for family, (evidence_count, kept_count) in typed_counts.items():
        grounded_miss_count = _trust_metric_value(
            metrics,
            f"trust_policy/evidence/{family}_grounded_miss_count",
        )
        if grounded_miss_count > 0.0:
            missing_counts[family] = grounded_miss_count
            continue
        grounded_evidence_count = _trust_metric_value(
            metrics,
            f"trust_policy/evidence/{family}_grounded_evidence_count",
        )
        grounded_kept_count = _trust_metric_value(
            metrics,
            f"trust_policy/evidence/{family}_grounded_kept_count",
        )
        if grounded_evidence_count > 0.0:
            missing_counts[family] = max(0.0, grounded_evidence_count - grounded_kept_count)
            continue
        missing_counts[family] = max(0.0, evidence_count - kept_count)
    return missing_counts


def _schema_refresh_failure_kinds(metrics: Dict[str, Any]) -> List[str]:
    failed_kinds: List[str] = []
    for kind in ("location", "progress", "state"):
        applicable = _trust_metric_value(metrics, f"trust_policy/schema/{kind}_refresh_applicable")
        if applicable <= 0.0:
            continue
        refresh_success = _trust_metric_value(metrics, f"trust_policy/schema/{kind}_refresh_success")
        fresh_selection = _trust_metric_value(metrics, f"trust_policy/schema/{kind}_fresh_selection")
        stale_activation = _trust_metric_value(metrics, f"trust_policy/schema/{kind}_stale_activation")
        if refresh_success <= 0.0 or fresh_selection <= 0.0 or stale_activation >= 0.50:
            failed_kinds.append(kind)
    return failed_kinds


def _infer_failure_families(metrics: Dict[str, Any]) -> List[str]:
    typed_counts = _typed_evidence_counts(metrics)
    missing_counts = _typed_evidence_missing_counts(metrics)
    failure_families: List[str] = []
    if typed_counts["location_pointer"][0] > 0.0 and missing_counts["location_pointer"] > 0.0:
        failure_families.append("missing_location_pointer")
    if typed_counts["inventory_pointer"][0] > 0.0 and missing_counts["inventory_pointer"] > 0.0:
        failure_families.append("missing_inventory_pointer")
    if typed_counts["progress_witness"][0] > 0.0 and missing_counts["progress_witness"] > 0.0:
        failure_families.append("missing_progress_witness")
    if typed_counts["state_witness"][0] > 0.0 and missing_counts["state_witness"] > 0.0:
        failure_families.append("missing_state_witness")
    for kind in _schema_refresh_failure_kinds(metrics):
        failure_families.append(f"stale_schema_not_refreshed:{kind}")
    return failure_families


def _infer_outcome_failure_families(
    metrics: Dict[str, Any],
    outcome_utility: Optional[float] = None,
) -> List[str]:
    failure_families = list(_infer_failure_families(metrics))
    explicit_action_error = _trust_metric_value(
        metrics,
        "trust_policy/mechanism/action_error_without_memory_fault",
    )
    if explicit_action_error > 0.0:
        failure_families.append("action_error_without_memory_fault")
    elif (
        outcome_utility is not None
        and float(outcome_utility) < 0.0
        and not failure_families
    ):
        failure_families.append("action_error_without_memory_fault")
    return list(dict.fromkeys(failure_families))


def _annotate_outcome_failure_metrics(
    metrics: Dict[str, Any],
    *,
    outcome_utility: float,
) -> Dict[str, Any]:
    annotated = dict(metrics)
    failure_families = _infer_failure_families(annotated)
    action_error_without_memory_fault = float(float(outcome_utility) < 0.0 and not failure_families)
    annotated["trust_policy/mechanism/outcome_utility"] = float(outcome_utility)
    annotated["trust_policy/mechanism/action_error_without_memory_fault"] = action_error_without_memory_fault
    return annotated


def _infer_failed_skill_kinds(metrics: Dict[str, Any]) -> List[str]:
    failed_skill_kinds: List[str] = []
    for family in _infer_failure_families(metrics):
        if family == "missing_location_pointer":
            failed_skill_kinds.append("location")
        elif family in {"missing_inventory_pointer", "missing_progress_witness"}:
            failed_skill_kinds.append("progress")
        elif family == "missing_state_witness":
            failed_skill_kinds.append("state")
        elif family.startswith("stale_schema_not_refreshed:"):
            failed_skill_kinds.append(family.split(":", 1)[1])
    return list(dict.fromkeys(kind for kind in failed_skill_kinds if kind))


def _infer_observed_skill_kinds(metrics: Dict[str, Any]) -> List[str]:
    typed_counts = _typed_evidence_counts(metrics)
    observed_skill_kinds: List[str] = []
    if typed_counts["location_pointer"][0] > 0.0:
        observed_skill_kinds.append("location")
    if typed_counts["progress_witness"][0] > 0.0 or typed_counts["inventory_pointer"][0] > 0.0:
        observed_skill_kinds.append("progress")
    if typed_counts["state_witness"][0] > 0.0:
        observed_skill_kinds.append("state")
    return observed_skill_kinds


def _infer_support_by_skill(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "location": min(1.0, float(metrics.get("trust_policy/skill/location_support_count", 0.0)) / 2.0),
        "progress": min(1.0, float(metrics.get("trust_policy/skill/progress_support_count", 0.0)) / 3.0),
        "state": min(1.0, float(metrics.get("trust_policy/skill/state_support_count", 0.0)) / 2.0),
    }


def _infer_conflict_by_skill(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "location": min(1.0, float(metrics.get("trust_policy/skill/location_conflict_count", 0.0))),
        "progress": min(1.0, float(metrics.get("trust_policy/skill/progress_conflict_count", 0.0))),
        "state": min(1.0, float(metrics.get("trust_policy/skill/state_conflict_count", 0.0))),
    }


def _infer_retained_ratio_by_skill(metrics: Dict[str, Any]) -> Dict[str, float]:
    typed_counts = _typed_evidence_counts(metrics)
    location_evidence, location_kept = typed_counts["location_pointer"]
    inventory_evidence, inventory_kept = typed_counts["inventory_pointer"]
    progress_evidence, progress_kept = typed_counts["progress_witness"]
    state_evidence, state_kept = typed_counts["state_witness"]
    return {
        "location": (location_kept / location_evidence) if location_evidence > 0.0 else 0.0,
        "progress": (
            (progress_kept + inventory_kept) / (progress_evidence + inventory_evidence)
            if (progress_evidence + inventory_evidence) > 0.0
            else 0.0
        ),
        "state": (state_kept / state_evidence) if state_evidence > 0.0 else 0.0,
    }


def _failure_family_boosts(
    metrics: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    support_boosts: Dict[str, float] = {}
    conflict_boosts: Dict[str, float] = {}
    role_support_boosts: Dict[str, Dict[str, float]] = {}

    def _boost_roles(kind: str, roles: Dict[str, float]) -> None:
        role_support_boosts.setdefault(kind, {})
        for role, value in roles.items():
            role_support_boosts[kind][role] = max(role_support_boosts[kind].get(role, 0.0), float(value))

    for family in _infer_failure_families(metrics):
        if family == "missing_location_pointer":
            support_boosts["location"] = max(support_boosts.get("location", 0.0), 1.0)
            _boost_roles(
                "location",
                {
                    "target_fact": 1.0,
                    "agent_location": 0.9,
                    "state_device_fact": 0.9,
                    "anchor": 0.6,
                },
            )
        elif family == "missing_inventory_pointer":
            support_boosts["progress"] = max(support_boosts.get("progress", 0.0), 1.0)
            _boost_roles(
                "progress",
                {
                    "holding": 1.0,
                    "inventory": 1.0,
                },
            )
        elif family == "missing_progress_witness":
            support_boosts["progress"] = max(support_boosts.get("progress", 0.0), 0.9)
            _boost_roles(
                "progress",
                {
                    "placed": 1.0,
                    "holding": 0.7,
                    "located": 0.6,
                },
            )
        elif family == "missing_state_witness":
            support_boosts["state"] = max(support_boosts.get("state", 0.0), 1.0)
            _boost_roles(
                "state",
                {
                    "light_state": 1.0,
                    "object_state": 1.0,
                    "receptacle_state": 0.8,
                },
            )
        elif family.startswith("stale_schema_not_refreshed:"):
            kind = family.split(":", 1)[1]
            support_boosts[kind] = max(support_boosts.get(kind, 0.0), 0.85)
            conflict_boosts[kind] = max(conflict_boosts.get(kind, 0.0), 0.85)
            role_map = {
                "location": {
                    "target_fact": 1.0,
                    "agent_location": 0.8,
                    "anchor": 0.6,
                },
                "progress": {
                    "placed": 1.0,
                    "holding": 0.8,
                    "inventory": 0.6,
                },
                "state": {
                    "light_state": 1.0,
                    "object_state": 1.0,
                    "receptacle_state": 0.8,
                },
            }.get(kind, {})
            if role_map:
                _boost_roles(kind, role_map)
    return support_boosts, conflict_boosts, role_support_boosts


def _failure_family_is_applicable(metrics: Dict[str, Any], family: str) -> bool:
    typed_counts = _typed_evidence_counts(metrics)
    if family == "missing_location_pointer":
        return typed_counts["location_pointer"][0] > 0.0
    if family == "missing_inventory_pointer":
        return typed_counts["inventory_pointer"][0] > 0.0
    if family == "missing_progress_witness":
        return typed_counts["progress_witness"][0] > 0.0
    if family == "missing_state_witness":
        return typed_counts["state_witness"][0] > 0.0
    if family.startswith("stale_schema_not_refreshed:"):
        kind = family.split(":", 1)[1]
        return _trust_metric_value(metrics, f"trust_policy/schema/{kind}_refresh_applicable") > 0.0
    return False


def _infer_outcome_utility_signal(reward: Any, done: Any, info: Dict[str, Any]) -> float:
    reward_value = float(reward)
    utility = float(np.tanh(reward_value))
    if bool(done):
        terminal_success = None
        if "won" in info:
            terminal_success = float(info.get("won", 0.0))
        elif "success" in info:
            terminal_success = 1.0 if bool(info.get("success")) else 0.0
        if terminal_success is not None:
            if terminal_success > 0.0:
                utility = 1.0
            else:
                utility = -1.0
    return float(min(1.0, max(-1.0, utility)))


def _infer_utility_by_skill(metrics: Dict[str, Any], outcome_utility: float) -> Dict[str, float]:
    retained_ratio_by_skill = _infer_retained_ratio_by_skill(metrics)
    return {
        kind: max(-1.0, min(1.0, outcome_utility * (0.35 + 0.65 * retained_ratio)))
        for kind, retained_ratio in retained_ratio_by_skill.items()
        if retained_ratio > 0.0
    }


def _infer_witness_utility_by_line(metrics: Dict[str, Any], outcome_utility: float) -> Dict[str, float]:
    utility_by_line: Dict[str, float] = {}
    for kind in ("location", "progress", "state"):
        kept_lines = metrics.get(f"trust_policy/witness/{kind}_kept_lines", ())
        if not isinstance(kept_lines, (list, tuple)):
            continue
        kept_ratio = float(metrics.get(f"trust_policy/witness/{kind}_kept_count", 0.0))
        total_ratio = float(metrics.get(f"trust_policy/witness/{kind}_count", 0.0))
        witness_retention = (kept_ratio / total_ratio) if total_ratio > 0.0 else 0.0
        signal = max(-1.0, min(1.0, outcome_utility * (0.40 + 0.60 * witness_retention)))
        if abs(signal) <= 1e-8:
            continue
        for line in kept_lines:
            if line and str(line).strip():
                utility_by_line[str(line)] = signal
    return utility_by_line


def _infer_witness_role_utility_by_skill(metrics: Dict[str, Any], outcome_utility: float) -> Dict[str, Dict[str, float]]:
    role_utility_by_skill: Dict[str, Dict[str, float]] = {}
    for kind in ("location", "progress", "state"):
        role_counts = metrics.get(f"trust_policy/witness/{kind}_role_counts", {})
        kept_role_counts = metrics.get(f"trust_policy/witness/{kind}_kept_role_counts", {})
        if not isinstance(role_counts, dict) or not role_counts:
            continue
        aggregated: Dict[str, float] = {}
        total_role_count = sum(float(value) for value in role_counts.values())
        for role, total_count in role_counts.items():
            total_value = float(total_count)
            kept_value = float(kept_role_counts.get(role, 0.0)) if isinstance(kept_role_counts, dict) else 0.0
            if total_value <= 0.0 or kept_value <= 0.0:
                continue
            role_retention = kept_value / total_value
            role_presence = (total_value / total_role_count) if total_role_count > 0.0 else 0.0
            role_signal = max(
                -1.0,
                min(
                    1.0,
                    outcome_utility * (0.20 + 0.55 * role_retention + 0.25 * role_presence),
                ),
            )
            if abs(role_signal) <= 1e-8:
                continue
            aggregated[str(role)] = role_signal
        if aggregated:
            role_utility_by_skill[kind] = aggregated
    return role_utility_by_skill


def _infer_witness_role_support_by_skill(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    role_support_by_skill: Dict[str, Dict[str, float]] = {}
    for kind in ("location", "progress", "state"):
        role_counts = metrics.get(f"trust_policy/witness/{kind}_role_counts", {})
        kept_role_counts = metrics.get(f"trust_policy/witness/{kind}_kept_role_counts", {})
        skill_support = float(metrics.get(f"trust_policy/skill/{kind}_support_count", 0.0))
        if not isinstance(role_counts, dict) or not role_counts:
            continue
        role_supports: Dict[str, float] = {}
        for role, total_count in role_counts.items():
            total_value = float(total_count)
            kept_value = float(kept_role_counts.get(role, 0.0)) if isinstance(kept_role_counts, dict) else 0.0
            if total_value <= 0.0:
                continue
            kept_ratio = kept_value / total_value
            role_supports[str(role)] = min(1.0, kept_ratio * (0.40 + 0.30 * skill_support))
        if role_supports:
            role_support_by_skill[kind] = role_supports
    return role_support_by_skill


def _infer_witness_role_conflict_by_skill(metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    role_conflict_by_skill: Dict[str, Dict[str, float]] = {}
    for kind in ("location", "progress", "state"):
        role_counts = metrics.get(f"trust_policy/witness/{kind}_role_counts", {})
        kept_role_counts = metrics.get(f"trust_policy/witness/{kind}_kept_role_counts", {})
        skill_conflict = float(metrics.get(f"trust_policy/skill/{kind}_conflict_count", 0.0))
        if not isinstance(role_counts, dict) or not role_counts:
            continue
        total_witness_count = sum(float(value) for value in role_counts.values())
        role_conflicts: Dict[str, float] = {}
        for role, total_count in role_counts.items():
            total_value = float(total_count)
            kept_value = float(kept_role_counts.get(role, 0.0)) if isinstance(kept_role_counts, dict) else 0.0
            if total_value <= 0.0:
                continue
            dropped_ratio = max(0.0, total_value - kept_value) / total_value
            presence_ratio = (total_value / total_witness_count) if total_witness_count > 0.0 else 0.0
            role_conflicts[str(role)] = min(1.0, 0.25 * dropped_ratio + min(1.0, skill_conflict) * presence_ratio)
        if role_conflicts:
            role_conflict_by_skill[kind] = role_conflicts
    return role_conflict_by_skill


def _metrics_family(metrics: Dict[str, Any]) -> Optional[str]:
    family = metrics.get("trust_policy/family_target", metrics.get("trust_policy/family"))
    if family is None:
        return None
    family_key = str(family).strip()
    return family_key or None


def _metrics_phase(metrics: Dict[str, Any]) -> Optional[str]:
    phase = metrics.get("trust_policy/phase_target", metrics.get("trust_policy/phase"))
    if phase is None:
        return None
    phase_key = str(phase).strip()
    return phase_key or None


def _namespace_skill_signal_by_family(
    metrics: Dict[str, Any],
    by_skill: Dict[str, float],
) -> Dict[str, float]:
    family = _metrics_family(metrics)
    phase = _metrics_phase(metrics)
    if not family:
        return dict(by_skill)
    return {
        (f"{family}::{phase}::{kind}" if phase else f"{family}::{kind}"): float(value)
        for kind, value in by_skill.items()
    }


def _namespace_skill_kinds_by_family(
    metrics: Dict[str, Any],
    skill_kinds: List[str],
) -> List[str]:
    family = _metrics_family(metrics)
    phase = _metrics_phase(metrics)
    if not family:
        return list(skill_kinds)
    return [(f"{family}::{phase}::{kind}" if phase else f"{family}::{kind}") for kind in skill_kinds if kind]


def _namespace_role_signal_by_family(
    metrics: Dict[str, Any],
    by_skill: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    family = _metrics_family(metrics)
    phase = _metrics_phase(metrics)
    if not family:
        return {kind: dict(role_values) for kind, role_values in by_skill.items()}
    return {
        (f"{family}::{phase}::{kind}" if phase else f"{family}::{kind}"): {
            str(role): float(value)
            for role, value in role_values.items()
        }
        for kind, role_values in by_skill.items()
    }


def _merge_role_signal_maps(
    *maps: Dict[str, Dict[str, float]],
    clamp_signed: bool = False,
) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    for signal_map in maps:
        for kind, role_values in signal_map.items():
            if not isinstance(role_values, dict):
                continue
            merged.setdefault(kind, {})
            for role, value in role_values.items():
                next_value = float(merged[kind].get(role, 0.0)) + float(value)
                if clamp_signed:
                    next_value = max(-1.0, min(1.0, next_value))
                else:
                    next_value = max(0.0, min(1.0, next_value))
                if abs(next_value) <= 1e-8:
                    merged[kind].pop(role, None)
                else:
                    merged[kind][str(role)] = next_value
            if not merged[kind]:
                merged.pop(kind, None)
    return merged


def _infer_counterfactual_role_credit_by_skill(
    metrics: Dict[str, Any],
    outcome_utility: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    role_support_by_skill: Dict[str, Dict[str, float]] = {}
    role_conflict_by_skill: Dict[str, Dict[str, float]] = {}
    role_utility_by_skill: Dict[str, Dict[str, float]] = {}
    for kind in ("location", "progress", "state"):
        role_counts = metrics.get(f"trust_policy/witness/{kind}_role_counts", {})
        kept_role_counts = metrics.get(f"trust_policy/witness/{kind}_kept_role_counts", {})
        if not isinstance(role_counts, dict) or not role_counts:
            continue
        sanitized_totals = {
            str(role): max(0.0, float(total_count))
            for role, total_count in role_counts.items()
            if float(total_count) > 0.0
        }
        if not sanitized_totals:
            continue
        sanitized_kept = {
            role: min(
                sanitized_totals[role],
                max(0.0, float(kept_role_counts.get(role, 0.0))) if isinstance(kept_role_counts, dict) else 0.0,
            )
            for role in sanitized_totals
        }
        total_role_count = sum(sanitized_totals.values())
        total_kept_count = sum(sanitized_kept.values())
        if total_role_count <= 0.0:
            continue
        factual_skill_retention = total_kept_count / total_role_count
        for role, total_value in sanitized_totals.items():
            kept_value = sanitized_kept.get(role, 0.0)
            role_retention = kept_value / total_value if total_value > 0.0 else 0.0
            without_role_total = total_role_count - total_value
            without_role_kept = max(0.0, total_kept_count - kept_value)
            without_role_retention = (without_role_kept / without_role_total) if without_role_total > 0.0 else 0.0
            leave_one_out_gap = factual_skill_retention - without_role_retention
            selection_lift = role_retention - factual_skill_retention
            best_substitute_retention = 0.0
            for peer_role, peer_total in sanitized_totals.items():
                if peer_role == role or peer_total <= 0.0:
                    continue
                peer_kept = sanitized_kept.get(peer_role, 0.0)
                peer_retention = peer_kept / peer_total if peer_total > 0.0 else 0.0
                peer_capacity = math.sqrt(max(0.0, min(1.0, peer_total / total_value))) if total_value > 0.0 else 0.0
                substitute_retention = peer_retention * peer_capacity
                best_substitute_retention = max(best_substitute_retention, substitute_retention)
            swapped_kept = min(
                total_role_count,
                max(0.0, without_role_kept + best_substitute_retention * total_value),
            )
            swapped_retention = swapped_kept / total_role_count if total_role_count > 0.0 else 0.0
            swap_gap = factual_skill_retention - swapped_retention
            role_presence = total_value / total_role_count
            presence_weight = 0.35 + 0.65 * math.sqrt(max(0.0, min(1.0, role_presence)))
            margin = (
                0.50 * leave_one_out_gap
                + 0.20 * selection_lift
                + 0.30 * swap_gap
            ) * presence_weight
            if abs(margin) <= 1e-8:
                continue
            utility_signal = max(-1.0, min(1.0, float(outcome_utility) * margin))
            if abs(utility_signal) > 1e-8:
                role_utility_by_skill.setdefault(kind, {})[role] = utility_signal
            support_signal = max(0.0, float(outcome_utility) * margin)
            conflict_signal = max(0.0, -float(outcome_utility) * margin)
            if support_signal > 1e-8:
                role_support_by_skill.setdefault(kind, {})[role] = min(1.0, support_signal)
            if conflict_signal > 1e-8:
                role_conflict_by_skill.setdefault(kind, {})[role] = min(1.0, conflict_signal)
    return role_support_by_skill, role_conflict_by_skill, role_utility_by_skill


def _record_trust_policy_metric_histories(
    metric_histories: List[List[Dict[str, Any]]],
    metrics_batch: List[Dict[str, Any]],
    *,
    max_history: int = 4,
) -> List[List[Dict[str, Any]]]:
    if len(metric_histories) < len(metrics_batch):
        metric_histories.extend([[] for _ in range(len(metrics_batch) - len(metric_histories))])
    for index, metrics in enumerate(metrics_batch):
        if not isinstance(metrics, dict) or not metrics:
            continue
        metric_histories[index].append(dict(metrics))
        if len(metric_histories[index]) > max_history:
            metric_histories[index] = metric_histories[index][-max_history:]
    return metric_histories


def summarize_trust_policy_recovery_metrics(trajectory_steps: List[Dict[str, Any]]) -> Dict[str, float]:
    active_steps = [step for step in trajectory_steps if bool(step.get("active_masks", True))]
    if not active_steps:
        return {}

    failed_families_by_step = [
        set(
            _infer_outcome_failure_families(
                step,
                outcome_utility=_trust_metric_value(step, "trust_policy/mechanism/outcome_utility"),
            )
        )
        for step in active_steps
    ]
    action_error_count = float(
        sum(int("action_error_without_memory_fault" in families) for families in failed_families_by_step)
    )
    candidate_families = sorted(
        {
            family
            for families in failed_families_by_step
            for family in families
            if family != "action_error_without_memory_fault"
        }
    )

    applicable_kinds = 0
    recovered_kinds = 0
    recovery_steps: List[float] = []
    stability_scores: List[float] = []

    for family in candidate_families:
        first_failure_idx = None
        first_recovery_idx = None
        for step_idx, step in enumerate(active_steps):
            if not _failure_family_is_applicable(step, family):
                continue
            if first_failure_idx is None and family in failed_families_by_step[step_idx]:
                first_failure_idx = step_idx
                continue
            if first_failure_idx is not None and step_idx > first_failure_idx and family not in failed_families_by_step[step_idx]:
                first_recovery_idx = step_idx
                break

        if first_failure_idx is None:
            continue
        applicable_kinds += 1
        if first_recovery_idx is None:
            continue

        recovered_kinds += 1
        recovery_steps.append(float(first_recovery_idx - first_failure_idx))
        post_recovery_kept: List[float] = []
        for step_idx, step in enumerate(active_steps[first_recovery_idx:], start=first_recovery_idx):
            if not _failure_family_is_applicable(step, family):
                continue
            post_recovery_kept.append(float(family not in failed_families_by_step[step_idx]))
        if post_recovery_kept:
            stability_scores.append(float(np.mean(post_recovery_kept)))

    recovery_success = (recovered_kinds / applicable_kinds) if applicable_kinds > 0 else 0.0
    return {
        "trust_policy/mechanism/recovery_applicable_rate": float(applicable_kinds > 0),
        "trust_policy/mechanism/recovery_failure_kind_count": float(applicable_kinds),
        "trust_policy/mechanism/recovery_success_rate": float(recovery_success),
        "trust_policy/mechanism/recovery_steps_mean": float(np.mean(recovery_steps)) if recovery_steps else 0.0,
        "trust_policy/mechanism/recovery_stability_rate": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "trust_policy/mechanism/action_error_without_memory_fault_count": action_error_count,
        "trust_policy/mechanism/action_error_without_memory_fault_rate": (
            action_error_count / float(len(active_steps)) if active_steps else 0.0
        ),
    }


def _update_feedback_with_discounted_utility(
    feedback: Optional[MemorySkillFeedback],
    metric_history: List[Dict[str, Any]],
    *,
    outcome_utility: float,
    horizon: int = 3,
    discount: float = 0.7,
) -> Optional[MemorySkillFeedback]:
    if not metric_history:
        return feedback
    recent_metrics = metric_history[-horizon:]
    recent_failure_families = [
        _infer_outcome_failure_families(metrics, outcome_utility=outcome_utility)
        for metrics in recent_metrics
    ]
    if float(outcome_utility) < 0.0 and not any(
        family != "action_error_without_memory_fault"
        for families in recent_failure_families
        for family in families
    ):
        return feedback
    updated_feedback = feedback
    for lag, metrics in enumerate(reversed(recent_metrics)):
        failure_families = _infer_outcome_failure_families(metrics, outcome_utility=outcome_utility)
        if failure_families == ["action_error_without_memory_fault"]:
            continue
        discounted_utility = float(outcome_utility) * (float(discount) ** lag)
        utility_by_skill = _infer_utility_by_skill(metrics, discounted_utility)
        witness_utility_by_line = _infer_witness_utility_by_line(metrics, discounted_utility)
        witness_role_support_by_skill = _infer_witness_role_support_by_skill(metrics)
        witness_role_conflict_by_skill = _infer_witness_role_conflict_by_skill(metrics)
        witness_role_utility_by_skill = _infer_witness_role_utility_by_skill(metrics, discounted_utility)
        (
            counterfactual_role_support_by_skill,
            counterfactual_role_conflict_by_skill,
            counterfactual_role_utility_by_skill,
        ) = _infer_counterfactual_role_credit_by_skill(metrics, discounted_utility)
        witness_role_support_by_skill = _merge_role_signal_maps(
            witness_role_support_by_skill,
            counterfactual_role_support_by_skill,
        )
        witness_role_conflict_by_skill = _merge_role_signal_maps(
            witness_role_conflict_by_skill,
            counterfactual_role_conflict_by_skill,
        )
        witness_role_utility_by_skill = _merge_role_signal_maps(
            witness_role_utility_by_skill,
            counterfactual_role_utility_by_skill,
            clamp_signed=True,
        )
        utility_by_skill = _namespace_skill_signal_by_family(metrics, utility_by_skill)
        witness_role_support_by_skill = _namespace_role_signal_by_family(metrics, witness_role_support_by_skill)
        witness_role_conflict_by_skill = _namespace_role_signal_by_family(metrics, witness_role_conflict_by_skill)
        witness_role_utility_by_skill = _namespace_role_signal_by_family(metrics, witness_role_utility_by_skill)
        if (
            not utility_by_skill
            and not witness_utility_by_line
            and not witness_role_support_by_skill
            and not witness_role_conflict_by_skill
            and not witness_role_utility_by_skill
        ):
            continue
        observed_skill_kinds = _namespace_skill_kinds_by_family(metrics, _infer_observed_skill_kinds(metrics))
        updated_feedback = auto_update_memory_skill_feedback(
            updated_feedback,
            observed_skill_kinds=tuple(observed_skill_kinds),
            failed_skill_kinds=(),
            utility_by_skill=utility_by_skill,
            witness_utility_by_line=witness_utility_by_line,
            witness_role_support_by_skill=witness_role_support_by_skill,
            witness_role_conflict_by_skill=witness_role_conflict_by_skill,
            witness_role_utility_by_skill=witness_role_utility_by_skill,
        )
    return updated_feedback


def _build_trust_policy_from_config(trust_policy_cfg: Any) -> Tuple[bool, bool, bool, Optional[str], Optional[TrustCalibratedRenderPolicy]]:
    if not trust_policy_cfg:
        return False, False, False, None, None
    enabled = bool(trust_policy_cfg.get("enable", False))
    query_conditioned = bool(trust_policy_cfg.get("query_conditioned", False))
    state_aware = bool(trust_policy_cfg.get("state_aware", False))
    context_mode = trust_policy_cfg.get("context_mode")
    if context_mode is not None:
        context_mode = str(context_mode).strip().lower() or None
    if context_mode not in {None, "query", "state", "slot", "auto"}:
        raise ValueError(f"Unsupported ocr.trust_policy.context_mode={context_mode}")
    if context_mode is None:
        context_mode = "state" if state_aware else "query"
    query_relevance_weight = float(trust_policy_cfg.get("query_relevance_weight", 0.30))
    context_budget_percent = float(trust_policy_cfg.get("context_budget_percent", 100.0))
    policy = None
    if enabled and query_conditioned:
        policy = TrustCalibratedRenderPolicy(
            TrustPolicyConfig(
                query_relevance_weight=query_relevance_weight,
                context_budget_percent=context_budget_percent,
            )
        )
    return enabled, query_conditioned, state_aware, context_mode, policy


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)
        try:
            self.model_path = str(config.actor_rollout_ref.model.path)
        except Exception:
            self.model_path = ""
        self.agent_select_compression_enable = self.ocr_config.agent_select_compression.get('enable', False)
        self.last_effective_compression_factors: List[float] = []
        self.trust_policy_last_metrics: List[Dict[str, Any]] = []
        self.trust_policy_skill_feedbacks: List[Optional[MemorySkillFeedback]] = []
        self.trust_policy_metric_histories: List[List[Dict[str, Any]]] = []
        self.qwen3_search_ocr_compact_history = False
        self.qwen3_search_ocr_doc_limit = 3
        self.qwen3_search_ocr_doc_char_cap = 0
        self.qwen3_search_ocr_newest_first = True
        self.qwen3_search_ocr_dedupe_docs_across_steps = True
        trust_policy_cfg = self.ocr_config.get('trust_policy', {})
        self.trust_policy_use_compressed_history = bool(trust_policy_cfg.get("use_compressed_history", True))
        self.qwen3_ocr_render_overrides: Dict[str, Any] = {}
        if is_qwen3_vl_model_path(self.model_path):
            self.qwen3_search_ocr_compact_history = (
                os.environ.get("AGENTOCR_QWEN3_SEARCH_OCR_COMPACT_HISTORY", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            self.qwen3_search_ocr_doc_limit = max(
                1,
                int(os.environ.get("AGENTOCR_QWEN3_SEARCH_OCR_DOC_LIMIT", "3") or 3),
            )
            self.qwen3_search_ocr_doc_char_cap = max(
                0,
                int(os.environ.get("AGENTOCR_QWEN3_SEARCH_OCR_DOC_CHAR_CAP", "0") or 0),
            )
            self.qwen3_search_ocr_newest_first = (
                os.environ.get("AGENTOCR_QWEN3_SEARCH_OCR_NEWEST_FIRST", "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            self.qwen3_search_ocr_dedupe_docs_across_steps = (
                os.environ.get("AGENTOCR_QWEN3_SEARCH_OCR_DEDUPE_DOCS_ACROSS_STEPS", "1").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            try:
                qwen3_font_size = int(os.environ.get("AGENTOCR_QWEN3_OCR_RENDER_FONT_SIZE", "0"))
            except Exception:
                qwen3_font_size = 0
            if qwen3_font_size > 0:
                self.qwen3_ocr_render_overrides["font_size"] = qwen3_font_size

            try:
                qwen3_max_width = int(os.environ.get("AGENTOCR_QWEN3_OCR_RENDER_MAX_WIDTH", "0"))
            except Exception:
                qwen3_max_width = 0
            if qwen3_max_width > 0:
                self.qwen3_ocr_render_overrides["max_width"] = qwen3_max_width
            self.qwen3_ocr_render_overrides["qwen3_history_pages"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGES", "1").strip().lower() in {"1", "true", "yes", "on"}
            )
            self.qwen3_ocr_render_overrides["qwen3_history_structured"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_STRUCTURED", "1").strip().lower() in {"1", "true", "yes", "on"}
            )
            self.qwen3_ocr_render_overrides["use_precise"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_USE_PRECISE", "1").strip().lower() in {"1", "true", "yes", "on"}
            )
            qwen3_page_width = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_WIDTH", "0") or 0))
            qwen3_page_height = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_HEIGHT", "0") or 0))
            qwen3_min_height = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_MIN_HEIGHT", "0") or 0))
            qwen3_page_padding = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_PADDING", "0") or 0))
            qwen3_page_gap = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_GAP", "0") or 0))
            qwen3_page_budget = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_BUDGET", "0") or 0))
            qwen3_page_columns = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_COLUMNS", "0") or 0))
            if qwen3_page_width > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_width"] = qwen3_page_width
                self.qwen3_ocr_render_overrides["min_width"] = qwen3_page_width
            if qwen3_page_height > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_height"] = qwen3_page_height
                if not self.trust_policy_use_compressed_history:
                    self.qwen3_ocr_render_overrides["min_height"] = qwen3_page_height
            if qwen3_min_height > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_min_height"] = qwen3_min_height
            if qwen3_page_padding > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_padding"] = qwen3_page_padding
            if qwen3_page_gap > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_gap"] = qwen3_page_gap
            if qwen3_page_budget > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_budget"] = qwen3_page_budget
            if qwen3_page_columns > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_columns"] = qwen3_page_columns
            self.qwen3_ocr_render_overrides["qwen3_history_dynamic_min_height"] = bool(
                self.trust_policy_use_compressed_history
            )
        self.trust_policy_use_prompt_summary = bool(trust_policy_cfg.get("use_prompt_summary", False))
        self.trust_policy_collect_diagnostics = bool(trust_policy_cfg.get("collect_diagnostics", False))
        self.trust_policy_min_compaction_lines = max(0, int(trust_policy_cfg.get("min_compaction_lines", 8) or 0))
        self.trust_policy_min_prompt_summary_lines = max(
            0,
            int(trust_policy_cfg.get("min_prompt_summary_lines", self.trust_policy_min_compaction_lines) or 0),
        )
        self.trust_policy_min_history_steps_for_compaction = max(
            0,
            int(trust_policy_cfg.get("min_history_steps_for_compaction", 0) or 0),
        )
        self.trust_policy_feedback_update_interval = max(
            1,
            int(trust_policy_cfg.get("feedback_update_interval", 4) or 1),
        )
        self.trust_policy_feedback_min_history_lines = max(
            0,
            int(trust_policy_cfg.get("feedback_min_history_lines", self.trust_policy_min_compaction_lines) or 0),
        )
        self.trust_policy_enable, self.trust_policy_query_conditioned, self.trust_policy_state_aware, self.trust_policy_context_mode, self.trust_policy_obj = (
            _build_trust_policy_from_config(trust_policy_cfg)
        )
        force_search_env = os.environ.get("AGENTOCR_SEARCH_FORCE_SEARCH_FIRST_STEP")
        if force_search_env is None:
            force_search_env = os.environ.get("AGENTOCR_SEARCH_NO_HISTORY_FORCE_SEARCH", "0")
        self.force_search_on_no_history_ocr = force_search_env.strip().lower() in {"1", "true", "yes", "on"}

        strict_grounding_env = os.environ.get("AGENTOCR_SEARCH_ENABLE_STRICT_GROUNDING_PROMPT")
        if strict_grounding_env is None:
            strict_grounding_env = os.environ.get("AGENTOCR_SEARCH_STRICT_ENTITY_GROUNDING", "0")
        self.strict_entity_grounding = strict_grounding_env.strip().lower() in {"1", "true", "yes", "on"}

        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.template_no_his = (
                SEARCH_TEMPLATE_NO_HIS_OCR_FORCE_SEARCH
                if self.force_search_on_no_history_ocr
                else SEARCH_TEMPLATE_NO_HIS_OCR
            )
            self.template = SEARCH_TEMPLATE_OCR
            if self.strict_entity_grounding:
                self.template_no_his += SEARCH_TEMPLATE_NO_HIS_OCR_STRICT_ENTITY_GROUNDING
                self.template += SEARCH_TEMPLATE_OCR_STRICT_ENTITY_GROUNDING
            if self.agent_select_compression_enable:
                self.template_no_his += (
                    SEARCH_COMPRESSION_TEMPLATE_NO_HIS_FORCE_SEARCH
                    if self.force_search_on_no_history_ocr
                    else SEARCH_COMPRESSION_TEMPLATE_NO_HIS
                )
                self.template += SEARCH_COMPRESSION_TEMPLATE
        else:
            self.template_no_his = SEARCH_TEMPLATE_NO_HIS
            self.template = SEARCH_TEMPLATE

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))
        self.active_masks = [True] * len(obs)
        self.trust_policy_skill_feedbacks = _build_initial_trust_policy_feedbacks(len(obs))
        self.trust_policy_metric_histories = _build_initial_trust_policy_metric_histories(len(obs))

        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.ocr_time = 0
            # Reset OCRTool to clear all caches and statistics
            self.ocr_tool.reset()

        full_text_obs, trajectory_images = self.build_text_obs(obs, init=True)
        observations = {
            "text": full_text_obs,
            "image": trajectory_images,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        # Extract actions, validity, and compression factors from LLM responses
        if self.ocr_tool and self.ocr_tool.is_enabled() and self.agent_select_compression_enable:
            actions, valids, compression_factors = self.projection_f(text_actions, check_compression_tag=True)
        else:
            actions, valids = self.projection_f(text_actions)
            compression_factors = None
        
        next_obs, rewards, dones, infos = self.envs.step(actions)
        if self.trust_policy_enable and self.trust_policy_query_conditioned:
            self._apply_trust_policy_utility_feedbacks(rewards, dones, infos)
        history_actions = [
            str(info.get("postprocessed_action") or action)
            for action, info in zip(actions, infos)
        ]
        self.memory.store({
            "search": history_actions,
            "information": next_obs,
        })
        
        for i, done in enumerate(dones):
            if done:
                self.active_masks[i] = False

        full_text_obs, trajectory_images = self.build_text_obs(next_obs, compression_factors=compression_factors)
        next_observations = {
            "text": full_text_obs,
            "image": trajectory_images,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            if compression_factors is not None:
                if i < len(self.last_effective_compression_factors):
                    info['compression_factor'] = self.last_effective_compression_factors[i]
                else:
                    info['compression_factor'] = compression_factors[i]
            if self.trust_policy_enable:
                metrics = (
                    self.trust_policy_last_metrics[i]
                    if i < len(self.trust_policy_last_metrics) and self.trust_policy_last_metrics[i]
                    else _build_default_trust_policy_metrics()
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        info[key] = float(value)

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        compression_factors: Optional[List[float]] = None,
        init: bool = False
    ) -> Tuple[List[str], Optional[List]]:
        """
        This function builds the text observation for the agent and optionally renders trajectory history as images.
        
        Returns:
            Tuple of (text_observations, trajectory_images):
            - text_observations: List of processed text observations
            - trajectory_images: List of PIL Images (or None if OCR is disabled/not available)
        """
        postprocess_text_obs = []
        trajectory_images = None
        memory_contexts, valid_lens = None, None
        qwen3_ocr_render_overrides = dict(getattr(self, "qwen3_ocr_render_overrides", {}) or {})
        no_history_render = bool(init or self.config.env.history_length <= 0)

        # Fetch memory contexts if needed (for both OCR and non-OCR cases)
        if not no_history_render:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )
        
        # If OCRTool is enabled, generate images (blank for init, or from history)
        if self.ocr_tool and self.ocr_tool.is_enabled():
            start_time = time.time()
            render_compression_factors = compression_factors
            render_memory_contexts = memory_contexts
            render_trust_policy = self.trust_policy_enable
            render_qwen3_ocr_overrides = dict(qwen3_ocr_render_overrides)

            # Get step count from memory (use first env's memory length as reference)
            step_info = str(len(self.memory[0])) if len(self.memory) > 0 else "0"
            
            # Use compression factors chosen by the LLM (per environment)
            # Pass individual compression factors as a list for per-image compression
            if render_compression_factors is None:
                # Default to no compression (1.0) for all images
                render_compression_factors = [1.0] * len(text_obs)

            if no_history_render:
                render_memory_contexts = [
                    build_search_task_card_context(_query_text_from_task(self.tasks[i]))
                    for i in range(len(text_obs))
                ]
                render_trust_policy = self.trust_policy_enable
                render_qwen3_ocr_overrides["qwen3_history_structured"] = False
                render_qwen3_ocr_overrides["qwen3_history_pages"] = False
                render_qwen3_ocr_overrides["qwen3_history_dynamic_min_height"] = True
            elif (
                render_memory_contexts is not None
                and self.qwen3_search_ocr_compact_history
            ):
                render_memory_contexts, _ = self.memory.fetch_compact_for_ocr(
                    self.config.env.history_length,
                    obs_key="information",
                    action_key="search",
                    doc_limit=self.qwen3_search_ocr_doc_limit,
                    doc_char_cap=self.qwen3_search_ocr_doc_char_cap,
                    newest_first=self.qwen3_search_ocr_newest_first,
                    dedupe_docs_across_steps=self.qwen3_search_ocr_dedupe_docs_across_steps,
                )
                if (
                    bool(render_qwen3_ocr_overrides.get("qwen3_history_pages", False))
                    or bool(render_qwen3_ocr_overrides.get("qwen3_history_structured", False))
                ):
                    render_qwen3_ocr_overrides["qwen3_history_preserve_input_order"] = True

            # Use use_precise=False for faster processing (significant speedup)
            ocr_render_kwargs: Dict[str, Any] = {
                "step_info": step_info,
                "use_precise": False,
                "enable_cache": not no_history_render,
                "current_steps": [len(self.memory[i]) for i in range(len(text_obs))],
                "trust_policy": render_trust_policy,
                "trust_policy_obj": self.trust_policy_obj,
                "trust_policy_query_texts": [
                    _query_text_from_task(self.tasks[i]) for i in range(len(text_obs))
                ] if render_trust_policy and self.trust_policy_query_conditioned else None,
                "trust_policy_skill_feedbacks": self.trust_policy_skill_feedbacks if render_trust_policy else None,
                "trust_policy_state_aware": self.trust_policy_state_aware,
                "trust_policy_context_mode": self.trust_policy_context_mode,
                "trust_policy_current_steps": [len(self.memory[i]) for i in range(len(text_obs))],
                "trust_policy_collect_diagnostics": self.trust_policy_collect_diagnostics,
                "trust_policy_use_compressed_history": self.trust_policy_use_compressed_history,
                "trust_policy_use_prompt_summary": self.trust_policy_use_prompt_summary,
                "trust_policy_min_compaction_lines": self.trust_policy_min_compaction_lines,
                "trust_policy_min_prompt_summary_lines": self.trust_policy_min_prompt_summary_lines,
                "trust_policy_min_history_steps_for_compaction": self.trust_policy_min_history_steps_for_compaction,
            }
            ocr_render_kwargs.update(render_qwen3_ocr_overrides)
            trajectory_images = self.ocr_tool.convert_texts_to_images(
                render_memory_contexts,
                batch_size=len(text_obs),
                active_masks=self.active_masks,
                compression_factor=render_compression_factors,
                save_img=False,
                **ocr_render_kwargs,
            )
            self.last_effective_compression_factors = (
                self.ocr_tool.get_last_applied_compression_factors()
            )
            self.trust_policy_last_metrics = (
                self.ocr_tool.get_last_trust_policy_diagnostics() if render_trust_policy else []
            )
            if render_trust_policy:
                self.trust_policy_metric_histories = _record_trust_policy_metric_histories(
                    self.trust_policy_metric_histories,
                    self.trust_policy_last_metrics,
                )
            if render_trust_policy and self.trust_policy_query_conditioned:
                self._update_trust_policy_skill_feedbacks()
            end_time = time.time()
            self.ocr_time += end_time - start_time
            # print(f"Step {len(self.memory[0])+1}, OCR time: {end_time - start_time}")

        for i in range(len(text_obs)):
            if no_history_render:
                obs_i = self.template_no_his.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = self.template.format(
                    task_description=self.tasks[i],
                    memory_context=memory_contexts[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i] if valid_lens else len(self.memory[i]),
                    compression_factor=compression_factors[i] if compression_factors is not None else 1.0,
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs, trajectory_images


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                if data_source:
                    success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
        success['success_rate'].append(0.0)
        if total_infos[batch_idx]:
            data_source = total_infos[batch_idx][-1].get("data_source")
            if data_source:
                success[f"{data_source}_success_rate"].append(0.0)

    def _update_trust_policy_skill_feedbacks(self) -> None:
        if not getattr(self, "_agentocr_rollout_is_train", True):
            return
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
            support_by_skill_raw = _infer_support_by_skill(metrics)
            conflict_by_skill_raw = _infer_conflict_by_skill(metrics)
            failure_support_boosts, failure_conflict_boosts, failure_role_support_boosts = _failure_family_boosts(metrics)
            for kind, value in failure_support_boosts.items():
                support_by_skill_raw[kind] = max(support_by_skill_raw.get(kind, 0.0), float(value))
            for kind, value in failure_conflict_boosts.items():
                conflict_by_skill_raw[kind] = max(conflict_by_skill_raw.get(kind, 0.0), float(value))
            support_by_skill = _namespace_skill_signal_by_family(metrics, support_by_skill_raw)
            conflict_by_skill = _namespace_skill_signal_by_family(metrics, conflict_by_skill_raw)
            witness_role_support_by_skill = _namespace_role_signal_by_family(
                metrics,
                _merge_role_signal_maps(
                    _infer_witness_role_support_by_skill(metrics),
                    failure_role_support_boosts,
                ),
            )
            witness_role_conflict_by_skill = _namespace_role_signal_by_family(metrics, _infer_witness_role_conflict_by_skill(metrics))
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
        if not getattr(self, "_agentocr_rollout_is_train", True):
            return
        if not self.trust_policy_last_metrics:
            return
        if len(self.trust_policy_skill_feedbacks) < len(self.trust_policy_metric_histories):
            self.trust_policy_skill_feedbacks.extend(
                [None] * (len(self.trust_policy_metric_histories) - len(self.trust_policy_skill_feedbacks))
            )
        for index, metric_history in enumerate(self.trust_policy_metric_histories):
            if index >= len(rewards) or index >= len(infos):
                continue
            history_len = (len(self.memory[index]) if index < len(self.memory) else 0) + 1
            if not bool(dones[index]):
                if history_len < self.trust_policy_feedback_min_history_lines:
                    continue
                if (
                    self.trust_policy_feedback_update_interval > 1
                    and history_len % self.trust_policy_feedback_update_interval != 0
                ):
                    continue
            outcome_utility = _infer_outcome_utility_signal(rewards[index], dones[index], infos[index])
            annotated_metrics = _annotate_outcome_failure_metrics(
                metric_history[-1],
                outcome_utility=outcome_utility,
            ) if metric_history else {}
            if metric_history:
                metric_history[-1] = annotated_metrics
            if index < len(self.trust_policy_last_metrics):
                self.trust_policy_last_metrics[index] = annotated_metrics
            self.trust_policy_skill_feedbacks[index] = _update_feedback_with_discounted_utility(
                self.trust_policy_skill_feedbacks[index],
                metric_history,
                outcome_utility=outcome_utility,
            )
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        try:
            self.model_path = str(config.actor_rollout_ref.model.path)
        except Exception:
            self.model_path = ""
        trust_policy_cfg = self.ocr_config.get('trust_policy', {})
        trust_policy_use_compressed_history = bool(trust_policy_cfg.get("use_compressed_history", True))
        self.qwen3_ocr_render_overrides: Dict[str, Any] = {}
        if is_qwen3_vl_model_path(self.model_path):
            try:
                qwen3_font_size = int(os.environ.get("AGENTOCR_QWEN3_OCR_RENDER_FONT_SIZE", "0"))
            except Exception:
                qwen3_font_size = 0
            if qwen3_font_size > 0:
                self.qwen3_ocr_render_overrides["font_size"] = qwen3_font_size

            try:
                qwen3_max_width = int(os.environ.get("AGENTOCR_QWEN3_OCR_RENDER_MAX_WIDTH", "0"))
            except Exception:
                qwen3_max_width = 0
            if qwen3_max_width > 0:
                self.qwen3_ocr_render_overrides["max_width"] = qwen3_max_width
            self.qwen3_ocr_render_overrides["qwen3_history_pages"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGES", "1").strip().lower() in {"1", "true", "yes", "on"}
            )
            self.qwen3_ocr_render_overrides["qwen3_history_structured"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_STRUCTURED", "1").strip().lower() in {"1", "true", "yes", "on"}
            )
            self.qwen3_ocr_render_overrides["use_precise"] = (
                os.environ.get("AGENTOCR_QWEN3_HISTORY_USE_PRECISE", "1").strip().lower() in {"1", "true", "yes", "on"}
            )
            qwen3_page_width = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_WIDTH", "0") or 0))
            qwen3_page_height = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_HEIGHT", "0") or 0))
            qwen3_min_height = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_MIN_HEIGHT", "0") or 0))
            qwen3_page_padding = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_PADDING", "0") or 0))
            qwen3_page_gap = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_GAP", "0") or 0))
            qwen3_page_budget = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_BUDGET", "0") or 0))
            qwen3_page_columns = max(0, int(os.environ.get("AGENTOCR_QWEN3_HISTORY_PAGE_COLUMNS", "0") or 0))
            if qwen3_page_width > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_width"] = qwen3_page_width
                self.qwen3_ocr_render_overrides["min_width"] = qwen3_page_width
            if qwen3_page_height > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_height"] = qwen3_page_height
                if not trust_policy_use_compressed_history:
                    self.qwen3_ocr_render_overrides["min_height"] = qwen3_page_height
            if qwen3_min_height > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_min_height"] = qwen3_min_height
            if qwen3_page_padding > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_padding"] = qwen3_page_padding
            if qwen3_page_gap > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_gap"] = qwen3_page_gap
            if qwen3_page_budget > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_budget"] = qwen3_page_budget
            if qwen3_page_columns > 0:
                self.qwen3_ocr_render_overrides["qwen3_history_page_columns"] = qwen3_page_columns
            # Keep the Qwen3 structured OCR layout, but restore the old
            # content-length -> image-size behavior when compact trust history
            # is enabled. Otherwise a few compact lines can still be rendered
            # into a large fixed page, increasing visual tokens instead of
            # reducing them.
            self.qwen3_ocr_render_overrides["qwen3_history_dynamic_min_height"] = bool(
                trust_policy_use_compressed_history
            )
        self.agent_select_compression_enable = self.ocr_config.agent_select_compression.get('enable', False)
        self.last_effective_compression_factors: List[float] = []
        self.trust_policy_last_metrics: List[Dict[str, Any]] = []
        self.trust_policy_skill_feedbacks: List[Optional[MemorySkillFeedback]] = []
        self.trust_policy_metric_histories: List[List[Dict[str, Any]]] = []
        self.trust_policy_use_compressed_history = bool(trust_policy_cfg.get("use_compressed_history", True))
        self.trust_policy_use_prompt_summary = bool(trust_policy_cfg.get("use_prompt_summary", False))
        self.trust_policy_collect_diagnostics = bool(trust_policy_cfg.get("collect_diagnostics", False))
        self.trust_policy_min_compaction_lines = max(0, int(trust_policy_cfg.get("min_compaction_lines", 8) or 0))
        self.trust_policy_min_prompt_summary_lines = max(
            0,
            int(trust_policy_cfg.get("min_prompt_summary_lines", self.trust_policy_min_compaction_lines) or 0),
        )
        self.trust_policy_min_history_steps_for_compaction = max(
            0,
            int(trust_policy_cfg.get("min_history_steps_for_compaction", 0) or 0),
        )
        self.trust_policy_feedback_update_interval = max(
            1,
            int(trust_policy_cfg.get("feedback_update_interval", 4) or 1),
        )
        self.trust_policy_feedback_min_history_lines = max(
            0,
            int(trust_policy_cfg.get("feedback_min_history_lines", self.trust_policy_min_compaction_lines) or 0),
        )
        self.trust_policy_enable, self.trust_policy_query_conditioned, self.trust_policy_state_aware, self.trust_policy_context_mode, self.trust_policy_obj = (
            _build_trust_policy_from_config(trust_policy_cfg)
        )

        if self.ocr_tool and self.ocr_tool.is_enabled():
            if is_qwen3_vl_model_path(self.model_path):
                self.template_no_his = ALFWORLD_TEMPLATE_NO_HIS_OCR_QWEN3
                self.template = ALFWORLD_TEMPLATE_OCR_QWEN3
            else:
                self.template_no_his = ALFWORLD_TEMPLATE_NO_HIS_OCR
                self.template = ALFWORLD_TEMPLATE_OCR
        else:
            self.template_no_his = ALFWORLD_TEMPLATE_NO_HIS
            self.template = ALFWORLD_TEMPLATE
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        self.active_masks = [True] * len(text_obs)
        self.trust_policy_skill_feedbacks = _build_initial_trust_policy_feedbacks(len(text_obs))
        self.trust_policy_metric_histories = _build_initial_trust_policy_metric_histories(len(text_obs))
        
        if self.ocr_tool and self.ocr_tool.is_enabled():
            self.ocr_time = 0
            # Reset OCRTool to clear all caches and statistics
            self.ocr_tool.reset()

        full_text_obs, trajectory_images = self.build_text_obs(text_obs, self.envs.get_admissible_commands, compression_factors=None, init=True)
        return {'text': full_text_obs, 'image': trajectory_images, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        # Extract actions, validity, and compression factors from LLM responses
        if self.ocr_tool and self.ocr_tool.is_enabled() and self.agent_select_compression_enable:
            actions, valids, compression_factors = self.projection_f(text_actions, self.envs.get_admissible_commands, check_compression_tag=True)
        else:
            actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
            compression_factors = None
        
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        if self.trust_policy_enable and self.trust_policy_query_conditioned:
            self._apply_trust_policy_utility_feedbacks(rewards, dones, infos)
        sanitized_history_obs = [_strip_embedded_task_text_for_memory(obs) for obs in self.pre_text_obs]
        self.memory.store({'text_obs': sanitized_history_obs, 'action': actions})
        self.pre_text_obs = text_obs

        for i, done in enumerate(dones):
            if done:
                self.active_masks[i] = False

        rewards = to_numpy(rewards) 

        full_text_obs, trajectory_images = self.build_text_obs(text_obs, self.envs.get_admissible_commands, compression_factors=compression_factors)
        if any(info.get("extra.gamefile") is None for info in infos):
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid and compression_factor to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            if compression_factors is not None:
                if i < len(self.last_effective_compression_factors):
                    info['compression_factor'] = self.last_effective_compression_factors[i]
                else:
                    info['compression_factor'] = compression_factors[i]
            if self.trust_policy_enable:
                metrics = (
                    self.trust_policy_last_metrics[i]
                    if i < len(self.trust_policy_last_metrics) and self.trust_policy_last_metrics[i]
                    else _build_default_trust_policy_metrics()
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float, bool)):
                        info[key] = float(value)

        next_observations = {'text': full_text_obs, 'image': trajectory_images, 'anchor': text_obs}
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], compression_factors: Optional[List[float]] = None, init: bool = False) -> Tuple[List[str], Optional[List]]:
        """
        This function builds the text observation for the agent and optionally renders trajectory history as images.
        
        Returns:
            Tuple of (text_observations, trajectory_images):
            - text_observations: List of processed text observations
            - trajectory_images: List of PIL Images (or None if OCR is disabled/not available)
        """
        postprocess_text_obs = []
        trajectory_images = None
        memory_contexts, valid_lens = None, None
        trust_policy_enable = bool(getattr(self, "trust_policy_enable", False))
        trust_policy_query_conditioned = bool(getattr(self, "trust_policy_query_conditioned", False))
        trust_policy_state_aware = bool(getattr(self, "trust_policy_state_aware", False))
        trust_policy_context_mode = getattr(self, "trust_policy_context_mode", None)
        trust_policy_obj = getattr(self, "trust_policy_obj", None)
        trust_policy_skill_feedbacks = getattr(self, "trust_policy_skill_feedbacks", [])
        trust_policy_collect_diagnostics = bool(getattr(self, "trust_policy_collect_diagnostics", False))
        trust_policy_use_compressed_history = bool(getattr(self, "trust_policy_use_compressed_history", True))
        trust_policy_use_prompt_summary = bool(getattr(self, "trust_policy_use_prompt_summary", False))
        trust_policy_min_compaction_lines = max(0, int(getattr(self, "trust_policy_min_compaction_lines", 8) or 0))
        trust_policy_min_prompt_summary_lines = max(
            0,
            int(getattr(self, "trust_policy_min_prompt_summary_lines", trust_policy_min_compaction_lines) or 0),
        )
        trust_policy_min_history_steps_for_compaction = max(
            0,
            int(getattr(self, "trust_policy_min_history_steps_for_compaction", 0) or 0),
        )
        qwen3_ocr_render_overrides = dict(getattr(self, "qwen3_ocr_render_overrides", {}) or {})
        
        # If OCRTool is enabled, generate images (blank for init, or from history)
        if self.ocr_tool and self.ocr_tool.is_enabled():
            start_time = time.time()
            render_compression_factors = compression_factors
            if init or self.config.env.history_length <= 0:
                memory_contexts, valid_lens = None, None
            else:
                memory_contexts, valid_lens = self.memory.fetch(self.config.env.history_length, obs_key="text_obs", action_key="action")

            # Get step count from memory (use first env's memory length as reference)
            step_info = str(len(self.memory[0])) if len(self.memory) > 0 else "0"
            
            # Use compression factors chosen by the LLM (per environment)
            # Pass individual compression factors as a list for per-image compression
            if render_compression_factors is None:
                # Default to no compression (1.0) for all images
                render_compression_factors = [1.0] * len(text_obs)
            
            # Use use_precise=False for faster processing (significant speedup)
            ocr_render_kwargs: Dict[str, Any] = {
                "step_info": step_info,
                "use_precise": False,
                "enable_cache": True,
                "current_steps": [len(self.memory[i]) for i in range(len(text_obs))],
                "trust_policy": trust_policy_enable,
                "trust_policy_obj": trust_policy_obj,
                "trust_policy_query_texts": [
                    _query_text_from_task(self.tasks[i]) for i in range(len(text_obs))
                ] if trust_policy_enable and trust_policy_query_conditioned else None,
                "trust_policy_skill_feedbacks": trust_policy_skill_feedbacks if trust_policy_enable else None,
                "trust_policy_state_aware": trust_policy_state_aware,
                "trust_policy_context_mode": trust_policy_context_mode,
                "trust_policy_current_steps": [len(self.memory[i]) for i in range(len(text_obs))],
                "trust_policy_collect_diagnostics": trust_policy_collect_diagnostics,
                "trust_policy_use_compressed_history": trust_policy_use_compressed_history,
                "trust_policy_use_prompt_summary": trust_policy_use_prompt_summary,
                "trust_policy_min_compaction_lines": trust_policy_min_compaction_lines,
                "trust_policy_min_prompt_summary_lines": trust_policy_min_prompt_summary_lines,
                "trust_policy_min_history_steps_for_compaction": trust_policy_min_history_steps_for_compaction,
            }
            ocr_render_kwargs.update(qwen3_ocr_render_overrides)
            trajectory_images = self.ocr_tool.convert_texts_to_images(
                memory_contexts,
                batch_size=len(text_obs),
                active_masks=self.active_masks,
                compression_factor=render_compression_factors,
                save_img=False,
                **ocr_render_kwargs,
            )
            self.last_effective_compression_factors = (
                self.ocr_tool.get_last_applied_compression_factors()
            )
            self.trust_policy_last_metrics = (
                self.ocr_tool.get_last_trust_policy_diagnostics() if trust_policy_enable else []
            )
            if trust_policy_enable:
                self.trust_policy_metric_histories = _record_trust_policy_metric_histories(
                    self.trust_policy_metric_histories,
                    self.trust_policy_last_metrics,
                )
            if trust_policy_enable and trust_policy_query_conditioned:
                self._update_trust_policy_skill_feedbacks()
            end_time = time.time()
            self.ocr_time += end_time - start_time
            # print(f"Step {len(self.memory[0])+1}, OCR time: {end_time - start_time}")
        elif not init and self.config.env.history_length > 0:
            # OCRTool not enabled, but we still need to fetch memory for text obs
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        prompt_summaries = (
            self.ocr_tool.get_last_trust_policy_prompt_summaries()
            if (
                trust_policy_enable
                and trust_policy_use_prompt_summary
                and self.ocr_tool
                and self.ocr_tool.is_enabled()
            )
            else []
        )

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            filtered_actions = [s for s in admissible_actions[i] if s != 'help']
            raw_action_history = memory_contexts[i] if (memory_contexts is not None and i < len(memory_contexts)) else ""
            action_history = raw_action_history
            filtered_actions = reorder_alfworld_admissible_actions(
                task_description=self.tasks[i],
                current_observation=text_obs[i],
                action_history=raw_action_history,
                admissible_actions=filtered_actions,
                model_path=self.model_path,
            )
            reformatted_admissible_actions = "\n".join(filtered_actions)
            goal_hints = ""
            state_strategy_hint = ""
            history_hint = ""
            failure_hint = ""
            memory_update_hint = ""
            rules = build_alfworld_rules_for_model(
                self.model_path,
                include_compression=bool(
                    self.ocr_tool and self.ocr_tool.is_enabled() and self.agent_select_compression_enable
                ),
            )
            if (
                trust_policy_enable
                and trust_policy_use_prompt_summary
                and is_qwen3_vl_model_path(self.model_path)
                and i < len(prompt_summaries)
            ):
                memory_update_hint = prompt_summaries[i]

            goal_hints = build_alfworld_goal_hints(
                task_description=self.tasks[i],
                model_path=self.model_path,
            )
            state_strategy_hint = build_alfworld_state_strategy_hint(
                task_description=self.tasks[i],
                current_observation=text_obs[i],
                action_history=raw_action_history,
                model_path=self.model_path,
            )
            history_hint = build_alfworld_history_hint(
                raw_action_history,
                self.model_path,
            )
            failure_hint = build_alfworld_failure_hint(
                task_description=self.tasks[i],
                action_history=raw_action_history,
                current_observation=text_obs[i],
                admissible_actions=filtered_actions,
                model_path=self.model_path,
            )

            effective_history_length = valid_lens[i] if valid_lens is not None else 0

            if init or self.config.env.history_length <= 0:
                obs = self.template_no_his.format(
                    task_description=self.tasks[i],
                    goal_hints=goal_hints,
                    state_strategy_hint=state_strategy_hint,
                    history_hint=history_hint,
                    memory_update_hint=memory_update_hint,
                    failure_hint=failure_hint,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    rules=rules,
                )
            else:
                obs = self.template.format(
                    task_description=self.tasks[i],
                    goal_hints=goal_hints,
                    state_strategy_hint=state_strategy_hint,
                    step_count=len(self.memory[i]),
                    history_length=effective_history_length,
                    action_history=action_history,
                    history_hint=history_hint,
                    memory_update_hint=memory_update_hint,
                    failure_hint=failure_hint,
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    compression_factor=compression_factors[i] if compression_factors is not None else 1.0,
                    rules=rules,
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs, trajectory_images

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)

                data_source = info.get("data_source")
                if data_source:
                    success[f"{data_source}_success_rate"].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask
        success['success_rate'].append(0.0)
        if total_infos[batch_idx]:
            info = total_infos[batch_idx][-1]
            data_source = info.get("data_source")
            if data_source:
                success[f"{data_source}_success_rate"].append(0.0)
            gamefile = info.get("extra.gamefile")
            if gamefile:
                self._process_gamefile(gamefile, 0.0, success)

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def _update_trust_policy_skill_feedbacks(self) -> None:
        if not getattr(self, "_agentocr_rollout_is_train", True):
            return
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
            support_by_skill_raw = _infer_support_by_skill(metrics)
            conflict_by_skill_raw = _infer_conflict_by_skill(metrics)
            failure_support_boosts, failure_conflict_boosts, failure_role_support_boosts = _failure_family_boosts(metrics)
            for kind, value in failure_support_boosts.items():
                support_by_skill_raw[kind] = max(support_by_skill_raw.get(kind, 0.0), float(value))
            for kind, value in failure_conflict_boosts.items():
                conflict_by_skill_raw[kind] = max(conflict_by_skill_raw.get(kind, 0.0), float(value))
            support_by_skill = _namespace_skill_signal_by_family(metrics, support_by_skill_raw)
            conflict_by_skill = _namespace_skill_signal_by_family(metrics, conflict_by_skill_raw)
            witness_role_support_by_skill = _namespace_role_signal_by_family(
                metrics,
                _merge_role_signal_maps(
                    _infer_witness_role_support_by_skill(metrics),
                    failure_role_support_boosts,
                ),
            )
            witness_role_conflict_by_skill = _namespace_role_signal_by_family(metrics, _infer_witness_role_conflict_by_skill(metrics))
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
        if not getattr(self, "_agentocr_rollout_is_train", True):
            return
        if not self.trust_policy_last_metrics:
            return
        if len(self.trust_policy_skill_feedbacks) < len(self.trust_policy_metric_histories):
            self.trust_policy_skill_feedbacks.extend(
                [None] * (len(self.trust_policy_metric_histories) - len(self.trust_policy_skill_feedbacks))
            )
        for index, metric_history in enumerate(self.trust_policy_metric_histories):
            if index >= len(rewards) or index >= len(infos):
                continue
            history_len = (len(self.memory[index]) if index < len(self.memory) else 0) + 1
            if not bool(dones[index]):
                if history_len < self.trust_policy_feedback_min_history_lines:
                    continue
                if (
                    self.trust_policy_feedback_update_interval > 1
                    and history_len % self.trust_policy_feedback_update_interval != 0
                ):
                    continue
            outcome_utility = _infer_outcome_utility_signal(rewards[index], dones[index], infos[index])
            annotated_metrics = _annotate_outcome_failure_metrics(
                metric_history[-1],
                outcome_utility=outcome_utility,
            ) if metric_history else {}
            if metric_history:
                metric_history[-1] = annotated_metrics
            if index < len(self.trust_policy_last_metrics):
                self.trust_policy_last_metrics[index] = annotated_metrics
            self.trust_policy_skill_feedbacks[index] = _update_feedback_with_discounted_utility(
                self.trust_policy_skill_feedbacks[index],
                metric_history,
                outcome_utility=outcome_utility,
            )


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return
        success['success_rate'].append(0.0)
        success['webshop_task_score (not success_rate)'].append(0.0)

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _agentocr_boot_debug(
            f"make_envs alfworld train build start train_batch={config.data.train_batch_size} group_n={group_n} resources={resources_per_worker}"
        )
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _agentocr_boot_debug(
            f"make_envs alfworld train build done num_processes={getattr(_envs, 'num_processes', 'unknown')}"
        )
        _agentocr_boot_debug(
            f"make_envs alfworld val build start val_batch={config.data.val_batch_size} resources={resources_per_worker}"
        )
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _agentocr_boot_debug(
            f"make_envs alfworld val build done num_processes={getattr(_val_envs, 'num_processes', 'unknown')}"
        )
        
        projection_f = partial(alfworld_projection)
        _agentocr_boot_debug("make_envs alfworld managers start")
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        _agentocr_boot_debug("make_envs alfworld managers done")
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
