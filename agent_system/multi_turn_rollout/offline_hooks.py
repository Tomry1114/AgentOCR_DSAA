# Copyright 2026 AgentOCR Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_system.reward_manager.counterfactual_credit import (
    CounterfactualCreditConfig,
    CounterfactualOpticalCreditAssigner,
    OpticalStepEvent,
)
from agentocr.trust_policy import (
    SegmentTrustMetadata,
    TrustCalibratedRenderPolicy,
    TrustPolicyConfig,
    build_trust_segments_from_lines,
)


@dataclass(frozen=True)
class OfflineHookSummary:
    """Parseable summary produced by AgentOCR offline research hooks."""

    counterfactual_credit: Dict[str, Any]
    trust_rendering: Dict[str, Any]


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _extract_prompt_text(step: Dict[str, Any]) -> str:
    raw_prompt = step.get("raw_prompt")
    if raw_prompt is None:
        return ""

    raw_prompt = _to_python_scalar(raw_prompt)
    if isinstance(raw_prompt, list) and raw_prompt:
        first_item = raw_prompt[0]
        if isinstance(first_item, dict):
            return str(first_item.get("content", ""))
    if isinstance(raw_prompt, dict):
        return str(raw_prompt.get("content", ""))
    return str(raw_prompt)


def build_step_events(trajectory_steps: Iterable[Dict[str, Any]]) -> List[OpticalStepEvent]:
    """Convert rollout step dictionaries into optical credit events."""

    events: List[OpticalStepEvent] = []
    for index, step in enumerate(trajectory_steps, start=1):
        events.append(
            OpticalStepEvent(
                step=int(step.get("step", index)),
                action_valid=bool(step.get("is_action_valid", True)),
                compression_factor=float(step.get("compression_factor", 1.0)),
                visual_token_count=float(step.get("memory_visual_token_count", 0.0)),
                segment_count=int(step.get("segment_count", 0)),
            )
        )
    return events


def infer_repair_step(
    events: List[OpticalStepEvent],
    is_success: bool,
    explicit_repair_step: Optional[int] = None,
) -> Optional[int]:
    """Infer a cheap repair step for offline debugging.

    This is intentionally conservative. If the caller already has a real
    counterfactual repair step, that value wins. Otherwise, invalid actions are
    used as the first likely failure point; if none exist, use the highest
    compression step as a proxy only for failed trajectories.
    """

    if explicit_repair_step is not None:
        return int(explicit_repair_step)
    if is_success or not events:
        return None

    for event in events:
        if not event.action_valid:
            return event.step

    return max(events, key=lambda event: event.compression_factor).step


def analyze_counterfactual_credit(
    trajectory_steps: Iterable[Dict[str, Any]],
    is_success: bool,
    repair_step: Optional[int] = None,
    config: Optional[CounterfactualCreditConfig] = None,
) -> Dict[str, Any]:
    events = build_step_events(trajectory_steps)
    inferred_repair_step = infer_repair_step(events, is_success, repair_step)
    result = CounterfactualOpticalCreditAssigner(config).assign(
        events,
        is_success=is_success,
        repair_step=inferred_repair_step,
    )
    return {
        "dense_rewards": result.dense_rewards,
        "blame_scores": result.blame_scores,
        "repair_step": result.repair_step,
        "summary": result.summary,
        "events": [event.__dict__ for event in events],
    }


def build_segment_metadata(
    segments: Iterable[Dict[str, Any]],
    default_source_id: str = "offline",
) -> List[SegmentTrustMetadata]:
    """Convert offline segment dictionaries into trust-policy metadata."""

    metadata: List[SegmentTrustMetadata] = []
    for index, segment in enumerate(segments, start=1):
        metadata.append(
            SegmentTrustMetadata(
                text=str(segment.get("text", "")),
                step=int(segment.get("step", index)),
                source_id=str(segment.get("source_id", default_source_id)),
                source_trust=float(segment.get("source_trust", 1.0)),
                support_count=int(segment.get("support_count", 1)),
                contradiction_count=int(segment.get("contradiction_count", 0)),
                suspicious_score=float(segment.get("suspicious_score", 0.0)),
                salience=float(segment.get("salience", 0.5)),
            )
        )
    return metadata


def build_memory_segments_from_context(
    memory_context: str,
    current_step: int,
    default_source_id: str = "memory_context",
) -> List[Dict[str, Any]]:
    """Convert plain history text into heuristic trust-policy segment dicts.

    This is intended for replay/debug analysis when the caller only has the
    flattened history string rather than structured segment metadata.
    """

    if not memory_context:
        return []

    metadata = build_trust_segments_from_lines(
        memory_context.split("\n"),
        current_step=current_step,
        default_source_id=default_source_id,
    )
    return [segment.__dict__ for segment in metadata]


def analyze_trust_rendering(
    segments: Iterable[Dict[str, Any]],
    current_step: int,
    config: Optional[TrustPolicyConfig] = None,
) -> Dict[str, Any]:
    policy = TrustCalibratedRenderPolicy(config)
    metadata = build_segment_metadata(segments)
    decisions = policy.decide_batch(metadata, current_step=current_step)
    action_counts: Dict[str, int] = {}
    for decision in decisions:
        action_counts[decision.action] = action_counts.get(decision.action, 0) + 1

    return {
        "decisions": [decision.__dict__ for decision in decisions],
        "action_counts": action_counts,
        "renderable_text": policy.renderable_text(decisions),
        "compression_factors": policy.compression_factors(decisions),
    }


def run_offline_agentocr_hooks(
    trajectory_steps: Iterable[Dict[str, Any]],
    memory_segments: Optional[Iterable[Dict[str, Any]]],
    is_success: bool,
    current_step: int,
    repair_step: Optional[int] = None,
    memory_context: Optional[str] = None,
) -> OfflineHookSummary:
    """Run both top-2 AgentOCR follow-up hooks on offline trajectory data."""

    resolved_segments = list(memory_segments) if memory_segments is not None else []
    if not resolved_segments and memory_context:
        resolved_segments = build_memory_segments_from_context(memory_context, current_step=current_step)

    return OfflineHookSummary(
        counterfactual_credit=analyze_counterfactual_credit(
            trajectory_steps,
            is_success=is_success,
            repair_step=repair_step,
        ),
        trust_rendering=analyze_trust_rendering(
            resolved_segments,
            current_step=current_step,
        ),
    )


def export_rollout_traces_for_replay(
    total_batch_list: Iterable[Iterable[Dict[str, Any]]],
    output_dir: str,
    max_trajectories: int = 2,
) -> List[str]:
    """Export collected rollout trajectories to replay-analysis friendly JSON.

    This is a debug-only helper for bridging live rollouts and offline analysis.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    exported_paths: List[str] = []
    task_match = str(__import__("os").environ.get("AGENTOCR_EXPORT_TASK_MATCH", "") or "").strip().lower()

    for traj_idx, trajectory in enumerate(total_batch_list):
        steps = list(trajectory)
        if not steps:
            continue

        active_steps = [step for step in steps if bool(step.get("active_masks", True))]
        if not active_steps:
            continue

        final_step = active_steps[-1]
        gamefile = str(final_step.get("debug_env_gamefile", "") or "")
        memory_context = _extract_prompt_text(final_step)
        if task_match:
            haystacks = [gamefile.lower(), memory_context.lower()]
            if not any(task_match in haystack for haystack in haystacks):
                continue
        payload = {
            "trajectory_steps": [
                {
                    "step": step_idx + 1,
                    "is_action_valid": bool(step.get("is_action_valid", True)),
                    "compression_factor": float(step.get("compression_factor", 1.0)),
                    "memory_visual_token_count": float(step.get("memory_visual_token_count", 0.0)),
                    "segment_count": int(step.get("segment_count", 0)),
                    "prompt_text": str(step.get("raw_prompt_text", "") or _extract_prompt_text(step)),
                    "response_text": str(step.get("debug_model_response_text", "") or ""),
                    "observation_text": str(step.get("debug_env_observation_text", "") or ""),
                    "admissible_actions": list(step.get("debug_env_admissible_actions", []) or []),
                    "gamefile": str(step.get("debug_env_gamefile", "") or ""),
                    "won": bool(step.get("debug_env_won", False)),
                }
                for step_idx, step in enumerate(active_steps)
            ],
            "memory_context": memory_context,
            "is_success": bool(final_step.get("is_success", False)),
            "current_step": len(active_steps),
            "traj_uid": str(final_step.get("traj_uid", traj_idx)),
            "uid": str(final_step.get("uid", traj_idx)),
            "gamefile": gamefile,
        }

        output_path = output_root / f"traj_{traj_idx:03d}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        exported_paths.append(str(output_path))
        if len(exported_paths) >= max_trajectories:
            break

    return exported_paths
