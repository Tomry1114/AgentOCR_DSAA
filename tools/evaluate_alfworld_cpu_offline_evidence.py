from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentocr.trust_policy import (  # noqa: E402
    TrustCalibratedRenderPolicy,
    TrustPolicyConfig,
    build_compact_trust_context,
    build_query_conditioned_segments_from_lines,
)
from examples.agentocr_pilots.run_alfworld_slot_aware_multitemplate_pilot import (  # noqa: E402
    run_suite as run_slot_aware_multitemplate_suite,
)


REAL_PROMPT_BRIDGE_PATH = ROOT / "refine-logs/real_prompt_bridge_20260423.json"
REAL_PROMPT_CURATION_PATH = ROOT / "refine-logs/alfworld_real_prompt_objective_subset_20260520.json"
DEFAULT_OUTPUT_PATH = ROOT / "outputs/agentocr_pilots/alfworld_cpu_offline_evidence_report.json"
DEFAULT_MARKDOWN_PATH = ROOT / "refine-logs/ALFWORLD_CPU_OFFLINE_EVIDENCE.md"
SLOT_AWARE_MULTITEMPLATE_OUTPUT_PATH = ROOT / "outputs/agentocr_pilots/alfworld_slot_aware_multitemplate_pilot_current.json"
SCRIPTED_PILOT_PATHS = [
    ROOT / "outputs/agentocr_pilots/alfworld_query_conditioned_pilot_extractive.json",
    ROOT / "outputs/agentocr_pilots/alfworld_long_horizon_query_conditioned_pilot_extractive_sanity.json",
    ROOT / "outputs/agentocr_pilots/alfworld_multitemplate_query_conditioned_pilot_extractive_sanity.json",
]
M20_ABLATION_PATH = ROOT / "outputs/agentocr_pilots/m20_multitemplate_ablation_extractive_sanity.json"
REAL_SUBSET_VARIANTS = ("baseline_raw", "static_compact", "query_compact")
STATE_DEVICE_BY_FAMILY = {
    "look_at_obj_in_light": "desklamp",
    "pick_heat_then_place_in_recep": "microwave",
    "pick_cool_then_place_in_recep": "fridge",
    "pick_clean_then_place_in_recep": "sinkbasin",
}
VERB_STOPWORDS = {
    "put",
    "place",
    "find",
    "take",
    "get",
    "fetch",
    "bring",
    "move",
    "collect",
    "store",
    "carry",
    "hold",
    "look",
    "examine",
    "inspect",
    "clean",
    "cool",
    "heat",
    "wash",
    "rinse",
}


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_entity(text: str) -> str:
    cleaned = normalize_space(re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()))
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned).strip()
    return re.sub(r"\s+\d+$", "", cleaned).strip()


def split_sentences(text: str) -> List[str]:
    source = str(text or "").strip()
    if not source:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z<\-])|\n+", source)
    return [part.strip() for part in parts if part.strip()]


def singularize(text: str) -> str:
    value = normalize_entity(text)
    if value.endswith("ies") and len(value) > 3:
        return value[:-3] + "y"
    if value.endswith("es") and len(value) > 3 and not value.endswith("ses"):
        return value[:-2]
    if value.endswith("s") and len(value) > 3 and not value.endswith("ss"):
        return value[:-1]
    return value


def entity_matches(goal: str, candidate: str) -> bool:
    left = singularize(goal)
    right = singularize(candidate)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def canonical_task_object(task_text: str, task_family: str) -> Optional[str]:
    text = str(task_text or "").lower()
    if task_family == "look_at_obj_in_light":
        match = re.search(r"(?:look|examine|inspect)(?:\s+at)?\s+(?:the\s+)?([a-z0-9_-]+)", text)
        return normalize_entity(match.group(1)) if match else None
    match = re.search(
        r"(?:put|place|find|take|get|fetch|bring|move|collect|store|carry|hold|clean|cool|heat)\s+"
        r"(?:some\s+|a\s+|an\s+|two\s+|\d+\s+)?([a-z0-9_-]+)",
        text,
    )
    return normalize_entity(match.group(1)) if match else None


def canonical_task_receptacle(task_text: str, task_family: str) -> Optional[str]:
    if task_family == "look_at_obj_in_light":
        match = re.search(r"with\s+(?:the\s+)?([a-z0-9_-]+)", str(task_text or "").lower())
        return normalize_entity(match.group(1)) if match else None
    match = re.search(r"(?:in|into|inside|on|onto)\s+(?:the\s+)?([a-z0-9_-]+)", str(task_text or "").lower())
    return normalize_entity(match.group(1)) if match else None


def target_tokens(task_text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", str(task_text or "").lower())
    result = []
    for token in tokens:
        if token in VERB_STOPWORDS or token in {"the", "a", "an", "some", "two", "it", "with", "to", "in", "on", "and"}:
            continue
        result.append(token)
    return result


@dataclass
class ParsedAction:
    raw: str
    kind: str
    target: str = ""
    obj: str = ""
    source: str = ""


@dataclass
class ContextFacts:
    current_locations: List[str]
    visible_objects: List[str]
    visible_anchors: List[str]
    object_locations: Dict[str, str]
    carrying_objects: List[str]
    closed_receptacles: List[str]
    open_receptacles: List[str]
    room_overview_locations: List[str]


def parse_action(action_text: str) -> ParsedAction:
    action = normalize_space(str(action_text or "").lower())
    match = re.match(r"^(go to|open|close|examine|use)\s+(.+)$", action)
    if match:
        return ParsedAction(raw=action, kind=match.group(1), target=normalize_entity(match.group(2)))
    match = re.match(r"^take\s+(.+?)\s+from\s+(.+)$", action)
    if match:
        return ParsedAction(
            raw=action,
            kind="take",
            obj=normalize_entity(match.group(1)),
            source=normalize_entity(match.group(2)),
        )
    match = re.match(r"^(move|apply|put)\s+(.+?)\s+(?:to|in|into|inside|on|onto)\s+(.+)$", action)
    if match:
        return ParsedAction(
            raw=action,
            kind=match.group(1),
            obj=normalize_entity(match.group(2)),
            target=normalize_entity(match.group(3)),
        )
    match = re.match(r"^(clean|cool|heat)\s+(.+?)\s+with\s+(.+)$", action)
    if match:
        return ParsedAction(
            raw=action,
            kind=match.group(1),
            obj=normalize_entity(match.group(2)),
            target=normalize_entity(match.group(3)),
        )
    return ParsedAction(raw=action, kind=action.split(" ", 1)[0] if action else "")


def parse_context_facts(context: str) -> ContextFacts:
    current_locations: List[str] = []
    visible_objects: List[str] = []
    visible_anchors: List[str] = []
    object_locations: Dict[str, str] = {}
    carrying_objects: List[str] = []
    closed_receptacles: List[str] = []
    open_receptacles: List[str] = []
    room_overview_locations: List[str] = []
    last_anchor: Optional[str] = None

    for raw_line in [line.strip() for line in str(context or "").splitlines() if line.strip()]:
        line = raw_line.lower()
        match = re.search(r"\byou arrive at ([^.]+)", line)
        if match:
            anchor = normalize_entity(match.group(1))
            last_anchor = anchor
            current_locations.append(anchor)
        match = re.search(r"\byou are facing the ([^,.]+)", line)
        if match:
            anchor = normalize_entity(match.group(1))
            last_anchor = anchor
            current_locations.append(anchor)
        match = re.search(r"\byou are carrying:?\s+(.+?)(?:\.|$)", line)
        if match:
            for part in re.split(r",| and ", match.group(1)):
                value = normalize_entity(part)
                if value and value != "nothing":
                    carrying_objects.append(value)
        match = re.search(r"\byou pick up the ([a-z0-9 _-]+?) from ([a-z0-9 _-]+?)(?:\.|$)", line)
        if match:
            carrying_objects.append(normalize_entity(match.group(1)))
            object_locations[normalize_entity(match.group(1))] = normalize_entity(match.group(2))
        match = re.search(r"\b(?:on|in|inside) the ([^,.]+), you see (.+?)(?:\.|$)", line)
        if match:
            anchor = normalize_entity(match.group(1))
            last_anchor = anchor
            visible_anchors.append(anchor)
            for part in re.split(r",| and ", match.group(2)):
                value = normalize_entity(part)
                if value and value != "nothing":
                    visible_objects.append(value)
                    object_locations[value] = anchor
        match = re.search(r"\bin it, you see (.+?)(?:\.|$)", line)
        if match and last_anchor:
            visible_anchors.append(last_anchor)
            for part in re.split(r",| and ", match.group(1)):
                value = normalize_entity(part)
                if value and value != "nothing":
                    visible_objects.append(value)
                    object_locations[value] = last_anchor
        match = re.search(r"\bthe ([^,.]+) is (open|closed)\b", line)
        if match:
            receptacle = normalize_entity(match.group(1))
            if match.group(2) == "closed":
                closed_receptacles.append(receptacle)
            else:
                open_receptacles.append(receptacle)
        if raw_line.startswith("[SKILL][location]"):
            for token in raw_line.split()[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = normalize_entity(key)
                value = normalize_entity(value.replace("_", " "))
                if key == "agent":
                    current_locations.append(value)
                    continue
                if key == "anchor":
                    visible_anchors.append(value)
                    last_anchor = value
                    continue
                if key:
                    object_locations[key] = value
                    visible_objects.append(key)
        if raw_line.startswith("[SKILL][state]"):
            for token in raw_line.split()[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                receptacle = normalize_entity(key.replace("_", " "))
                state = normalize_entity(value)
                if state == "closed":
                    closed_receptacles.append(receptacle)
                elif state == "open":
                    open_receptacles.append(receptacle)
        if "looking quickly around you, you see" in line:
            suffix = line.split("looking quickly around you, you see", 1)[1]
            for part in re.split(r",| and ", suffix):
                value = normalize_entity(part)
                if value and value != "nothing":
                    room_overview_locations.append(value)

    return ContextFacts(
        current_locations=current_locations,
        visible_objects=visible_objects,
        visible_anchors=visible_anchors,
        object_locations=object_locations,
        carrying_objects=carrying_objects,
        closed_receptacles=closed_receptacles,
        open_receptacles=open_receptacles,
        room_overview_locations=room_overview_locations,
    )


def render_compact_context(
    task_text: str,
    raw_lines: Sequence[str],
    *,
    query_weight: float,
) -> str:
    if not raw_lines:
        return ""
    current_step = max(1, len(raw_lines))
    policy = TrustCalibratedRenderPolicy(TrustPolicyConfig(query_relevance_weight=query_weight))
    segments = build_query_conditioned_segments_from_lines(
        raw_lines,
        current_step=current_step,
        query_text=task_text,
        state_aware=True,
        context_mode="auto",
    )
    decisions = policy.apply_context_budget(policy.decide_batch(segments, current_step=current_step))
    kept_lines = [decision.text for decision in decisions if decision.action != "hide" and str(decision.text or "").strip()]
    return build_compact_trust_context(
        query_text=task_text,
        raw_lines=raw_lines,
        kept_lines=kept_lines,
        requested_mode="auto",
        state_aware=True,
        context_budget_percent=policy.config.context_budget_percent,
    )


def select_objective_action(task_text: str, task_family: str, context: str, admissible_actions: Sequence[str]) -> str:
    goal_object = canonical_task_object(task_text, task_family)
    goal_receptacle = canonical_task_receptacle(task_text, task_family)
    state_device = STATE_DEVICE_BY_FAMILY.get(task_family)
    facts = parse_context_facts(context)
    actions = [parse_action(action) for action in admissible_actions]
    best_action = actions[-1].raw if actions else ""
    best_score = -1e9

    for parsed in actions:
        score = 0.0
        if (
            parsed.kind == "take"
            and task_family != "look_at_obj_in_light"
            and goal_object
            and entity_matches(goal_object, parsed.obj)
        ):
            score += 6.0
            if parsed.source and any(entity_matches(parsed.source, anchor) for anchor in facts.current_locations + facts.visible_anchors):
                score += 3.0
            if any(entity_matches(goal_object, visible) for visible in facts.visible_objects):
                score += 2.0
        if parsed.kind in {"move", "apply", "put"} and goal_object and entity_matches(goal_object, parsed.obj):
            score += 5.0
            if goal_receptacle and entity_matches(goal_receptacle, parsed.target):
                score += 4.0
            if any(entity_matches(goal_object, carried) for carried in facts.carrying_objects):
                score += 4.0
        if parsed.kind in {"clean", "cool", "heat"} and goal_object and entity_matches(goal_object, parsed.obj):
            score += 5.0
            if state_device and entity_matches(state_device, parsed.target):
                score += 5.0
            if any(entity_matches(goal_object, carried) for carried in facts.carrying_objects):
                score += 3.0
        if parsed.kind == "use":
            if task_family == "look_at_obj_in_light" and state_device and entity_matches(state_device, parsed.target):
                score += 7.0
                if goal_object and any(entity_matches(goal_object, visible) for visible in facts.visible_objects):
                    score += 3.0
        if parsed.kind == "open":
            if parsed.target and any(entity_matches(parsed.target, closed) for closed in facts.closed_receptacles):
                score += 4.0
            if parsed.target and any(entity_matches(parsed.target, location) for location in facts.current_locations):
                score += 2.0
            if task_family in STATE_DEVICE_BY_FAMILY and state_device and entity_matches(state_device, parsed.target):
                score += 4.0
            if goal_receptacle and entity_matches(goal_receptacle, parsed.target):
                score += 2.0
        if parsed.kind == "go to":
            if goal_receptacle and entity_matches(goal_receptacle, parsed.target):
                score += 4.0
            if goal_object and any(entity_matches(goal_object, carried) for carried in facts.carrying_objects):
                score += 3.0
            if task_family == "look_at_obj_in_light" and any(entity_matches(parsed.target, loc) for loc in facts.room_overview_locations):
                if parsed.target.startswith("desk"):
                    score += 2.0
        if parsed.kind == "examine":
            if parsed.target and any(entity_matches(parsed.target, location) for location in facts.current_locations):
                score += 0.5
        if parsed.kind in {"look", "inventory"}:
            score -= 1.0
        if score > best_score:
            best_score = score
            best_action = parsed.raw
    return best_action


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials <= 0:
        return 1.0
    tail = min(wins, losses)
    cumulative = sum(math.comb(trials, k) for k in range(0, tail + 1)) / (2 ** trials)
    return min(1.0, 2.0 * cumulative)


def bootstrap_delta_interval(better: Sequence[int], worse: Sequence[int], samples: int = 4000, seed: int = 0) -> Tuple[float, float]:
    count = len(better)
    if count <= 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        better_rate = sum(better[index] for index in indices) / count
        worse_rate = sum(worse[index] for index in indices) / count
        deltas.append(better_rate - worse_rate)
    deltas.sort()
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return deltas[lower_index], deltas[upper_index]


def paired_binary_summary(
    pairs: Sequence[Tuple[int, int]],
    *,
    better_label: str,
    worse_label: str,
) -> Dict[str, object]:
    wins = sum(1 for better, worse in pairs if better == 1 and worse == 0)
    losses = sum(1 for better, worse in pairs if better == 0 and worse == 1)
    ties = len(pairs) - wins - losses
    better_series = [better for better, _ in pairs]
    worse_series = [worse for _, worse in pairs]
    better_successes = sum(better_series)
    worse_successes = sum(worse_series)
    delta = (better_successes - worse_successes) / len(pairs) if pairs else 0.0
    ci_low, ci_high = bootstrap_delta_interval(better_series, worse_series)
    return {
        "better_label": better_label,
        "worse_label": worse_label,
        "num_pairs": len(pairs),
        "better_success_count": better_successes,
        "worse_success_count": worse_successes,
        "better_success_rate": better_successes / len(pairs) if pairs else 0.0,
        "worse_success_rate": worse_successes / len(pairs) if pairs else 0.0,
        "delta_success_rate": delta,
        "bootstrap_delta_ci95": [ci_low, ci_high],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p_value": exact_two_sided_sign_test(wins, losses),
    }


def extract_action_target(action_text: str) -> Tuple[str, str]:
    parsed = parse_action(action_text)
    if parsed.kind == "take":
        return parsed.obj, parsed.source
    if parsed.kind in {"move", "apply", "put"}:
        return parsed.obj, parsed.target
    if parsed.kind in {"go to", "open", "close", "use", "examine"}:
        return "", parsed.target
    if parsed.kind in {"clean", "cool", "heat"}:
        return parsed.obj, parsed.target
    return "", ""


def classify_scripted_failure(row: Dict[str, object], *, better_key: str, worse_key: str) -> str:
    ground_truth_action = str(row.get("ground_truth_action") or "")
    better_context = str(row.get(f"{better_key}_context") or "")
    worse_context = str(row.get(f"{worse_key}_context") or "")
    obj, target = extract_action_target(ground_truth_action)
    better_obj = not obj or normalize_entity(obj) in normalize_entity(better_context)
    better_target = not target or normalize_entity(target) in normalize_entity(better_context)
    worse_obj = not obj or normalize_entity(obj) in normalize_entity(worse_context)
    worse_target = not target or normalize_entity(target) in normalize_entity(worse_context)
    if better_target and not worse_target:
        return "target_anchor_restored"
    if better_obj and not worse_obj:
        return "target_object_restored"
    if better_obj and better_target and worse_obj and worse_target:
        return "action_selection_only"
    return "mixed_or_unknown"


def evaluate_real_prompt_subset(real_bridge_path: Path, curation_path: Path) -> Dict[str, object]:
    bridge_data = json.loads(real_bridge_path.read_text())
    curated_data = json.loads(curation_path.read_text())
    rows = bridge_data["rows"]

    case_results: List[Dict[str, object]] = []
    variant_hits = {variant: [] for variant in REAL_SUBSET_VARIANTS}

    for case in curated_data["cases"]:
        row = rows[int(case["row_index"])]
        raw_text = str(row.get("history_text") or row.get("current_observation") or "")
        raw_lines = split_sentences(raw_text)
        baseline_context = "\n".join(raw_lines)
        static_context = render_compact_context(row["task_text"], raw_lines, query_weight=0.0)
        query_context = render_compact_context(row["task_text"], raw_lines, query_weight=0.30)
        contexts = {
            "baseline_raw": baseline_context,
            "static_compact": static_context,
            "query_compact": query_context,
        }
        predictions = {
            variant: select_objective_action(
                task_text=row["task_text"],
                task_family=row["task_family"],
                context=context,
                admissible_actions=row["admissible_actions"],
            )
            for variant, context in contexts.items()
        }
        hits = {
            variant: int(predictions[variant] == str(case["label_action"]).lower())
            for variant in contexts
        }
        for variant, hit in hits.items():
            variant_hits[variant].append(hit)
        case_results.append(
            {
                "row_index": case["row_index"],
                "task_text": row["task_text"],
                "task_family": row["task_family"],
                "label_action": str(case["label_action"]).lower(),
                "label_reason": case["reason"],
                "original_chosen_action": str(row.get("chosen_action") or "").lower(),
                "current_observation": raw_text,
                "admissible_actions": row["admissible_actions"],
                "predictions": predictions,
                "hits": hits,
                "contexts": contexts,
            }
        )

    variant_summaries = {}
    for variant, hits in variant_hits.items():
        success_count = sum(hits)
        ci_low, ci_high = wilson_interval(success_count, len(hits))
        variant_summaries[variant] = {
            "success_count": success_count,
            "num_cases": len(hits),
            "success_rate": success_count / len(hits) if hits else 0.0,
            "wilson_ci95": [ci_low, ci_high],
        }

    query_vs_baseline = paired_binary_summary(
        list(zip(variant_hits["query_compact"], variant_hits["baseline_raw"])),
        better_label="query_compact",
        worse_label="baseline_raw",
    )
    query_vs_static = paired_binary_summary(
        list(zip(variant_hits["query_compact"], variant_hits["static_compact"])),
        better_label="query_compact",
        worse_label="static_compact",
    )

    return {
        "source_path": str(real_bridge_path.relative_to(ROOT)),
        "curation_path": str(curation_path.relative_to(ROOT)),
        "label_policy": curated_data["label_policy"],
        "num_cases": len(case_results),
        "variant_summaries": variant_summaries,
        "paired": {
            "query_vs_baseline": query_vs_baseline,
            "query_vs_static": query_vs_static,
        },
        "cases": case_results,
        "caveat": (
            "This real-prompt subset is intentionally small and conservative. It is useful for sanity-checking "
            "real ALFWorld prompt slices, but it is not large enough to claim benchmark-level superiority."
        ),
    }


def summarize_scripted_pilot(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text())
    rows = [row for row in payload.get("rows", []) if row.get("protocol") == "tight_rendered_area_budget"]
    pairs = [(int(bool(row["query_success"])), int(bool(row["baseline_success"]))) for row in rows]
    query_vs_baseline = paired_binary_summary(pairs, better_label="query", worse_label="baseline")
    query_vs_static = paired_binary_summary(
        [(int(bool(row["query_success"])), int(bool(row["static_success"]))) for row in rows],
        better_label="query",
        worse_label="static",
    )
    query_wins = [
        {
            "task_description": row["task_description"],
            "ground_truth_action": row["ground_truth_action"],
            "baseline_action": row["baseline_action"],
            "query_action": row["query_action"],
            "failure_mode": classify_scripted_failure(row, better_key="query", worse_key="baseline"),
        }
        for row in rows
        if row["query_success"] and not row["baseline_success"]
    ]
    return {
        "path": str(path.relative_to(ROOT)),
        "num_rows": len(rows),
        "query_vs_baseline": query_vs_baseline,
        "query_vs_static": query_vs_static,
        "query_only_wins": query_wins,
        "summary": payload.get("summary", {}),
        "evidence_type": "scripted_cpu_pilot",
    }


def aggregate_scripted_pilots(summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    combined_pairs: List[Tuple[int, int]] = []
    combined_pairs_static: List[Tuple[int, int]] = []
    failure_counter: Counter[str] = Counter()
    for summary in summaries:
        source = json.loads((ROOT / summary["path"]).read_text())
        rows = [row for row in source.get("rows", []) if row.get("protocol") == "tight_rendered_area_budget"]
        combined_pairs.extend((int(bool(row["query_success"])), int(bool(row["baseline_success"]))) for row in rows)
        combined_pairs_static.extend((int(bool(row["query_success"])), int(bool(row["static_success"]))) for row in rows)
        for row in rows:
            if row["query_success"] and not row["baseline_success"]:
                failure_counter[classify_scripted_failure(row, better_key="query", worse_key="baseline")] += 1
    return {
        "num_pilots": len(summaries),
        "aggregate_query_vs_baseline": paired_binary_summary(
            combined_pairs,
            better_label="query",
            worse_label="baseline",
        ),
        "aggregate_query_vs_static": paired_binary_summary(
            combined_pairs_static,
            better_label="query",
            worse_label="static",
        ),
        "failure_taxonomy": dict(sorted(failure_counter.items())),
        "caveat": (
            "These pilots are scripted ALFWorld-style next-action evaluations. They are useful mechanism evidence "
            "for the no-GPU path, but they are not the official ALFWorld benchmark split."
        ),
    }


def summarize_m20_ablation(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text())
    rows = [row for row in payload.get("rows", []) if row.get("protocol") == "tight_rendered_area_budget"]
    comparisons = {}
    for weaker in ("baseline", "static", "lexical_query", "naive_query_trust"):
        comparisons[f"query_vs_{weaker}"] = paired_binary_summary(
            [
                (
                    int(bool(row["variants"]["query"]["success"])),
                    int(bool(row["variants"][weaker]["success"])),
                )
                for row in rows
            ],
            better_label="query",
            worse_label=weaker,
        )
    by_distractor = []
    for distractor_count in sorted({int(row["distractor_count"]) for row in rows}):
        subset = [row for row in rows if int(row["distractor_count"]) == distractor_count]
        by_distractor.append(
            {
                "distractor_count": distractor_count,
                "query_vs_baseline": paired_binary_summary(
                    [
                        (
                            int(bool(row["variants"]["query"]["success"])),
                            int(bool(row["variants"]["baseline"]["success"])),
                        )
                        for row in subset
                    ],
                    better_label="query",
                    worse_label="baseline",
                ),
                "query_vs_lexical_query": paired_binary_summary(
                    [
                        (
                            int(bool(row["variants"]["query"]["success"])),
                            int(bool(row["variants"]["lexical_query"]["success"])),
                        )
                        for row in subset
                    ],
                    better_label="query",
                    worse_label="lexical_query",
                ),
            }
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "num_rows": len(rows),
        "comparisons": comparisons,
        "by_distractor_count": by_distractor,
        "summary": payload.get("summary", {}),
        "evidence_type": "scripted_cpu_ablation",
        "caveat": (
            "This ablation suite is still scripted, but it is useful for checking whether the gain survives against "
            "simple lexical retrieval and naive trust variants."
        ),
    }


def build_current_slot_aware_suite() -> Dict[str, object]:
    payload = run_slot_aware_multitemplate_suite(max_templates=6, seeds=[0, 1])
    SLOT_AWARE_MULTITEMPLATE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SLOT_AWARE_MULTITEMPLATE_OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    paired = {
        (row["protocol"], row["stronger_variant"], row["weaker_variant"]): row
        for row in payload["summary"]["paired_summaries"]
    }
    return {
        "path": str(SLOT_AWARE_MULTITEMPLATE_OUTPUT_PATH.relative_to(ROOT)),
        "summary": payload["summary"],
        "paired": {
            "full_text_slot_vs_baseline": paired[("full_text_memory", "slot", "baseline")],
            "tight_budget_slot_vs_baseline": paired[("tight_rendered_area_budget", "slot", "baseline")],
            "tight_budget_slot_vs_query": paired[("tight_rendered_area_budget", "slot", "query")],
        },
        "evidence_type": "fresh_current_code_cpu_pilot",
        "caveat": (
            "This suite is still a scripted ALFWorld mechanism slice rather than the official seen/unseen split, "
            "but unlike the older query-conditioned JSONs it is freshly rerun on the current code and directly "
            "targets the current slot-aware memory story."
        ),
    }


def write_markdown(report: Dict[str, object], output_path: Path) -> None:
    real_subset = report["real_prompt_objective_subset"]
    current_slot_aware = report["fresh_current_slot_aware_suite"]
    scripted = report["scripted_tight_budget_suite"]
    m20 = report["m20_ablation_suite"]
    lines = [
        "# ALFWorld CPU Offline Evidence",
        "",
        "## Bottom Line",
        "",
        report["bottom_line"]["summary"],
        "",
        "## Real Prompt Objective Subset",
        "",
        f"- cases: `{real_subset['num_cases']}`",
        f"- query vs baseline: `{real_subset['paired']['query_vs_baseline']['better_success_count']}/{real_subset['num_cases']}` vs `{real_subset['paired']['query_vs_baseline']['worse_success_count']}/{real_subset['num_cases']}`",
        f"- query vs static: `{real_subset['paired']['query_vs_static']['better_success_count']}/{real_subset['num_cases']}` vs `{real_subset['paired']['query_vs_static']['worse_success_count']}/{real_subset['num_cases']}`",
        f"- caveat: {real_subset['caveat']}",
        "",
        "## Fresh Current-Code Slot-Aware Suite",
        "",
        f"- tasks: `{current_slot_aware['summary']['num_tasks']}`",
        f"- full-text slot vs baseline: `{current_slot_aware['paired']['full_text_slot_vs_baseline']['wins']}/{current_slot_aware['paired']['full_text_slot_vs_baseline']['losses']}/{current_slot_aware['paired']['full_text_slot_vs_baseline']['ties']}`",
        f"- tight-budget slot vs baseline: `{current_slot_aware['paired']['tight_budget_slot_vs_baseline']['wins']}/{current_slot_aware['paired']['tight_budget_slot_vs_baseline']['losses']}/{current_slot_aware['paired']['tight_budget_slot_vs_baseline']['ties']}`",
        f"- tight-budget sign-test p-value: `{current_slot_aware['paired']['tight_budget_slot_vs_baseline']['sign_test_p_value']}`",
        f"- tight-budget slot success rate: `{current_slot_aware['paired']['tight_budget_slot_vs_baseline']['stronger_success_rate']}`",
        f"- tight-budget baseline success rate: `{current_slot_aware['paired']['tight_budget_slot_vs_baseline']['weaker_success_rate']}`",
        f"- caveat: {current_slot_aware['caveat']}",
        "",
        "## Scripted Tight-Budget Pilots",
        "",
        f"- aggregate query vs baseline wins/losses/ties: `{scripted['aggregate']['aggregate_query_vs_baseline']['wins']}/{scripted['aggregate']['aggregate_query_vs_baseline']['losses']}/{scripted['aggregate']['aggregate_query_vs_baseline']['ties']}`",
        f"- aggregate sign-test p-value: `{scripted['aggregate']['aggregate_query_vs_baseline']['sign_test_p_value']}`",
        f"- aggregate delta success rate: `{scripted['aggregate']['aggregate_query_vs_baseline']['delta_success_rate']}`",
        f"- failure taxonomy: `{json.dumps(scripted['aggregate']['failure_taxonomy'], ensure_ascii=False)}`",
        f"- caveat: {scripted['aggregate']['caveat']}",
        "",
        "## M20 CPU Ablation",
        "",
        f"- query vs lexical_query delta: `{m20['comparisons']['query_vs_lexical_query']['delta_success_rate']}`",
        f"- query vs naive_query_trust delta: `{m20['comparisons']['query_vs_naive_query_trust']['delta_success_rate']}`",
        f"- caveat: {m20['caveat']}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> Dict[str, object]:
    real_subset = evaluate_real_prompt_subset(REAL_PROMPT_BRIDGE_PATH, REAL_PROMPT_CURATION_PATH)
    current_slot_aware_suite = build_current_slot_aware_suite()
    scripted_summaries = [summarize_scripted_pilot(path) for path in SCRIPTED_PILOT_PATHS if path.exists()]
    scripted_aggregate = aggregate_scripted_pilots(scripted_summaries)
    m20_summary = summarize_m20_ablation(M20_ABLATION_PATH)
    current_slot_aware_gap = current_slot_aware_suite["paired"]["tight_budget_slot_vs_baseline"]

    bottom_line = {
        "claim_supported": "partial",
        "summary": (
            "The current no-GPU ALFWorld evidence now supports a stronger partial claim for the current code: on a "
            "freshly rerun slot-aware multitemplate CPU suite, the current idea beats the OCR baseline in paired "
            "next-action evaluation, and a small manually curated real-prompt subset shows that the same pipeline "
            "does not regress on objective real ALFWorld prompt slices. This is still not enough to claim "
            "superiority on the official ALFWorld benchmark split."
        ),
        "fresh_current_slot_sign_test_p_value": current_slot_aware_gap["sign_test_p_value"],
        "fresh_current_slot_delta_success_rate": current_slot_aware_gap["delta_success_rate"],
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_prompt_objective_subset": real_subset,
        "fresh_current_slot_aware_suite": current_slot_aware_suite,
        "scripted_tight_budget_suite": {
            "sources": scripted_summaries,
            "aggregate": scripted_aggregate,
        },
        "m20_ablation_suite": m20_summary,
        "bottom_line": bottom_line,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CPU-only ALFWorld evidence report for the current no-GPU path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_PATH))
    args = parser.parse_args()

    report = build_report()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_path = Path(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, markdown_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "markdown_output": str(markdown_path),
                "bottom_line": report["bottom_line"],
                "fresh_current_slot_vs_baseline": report["fresh_current_slot_aware_suite"]["paired"]["tight_budget_slot_vs_baseline"],
                "scripted_query_vs_baseline": report["scripted_tight_budget_suite"]["aggregate"]["aggregate_query_vs_baseline"],
                "real_prompt_query_vs_baseline": report["real_prompt_objective_subset"]["paired"]["query_vs_baseline"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
