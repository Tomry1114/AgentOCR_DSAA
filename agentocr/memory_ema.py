from __future__ import annotations

from dataclasses import replace
from typing import Any


def schema_uncertainty(
    *,
    confidence: float,
    support: float,
    conflict: float,
    freshness: float,
    coverage: float,
    hit_count: int,
    miss_count: int,
) -> float:
    evidence = min(
        1.0,
        0.24 * min(1.0, max(0.0, coverage))
        + 0.28 * min(1.0, max(0.0, support))
        + 0.18 * min(1.0, max(0.0, freshness))
        + 0.18 * min(1.0, max(0.0, float(confidence)))
        + 0.12 * min(1.0, float(hit_count) / float(max(1, hit_count + miss_count))),
    )
    instability = min(
        1.0,
        0.50 * min(1.0, max(0.0, conflict))
        + 0.25 * min(1.0, float(miss_count) / float(max(1, hit_count + miss_count)))
        + 0.25 * max(0.0, 1.0 - min(1.0, max(0.0, freshness))),
    )
    return min(1.0, max(0.0, 0.75 * (1.0 - evidence) + 0.25 * instability))


def schema_lifecycle_stage(
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
    if (
        conflict >= 0.85
        or eviction_pressure >= 0.72
        or (age >= 3 and miss_count >= max(2, hit_count + 1) and freshness <= 0.22)
    ):
        return "retired"
    if (
        conflict >= 0.45
        or eviction_pressure >= 0.35
        or (miss_count > hit_count and freshness <= 0.45)
    ):
        return "stale"
    if (
        age >= 2
        and hit_count >= 2
        and support >= 0.58
        and freshness >= 0.45
        and uncertainty <= 0.42
        and conflict <= 0.30
    ):
        return "stable"
    return "proto"


def schema_verification_signal(
    line_text: str,
    schema_witnesses: list[Any] | tuple[Any, ...],
    *,
    normalized_line_key,
) -> float:
    line_key = normalized_line_key(line_text)
    signal = 0.0
    for schema in schema_witnesses:
        schema_keys = {normalized_line_key(line) for line in schema.lines}
        if line_key not in schema_keys or schema.lifecycle_stage == "retired":
            continue
        stage_bonus = {"proto": 0.06, "stable": 0.03, "stale": 0.10}.get(schema.lifecycle_stage, 0.0)
        signal = max(
            signal,
            min(
                1.0,
                0.08
                + 0.28 * float(schema.uncertainty)
                + 0.16 * float(schema.coverage)
                + 0.10 * float(schema.support)
                + stage_bonus
                - 0.18 * float(schema.eviction_pressure),
            ),
        )
    return max(0.0, signal)


def schema_component_tags(
    schema,
    kind: str,
    goal_slots,
    *,
    normalize_fact_token,
    strip_xml_tags,
    extract_current_location,
    extract_fact_signature,
    goal_object_for_entity,
    extract_visible_anchor,
    extract_anchor_object_locations,
    extract_pickup_object,
    extract_placement,
    extract_receptacle_state,
) -> tuple[str, ...]:
    tags: set[str] = set()
    for line in schema.lines:
        lowered = normalize_fact_token(strip_xml_tags(line))
        if kind == "location":
            if extract_current_location(line) is not None:
                tags.add("arrival")
            signature = extract_fact_signature(line)
            if signature is not None:
                subject, _ = signature
                if goal_object_for_entity(subject, goal_slots):
                    tags.add("target_fact")
            anchor_locations = extract_anchor_object_locations(line, goal_slots)
            if anchor_locations:
                tags.add("target_fact")
                tags.add("anchor")
            placement = extract_placement(line)
            if placement is not None:
                placed_object, _ = placement
                if goal_object_for_entity(placed_object, goal_slots):
                    tags.add("target_fact")
                    tags.add("placement")
            pickup_object = extract_pickup_object(line)
            if pickup_object and goal_object_for_entity(pickup_object, goal_slots):
                tags.add("inventory_pointer")
            if extract_visible_anchor(line) is not None:
                tags.add("anchor")
            if (
                "arrive" not in lowered
                and signature is None
                and not anchor_locations
                and placement is None
                and pickup_object is None
                and extract_visible_anchor(line) is None
            ):
                tags.add("context")
        elif kind == "progress":
            if extract_pickup_object(line):
                tags.add("pickup")
            if extract_placement(line) is not None:
                tags.add("placement")
            if extract_receptacle_state(line) is not None:
                tags.add("receptacle_state")
        elif kind == "state":
            if extract_receptacle_state(line) is not None:
                tags.add("receptacle_state")
            if any(token in lowered for token in ("steaming", "heated", "chilled", "cooled", "rinsed", "washed", "clean", "dirty")):
                tags.add("object_state")
            if any(token in lowered for token in ("illuminated", "bright", "dark")):
                tags.add("light_state")
    return tuple(sorted(tags))


def schema_template_signature(schema, kind: str, goal_slots, **kwargs) -> tuple[str, ...] | None:
    components = schema_component_tags(schema, kind, goal_slots, **kwargs)
    if len(components) < 2:
        return None
    return (kind, *components)


def schema_primary_subject(
    schema,
    kind: str,
    goal_slots,
    *,
    schema_fact_signature,
    schema_progress_signature,
    schema_state_signature,
    goal_object_for_entity,
    normalize_fact_token,
) -> str:
    if kind == "location":
        signature = schema_fact_signature(schema, goal_slots)
        if signature is not None:
            subject, _ = signature
            return goal_object_for_entity(subject, goal_slots) or normalize_fact_token(subject)
    if kind == "progress":
        signature = schema_progress_signature(schema, goal_slots)
        if signature is not None:
            subject, _ = signature
            return goal_object_for_entity(subject, goal_slots) or normalize_fact_token(subject)
    if kind == "state":
        signature = schema_state_signature(schema, goal_slots)
        if signature is not None:
            subject, _ = signature
            return normalize_fact_token(subject)
    return "generic"


def best_schema_representative(schemas) -> Any:
    return max(
        schemas,
        key=lambda schema: (
            float(schema.freshness) + 0.8 * float(schema.support) - 0.6 * float(schema.conflict),
            float(schema.coverage),
            -float(schema.uncertainty),
            len(schema.lines),
        ),
    )


def schema_split_lines(
    schema,
    kind: str,
    goal_slots,
    line_key_set: set[str],
    *,
    normalized_line_key,
    extract_fact_signature,
    goal_object_for_entity,
    extract_visible_anchor,
    extract_anchor_object_locations,
    extract_current_location,
    extract_pickup_object,
    extract_placement,
    extract_receptacle_state,
    normalize_fact_token,
    strip_xml_tags,
) -> tuple[str, ...]:
    present_lines = [
        line for line in schema.lines
        if normalized_line_key(line) in line_key_set
    ]
    if len(present_lines) < 2:
        return ()
    if len(present_lines) == 2:
        return tuple(present_lines)

    prioritized: list[str] = []
    if kind == "location":
        fact_lines = []
        anchor_lines = []
        arrival_lines = []
        placement_lines = []
        inventory_lines = []
        for line in present_lines:
            signature = extract_fact_signature(line)
            if signature is not None:
                subject, _ = signature
                if goal_object_for_entity(subject, goal_slots):
                    fact_lines.append(line)
                    continue
            if extract_anchor_object_locations(line, goal_slots):
                fact_lines.append(line)
                continue
            placement = extract_placement(line)
            if placement is not None:
                placed_object, _ = placement
                if goal_object_for_entity(placed_object, goal_slots):
                    placement_lines.append(line)
                    continue
            if extract_visible_anchor(line) is not None:
                anchor_lines.append(line)
                continue
            pickup_object = extract_pickup_object(line)
            if pickup_object and goal_object_for_entity(pickup_object, goal_slots):
                inventory_lines.append(line)
                continue
            if extract_current_location(line) is not None:
                arrival_lines.append(line)
        prioritized = placement_lines[:1] + fact_lines[:1] + anchor_lines[:1] + inventory_lines[:1] + arrival_lines[:1]
    elif kind == "progress":
        pickup_lines = [line for line in present_lines if extract_pickup_object(line)]
        placement_lines = [line for line in present_lines if extract_placement(line) is not None]
        state_lines = [line for line in present_lines if extract_receptacle_state(line) is not None]
        prioritized = pickup_lines[:1] + placement_lines[:1] + state_lines[:1]
    elif kind == "state":
        receptacle_lines = [line for line in present_lines if extract_receptacle_state(line) is not None]
        lexical_lines = [
            line for line in present_lines
            if any(token in normalize_fact_token(strip_xml_tags(line)) for token in ("steaming", "heated", "chilled", "cooled", "rinsed", "washed", "clean", "dirty", "illuminated", "bright", "dark"))
        ]
        prioritized = receptacle_lines[:1] + lexical_lines[:1]

    seen_keys: set[str] = set()
    compact: list[str] = []
    for line in prioritized + present_lines:
        line_key = normalized_line_key(line)
        if line_key in seen_keys:
            continue
        seen_keys.add(line_key)
        compact.append(line)
        if len(compact) >= 2:
            break
    return tuple(compact[:2]) if len(compact) >= 2 else tuple(present_lines[:2])


def evolve_schema_witnesses(
    skills,
    lines,
    goal_slots,
    *,
    schema_cls,
    normalized_line_key,
    schema_line_key,
    normalize_emergent_role_suffix,
    schema_template_signature_fn,
    schema_primary_subject_fn,
    best_schema_representative_fn,
    schema_split_lines_fn,
) -> None:
    line_key_set = {
        normalized_line_key(line)
        for line in lines
        if line and str(line).strip()
    }
    for kind in ("location", "progress", "state"):
        base_schemas = list(skills.schema_witnesses_by_skill.get(kind, ()))
        if not base_schemas:
            continue
        split_candidates = [
            schema for schema in base_schemas
            if not str(getattr(schema, "role", "")).startswith("typed_")
        ]
        existing_keys = {
            (schema.role, schema_line_key(schema.lines))
            for schema in base_schemas
        }
        additions: list[Any] = []

        for schema in split_candidates:
            split_lines = schema_split_lines_fn(schema, kind, goal_slots, line_key_set)
            if len(schema.lines) <= 2 or len(split_lines) < 2:
                continue
            if schema.coverage >= 0.999 and schema.conflict <= 0.20:
                continue
            split_role = f"split_{schema.role}"
            split_key = (split_role, schema_line_key(split_lines))
            if split_key in existing_keys:
                continue
            split_support = min(1.0, max(0.18, schema.support * (0.75 + 0.25 * schema.coverage)))
            split_uncertainty = max(0.10, min(1.0, schema.uncertainty * (0.80 if schema.coverage < 1.0 else 0.92)))
            additions.append(
                schema_cls(
                    lines=split_lines,
                    role=split_role,
                    confidence=min(1.0, max(0.45, schema.confidence * 0.92)),
                    support=split_support,
                    conflict=max(0.0, schema.conflict * 0.65),
                    freshness=schema.freshness,
                    eviction_pressure=max(0.0, schema.eviction_pressure * 0.55),
                    age=0,
                    hit_count=max(1, schema.hit_count),
                    miss_count=max(0, schema.miss_count // 2),
                    coverage=len(split_lines) / float(max(1, len(schema.lines))),
                    uncertainty=split_uncertainty,
                    lifecycle_stage="proto",
                )
            )
            existing_keys.add(split_key)

        grouped: dict[tuple[str, ...], list[Any]] = {}
        for schema in base_schemas:
            if schema.lifecycle_stage == "retired":
                continue
            signature = schema_template_signature_fn(schema, kind, goal_slots)
            if signature is None:
                continue
            grouped.setdefault(signature, []).append(schema)

        for signature, members in grouped.items():
            if len(members) < 2:
                continue
            representative = best_schema_representative_fn(members)
            general_role = f"generalized_{kind}_{'_'.join(signature[1:])}"
            general_key = (general_role, schema_line_key(representative.lines))
            if general_key in existing_keys:
                continue
            mean_support = sum(float(schema.support) for schema in members) / float(len(members))
            mean_conflict = sum(float(schema.conflict) for schema in members) / float(len(members))
            mean_uncertainty = sum(float(schema.uncertainty) for schema in members) / float(len(members))
            mean_coverage = sum(float(schema.coverage) for schema in members) / float(len(members))
            support_bonus = min(0.18, 0.06 * (len(members) - 1))
            generalized_support = min(1.0, mean_support + support_bonus)
            generalized_uncertainty = max(0.08, mean_uncertainty * max(0.55, 0.85 - 0.08 * (len(members) - 1)))
            generalized_conflict = max(0.0, mean_conflict * max(0.55, 0.90 - 0.12 * (len(members) - 1)))
            generalized_stage = (
                "stable"
                if generalized_support >= 0.62 and generalized_uncertainty <= 0.40 and generalized_conflict <= 0.28
                else "proto"
            )
            additions.append(
                schema_cls(
                    lines=representative.lines,
                    role=general_role,
                    confidence=min(1.0, max(0.50, representative.confidence + 0.05 * min(3, len(members) - 1))),
                    support=generalized_support,
                    conflict=generalized_conflict,
                    freshness=max(float(schema.freshness) for schema in members),
                    eviction_pressure=min(float(schema.eviction_pressure) for schema in members),
                    age=max(int(schema.age) for schema in members),
                    hit_count=sum(int(schema.hit_count) for schema in members),
                    miss_count=sum(int(schema.miss_count) for schema in members),
                    coverage=max(mean_coverage, float(representative.coverage)),
                    uncertainty=generalized_uncertainty,
                    lifecycle_stage=generalized_stage,
                )
            )
            existing_keys.add(general_key)

            subject_groups: dict[str, list[Any]] = {}
            for schema in members:
                subject_groups.setdefault(schema_primary_subject_fn(schema, kind, goal_slots), []).append(schema)
            for subject_key, subject_members in subject_groups.items():
                if len(subject_members) < 2:
                    continue
                subject_rep = best_schema_representative_fn(subject_members)
                merged_role = f"merged_{kind}_{'_'.join(signature[1:])}_{normalize_emergent_role_suffix(subject_key)}"
                merged_key = (merged_role, schema_line_key(subject_rep.lines))
                if merged_key in existing_keys:
                    continue
                merged_support = min(
                    1.0,
                    max(float(subject_rep.support), generalized_support) + min(0.12, 0.05 * (len(subject_members) - 1)),
                )
                merged_uncertainty = max(0.06, min(generalized_uncertainty, float(subject_rep.uncertainty) * 0.82))
                additions.append(
                    schema_cls(
                        lines=subject_rep.lines,
                        role=merged_role,
                        confidence=min(1.0, max(float(subject_rep.confidence), 0.55)),
                        support=merged_support,
                        conflict=max(0.0, float(subject_rep.conflict) * 0.75),
                        freshness=float(subject_rep.freshness),
                        eviction_pressure=float(subject_rep.eviction_pressure) * 0.70,
                        age=int(subject_rep.age),
                        hit_count=sum(int(schema.hit_count) for schema in subject_members),
                        miss_count=sum(int(schema.miss_count) for schema in subject_members),
                        coverage=float(subject_rep.coverage),
                        uncertainty=merged_uncertainty,
                        lifecycle_stage="stable" if merged_support >= 0.66 and merged_uncertainty <= 0.38 else "proto",
                    )
                )
                existing_keys.add(merged_key)

        if additions:
            skills.schema_witnesses_by_skill[kind] = tuple(base_schemas + additions)


def refresh_schema_witnesses(
    skills,
    lines,
    goal_slots,
    *,
    schema_last_line_index,
    schema_fact_signature,
    schema_state_signature,
    schema_progress_signature,
    goal_object_for_entity,
    extract_fact_signature,
    extract_anchor_object_locations,
    extract_placement,
    extract_pickup_object,
) -> None:
    if not lines:
        return

    line_index_by_key = {
        " ".join(str(line_text).lower().split()): index
        for index, line_text in enumerate(lines)
        if line_text and str(line_text).strip()
    }
    latest_location_by_subject: dict[str, tuple[int, str]] = {}
    latest_state_by_subject: dict[str, tuple[int, str]] = {}
    latest_progress_by_subject: dict[str, tuple[int, str]] = {}
    latest_inventory_by_subject: dict[str, tuple[int, str]] = {}
    latest_query_result_by_subject: dict[str, tuple[int, str]] = {}
    extract_search_query = schema_state_signature.__globals__.get("_extract_search_query")
    extract_information_payload = schema_state_signature.__globals__.get("_extract_information_payload")
    normalize_search_query_anchor = schema_state_signature.__globals__.get("_normalize_search_query_anchor")
    canonicalize_query_fact_value = schema_state_signature.__globals__.get("_canonicalize_query_fact_value")
    active_search_anchor = ""

    for index, line_text in enumerate(lines):
        lowered = " ".join(str(line_text).lower().split())
        if callable(extract_search_query):
            search_query = extract_search_query(line_text)
            if search_query and callable(normalize_search_query_anchor):
                active_search_anchor = normalize_search_query_anchor(search_query) or active_search_anchor
        current_location = schema_state_signature.__globals__["_extract_current_location"](line_text)
        if current_location:
            latest_location_by_subject["agent"] = (index, current_location)
        signature = extract_fact_signature(line_text)
        if signature is not None:
            subject, value = signature
            goal_object = goal_object_for_entity(subject, goal_slots)
            if goal_object:
                latest_location_by_subject[goal_object] = (index, value)
        if active_search_anchor and callable(extract_information_payload) and callable(canonicalize_query_fact_value):
            info_payload = extract_information_payload(line_text)
            fact_value = canonicalize_query_fact_value(info_payload) if info_payload else ""
            if fact_value:
                latest_query_result_by_subject[active_search_anchor] = (index, fact_value)
        for subject, value in extract_anchor_object_locations(line_text, goal_slots):
            latest_location_by_subject[subject] = (index, value)
        placement = extract_placement(line_text)
        if placement is not None:
            placed_object, receptacle = placement
            goal_object = goal_object_for_entity(placed_object, goal_slots)
            if goal_object:
                latest_location_by_subject[goal_object] = (index, receptacle)
        pickup_object = extract_pickup_object(line_text)
        if pickup_object:
            goal_object = goal_object_for_entity(pickup_object, goal_slots)
            if goal_object:
                latest_location_by_subject[goal_object] = (index, "inventory")
                latest_inventory_by_subject["agent"] = (index, goal_object)
        receptacle_state = schema_state_signature.__globals__["_extract_receptacle_state"](line_text)
        if receptacle_state is not None:
            receptacle, is_open = receptacle_state
            latest_state_by_subject[receptacle] = (index, "open" if is_open else "closed")
        for token, state_name in (
            ("steaming", "heated"),
            ("heated", "heated"),
            ("chilled", "cooled"),
            ("cooled", "cooled"),
            ("rinsed", "cleaned"),
            ("washed", "cleaned"),
            ("clean", "cleaned"),
            ("dirty", "dirty"),
            ("illuminated", "light_on"),
            ("bright", "light_on"),
            ("dark", "light_off"),
        ):
            if token in lowered:
                subject = next(iter(goal_slots.target_objects), "environment")
                latest_state_by_subject[subject] = (index, state_name)
        placement = extract_placement(line_text)
        if placement is not None:
            placed_object, receptacle = placement
            goal_object = goal_object_for_entity(placed_object, goal_slots)
            if goal_object:
                latest_progress_by_subject[goal_object] = (index, f"placed:{receptacle}")
                latest_inventory_by_subject["agent"] = (index, "empty")
        pickup_object = extract_pickup_object(line_text)
        if pickup_object:
            goal_object = goal_object_for_entity(pickup_object, goal_slots)
            if goal_object:
                latest_progress_by_subject[goal_object] = (index, "holding")

    for kind in ("location", "progress", "state"):
        refreshed: list[Any] = []
        for schema in skills.schema_witnesses_by_skill.get(kind, ()):
            matched_lines = sum(1 for line in schema.lines if " ".join(str(line).lower().split()) in line_index_by_key)
            coverage = matched_lines / float(max(1, len(schema.lines)))
            schema_last_idx = schema_last_line_index(schema, line_index_by_key)
            recency_ratio = float(schema_last_idx + 1) / float(max(1, len(lines))) if schema_last_idx >= 0 else 0.0
            age = int(schema.age) + 1
            hit_count = int(schema.hit_count) + (1 if coverage >= 0.67 else 0)
            miss_count = int(schema.miss_count) + (1 if coverage < 0.67 else 0)
            freshness = min(1.0, max(0.10, 0.35 + 0.65 * recency_ratio))
            support = min(
                1.0,
                0.18
                + 0.24 * min(3, len(schema.lines))
                + 0.18 * float(schema.confidence)
                + 0.22 * coverage
                + 0.10 * min(1.0, hit_count / float(max(1, age))),
            )
            conflict = 0.0
            replacement_strength = 0.0
            if kind == "location":
                explicit_subject = str(getattr(schema, "subject", "") or "")
                explicit_value = str(getattr(schema, "current_value", "") or "")
                relation_family = str(getattr(schema, "relation_family", "") or "")
                if explicit_subject and explicit_value and relation_family in {"agent_room", "object_location", "device_location"}:
                    latest_location = latest_location_by_subject.get(explicit_subject)
                    if latest_location is not None:
                        latest_index, latest_value = latest_location
                        if latest_index > schema_last_idx and latest_value != explicit_value:
                            conflict = max(conflict, 0.90)
                            replacement_strength = max(replacement_strength, 0.85)
                elif explicit_subject and explicit_value and relation_family == "query_result":
                    latest_query_result = latest_query_result_by_subject.get(explicit_subject)
                    if latest_query_result is not None:
                        latest_index, latest_value = latest_query_result
                        if latest_index > schema_last_idx and latest_value != explicit_value:
                            conflict = max(conflict, 0.88)
                            replacement_strength = max(replacement_strength, 0.82)
                else:
                    signature = schema_fact_signature(schema, goal_slots)
                    if signature is not None:
                        subject, value = signature
                        goal_object = goal_object_for_entity(subject, goal_slots) or subject
                        latest_location = latest_location_by_subject.get(goal_object)
                        if latest_location is not None:
                            latest_index, latest_value = latest_location
                            if latest_index > schema_last_idx and latest_value != value:
                                conflict = max(conflict, 0.90)
                                replacement_strength = max(replacement_strength, 0.85)
            elif kind == "state":
                explicit_subject = str(getattr(schema, "subject", "") or "")
                explicit_value = str(getattr(schema, "current_value", "") or "")
                relation_family = str(getattr(schema, "relation_family", "") or "")
                if explicit_subject and explicit_value and relation_family in {"receptacle_state", "object_state", "light_state", "emergent_state"}:
                    latest_state = latest_state_by_subject.get(explicit_subject)
                    if latest_state is not None:
                        latest_index, latest_value = latest_state
                        if latest_index > schema_last_idx and latest_value != explicit_value:
                            conflict = max(conflict, 0.85)
                            replacement_strength = max(replacement_strength, 0.80)
                else:
                    signature = schema_state_signature(schema, goal_slots)
                    if signature is not None:
                        subject, value = signature
                        latest_state = latest_state_by_subject.get(subject)
                        if latest_state is not None:
                            latest_index, latest_value = latest_state
                            if latest_index > schema_last_idx and latest_value != value:
                                conflict = max(conflict, 0.85)
                                replacement_strength = max(replacement_strength, 0.80)
            elif kind == "progress":
                explicit_subject = str(getattr(schema, "subject", "") or "")
                explicit_value = str(getattr(schema, "current_value", "") or "")
                relation_family = str(getattr(schema, "relation_family", "") or "")
                if explicit_subject and explicit_value and relation_family == "inventory":
                    latest_progress = latest_inventory_by_subject.get(explicit_subject)
                    if latest_progress is not None:
                        latest_index, latest_value = latest_progress
                        if latest_index > schema_last_idx and latest_value != explicit_value:
                            conflict = max(conflict, 0.82)
                            replacement_strength = max(replacement_strength, 0.78)
                elif explicit_subject and explicit_value and relation_family == "task_progress":
                    latest_progress = latest_progress_by_subject.get(explicit_subject)
                    if latest_progress is not None:
                        latest_index, latest_value = latest_progress
                        if latest_index > schema_last_idx and latest_value != explicit_value:
                            conflict = max(conflict, 0.80)
                            replacement_strength = max(replacement_strength, 0.75)
                else:
                    signature = schema_progress_signature(schema, goal_slots)
                    if signature is not None:
                        subject, value = signature
                        latest_progress = latest_progress_by_subject.get(subject)
                        if latest_progress is not None:
                            latest_index, latest_value = latest_progress
                            if latest_index > schema_last_idx and latest_value != value:
                                conflict = max(conflict, 0.80)
                                replacement_strength = max(replacement_strength, 0.75)
            if conflict > 0.0:
                freshness = max(0.10, freshness * (1.0 - 0.65 * conflict))
                support = max(0.05, support * (1.0 - 0.45 * conflict))
            eviction_pressure = min(1.0, max(0.0, 0.65 * conflict + 0.35 * replacement_strength) * max(0.0, 1.0 - freshness))
            uncertainty = schema_uncertainty(
                confidence=schema.confidence,
                support=support,
                conflict=conflict,
                freshness=freshness,
                coverage=coverage,
                hit_count=hit_count,
                miss_count=miss_count,
            )
            lifecycle = schema_lifecycle_stage(
                support=support,
                conflict=conflict,
                freshness=freshness,
                eviction_pressure=eviction_pressure,
                uncertainty=uncertainty,
                age=age,
                hit_count=hit_count,
                miss_count=miss_count,
            )
            refreshed.append(
                replace(
                    schema,
                    support=support,
                    conflict=conflict,
                    freshness=freshness,
                    eviction_pressure=eviction_pressure,
                    age=age,
                    hit_count=hit_count,
                    miss_count=miss_count,
                    coverage=coverage,
                    uncertainty=uncertainty,
                    lifecycle_stage=lifecycle,
                )
            )
        skills.schema_witnesses_by_skill[kind] = tuple(refreshed)
