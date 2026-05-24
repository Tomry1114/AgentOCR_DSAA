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

import importlib.util
import re
from functools import lru_cache
from pathlib import Path


# --------------------- ALFWorld --------------------- #
ALFWORLD_QWEN3VL_SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED embodied Environment."
)


ALFWORLD_ACTION_RULES = """
Rules:
1. Focus on the task goal, not just the most recently mentioned or most visible object.
2. If the current object is unrelated to the task goal, prefer an exploration or navigation action that helps find the goal object or target location.
3. Never output an action that is not listed in the admissible actions.
4. Before finalizing, verify that the action string appears exactly, character-for-character, in the admissible actions list.
5. If the ideal action is unavailable, choose the listed exploration or navigation action that best advances the task goal.
6. Keep the reasoning brief and task-directed.
7. Put all reasoning only inside <thinking>...</thinking>.
8. Do not use headings or free-form prefixes such as "Reasoning:", "Thought:", or "Analysis:".
9. After reasoning, output exactly one action inside <action>...</action>.
10. Do not output any free-form text outside the required tags.
11. The output format is strict:
first one <thinking>...</thinking> block, then one <action>...</action> block.
12. Inside <action>...</action>, write the exact admissible action string.
13. Do not copy wording from these instructions into the final answer.
""".strip()


ALFWORLD_QWEN3_ACTION_RULES = """
Rules:
1. Focus on the task goal and choose the best next admissible action.
2. If the goal requires a state change such as clean, cool, heat, or light, do not treat final placement or examination as complete until that required state change is done.
3. Keep the reasoning short and task-directed.
4. Reason briefly inside one <thinking>...</thinking> block.
5. Then output exactly one <action>...</action> block.
6. Inside <action>...</action>, copy one admissible action exactly, character for character.
7. Do not output free-form text outside the required tags.
""".strip()


ALFWORLD_COMPRESSION_RULE = (
    "12. Additionally, choose a compression factor larger than or equal to 1.0 for the next image "
    "and put it inside <compression>...</compression> after the action block.\n"
    "13. If uncertain, output 1.0 to preserve readability.\n"
    "14. Inside <compression>...</compression>, write a concrete numeric value such as 1.0.\n"
    "15. Do not echo instruction text inside <compression>...</compression>."
)


ALFWORLD_QWEN3_COMPRESSION_RULE = (
    "After <action>...</action>, output one <compression>...</compression> block for the next image.\n"
    "Use a numeric value greater than or equal to 1.0, such as 1.0.\n"
    "If uncertain, output 1.0."
)


def is_qwen3_vl_model_path(model_path) -> bool:
    if not model_path:
        return False
    normalized = str(model_path).lower()
    return (
        "qwen3-vl" in normalized
        or "qwen3vl" in normalized
        or "qwen3.5" in normalized
        or "qwen3_5" in normalized
    )


_GOAL_STATE_WORDS = {
    "clean",
    "cool",
    "cold",
    "hot",
    "heated",
    "cooled",
    "sliced",
    "slice",
}
_GOAL_QUANTITY_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
_GOAL_STOPWORDS = {"a", "an", "the", "some", "of", "it", "under", "on", "in", "and"} | _GOAL_QUANTITY_WORDS


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


def _canonicalize_goal_object(phrase: str, *, singularize_plural: bool = False) -> str:
    tokens = [token for token in re.findall(r"[a-z0-9]+", str(phrase or "").lower()) if token]
    filtered = [
        _singularize_goal_token(token) if singularize_plural else token
        for token in tokens
        if token not in _GOAL_STOPWORDS and token not in _GOAL_STATE_WORDS
    ]
    return " ".join(filtered).strip()


def _extract_goal_terms(phrase: str) -> list[str]:
    canonical = _canonicalize_goal_object(phrase)
    if not canonical:
        return []
    terms = []
    for token in canonical.split():
        if token not in terms:
            terms.append(token)
    return terms


def _text_mentions_any_term(text: str, terms: list[str]) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in terms if term)


def _extract_alfworld_goal_slots(task_description: str) -> dict:
    task = str(task_description or "").strip().lower()
    normalized = re.sub(r"\s+", " ", task)
    multi_object_goal = bool(re.search(r"^put\s+(?:two|\d+)\b", normalized))

    target_object = ""
    target_receptacle = ""
    required_state = ""
    required_tool = ""

    official_patterns = [
        (
            r"^put (?:a|some|two|\d+) (?P<object>.+?) (?:in|on) (?P<receptacle>.+?)(?:\.|$)",
            None,
            None,
        ),
        (
            r"^clean some (?P<object>.+?) and put it in (?P<receptacle>.+?)(?:\.|$)",
            "clean",
            "sinkbasin",
        ),
        (
            r"^heat some (?P<object>.+?) and put it in (?P<receptacle>.+?)(?:\.|$)",
            "heat",
            "microwave",
        ),
        (
            r"^cool some (?P<object>.+?) and put it in (?P<receptacle>.+?)(?:\.|$)",
            "cool",
            "fridge",
        ),
        (
            r"^look at (?P<object>.+?) under the (?P<toggle>.+?)(?:\.|$)",
            "light",
            None,
        ),
    ]
    for pattern, state_name, default_tool in official_patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        target_object = match.groupdict().get("object", "").strip()
        target_receptacle = match.groupdict().get("receptacle", "").strip()
        toggle_name = match.groupdict().get("toggle", "").strip()
        if state_name:
            required_state = state_name
        if default_tool:
            required_tool = default_tool
        elif toggle_name:
            required_tool = toggle_name
        break

    if not target_object:
        object_match = re.search(
            r"(?:put|clean|cool|heat|look at)\s+(?:some\s+|a\s+|two\s+|\d+\s+)?(.+?)(?:\s+and\s+put|\s+on\s+|\s+in\s+|\s+under\s+|\.|$)",
            normalized,
        )
        if object_match:
            target_object = object_match.group(1).strip()

    if not target_receptacle:
        receptacle_match = re.search(
            r"\bput(?:\s+(?:some|a|two|\d+)\s+.+?)?\s+(?:it\s+)?(?:on|in)\s+(.+?)(?:\.|$)",
            normalized,
        )
        if receptacle_match:
            target_receptacle = receptacle_match.group(1).strip()

    if not required_state:
        if normalized.startswith("clean ") or " put a clean " in f" {normalized} ":
            required_state = "clean"
            required_tool = required_tool or "sinkbasin"
        elif normalized.startswith("cool ") or " put a cool " in f" {normalized} " or " put a cold " in f" {normalized} ":
            required_state = "cool"
            required_tool = required_tool or "fridge"
        elif normalized.startswith("heat ") or " put a hot " in f" {normalized} ":
            required_state = "heat"
            required_tool = required_tool or "microwave"
        elif " under the " in normalized and ("lamp" in normalized or "light" in normalized):
            required_state = "light"

    target_object = _canonicalize_goal_object(target_object, singularize_plural=multi_object_goal)
    target_receptacle = _canonicalize_goal_object(target_receptacle)
    required_tool = _canonicalize_goal_object(required_tool)

    return {
        "target_object": target_object,
        "target_receptacle": target_receptacle,
        "required_state": required_state,
        "required_tool": required_tool,
        "target_object_terms": _extract_goal_terms(target_object),
        "target_receptacle_terms": _extract_goal_terms(target_receptacle),
        "required_tool_terms": _extract_goal_terms(required_tool),
    }


def build_alfworld_goal_hints(task_description: str, model_path=None) -> str:
    if not is_qwen3_vl_model_path(model_path):
        return ""

    slots = _extract_alfworld_goal_slots(task_description)
    lines = ["Goal summary:"]
    if slots["target_object"]:
        lines.append(f"- target object: {slots['target_object']}")
    if slots["target_receptacle"]:
        lines.append(f"- target location: {slots['target_receptacle']}")
    if slots["required_state"]:
        state_line = f"- required state: {slots['required_state']}"
        if slots["required_tool"]:
            state_line += f" via {slots['required_tool']}"
        lines.append(state_line)
    if slots["target_object"] and not slots["required_state"]:
        lines.append(
            f"- if {slots['target_object']} is not visible, prefer admissible locations that can actually hold it"
        )
    lines.append("- prefer task-relevant navigation over unrelated object interactions")
    return "\n".join(lines)


def _strip_embedded_task_text(text: str) -> str:
    cleaned_lines = []
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if lowered.startswith("your task is to:"):
            continue
        if lowered.startswith("task:"):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def _context_has_object_state_completion(
    context_text: str,
    *,
    object_terms: list[str],
    required_state: str,
    tool_terms: list[str],
) -> bool:
    lowered = str(context_text or "").lower()
    if not lowered or not required_state:
        return False

    object_terms = [term for term in object_terms if term]
    tool_terms = [term for term in tool_terms if term]

    if required_state == "light":
        for tool_term in tool_terms:
            escaped = re.escape(tool_term)
            if re.search(rf"\b{escaped}\b[^.\n]{{0,32}}\b(on|turned on|is on|now on)\b", lowered):
                return True
            if re.search(rf"\b(turn on|use)\b[^.\n]{{0,24}}\b{escaped}\b", lowered):
                return True
        return False

    state_words_by_family = {
        "clean": ("clean", "cleaned", "wash", "washed", "rinse", "rinsed"),
        "cool": ("cool", "cooled", "cold", "chilled"),
        "heat": ("heat", "heated", "hot", "warmed", "warm"),
    }
    state_words = state_words_by_family.get(required_state, ())
    if not state_words or not object_terms:
        return False

    for object_term in object_terms:
        escaped_object = re.escape(object_term)
        for state_word in state_words:
            escaped_state = re.escape(state_word)
            if re.search(rf"\b{escaped_object}\b[^.\n]{{0,32}}\b{escaped_state}\b", lowered):
                return True
            if re.search(rf"\b{escaped_state}\b[^.\n]{{0,32}}\b{escaped_object}\b", lowered):
                return True
    return False


def build_alfworld_state_strategy_hint(
    task_description: str,
    current_observation: str,
    action_history: str,
    model_path=None,
) -> str:
    if not is_qwen3_vl_model_path(model_path):
        return ""

    slots = _extract_alfworld_goal_slots(task_description)
    required_state = slots["required_state"]
    if required_state not in {"clean", "cool", "heat", "light"}:
        return ""

    state_context = "\n".join(
        part for part in (str(action_history or ""), _strip_embedded_task_text(current_observation)) if part
    )
    state_complete = _context_has_object_state_completion(
        state_context,
        object_terms=slots["target_object_terms"],
        required_state=required_state,
        tool_terms=slots["required_tool_terms"],
    )
    target_object_grounded = _context_has_target_object_grounding(
        state_context,
        slots["target_object_terms"],
    )

    target_object = slots["target_object"] or "target object"
    target_receptacle = slots["target_receptacle"] or "the final goal location"
    required_tool = slots["required_tool"] or "the required device"

    lines = ["State-task strategy:"]
    lines.append(f"- Required state: {required_state} via {required_tool}.")
    if state_complete:
        lines.append("- The required state change is already complete.")
        lines.append(f"- Current subgoal: prioritize the final goal for {target_object}, such as {target_receptacle}.")
    else:
        lines.append(
            f"- While the required state is still pending, do not treat final placement or examination as success for {target_object}."
        )
        if target_object_grounded:
            lines.append(
                f"- The target {target_object} is already grounded in the recent context; do not restart broad search."
            )
            lines.append(
                f"- Current subgoal: keep {target_object} and use {required_tool} before finishing at {target_receptacle}."
            )
        else:
            lines.append(
                f"- Before {target_object} is found, search compatible object locations first; {required_tool} is for the state change after you have {target_object}."
            )
            arrived_match = re.search(r"you arrive at ([^.]+)", str(current_observation or "").lower())
            arrived_location = arrived_match.group(1).strip() if arrived_match else ""
            if (
                arrived_location
                and _location_family_can_hold_target_object(arrived_location, target_object) is True
                and _location_family_is_openable(arrived_location)
                and _observation_location_has_state(current_observation, arrived_location, "closed")
            ):
                lines.append(f"- The current {arrived_location} is a compatible closed container; open it before leaving.")
            lines.append(
                f"- Current subgoal: find or keep {target_object}, then use {required_tool} before finishing at {target_receptacle}."
            )
    return "\n".join(lines)


def build_alfworld_history_hint(action_history: str, model_path=None, max_lines: int = 4) -> str:
    if not is_qwen3_vl_model_path(model_path):
        return ""

    history = str(action_history or "").strip()
    if not history or history.lower() == "none":
        return ""

    lines = [line.strip() for line in history.splitlines() if line.strip()]
    if not lines:
        return ""

    recent_lines = lines[-max_lines:]
    return "Recent trajectory summary (text backup):\n" + "\n".join(recent_lines)


def build_alfworld_failure_hint(
    action_history: str,
    current_observation: str,
    admissible_actions,
    model_path=None,
    task_description: str = "",
) -> str:
    if not is_qwen3_vl_model_path(model_path):
        return ""

    slots = _extract_alfworld_goal_slots(task_description)
    stripped_observation = _strip_embedded_task_text(current_observation)
    search_context = "\n".join(part for part in (str(action_history or ""), stripped_observation) if part)
    target_object_grounded = _context_has_target_object_grounding(search_context, slots["target_object_terms"])
    target_object_visible = _observation_mentions_target_object(stripped_observation, slots["target_object_terms"])
    state_complete = _context_has_object_state_completion(
        search_context,
        object_terms=slots["target_object_terms"],
        required_state=slots["required_state"],
        tool_terms=slots["required_tool_terms"],
    )
    target_state_action = _select_target_state_action(
        admissible_actions,
        target_object_terms=slots["target_object_terms"],
        required_state=slots["required_state"],
        required_tool_terms=slots["required_tool_terms"],
    )
    direct_target_examine_action = _select_target_direct_examine_action(
        admissible_actions,
        target_object_terms=slots["target_object_terms"],
    )
    target_take_action = _select_target_take_action(
        admissible_actions,
        target_object_terms=slots["target_object_terms"],
    )
    target_take_from_required_tool_action = _select_target_take_action(
        admissible_actions,
        target_object_terms=slots["target_object_terms"],
        preferred_location_terms=slots["required_tool_terms"],
    )
    searched_without_target_locations = []
    if not slots["required_state"]:
        searched_without_target_locations = sorted(
            _extract_searched_without_target_locations(
                action_history,
                target_object_terms=slots["target_object_terms"],
            ),
            key=_alfworld_location_sort_key,
        )
    next_search_action = _select_next_unseen_search_action(
        admissible_actions,
        set(searched_without_target_locations),
        target_object=slots["target_object"],
    )
    compatible_families = []
    incompatible_families = []
    for action in admissible_actions or []:
        action_location = _extract_alfworld_action_location(action)
        if not action_location:
            continue
        family_name = _alfworld_location_family_name(action_location)
        if not family_name:
            continue
        compatibility = _location_family_can_hold_target_object(action_location, slots["target_object"])
        if compatibility is True and family_name not in compatible_families:
            compatible_families.append(family_name)
        elif compatibility is False and family_name not in incompatible_families:
            incompatible_families.append(family_name)
    proactive_search_guidance = bool(
        not slots["required_state"]
        and slots["target_object"]
        and not target_object_grounded
        and not target_object_visible
        and compatible_families
        and incompatible_families
    )
    recent_actions = _extract_recent_actions(action_history, max_actions=3)
    lowered_obs = str(current_observation or "").lower()
    current_action_pool = {str(action).lower() for action in (admissible_actions or [])}
    repeated_recently = len(recent_actions) >= 2 and recent_actions[-1] == recent_actions[-2]
    failed_recently = "nothing happens" in lowered_obs or repeated_recently
    if not failed_recently and not searched_without_target_locations and not proactive_search_guidance:
        return ""

    lines = ["Failure avoidance:"]
    if recent_actions:
        last_action = recent_actions[-1]
        if failed_recently:
            if last_action not in current_action_pool:
                lines.append(f'- do not repeat "{last_action}"; it is not in the current admissible list')
                if (
                    slots["required_state"] in {"clean", "cool", "heat"}
                    and not state_complete
                    and target_state_action
                ):
                    lines.append(
                        f'- the direct {slots["required_state"]} action is already available; prefer "{target_state_action}"'
                    )
                elif target_take_from_required_tool_action and last_action.startswith("open "):
                    lines.append(
                        f'- the required tool is already accessible; prefer "{target_take_from_required_tool_action}"'
                    )
                elif (
                    slots["required_state"] == "light"
                    and state_complete
                    and target_take_action
                    and not direct_target_examine_action
                ):
                    lines.append(
                        f'- direct target examine is unavailable; prefer the listed proxy action "{target_take_action}"'
                    )
                elif target_take_action and target_object_visible:
                    lines.append(f'- if you need {slots["target_object"]}, prefer "{target_take_action}"')
            else:
                lines.append(f'- avoid repeating the failed recent action "{last_action}"')
    if proactive_search_guidance:
        lines.append(f"- when searching for {slots['target_object']}, prefer compatible locations from the admissible list")
        lines.append(
            f"- {compatible_families[0]} can hold {slots['target_object']}, but {incompatible_families[0]} cannot"
        )
        if next_search_action:
            lines.append(f'- if you continue searching, prefer "{next_search_action}"')
    if searched_without_target_locations:
        displayed_locations = ", ".join(searched_without_target_locations[:4])
        lines.append(f"- already searched without target: {displayed_locations}")
        if slots["target_object"]:
            lines.append(
                f"- prefer an unseen compatible location for {slots['target_object']} before revisiting a searched-empty one"
            )
        else:
            lines.append("- prefer an unseen candidate location before revisiting a searched-empty one")
        if next_search_action:
            lines.append(f'- if you continue searching, prefer "{next_search_action}"')
    lines.append("- copy a different action exactly from the admissible actions above")
    return "\n".join(lines)


def _extract_recent_actions(action_history: str, max_actions: int = 4) -> list[str]:
    history = str(action_history or "").strip()
    if not history or history.lower() == "none":
        return []

    actions = []
    for line in history.splitlines():
        match = re.search(r"\[Action\]:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
        if match:
            actions.append(match.group(1).strip().lower())
    return actions[-max_actions:]


def _normalize_alfworld_location_name(location: str) -> str:
    normalized = str(location or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.startswith("the "):
        normalized = normalized[4:].strip()
    return normalized


def _alfworld_entity_symbol(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _alfworld_location_family_symbol(location: str) -> str:
    normalized = _normalize_alfworld_location_name(location)
    match = re.fullmatch(r"(.+?)\s+\d+", normalized)
    family = match.group(1) if match else normalized
    return _alfworld_entity_symbol(family)


def _alfworld_location_family_name(location: str) -> str:
    normalized = _normalize_alfworld_location_name(location)
    match = re.fullmatch(r"(.+?)\s+\d+", normalized)
    return (match.group(1) if match else normalized).strip()


@lru_cache(maxsize=1)
def _load_alfworld_search_metadata() -> tuple[dict[str, set[str]], set[str]]:
    constants_path = (
        Path(__file__).resolve().parents[1]
        / "env_package"
        / "alfworld"
        / "alfworld"
        / "gen"
        / "constants.py"
    )
    spec = importlib.util.spec_from_file_location("agentocr_alfworld_constants", constants_path)
    if spec is None or spec.loader is None:
        return {}, set()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    raw_affordances = getattr(module, "VAL_RECEPTACLE_OBJECTS", {})
    affordances: dict[str, set[str]] = {}
    for receptacle_name, object_names in raw_affordances.items():
        receptacle_symbol = _alfworld_entity_symbol(receptacle_name)
        if not receptacle_symbol:
            continue
        affordances[receptacle_symbol] = {
            symbol
            for symbol in (_alfworld_entity_symbol(object_name) for object_name in object_names)
            if symbol
        }

    openable_families = {
        symbol
        for symbol in (
            _alfworld_entity_symbol(name)
            for name in getattr(module, "OPENABLE_CLASS_LIST", ())
        )
        if symbol
    }
    return affordances, openable_families


def _location_family_can_hold_target_object(location: str, target_object: str) -> bool | None:
    target_symbol = _alfworld_entity_symbol(target_object)
    family_symbol = _alfworld_location_family_symbol(location)
    if not target_symbol or not family_symbol:
        return None
    affordances, _ = _load_alfworld_search_metadata()
    allowed_objects = affordances.get(family_symbol)
    if allowed_objects is None:
        return None
    return target_symbol in allowed_objects


def _location_family_is_openable(location: str) -> bool:
    family_symbol = _alfworld_location_family_symbol(location)
    if not family_symbol:
        return False
    _, openable_families = _load_alfworld_search_metadata()
    return family_symbol in openable_families


def _extract_alfworld_action_location(action: str) -> str:
    match = re.match(
        r"^(?:go to|examine|inspect|look at|open|close)\s+(.+)$",
        str(action or "").strip().lower(),
    )
    if not match:
        return ""
    return _normalize_alfworld_location_name(match.group(1))


def _extract_numbered_location_key(location: str):
    normalized = _normalize_alfworld_location_name(location)
    match = re.fullmatch(r"([a-z]+(?:\s+[a-z]+)*)\s+(\d+)", normalized)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _alfworld_location_sort_key(location: str) -> tuple[str, int, str]:
    numbered = _extract_numbered_location_key(location)
    if numbered:
        prefix, index = numbered
        return (prefix, index, location)
    return (_normalize_alfworld_location_name(location), 10**9, location)


def _action_mentions_target_object(action: str, target_object_terms: list[str]) -> bool:
    return _text_mentions_any_term(action, target_object_terms)


def _observation_mentions_target_object(observation_text: str, object_terms: list[str]) -> bool:
    return _text_mentions_any_term(observation_text, object_terms)


def _select_target_state_action(
    admissible_actions,
    *,
    target_object_terms: list[str],
    required_state: str,
    required_tool_terms: list[str],
) -> str:
    if required_state not in {"clean", "cool", "heat"}:
        return ""
    for action in admissible_actions or []:
        lowered = str(action).lower()
        if not lowered.startswith(f"{required_state} "):
            continue
        if target_object_terms and not _action_mentions_target_object(action, target_object_terms):
            continue
        if required_tool_terms and not _text_mentions_any_term(action, required_tool_terms):
            continue
        return str(action)
    return ""


def _select_target_direct_examine_action(admissible_actions, *, target_object_terms: list[str]) -> str:
    for action in admissible_actions or []:
        lowered = str(action).lower()
        if not lowered.startswith(("examine ", "look at ")):
            continue
        if target_object_terms and not _action_mentions_target_object(action, target_object_terms):
            continue
        return str(action)
    return ""


def _select_target_take_action(
    admissible_actions,
    *,
    target_object_terms: list[str],
    preferred_location_terms: list[str] | None = None,
) -> str:
    preferred_location_terms = [term for term in (preferred_location_terms or []) if term]
    fallback_action = ""
    for action in admissible_actions or []:
        lowered = str(action).lower()
        if not lowered.startswith("take "):
            continue
        if target_object_terms and not _action_mentions_target_object(action, target_object_terms):
            continue
        if preferred_location_terms and _text_mentions_any_term(action, preferred_location_terms):
            return str(action)
        if not fallback_action:
            fallback_action = str(action)
    return fallback_action


def _action_pool_implies_location_open(admissible_actions, location: str) -> bool:
    normalized_location = _normalize_alfworld_location_name(location)
    if not normalized_location:
        return False
    escaped_location = re.escape(normalized_location)
    for action in admissible_actions or []:
        lowered = str(action).lower()
        if lowered == f"close {normalized_location}":
            return True
        if re.match(rf"^take .+ from {escaped_location}$", lowered):
            return True
    return False


def _context_has_target_object_grounding(context_text: str, object_terms: list[str]) -> bool:
    lowered = str(context_text or "").lower()
    if not lowered:
        return False

    for term in object_terms:
        escaped = re.escape(term)
        patterns = (
            rf"\byou pick up\b[^.\n]{{0,32}}\b{escaped}\b",
            rf"\byou are carrying\b[^.\n]{{0,32}}\b{escaped}\b",
            rf"\bholding\b[^.\n]{{0,32}}\b{escaped}\b",
            rf"\b{escaped}\b[^.\n]{{0,32}}\b(?:is in|is on|is inside|is at)\b",
        )
        if any(re.search(pattern, lowered) for pattern in patterns):
            return True
    return False


def _observation_location_has_state(observation_text: str, location: str, state: str) -> bool:
    normalized_location = _normalize_alfworld_location_name(location)
    if not normalized_location:
        return False
    lowered = str(observation_text or "").lower()
    escaped = re.escape(normalized_location)
    return bool(re.search(rf"\b{escaped}\b[^.\n]{{0,24}}\bis {re.escape(state)}\b", lowered))


def _extract_searched_without_target_locations(
    action_history: str,
    *,
    target_object_terms: list[str],
) -> set[str]:
    history = str(action_history or "").strip()
    if not history or history.lower() == "none":
        return set()

    searched_locations: set[str] = set()
    found_locations: set[str] = set()

    for raw_line in history.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()

        for skill_match in re.finditer(r"searched_without_target=([a-z0-9_,\s]+)", lowered):
            for token in skill_match.group(1).split(","):
                normalized = _normalize_alfworld_location_name(token)
                if normalized:
                    searched_locations.add(normalized)

        action_match = re.search(r"\[action\]:\s*(.+)$", line, flags=re.IGNORECASE)
        action_text = action_match.group(1).strip() if action_match else ""
        action_location = _extract_alfworld_action_location(action_text)

        location_candidates = []
        for pattern in (
            r"\byou arrive at ([^.]+?)(?:\.|,|$)",
            r"\bon the ([^.]+?), you see\b",
            r"\bin the ([^.]+?), you see\b",
            r"\byou inspect ([^.]+?)(?:\.|,|$)",
            r"\bthe ([^.]+?) is empty\b",
            r"\bthe ([^.]+?) is open\b",
            r"\bthe ([^.]+?) is closed\b",
        ):
            for match in re.finditer(pattern, lowered):
                normalized = _normalize_alfworld_location_name(match.group(1))
                if normalized:
                    location_candidates.append(normalized)
        if action_location:
            location_candidates.append(action_location)

        location_candidates = list(dict.fromkeys(location_candidates))
        if not location_candidates:
            continue

        target_found_here = any(
            re.search(rf"\b{re.escape(term)}\b", lowered)
            for term in target_object_terms
            if term
        )
        if target_found_here:
            for location in location_candidates:
                found_locations.add(location)
                searched_locations.discard(location)
            continue

        empty_or_absent = (
            "see nothing" in lowered
            or " is empty" in lowered
            or " it is empty" in lowered
        )
        explicit_search_observation = (
            "you see" in lowered
            or "you inspect" in lowered
            or "you arrive at" in lowered
        )
        if empty_or_absent or (
            explicit_search_observation and not _action_mentions_target_object(action_text, target_object_terms)
        ):
            for location in location_candidates:
                searched_locations.add(location)

    return searched_locations - found_locations


def _select_next_unseen_search_action(
    admissible_actions,
    searched_without_target_locations: set[str],
    *,
    target_object: str = "",
) -> str:
    searched_prefixes = {
        numbered[0]
        for numbered in (
            _extract_numbered_location_key(location)
            for location in searched_without_target_locations
        )
        if numbered
    }

    family_counts: dict[str, int] = {}
    for action in admissible_actions or []:
        family_symbol = _alfworld_location_family_symbol(_extract_alfworld_action_location(action))
        if family_symbol:
            family_counts[family_symbol] = family_counts.get(family_symbol, 0) + 1

    ranked_candidates = []
    for index, action in enumerate(admissible_actions or []):
        action_location = _extract_alfworld_action_location(action)
        if not action_location:
            continue
        numbered = _extract_numbered_location_key(action_location)
        prefix = numbered[0] if numbered else ""
        location_index = numbered[1] if numbered else 10**9
        compatibility = _location_family_can_hold_target_object(action_location, target_object)
        openable_rank = 0 if compatibility is True and _location_family_is_openable(action_location) else 1
        compatibility_rank = 0 if compatibility is True else 1 if compatibility is None else 2
        family_count_rank = family_counts.get(_alfworld_location_family_symbol(action_location), 10**9)
        same_family_rank = 0 if numbered and prefix in searched_prefixes and compatibility is not False else 1
        ranked_candidates.append(
            (
                action_location in searched_without_target_locations,
                compatibility_rank,
                openable_rank,
                family_count_rank,
                same_family_rank,
                location_index,
                index,
                action,
            )
        )

    if not ranked_candidates:
        return ""

    ranked_candidates.sort()
    return str(ranked_candidates[0][-1])


def reorder_alfworld_admissible_actions(
    task_description: str,
    current_observation: str,
    action_history: str,
    admissible_actions,
    model_path=None,
):
    actions = list(admissible_actions or [])
    if not actions or not is_qwen3_vl_model_path(model_path):
        return actions

    slots = _extract_alfworld_goal_slots(task_description)
    target_object = slots["target_object"]
    target_receptacle = slots["target_receptacle"]
    required_tool = slots["required_tool"]
    target_object_terms = slots["target_object_terms"]
    target_receptacle_terms = slots["target_receptacle_terms"]
    required_tool_terms = slots["required_tool_terms"]
    recent_actions = _extract_recent_actions(action_history)
    searched_without_target_locations = _extract_searched_without_target_locations(
        action_history,
        target_object_terms=target_object_terms,
    )
    searched_numbered_prefixes = {
        numbered[0]
        for numbered in (
            _extract_numbered_location_key(location)
            for location in searched_without_target_locations
        )
        if numbered
    }
    lowered_obs = str(current_observation or "").lower()
    stripped_observation = _strip_embedded_task_text(current_observation)
    lowered_context = f"{action_history}\n{stripped_observation}".lower()
    arrived_match = re.search(r"you arrive at ([^.]+)", lowered_obs)
    arrived_location = arrived_match.group(1).strip() if arrived_match else ""
    requires_preparation = slots["required_state"] in {"clean", "cool", "heat"}
    state_already_satisfied = _context_has_object_state_completion(
        lowered_context,
        object_terms=target_object_terms,
        required_state=slots["required_state"],
        tool_terms=required_tool_terms,
    )
    needs_preparation_first = requires_preparation and not state_already_satisfied
    target_object_grounded = _context_has_target_object_grounding(
        lowered_context,
        target_object_terms,
    )
    target_object_visible = _observation_mentions_target_object(stripped_observation, target_object_terms)
    target_ready_for_preparation = target_object_grounded or target_object_visible
    target_object_needs_search = bool(target_object_terms) and not target_object_grounded and not target_object_visible
    search_history_active = target_object_needs_search
    search_continuation_active = search_history_active
    current_location_symbol = _alfworld_location_family_symbol(arrived_location)
    target_receptacle_symbol = _alfworld_entity_symbol(target_receptacle)
    at_target_receptacle = bool(current_location_symbol and current_location_symbol == target_receptacle_symbol)
    required_tool_family_symbol = _alfworld_location_family_symbol(required_tool)
    current_required_tool_location = ""
    if required_tool_terms:
        required_tool_pattern = r"|".join(re.escape(term) for term in required_tool_terms if term)
        if required_tool_pattern:
            required_tool_match = re.search(rf"\b({required_tool_pattern})\s+\d+\b", lowered_obs)
            if required_tool_match:
                current_required_tool_location = _normalize_alfworld_location_name(required_tool_match.group(0))
    required_tool_closed = bool(
        current_required_tool_location
        and _observation_location_has_state(current_observation, current_required_tool_location, "closed")
    )
    required_tool_open = bool(
        current_required_tool_location
        and _observation_location_has_state(current_observation, current_required_tool_location, "open")
    )
    if not required_tool_open and current_required_tool_location:
        required_tool_open = _action_pool_implies_location_open(actions, current_required_tool_location)
    current_search_container_closed = bool(
        target_object_needs_search
        and arrived_location
        and _location_family_can_hold_target_object(arrived_location, target_object) is True
        and _location_family_is_openable(arrived_location)
        and _observation_location_has_state(current_observation, arrived_location, "closed")
    )
    action_location_family_counts: dict[str, int] = {}
    for action in actions:
        family_symbol = _alfworld_location_family_symbol(_extract_alfworld_action_location(action))
        if family_symbol:
            action_location_family_counts[family_symbol] = action_location_family_counts.get(family_symbol, 0) + 1
    compatible_non_tool_search_available = any(
        str(action).lower().startswith(("go to ", "open ", "close ", "examine ", "inspect ", "look at "))
        and (action_location := _extract_alfworld_action_location(action))
        and _location_family_can_hold_target_object(action_location, target_object) is True
        and _alfworld_location_family_symbol(action_location) != required_tool_family_symbol
        and _alfworld_location_family_symbol(action_location) != target_receptacle_symbol
        for action in actions
    )
    target_state_action = _select_target_state_action(
        actions,
        target_object_terms=target_object_terms,
        required_state=slots["required_state"],
        required_tool_terms=required_tool_terms,
    )
    direct_target_examine_action = _select_target_direct_examine_action(
        actions,
        target_object_terms=target_object_terms,
    )
    target_take_action = _select_target_take_action(
        actions,
        target_object_terms=target_object_terms,
    )
    target_take_from_required_tool_action = _select_target_take_action(
        actions,
        target_object_terms=target_object_terms,
        preferred_location_terms=required_tool_terms,
    )

    def score_action(action: str) -> tuple:
        lowered = str(action).lower()
        score = 0
        action_location = _extract_alfworld_action_location(action)
        numbered_location = _extract_numbered_location_key(action_location)
        family_symbol = _alfworld_location_family_symbol(action_location)
        action_is_search_like = lowered.startswith(("go to ", "open ", "close ", "examine ", "inspect ", "look at "))
        mentions_target_object = _action_mentions_target_object(action, target_object_terms)
        mentions_target_receptacle = _text_mentions_any_term(action, target_receptacle_terms)
        mentions_required_tool = _text_mentions_any_term(action, required_tool_terms)
        compatible_with_target = _location_family_can_hold_target_object(action_location, target_object)
        location_is_openable = _location_family_is_openable(action_location)

        if target_object and re.search(rf"\b{re.escape(target_object)}\b", lowered):
            score += 8
            if lowered.startswith(("take ", "examine ", "go to ")):
                score += 2
        elif mentions_target_object:
            score += 5

        if needs_preparation_first and target_ready_for_preparation and target_state_action:
            if lowered == target_state_action.lower():
                score += 18
            elif lowered.startswith("move ") and mentions_required_tool:
                score -= 8
        if (
            needs_preparation_first
            and required_tool_open
            and not target_state_action
            and target_take_from_required_tool_action
            and lowered == target_take_from_required_tool_action.lower()
        ):
            score += 10

        if slots["required_state"] == "light" and state_already_satisfied:
            if direct_target_examine_action and lowered == direct_target_examine_action.lower():
                score += 18
            elif (
                not direct_target_examine_action
                and target_object_visible
                and target_take_action
                and lowered == target_take_action.lower()
            ):
                score += 16
            if mentions_required_tool and lowered.startswith("use "):
                score -= 8
            if target_take_action and lowered == "look":
                score -= 2

        if mentions_target_receptacle:
            if action_is_search_like and not (target_object_needs_search and at_target_receptacle):
                if needs_preparation_first and not target_ready_for_preparation:
                    score -= 4
                else:
                    score += 7
                    if lowered.startswith(("go to ", "open ", "close ", "examine ")):
                        score += 2
            elif target_object_grounded and lowered.startswith(("move ", "put ")):
                score += 12
            elif target_object_needs_search and lowered.startswith("take "):
                score -= 4
        if needs_preparation_first and mentions_target_receptacle:
            score -= 3

        if mentions_required_tool and action_is_search_like:
            score += 6
            if lowered.startswith(("go to ", "use ", "examine ")):
                score += 2
        if (
            needs_preparation_first
            and target_object_needs_search
            and compatible_non_tool_search_available
            and mentions_required_tool
            and action_is_search_like
        ):
            score -= 10
            if lowered.startswith("go to "):
                score -= 2
        if (
            needs_preparation_first
            and not target_ready_for_preparation
            and mentions_required_tool
            and lowered.startswith("go to ")
            and current_required_tool_location
            and action_location
            and _alfworld_location_family_symbol(action_location)
            == _alfworld_location_family_symbol(current_required_tool_location)
            and _normalize_alfworld_location_name(action_location)
            != _normalize_alfworld_location_name(current_required_tool_location)
        ):
            score -= 8
        if needs_preparation_first and mentions_required_tool and action_is_search_like and target_ready_for_preparation:
            score += 4
            if target_object_grounded:
                score += 6

        if needs_preparation_first and target_object_grounded and mentions_target_receptacle:
            score -= 6

        if current_search_container_closed:
            if lowered == f"open {arrived_location}":
                score += 18
            elif lowered.startswith("go to ") and action_location and action_location != arrived_location:
                score -= 4
                if mentions_required_tool:
                    score -= 8

        if lowered == "inventory":
            score += 1
        elif lowered == "look":
            score -= 1

        if lowered.startswith("go to "):
            score += 1

        if target_object_needs_search and action_is_search_like and action_location:
            same_open_required_tool = bool(
                required_tool_open
                and current_required_tool_location
                and lowered == f"open {current_required_tool_location}"
            )
            if compatible_with_target is True and not same_open_required_tool:
                score += 5
                if location_is_openable:
                    score += 3
            elif compatible_with_target is False and not mentions_target_receptacle:
                score -= 8
                if numbered_location:
                    score -= 2

        if required_tool_closed and current_required_tool_location:
            if lowered == f"open {current_required_tool_location}":
                score += 12
            elif lowered == f"go to {current_required_tool_location}":
                score -= 12
        if required_tool_open and current_required_tool_location:
            if lowered == f"open {current_required_tool_location}":
                score -= 12
            elif lowered == f"go to {current_required_tool_location}":
                score -= 10

        if search_continuation_active and action_location and action_location in searched_without_target_locations:
            score -= 10
        elif search_continuation_active and numbered_location and numbered_location[0] in searched_numbered_prefixes:
            score += 3

        if lowered in recent_actions:
            score -= 3
            if "nothing happens" in lowered_obs or "arrive at" in lowered_obs:
                score -= 4
        if recent_actions and lowered == recent_actions[-1]:
            score -= 2

        if arrived_location and lowered == f"go to {arrived_location}":
            score -= 8
        if (
            arrived_location
            and action_location
            and _normalize_alfworld_location_name(action_location) == _normalize_alfworld_location_name(arrived_location)
            and lowered.startswith(("examine ", "look at "))
            and "you see" in lowered_obs
        ):
            score -= 6

        if "nothing happens" in lowered_obs and lowered.startswith("take "):
            score += 1

        location_family_rank = 1
        location_index_rank = 10**9
        if numbered_location:
            location_family_rank = (
                0
                if search_continuation_active and numbered_location[0] in searched_numbered_prefixes and compatible_with_target is not False
                else 1
            )
            location_index_rank = numbered_location[1]

        revisit_rank = 1 if search_continuation_active and action_location in searched_without_target_locations else 0
        compatibility_rank = 0 if compatible_with_target is True else 1 if compatible_with_target is None else 2
        openable_rank = 0 if compatible_with_target is True and location_is_openable else 1
        family_count_rank = action_location_family_counts.get(family_symbol, 10**9)

        return (
            -score,
            revisit_rank,
            compatibility_rank,
            openable_rank,
            family_count_rank,
            location_family_rank,
            location_index_rank,
            actions.index(action),
        )

    return sorted(actions, key=score_action)


def build_alfworld_rules_for_model(model_path=None, include_compression: bool = False) -> str:
    if is_qwen3_vl_model_path(model_path):
        rules = ALFWORLD_QWEN3_ACTION_RULES
        if include_compression:
            rules = f"{rules}\n{ALFWORLD_QWEN3_COMPRESSION_RULE}"
        return rules

    rules = ALFWORLD_ACTION_RULES
    if include_compression:
        compression_rule = ALFWORLD_COMPRESSION_RULE
        rules = f"{rules}\n{compression_rule}"
    return rules


def build_alfworld_qwen3vl_messages(
    task_description,
    current_observation,
    admissible_actions,
    action_history="None",
    current_step=0,
):
    history_hint = build_alfworld_history_hint(action_history, "/tmp/Qwen3-VL-2B-Instruct")
    state_strategy_hint = build_alfworld_state_strategy_hint(
        task_description=task_description,
        current_observation=current_observation,
        action_history=action_history,
        model_path="/tmp/Qwen3-VL-2B-Instruct",
    )
    state_strategy_block = f"\n\n{state_strategy_hint}" if state_strategy_hint else ""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": ALFWORLD_QWEN3VL_SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
You are an expert agent operating in the ALFRED embodied Environment. Your task is to:
{task_description}. Prior to this step, you have already taken {current_step - 1} step(s). The provided image shows the most recent observations and the corresponding actions you took.

You are now at step {current_step} and your current textual observation is: {current_observation}
Your admissible actions of the current situation are:
{admissible_actions}
{state_strategy_block}

{history_hint}

{ALFWORLD_QWEN3_ACTION_RULES}
""".strip(),
                }
            ],
        },
    ]


ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED embodied Environment.

Task:
{task_description}

{goal_hints}

Current step:
{current_step}

Current textual observation:
{current_observation}

{failure_hint}

Admissible actions:
{admissible_actions}

{rules}
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED embodied Environment.

Task:
{task_description}

{goal_hints}

Current step:
{current_step}

Previous observations and actions:
{action_history}

Current textual observation:
{current_observation}

{failure_hint}

Admissible actions:
{admissible_actions}

{rules}
"""

ALFWORLD_TEMPLATE_NO_HIS_OCR = """
<image>
You are an expert agent operating in the ALFRED embodied Environment.

Task:
{task_description}

{goal_hints}

Current step:
{current_step}

{memory_update_hint}

Current textual observation:
{current_observation}

{failure_hint}

Admissible actions:
{admissible_actions}

{rules}
"""

ALFWORLD_TEMPLATE_NO_HIS_OCR_QWEN3 = """
<image>
Task:
{task_description}

{state_strategy_hint}

Current step:
{current_step}

{memory_update_hint}

Current textual observation:
{current_observation}

{failure_hint}

Admissible actions:
{admissible_actions}

{rules}
"""

ALFWORLD_TEMPLATE_OCR = """
<image>
You are an expert agent operating in the ALFRED embodied Environment.

Task:
{task_description}

{goal_hints}

Current step:
{current_step}

Previous observations and actions:
The provided image shows the most recent {history_length} observations and the corresponding actions you took.

{history_hint}

{memory_update_hint}

Current textual observation:
{current_observation}

{failure_hint}

Admissible actions:
{admissible_actions}

{rules}
"""

ALFWORLD_TEMPLATE_OCR_QWEN3 = """
<image>
Task:
{task_description}

{state_strategy_hint}

Current step:
{current_step}

History image: the image shows your most recent {history_length} observations and actions.

{memory_update_hint}

Current textual observation:
{current_observation}

{failure_hint}

Admissible actions:
{admissible_actions}

{rules}
"""

ALFWORLD_QWEN3_TEMPLATE_NO_HIS_OCR = """
<image>
You are an expert agent operating in the ALFRED embodied Environment. Your task is to:
{task_description}

{state_strategy_hint}

You are now at step {current_step} and your current textual observation is: {current_observation}
Your admissible actions of the current situation are:
{admissible_actions}

{rules}
"""

ALFWORLD_QWEN3_TEMPLATE_OCR = """
<image>
You are an expert agent operating in the ALFRED embodied Environment. Your task is to:
{task_description}. Prior to this step, you have already taken {step_count} step(s). The provided image shows the most recent {history_length} observations and the corresponding actions you took.

{state_strategy_hint}

You are now at step {current_step} and your current textual observation is: {current_observation}
Your admissible actions of the current situation are:
{admissible_actions}

{rules}
"""

ALFWORLD_COMPRESSION_TEMPLATE_NO_HIS = """
Additionally, select an image compression factor larger than or equal to 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. If uncertain, output 1.0 to preserve readability. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.0</compression>).
"""

ALFWORLD_COMPRESSION_TEMPLATE = """
Additionally, select an image compression factor larger than or equal to 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. If uncertain, output 1.0 to preserve readability. You must output the selected value within <compression> </compression> tags (e.g., <compression>1.0</compression>).
"""
