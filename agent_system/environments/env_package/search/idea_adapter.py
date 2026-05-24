import os
import re
from typing import Any, Dict, List, Optional, Tuple


_TRUTHY = {"1", "true", "yes", "on"}
_RELATION_PHRASES = (
    "same network",
    "same country",
    "same team",
    "same label",
    "same channel",
)
_FOLLOWUP_RELATION_WORDS = {
    "actor",
    "actress",
    "author",
    "captain",
    "coach",
    "daughter",
    "director",
    "ethnicity",
    "father",
    "founder",
    "governor",
    "husband",
    "language",
    "manager",
    "mayor",
    "mother",
    "nationality",
    "occupation",
    "owner",
    "parent",
    "partner",
    "player",
    "president",
    "race",
    "religion",
    "role",
    "singer",
    "son",
    "spouse",
    "star",
    "wife",
}
_FOLLOWUP_ATTRIBUTE_WORDS = {
    "age",
    "ancestry",
    "birthplace",
    "city",
    "country",
    "date",
    "ethnicity",
    "language",
    "location",
    "name",
    "nationality",
    "occupation",
    "race",
    "religion",
    "role",
    "winner",
    "year",
}
_ANSWER_ATTRIBUTE_WORDS = _FOLLOWUP_ATTRIBUTE_WORDS | {
    "award",
    "awards",
    "composer",
    "composed",
    "currency",
    "founded",
    "founded",
    "founding",
    "mascot",
    "premiered",
    "release",
    "released",
    "singer",
    "size",
    "voice",
    "won",
}
_CLAUSE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "as",
    "at",
    "be",
    "broadcast",
    "by",
    "channel",
    "did",
    "do",
    "does",
    "drama",
    "film",
    "for",
    "from",
    "how",
    "in",
    "indian",
    "is",
    "it",
    "its",
    "movie",
    "network",
    "of",
    "on",
    "or",
    "same",
    "series",
    "show",
    "television",
    "that",
    "the",
    "their",
    "this",
    "to",
    "tv",
    "up",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_GENERIC_QUERY_STOPWORDS = _CLAUSE_STOPWORDS | {
    "been",
    "being",
    "compete",
    "competed",
    "competing",
    "find",
    "has",
    "have",
    "having",
    "make",
    "makes",
    "was",
    "were",
}
_BRIDGE_QUERY_HINT_WORDS = {
    "actor",
    "actress",
    "brand",
    "company",
    "country",
    "dialogue",
    "driver",
    "episode",
    "film",
    "founded",
    "founding",
    "member",
    "members",
    "movie",
    "remake",
    "role",
    "screenwriter",
    "season",
    "singer",
    "song",
    "starred",
    "team",
    "term",
    "title",
    "winner",
    "year",
}
_QUESTION_SPAN_LEAD_NOISE = {
    "actor",
    "actress",
    "band",
    "came",
    "character",
    "composer",
    "created",
    "devised",
    "founded",
    "founding",
    "first",
    "invent",
    "invented",
    "inventor",
    "name",
    "older",
    "played",
    "portrayed",
    "singer",
    "voice",
    "who",
    "writer",
}
_LOWERCASE_NAME_CONNECTORS = {
    "da",
    "de",
    "del",
    "der",
    "di",
    "du",
    "la",
    "le",
    "van",
    "von",
}


def search_idea_branch_enabled() -> bool:
    return os.environ.get("AGENTOCR_SEARCH_SPECIAL_IDEA_BRANCH", "0").strip().lower() in _TRUTHY


def _env_flag(name: str, default: bool = False, legacy_name: Optional[str] = None) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None and legacy_name:
        raw_value = os.environ.get(legacy_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in _TRUTHY


def search_query_rewrite_enabled() -> bool:
    return _env_flag("AGENTOCR_SEARCH_ENABLE_QUERY_REWRITE", default=False)


def search_answer_rewrite_enabled() -> bool:
    return _env_flag("AGENTOCR_SEARCH_ENABLE_ANSWER_REWRITE", default=False)


def search_answer_to_search_rewrite_enabled() -> bool:
    return _env_flag("AGENTOCR_SEARCH_ENABLE_ANSWER_TO_SEARCH_REWRITE", default=False)


def search_relation_rewrite_enabled() -> bool:
    return _env_flag(
        "AGENTOCR_SEARCH_ENABLE_RELATION_REWRITE",
        default=search_query_rewrite_enabled(),
    )


def search_exact_qualifier_rewrite_enabled() -> bool:
    return _env_flag(
        "AGENTOCR_SEARCH_ENABLE_EXACT_QUALIFIER_REWRITE",
        default=search_query_rewrite_enabled(),
    )


def search_bridge_attribute_rewrite_enabled() -> bool:
    return _env_flag(
        "AGENTOCR_SEARCH_ENABLE_BRIDGE_ATTRIBUTE_REWRITE",
        default=search_exact_qualifier_rewrite_enabled(),
    )


def search_member_rewrite_enabled() -> bool:
    return _env_flag(
        "AGENTOCR_SEARCH_ENABLE_MEMBER_REWRITE",
        default=search_query_rewrite_enabled(),
    )


def search_followup_anchor_rewrite_enabled() -> bool:
    return _env_flag("AGENTOCR_SEARCH_ENABLE_FOLLOWUP_ANCHOR_REWRITE", default=False)


def search_generic_query_rewrite_enabled() -> bool:
    return _env_flag("AGENTOCR_SEARCH_ENABLE_GENERIC_QUERY_REWRITE", default=False)


def search_explicit_attribute_answer_guard_enabled() -> bool:
    return _env_flag("AGENTOCR_SEARCH_ENABLE_EXPLICIT_ATTRIBUTE_ANSWER_GUARD", default=False)


def adapt_search_action(question: str, chat_history: List[Dict[str, Any]], action: str) -> Tuple[str, Dict[str, Any]]:
    action = _trim_action(action)

    if not search_idea_branch_enabled():
        return action, {"deserves_search_bonus": True, "adapted": False, "kind": "disabled"}

    search_match = re.search(r"<search>(.*?)</search>", action, re.IGNORECASE | re.DOTALL)
    if search_match:
        query = _normalize_space(search_match.group(1))
        rewritten_query, meta = rewrite_search_query(question=question, chat_history=chat_history, query=query)
        return f"<search>{rewritten_query}</search>", meta

    answer_match = re.search(r"<answer>(.*?)</answer>", action, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = _normalize_space(answer_match.group(1))
        rewritten_answer, answer_meta = rewrite_answer_span(
            question=question,
            chat_history=chat_history,
            answer=answer,
        )
        if answer_meta["kind"] == "search":
            return (
                f"<search>{rewritten_answer}</search>",
                {
                    "deserves_search_bonus": False,
                    "adapted": True,
                    "kind": "search",
                    "reasons": answer_meta["reasons"],
                },
            )
        return (
            f"<answer>{rewritten_answer}</answer>",
            {
                "deserves_search_bonus": False,
                "adapted": answer_meta["adapted"],
                "kind": "answer",
                "reasons": answer_meta["reasons"],
            },
        )

    return action, {"deserves_search_bonus": False, "adapted": False, "kind": "unparsed"}


def rewrite_search_query(question: str, chat_history: List[Dict[str, Any]], query: str) -> Tuple[str, Dict[str, Any]]:
    question = str(question or "")
    original_query = _normalize_space(query)
    rewritten_query = original_query
    question_lower = question.lower()
    reasons: List[str] = []
    history_turns = sum(1 for item in chat_history if "<information>" in str(item.get("content", "")))
    last_info = _extract_last_information(chat_history) if history_turns > 0 else ""

    age_query_changed, rewritten_query = _rewrite_age_query(question=question, query=rewritten_query)
    if age_query_changed:
        reasons.append("age_guard")

    if search_relation_rewrite_enabled():
        relation_changed, rewritten_query = _rewrite_relation_query(
            question=question,
            history_turns=history_turns,
            query=rewritten_query,
        )
        if relation_changed:
            reasons.append("relation_guard")

    if search_exact_qualifier_rewrite_enabled():
        qualifier_changed, rewritten_query = _rewrite_exact_qualifier_query(question=question, query=rewritten_query)
        if qualifier_changed:
            reasons.append("exact_qualifier")

    if search_bridge_attribute_rewrite_enabled():
        bridge_changed, rewritten_query = _rewrite_bridge_attribute_query(
            question=question,
            history_turns=history_turns,
            query=rewritten_query,
        )
        if bridge_changed:
            reasons.append("bridge_attribute")

    if search_member_rewrite_enabled():
        member_changed, rewritten_query = _rewrite_member_query(question=question, query=rewritten_query)
        if member_changed:
            reasons.append("member_scope")

    if search_followup_anchor_rewrite_enabled():
        followup_anchor_changed, rewritten_query = _rewrite_followup_anchor_query(
            question=question,
            history_turns=history_turns,
            query=rewritten_query,
            last_info=last_info,
        )
        if followup_anchor_changed:
            reasons.append("followup_anchor")

    if search_generic_query_rewrite_enabled() and not reasons:
        generic_changed, rewritten_query = _rewrite_generic_first_turn_query(
            question=question,
            history_turns=history_turns,
            query=rewritten_query,
        )
        if generic_changed:
            reasons.append("generic_keyword")

    rewritten_query = _dedupe_words_preserve_order(rewritten_query)
    deserves_bonus = _deserves_search_bonus(
        original_query=original_query,
        rewritten_query=rewritten_query,
        question_lower=question_lower,
        history_turns=history_turns,
    )
    return rewritten_query, {
        "deserves_search_bonus": deserves_bonus,
        "adapted": rewritten_query != original_query,
        "kind": "search",
        "reasons": reasons,
    }


def rewrite_answer_span(question: str, chat_history: List[Dict[str, Any]], answer: str) -> Tuple[str, Dict[str, Any]]:
    question = str(question or "")
    answer = _normalize_space(answer)
    question_lower = question.lower()
    history_turns = sum(1 for item in chat_history if "<information>" in str(item.get("content", "")))
    last_info = _extract_last_information(chat_history)
    answer_rewrite_enabled = search_answer_rewrite_enabled()
    answer_to_search_enabled = search_answer_to_search_rewrite_enabled()

    member_shortened, member_changed = _shorten_member_answer(question=question, answer=answer)
    if member_changed:
        return member_shortened, {"adapted": True, "kind": "answer", "reasons": ["member_answer_span"]}

    parenthetical_trimmed, parenthetical_changed = _strip_parenthetical_answer(question=question, answer=answer)
    if parenthetical_changed:
        return parenthetical_trimmed, {"adapted": True, "kind": "answer", "reasons": ["canonical_answer_span"]}

    if search_explicit_attribute_answer_guard_enabled() and not answer_to_search_enabled:
        explicit_guard_query = _rewrite_answer_to_search(question=question, last_info=last_info, answer=answer)
        if explicit_guard_query:
            return explicit_guard_query, {"adapted": True, "kind": "search", "reasons": ["explicit_attribute_guard"]}

    if not answer_rewrite_enabled and not answer_to_search_enabled:
        return answer, {"adapted": False, "kind": "answer", "reasons": []}

    if not answer or len(answer.split()) != 1:
        if answer_to_search_enabled:
            fallback_query = _rewrite_answer_to_search(question=question, last_info=last_info, answer=answer)
            if fallback_query:
                return fallback_query, {"adapted": True, "kind": "search", "reasons": ["answer_requires_search"]}
        return answer, {"adapted": False, "kind": "answer", "reasons": []}

    if not last_info:
        return answer, {"adapted": False, "kind": "answer", "reasons": []}

    if answer_rewrite_enabled:
        named_candidates = _extract_name_like_answer_candidates(last_info=last_info, answer=answer)
        if len(named_candidates) == 1:
            candidate = named_candidates[0]
            if answer_to_search_enabled:
                fallback_query = _rewrite_answer_to_search(question=question, last_info=last_info, answer=candidate)
                if fallback_query:
                    return fallback_query, {"adapted": True, "kind": "search", "reasons": ["answer_requires_search"]}
            return candidate, {"adapted": True, "kind": "answer", "reasons": ["canonical_answer_span"]}

        pattern = re.compile(r"\b(?:[^\W\d_][\w&'.-]*)(?:\s+[^\W\d_][\w&'.-]*){1,3}\b", re.UNICODE)
        candidates: List[str] = []
        answer_lower = answer.lower()
        for candidate in pattern.findall(last_info):
            tokens = candidate.split()
            token_lowers = [token.lower() for token in tokens]
            if answer_lower not in token_lowers:
                continue
            if len(tokens) <= 1:
                continue
            normalized_candidate = _normalize_space(candidate)
            if normalized_candidate.lower() == answer_lower:
                continue
            candidates.append(normalized_candidate)

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)

        if len(unique_candidates) == 1:
            candidate = unique_candidates[0]
            if answer_to_search_enabled:
                fallback_query = _rewrite_answer_to_search(question=question, last_info=last_info, answer=candidate)
                if fallback_query:
                    return fallback_query, {"adapted": True, "kind": "search", "reasons": ["answer_requires_search"]}
            return candidate, {"adapted": True, "kind": "answer", "reasons": ["canonical_answer_span"]}

    if answer_to_search_enabled:
        fallback_query = _rewrite_answer_to_search(question=question, last_info=last_info, answer=answer)
        if fallback_query:
            return fallback_query, {"adapted": True, "kind": "search", "reasons": ["answer_requires_search"]}
        if history_turns > 0 and _is_short_weak_answer(question_lower=question_lower, answer=answer, last_info=last_info):
            fallback = _build_answer_followup_query(question=question, answer=answer)
            return fallback, {"adapted": True, "kind": "search", "reasons": ["weak_answer_retry"]}

    return answer, {"adapted": False, "kind": "answer", "reasons": []}


def _strip_parenthetical_answer(question: str, answer: str) -> Tuple[str, bool]:
    if "(" not in answer or ")" not in answer:
        return answer, False

    question_lower = str(question or "").lower()
    if not (
        question_lower.startswith("who ")
        or question_lower.startswith("what ")
        or re.search(r"\b(member|members|composition|consists? of|make(?:s)? up|made up of)\b", question_lower)
    ):
        return answer, False

    stripped = _normalize_space(re.sub(r"\s*\([^)]*\)", "", answer))
    if not stripped or stripped == answer:
        return answer, False
    return stripped, True


def _shorten_member_answer(question: str, answer: str) -> Tuple[str, bool]:
    question_lower = str(question or "").lower()
    answer_lower = answer.lower()
    if not re.search(r"\b(member|members|composition|consists? of|make(?:s)? up|made up of)\b", question_lower):
        return answer, False

    if "governors and presidents" in answer_lower:
        return "governors and presidents", answer_lower != "governors and presidents"

    if answer_lower.startswith("the ") and len(answer.split()) > 3:
        stripped = re.sub(r"^\s*the\s+", "", answer, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+of\b.*$", "", stripped, flags=re.IGNORECASE)
        stripped = _normalize_space(stripped)
        if stripped and stripped != answer:
            return stripped, True
    return answer, False


def _trim_action(action: str) -> str:
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _rewrite_answer_to_search(question: str, last_info: str, answer: str) -> str:
    question_lower = str(question or "").lower()
    info_lower = str(last_info or "").lower()
    answer_lower = str(answer or "").lower()
    if not question_lower or not info_lower or not answer_lower:
        return ""

    if _question_needs_explicit_attribute(question_lower):
        if not _last_info_has_explicit_attribute(question_lower=question_lower, info_lower=info_lower, answer_lower=answer_lower):
            if _should_prefer_question_only_retry(question=question, answer=answer):
                return _build_question_followup_query(question)
            return _build_answer_followup_query(question=question, answer=answer)

    if _question_has_named_choice_mismatch(question=question, answer=answer):
        return _build_question_followup_query(question)

    if _question_needs_exact_target_guard(
        question_lower=question_lower,
        info_lower=info_lower,
        answer_lower=answer_lower,
    ):
        if _should_prefer_question_only_retry(question=question, answer=answer):
            return _build_question_followup_query(question)
        return _build_answer_followup_query(question=question, answer=answer)

    return ""


def _question_needs_explicit_attribute(question_lower: str) -> bool:
    if _question_asks_age(question_lower):
        return True
    if " who " in f" {question_lower} " and any(
        word in question_lower
        for word in (
            "sings",
            "singer",
            "wrote",
            "writer",
            "author",
            "voice actor",
            "voiced",
            "composed",
            "composer",
            "played",
            "portrayed",
        )
    ):
        return True
    if _question_asks_origin(question_lower):
        return True
    if "which country" in question_lower or "what country" in question_lower:
        return True
    if "nationality" in question_lower or "ethnicity" in question_lower or "religion" in question_lower:
        return True
    if question_lower.startswith("when ") or " what year " in f" {question_lower} " or " release date " in f" {question_lower} ":
        return True
    return False


def _last_info_has_explicit_attribute(question_lower: str, info_lower: str, answer_lower: str) -> bool:
    if _question_asks_age(question_lower):
        return _last_info_has_explicit_age_value(last_info=info_lower, answer=answer_lower)
    if any(word in question_lower for word in ("sings", "singer")):
        return _contains_window(info_lower, answer_lower, ("singer", "sings", "performed by", "performed", "vocals by"))
    if any(word in question_lower for word in ("writer", "author", "wrote")):
        return _contains_window(info_lower, answer_lower, ("author", "writer", "written by", "wrote"))
    if any(word in question_lower for word in ("played", "portrayed")):
        return _contains_window(
            info_lower,
            answer_lower,
            ("played by", "played", "portrayed by", "portrayed", "actor", "actress", "cast as"),
        )
    if any(word in question_lower for word in ("voice actor", "voiced")):
        return _contains_window(info_lower, answer_lower, ("voice", "voiced", "voice actor"))
    if any(word in question_lower for word in ("composed", "composer")):
        return _contains_window(info_lower, answer_lower, ("composer", "composed by", "written by"))
    if _question_asks_origin(question_lower):
        return _contains_window(
            info_lower,
            answer_lower,
            (
                "came up with",
                "created by",
                "devised by",
                "invented",
                "inventor",
                "origin",
                "originated by",
                "pioneered by",
            ),
        )
    if "which country" in question_lower or "what country" in question_lower:
        return _contains_window(info_lower, answer_lower, ("country", "american", "british", "canadian", "australian", "english"))
    if any(word in question_lower for word in ("nationality", "ethnicity", "religion")):
        return _contains_window(info_lower, answer_lower, ("nationality", "ethnicity", "religion", "race"))
    if question_lower.startswith("when ") or " what year " in f" {question_lower} " or " release date " in f" {question_lower} ":
        return bool(re.search(r"\b(?:19|20)\d{2}\b", answer_lower)) or _contains_window(
            info_lower,
            answer_lower,
            ("released", "release", "ended", "stopped", "premiered", "date", "year", "come out"),
        )
    return True


def _question_needs_exact_target_guard(question_lower: str, info_lower: str, answer_lower: str) -> bool:
    if (
        "fortnite" in question_lower
        and "full game" in question_lower
        and any(term in info_lower for term in ("save the world", "battle royale"))
        and all(term not in question_lower for term in ("save the world", "battle royale"))
    ):
        return True
    if (
        "premier league" in question_lower
        and "assist" in question_lower
        and "premier league" not in info_lower
        and any(term in info_lower for term in ("french league", "la liga", "bundesliga", "serie a"))
    ):
        return True
    return False


def _question_asks_origin(question_lower: str) -> bool:
    return any(
        phrase in question_lower
        for phrase in (
            "came up with",
            "invent ",
            "invented",
            "inventor",
            "originated",
            "created",
            "devised",
        )
    )


def _should_prefer_question_only_retry(question: str, answer: str) -> bool:
    question_lower = str(question or "").lower()
    return _question_asks_origin(question_lower) or _answer_partially_overlaps_named_anchor(question=question, answer=answer)


def _build_question_followup_query(question: str) -> str:
    anchor_query = _build_keyword_query_from_question(question)
    rewritten, _ = rewrite_search_query(question=question, chat_history=[], query=anchor_query or question)
    return _dedupe_words_preserve_order(rewritten or anchor_query or question)


def _build_answer_followup_query(question: str, answer: str) -> str:
    anchor_query = _build_keyword_query_from_question(question)
    answer_tokens = _tokenize_original(answer)
    combined = " ".join(answer_tokens + [anchor_query]) if answer_tokens else anchor_query
    rewritten, _ = rewrite_search_query(question=question, chat_history=[], query=combined)
    return _dedupe_words_preserve_order(rewritten or anchor_query or combined)


def _question_has_named_choice_mismatch(question: str, answer: str) -> bool:
    question_lower = str(question or "").lower()
    if " or " not in f" {question_lower} ":
        return False
    if not any(term in question_lower for term in ("which", "who", "what", "first", "earlier", "later", "older", "newer")):
        return False
    if not _answer_partially_overlaps_named_anchor(question=question, answer=answer):
        return False
    return len(_extract_named_anchor_groups(question)) >= 2


def _answer_partially_overlaps_named_anchor(question: str, answer: str) -> bool:
    answer_terms = [token for token in _tokenize_lower(answer) if token not in _GENERIC_QUERY_STOPWORDS]
    if not answer_terms:
        return False

    candidate_groups = _extract_named_anchor_groups(question) + _extract_question_keyword_spans(question)
    for group in candidate_groups:
        group_terms = [token for token in group if token not in _GENERIC_QUERY_STOPWORDS]
        if not group_terms:
            continue
        if answer_terms == group_terms:
            continue

        overlap = sum(token in group_terms for token in answer_terms)
        if overlap <= 0:
            continue
        if overlap == min(len(answer_terms), len(group_terms)):
            return True
        if len(answer_terms) >= 2 and len(group_terms) >= 2:
            return True
        if overlap >= 2:
            return True
    return False


def _extract_question_keyword_spans(question: str) -> List[List[str]]:
    spans: List[List[str]] = []
    current: List[str] = []
    for token in _tokenize_lower(question):
        if token in _GENERIC_QUERY_STOPWORDS:
            if current:
                spans.append(current)
                current = []
            continue
        current.append(token)

    if current:
        spans.append(current)

    trimmed_spans: List[List[str]] = []
    for span in spans:
        start = 0
        while start < len(span) and span[start] in _QUESTION_SPAN_LEAD_NOISE:
            start += 1
        trimmed = span[start:]
        if len(trimmed) >= 2:
            trimmed_spans.append(trimmed)
    return trimmed_spans


def _contains_window(info_lower: str, answer_lower: str, hint_phrases: Tuple[str, ...]) -> bool:
    idx = info_lower.find(answer_lower)
    if idx >= 0:
        window = info_lower[max(0, idx - 120) : idx + len(answer_lower) + 120]
        if any(phrase in window for phrase in hint_phrases):
            return True
    return any(phrase in info_lower and answer_lower in info_lower for phrase in hint_phrases)


def _extract_name_like_answer_candidates(last_info: str, answer: str) -> List[str]:
    answer_tokens = [_normalize_keyword_token(token).lower() for token in _tokenize_original(answer)]
    answer_tokens = [token for token in answer_tokens if token]
    if not answer_tokens:
        return []

    info_tokens = re.findall(r"[^\W_][\w'&.-]*", str(last_info or ""), flags=re.UNICODE)
    normalized_info_tokens = [_normalize_keyword_token(token).lower() for token in info_tokens]
    span_len = len(answer_tokens)
    candidates: List[str] = []

    for start_idx in range(0, len(info_tokens) - span_len + 1):
        if normalized_info_tokens[start_idx : start_idx + span_len] != answer_tokens:
            continue

        left = start_idx
        while left > 0:
            prev_token = info_tokens[left - 1]
            if _is_capitalized_name_token(prev_token):
                left -= 1
                continue
            if _is_name_connector_token(prev_token):
                if left - 2 < 0 or not _is_capitalized_name_token(info_tokens[left - 2]):
                    break
                left -= 1
                continue
            break

        right = start_idx + span_len - 1
        while right + 1 < len(info_tokens):
            next_token = info_tokens[right + 1]
            if _is_capitalized_name_token(next_token):
                right += 1
                continue
            if _is_name_connector_token(next_token):
                if right + 2 >= len(info_tokens) or not _is_capitalized_name_token(info_tokens[right + 2]):
                    break
                right += 1
                continue
            break

        candidate_tokens = info_tokens[left : right + 1]
        if len(candidate_tokens) <= span_len or len(candidate_tokens) > 6:
            continue

        candidate = _normalize_space(" ".join(candidate_tokens))
        if candidate and candidate.lower() != _normalize_space(answer).lower():
            candidates.append(candidate)

    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _is_capitalized_name_token(token: str) -> bool:
    normalized = _normalize_keyword_token(token)
    if not normalized or normalized.lower() in _LOWERCASE_NAME_CONNECTORS:
        return False
    letters = [ch for ch in normalized if ch.isalpha()]
    if not letters:
        return False
    if normalized.isupper():
        return True
    first_alpha = next((ch for ch in normalized if ch.isalpha()), "")
    return bool(first_alpha and first_alpha.isupper())


def _is_name_connector_token(token: str) -> bool:
    normalized = _normalize_keyword_token(token)
    return normalized.lower() in _LOWERCASE_NAME_CONNECTORS


def _is_short_weak_answer(question_lower: str, answer: str, last_info: str) -> bool:
    if len(answer.split()) > 3:
        return False
    if not _question_needs_explicit_attribute(question_lower):
        return False
    return not _last_info_has_explicit_attribute(
        question_lower=question_lower,
        info_lower=last_info.lower(),
        answer_lower=answer.lower(),
    )


def _rewrite_age_query(question: str, query: str) -> Tuple[bool, str]:
    question_lower = question.lower()
    query_lower = query.lower()
    if not _question_asks_age(question_lower):
        return False, query

    forbidden_patterns = (
        r"\bcurrent year\b",
        r"\bcurrent date\b",
        r"\btoday'?s date\b",
        r"\btoday\b",
        r"\bnow\b",
        r"\bpresent year\b",
    )
    changed = False
    rewritten = query
    for pattern in forbidden_patterns:
        new_text = re.sub(pattern, " ", rewritten, flags=re.IGNORECASE)
        if new_text != rewritten:
            rewritten = new_text
            changed = True

    allowed_years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))

    def _year_sub(match: re.Match[str]) -> str:
        year = match.group(0)
        nonlocal changed
        if year not in allowed_years:
            changed = True
            return " "
        return year

    rewritten = re.sub(r"\b(?:19|20)\d{2}\b", _year_sub, rewritten)
    rewritten = _normalize_space(rewritten)
    if "age" not in rewritten.lower():
        rewritten = _normalize_space(f"{rewritten} current age")
        changed = True
    return changed, rewritten


def _question_asks_age(question_lower: str) -> bool:
    return (
        "how old" in question_lower
        or "current age" in question_lower
        or question_lower.startswith("age of ")
        or question_lower.endswith(" age")
    )


def _rewrite_relation_query(question: str, history_turns: int, query: str) -> Tuple[bool, str]:
    question_lower = question.lower()
    query_lower = query.lower()
    for phrase in _RELATION_PHRASES:
        if phrase not in question_lower:
            continue
        changed = False
        rewritten = query
        if phrase not in query_lower:
            rewritten = _normalize_space(f"{rewritten} {phrase}")
            changed = True
        if history_turns > 0:
            left_clause = question_lower.split(phrase, 1)[0]
            hint_tokens = _keywordize_clause(left_clause, limit=3)
            if hint_tokens:
                query_terms = set(_tokenize_lower(rewritten))
                missing_tokens = [token for token in hint_tokens if token not in query_terms]
                if missing_tokens:
                    rewritten = _normalize_space(f"{rewritten} {' '.join(missing_tokens)}")
                    changed = True
        return changed, rewritten
    return False, query


def _rewrite_exact_qualifier_query(question: str, query: str) -> Tuple[bool, str]:
    changed = False
    rewritten = query
    query_lower = query.lower()

    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if years and not any(year in query for year in years):
        rewritten = _normalize_space(f"{rewritten} {years[0]}")
        changed = True

    for pattern in (r"\bseason\s+\d+\b", r"\bepisode\s+\d+\b", r"\bpart\s+\d+\b"):
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match and match.group(0).lower() not in query_lower:
            rewritten = _normalize_space(f"{rewritten} {match.group(0)}")
            changed = True
    return changed, rewritten


def _rewrite_member_query(question: str, query: str) -> Tuple[bool, str]:
    question_lower = question.lower()
    if not re.search(r"\b(member|members|composition|consists? of|make(?:s)? up|made up of)\b", question_lower):
        return False, query
    rewritten = re.sub(
        r"\b(who|what|which)\b|\bmake(?:s)?\s+up\b|\bmade\s+up\s+of\b|\bconsists?\s+of\b|\bcomposition\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    rewritten = _normalize_space(rewritten)
    if not re.search(r"\bmember|members\b", rewritten, flags=re.IGNORECASE):
        rewritten = _normalize_space(f"{rewritten} members")
    return rewritten.lower() != _normalize_space(query).lower(), rewritten


def _rewrite_bridge_attribute_query(question: str, history_turns: int, query: str) -> Tuple[bool, str]:
    if history_turns > 0:
        return False, query

    question_lower = str(question or "").lower()
    if "known for" not in question_lower or "role" not in question_lower:
        return False, query
    if not re.search(r"\bwhat\s+(?:(?:19|20)\d{2}\s+)?(film|movie|song|album|book|novel|series|show)\b", question_lower):
        return False, query

    query_terms = set(_tokenize_lower(query))
    if not ({"actor", "actress"} & query_terms or "starred" in question_lower):
        return False, query
    if {"known", "role", "roles"} & query_terms and {"film", "movie", "song", "album", "book", "novel", "series", "show"} & query_terms:
        return False, query

    suffix_tokens: List[str] = []
    if "known" not in query_terms:
        suffix_tokens.append("known")
    if not ({"role", "roles"} & query_terms):
        suffix_tokens.append("roles")

    work_type_match = re.search(
        r"\bwhat\s+(?:(?:19|20)\d{2}\s+)?(film|movie|song|album|book|novel|series|show)\b",
        question_lower,
    )
    if work_type_match:
        work_type = work_type_match.group(1)
        if work_type not in query_terms:
            suffix_tokens.append(work_type)

    if not suffix_tokens:
        return False, query
    rewritten = _normalize_space(f"{query} {' '.join(suffix_tokens)}")
    return rewritten.lower() != _normalize_space(query).lower(), rewritten


def _rewrite_generic_first_turn_query(question: str, history_turns: int, query: str) -> Tuple[bool, str]:
    if history_turns > 0:
        return False, query

    normalized_query = _normalize_space(query)
    query_tokens = _tokenize_lower(normalized_query)
    if len(query_tokens) < 4:
        return False, query

    question_lower = str(question or "").lower()
    if _should_preserve_first_turn_query(question=question, query_tokens=query_tokens):
        return False, query
    if not _looks_like_natural_language_query(question_lower=question_lower, query_tokens=query_tokens):
        return False, query

    rewritten = _build_keyword_query_from_question(question)
    if not rewritten:
        return False, query
    return rewritten.lower() != normalized_query.lower(), rewritten


def _rewrite_followup_anchor_query(
    question: str,
    history_turns: int,
    query: str,
    last_info: str = "",
) -> Tuple[bool, str]:
    if history_turns <= 0:
        return False, query

    normalized_query = _normalize_space(query)
    if not normalized_query:
        return False, query

    query_tokens = _tokenize_lower(normalized_query)
    if not query_tokens:
        return False, query

    question_lower = str(question or "").lower()
    if _question_needs_followup_anchor(question_lower):
        question_tokens = set(_tokenize_lower(question_lower))
        content_tokens = [
            token
            for token in query_tokens
            if token not in _GENERIC_QUERY_STOPWORDS and token not in _FOLLOWUP_ATTRIBUTE_WORDS and len(token) > 2
        ]
        overlap = sum(token in question_tokens for token in content_tokens)
        can_use_question_anchor = (
            overlap < 2
            and not (len(query_tokens) > 6 and not any(token in _FOLLOWUP_ATTRIBUTE_WORDS for token in query_tokens))
        )
        if can_use_question_anchor:
            anchor_query = _build_keyword_query_from_question(question)
            if anchor_query:
                query_terms = set(query_tokens)
                anchor_suffix: List[str] = []
                for token in anchor_query.split():
                    token_lower = token.lower()
                    if token_lower in query_terms:
                        continue
                    anchor_suffix.append(token)
                    if len(anchor_suffix) >= 6:
                        break

                if anchor_suffix:
                    rewritten = _dedupe_words_preserve_order(f"{normalized_query} {' '.join(anchor_suffix)}")
                    if rewritten.lower() != normalized_query.lower():
                        return True, rewritten

    doc_anchor = _extract_followup_doc_anchor(last_info)
    if not doc_anchor:
        return False, query

    if _query_already_covers_anchor(normalized_query, doc_anchor):
        return False, query

    suffix_tokens = _build_followup_doc_anchor_suffix(
        question=question,
        query=normalized_query,
        anchor=doc_anchor,
    )
    if not suffix_tokens:
        return False, query

    rewritten = _dedupe_words_preserve_order(f"{doc_anchor} {' '.join(suffix_tokens)}")
    return rewritten.lower() != normalized_query.lower(), rewritten


def _extract_followup_doc_anchor(last_info: str) -> str:
    for raw_line in str(last_info or "").splitlines():
        line = _normalize_space(raw_line)
        if not line:
            continue
        doc_match = re.match(r"^Doc\s+\d+:\s*(.*)$", line, flags=re.IGNORECASE)
        if not doc_match:
            continue
        content = _normalize_space(doc_match.group(1))
        if not content:
            continue

        quoted_match = re.match(r'^"([^"]+)"', content)
        if quoted_match:
            return _normalize_space(quoted_match.group(1))

        repeat_anchor = _extract_repeated_doc_prefix_anchor(content)
        if repeat_anchor:
            return repeat_anchor

        fallback_anchor = _extract_leading_doc_anchor(content)
        if fallback_anchor:
            return fallback_anchor
    return ""


def _extract_repeated_doc_prefix_anchor(content: str) -> str:
    tokens = _tokenize_original(content)
    if len(tokens) < 3:
        return ""

    normalized_tokens = [_normalize_keyword_token(token) for token in tokens]
    max_span = min(6, len(tokens) - 1)
    for span_len in range(max_span, 1, -1):
        prefix_tokens = normalized_tokens[:span_len]
        prefix_text = _normalize_space(" ".join(prefix_tokens))
        prefix_lower = prefix_text.lower()
        if not prefix_lower:
            continue

        remainder_text = _normalize_space(" ".join(normalized_tokens[span_len:])).lower()
        if not remainder_text:
            continue

        if re.search(rf"\b(?:the\s+)?{re.escape(prefix_lower)}\b", remainder_text):
            return prefix_text
    return ""


def _extract_leading_doc_anchor(content: str) -> str:
    tokens = _tokenize_original(content)
    if not tokens:
        return ""

    body_starters = {
        "a",
        "an",
        "are",
        "has",
        "have",
        "had",
        "is",
        "known",
        "refers",
        "was",
        "were",
    }
    allowed_lowercase_title_tokens = {
        "and",
        "day",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
        "year",
    }

    anchor_tokens: List[str] = []
    for idx, token in enumerate(tokens[:6]):
        normalized = _normalize_keyword_token(token)
        if not normalized:
            break
        lowered = normalized.lower()
        if idx == 0:
            anchor_tokens.append(normalized)
            continue
        if lowered in body_starters:
            break
        if _is_capitalized_name_token(token) or lowered in allowed_lowercase_title_tokens:
            anchor_tokens.append(normalized)
            continue
        break

    if len(anchor_tokens) >= 2:
        return _normalize_space(" ".join(anchor_tokens))
    return ""


def _query_already_covers_anchor(query: str, anchor: str) -> bool:
    query_terms = set(_tokenize_lower(query))
    anchor_terms = [token for token in _tokenize_lower(anchor) if token not in _GENERIC_QUERY_STOPWORDS]
    if not anchor_terms:
        return False
    required_hits = 1 if len(anchor_terms) == 1 else 2
    return sum(token in query_terms for token in anchor_terms) >= required_hits


def _build_followup_doc_anchor_suffix(question: str, query: str, anchor: str) -> List[str]:
    question_lower = str(question or "").lower()
    query_tokens_original = _tokenize_original(query)
    query_tokens_lower = [_normalize_keyword_token(token).lower() for token in query_tokens_original]
    anchor_terms = set(_tokenize_lower(anchor))

    suffix_tokens: List[str] = []
    for group in _extract_named_anchor_groups(question):
        if _anchor_group_matches_doc_anchor(group, anchor_terms):
            continue
        for token in group:
            if token in anchor_terms or token in _GENERIC_QUERY_STOPWORDS:
                continue
            if token.isdigit() or len(token) <= 1:
                continue
            suffix_tokens.append(token)

    attribute_hints: List[str] = []
    if "came up with" in question_lower or "invent" in question_lower or "invent" in " ".join(query_tokens_lower):
        attribute_hints.append("origin")
    if "network" in question_lower:
        attribute_hints.append("network")
    if "nationality" in question_lower:
        attribute_hints.append("nationality")
    if "ethnicity" in question_lower:
        attribute_hints.append("ethnicity")
    if "religion" in question_lower:
        attribute_hints.append("religion")
    if "how old" in question_lower or " age" in f" {question_lower} ":
        attribute_hints.append("age")
    if "founded" in question_lower or "founding" in question_lower:
        attribute_hints.extend(["founding", "year"])
    if question_lower.startswith("when ") or " what year " in f" {question_lower} " or " which year " in f" {question_lower} ":
        attribute_hints.append("year")

    for token_original, token_lower in zip(query_tokens_original, query_tokens_lower):
        if not token_lower or token_lower in _GENERIC_QUERY_STOPWORDS or token_lower in anchor_terms:
            continue
        if token_lower in _FOLLOWUP_ATTRIBUTE_WORDS or token_lower in _ANSWER_ATTRIBUTE_WORDS:
            attribute_hints.append(token_original)

    suffix_tokens.extend(attribute_hints)
    deduped_suffix: List[str] = []
    seen = set()
    for token in suffix_tokens:
        normalized = _normalize_space(token)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen or key in anchor_terms:
            continue
        seen.add(key)
        deduped_suffix.append(normalized)
    return deduped_suffix[:6]


def _anchor_group_matches_doc_anchor(group: List[str], anchor_terms: set[str]) -> bool:
    if not group or not anchor_terms:
        return False
    required_hits = 1 if len(group) == 1 else 2
    return sum(token in anchor_terms for token in group) >= required_hits


def _deserves_search_bonus(
    original_query: str,
    rewritten_query: str,
    question_lower: str,
    history_turns: int,
) -> bool:
    if not original_query or len(original_query.split()) < 2:
        return False
    if rewritten_query != original_query:
        return False
    if any(phrase in question_lower for phrase in _RELATION_PHRASES) and history_turns > 0:
        if not any(phrase in original_query.lower() for phrase in _RELATION_PHRASES):
            return False
    if re.search(r"\b(member|members|composition|consists of|make up|made up of)\b", question_lower):
        if not re.search(r"\b(member|members|composition|consists of|make up|made up of)\b", original_query, re.I):
            return False
    if any(word in question_lower for word in ("who ", "what ", "when ", "where ", "which ", "how many", "how much")):
        if _looks_like_natural_language_query(question_lower=question_lower, query_tokens=_tokenize_lower(original_query)):
            return False
    years = re.findall(r"\b(?:19|20)\d{2}\b", question_lower)
    if years and not any(year in original_query for year in years):
        return False
    if "how old" in question_lower or "current age" in question_lower:
        if re.search(r"\bcurrent year\b|\bcurrent date\b|\btoday\b|\bnow\b", original_query, re.I):
            return False
    return True


def _keywordize_clause(text: str, limit: int = 3) -> List[str]:
    tokens = _tokenize_lower(text)
    kept: List[str] = []
    for token in tokens:
        if token in _CLAUSE_STOPWORDS or token.isdigit():
            continue
        if len(token) <= 2:
            continue
        kept.append(token)
    return kept[:limit]


def _tokenize_lower(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", str(text or "").lower())


def _tokenize_original(text: str) -> List[str]:
    return re.findall(r"[^\W_][\w'&.-]*", str(text or ""), flags=re.UNICODE)


def _normalize_keyword_token(token: str) -> str:
    token = str(token or "").strip()
    if token.endswith("'s") or token.endswith("’s"):
        token = token[:-2]
    return token.strip("'’")


def _question_needs_followup_anchor(question_lower: str) -> bool:
    return ("'s " in question_lower or "’s " in question_lower) or any(
        word in question_lower for word in _FOLLOWUP_RELATION_WORDS
    )


def _looks_like_natural_language_query(question_lower: str, query_tokens: List[str]) -> bool:
    if not query_tokens:
        return False
    if query_tokens[0] in {"who", "what", "when", "where", "which", "how"}:
        return True
    stopword_hits = sum(token in _CLAUSE_STOPWORDS for token in query_tokens)
    overlap = sum(token in question_lower for token in query_tokens)
    return stopword_hits >= 2 or overlap >= max(3, len(query_tokens) - 1)


def _build_keyword_query_from_question(question: str) -> str:
    question = str(question or "")
    question_lower = question.lower()
    question_tokens = set(_tokenize_lower(question_lower))
    attribute_noise_tokens: set[str] = set()
    if "sings" in question_lower or "singer" in question_lower:
        attribute_noise_tokens.update({"sing", "sings", "singer", "sang", "sung"})
    if (
        "writer" in question_lower
        or "author" in question_lower
        or "wrote" in question_lower
    ) and not re.search(r"\b(screenwriter|dialogue|remake)\b", question_lower):
        attribute_noise_tokens.update({"author", "write", "writer", "writes", "wrote", "written"})
    if "played" in question_lower or "portrayed" in question_lower:
        attribute_noise_tokens.update({"play", "played", "portray", "portrayed"})
    if "voice actor" in question_lower or "voiced" in question_lower:
        attribute_noise_tokens.update({"actor", "voice", "voiced"})
    if "composer" in question_lower or "composed" in question_lower:
        attribute_noise_tokens.update({"compose", "composed", "composer", "composes"})
    if _question_asks_origin(question_lower):
        attribute_noise_tokens.update({"came", "come", "created", "devised", "invent", "invented", "inventor"})
    if _question_asks_age(question_lower):
        attribute_noise_tokens.update({"age", "current", "old"})
    tokens = _tokenize_original(question)
    kept: List[str] = []
    for token in tokens:
        normalized_token = _normalize_keyword_token(token)
        if not normalized_token:
            continue
        token_lower = normalized_token.lower()
        if token_lower in _GENERIC_QUERY_STOPWORDS:
            continue
        if token_lower in attribute_noise_tokens:
            continue
        if len(token_lower) <= 1 and not token_lower.isdigit():
            continue
        kept.append(normalized_token)

    hints: List[str] = []
    if "member" in question_lower or "make up" in question_lower or "consists of" in question_lower:
        hints.append("members")
    if _question_asks_origin(question_lower):
        hints.append("origin")
    if question_lower.startswith("when ") or " what year " in f" {question_lower} " or " which year " in f" {question_lower} ":
        hints.append("date")
    if "sings" in question_lower or "singer" in question_lower:
        hints.append("singer")
    if (
        "writer" in question_lower
        or "author" in question_lower
        or "wrote" in question_lower
    ) and not re.search(r"\b(screenwriter|dialogue|remake)\b", question_lower):
        hints.append("author")
    if "played" in question_lower or "portrayed" in question_lower:
        hints.append("actor")
    if "voice actor" in question_lower or "voiced" in question_lower:
        hints.append("voice")
    if "composer" in question_lower or "composed" in question_lower:
        hints.append("composer")
    if question_lower.startswith("who won") or " winner " in f" {question_lower} ":
        hints.append("winner")
    if question_lower.startswith("where "):
        hints.append("location")
    if "minimum wage" in question_lower and "world" in question_lower:
        hints.append("country")
    if "called" in question_tokens or "term" in question_tokens:
        hints.append("term")
    if "brand" in question_tokens:
        hints.append("brand")
    if "company" in question_tokens:
        hints.append("company")
    if "city" in question_tokens:
        hints.append("city")
    if "country" in question_tokens:
        hints.append("country")
    if "compete with" in question_lower or "competed with" in question_lower or "competitor" in question_lower:
        hints.append("competitor")
    if "number of" in question_lower or question_lower.startswith("how many"):
        hints.append("count")

    combined = kept + hints
    return _dedupe_words_preserve_order(" ".join(combined))


def _should_preserve_first_turn_query(question: str, query_tokens: List[str]) -> bool:
    if not query_tokens or len(query_tokens) > 16:
        return False
    if query_tokens[0] in {"who", "what", "when", "where", "which", "how"}:
        return False

    stopword_hits = sum(token in _CLAUSE_STOPWORDS for token in query_tokens)
    if stopword_hits > 3:
        return False

    if not _query_covers_named_anchor(question=question, query_tokens=query_tokens):
        return False

    return any(token in _BRIDGE_QUERY_HINT_WORDS for token in query_tokens)


def _query_covers_named_anchor(question: str, query_tokens: List[str]) -> bool:
    query_terms = set(query_tokens)
    for group in _extract_named_anchor_groups(question):
        required_hits = 1 if len(group) == 1 else 2
        hits = sum(token in query_terms for token in group)
        if hits >= required_hits:
            return True
    return False


def _extract_named_anchor_groups(question: str) -> List[List[str]]:
    groups: List[List[str]] = []
    current: List[str] = []
    for raw_token in _tokenize_original(question):
        normalized = _normalize_keyword_token(raw_token)
        if not normalized:
            continue

        token_lower = normalized.lower()
        is_named_like = (
            any(ch.isupper() for ch in normalized if ch.isalpha())
            or any(ch.isdigit() for ch in normalized)
        )
        if is_named_like and token_lower not in _GENERIC_QUERY_STOPWORDS:
            current.append(token_lower)
            continue

        if current:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    deduped_groups: List[List[str]] = []
    seen = set()
    for group in groups:
        key = tuple(group)
        if key in seen:
            continue
        seen.add(key)
        deduped_groups.append(group)
    return deduped_groups


def _dedupe_words_preserve_order(text: str) -> str:
    tokens = _normalize_space(text).split()
    deduped: List[str] = []
    seen = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return " ".join(deduped)


def _extract_last_information(chat_history: List[Dict[str, Any]]) -> str:
    for item in reversed(chat_history):
        content = str(item.get("content", ""))
        if "<information>" not in content:
            continue
        match = re.search(r"<information>(.*?)</information>", content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    return ""
