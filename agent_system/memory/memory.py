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

import json
import re
from typing import List, Dict, Any, Tuple
from .base import BaseMemory


def _normalize_search_history_action(text: Any) -> str:
    return " ".join(str(text or "").split())


def _normalize_search_history_observation(text: Any) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    if raw_text.startswith("<information>") and raw_text.endswith("</information>"):
        inner = raw_text[len("<information>") : -len("</information>")].strip()
        try:
            payload = json.loads(inner)
            if isinstance(payload, dict) and "result" in payload:
                inner = str(payload.get("result") or "")
        except Exception:
            inner = inner.replace("\\n", "\n")
        normalized_lines = [" ".join(line.split()) for line in inner.splitlines() if line.strip()]
        if not normalized_lines:
            return "<information></information>"
        return "<information>\n" + "\n".join(normalized_lines) + "\n</information>"

    return "\n".join(" ".join(line.split()) for line in raw_text.splitlines() if line.strip())


_SEARCH_DOC_LINE_RE = re.compile(r"^(Doc\s+\d+:)\s*(.*)$", re.IGNORECASE)
_QUOTED_SEARCH_TITLE_RE = re.compile(r'^"([^"]+)"\s*(.*)$')


def _strip_repeated_search_doc_title(text: Any) -> str:
    compact_text = " ".join(str(text or "").split()).strip()
    if not compact_text:
        return ""

    title_match = _QUOTED_SEARCH_TITLE_RE.match(compact_text)
    if not title_match:
        return compact_text

    title = " ".join(title_match.group(1).split()).strip()
    rest = " ".join(title_match.group(2).split()).strip()
    if title and rest.lower().startswith(title.lower()):
        rest = rest[len(title) :].lstrip(" -:;,.")
    if rest:
        return f"{title}: {rest}"
    return title


def _truncate_search_doc_text(text: str, char_cap: int) -> str:
    if char_cap <= 0 or len(text) <= char_cap:
        return text
    truncated = text[:char_cap]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    truncated = truncated.rstrip(" ,;:")
    if not truncated:
        truncated = text[:char_cap].rstrip(" ,;:")
    return truncated + " ..."


def _compact_search_doc_line(line: str, doc_char_cap: int) -> str:
    compact_line, _ = _compact_search_doc_line_with_metadata(line, doc_char_cap)
    return compact_line


def _compact_search_doc_line_with_metadata(line: str, doc_char_cap: int) -> Tuple[str, bool]:
    cleaned_line = " ".join(str(line or "").split()).strip()
    if not cleaned_line:
        return "", False
    doc_match = _SEARCH_DOC_LINE_RE.match(cleaned_line)
    if not doc_match:
        return _truncate_search_doc_text(cleaned_line, doc_char_cap), False

    prefix = doc_match.group(1)
    raw_content = " ".join(str(doc_match.group(2) or "").split()).strip()
    title_only = False
    title_match = _QUOTED_SEARCH_TITLE_RE.match(raw_content)
    if title_match:
        title = " ".join(title_match.group(1).split()).strip()
        rest = " ".join(title_match.group(2).split()).strip()
        if title and rest.lower().startswith(title.lower()):
            rest = rest[len(title) :].lstrip(" -:;,.")
        if rest:
            content = f"{title}: {rest}"
        else:
            content = f"[TITLE ONLY] {title}"
            title_only = True
    else:
        content = _strip_repeated_search_doc_title(raw_content)
    content = _truncate_search_doc_text(content, doc_char_cap)
    return f"{prefix} {content}".strip(), title_only


def _group_search_observation_lines(obs_clean: str) -> Tuple[List[str], List[str]]:
    doc_lines: List[str] = []
    fallback_lines: List[str] = []
    current_doc_parts: List[str] = []

    for raw_line in obs_clean.splitlines():
        cleaned_line = raw_line.strip()
        if not cleaned_line or cleaned_line in {"<information>", "</information>"}:
            continue

        if _SEARCH_DOC_LINE_RE.match(cleaned_line):
            if current_doc_parts:
                doc_lines.append(" ".join(part for part in current_doc_parts if part).strip())
            current_doc_parts = [cleaned_line]
            continue

        if current_doc_parts:
            current_doc_parts.append(cleaned_line)
        else:
            fallback_lines.append(cleaned_line)

    if current_doc_parts:
        doc_lines.append(" ".join(part for part in current_doc_parts if part).strip())

    return doc_lines, fallback_lines


def _search_line_dedupe_key(text: str) -> str:
    cleaned_text = " ".join(str(text or "").split()).strip()
    if not cleaned_text:
        return ""
    doc_match = _SEARCH_DOC_LINE_RE.match(cleaned_text)
    if doc_match:
        cleaned_text = doc_match.group(2).strip()
    return cleaned_text.lower()

class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                # lines.append(
                #     # remove \n
                #     f"[Observation]: {obs.replace('\n', ' ')} [Action]: {act.replace('\n', ' ')}"
                # )
                obs_clean = obs.replace("\n", " ")
                act_clean = act.replace("\n", " ")
                lines.append(
                    f"[Observation]: {obs_clean} [Action]: {act_clean}"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
    

class SearchMemory(BaseMemory):
    """
    Memory manager for search tasks: responsible for storing & fetching
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                act_clean = _normalize_search_history_action(act)
                obs_clean = _normalize_search_history_observation(obs)
                block_lines = [f"[Step {step_num}]"]
                if act_clean:
                    block_lines.append(act_clean)
                if obs_clean:
                    block_lines.append(obs_clean)
                lines.append("\n".join(block_lines))

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

    def fetch_compact_for_ocr(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
        doc_limit: int = 3,
        doc_char_cap: int = 384,
        newest_first: bool = True,
        dedupe_docs_across_steps: bool = True,
    ) -> Tuple[List[str], List[int]]:
        """Fetch search history formatted specifically for compact OCR rendering."""
        doc_limit = max(1, int(doc_limit))
        doc_char_cap = max(0, int(doc_char_cap))
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len
            indexed_recent = [
                (start_idx + j + 1, rec)
                for j, rec in enumerate(recent)
            ]
            if newest_first:
                indexed_recent = list(reversed(indexed_recent))

            lines = []
            seen_doc_keys = set()
            for step_num, rec in indexed_recent:
                act_clean = _normalize_search_history_action(rec[action_key])
                obs_clean = _normalize_search_history_observation(rec[obs_key])

                block_lines = [f"[Step {step_num}]"]
                if act_clean:
                    block_lines.append(act_clean)

                doc_lines, fallback_lines = _group_search_observation_lines(obs_clean)

                kept_info_lines = []
                title_only_info_lines = []
                for doc_line in doc_lines[:doc_limit]:
                    compact_line, is_title_only = _compact_search_doc_line_with_metadata(
                        doc_line,
                        doc_char_cap,
                    )
                    doc_key = _search_line_dedupe_key(compact_line)
                    if not compact_line or not (
                        not dedupe_docs_across_steps
                        or not doc_key
                        or doc_key not in seen_doc_keys
                    ):
                        continue
                    if is_title_only:
                        title_only_info_lines.append((compact_line, doc_key))
                    else:
                        kept_info_lines.append(compact_line)
                        if dedupe_docs_across_steps and doc_key:
                            seen_doc_keys.add(doc_key)

                title_hint_budget = 0
                if not kept_info_lines:
                    title_hint_budget = doc_limit

                if title_hint_budget > 0:
                    for compact_line, doc_key in title_only_info_lines[:title_hint_budget]:
                        kept_info_lines.append(compact_line)
                        if dedupe_docs_across_steps and doc_key:
                            seen_doc_keys.add(doc_key)

                if not kept_info_lines:
                    for extra_line in fallback_lines[:doc_limit]:
                        compact_line, _ = _compact_search_doc_line_with_metadata(extra_line, doc_char_cap)
                        doc_key = _search_line_dedupe_key(compact_line)
                        if compact_line and (
                            not dedupe_docs_across_steps
                            or not doc_key
                            or doc_key not in seen_doc_keys
                        ):
                            kept_info_lines.append(compact_line)
                            if dedupe_docs_across_steps and doc_key:
                                seen_doc_keys.add(doc_key)

                if kept_info_lines:
                    block_lines.append("<information>")
                    block_lines.extend(kept_info_lines)
                    block_lines.append("</information>")

                lines.append("\n".join(block_lines))

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths
