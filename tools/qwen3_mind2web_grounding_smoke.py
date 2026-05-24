#!/usr/bin/env python3
import argparse
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from agentocr import OCRTool
from agentocr.trust_policy import TrustCalibratedRenderPolicy, TrustPolicyConfig


DEFAULT_MODEL_PATH = "/hpc2hdd/home/rtang906/hf_models/Qwen3-VL-4B-Instruct"
DEFAULT_DATA_PATH = (
    "/hpc2hdd/home/rtang906/.cache/huggingface/hub/"
    "datasets--osunlp--Mind2Web/snapshots/17ece8eb89862368edc0cc806acee6fca5163474/"
    "data/train/train_0.json"
)
DEFAULT_OUTPUT = "/hpc2hdd/home/rtang906/AgentOCR/outputs/v2_debug/mind2web_grounding_smoke.json"
CONTAINER_TAGS = {"html", "body"}
SIGNAL_ATTRS = ("aria_label", "placeholder", "name", "id", "title", "value", "input_value", "type", "role")


@dataclass
class CandidateRecord:
    backend_node_id: str
    tag: str
    text: str
    summary: str
    attrs: Dict[str, Any]
    raw: Dict[str, Any]
    score_hint: float = 0.0


@dataclass
class CaseRecord:
    annotation_id: str
    website: str
    domain: str
    subdomain: str
    task: str
    step_index: int
    total_steps: int
    action_repr: str
    operation: Dict[str, Any]
    positive_candidate: CandidateRecord
    candidates: List[CandidateRecord]
    history_context: str
    query_text: str


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _meaningful_texts(texts: Iterable[str], limit: int = 2) -> str:
    kept: List[str] = []
    for text in texts:
        cleaned = _normalize_space(text)
        if not cleaned:
            continue
        if cleaned in kept:
            continue
        kept.append(cleaned)
        if len(kept) >= limit:
            break
    return " | ".join(kept)


def _find_element(soup: BeautifulSoup, backend_node_id: str):
    return soup.find(attrs={"backend_node_id": backend_node_id})


def _element_text(soup: BeautifulSoup, backend_node_id: str) -> str:
    element = _find_element(soup, backend_node_id)
    if element is None:
        return ""
    texts = list(element.stripped_strings)
    summary = _meaningful_texts(texts, limit=2)
    if summary:
        return summary
    for parent in element.parents:
        if getattr(parent, "attrs", None) and parent.attrs.get("backend_node_id"):
            parent_text = _meaningful_texts(list(parent.stripped_strings), limit=2)
            if parent_text:
                return parent_text
            parent_attrs = []
            for key in ("aria_label", "placeholder", "name", "title"):
                value = _normalize_space(parent.attrs.get(key, ""))
                if value:
                    parent_attrs.append(f"{key}={value}")
            if parent_attrs:
                return " | ".join(parent_attrs[:2])
    return ""


def _candidate_summary(candidate: Dict[str, Any], soup: BeautifulSoup) -> CandidateRecord:
    attrs = _safe_json_loads(str(candidate.get("attributes", "")))
    backend_node_id = str(candidate.get("backend_node_id", attrs.get("backend_node_id", ""))).strip()
    tag = str(candidate.get("tag", "")).strip().lower()
    text_value = _element_text(soup, backend_node_id)
    parts = [f"id={backend_node_id}", f"tag={tag}"]
    for key in SIGNAL_ATTRS:
        value = _normalize_space(attrs.get(key, ""))
        if value:
            parts.append(f"{key}={value[:80]}")
    if text_value:
        parts.append(f"text={text_value[:120]}")
    summary = " | ".join(parts)
    return CandidateRecord(
        backend_node_id=backend_node_id,
        tag=tag,
        text=text_value,
        summary=summary,
        attrs=attrs,
        raw=candidate,
    )


def _tokenize_signal(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text or "").lower())


def _candidate_signal_strength(candidate: CandidateRecord) -> float:
    strength = 0.0
    if candidate.text:
        strength += 2.0
    for key in SIGNAL_ATTRS:
        if _normalize_space(candidate.attrs.get(key, "")):
            strength += 1.0
    if candidate.tag in {"button", "input", "select", "textarea", "a"}:
        strength += 1.0
    if candidate.tag in CONTAINER_TAGS:
        strength -= 2.0
    return strength


def _negative_ranking(
    candidate: CandidateRecord,
    positive: CandidateRecord,
    task: str,
    action_repr: str,
    operation: Dict[str, Any],
) -> float:
    score = 0.0
    pos_tokens = set(_tokenize_signal(positive.summary))
    cand_tokens = set(_tokenize_signal(candidate.summary))
    query_tokens = set(_tokenize_signal(task)) | set(_tokenize_signal(action_repr))
    value_tokens = set(_tokenize_signal(operation.get("value", "")))
    if candidate.tag == positive.tag:
        score += 5.0
    if candidate.attrs.get("type") == positive.attrs.get("type"):
        score += 2.0
    for key in ("name", "placeholder", "aria_label", "id", "role"):
        pos_value = _normalize_space(positive.attrs.get(key, ""))
        cand_value = _normalize_space(candidate.attrs.get(key, ""))
        if pos_value and cand_value:
            if pos_value == cand_value:
                score += 4.0
            elif pos_value.lower() in cand_value.lower() or cand_value.lower() in pos_value.lower():
                score += 2.0
    score += 0.5 * len(pos_tokens & cand_tokens)
    score += 0.35 * len(query_tokens & cand_tokens)
    score += 0.75 * len(value_tokens & cand_tokens)
    score += _candidate_signal_strength(candidate)
    if candidate.tag in CONTAINER_TAGS:
        score -= 4.0
    return score


def _build_history_context(item: Dict[str, Any], step_index: int) -> str:
    lines = [
        f"[META] website={item['website']} domain={item['domain']} subdomain={item['subdomain']}",
    ]
    for idx, action_repr in enumerate(item["action_reprs"][:step_index], start=1):
        lines.append(f"[STEP {idx:02d}] [Action]: {_normalize_space(action_repr)}")
    return "\n".join(lines)


def _build_query_text(item: Dict[str, Any], action_repr: str, operation: Dict[str, Any]) -> str:
    op = str(operation.get("op", "")).strip()
    value = _normalize_space(operation.get("value", ""))
    extra = f"{op}:{value}" if value else op
    extra = extra.strip(":")
    if extra:
        return f"{item['confirmed_task']}\nNext step: {_normalize_space(action_repr)}\nOperation: {extra}"
    return f"{item['confirmed_task']}\nNext step: {_normalize_space(action_repr)}"


def _pick_cases(
    data: Sequence[Dict[str, Any]],
    limit: int,
    candidates_per_case: int,
    max_items_to_scan: Optional[int] = None,
) -> List[CaseRecord]:
    ranked: List[Tuple[float, CaseRecord]] = []
    scan_count = min(len(data), max_items_to_scan) if max_items_to_scan is not None else len(data)
    for item in data[:scan_count]:
        actions = item.get("actions", [])
        action_reprs = item.get("action_reprs", [])
        for step_index, action in enumerate(actions):
            if step_index < 3 or step_index >= len(action_reprs):
                continue
            pos_list = action.get("pos_candidates") or []
            neg_list = action.get("neg_candidates") or []
            if len(pos_list) != 1 or len(neg_list) < max(4, candidates_per_case - 1):
                continue
            operation = action.get("operation") or {}
            op_name = str(operation.get("op", "")).strip().upper()
            if op_name not in {"CLICK", "TYPE", "SELECT"}:
                continue
            soup = BeautifulSoup(str(action.get("cleaned_html", "")), "html.parser")
            positive = _candidate_summary(pos_list[0], soup)
            if _candidate_signal_strength(positive) < 1.0:
                continue
            negatives: List[CandidateRecord] = []
            for neg in neg_list:
                candidate = _candidate_summary(neg, soup)
                if not candidate.backend_node_id or candidate.backend_node_id == positive.backend_node_id:
                    continue
                candidate.score_hint = _negative_ranking(
                    candidate=candidate,
                    positive=positive,
                    task=str(item.get("confirmed_task", "")),
                    action_repr=str(action_reprs[step_index]),
                    operation=operation,
                )
                negatives.append(candidate)
            negatives.sort(key=lambda record: (-record.score_hint, record.backend_node_id))
            chosen_negatives = negatives[: max(1, candidates_per_case - 1)]
            if len(chosen_negatives) < max(1, candidates_per_case - 1):
                continue
            candidates = [positive, *chosen_negatives]
            candidates.sort(key=lambda record: (record.backend_node_id != positive.backend_node_id, record.backend_node_id))
            case = CaseRecord(
                annotation_id=str(item["annotation_id"]),
                website=str(item["website"]),
                domain=str(item["domain"]),
                subdomain=str(item["subdomain"]),
                task=str(item["confirmed_task"]),
                step_index=step_index,
                total_steps=len(actions),
                action_repr=str(action_reprs[step_index]),
                operation=operation,
                positive_candidate=positive,
                candidates=candidates,
                history_context=_build_history_context(item, step_index),
                query_text=_build_query_text(item, str(action_reprs[step_index]), operation),
            )
            rank_score = float(step_index) + 0.15 * _candidate_signal_strength(positive) + 0.02 * sum(
                neg.score_hint for neg in chosen_negatives
            )
            ranked.append((rank_score, case))

    ranked.sort(key=lambda item: (-item[0], item[1].annotation_id, item[1].step_index))
    selected: List[CaseRecord] = []
    seen_keys = set()
    seen_websites = set()
    for _, case in ranked:
        key = (case.annotation_id, case.step_index)
        if key in seen_keys:
            continue
        if case.website in seen_websites and len(selected) < limit // 2:
            continue
        seen_keys.add(key)
        seen_websites.add(case.website)
        selected.append(case)
        if len(selected) >= limit:
            break
    return selected


def _build_prompt(case: CaseRecord) -> str:
    candidate_lines = "\n".join(f"- {candidate.summary}" for candidate in case.candidates)
    op_name = str(case.operation.get("op", "")).strip().upper()
    op_value = _normalize_space(case.operation.get("value", ""))
    op_line = f"{op_name}: {op_value}" if op_value else op_name
    return (
        "<image>\n"
        "You are selecting the correct webpage element for the next grounded browser action.\n\n"
        f"Task:\n{case.task}\n\n"
        f"Website:\n{case.website} | {case.domain} | {case.subdomain}\n\n"
        f"Current step:\n{case.step_index + 1} / {case.total_steps}\n\n"
        "History image:\n"
        "The image shows previous interaction history only.\n\n"
        "Target next-step description:\n"
        f"{case.action_repr}\n\n"
        "Required operation:\n"
        f"{op_line}\n\n"
        "Candidate elements:\n"
        f"{candidate_lines}\n\n"
        "Rules:\n"
        "1. Use the task goal and the history image to infer what constraints have already been set.\n"
        "2. Use the next-step description and the candidate list to choose the exact target element.\n"
        "3. Output exactly one tag in the format <candidate_id>...</candidate_id>.\n"
        "4. The content inside <candidate_id>...</candidate_id> must be one backend_node_id copied from the candidate list.\n"
    )


def _build_chat_prompt(processor, prompt_text: str, image_count: int) -> str:
    prompt_body = prompt_text.replace("<image>\n", "", 1).replace("<image>", "", 1)
    chat = [
        {
            "role": "user",
            "content": [
                *([{"type": "image"}] * max(1, image_count)),
                {"type": "text", "text": prompt_body},
            ],
        }
    ]
    return processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def _load_model(model_path: str, device: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return processor, model


def _run_generation(
    *,
    model,
    processor,
    prompt_text: str,
    images: List[Any],
    max_prompt_length: int,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any], List[Image.Image]]:
    raw_prompt = _build_chat_prompt(processor, prompt_text=prompt_text, image_count=len(images))
    processed_images = [
        image if isinstance(image, Image.Image) else Image.fromarray(image)
        for image in images
    ]
    model_inputs = processor(
        text=[raw_prompt],
        images=processed_images,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    input_len = model_inputs["input_ids"].shape[1]
    output_tokens = generated[0][input_len:]
    output_text = processor.decode(output_tokens, skip_special_tokens=False)
    mm_diagnostics = {
        "prompt_char_count": len(raw_prompt),
        "input_token_count": int(input_len),
        "image_count": len(processed_images),
        "visual_tokens": 0,
        "max_prompt_length": int(max_prompt_length),
    }
    return output_text, mm_diagnostics, processed_images


def _parse_candidate_id(output_text: str, candidates: Sequence[CandidateRecord]) -> str:
    match = re.search(r"<candidate_id>\s*([0-9]+)\s*</candidate_id>", output_text, flags=re.I)
    if match:
        return match.group(1)
    candidate_ids = {candidate.backend_node_id for candidate in candidates}
    for token in re.findall(r"\b[0-9]{2,}\b", output_text):
        if token in candidate_ids:
            return token
    return ""


def _build_ocr_tool() -> OCRTool:
    return OCRTool(
        enabled=True,
        font_size=10,
        padding=2,
        min_width=28,
        max_width=392,
        min_height=28,
        max_height=1024,
        max_workers=1,
        use_parallel=False,
        use_precise=False,
        enable_cache=False,
    )


def _render_history(
    *,
    case: CaseRecord,
    variant: str,
    tool: OCRTool,
) -> Dict[str, Any]:
    trust_enabled = variant == "full"
    image_arrays = tool.convert_texts_to_images(
        [case.history_context],
        batch_size=1,
        current_steps=[case.step_index],
        trust_policy=trust_enabled,
        trust_policy_obj=TrustCalibratedRenderPolicy(
            TrustPolicyConfig(query_relevance_weight=0.30, context_budget_percent=50.0)
        ) if trust_enabled else None,
        trust_policy_query_texts=[case.query_text] if trust_enabled else None,
        trust_policy_state_aware=False,
        trust_policy_context_mode="auto" if trust_enabled else None,
        trust_policy_use_compressed_history=trust_enabled,
        trust_policy_use_prompt_summary=False,
        trust_policy_collect_diagnostics=True,
        trust_policy_min_compaction_lines=4,
        trust_policy_min_prompt_summary_lines=4,
        qwen3_history_structured=True,
        qwen3_history_page_width=392,
        qwen3_history_page_height=512,
        qwen3_history_dynamic_min_height=True,
        enable_cache=False,
    )
    image_like = image_arrays[0]
    if isinstance(image_like, list):
        rendered = [Image.fromarray(image) for image in image_like]
    else:
        rendered = [Image.fromarray(image_like)]
    return {
        "images": rendered,
        "trust_diagnostics": tool.get_last_trust_policy_diagnostics()[0] if trust_enabled else {},
        "processed_context": (
            tool.get_last_trust_policy_processed_contexts()[0]
            if trust_enabled and tool.get_last_trust_policy_processed_contexts()
            else case.history_context
        ),
        "effective_compression": (
            tool.get_last_applied_compression_factors()[0]
            if tool.get_last_applied_compression_factors()
            else 1.0
        ),
    }


def _save_case_artifacts(
    case_dir: Path,
    *,
    prompt_text: str,
    raw_output: str,
    rendered_images: Sequence[Image.Image],
    candidate_descriptions: Sequence[str],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (case_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
    (case_dir / "candidates.txt").write_text("\n".join(candidate_descriptions), encoding="utf-8")
    for index, image in enumerate(rendered_images):
        image.save(case_dir / f"history_{index}.png")


def _summarize(rows: Sequence[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    variant_rows = [row for row in rows if row["variant"] == variant]
    success_count = sum(1 for row in variant_rows if row["success"])
    visual_tokens = [float(row["mm_diagnostics"].get("visual_tokens", 0.0)) for row in variant_rows]
    compression = [float(row["effective_render_compression"]) for row in variant_rows]
    return {
        "variant": variant,
        "num_cases": len(variant_rows),
        "grounding_rate": (success_count / len(variant_rows)) if variant_rows else 0.0,
        "success_count": success_count,
        "mean_visual_tokens": (sum(visual_tokens) / len(visual_tokens)) if visual_tokens else 0.0,
        "mean_effective_compression": (sum(compression) / len(compression)) if compression else 0.0,
    }


@contextmanager
def _temporary_env(overrides: Dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Mind2Web offline grounding smoke with AgentOCR history rendering.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--candidates-per-case", type=int, default=8)
    parser.add_argument("--scan-items", type=int, default=20)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()

    output_path = Path(args.output)
    artifact_root = output_path.parent / f"{output_path.stem}_artifacts"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.data_path).open() as f:
        data = json.load(f)
    cases = _pick_cases(
        data,
        limit=args.limit,
        candidates_per_case=args.candidates_per_case,
        max_items_to_scan=args.scan_items,
    )
    if not cases:
        raise RuntimeError("No suitable Mind2Web grounding cases were found.")

    processor, model = _load_model(args.model_path, args.device)
    rows: List[Dict[str, Any]] = []

    with _temporary_env({"AGENTOCR_QWEN3_HISTORY_PAGES": "0"}):
        for case in cases:
            for variant in ("baseline", "full"):
                ocr_tool = _build_ocr_tool()
                rendered = _render_history(case=case, variant=variant, tool=ocr_tool)
                prompt_text = _build_prompt(case)
                raw_output, mm_diagnostics, processed_images = _run_generation(
                    model=model,
                    processor=processor,
                    prompt_text=prompt_text,
                    images=rendered["images"],
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                )
                predicted_id = _parse_candidate_id(raw_output, case.candidates)
                success = predicted_id == case.positive_candidate.backend_node_id
                case_dir = artifact_root / variant / f"{case.annotation_id}_step{case.step_index + 1}"
                _save_case_artifacts(
                    case_dir,
                    prompt_text=prompt_text,
                    raw_output=raw_output,
                    rendered_images=processed_images,
                    candidate_descriptions=[candidate.summary for candidate in case.candidates],
                )
                rows.append(
                    {
                        "variant": variant,
                        "annotation_id": case.annotation_id,
                        "website": case.website,
                        "domain": case.domain,
                        "subdomain": case.subdomain,
                        "task": case.task,
                        "step_index": case.step_index,
                        "total_steps": case.total_steps,
                        "action_repr": case.action_repr,
                        "operation": case.operation,
                        "positive_backend_node_id": case.positive_candidate.backend_node_id,
                        "predicted_backend_node_id": predicted_id,
                        "success": success,
                        "candidate_count": len(case.candidates),
                        "candidate_descriptions": [candidate.summary for candidate in case.candidates],
                        "history_context": case.history_context,
                        "processed_context": rendered["processed_context"],
                        "prompt_text": prompt_text,
                        "raw_output": raw_output,
                        "mm_diagnostics": mm_diagnostics,
                        "trust_diagnostics": rendered["trust_diagnostics"],
                        "effective_render_compression": rendered["effective_compression"],
                        "artifact_dir": str(case_dir),
                    }
                )

    payload = {
        "summary": {
            "model_path": args.model_path,
            "device": args.device,
            "data_path": args.data_path,
            "num_cases": len(cases),
            "baseline": _summarize(rows, "baseline"),
            "full": _summarize(rows, "full"),
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote Mind2Web grounding smoke to {output_path}")


if __name__ == "__main__":
    main()
