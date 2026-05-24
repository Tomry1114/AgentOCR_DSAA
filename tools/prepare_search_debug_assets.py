#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import faiss
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.search.retriever.retrieval_server import Encoder


def _stringify_targets(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "target" in value:
            return _stringify_targets(value["target"])
        return json.dumps(value, ensure_ascii=True)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    return str(value).strip()


def _extract_question(row: pd.Series) -> str:
    env_kwargs = row.get("env_kwargs", {}) or {}
    extra_info = row.get("extra_info", {}) or {}
    return str(env_kwargs.get("question") or extra_info.get("question") or "").strip()


def _extract_answer(row: pd.Series) -> str:
    env_kwargs = row.get("env_kwargs", {}) or {}
    reward_model = row.get("reward_model", {}) or {}
    ground_truth = env_kwargs.get("ground_truth") or reward_model.get("ground_truth") or {}
    return _stringify_targets(ground_truth)


def _build_corpus_rows(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    combined = pd.concat([train_df, val_df], ignore_index=True)
    for index, row in combined.iterrows():
        question = _extract_question(row)
        answer = _extract_answer(row)
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        answer_text = answer or "unknown"
        rows.append(
            {
                "contents": (
                    f"{question}\n"
                    f"Reference answer: {answer_text}.\n"
                    f"This passage is a tiny debug retrieval note for the question: {question}"
                )
            }
        )
        rows.append(
            {
                "contents": (
                    f"Background for: {question}\n"
                    f"A concise explanation says the likely answer is {answer_text}."
                )
            }
        )
        rows.append(
            {
                "contents": (
                    f"Decoy context {index}\n"
                    f"This note repeats some query words from '{question}' but the important answer token remains {answer_text}."
                )
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_faiss_index(corpus_rows: list[dict[str, str]], model_path: str, device: str, output_path: Path) -> None:
    encoder = Encoder(
        model_name="e5",
        model_path=model_path,
        pooling_method="mean",
        max_length=256,
        use_fp16=False,
        device=device,
    )
    passages = [row["contents"] for row in corpus_rows]
    embeddings = encoder.encode(passages, is_query=False)
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    faiss.write_index(index, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tiny Search debug parquet/corpus/index assets.")
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--val-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train", type=int, default=2)
    parser.add_argument("--num-val", type=int, default=2)
    parser.add_argument("--retriever-model", default="intfloat/e5-base-v2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_source).head(args.num_train).copy()
    val_df = pd.read_parquet(args.val_source).head(args.num_val).copy()

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"
    corpus_path = output_dir / "wiki-18.jsonl"
    index_path = output_dir / "e5_Flat.index"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    corpus_rows = _build_corpus_rows(train_df, val_df)
    _write_jsonl(corpus_path, corpus_rows)
    _build_faiss_index(corpus_rows, args.retriever_model, args.device, index_path)

    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "corpus_path": str(corpus_path),
                "index_path": str(index_path),
                "num_train": len(train_df),
                "num_val": len(val_df),
                "num_corpus_docs": len(corpus_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
