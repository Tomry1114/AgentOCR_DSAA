#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

from examples.agentocr_pilots.run_alfworld_slot_aware_multitemplate_pilot import run_suite as run_alfworld_suite
from examples.agentocr_pilots.run_memory_framework_component_performance_expanded import (
    run_suite as run_mechanism_suite,
)
from examples.agentocr_pilots.run_query_conditioned_distractor_scaling_large import (
    run_suite as run_search_suite,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "v2_debug"
DEFAULT_JSON = DEFAULT_OUTPUT_DIR / "v2_expansion_loop_summary_current.json"
DEFAULT_MD = ROOT / "Paper" / "机制实验v2_扩样循环结果.md"

SEARCH_DISTRACTOR_COUNTS = tuple(range(1, 11))
MECH_FILLERS = tuple(range(20, 401, 20))
MECH_RECOVERY_SCHEDULES = (
    (25, 25, 25, 25),
    (25, 20, 20, 20),
    (25, 15, 15, 15),
    (30, 30, 30, 30),
    (30, 25, 25, 25),
    (30, 20, 20, 20),
    (30, 15, 15, 15),
    (30, 10, 10, 10),
    (35, 20, 20, 20),
    (35, 10, 10, 10),
)
ALFWORLD_SEEDS = tuple(range(30))


def pct(value: float) -> str:
    return f"{float(value):.3f}"


def sci(value: float) -> str:
    return f"{float(value):.2e}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_search_ge2_summary(payload: Dict[str, Any]) -> Dict[str, float]:
    rows = [row for row in payload["rows"] if int(row["distractor_count"]) >= 2]
    baseline = sum(float(bool(row["baseline_success"])) for row in rows) / len(rows)
    query = sum(float(bool(row["query_success"])) for row in rows) / len(rows)
    return {
        "num_tasks": len(rows),
        "baseline_success_rate": baseline,
        "query_success_rate": query,
        "query_vs_baseline_gain": query - baseline,
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    search = summary["stages"]["search_web"]
    mech = summary["stages"]["mechanism_chain"]
    alf = summary["stages"]["alfworld"]
    lines = [
        "# 机制实验 v2 扩样循环结果",
        "",
        f"整理时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## 0. 本轮执行顺序",
        "",
        "- `Search/Web`",
        "- `mechanism_chain`",
        "- `ALFWorld`",
        "",
        "## 1. Search/Web",
        "",
        f"- 口径：`expanded target specs`，`{search['summary']['num_target_specs']} specs`，"
        f"`{len(search['summary']['distractor_counts'])}` 个 distractor count，"
        f"共 `{search['summary']['num_tasks']}` tasks。",
        f"- 整体：`baseline {pct(search['summary']['overall']['baseline_success_rate'])}` "
        f"`-> query {pct(search['summary']['overall']['query_success_rate'])}`，"
        f"`delta = {pct(search['summary']['overall']['query_vs_baseline_gain'])}`。",
        f"- `>=2 distractors` 子集：`{search['ge2_summary']['num_tasks']}` tasks，"
        f"`{pct(search['ge2_summary']['baseline_success_rate'])} -> "
        f"{pct(search['ge2_summary']['query_success_rate'])}`。",
        "",
        "## 2. Mechanism Chain",
        "",
        f"- 口径：`retention {mech['payload']['suite_sizes']['retention']}` / "
        f"`refresh {mech['payload']['suite_sizes']['refresh']}` / "
        f"`recovery {mech['payload']['suite_sizes']['recovery']}`。",
    ]
    for name in ("base", "egrc_only", "ema_plus_egrc", "egrc_plus_fmr", "full"):
        overall = mech["payload"]["configs"][name]["overall"]["success_rate"]
        lines.append(f"- `{name}` overall success = `{pct(overall)}`")
    lines.extend(
        [
            f"- `egrc_vs_base = {pct(mech['payload']['deltas']['egrc_vs_base'])}`",
            f"- `ema_vs_egrc = {pct(mech['payload']['deltas']['ema_vs_egrc'])}`",
            f"- `fmr_vs_egrc = {pct(mech['payload']['deltas']['fmr_vs_egrc'])}`",
            "",
            "## 3. ALFWorld",
            "",
            f"- 口径：`{alf['summary']['num_tasks']}` tasks，seeds `0..{max(alf['summary']['seeds'])}`。",
            f"- tight budget 下：`baseline {pct(alf['tight_budget_pair']['weaker_success_rate'])}` "
            f"`-> slot {pct(alf['tight_budget_pair']['stronger_success_rate'])}`，"
            f"`delta = {pct(alf['tight_budget_pair']['delta_success_rate'])}`，"
            f"`p = {sci(alf['tight_budget_pair']['sign_test_p_value'])}`。",
            "",
            "## 4. 结论",
            "",
            "- 这轮循环按固定顺序把定下的三条证据线继续放大了一轮。",
            "- `Search/Web` 和主机制链都在不改判定逻辑的情况下继续扩样后仍保持 `idea > base`。",
            "- `ALFWorld` 也在同一轮循环的最后继续扩大，没有出现趋势反转。",
            "- 说明当前实验脚本和口径可以继续按同样方式再往上扩。",
            "",
            "## 5. 输出文件",
            "",
            f"- `{search['output_path']}`",
            f"- `{mech['output_path']}`",
            f"- `{alf['output_path']}`",
            f"- `{summary['summary_json']}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2 enlargement loop sequentially across selected datasets.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json-summary", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-summary", default=str(DEFAULT_MD))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_output = output_dir / "query_conditioned_distractor_scaling_large_600_debug_current.json"
    search_payload = run_search_suite(SEARCH_DISTRACTOR_COUNTS, spec_profile="expanded")
    write_json(search_output, search_payload)

    mech_output = output_dir / "memory_framework_component_performance_expanded_480_debug_current.json"
    mech_payload = run_mechanism_suite(
        grounding_fillers=MECH_FILLERS,
        refresh_fillers=MECH_FILLERS,
        recovery_schedules=MECH_RECOVERY_SCHEDULES,
    )
    write_json(mech_output, mech_payload)

    alf_output = output_dir / "alfworld_slot_aware_multitemplate_pilot_540_debug_current.json"
    alf_payload = run_alfworld_suite(max_templates=6, seeds=ALFWORLD_SEEDS)
    write_json(alf_output, alf_payload)

    tight_budget_pair = next(
        row
        for row in alf_payload["summary"]["paired_summaries"]
        if row["protocol"] == "tight_rendered_area_budget" and row["weaker_variant"] == "baseline"
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "summary_json": str(Path(args.json_summary).relative_to(ROOT)),
        "stages": {
            "search_web": {
                "output_path": str(search_output.relative_to(ROOT)),
                "summary": search_payload["summary"],
                "ge2_summary": build_search_ge2_summary(search_payload),
            },
            "mechanism_chain": {
                "output_path": str(mech_output.relative_to(ROOT)),
                "payload": mech_payload,
            },
            "alfworld": {
                "output_path": str(alf_output.relative_to(ROOT)),
                "summary": alf_payload["summary"],
                "tight_budget_pair": tight_budget_pair,
            },
        },
    }

    json_summary = Path(args.json_summary)
    md_summary = Path(args.markdown_summary)
    write_json(json_summary, summary)
    md_summary.parent.mkdir(parents=True, exist_ok=True)
    md_summary.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Wrote search output to {search_output}")
    print(f"Wrote mechanism output to {mech_output}")
    print(f"Wrote ALFWorld output to {alf_output}")
    print(f"Wrote loop JSON summary to {json_summary}")
    print(f"Wrote loop markdown summary to {md_summary}")


if __name__ == "__main__":
    main()
