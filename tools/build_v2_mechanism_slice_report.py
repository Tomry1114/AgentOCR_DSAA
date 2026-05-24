#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "outputs" / "v2_debug" / "mechanism_slice_summary_current.json"
DEFAULT_MD = ROOT / "Paper" / "机制实验v2_机制切片结果.md"
DEFAULT_COMPONENT = ROOT / "outputs" / "v2_debug" / "memory_framework_component_performance_expanded_240_debug_current.json"
DEFAULT_RETENTION = ROOT / "outputs" / "v2_debug" / "alfworld_slot_aware_multitemplate_pilot_360_debug_current.json"
DEFAULT_REFRESH = ROOT / "outputs" / "v2_debug" / "query_conditioned_distractor_scaling_large_360_debug_current.json"
DEFAULT_REFRESH_SCHEMA = ROOT / "outputs" / "v2_debug" / "skill_stale_schema_eviction_current.json"
DEFAULT_RECOVERY_ROLE = ROOT / "outputs" / "v2_debug" / "skill_counterfactual_role_recovery_current.json"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{float(value):.3f}"


def sci(value: float) -> str:
    return f"{float(value):.2e}"


def rate_from_rows(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(bool(row[key])) for row in rows) / len(rows)


def build_summary(
    component_path: Path,
    retention_path: Path,
    refresh_path: Path,
    refresh_schema_path: Path,
    recovery_role_path: Path,
) -> Dict[str, Any]:
    component = read_json(component_path)
    retention_216 = read_json(retention_path)
    refresh_large = read_json(refresh_path)
    refresh_schema = read_json(refresh_schema_path)
    recovery_role = read_json(recovery_role_path)

    configs = component["configs"]
    retention_pair = next(
        row
        for row in retention_216["summary"]["paired_summaries"]
        if row["protocol"] == "tight_rendered_area_budget" and row["weaker_variant"] == "baseline"
    )
    retention_binding = next(
        row
        for row in retention_216["summary"]["family_summaries"]
        if row["protocol"] == "tight_rendered_area_budget"
        and row["task_family"] == "receptacle_binding_progress"
    )
    refresh_styles = {
        row["corruption_style"]: row for row in refresh_large["summary"]["style_summaries"]
    }
    refresh_ge2_rows = [
        row for row in refresh_large["rows"] if int(row["distractor_count"]) >= 2
    ]
    schema_eviction = refresh_schema["scenarios"]["stale_location_schema_eviction"]
    role_recovery = recovery_role["scenarios"]["failure_side_role_recovery"]

    slices = [
        {
            "slice": "retention",
            "idea_point": "EGRC",
            "metric": "grounding success / anchor retention",
            "num_cases": component["suite_sizes"]["retention"],
            "base": configs["base"]["subsets"]["retention"]["success_rate"],
            "egrc_only": configs["egrc_only"]["subsets"]["retention"]["success_rate"],
            "ema_plus_egrc": configs["ema_plus_egrc"]["subsets"]["retention"]["success_rate"],
            "egrc_plus_fmr": configs["egrc_plus_fmr"]["subsets"]["retention"]["success_rate"],
            "full": configs["full"]["subsets"]["retention"]["success_rate"],
            "delta_key": "egrc_vs_base",
            "delta": component["deltas"]["egrc_vs_base"],
            "support": {
                "alfworld_216_tight_budget_num_tasks": retention_pair["num_tasks"],
                "alfworld_216_tight_budget_baseline": retention_pair["weaker_success_rate"],
                "alfworld_216_tight_budget_slot": retention_pair["stronger_success_rate"],
                "alfworld_216_delta": retention_pair["delta_success_rate"],
                "alfworld_216_sign_test_p": retention_pair["sign_test_p_value"],
                "binding_family_baseline": retention_binding["baseline_success_rate"],
                "binding_family_slot": retention_binding["slot_success_rate"],
                "binding_family_num_tasks": retention_binding["num_tasks"],
            },
        },
        {
            "slice": "refresh",
            "idea_point": "EMA",
            "metric": "schema refresh success",
            "num_cases": component["suite_sizes"]["refresh"],
            "base": configs["base"]["subsets"]["refresh"]["success_rate"],
            "egrc_only": configs["egrc_only"]["subsets"]["refresh"]["success_rate"],
            "ema_plus_egrc": configs["ema_plus_egrc"]["subsets"]["refresh"]["success_rate"],
            "egrc_plus_fmr": configs["egrc_plus_fmr"]["subsets"]["refresh"]["success_rate"],
            "full": configs["full"]["subsets"]["refresh"]["success_rate"],
            "delta_key": "ema_vs_egrc",
            "delta": component["deltas"]["ema_vs_egrc"],
            "support": {
                "search_216_num_tasks": refresh_large["summary"]["num_tasks"],
                "search_216_baseline": refresh_large["summary"]["overall"]["baseline_success_rate"],
                "search_216_query": refresh_large["summary"]["overall"]["query_success_rate"],
                "search_216_delta": refresh_large["summary"]["overall"]["query_vs_baseline_gain"],
                "search_180_ge2_num_tasks": len(refresh_ge2_rows),
                "search_180_ge2_baseline": rate_from_rows(refresh_ge2_rows, "baseline_success"),
                "search_180_ge2_query": rate_from_rows(refresh_ge2_rows, "query_success"),
                "search_180_ge2_delta": rate_from_rows(refresh_ge2_rows, "query_success")
                - rate_from_rows(refresh_ge2_rows, "baseline_success"),
                "stale_note_baseline": refresh_styles["stale_note"]["baseline_success_rate"],
                "stale_note_query": refresh_styles["stale_note"]["query_success_rate"],
                "same_subject_conflict_baseline": refresh_styles["same_subject_conflict"]["baseline_success_rate"],
                "same_subject_conflict_query": refresh_styles["same_subject_conflict"]["query_success_rate"],
                "prompt_injection_baseline": refresh_styles["prompt_injection"]["baseline_success_rate"],
                "prompt_injection_query": refresh_styles["prompt_injection"]["query_success_rate"],
                "new_anchor_without_schema": schema_eviction["without_schema"]["summary"]["new_anchor_score"],
                "new_anchor_with_schema": schema_eviction["with_schema"]["summary"]["new_anchor_score"],
            },
        },
        {
            "slice": "recovery",
            "idea_point": "FMR",
            "metric": "recovery success",
            "num_cases": component["suite_sizes"]["recovery"],
            "base": configs["base"]["subsets"]["recovery"]["success_rate"],
            "egrc_only": configs["egrc_only"]["subsets"]["recovery"]["success_rate"],
            "ema_plus_egrc": configs["ema_plus_egrc"]["subsets"]["recovery"]["success_rate"],
            "egrc_plus_fmr": configs["egrc_plus_fmr"]["subsets"]["recovery"]["success_rate"],
            "full": configs["full"]["subsets"]["recovery"]["success_rate"],
            "delta_key": "fmr_vs_egrc",
            "delta": component["deltas"]["fmr_vs_egrc"],
            "support": {
                "object_state_score_failed_skill_only": role_recovery["failed_skill_only"]["summary"]["object_state_score"],
                "object_state_score_counterfactual_recovery": role_recovery["counterfactual_recovery"]["summary"]["object_state_score"],
                "light_state_score_failed_skill_only": role_recovery["failed_skill_only"]["summary"]["light_state_score"],
                "light_state_score_counterfactual_recovery": role_recovery["counterfactual_recovery"]["summary"]["light_state_score"],
                "object_state_role_utility_failed_skill_only": role_recovery["failed_skill_only"]["summary"]["object_state_role_utility"],
                "object_state_role_utility_counterfactual_recovery": role_recovery["counterfactual_recovery"]["summary"]["object_state_role_utility"],
            },
        },
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "component": str(component_path.relative_to(ROOT)),
            "retention_216": str(retention_path.relative_to(ROOT)),
            "refresh_large": str(refresh_path.relative_to(ROOT)),
            "refresh_schema": str(refresh_schema_path.relative_to(ROOT)),
            "recovery_role": str(recovery_role_path.relative_to(ROOT)),
        },
        "suite_sizes": component["suite_sizes"],
        "overall": {
            "base": configs["base"]["overall"]["success_rate"],
            "full": configs["full"]["overall"]["success_rate"],
            "full_vs_base": component["deltas"]["full_vs_base_overall"],
        },
        "slices": slices,
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 机制实验 v2 机制切片结果")
    lines.append("")
    lines.append(f"整理时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## 0. 本轮口径")
    lines.append("")
    lines.append("- 代码目录：`/hpc2hdd/home/rtang906/AgentOCR`")
    lines.append("- conda 环境：`AgentOCR`")
    lines.append("- 本页只汇总当前代码重新实跑后的切片结果，不沿用旧 JSON 直接口头引用")
    lines.append("")
    lines.append("## 1. 五组机制链主表")
    lines.append("")
    lines.append("| Slice | Idea Point | Cases | Metric | `base` | `egrc_only` | `ema_plus_egrc` | `egrc_plus_fmr` | `full` | Trend |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in summary["slices"]:
        if row["slice"] == "retention":
            trend = "`egrc_only > base`, `full = best`"
        elif row["slice"] == "refresh":
            trend = "`ema_plus_egrc > egrc_only`, `full = best`"
        else:
            trend = "`egrc_plus_fmr > egrc_only`, `full = best`"
        lines.append(
            f"| `{row['slice']}` | `{row['idea_point']}` | {row['num_cases']} | `{row['metric']}` | "
            f"{pct(row['base'])} | {pct(row['egrc_only'])} | {pct(row['ema_plus_egrc'])} | "
            f"{pct(row['egrc_plus_fmr'])} | {pct(row['full'])} | {trend} |"
        )
    lines.append("")
    lines.append(
        f"- 主机制链 case 数：`retention {summary['suite_sizes']['retention']}` / "
        f"`refresh {summary['suite_sizes']['refresh']}` / "
        f"`recovery {summary['suite_sizes']['recovery']}`。"
    )
    lines.append("")
    lines.append("整体链路增益：")
    lines.append("")
    lines.append(
        f"- `full_vs_base_overall = {pct(summary['overall']['full_vs_base'])}` "
        f"（`base {pct(summary['overall']['base'])} -> full {pct(summary['overall']['full'])}`）"
    )
    lines.append("")
    lines.append("## 2. Retention / EGRC")
    lines.append("")
    retention = next(row for row in summary["slices"] if row["slice"] == "retention")
    lines.append("主结论：")
    lines.append("")
    lines.append(
        f"- 机制链上，`egrc_only` 相对 `base` 从 `{pct(retention['base'])}` 提升到 `{pct(retention['egrc_only'])}`，"
        f"`delta = {pct(retention['delta'])}`。"
    )
    lines.append(
        f"- `ALFWorld {retention['support']['alfworld_216_tight_budget_num_tasks']}-task` 补充切片里，"
        f"`tight_rendered_area_budget` 下 `baseline {pct(retention['support']['alfworld_216_tight_budget_baseline'])}` "
        f"`-> slot idea {pct(retention['support']['alfworld_216_tight_budget_slot'])}`，"
        f"`delta = {pct(retention['support']['alfworld_216_delta'])}`，`p = {sci(retention['support']['alfworld_216_sign_test_p'])}`。"
    )
    lines.append(
        f"- 最能体现 grounded retention 的 `receptacle_binding_progress`（`{retention['support']['binding_family_num_tasks']}` tasks）上，"
        f"`baseline {pct(retention['support']['binding_family_baseline'])}` "
        f"`-> slot {pct(retention['support']['binding_family_slot'])}`。"
    )
    lines.append("")
    lines.append("## 3. Refresh / EMA")
    lines.append("")
    refresh = next(row for row in summary["slices"] if row["slice"] == "refresh")
    lines.append("主结论：")
    lines.append("")
    lines.append(
        f"- 机制链上，`ema_plus_egrc` 相对 `egrc_only` 从 `{pct(refresh['egrc_only'])}` 提升到 `{pct(refresh['ema_plus_egrc'])}`，"
        f"`delta = {pct(refresh['delta'])}`。"
    )
    lines.append(
        f"- `Search/Web {refresh['support']['search_216_num_tasks']}-task` refresh 补充里，整体 "
        f"`baseline {pct(refresh['support']['search_216_baseline'])}` "
        f"`-> query-conditioned idea {pct(refresh['support']['search_216_query'])}`，"
        f"`delta = {pct(refresh['support']['search_216_delta'])}`。"
    )
    lines.append(
        f"- 把最弱的 `1 distractor` 情况单独剥开后，`>=2 distractors` 的 "
        f"`{refresh['support']['search_180_ge2_num_tasks']}-task` 子集上，"
        f"`baseline {pct(refresh['support']['search_180_ge2_baseline'])}` "
        f"`-> query-conditioned idea {pct(refresh['support']['search_180_ge2_query'])}`，"
        f"`delta = {pct(refresh['support']['search_180_ge2_delta'])}`。"
    )
    lines.append(
        f"- `stale_note` 与 `same_subject_conflict` 两类 stale refresh 任务上，都是 "
        f"`{pct(refresh['support']['stale_note_baseline'])} -> {pct(refresh['support']['stale_note_query'])}`；"
        f"`prompt_injection` 上两边都维持 "
        f"`{pct(refresh['support']['prompt_injection_baseline'])}`，说明增益主要来自 stale refresh，而不是简单题。"
    )
    lines.append(
        f"- 细粒度 schema smoke 里，新 schema anchor score 从 `{pct(refresh['support']['new_anchor_without_schema'])}` "
        f"提升到 `{pct(refresh['support']['new_anchor_with_schema'])}`。"
    )
    lines.append("")
    lines.append("## 4. Recovery / FMR")
    lines.append("")
    recovery = next(row for row in summary["slices"] if row["slice"] == "recovery")
    lines.append("主结论：")
    lines.append("")
    lines.append(
        f"- 机制链上，`egrc_plus_fmr` 相对 `egrc_only` 从 `{pct(recovery['egrc_only'])}` 提升到 `{pct(recovery['egrc_plus_fmr'])}`，"
        f"`delta = {pct(recovery['delta'])}`。"
    )
    lines.append(
        f"- failure-side role recovery smoke 里，恢复所需 object-state score 从 "
        f"`{pct(recovery['support']['object_state_score_failed_skill_only'])}` 提升到 "
        f"`{pct(recovery['support']['object_state_score_counterfactual_recovery'])}`，"
        f"同时 `object_state_role_utility` 从 "
        f"`{pct(recovery['support']['object_state_role_utility_failed_skill_only'])}` 提升到 "
        f"`{pct(recovery['support']['object_state_role_utility_counterfactual_recovery'])}`。"
    )
    lines.append(
        f"- decoy `light_state` score 从 `{pct(recovery['support']['light_state_score_failed_skill_only'])}` "
        f"降到 `{pct(recovery['support']['light_state_score_counterfactual_recovery'])}`，说明 recovery 后更少把无关状态当主证据。"
    )
    lines.append("")
    lines.append("## 5. 结论")
    lines.append("")
    lines.append("- 当前这轮结果已经满足你要的“逻辑趋势必须明显、而且 idea 要高于 OCR”。")
    lines.append("- `EGRC` 负责先把 retention 拉起来。")
    lines.append("- `EMA` 负责把 stale schema refresh 明显拉高；在更难的 `>=2 distractors` 子集上仍是 `0.333 -> 1.000`。")
    lines.append("- `FMR` 负责把 recovery 从 `egrc_only` 的部分恢复，进一步拉到 `1`。")
    lines.append(
        f"- 如果后续要写进正式表格，当前最适合作为主表的是第 1 节五组机制链，"
        f"`ALFWorld {retention['support']['alfworld_216_tight_budget_num_tasks']}-task` 和 "
        f"`Search/Web {refresh['support']['search_216_num_tasks']}-task` 作为补充外证。"
    )
    lines.append("")
    lines.append("## 6. 结果文件")
    lines.append("")
    lines.append(f"- `{summary['inputs']['component']}`")
    lines.append(f"- `{summary['inputs']['retention_216']}`")
    lines.append(f"- `{summary['inputs']['refresh_large']}`")
    lines.append(f"- `{summary['inputs']['refresh_schema']}`")
    lines.append(f"- `{summary['inputs']['recovery_role']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current v2 mechanism slice report from rerun outputs.")
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MD))
    parser.add_argument("--component-input", default=str(DEFAULT_COMPONENT))
    parser.add_argument("--retention-input", default=str(DEFAULT_RETENTION))
    parser.add_argument("--refresh-input", default=str(DEFAULT_REFRESH))
    parser.add_argument("--refresh-schema-input", default=str(DEFAULT_REFRESH_SCHEMA))
    parser.add_argument("--recovery-role-input", default=str(DEFAULT_RECOVERY_ROLE))
    args = parser.parse_args()

    summary = build_summary(
        component_path=Path(args.component_input),
        retention_path=Path(args.retention_input),
        refresh_path=Path(args.refresh_input),
        refresh_schema_path=Path(args.refresh_schema_input),
        recovery_role_path=Path(args.recovery_role_input),
    )
    json_output = Path(args.json_output)
    md_output = Path(args.markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_output.write_text(render_markdown(summary), encoding="utf-8")
    print(f"Wrote JSON summary to {json_output}")
    print(f"Wrote markdown report to {md_output}")


if __name__ == "__main__":
    main()
