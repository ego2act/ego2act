from __future__ import annotations

from collections.abc import Iterable
from statistics import fmean
from typing import Any


PHYSICAL_LABELS = ("newton", "mass_or_solid", "fluid", "penetration", "gravity")
COMMON_SENSE_LABELS = ("aesthetics", "temporal_consistency")


def native_scores(record: dict[str, Any]) -> dict[str, float]:
    """Return the four scores defined by the official WorldModelBench runner."""
    instruction = float(record["instruction"]["score"])
    physical = sum(float(item["pass"]) for item in record["physical_laws"])
    common = sum(float(item["pass"]) for item in record["common_sense"])
    return {
        "instruction_0_to_3": instruction,
        "physical_laws_0_to_5": physical,
        "common_sense_0_to_2": common,
        "official_total_0_to_10": instruction + physical + common,
    }


def affine_1_to_4(score: float, native_max: float) -> float:
    return 1.0 + 3.0 * float(score) / native_max


def enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    scores = native_scores(record)
    common = record["common_sense"]
    if len(common) != 2:
        raise ValueError("WorldModelBench common-sense output must contain two checks")
    temporal = float(common[1]["pass"])
    return {
        **record,
        "native_scores": scores,
        "affine_scores_1_to_4": {
            "official_total": affine_1_to_4(scores["official_total_0_to_10"], 10),
            "instruction": affine_1_to_4(scores["instruction_0_to_3"], 3),
            "physical_laws": affine_1_to_4(scores["physical_laws_0_to_5"], 5),
            "temporal_consistency": affine_1_to_4(temporal, 1),
        },
    }


def rubric_proxy_scores(record: dict[str, Any]) -> dict[str, float]:
    """Map native components to the current Task/Physics rubric for diagnostics.

    The official WorldModelBench total remains the primary score.  These fixed,
    non-fitted proxies only make the closest available component comparison
    explicit: instruction follows Task, while the five physical-law checks plus
    temporal consistency cover the closest observable subset of Physics.
    """
    scores = record["native_scores"]
    common = record["common_sense"]
    if len(common) != 2:
        raise ValueError("WorldModelBench common-sense output must contain two checks")
    temporal = float(common[1]["pass"])
    task = float(scores["instruction_0_to_3"])
    physics = 4.0 * (float(scores["physical_laws_0_to_5"]) + temporal) / 6.0
    return {
        "task_proxy_0_to_3": round(task, 6),
        "physics_proxy_0_to_4": round(physics, 6),
        "unified_proxy_0_to_1": round(0.5 * (task / 3.0 + physics / 4.0), 6),
    }


def _mean(items: Iterable[float]) -> float | None:
    values = list(items)
    return round(fmean(values), 6) if values else None


def _groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "all": records,
        "correct": [record for record in records if record["category"] == "correct"],
        "wrong": [record for record in records if record["category"] == "wrong"],
        "ai": [record for record in records if record["category"] == "ai"],
    }
    for generator in sorted({record.get("generator") for record in records} - {None}):
        groups[f"ai/{generator}"] = [
            record for record in records if record.get("generator") == generator
        ]
    return groups


def native_score_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize original scores without fitting or construct remapping."""
    successful = [record for record in records if "native_scores" in record]
    groups = _groups(successful)
    keys = (
        "instruction_0_to_3",
        "physical_laws_0_to_5",
        "common_sense_0_to_2",
        "official_total_0_to_10",
    )
    summary: dict[str, Any] = {}
    for name, rows in groups.items():
        summary[name] = {
            "videos": len(rows),
            "mean": {
                key: _mean(record["native_scores"][key] for record in rows)
                for key in keys
            },
        }
    correct = summary["correct"]["mean"]
    wrong = summary["wrong"]["mean"]
    summary["correct_minus_wrong"] = {
        key: (
            round(float(correct[key]) - float(wrong[key]), 6)
            if correct[key] is not None and wrong[key] is not None
            else None
        )
        for key in keys
    }
    return {
        "definition": (
            "Official per-video total = instruction (0-3) + five physical-law "
            "pass indicators (0-5) + two common-sense pass indicators (0-2)."
        ),
        "higher_is_better": True,
        "groups": summary,
    }


def rubric_proxy_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if "rubric_proxy_scores" in record]
    fields = (
        "task_proxy_0_to_3",
        "physics_proxy_0_to_4",
        "unified_proxy_0_to_1",
    )
    return {
        "definition": (
            "Fixed Task/Physics diagnostics from native components; not an "
            "official WorldModelBench aggregate."
        ),
        "groups": {
            name: {
                "videos": len(rows),
                "mean": {
                    field: _mean(record["rubric_proxy_scores"][field] for record in rows)
                    for field in fields
                },
            }
            for name, rows in _groups(successful).items()
        },
    }


def render_report(report: dict[str, Any]) -> str:
    native = report["native_score_report"]["groups"]

    def cell(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    lines = [
        "# WorldModelBench on Ego2Act",
        "",
        f"Successful videos: **{report['successful_videos']}/{report['requested_videos']}**  ",
        f"Execution: {report['execution']}  ",
        f"Judge: `{report['judge']['repository']}@{report['judge']['revision']}`",
        "",
        "## Original WorldModelBench scores",
        "",
        "These are the unmodified upstream score components and total. Higher is better.",
        "No Ego2Act label, category, failure description, or object list was supplied to the judge.",
        "",
        "| Group | n | Instruction (0-3) | Physical laws (0-5) | Common sense (0-2) | Official total (0-10) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in native:
        if group == "correct_minus_wrong":
            continue
        block = native[group]
        means = block["mean"]
        lines.append(
            f"| {group} | {block['videos']} | {cell(means['instruction_0_to_3'])} | "
            f"{cell(means['physical_laws_0_to_5'])} | "
            f"{cell(means['common_sense_0_to_2'])} | "
            f"{cell(means['official_total_0_to_10'])} |"
        )
    gap = native["correct_minus_wrong"]
    lines.append(
        "| correct - wrong | - | "
        f"{cell(gap['instruction_0_to_3'])} | {cell(gap['physical_laws_0_to_5'])} | "
        f"{cell(gap['common_sense_0_to_2'])} | {cell(gap['official_total_0_to_10'])} |"
    )
    lines.extend([
        "",
        "## Task/Physics alignment",
        "",
        "The official 0-10 total remains primary. For later rubric analysis, instruction",
        "is retained as a Task 0-3 proxy. The five physical-law checks plus temporal",
        "consistency are fixed to a Physics 0-4 proxy; aesthetics is excluded.",
        "No agreement metric is computed until Task/Physics annotations are joined.",
        "",
        "",
        "## Provenance",
        "",
        f"- WorldModelBench commit: `{report['official_commit']}`",
        f"- VILA commit: `{report['vila_commit']}`",
        f"- Runtime lock SHA-256: `{report['runtime_lock_sha256']}`",
        f"- Upstream tracked trees clean: `{report['tracked_trees_clean']}`",
        f"- Prompt/source integrity: `{report['source_integrity']}`",
        f"- Raw upstream responses: `{report['artifacts']['upstream_output']}`",
        f"- Adapter manifest: `{report['artifacts']['manifest']}`",
        f"- Label-free judge input: `{report['artifacts']['upstream_input']}`",
        "",
    ])
    return "\n".join(lines)
