"""Compile baseline outputs without normalizing or combining native scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Callable

from baselines.common import write_json
from baselines.videoscore.mapping import ASPECTS as VIDEOSCORE_ASPECTS


Component = tuple[str, tuple[float, float], Callable[[dict[str, Any]], float]]


def _field(name: str) -> Callable[[dict[str, Any]], float]:
    return lambda row: float(row[name])


def _nested(parent: str, name: str) -> Callable[[dict[str, Any]], float]:
    return lambda row: float(row[parent][name])


BENCHMARKS: dict[str, dict[str, Any]] = {
    "pqsg": {
        "path": Path("pqsg/report.json"),
        "label": "PQSG",
        "components": [("final", (0.0, 1.0), _field("native_score_0_to_1"))],
    },
    "wr_arena": {
        "path": Path("wr_arena/report.json"),
        "label": "WR-Arena Action Simulation Fidelity",
        "components": [("action_fidelity", (0.0, 3.0), _field("native_score_0_to_3"))],
    },
    "rbench": {
        "path": Path("rbench/report.json"),
        "label": "RBench-Ego2Act Common Manipulation",
        "components": [("final", (1.0, 5.0), _field("native_score_1_to_5"))],
    },
    "worldmodelbench": {
        "path": Path("worldmodelbench/report.json"),
        "label": "WorldModelBench",
        "components": [
            ("instruction", (0.0, 3.0), _nested("native_scores", "instruction_0_to_3")),
            ("physical_laws", (0.0, 5.0), _nested("native_scores", "physical_laws_0_to_5")),
            ("common_sense", (0.0, 2.0), _nested("native_scores", "common_sense_0_to_2")),
            ("official_total", (0.0, 10.0), _nested("native_scores", "official_total_0_to_10")),
        ],
    },
    "videoscore": {
        "path": Path("videoscore/report.json"),
        "label": "VideoScore",
        "components": [
            (aspect, (1.0, 4.0), _nested("prediction", aspect))
            for aspect in VIDEOSCORE_ASPECTS
        ],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile native baseline scores without remapping."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "all": rows,
        "correct": [row for row in rows if row.get("category") == "correct"],
        "wrong": [row for row in rows if row.get("category") == "wrong"],
        "ai": [row for row in rows if row.get("category") == "ai"],
    }
    for generator in sorted({row.get("generator") for row in rows} - {None}):
        groups[f"ai/{generator}"] = [
            row for row in rows if row.get("generator") == generator
        ]
    return groups


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "videos": len(values),
        "mean": round(fmean(values), 6) if values else None,
        "median": round(median(values), 6) if values else None,
        "std": round(pstdev(values), 6) if values else None,
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def compile_results(results_root: Path) -> dict[str, Any]:
    compiled: dict[str, Any] = {}
    id_sets: dict[str, set[str]] = {}
    for name, spec in BENCHMARKS.items():
        path = results_root / spec["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        identifiers = [str(row["id"]) for row in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} contains duplicate sample IDs")
        id_sets[name] = set(identifiers)

        successful: list[dict[str, Any]] = []
        component_values: dict[str, dict[str, float]] = {}
        for row in records:
            values: dict[str, float] = {}
            try:
                for component, score_range, getter in spec["components"]:
                    value = getter(row)
                    if not math.isfinite(value) or not score_range[0] <= value <= score_range[1]:
                        raise ValueError(
                            f"{name}/{row['id']}/{component}={value} outside {score_range}"
                        )
                    values[component] = value
            except (KeyError, TypeError):
                continue
            successful.append(row)
            component_values[str(row["id"])] = values

        group_summaries: dict[str, Any] = {}
        for group, rows in _groups(successful).items():
            group_summaries[group] = {
                component: _statistics([
                    component_values[str(row["id"])][component] for row in rows
                ])
                for component, _, _ in spec["components"]
            }
        compiled[name] = {
            "label": spec["label"],
            "source": str(path),
            "requested_videos": payload.get("requested_videos", len(records)),
            "recorded_videos": len(records),
            "successful_videos": len(successful),
            "failed_videos": len(records) - len(successful),
            "native_components": {
                component: {"range": list(score_range), "higher_is_better": True}
                for component, score_range, _ in spec["components"]
            },
            "groups": group_summaries,
        }

    union = set().union(*id_sets.values())
    sample_set_differences = {
        name: sorted(union - identifiers)
        for name, identifiers in id_sets.items()
        if identifiers != union
    }
    return {
        "schema_version": 1,
        "score_policy": (
            "Native benchmark brackets only; no affine remapping, normalization, "
            "cross-benchmark averaging, or synthetic VideoScore total."
        ),
        "sample_set_union": len(union),
        "sample_set_differences": sample_set_differences,
        "benchmarks": compiled,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native baseline results",
        "",
        report["score_policy"],
        "",
        f"Sample-set union: **{report['sample_set_union']} videos**.",
        "",
    ]
    if report["sample_set_differences"]:
        lines.extend([
            "Warning: benchmark sample sets differ; see the JSON report for IDs.",
            "",
        ])
    for benchmark in report["benchmarks"].values():
        lines.extend([
            f"## {benchmark['label']}",
            "",
            f"Coverage: **{benchmark['successful_videos']}/{benchmark['recorded_videos']}**.",
            "",
            "| Group | Component | Native range | n | Mean | Median | Std | Min | Max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for group, components in benchmark["groups"].items():
            for component, stats in components.items():
                score_range = benchmark["native_components"][component]["range"]
                value = lambda key: "n/a" if stats[key] is None else f"{stats[key]:.4f}"
                lines.append(
                    f"| {group} | {component} | {score_range[0]:g}–{score_range[1]:g} | "
                    f"{stats['videos']} | {value('mean')} | {value('median')} | "
                    f"{value('std')} | {value('min')} | {value('max')} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    root = args.results_root.expanduser().resolve()
    report = compile_results(root)
    output_json = (args.output_json or root / "native_results.json").expanduser().resolve()
    output_md = (args.output_md or root / "native_results.md").expanduser().resolve()
    write_json(output_json, report)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Native JSON: {output_json}")
    print(f"Native Markdown: {output_md}")
    return 0 if not report["sample_set_differences"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
