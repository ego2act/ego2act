"""Build a current-schema VideoScore report from GPU scoring output.

The official five 1-4 aspects remain the native result. This report adds only
fixed Task/Physics diagnostic proxies and never depends on retired data.yml
labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from baselines.common import ROOT, sha256, write_json
from baselines.videoscore import mapping
from ego2act.data import evaluable_videos, load_dataset_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report native VideoScore results.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".results/baselines/videoscore/report.json",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ego2act")
    parser.add_argument(
        "--allow-input-drift",
        action="store_true",
        help="Report scores whose current video/goal fingerprint no longer matches.",
    )
    return parser


def _prediction(item: dict[str, Any]) -> dict[str, float]:
    if isinstance(item.get("prediction"), dict):
        return {
            aspect: float(item["prediction"][aspect]) for aspect in mapping.ASPECTS
        }
    values = item.get("scores")
    if not isinstance(values, list) or len(values) != len(mapping.ASPECTS):
        raise ValueError(f"Incomplete VideoScore output for {item.get('id')}")
    return dict(zip(mapping.ASPECTS, map(float, values)))


def _current_fingerprint(case: dict[str, Any], video: dict[str, Any]) -> str:
    video_hash = sha256(Path(video["video_path"]).resolve())
    prompt_hash = hashlib.sha256(case["goal"].encode("utf-8")).hexdigest()
    sample_id = f"{case['id']}::{video['key']}"
    return hashlib.sha256(json.dumps(
        {
            "id": sample_id,
            "video_sha256": video_hash,
            "prompt_sha256": prompt_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def load_scored_records(
    scores_path: Path,
    data_root: Path,
    allow_drift: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    successful = [item for item in scores.get("results", []) if "scores" in item]
    if not successful:
        raise ValueError(f"No successfully scored videos in {scores_path}")

    cases: dict[str, dict[str, Any]] = {}
    videos: dict[tuple[str, str], dict[str, Any]] = {}
    drift: list[str] = []
    records = []
    for item in successful:
        case_id = str(item["case"])
        video_key = str(item["video"])
        if case_id not in cases:
            cases[case_id] = load_dataset_case(case_id, data_root)
            videos.update({
                (case_id, video["key"]): video
                for video in evaluable_videos(cases[case_id])
            })
        video = videos.get((case_id, video_key))
        if video is None:
            drift.append(f"{case_id}::{video_key}: video is absent from current metadata")
        else:
            current = _current_fingerprint(cases[case_id], video)
            if item.get("sample_fingerprint") != current:
                drift.append(f"{case_id}::{video_key}: video or goal fingerprint changed")

        prediction = _prediction(item)
        records.append({
            **item,
            "id": item.get("id", f"{case_id}::{video_key}"),
            "prediction": prediction,
            "rubric_proxy_scores": mapping.rubric_proxy_scores(prediction),
        })

    if drift and not allow_drift:
        raise RuntimeError(
            "VideoScore inputs do not match the current metadata/video bytes:\n  "
            + "\n  ".join(drift)
            + "\nRe-score these videos, or explicitly pass --allow-input-drift."
        )
    return scores, records, drift


def _groups(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "all": records,
        "correct": [row for row in records if row["category"] == "correct"],
        "wrong": [row for row in records if row["category"] == "wrong"],
        "ai": [row for row in records if row["category"] == "ai"],
    }
    for generator in sorted({row.get("generator") for row in records} - {None}):
        groups[f"ai/{generator}"] = [
            row for row in records if row.get("generator") == generator
        ]
    return groups


def native_score_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    report = {}
    proxy_fields = (
        "task_proxy_0_to_3",
        "physics_proxy_0_to_4",
        "unified_proxy_0_to_1",
    )
    for name, rows in _groups(records).items():
        report[name] = {
            "videos": len(rows),
            "native_mean": {
                aspect: (
                    round(fmean(row["prediction"][aspect] for row in rows), 6)
                    if rows else None
                )
                for aspect in mapping.ASPECTS
            },
            "rubric_proxy_mean": {
                field: (
                    round(
                        fmean(row["rubric_proxy_scores"][field] for row in rows), 6
                    )
                    if rows else None
                )
                for field in proxy_fields
            },
        }
    return {
        "definition": "five independent upstream VideoScore aspects; no official scalar aggregate",
        "higher_is_better": True,
        "groups": report,
    }


def _failure_rate(scores: dict[str, Any]) -> float | None:
    requested = scores.get("requested_videos", scores.get("annotated_videos"))
    failed = scores.get("failed_videos")
    if not requested or failed is None:
        return None
    return round(float(failed) / float(requested), 6)


def main() -> int:
    args = build_parser().parse_args()
    scores_path = args.scores.expanduser().resolve()
    scores, records, drift = load_scored_records(
        scores_path,
        args.data_root.expanduser().resolve(),
        args.allow_input_drift,
    )
    report = {
        "method": "VideoScore",
        "execution": "official five-head VideoScore inference with current Ego2Act inputs",
        "upstream_repository": "https://github.com/TIGER-AI-Lab/VideoScore",
        "upstream_commit": scores.get("upstream_commit"),
        "mantis_repository": scores.get("mantis_repository"),
        "mantis_version": scores.get("mantis_version"),
        "mantis_commit": scores.get("mantis_commit"),
        "source_integrity": scores.get("source_integrity"),
        "model": scores.get("model"),
        "model_revision": scores.get("model_revision"),
        "model_verified": scores.get("model_verified"),
        "evaluator_fingerprint": scores.get("evaluator_fingerprint"),
        "native_scale": scores.get("native_scale"),
        "native_aspects": list(mapping.ASPECTS),
        "max_frames": scores.get("max_frames"),
        "prompt_sha256": scores.get("prompt_sha256"),
        "runtime_lock_sha256": scores.get("runtime_lock_sha256"),
        "scoring_runtime": scores.get("runtime"),
        "ego2act_commit": scores.get("ego2act_commit"),
        "scores_file": str(scores_path),
        "requested_videos": scores.get("requested_videos"),
        "successful_videos": len(records),
        "failed_videos": scores.get("failed_videos"),
        "failure_rate": _failure_rate(scores),
        "input_drift": drift,
        "current_inputs_verified": not drift,
        "primary_score": {
            "fields": list(mapping.ASPECTS),
            "range": [1.0, 4.0],
            "definition": "official independent VideoScore aspect vector; upstream defines no total",
        },
        "rubric_alignment": {
            "task": "text_to_video_alignment mapped from 1-4 to Task 0-3",
            "physics": (
                "mean(temporal_consistency, factual_consistency) mapped from 1-4 to Physics 0-4"
            ),
            "excluded": list(mapping.EXCLUDED_FROM_RUBRIC_PROXY),
            "unified": "0.5 * (task / 3 + physics / 4)",
            "status": "fixed diagnostic proxy; not an official VideoScore aggregate",
        },
        "native_score_report": native_score_report(records),
        "metrics": None,
        "metrics_note": (
            "Task/Physics human scores are not stored in metadata.json. Join them "
            "during analysis before computing agreement."
        ),
        "efficiency": {
            "scoring_elapsed_seconds": scores.get("elapsed_seconds"),
            "model_calls": len(records),
        },
        "records": records,
    }
    output = args.output.expanduser().resolve()
    write_json(output, report)
    print(json.dumps(report["native_score_report"], indent=2))
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
