from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Any

from dotenv import load_dotenv

from baselines.common import (
    DEFAULT_API_JUDGE_MODEL,
    ROOT,
    add_dataset_arguments,
    load_samples,
    require_commit,
    run,
    tracked_file_matches_head,
    write_json,
)
from baselines.rbench.prompt import PROMPT_VERSION, prompt_sha256


REPOSITORY = Path(__file__).resolve().parent / "external" / "ReVidgen"
COMMIT = "b03df27f0376faa148dcd8cd620a1989a32ca979"
SOURCE = "eval/5_tasks/common_manipulation.py"
UPSTREAM_RUNNER = REPOSITORY / SOURCE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RBench Common Manipulation scoring rule with a controlled "
            "first-person human-manipulation prompt adaptation."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--view", default="first-person")
    parser.add_argument("--model", default=DEFAULT_API_JUDGE_MODEL)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only records with identical video, goal, model, and prompt.",
    )
    add_dataset_arguments(parser)
    return parser


def _fingerprint(model: str, view: str) -> str:
    return hashlib.sha256(json.dumps({
        "upstream_commit": COMMIT,
        "model": model,
        "view": view,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "frame_protocol": "upstream 16-frame sampling then 6-frame 3x2 JPEG grid",
        "score_protocol": "upstream compute_final_score",
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    successful = [row for row in records if "native_score_1_to_5" in row]
    return {
        "definition": "RBench Common Manipulation final score after native post-processing",
        "range": [1.0, 5.0],
        "higher_is_better": True,
        "groups": {
            name: {
                "videos": len(rows),
                "mean": (
                    round(fmean(row["native_score_1_to_5"] for row in rows), 6)
                    if rows else None
                ),
                "median": (
                    round(median(row["native_score_1_to_5"] for row in rows), 6)
                    if rows else None
                ),
            }
            for name, rows in _groups(successful).items()
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    load_dotenv(ROOT / ".env")
    commit = require_commit(REPOSITORY, COMMIT)
    integrity = {SOURCE: tracked_file_matches_head(REPOSITORY, SOURCE)}
    if not integrity[SOURCE]:
        raise RuntimeError(f"Official RBench source file was modified: {SOURCE}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "upstream_input.json"
    raw_output = output_dir / "upstream_output.json"
    usage_output = output_dir / "usage.json"
    evaluator_fingerprint = _fingerprint(args.model, args.view)
    samples = load_samples(args)
    write_json(input_path, [
        {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "video_path": sample["video_path"],
            "sample_fingerprint": sample["sample_fingerprint"],
            "view": args.view,
        }
        for sample in samples
    ])
    manifest = {
        "method": "RBench-Ego2Act Common Manipulation",
        "adaptation": (
            "controlled robot-to-first-person-human prompt wording; upstream "
            "frame grid and native score post-processing retained"
        ),
        "official_commit": commit,
        "source_integrity": integrity,
        "model": args.model,
        "view": args.view,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "evaluator_fingerprint": evaluator_fingerprint,
        "judge_visible_fields": ["goal", "view", "six-frame image grid"],
        "judge_excluded_fields": [
            "id", "case", "category", "generator", "seed", "gt", "description"
        ],
        "samples": samples,
        "upstream_input": str(input_path),
    }
    write_json(output_dir / "manifest.json", manifest)
    if args.prepare_only:
        print(f"Prepared {len(samples)} samples in {input_path}")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("RBench evaluator requires OPENROUTER_API_KEY")
    if not args.resume:
        raw_output.unlink(missing_ok=True)
        usage_output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "baselines.rbench.openrouter_entry",
        "--upstream-script",
        str(UPSTREAM_RUNNER),
        "--input",
        str(input_path),
        "--output",
        str(raw_output),
        "--usage-output",
        str(usage_output),
        "--model",
        args.model,
        "--evaluator-fingerprint",
        evaluator_fingerprint,
        "--workers",
        str(args.workers),
    ]
    if args.resume:
        command.append("--resume")
    run(command, cwd=ROOT)

    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    raw_by_id = {row["id"]: row for row in raw.get("records", [])}
    records = []
    for sample in samples:
        item = raw_by_id.get(sample["id"])
        if item is None:
            records.append({**sample, "error": "missing upstream output"})
        elif item.get("sample_fingerprint") != sample["sample_fingerprint"]:
            records.append({**sample, "error": "stale upstream sample fingerprint"})
        elif item.get("evaluator_fingerprint") != evaluator_fingerprint:
            records.append({**sample, "error": "stale evaluator fingerprint"})
        else:
            records.append({**sample, **item})

    successful = [row for row in records if "native_score_1_to_5" in row]
    report = {
        "method": "RBench-Ego2Act Common Manipulation",
        "execution": "controlled prompt adaptation with upstream frame and score protocol",
        "official_repository": "https://github.com/DAGroup-PKU/ReVidgen",
        "official_commit": commit,
        "source_integrity": integrity,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "evaluator_fingerprint": evaluator_fingerprint,
        "primary_score": {
            "field": "native_score_1_to_5",
            "range": [1.0, 5.0],
            "definition": "RBench native final-score post-processing",
        },
        "requested_videos": len(samples),
        "successful_videos": len(successful),
        "failed_videos": len(samples) - len(successful),
        "native_score_report": native_score_report(records),
        "token_usage": (
            json.loads(usage_output.read_text(encoding="utf-8"))
            if usage_output.is_file() else None
        ),
        "metrics": None,
        "metrics_note": "Task/Physics annotations are joined only in later analysis.",
        "records": records,
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report["native_score_report"], indent=2))
    return 0 if len(successful) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
