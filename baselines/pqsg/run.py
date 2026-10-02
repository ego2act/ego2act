from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

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


REPOSITORY = Path(__file__).resolve().parent / "external" / "pqsg"
COMMIT = "03afdb0e6f9ed68d363a68eea5ca4fd90d53848e"
SOURCE_FILES = (
    "scripts/run.py",
    "pqsg/qg.py",
    "pqsg/qa.py",
    "pqsg/score.py",
    "prompts/qg_prompt.txt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned official PQSG CLI on Ego2Act videos."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_API_JUDGE_MODEL)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only records with identical video and goal fingerprints.",
    )
    add_dataset_arguments(parser)
    return parser


def _native_summary(records: list[dict]) -> dict:
    successful = [record for record in records if "native_score_0_to_1" in record]
    groups: dict[str, list[dict]] = {
        "all": successful,
        "correct": [record for record in successful if record["category"] == "correct"],
        "wrong": [record for record in successful if record["category"] == "wrong"],
        "ai": [record for record in successful if record["category"] == "ai"],
    }
    for generator in sorted({record.get("generator") for record in successful} - {None}):
        groups[f"ai/{generator}"] = [
            record for record in successful if record.get("generator") == generator
        ]
    return {
        name: {
            "videos": len(rows),
            "mean": (
                round(fmean(row["native_score_0_to_1"] for row in rows), 6)
                if rows else None
            ),
        }
        for name, rows in groups.items()
    }


def _prepare_resume(path: Path, samples: list[dict], enabled: bool) -> None:
    if not enabled:
        path.unlink(missing_ok=True)
        return
    if not path.is_file():
        return
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        path.unlink()
        return
    by_id = {
        str(item.get("id")): item
        for item in loaded
        if isinstance(item, dict) and item.get("answers")
    }
    safe = []
    for sample in samples:
        item = by_id.get(sample["id"], {})
        safe.append(
            item
            if item.get("sample_fingerprint") == sample["sample_fingerprint"]
            else {}
        )
    write_json(path, safe)


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(ROOT / ".env")
    commit = require_commit(REPOSITORY, COMMIT)
    integrity = {
        path: tracked_file_matches_head(REPOSITORY, path)
        for path in SOURCE_FILES
    }
    if not all(integrity.values()):
        changed = [path for path, matches in integrity.items() if not matches]
        raise RuntimeError(f"Official PQSG source files were modified: {changed}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args)
    native_input = output_dir / "upstream_input.json"
    native_output = output_dir / "upstream_output.json"
    usage_output = output_dir / "usage.json"
    write_json(native_input, [
        {
            "id": sample["id"],
            "model": sample["category"],
            "prompt": sample["prompt"],
            "video_path": sample["video_path"],
            "video_sha256": sample["video_sha256"],
            "prompt_sha256": sample["prompt_sha256"],
            "sample_fingerprint": sample["sample_fingerprint"],
        }
        for sample in samples
    ])
    write_json(output_dir / "manifest.json", {
        "method": "PQSG",
        "official_commit": commit,
        "source_integrity": integrity,
        "samples": samples,
        "upstream_input": str(native_input),
        "provider": "openrouter",
        "model": args.model,
        "allowed_delta": "model client/transport only",
        "provider_media_serialization": "shared 480p/24fps inference proxy at OpenRouter client boundary",
    })
    if args.prepare_only:
        print(f"Prepared {native_input}")
        return 0

    _prepare_resume(native_output, samples, args.resume)

    command = [
        sys.executable,
        "-m",
        "baselines.pqsg.openrouter_entry",
        "--upstream",
        str(REPOSITORY),
        "--input",
        str(native_input),
        "--output",
        str(native_output),
        "--model",
        args.model,
        "--media-cache",
        str(output_dir / "provider_media"),
        "--usage-output",
        str(usage_output),
    ]
    run(command, cwd=ROOT)

    native = json.loads(native_output.read_text(encoding="utf-8"))
    by_id = {sample["id"]: sample for sample in samples}
    records = []
    for item in native:
        sample = by_id[str(item["id"])]
        record = {**sample, "upstream": item}
        if "score" in item:
            record["native_score_0_to_1"] = float(item["score"])
        if "error" in item:
            record["error"] = item["error"]
        records.append(record)

    report = {
        "method": "PQSG",
        "execution": "official CLI with provider-only OpenRouter shim",
        "official_repository": "https://github.com/atinpothiraj/pqsg",
        "official_commit": commit,
        "source_integrity": integrity,
        "provider": "openrouter",
        "model": args.model,
        "qg": args.model,
        "qa": "gemini-compatible video path",
        "qa_model": args.model,
        "allowed_delta": "model client/transport only",
        "provider_media_serialization": "shared 480p/24fps inference proxy at OpenRouter client boundary",
        "primary_score": {
            "field": "native_score_0_to_1",
            "range": [0, 1],
            "definition": "official dependency-weighted PQSG final score",
        },
        "native_score_report": _native_summary(records),
        "token_usage": (
            json.loads(usage_output.read_text(encoding="utf-8"))
            if usage_output.is_file()
            else None
        ),
        "metrics": None,
        "metrics_note": (
            "Task/Physics human scores are not stored in metadata.json. Compare "
            "native_score_0_to_1 with the normalized Ego2Act unified score "
            "when those annotations are joined during analysis."
        ),
        "records": records,
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report["native_score_report"], indent=2))
    return 0 if all("native_score_0_to_1" in row for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
