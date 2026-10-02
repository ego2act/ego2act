from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from baselines.common import (
    ROOT,
    add_dataset_arguments,
    load_samples,
    require_commit,
    run,
    sha256,
    tracked_file_matches_head,
    write_json,
)
from baselines.worldmodelbench.protocol import (
    enrich_record,
    native_score_report,
    render_report,
    rubric_proxy_report,
    rubric_proxy_scores,
)


BASELINE_DIR = Path(__file__).resolve().parent
EXTERNAL = BASELINE_DIR / "external"
REPOSITORY = EXTERNAL / "WorldModelBench"
VILA_REPOSITORY = EXTERNAL / "VILA"
UPSTREAM_EVALUATION = REPOSITORY / "evaluation.py"
DEFAULT_PYTHON = EXTERNAL / ".venv" / "bin" / "python"
DEFAULT_JUDGE = EXTERNAL / "models" / "vila-ewm-qwen2-1.5b"
RUNTIME_LOCK = BASELINE_DIR / "runtime-lock.txt"
JUDGE_MARKER = ".worldmodelbench_revision.json"

WORLDMODELBENCH_COMMIT = "00b7aa17a05f9fd1ab5c8f66bcf476d04c9c33bf"
VILA_COMMIT = "361c9317a2787ebf1814e115de6a4cf804ffaf51"
JUDGE_REPOSITORY = "Efficient-Large-Model/vila-ewm-qwen2-1.5b"
JUDGE_REVISION = "7a269c562f83e902f5ff1ef57015bd72f25f461a"
SOURCE_FILES = ("evaluation.py",)
VILA_SOURCE_FILES = ("llava/entry.py", "llava/media.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned official WorldModelBench VILA judge on Ego2Act "
            "videos and retain its native scores plus fixed Task/Physics proxies."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Rebuild report.json/report.md from an existing upstream_output.json.",
    )
    parser.add_argument(
        "--allow-unverified-judge",
        action="store_true",
        help="Allow a checkpoint without the setup script's pinned-revision marker.",
    )
    add_dataset_arguments(parser)
    return parser


def _source_integrity(repository: Path, paths: tuple[str, ...]) -> dict[str, bool]:
    return {path: tracked_file_matches_head(repository, path) for path in paths}


def _tracked_tree_clean(repository: Path) -> bool:
    return subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--"],
        check=False,
    ).returncode == 0


def _judge_provenance(judge: Path, allow_unverified: bool) -> dict[str, str | bool]:
    if not judge.is_dir():
        raise FileNotFoundError(
            f"WorldModelBench judge is missing: {judge}. Run: ego2act baseline worldmodelbench --setup-only"
        )
    marker_path = judge / JUDGE_MARKER
    if not marker_path.is_file():
        if not allow_unverified:
            raise RuntimeError(
                f"Judge has no pinned-revision marker: {marker_path}. "
                "Use ego2act baseline worldmodelbench --setup-only, or explicitly pass --allow-unverified-judge."
            )
        return {
            "repository": JUDGE_REPOSITORY,
            "revision": "unverified",
            "verified": False,
            "path": str(judge),
        }
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {"repository": JUDGE_REPOSITORY, "revision": JUDGE_REVISION}
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Judge revision marker mismatch: expected {expected}, found {marker}")
    return {**expected, "verified": True, "path": str(judge)}


def _input_sample(sample: dict) -> dict[str, str]:
    """Label-free adapter input; only the first three fields reach the model."""
    return {
        "id": sample["id"],
        "instruction": sample["prompt"],
        "video_path": sample["video_path"],
        "video_sha256": sample["video_sha256"],
        "prompt_sha256": sample["prompt_sha256"],
        "sample_fingerprint": sample["sample_fingerprint"],
    }


def _absolute_without_resolving(path: Path) -> Path:
    """Make a CLI path absolute without dereferencing a virtualenv Python symlink."""
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (Path.cwd() / expanded).absolute()


def _build_records(samples: list[dict], raw: dict) -> list[dict]:
    raw_by_id = {item["id"]: item for item in raw.get("records", [])}
    expected = {sample["id"] for sample in samples}
    extra = set(raw_by_id) - expected
    if extra:
        raise RuntimeError(f"Upstream output contains unselected sample IDs: {sorted(extra)}")

    records = []
    for sample in samples:
        upstream = raw_by_id.get(sample["id"])
        if upstream is None:
            records.append({**sample, "error": "missing upstream output"})
        elif upstream.get("sample_fingerprint") != sample["sample_fingerprint"]:
            records.append({**sample, "error": "stale upstream output fingerprint"})
        elif "native_scores" not in upstream:
            records.append({**sample, "upstream": upstream, "error": upstream.get("error", "incomplete output")})
        else:
            enriched = enrich_record(upstream)
            if enriched["native_scores"] != upstream["native_scores"]:
                raise RuntimeError(f"Native score reconstruction mismatch for {sample['id']}")
            records.append({
                **sample,
                **enriched,
                "rubric_proxy_scores": rubric_proxy_scores(enriched),
            })
    return records


def main() -> int:
    args = build_parser().parse_args()
    official_commit = require_commit(REPOSITORY, WORLDMODELBENCH_COMMIT)
    vila_commit = require_commit(VILA_REPOSITORY, VILA_COMMIT)
    source_integrity = {
        "WorldModelBench": _source_integrity(REPOSITORY, SOURCE_FILES),
        "VILA": _source_integrity(VILA_REPOSITORY, VILA_SOURCE_FILES),
    }
    tracked_trees_clean = {
        "WorldModelBench": _tracked_tree_clean(REPOSITORY),
        "VILA": _tracked_tree_clean(VILA_REPOSITORY),
    }
    changed = [
        f"{repository}/{path}"
        for repository, results in source_integrity.items()
        for path, matches in results.items()
        if not matches
    ]
    if changed:
        raise RuntimeError(f"Pinned upstream source files were modified: {changed}")
    if not all(tracked_trees_clean.values()):
        raise RuntimeError(f"Pinned upstream checkout has tracked modifications: {tracked_trees_clean}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_input = output_dir / "upstream_input.json"
    upstream_output = output_dir / "upstream_output.json"
    manifest_path = output_dir / "manifest.json"
    samples = load_samples(args)
    write_json(upstream_input, [_input_sample(sample) for sample in samples])
    write_json(manifest_path, {
        "method": "WorldModelBench",
        "adaptation": "controlled dataset/manifest adaptation; official prompts, judge, video loader, parser, and score retained",
        "official_repository": "https://github.com/WorldModelBench-Team/WorldModelBench",
        "official_commit": official_commit,
        "vila_repository": "https://github.com/NVlabs/VILA",
        "vila_commit": vila_commit,
        "runtime_lock_sha256": sha256(RUNTIME_LOCK),
        "source_integrity": source_integrity,
        "tracked_trees_clean": tracked_trees_clean,
        "judge_repository": JUDGE_REPOSITORY,
        "judge_revision": JUDGE_REVISION,
        "cot": args.cot,
        "judge_visible_fields": ["instruction", "video_path"],
        "adapter_only_fields": [
            "id", "video_sha256", "prompt_sha256", "sample_fingerprint"
        ],
        "judge_excluded_fields": ["objects", "description", "category", "gt", "start_frame"],
        "samples": samples,
        "upstream_input": str(upstream_input),
    })
    if args.prepare_only:
        print(f"Prepared {len(samples)} videos in {upstream_input}")
        return 0

    judge = args.judge.expanduser().resolve()
    judge_provenance = _judge_provenance(judge, args.allow_unverified_judge)
    python = _absolute_without_resolving(args.python)
    if not args.analyze_only:
        if not python.is_file():
            raise FileNotFoundError(
                f"Pinned VILA Python is missing: {python}. Run: ego2act baseline worldmodelbench --setup-only"
            )
        if not args.resume:
            upstream_output.unlink(missing_ok=True)
        command = [
            str(python),
            str(BASELINE_DIR / "upstream_entry.py"),
            "--upstream-evaluation",
            str(UPSTREAM_EVALUATION),
            "--input",
            str(upstream_input),
            "--output",
            str(upstream_output),
            "--judge",
            str(judge),
        ]
        if args.resume:
            command.append("--resume")
        if args.cot:
            command.append("--cot")
        run(command, cwd=ROOT)

    if not upstream_output.is_file():
        raise FileNotFoundError(f"Missing upstream result: {upstream_output}")
    raw = json.loads(upstream_output.read_text(encoding="utf-8"))
    if bool(raw.get("cot")) != args.cot:
        raise RuntimeError("Upstream output --cot setting does not match this report run")
    records = _build_records(samples, raw)
    successful = [record for record in records if "native_scores" in record]
    report = {
        "method": "WorldModelBench",
        "execution": "official VILA judge and protocol with a resumable Ego2Act manifest adapter",
        "adaptation_category": "controlled adaptation",
        "official_repository": "https://github.com/WorldModelBench-Team/WorldModelBench",
        "official_commit": official_commit,
        "vila_repository": "https://github.com/NVlabs/VILA",
        "vila_commit": vila_commit,
        "runtime_lock_sha256": sha256(RUNTIME_LOCK),
        "source_integrity": source_integrity,
        "tracked_trees_clean": tracked_trees_clean,
        "judge": judge_provenance,
        "cot": args.cot,
        "requested_videos": len(samples),
        "successful_videos": len(successful),
        "failed_videos": len(samples) - len(successful),
        "failure_rate": round((len(samples) - len(successful)) / len(samples), 6),
        "efficiency": {
            "model_calls": len(successful) * 8,
            "summed_per_video_elapsed_seconds": round(
                sum(float(record.get("elapsed_seconds", 0)) for record in successful), 3
            ),
        },
        "native_score_report": native_score_report(records),
        "primary_score": {
            "field": "native_scores.official_total_0_to_10",
            "range": [0, 10],
            "definition": "official WorldModelBench total",
        },
        "rubric_alignment": {
            "task": "instruction_0_to_3 (native)",
            "physics": (
                "4 * (physical_laws_0_to_5 + temporal_consistency_0_to_1) / 6"
            ),
            "unified": "0.5 * (task / 3 + physics / 4)",
            "status": "fixed diagnostic proxy; not an official WorldModelBench score",
        },
        "rubric_proxy_report": rubric_proxy_report(records),
        "metrics": None,
        "metrics_note": (
            "Task/Physics human scores are not stored in metadata.json. Join them "
            "during analysis before computing agreement."
        ),
        "artifacts": {
            "manifest": str(manifest_path),
            "upstream_input": str(upstream_input),
            "upstream_output": str(upstream_output),
        },
        "records": records,
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    (output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({
        "native_score_report": report["native_score_report"],
        "rubric_alignment": report["rubric_alignment"],
        "failed_videos": report["failed_videos"],
    }, indent=2))
    print(f"Report: {report_path}")
    return 0 if len(successful) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
