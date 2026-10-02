"""Score current Ego2Act videos with VideoScore on a CUDA host.

This is the GPU half of the baseline and is normally executed in Colab. It
writes raw aspect scores; run.py verifies current inputs and builds summaries.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from baselines.video import standardize_video_for_inference
from baselines.common import (
    add_dataset_arguments,
    git_commit,
    load_samples,
    require_commit,
    sha256,
    tracked_file_matches_head,
    write_json,
)
from baselines.videoscore import method


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
EXTERNAL = BASELINE_DIR / "external"
VIDEOSCORE_REPOSITORY = EXTERNAL / "VideoScore"
MANTIS_REPOSITORY = EXTERNAL / "Mantis"
DEFAULT_MODEL = EXTERNAL / "models" / "VideoScore-v1.1"
RUNTIME_LOCK = BASELINE_DIR / "runtime-lock.txt"
MODEL_MARKER = ".videoscore_revision.json"

VIDEOSCORE_COMMIT = "194736f913018188105a6e0c2b0667eff5bd3be5"
MANTIS_COMMIT = "62b7438199fb86aca2b2ab1340cc1d646d44be58"
VIDEOSCORE_SOURCE_FILES = (
    "benchmark/eval_videoscore.py",
    "benchmark/utils_tools.py",
    "examples/run_videoscore.py",
)
MANTIS_SOURCE_FILES = ("mantis/models/idefics2/modeling_idefics2.py",)
LFS_POINTER_PREFIX = b"version https://git-lfs"


def assert_real_video_bytes(paths: list[Path]) -> None:
    stubs = []
    for path in paths:
        with path.open("rb") as handle:
            if handle.read(23) == LFS_POINTER_PREFIX:
                stubs.append(path)
    if stubs:
        raise RuntimeError(
            f"{len(stubs)} of {len(paths)} videos are Git LFS pointer stubs, for example "
            f"{stubs[0]}. Run 'git lfs install && git lfs pull' before scoring."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VideoScore over selected videos in the current data/."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        help="Allow a custom model without the pinned revision marker.",
    )
    parser.add_argument("--max-frames", type=int, default=method.DEFAULT_MAX_FRAMES)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only records with identical input and evaluator fingerprints.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the selected manifest without loading the model.",
    )
    add_dataset_arguments(parser)
    return parser


def _assigned_string(path: Path, variable: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"Could not find string assignment {variable} in {path}")


def verify_upstream_contract() -> dict[str, Any]:
    videoscore_commit = require_commit(VIDEOSCORE_REPOSITORY, VIDEOSCORE_COMMIT)
    mantis_commit = require_commit(MANTIS_REPOSITORY, MANTIS_COMMIT)
    source_integrity = {
        "VideoScore": {
            path: tracked_file_matches_head(VIDEOSCORE_REPOSITORY, path)
            for path in VIDEOSCORE_SOURCE_FILES
        },
        "Mantis": {
            path: tracked_file_matches_head(MANTIS_REPOSITORY, path)
            for path in MANTIS_SOURCE_FILES
        },
    }
    failures = [
        f"{repository}/{path}"
        for repository, files in source_integrity.items()
        for path, matches in files.items()
        if not matches
    ]
    if failures:
        raise RuntimeError(f"Pinned VideoScore source differs from Git: {failures}")

    upstream_prompt = _assigned_string(
        VIDEOSCORE_REPOSITORY / "benchmark" / "utils_tools.py",
        "REGRESSION_QUERY_PROMPT",
    )
    if upstream_prompt != method.REGRESSION_QUERY_PROMPT:
        raise RuntimeError("Local VideoScore prompt differs from the pinned upstream prompt")
    return {
        "videoscore_commit": videoscore_commit,
        "mantis_commit": mantis_commit,
        "source_integrity": source_integrity,
    }


def model_provenance(model_name: str, allow_unverified: bool) -> dict[str, Any]:
    path = Path(model_name).expanduser()
    expected = {
        "repository": method.MODEL_REPOSITORY,
        "revision": method.MODEL_REVISION,
    }
    if path.is_dir():
        marker_path = path / MODEL_MARKER
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if any(marker.get(key) != value for key, value in expected.items()):
                raise RuntimeError(
                    f"VideoScore model revision mismatch: expected {expected}, found {marker}"
                )
            return {**expected, "verified": True, "path": str(path.resolve())}
        if not allow_unverified:
            raise RuntimeError(
                f"VideoScore model has no pinned-revision marker: {marker_path}. "
                f"Run: ego2act baseline videoscore --setup-only, or pass --allow-unverified-model."
            )
        return {
            "repository": model_name,
            "revision": "unverified",
            "verified": False,
            "path": str(path.resolve()),
        }

    if model_name == method.MODEL_REPOSITORY:
        return {**expected, "verified": True, "path": model_name}
    if not allow_unverified:
        raise RuntimeError(
            f"Custom VideoScore model is unverified: {model_name}. "
            "Use ego2act baseline videoscore --setup-only, or explicitly pass --allow-unverified-model."
        )
    return {
        "repository": model_name,
        "revision": "unverified",
        "verified": False,
        "path": model_name,
    }


def evaluate_video(
    model: Any,
    processor: Any,
    sample: dict,
    proxy_root: Path,
    max_frames: int,
) -> dict[str, Any]:
    started = time.monotonic()
    proxy_path, standardization = standardize_video_for_inference(
        Path(sample["video_path"]), proxy_root / sample["case"] / sample["video"]
    )
    prediction, frames_used = method.score_video(
        model, processor, sample["prompt"], proxy_path, max_frames
    )
    return {
        "id": sample["id"],
        "case": sample["case"],
        "video": sample["video"],
        "category": sample["category"],
        "generator": sample["generator"],
        "seed": sample["seed"],
        "goal": sample["prompt"],
        "video_sha256": sample["video_sha256"],
        "prompt_sha256": sample["prompt_sha256"],
        "sample_fingerprint": sample["sample_fingerprint"],
        "scores": [prediction[aspect] for aspect in method.ASPECTS],
        "prediction": prediction,
        "frames_used": frames_used,
        "video_standardization": standardization,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _evaluator_fingerprint(
    model_info: dict[str, Any],
    max_frames: int,
    device: str | None,
    dtype: str | None,
) -> str:
    import hashlib

    return hashlib.sha256(json.dumps({
        "model": model_info["repository"],
        "revision": model_info["revision"],
        "prompt_sha256": method.prompt_sha256(),
        "max_frames": max_frames,
        "device": device,
        "dtype": dtype,
        "runtime_lock_sha256": sha256(RUNTIME_LOCK),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _base_payload(
    *,
    upstream: dict[str, Any],
    model_info: dict[str, Any],
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    evaluator_fingerprint: str,
) -> dict[str, Any]:
    return {
        "method": "videoscore",
        "label": "VideoScore",
        "execution": "upstream scoring logic and prompt with current Ego2Act inputs",
        "upstream_repository": "https://github.com/TIGER-AI-Lab/VideoScore",
        "upstream_commit": upstream["videoscore_commit"],
        "mantis_repository": "https://github.com/TIGER-AI-Lab/Mantis",
        "mantis_commit": upstream["mantis_commit"],
        "source_integrity": upstream["source_integrity"],
        "model": model_info["repository"],
        "model_revision": model_info["revision"],
        "model_verified": model_info["verified"],
        "model_path": model_info["path"],
        "evaluator_fingerprint": evaluator_fingerprint,
        "aspects": list(method.ASPECTS),
        "native_scale": [1.0, 4.0],
        "max_frames": args.max_frames,
        "prompt_sha256": method.prompt_sha256(),
        "runtime_lock_sha256": sha256(RUNTIME_LOCK),
        "ego2act_commit": git_commit(ROOT),
        "requested_videos": len(samples),
        "selected_samples": samples,
    }


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    proxy_root = (args.proxy_root or output.parent / "videos").expanduser().resolve()

    upstream = verify_upstream_contract()
    model_info = model_provenance(args.model, args.allow_unverified_model)

    samples = load_samples(args)
    assert_real_video_bytes([Path(sample["video_path"]) for sample in samples])
    resolved_device = method.resolve_device(args.device)
    resolved_dtype = method.resolve_dtype(args.dtype, resolved_device)
    evaluator_fingerprint = _evaluator_fingerprint(
        model_info,
        args.max_frames,
        str(resolved_device),
        str(resolved_dtype).replace("torch.", ""),
    )
    base_payload = _base_payload(
        upstream=upstream,
        model_info=model_info,
        samples=samples,
        args=args,
        evaluator_fingerprint=evaluator_fingerprint,
    )
    if args.prepare_only:
        write_json(output, {**base_payload, "prepared_only": True, "results": []})
        print(f"Prepared {len(samples)} videos in {output}")
        return 0

    existing: dict[str, dict] = {}
    previous: dict[str, Any] = {}
    if args.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        existing = {
            item["id"]: item
            for item in previous.get("results", [])
            if "scores" in item
            and item.get("evaluator_fingerprint") == evaluator_fingerprint
        }
    reused = {
        sample["id"]: existing[sample["id"]]
        for sample in samples
        if sample["id"] in existing
        and existing[sample["id"]].get("sample_fingerprint")
        == sample["sample_fingerprint"]
    }
    pending = [
        sample for sample in samples if sample["id"] not in reused
    ]

    if not pending:
        results = [reused[sample["id"]] for sample in samples]
        write_json(output, {
            **base_payload,
            "mantis_version": previous.get("mantis_version"),
            "runtime": previous.get("runtime"),
            "transformers": previous.get("transformers"),
            "successful_videos": len(results),
            "failed_videos": 0,
            "resumed_successes": len(results),
            "elapsed_seconds": 0.0,
            "results": results,
        })
        print(f"Scores: {output} ({len(results)} reused)")
        return 0

    import transformers

    model, processor, runtime_description = method.load_model(
        model_info["path"],
        model_info["revision"] if model_info["path"] == method.MODEL_REPOSITORY else None,
        resolved_device,
        resolved_dtype,
    )
    print(json.dumps(runtime_description, indent=2), flush=True)

    results_by_id = dict(reused)
    started = time.monotonic()
    runtime_metadata = {
        **base_payload,
        "mantis_version": version("mantis-vl"),
        "runtime": runtime_description,
        "transformers": transformers.__version__,
    }
    for sample in pending:
        try:
            result = evaluate_video(
                model, processor, sample, proxy_root, args.max_frames
            )
            result["evaluator_fingerprint"] = evaluator_fingerprint
            results_by_id[sample["id"]] = result
            print(f"{sample['id']}: {result['scores']}", flush=True)
        except Exception as error:
            results_by_id[sample["id"]] = {
                "id": sample["id"],
                "case": sample["case"],
                "video": sample["video"],
                "sample_fingerprint": sample["sample_fingerprint"],
                "evaluator_fingerprint": evaluator_fingerprint,
                "error": f"{type(error).__name__}: {error}",
            }
            print(
                f"{sample['id']}: FAILED {type(error).__name__}: {error}",
                flush=True,
            )
        ordered = [results_by_id[sample["id"]] for sample in samples if sample["id"] in results_by_id]
        successful = [item for item in ordered if "scores" in item]
        write_json(output, {
            **runtime_metadata,
            "successful_videos": len(successful),
            "failed_videos": len(samples) - len(successful),
            "resumed_successes": len(reused),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "results": ordered,
        })

    results = [results_by_id[sample["id"]] for sample in samples]
    successful = [item for item in results if "scores" in item]

    write_json(output, {
        **runtime_metadata,
        "successful_videos": len(successful),
        "failed_videos": len(samples) - len(successful),
        "resumed_successes": len(reused),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "results": results,
    })
    print(f"Scores: {output}")
    return 0 if len(successful) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
