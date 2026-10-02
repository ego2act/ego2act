from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from baselines.common import add_dataset_arguments, load_samples, write_json
from baselines.simple_vqa.method import (
    SCORE_KEYS,
    prompt_sha256,
    score,
    video_to_data_url,
)
from baselines.video import (
    create_openrouter_client,
    standardize_video_for_inference,
)
from baselines.config import get_experiment_config


ROOT = Path(__file__).resolve().parents[2]


def _append_score(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": result["id"],
        "case": result["case"],
        "video": result["video"],
        "task_0_to_3": result["prediction"]["task_0_to_3"],
        "physics_0_to_4": result["prediction"]["physics_0_to_4"],
        "unified_0_to_1": result["prediction"]["unified_0_to_1"],
        "evaluator_fingerprint": result["evaluator_fingerprint"],
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _evaluator_fingerprint(model: str) -> str:
    return hashlib.sha256(json.dumps(
        {"model": model, "prompt_sha256": prompt_sha256()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def evaluate_video(
    sample: dict,
    output_root: Path,
    model: str,
    evaluator_fingerprint: str,
) -> dict:
    started = time.monotonic()
    video_dir = output_root / "videos" / sample["case"] / sample["video"]
    inference_video, video_report = standardize_video_for_inference(
        sample["video_path"], video_dir
    )
    predictions, usage = score(
        create_openrouter_client(),
        video_data_url=video_to_data_url(inference_video),
        goal=sample["prompt"],
        model=model,
    )
    return {
        **sample,
        "scores": [predictions[key] for key in SCORE_KEYS],
        "prediction": predictions,
        "evaluator_fingerprint": evaluator_fingerprint,
        "token_usage": usage,
        "video_standardization": video_report,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the graph-free Simple VQA baseline with separate Task and "
            "Physics rubric calls over current metadata.json data."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "analysis" / "csv" / "metadata.csv",
        help="Authoritative case manifest; defaults to analysis/csv/metadata.csv.",
    )
    parser.add_argument(
        "--id-file",
        type=Path,
        help="JSON list of exact sample IDs to evaluate after dataset discovery.",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--judge-model",
        help="Override the configured OpenRouter judge model.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only records with identical video, goal, model, and prompts.",
    )
    add_dataset_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(ROOT / ".env")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    output = args.output.expanduser().resolve()
    scores_path = output.parent / "scores.jsonl"
    existing_score_ids = set()
    if scores_path.is_file():
        existing_score_ids = {
            json.loads(line)["id"]
            for line in scores_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    model = args.judge_model or get_experiment_config()["baselines"]["model"]
    evaluator_fingerprint = _evaluator_fingerprint(model)
    requested_ids = None
    if args.id_file:
        requested_ids = set(json.loads(args.id_file.expanduser().read_text(encoding="utf-8")))
        args.case = sorted({sample_id.split("::", 1)[0] for sample_id in requested_ids})
    samples = load_samples(args)
    if args.id_file:
        selected_ids = requested_ids
        samples = [sample for sample in samples if sample["id"] in selected_ids]
        if len(samples) != len(selected_ids):
            missing = selected_ids - {sample["id"] for sample in samples}
            raise ValueError(f"ID file contains samples not found in dataset: {sorted(missing)[:5]}")
    manifest_cases = {
        row["case_id"]
        for row in csv.DictReader(args.manifest.expanduser().open(newline=""))
    }
    samples = [sample for sample in samples if sample["case"] in manifest_cases]
    if not samples:
        raise ValueError(f"No videos matched manifest cases in {args.manifest}")
    samples_by_id = {sample["id"]: sample for sample in samples}
    selected_ids = {sample["id"] for sample in samples}
    if args.prepare_only:
        write_json(output, {
            "method": "simple_vqa",
            "label": "Simple VQA",
            "rubric": "Task/Physics",
            "executed": False,
            "calls_per_video": 1,
            "model": model,
            "prompt_sha256": prompt_sha256(),
            "evaluator_fingerprint": evaluator_fingerprint,
            "samples": samples,
        })
        print(f"Prepared {len(samples)} samples: {output}")
        return 0

    existing = {}
    if args.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        existing = {
            item["id"]: item
            for item in previous.get("results", [])
            if "prediction" in item
            and item.get("sample_fingerprint")
            and item.get("evaluator_fingerprint") == evaluator_fingerprint
        }
    selected_existing = {
        sample_id: item
        for sample_id, item in existing.items()
        if sample_id in selected_ids
        and item["sample_fingerprint"] == samples_by_id[sample_id]["sample_fingerprint"]
    }
    pending = [sample for sample in samples if sample["id"] not in selected_existing]
    results = list(selected_existing.values())
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_video,
                sample,
                ROOT / ".tmp" / "simple_vqa_media",
                model,
                evaluator_fingerprint,
            ): sample
            for sample in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simple VQA"):
            sample = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["id"] not in existing_score_ids:
                    _append_score(scores_path, result)
                    existing_score_ids.add(result["id"])
                print(
                    f"{sample['case']}/{sample['video']}: {result['prediction']}",
                    flush=True,
                )
            except Exception as error:
                results.append({
                    **sample,
                    "evaluator_fingerprint": evaluator_fingerprint,
                    "error": f"{type(error).__name__}: {error}",
                })
                print(
                    f"{sample['case']}/{sample['video']}: FAILED "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
    order = {sample["id"]: index for index, sample in enumerate(samples)}
    results.sort(key=lambda item: order[item["id"]])
    successful = [item for item in results if "prediction" in item]
    usage = {
        "prompt_tokens": sum(item["token_usage"]["prompt_tokens"] for item in successful),
        "completion_tokens": sum(item["token_usage"]["completion_tokens"] for item in successful),
        "total_tokens": sum(item["token_usage"]["total_tokens"] for item in successful),
        "cost": sum(item["token_usage"]["cost"] for item in successful),
        "model_calls": sum(item["token_usage"]["model_calls"] for item in successful),
    }
    report = {
        "method": "simple_vqa",
        "label": "Simple VQA",
        "rubric": "Task/Physics",
        "use_graph": False,
        "calls_per_video": 1,
            "aggregation": "sqrt((task / 3) * (physics / 4)); null when physics is N/A",
            "structured_output": "one forced function call with Task/Physics reasoning and evidence",
        "model": model,
        "prompt_sha256": prompt_sha256(),
        "evaluator_fingerprint": evaluator_fingerprint,
        "requested_videos": len(samples),
        "successful_videos": len(successful),
        "failed_videos": len(samples) - len(successful),
        "workers": args.workers,
        "resumed_successes": len(selected_existing),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "token_usage": usage,
        "results": results,
    }
    write_json(output, report)
    print(f"Summary: {output}")
    return 0 if len(successful) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
