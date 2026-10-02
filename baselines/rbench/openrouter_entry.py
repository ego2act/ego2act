from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any

from openai import OpenAI

from baselines.common import UsageRecorder, write_json
from baselines.rbench.prompt import create_prompt


def install_unused_dependency_stubs() -> None:
    """The upstream API path imports local-model packages that it never uses."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        module = types.ModuleType("transformers")
        module.AutoModelForImageTextToText = object
        module.AutoProcessor = object
        sys.modules["transformers"] = module
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")
    try:
        import torchvision  # noqa: F401
    except (ImportError, RuntimeError):
        torchvision = types.ModuleType("torchvision")
        torchvision_io = types.ModuleType("torchvision.io")
        torchvision_io.write_video = None
        torchvision.io = torchvision_io
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.io"] = torchvision_io


def load_upstream(path: Path):
    install_unused_dependency_stubs()
    spec = importlib.util.spec_from_file_location("rbench_common_manipulation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RBench evaluator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usage-output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluator-fingerprint", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("response contained no JSON object")
    value = json.loads(text[start:end])
    if not isinstance(value, dict):
        raise ValueError("response JSON was not an object")
    return value


def _validate_details(details: dict[str, Any]) -> None:
    fields = (
        "action_execution",
        "task_completion",
        "object_consistency",
        "hand_consistency",
        "physical_plausibility",
        "total",
    )
    for field in fields:
        item = details.get(field)
        if not isinstance(item, dict) or not isinstance(item.get("score"), (int, float)):
            raise ValueError(f"missing numeric {field}.score")
        score = float(item["score"])
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"{field}.score outside 1-5")


def _grid_jpeg(upstream: Any, video_path: str) -> bytes:
    import cv2
    import numpy as np

    processor = upstream.Video_preprocess()
    frames = processor.extract_frames(video_path, num_frames=16)
    if len(frames) < 6:
        raise ValueError(f"RBench requires six decodable frames, got {len(frames)}")
    indices = np.linspace(0, len(frames) - 1, 6, dtype=int)
    grid = processor.merge_grid([frames[index] for index in indices])
    ok, encoded = cv2.imencode(".jpg", grid)
    if not ok:
        raise ValueError("OpenCV could not encode the RBench frame grid")
    return encoded.tobytes()


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    samples = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("RBench input must be a non-empty JSON list")
    upstream = load_upstream(args.upstream_script.resolve())

    previous_records: dict[str, dict[str, Any]] = {}
    previous_usage: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_records = {
            item["id"]: item
            for item in previous.get("records", [])
            if item.get("evaluator_fingerprint") == args.evaluator_fingerprint
            and "native_score_1_to_5" in item
            and "error" not in item
        }
    if args.resume and args.usage_output.is_file():
        previous_usage = json.loads(args.usage_output.read_text(encoding="utf-8")).get(
            "generations", []
        )

    reusable = {
        sample["id"]: previous_records[sample["id"]]
        for sample in samples
        if sample["id"] in previous_records
        and previous_records[sample["id"]].get("sample_fingerprint")
        == sample["sample_fingerprint"]
    }
    pending = [sample for sample in samples if sample["id"] not in reusable]
    results = dict(reusable)
    usage = UsageRecorder()
    usage.items = list(previous_usage)
    usage_lock = threading.Lock()
    thread_state = threading.local()

    def client() -> OpenAI:
        if not hasattr(thread_state, "client"):
            thread_state.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                timeout=180,
                max_retries=2,
            )
        return thread_state.client

    def evaluate(sample: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            jpeg = _grid_jpeg(upstream, sample["video_path"])
            encoded = base64.b64encode(jpeg).decode("ascii")
            response = client().chat.completions.create(
                model=args.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": create_prompt(sample["prompt"], sample["view"]),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }],
                temperature=0.0,
                max_tokens=1500,
                seed=2026,
                extra_body={"provider": {"require_parameters": True}},
            )
            with usage_lock:
                usage.add(response)
            raw = (response.choices[0].message.content or "").strip()
            details = _extract_json(raw)
            _validate_details(details)
            score = float(upstream.compute_final_score(details))
            if not 1.0 <= score <= 5.0:
                raise ValueError(f"native score outside 1-5: {score}")
            return {
                "id": sample["id"],
                "sample_fingerprint": sample["sample_fingerprint"],
                "evaluator_fingerprint": args.evaluator_fingerprint,
                "native_score_1_to_5": score,
                "details": details,
                "raw_response": raw,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as error:
            return {
                "id": sample["id"],
                "sample_fingerprint": sample["sample_fingerprint"],
                "evaluator_fingerprint": args.evaluator_fingerprint,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }

    def checkpoint() -> None:
        ordered = [results[sample["id"]] for sample in samples if sample["id"] in results]
        write_json(args.output, {
            "evaluator_fingerprint": args.evaluator_fingerprint,
            "records": ordered,
        })
        with usage_lock:
            usage_payload = usage.summary()
        write_json(args.usage_output, usage_payload)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, sample): sample for sample in pending}
        completed = len(reusable)
        for future in as_completed(futures):
            sample = futures[future]
            record = future.result()
            results[sample["id"]] = record
            completed += 1
            label = record.get("native_score_1_to_5", record.get("error"))
            print(f"[{completed}/{len(samples)}] {sample['id']}: {label}", flush=True)
            checkpoint()
    checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
