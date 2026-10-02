"""Run the official WorldModelBench protocol over an adapted video manifest.

This file runs inside the pinned VILA environment. It imports the official
EvaluationConfig and WorldModelEvaluator, so the prompts, video serialization,
judge loading, and generation call stay upstream-defined. The only adaptation
is iterating an Ego2Act manifest and checkpointing after each video.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-evaluation", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cot", action="store_true")
    return parser


def load_upstream(path: Path):
    spec = importlib.util.spec_from_file_location("worldmodelbench_official_evaluation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official evaluator at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_instruction(response: str) -> tuple[float, str | None]:
    """Mirror evaluation.py's official instruction parser exactly."""
    try:
        return float(response.split(":")[-1].strip(" .")), None
    except ValueError:
        return 0.0, "official parser defaulted an unparseable response to 0"


def parse_pass(response: str) -> bool:
    """Mirror evaluation.py: a physical/common-sense check passes iff it contains 'no'."""
    return "no" in response.lower()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate_sample(module: Any, evaluator: Any, config: Any, sample: dict[str, Any], cot: bool) -> dict[str, Any]:
    import llava

    started = time.monotonic()
    video = llava.Video(sample["video_path"])
    record: dict[str, Any] = {
        "id": sample["id"],
        "sample_fingerprint": sample["sample_fingerprint"],
    }
    for evaluation_type in module.EvaluationType:
        prompt_template = config.PROMPT_TEMPLATES[evaluation_type.value]
        questions = config.QUESTION_POOL[evaluation_type.value]
        if questions is None:
            prompt = prompt_template.format(instruction=sample["instruction"])
            response = evaluator.evaluate_video(video, prompt, cot)
            score, parse_warning = parse_instruction(response)
            record["instruction"] = {
                "prompt": prompt,
                "response": response,
                "score": score,
                "parse_warning": parse_warning,
            }
            continue

        items = []
        for question in questions:
            prompt = prompt_template.format(**{evaluation_type.value: question.lower()})
            response = evaluator.evaluate_video(video, prompt, cot)
            items.append({
                "question": question,
                "prompt": prompt,
                "response": response,
                "pass": parse_pass(response),
            })
        record[evaluation_type.value] = items

    physical = sum(item["pass"] for item in record["physical_laws"])
    common = sum(item["pass"] for item in record["common_sense"])
    instruction = float(record["instruction"]["score"])
    record["native_scores"] = {
        "instruction_0_to_3": instruction,
        "physical_laws_0_to_5": physical,
        "common_sense_0_to_2": common,
        "official_total_0_to_10": instruction + physical + common,
    }
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return record


def main() -> int:
    args = build_parser().parse_args()
    module = load_upstream(args.upstream_evaluation.resolve())
    samples = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("WorldModelBench input must be a non-empty JSON list")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "judge": str(args.judge.resolve()),
        "cot": args.cot,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "records": [],
    }
    if args.resume and args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if bool(payload.get("cot")) != args.cot:
            raise RuntimeError("Cannot resume with a different --cot setting")
        if Path(payload.get("judge", "")).resolve() != args.judge.resolve():
            raise RuntimeError("Cannot resume with a different judge checkpoint")
    payload["schema_version"] = 2

    selected = {sample["id"]: sample["sample_fingerprint"] for sample in samples}
    payload["records"] = [
        record
        for record in payload.get("records", [])
        if record.get("id") in selected
        and record.get("sample_fingerprint") == selected[record["id"]]
    ]

    completed = {
        record["id"]
        for record in payload.get("records", [])
        if "native_scores" in record
        and "error" not in record
        and record.get("sample_fingerprint") == selected.get(record["id"])
    }
    evaluator = module.WorldModelEvaluator(
        str(args.judge.resolve()), "", module.EvaluationConfig()
    )
    config = evaluator.config

    for index, sample in enumerate(samples, start=1):
        if sample["id"] in completed:
            print(f"[{index}/{len(samples)}] {sample['id']}: resumed", flush=True)
            continue
        try:
            record = evaluate_sample(module, evaluator, config, sample, args.cot)
            print(
                f"[{index}/{len(samples)}] {sample['id']}: "
                f"{record['native_scores']['official_total_0_to_10']}/10",
                flush=True,
            )
        except Exception as error:
            record = {
                "id": sample["id"],
                "sample_fingerprint": sample["sample_fingerprint"],
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"[{index}/{len(samples)}] {sample['id']}: FAILED {record['error']}", flush=True)

        payload["records"] = [
            item for item in payload.get("records", []) if item["id"] != sample["id"]
        ] + [record]
        payload["records"].sort(key=lambda item: item["id"])
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output, payload)

    # Per-sample failures are data, not a launcher failure. The outer adapter
    # builds a partial report and returns non-zero after preserving it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
