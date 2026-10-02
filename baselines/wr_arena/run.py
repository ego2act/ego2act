from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path
from statistics import fmean, median

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from baselines.common import (
    DEFAULT_API_JUDGE_MODEL,
    ROOT,
    add_dataset_arguments,
    load_samples,
    require_commit,
    tracked_file_matches_head,
    UsageRecorder,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parent / "external" / "WR-Arena"
COMMIT = "7b3e4af4d9346bc58fe2c6c75626f15c3a464fca"
SOURCE = "action_simulation_fidelity_scripts/action_simulation_fidelity_eval.py"
UPSTREAM_EVALUATOR = REPOSITORY / SOURCE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run WR-Arena's official per-video Action Simulation Fidelity "
            "function on Ego2Act videos."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_API_JUDGE_MODEL)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only records with identical video, goal, model, and protocol.",
    )
    add_dataset_arguments(parser)
    return parser


def load_upstream_module():
    try:
        import mmengine  # noqa: F401
    except ImportError:
        mmengine = types.ModuleType("mmengine")
        mmengine.load = None
        mmengine.dump = None
        sys.modules["mmengine"] = mmengine
    spec = importlib.util.spec_from_file_location(
        "wr_arena_action_simulation_fidelity",
        UPSTREAM_EVALUATOR,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {UPSTREAM_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_summary(records: list[dict]) -> dict:
    successful = [row for row in records if "native_score_0_to_3" in row]
    groups = {
        "all": successful,
        "correct": [row for row in successful if row["category"] == "correct"],
        "wrong": [row for row in successful if row["category"] == "wrong"],
        "ai": [row for row in successful if row["category"] == "ai"],
    }
    for generator in sorted({row.get("generator") for row in successful} - {None}):
        groups[f"ai/{generator}"] = [
            row for row in successful if row.get("generator") == generator
        ]
    return {
        "definition": "WR-Arena per-video Action Simulation Fidelity",
        "range": [0, 3],
        "higher_is_better": True,
        "groups": {
            name: {
                "videos": len(rows),
                "mean": (
                    round(fmean(row["native_score_0_to_3"] for row in rows), 6)
                    if rows else None
                ),
                "median": (
                    round(median(row["native_score_0_to_3"] for row in rows), 6)
                    if rows else None
                ),
            }
            for name, rows in groups.items()
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(ROOT / ".env")
    commit = require_commit(REPOSITORY, COMMIT)
    integrity = {SOURCE: tracked_file_matches_head(REPOSITORY, SOURCE)}
    if not integrity[SOURCE]:
        raise RuntimeError(f"Official WR-Arena source file was modified: {SOURCE}")

    samples = load_samples(args)
    if args.prepare_only:
        write_json(args.output.expanduser().resolve(), {
            "method": "WR-Arena Action Simulation Fidelity",
            "official_commit": commit,
            "source_integrity": integrity,
            "samples": samples,
            "provider": "openrouter",
            "model": args.model,
            "allowed_delta": "model client/transport only",
        })
        print(f"Prepared {len(samples)} samples")
        return 0

    key_name = "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"WR-Arena evaluator requires {key_name}")
    upstream = load_upstream_module()
    client_kwargs = {
        "api_key": api_key,
        "timeout": 180,
        "max_retries": 2,
    }
    client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
    delegate = OpenAI(**client_kwargs)
    usage = UsageRecorder()

    class Completions:
        def create(self, **kwargs):
            messages = copy.deepcopy(kwargs["messages"])
            for message in messages:
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    image = part.get("image_url", {})
                    url = image.get("url", "")
                    prefix = "data:image/png;base64,"
                    if not url.startswith(prefix):
                        continue
                    raw = base64.b64decode(url[len(prefix):])
                    buffer = io.BytesIO()
                    Image.open(io.BytesIO(raw)).convert("RGB").save(
                        buffer,
                        format="JPEG",
                        quality=90,
                    )
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    image["url"] = f"data:image/jpeg;base64,{encoded}"
            kwargs["messages"] = messages
            response = delegate.chat.completions.create(**kwargs)
            usage.add(response)
            return response

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions())
    )
    model = args.model
    evaluator_fingerprint = f"{commit}:openrouter:{model}"
    output_path = args.output.expanduser().resolve()
    previous = {}
    previous_usage = []
    if not args.resume:
        output_path.unlink(missing_ok=True)
    if args.resume and output_path.is_file():
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        previous_usage = loaded.get("token_usage", {}).get("generations", [])
        previous = {
            row["id"]: row
            for row in loaded.get("records", [])
            if "native_score_0_to_3" in row
            and row.get("evaluator_fingerprint") == evaluator_fingerprint
        }
    usage.items = list(previous_usage)
    records = []
    for index, sample in enumerate(samples, start=1):
        reusable = previous.get(sample["id"])
        if reusable and reusable.get("sample_fingerprint") == sample["sample_fingerprint"]:
            records.append(reusable)
            print(
                f"[{index}/{len(samples)}] {sample['id']}: reused",
                flush=True,
            )
            continue
        try:
            raw = upstream.evaluate_part(
                sample["video_path"],
                sample["prompt"],
                client,
                model,
            )
            text = str(raw).strip()
            if not text.isdigit() or int(text) not in {0, 1, 2, 3}:
                record = {
                    **sample,
                    "raw_upstream_response": raw,
                    "evaluator_fingerprint": evaluator_fingerprint,
                    "error": "invalid_score",
                }
            else:
                record = {
                    **sample,
                    "native_score_0_to_3": int(text),
                    "raw_upstream_response": raw,
                    "evaluator_fingerprint": evaluator_fingerprint,
                }
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            record = {
                **sample,
                "evaluator_fingerprint": evaluator_fingerprint,
                "error": text,
            }
        records.append(record)
        print(f"[{index}/{len(samples)}] {sample['id']}: {text}", flush=True)
        write_json(output_path, {
            "evaluator_fingerprint": evaluator_fingerprint,
            "token_usage": usage.summary(),
            "records": records,
        })

    successful = [row for row in records if "native_score_0_to_3" in row]
    report = {
        "method": "WR-Arena Action Simulation Fidelity",
        "execution": "official evaluate_part with provider-only OpenRouter client",
        "official_repository": "https://github.com/MBZUAI-IFM/WR-Arena",
        "official_commit": commit,
        "source_integrity": integrity,
        "provider": "openrouter",
        "model": model,
        "evaluator_fingerprint": evaluator_fingerprint,
        "allowed_delta": "model client/transport only",
        "provider_media_serialization": "same 8 upstream frames; PNG data URLs converted to JPEG quality 90 at client boundary",
        "official_frame_count": 8,
        "scope": (
            "Native per-video action-fidelity score. WR-Arena's published "
            "agent/environment aggregate requires six benchmark rounds and "
            "is not defined for a single Ego2Act clip."
        ),
        "primary_score": {
            "field": "native_score_0_to_3",
            "range": [0, 3],
            "definition": "official per-video Action Simulation Fidelity score",
        },
        "requested_videos": len(samples),
        "successful_videos": len(successful),
        "failed_videos": len(samples) - len(successful),
        "native_score_report": _native_summary(records),
        "token_usage": usage.summary(),
        "metrics": None,
        "metrics_note": (
            "Task/Physics human scores are not stored in metadata.json. Join "
            "native_score_0_to_3 to the Task 0-3 annotation during analysis."
        ),
        "records": records,
    }
    write_json(output_path, report)
    print(json.dumps(report["primary_score"], indent=2))
    return 0 if len(successful) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
