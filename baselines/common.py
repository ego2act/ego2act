from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from ego2act.data import (
    discover_dataset_cases,
    evaluable_videos,
    load_dataset_case,
)


ROOT = Path(__file__).resolve().parents[1]
AXIS_INDEX = {"task": 0, "object": 1, "interaction": 2}
DEFAULT_API_JUDGE_MODEL = "google/gemini-3.7-flash"


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ego2act")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--case-file",
        action="append",
        type=Path,
        default=[],
        help="Text file containing one case ID per line; order is preserved.",
    )
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument(
        "--category",
        action="append",
        choices=("correct", "wrong", "ai"),
        default=[],
    )
    parser.add_argument(
        "--generator",
        action="append",
        default=[],
        help="Run only AI videos from this generator alias; repeat as needed.",
    )
    parser.add_argument(
        "--exclude-generator",
        action="append",
        default=[],
        help=(
            "Exclude AI videos from this generator while retaining human "
            "controls and other generators; repeat as needed."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="Run only generated videos with this seed; repeat as needed.",
    )
    parser.add_argument("--limit", type=int)


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    data_root = args.data_root.expanduser().resolve()
    selected_cases = _selected_case_ids(args)
    selected_videos = set(getattr(args, "video", []) or [])
    selected_categories = set(getattr(args, "category", []) or [])
    selected_generators = set(getattr(args, "generator", []) or [])
    excluded_generators = set(getattr(args, "exclude_generator", []) or [])
    selected_seeds = set(getattr(args, "seed", []) or [])
    overlap = selected_generators & excluded_generators
    if overlap:
        raise ValueError(
            f"Generator aliases cannot be both included and excluded: {sorted(overlap)}"
        )
    if selected_cases:
        cases = [load_dataset_case(case_id, data_root) for case_id in selected_cases]
    else:
        cases = discover_dataset_cases(data_root)

    samples = []
    for case in cases:
        videos = evaluable_videos(case)
        if selected_videos:
            order = {key: index for index, key in enumerate(args.video)}
            videos.sort(key=lambda item: (order.get(item["key"], len(order)), item["key"]))
        for video in videos:
            if selected_videos and video["key"] not in selected_videos:
                continue
            if selected_categories and video["category"] not in selected_categories:
                continue
            generation = video.get("generation") or {}
            generator = generation.get("model_alias")
            seed = generation.get("sample_id")
            if selected_generators and generator not in selected_generators:
                continue
            if generator in excluded_generators:
                continue
            if selected_seeds and seed not in selected_seeds:
                continue
            gt = (
                [float(value) for value in video["gt"]]
                if video.get("gt") is not None
                else None
            )
            video_path = Path(video["video_path"]).resolve()
            video_sha256 = sha256(video_path)
            prompt_sha256 = hashlib.sha256(case["goal"].encode("utf-8")).hexdigest()
            sample_id = f"{case['id']}::{video['key']}"
            fingerprint = hashlib.sha256(json.dumps(
                {
                    "id": sample_id,
                    "video_sha256": video_sha256,
                    "prompt_sha256": prompt_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            samples.append({
                "id": sample_id,
                "case": case["id"],
                "video": video["key"],
                "category": video["category"],
                "generator": generator,
                "seed": seed,
                "prompt": case["goal"],
                "objects": list(case["objects"]),
                "video_path": str(video_path),
                "video_sha256": video_sha256,
                "prompt_sha256": prompt_sha256,
                "sample_fingerprint": fingerprint,
                "gt": gt,
                "human_final_1_to_4": fmean(gt) if gt else None,
                "description": video["description"],
            })
    if selected_videos:
        found = {sample["video"] for sample in samples}
        missing = selected_videos - found
        if missing:
            raise ValueError(f"Unknown video keys in selected cases: {sorted(missing)}")
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("No videos matched the requested dataset filters")
    return samples


def _selected_case_ids(args: argparse.Namespace) -> list[str]:
    values = list(getattr(args, "case", []) or [])
    for path in getattr(args, "case_file", []) or []:
        for raw in path.expanduser().read_text(encoding="utf-8").splitlines():
            value = raw.split("#", 1)[0].strip()
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_commit(repository: Path, expected: str) -> str:
    if not (repository / ".git").exists():
        raise FileNotFoundError(f"Missing upstream checkout: {repository}")
    actual = git_commit(repository)
    if actual != expected:
        raise RuntimeError(
            f"Upstream commit mismatch for {repository.name}: "
            f"expected {expected}, found {actual}"
        )
    return actual


def tracked_file_matches_head(repository: Path, relative_path: str) -> bool:
    worktree = (repository / relative_path).read_bytes()
    committed = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"HEAD:{relative_path}"]
    )
    return worktree == committed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def usage_dict(usage: Any) -> dict:
    extra = getattr(usage, "model_extra", None) or {}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cost": float(extra.get("cost", 0) or 0),
        "model_calls": 1,
    }


def merge_usage(*items: dict) -> dict:
    if not items:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "model_calls": 0,
        }
    return {
        key: sum(float(item[key]) for item in items) if key == "cost" else sum(int(item[key]) for item in items)
        for key in items[0]
    }


class UsageRecorder:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, response: Any) -> None:
        item = usage_dict(response.usage)
        item["generation_id"] = getattr(response, "id", None)
        item["model"] = getattr(response, "model", None)
        self.items.append(item)

    def summary(self) -> dict[str, Any]:
        totals = merge_usage(*[
            {key: item[key] for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
                "model_calls",
            )}
            for item in self.items
        ])
        return {**totals, "generations": self.items}


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    display_command: list[str] | None = None,
) -> None:
    print("$ " + " ".join(display_command or command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def agreement(
    records: Iterable[dict[str, Any]],
    *,
    score_key: str,
    native_min: float,
    native_max: float,
    target_axes: tuple[str, ...] = ("task", "object", "interaction"),
) -> dict[str, Any]:
    rows = [
        record
        for record in records
        if score_key in record
        and isinstance(record.get("gt"), list)
        and len(record["gt"]) == len(AXIS_INDEX)
    ]
    if not target_axes or any(axis not in AXIS_INDEX for axis in target_axes):
        raise ValueError(f"Unknown or empty target_axes={target_axes}")
    truth = [
        fmean(float(record["gt"][AXIS_INDEX[axis]]) for axis in target_axes)
        for record in rows
    ]
    native = [float(record[score_key]) for record in rows]
    prediction = [
        1.0 + 3.0 * (score - native_min) / (native_max - native_min)
        for score in native
    ]
    return {
        "target": "mean(" + ", ".join(target_axes) + ")",
        "videos": len(rows),
        "native_scale": [native_min, native_max],
        "human_scale": [1, 4],
        "mae_after_affine_scale_alignment": _rounded(
            fmean(abs(a - b) for a, b in zip(truth, prediction))
            if rows else None
        ),
        "pearson_r": _rounded(_pearson(truth, native)),
        "kendall_tau_b": _rounded(_kendall_tau_b(truth, native)),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right)
    )
    denominator = sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def _kendall_tau_b(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_direction = (left[first] > left[second]) - (
                left[first] < left[second]
            )
            right_direction = (right[first] > right[second]) - (
                right[first] < right[second]
            )
            if left_direction == 0 and right_direction == 0:
                continue
            if left_direction == 0:
                left_ties += 1
            elif right_direction == 0:
                right_ties += 1
            elif left_direction == right_direction:
                concordant += 1
            else:
                discordant += 1
    denominator = sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
