from __future__ import annotations

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


RUNNER_KINDS = {"cosmos3", "minimax_h3", "openrouter", "stub"}


def _default_config_path() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "config.yaml",
        Path(sys.prefix).resolve() / "share/ego2act/config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find config.yaml at the repository root or in the active "
        "environment's share/ego2act directory."
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{label} keys are invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return float(value)


def _validate_video_policy(video: dict[str, Any]) -> None:
    _keys(
        video,
        {
            "codec",
            "pixel_format",
            "max_long_side",
            "max_short_side",
            "max_fps",
            "max_bytes",
            "crf_attempts",
        },
        "generation.video",
    )
    for key in ("codec", "pixel_format"):
        if not isinstance(video[key], str) or not video[key].strip():
            raise ValueError(f"generation.video.{key} must be a non-empty string")
    for key in ("max_long_side", "max_short_side", "max_bytes"):
        _positive_integer(video[key], f"generation.video.{key}")
    _positive_number(video["max_fps"], "generation.video.max_fps")
    attempts = video["crf_attempts"]
    if (
        not isinstance(attempts, list)
        or not attempts
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or not 0 <= item <= 51
            for item in attempts
        )
    ):
        raise ValueError(
            "generation.video.crf_attempts must contain CRF integers from 0 to 51"
        )


def _validate_generation(generation: dict[str, Any], root: dict[str, Any]) -> None:
    _keys(
        generation,
        {
            "api_root",
            "data_root",
            "protocol",
            "output_template",
            "video",
            "models",
        },
        "generation",
    )
    for key in ("api_root", "data_root", "output_template"):
        if not isinstance(generation[key], str) or not generation[key].strip():
            raise ValueError(f"generation.{key} must be a non-empty string")
    for field in ("{model_name}", "{sample_id}"):
        if field not in generation["output_template"]:
            raise ValueError(f"generation.output_template must contain {field}")

    protocol = _mapping(generation["protocol"], "generation.protocol")
    _keys(
        protocol,
        {
            "duration_seconds",
            "resolution",
            "aspect_ratio",
            "generate_audio",
            "sample_ids",
        },
        "generation.protocol",
    )
    _positive_integer(
        protocol["duration_seconds"], "generation.protocol.duration_seconds"
    )
    for key in ("resolution", "aspect_ratio"):
        if not isinstance(protocol[key], str) or not protocol[key].strip():
            raise ValueError(
                f"generation.protocol.{key} must be a non-empty string"
            )
    if not isinstance(protocol["generate_audio"], bool):
        raise TypeError("generation.protocol.generate_audio must be boolean")
    sample_ids = protocol["sample_ids"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len(sample_ids) != len(set(sample_ids))
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in sample_ids
        )
    ):
        raise ValueError(
            "generation.protocol.sample_ids must be unique non-negative integers"
        )

    video = _mapping(generation["video"], "generation.video")
    _validate_video_policy(video)
    judge = root.get("baselines")
    if isinstance(judge, dict) and isinstance(judge.get("video"), dict):
        if video != judge["video"]:
            raise ValueError(
                "generation.video must match baselines.video so every generated "
                "video uses the same MP4 contract as the evaluators"
            )

    models = _mapping(generation["models"], "generation.models")
    if not models:
        raise ValueError("generation.models must not be empty")
    for alias, raw_spec in models.items():
        if not re.fullmatch(r"[a-z0-9_]+", alias):
            raise ValueError(f"Invalid generation model alias: {alias}")
        spec = _mapping(raw_spec, f"generation.models.{alias}")
        runner = spec.get("runner")
        if runner not in RUNNER_KINDS:
            raise ValueError(
                f"generation.models.{alias}.runner must be one of "
                f"{sorted(RUNNER_KINDS)}"
            )
        expected = {"runner", "model_id"}
        if runner != "openrouter":
            expected.add("reason")
        if "protocol" in spec:
            expected.add("protocol")
        _keys(spec, expected, f"generation.models.{alias}")
        if not isinstance(spec["model_id"], str) or not spec["model_id"].strip():
            raise ValueError(
                f"generation.models.{alias}.model_id must be a non-empty string"
            )
        if runner != "openrouter" and (
            not isinstance(spec["reason"], str) or not spec["reason"].strip()
        ):
            raise ValueError(
                f"generation.models.{alias}.reason must be a non-empty string"
            )
        if "protocol" in spec:
            model_protocol = _mapping(
                spec["protocol"], f"generation.models.{alias}.protocol"
            )
            unknown = set(model_protocol) - {"duration_seconds", "resolution", "aspect_ratio"}
            if unknown or not model_protocol:
                raise ValueError(
                    f"generation.models.{alias}.protocol may override only "
                    "duration_seconds, resolution, and aspect_ratio"
                )
            if "duration_seconds" in model_protocol:
                _positive_integer(
                    model_protocol["duration_seconds"],
                    f"generation.models.{alias}.protocol.duration_seconds",
                )
            for key in ("resolution", "aspect_ratio"):
                if key in model_protocol and (
                    not isinstance(model_protocol[key], str)
                    or not model_protocol[key].strip()
                ):
                    raise ValueError(
                        f"generation.models.{alias}.protocol.{key} must be a non-empty string"
                    )


def load_generation_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate only the top-level video-generation configuration."""
    config_path = Path(path).resolve() if path is not None else _default_config_path()
    with config_path.open("r", encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream), "config")
    generation = _mapping(root.get("generation"), "generation")
    _validate_generation(generation, root)

    normalized = json.dumps(generation, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    output = {"generation": deepcopy(generation)}
    output["_meta"] = {
        "config_path": config_path,
        "generation_config_sha256": digest,
        "data_root": (config_path.parent / generation["data_root"]).resolve(),
    }
    return output
