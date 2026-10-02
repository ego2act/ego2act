from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class GenerationRunner(ABC):
    """Provider boundary for one configured video-generation model."""

    def __init__(self, alias: str, model_id: str) -> None:
        self.alias = alias
        self.model_id = model_id

    @property
    @abstractmethod
    def runner_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def poll(self, url: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def download(self, job_id: str, output: Path) -> None:
        raise NotImplementedError


def api_json(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {error.code}: {detail}") from error


def download_openrouter_video(url: str, api_key: str, output: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    temporary = output.with_suffix(output.suffix + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=300) as response:
        if not (response.headers.get_content_type() or "").startswith("video/"):
            detail = response.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Expected video response, received: {detail}")
        with temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    temporary.replace(output)


class OpenRouterGenerationRunner(GenerationRunner):
    runner_id = "openrouter"

    def __init__(self, alias: str, model_id: str, api_key: str, api_root: str) -> None:
        super().__init__(alias, model_id)
        self.api_key = api_key
        self.api_root = api_root.rstrip("/")

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return api_json(
            f"{self.api_root}/videos",
            self.api_key,
            method="POST",
            payload=payload,
            timeout=180,
        )

    def poll(self, url: str) -> dict[str, Any]:
        if url.startswith("/"):
            url = f"{self.api_root.removesuffix('/api/v1')}{url}"
        return api_json(url, self.api_key, timeout=60)

    def download(self, job_id: str, output: Path) -> None:
        download_openrouter_video(
            f"{self.api_root}/videos/{job_id}/content?index=0",
            self.api_key,
            output,
        )


class StubGenerationRunner(GenerationRunner):
    runner_id = "stub"

    def __init__(self, alias: str, model_id: str, reason: str) -> None:
        super().__init__(alias, model_id)
        self.reason = reason

    def _unavailable(self) -> None:
        raise NotImplementedError(
            f"Generation runner {self.alias!r} ({self.model_id}) is unavailable: "
            f"{self.reason}"
        )

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._unavailable()

    def poll(self, url: str) -> dict[str, Any]:
        self._unavailable()

    def download(self, job_id: str, output: Path) -> None:
        self._unavailable()


def build_runner(
    alias: str,
    spec: dict[str, Any],
    api_key: str,
    api_root: str,
) -> GenerationRunner:
    if spec["runner"] == "openrouter":
        return OpenRouterGenerationRunner(
            alias, spec["model_id"], api_key, api_root
        )
    if spec["runner"] == "stub":
        return StubGenerationRunner(
            alias, spec["model_id"], spec["reason"]
        )
    if spec["runner"] == "cosmos3":
        from ego2act.vidgen.cosmos3 import Cosmos3GenerationRunner

        return Cosmos3GenerationRunner(alias, spec["model_id"])
    if spec["runner"] == "minimax_h3":
        from ego2act.vidgen.minimax_h3 import MiniMaxH3GenerationRunner

        return MiniMaxH3GenerationRunner(alias, spec["model_id"])
    raise ValueError(f"Unknown generation runner: {spec['runner']}")


def model_catalog(api_key: str, api_root: str) -> dict[str, dict[str, Any]]:
    response = api_json(f"{api_root.rstrip('/')}/videos/models", api_key)
    return {item["id"]: item for item in response["data"]}


def live_price_per_second(
    model: dict[str, Any],
    resolution: str,
    generate_audio: bool = False,
) -> float | None:
    prices = model.get("pricing_skus") or {}
    for key in (
        f"image_to_video_duration_seconds_{resolution}",
        (
            "duration_seconds_with_audio"
            if generate_audio
            else "duration_seconds_without_audio"
        ),
        "duration_seconds",
        f"cents_per_video_output_second_{resolution}",
        "cents_per_second_output",
    ):
        if key not in prices:
            continue
        value = float(prices[key])
        return value / 100 if key.startswith("cents_per_") else value
    return None


def validate_openrouter_model(
    alias: str,
    model_id: str,
    live: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "first_frame": "first_frame" in (live.get("supported_frame_images") or []),
        "duration": protocol["duration_seconds"]
        in (live.get("supported_durations") or []),
        "resolution": protocol["resolution"]
        in (live.get("supported_resolutions") or []),
        "aspect_ratio": protocol["aspect_ratio"]
        in (live.get("supported_aspect_ratios") or []),
        "audio": not protocol["generate_audio"] or bool(live.get("generate_audio")),
        "seed_declared": isinstance(live.get("seed"), bool),
    }
    required_checks = {
        "first_frame",
        "duration",
        "resolution",
        "aspect_ratio",
        "audio",
    }
    failed = [
        name for name, passed in checks.items()
        if name in required_checks and not passed
    ]
    if failed:
        raise ValueError(f"{alias} live capability mismatch: {failed}")
    snapshot_keys = (
        "id",
        "canonical_slug",
        "name",
        "supported_durations",
        "supported_resolutions",
        "supported_aspect_ratios",
        "supported_frame_images",
        "generate_audio",
        "seed",
        "pricing_skus",
        "allowed_passthrough_parameters",
    )
    return {
        "alias": alias,
        "runner": "openrouter",
        "available": True,
        "model_id": model_id,
        "display_name": live.get("name") or alias,
        "seed_support": live.get("seed") is True,
        "usd_per_second": live_price_per_second(
            live,
            protocol["resolution"],
            protocol["generate_audio"],
        ),
        "checks": checks,
        "catalog_snapshot": {key: live.get(key) for key in snapshot_keys},
    }


def unavailable_model_report(alias: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "alias": alias,
        "runner": spec["runner"],
        "available": False,
        "model_id": spec["model_id"],
        "display_name": alias,
        "seed_support": False,
        "usd_per_second": None,
        "checks": {},
        "catalog_snapshot": None,
        "reason": spec["reason"],
    }


def ego2act_model_report(alias: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "alias": alias,
        "runner": spec["runner"],
        "available": True,
        "model_id": spec["model_id"],
        "display_name": alias,
        "seed_support": True,
        "usd_per_second": None,
        "checks": {
            "first_frame": True,
            "duration": True,
            "resolution": True,
            "aspect_ratio": True,
            "audio": True,
        },
        "catalog_snapshot": None,
        "reason": spec.get("reason"),
    }
