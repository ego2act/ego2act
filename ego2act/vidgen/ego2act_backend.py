from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from ego2act.vidgen.runners import GenerationRunner


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
STATUS_MAP = {
    "queued": "pending",
    "running": "processing",
    "succeeded": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}


def env_positive_int(name: str, default: int | None = None) -> int | None:
    """Positive integer from the environment, or `default` when unset."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def configured_backend_url() -> str:
    return (
        os.environ.get("EGO2ACT_API_URL")
        or DEFAULT_BACKEND_URL
    ).rstrip("/")


def resolution_dimensions(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    if not resolution.endswith("p") or not resolution[:-1].isdigit():
        raise ValueError(
            f"Ego2Act backend runners require a resolution such as '720p', got {resolution!r}"
        )
    try:
        ratio_width, ratio_height = (int(value) for value in aspect_ratio.split(":"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Ego2Act backend runners require an aspect ratio such as '16:9', got {aspect_ratio!r}"
        ) from error
    if ratio_width <= 0 or ratio_height <= 0:
        raise ValueError("Ego2Act backend aspect-ratio components must be positive")

    short_side = int(resolution[:-1])
    ratio = ratio_width / ratio_height
    if ratio >= 1:
        height = short_side
        width = round(height * ratio)
    else:
        width = short_side
        height = round(width / ratio)
    return width // 2 * 2, height // 2 * 2


def constrained_frame_count(duration_seconds: int, fps: int) -> int:
    """Return the nearest Wan/Cosmos-compatible 4n+1 frame count."""
    requested = duration_seconds * fps
    lower = max(1, requested - ((requested - 1) % 4))
    upper = lower + 4
    return min((lower, upper), key=lambda value: (abs(value - requested), -value))


def first_frame_data_url(payload: dict[str, Any]) -> str:
    for frame in payload.get("frame_images") or []:
        if frame.get("frame_type") != "first_frame":
            continue
        image = frame.get("image_url") or {}
        value = image.get("url")
        if isinstance(value, str) and value.startswith("data:image/"):
            return value
    raise ValueError(
        "Ego2Act backend image-to-video submission requires one first-frame image data URL"
    )


class Ego2ActGenerationRunner(GenerationRunner):
    """HTTP transport shared by dedicated Ego2Act-backend model runners."""

    fps: int
    inference_steps: int
    job_timeout_seconds = 1800

    def __init__(
        self,
        alias: str,
        model_id: str,
        backend_url: str | None = None,
    ) -> None:
        super().__init__(alias, model_id)
        self.backend_url = (backend_url or configured_backend_url()).rstrip("/")

    def provider_options(self) -> dict[str, Any]:
        return {}

    def extra_options(self) -> dict[str, Any]:
        return {}

    def build_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        duration = payload.get("duration")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration < 1:
            raise ValueError("Ego2Act backend submission duration must be a positive integer")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Ego2Act backend submission prompt must be a non-empty string")
        width, height = resolution_dimensions(
            payload.get("resolution", ""), payload.get("aspect_ratio", "")
        )
        options: dict[str, Any] = {
            "height": height,
            "width": width,
            "num_frames": constrained_frame_count(duration, self.fps),
            "fps": self.fps,
            "num_inference_steps": self.inference_steps,
            "generate_audio": bool(payload.get("generate_audio", False)),
            "timeout_seconds": self.job_timeout_seconds,
            **self.extra_options(),
        }
        seed = payload.get("seed")
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("Ego2Act backend submission seed must be a non-negative integer")
            options["seed"] = seed
        provider_options = self.provider_options()
        if provider_options:
            options["provider_options"] = provider_options
        return {
            "provider": "sglang",
            "model": self.model_id,
            "mode": "image_to_video",
            "prompt": prompt.strip(),
            "image_base64": first_frame_data_url(payload),
            "options": options,
            "metadata": {
                "client": "ego2act",
                "model_alias": self.alias,
            },
        }

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._json_request(
            "/v1/video/generations",
            method="POST",
            payload=self.build_request(payload),
        )
        job_id = response.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"Ego2Act backend submission response has no job id: {response}")
        normalized = self._normalize(response)
        normalized["polling_url"] = f"{self.backend_url}/v1/video/generations/{job_id}"
        return normalized

    def poll(self, url: str) -> dict[str, Any]:
        return self._normalize(self._json_request(url))

    def download(self, job_id: str, output: Path) -> None:
        url = f"{self.backend_url}/v1/video/generations/{job_id}/video"
        request = urllib.request.Request(url)
        temporary = output.with_suffix(output.suffix + ".part")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                content_type = response.headers.get_content_type() or ""
                if not content_type.startswith("video/"):
                    detail = response.read(1000).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Ego2Act backend returned {content_type or 'unknown content'} "
                        f"instead of video: {detail}"
                    )
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Ego2Act backend returned an empty video")
            temporary.replace(output)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ego2Act backend HTTP {error.code} while downloading {job_id}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Ego2Act backend is unreachable at {self.backend_url}: {error.reason}"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    def _json_request(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = url if url.startswith(("http://", "https://")) else urljoin(
            f"{self.backend_url}/", url.lstrip("/")
        )
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            resolved,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ego2Act backend HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Ego2Act backend is unreachable at {self.backend_url}: {error.reason}. "
                "Start the Ego2Act inference server (scripts/run_backend.sh) and "
                "the selected model server first."
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError("Ego2Act backend returned a non-object JSON response")
        return result

    @staticmethod
    def _normalize(response: dict[str, Any]) -> dict[str, Any]:
        result = dict(response)
        status = result.get("status")
        if isinstance(status, str):
            result["provider_status"] = status
            result["status"] = STATUS_MAP.get(status, status)
        return result
