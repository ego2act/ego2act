from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ego2act.vidgen.ego2act_backend import (
    Ego2ActGenerationRunner,
    env_positive_float,
    env_positive_int,
    first_frame_data_url,
)


class MiniMaxH3GenerationRunner(Ego2ActGenerationRunner):
    """MiniMax-H3 FL2VA adapter for text plus one first-frame image."""

    runner_id = "minimax_h3"
    fps = 24
    inference_steps = 35
    job_timeout_seconds = 3600

    def __init__(
        self,
        alias: str,
        model_id: str,
        backend_url: str | None = None,
    ) -> None:
        super().__init__(alias, model_id, backend_url)
        self.inference_steps = env_positive_int("MINIMAX_H3_STEPS", self.inference_steps)
        self.flow_shift = env_positive_float("MINIMAX_H3_FLOW_SHIFT", 12.0)
        self.audio_flow_shift = env_positive_float("MINIMAX_H3_AUDIO_FLOW_SHIFT", 3.0)

    def build_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        duration = payload.get("duration")
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or not 4 <= duration <= 15
        ):
            raise ValueError("MiniMax-H3 duration must be an integer from 4 through 15 seconds")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("MiniMax-H3 prompt must be a non-empty string")
        aspect_ratio = _aspect_ratio(payload.get("aspect_ratio"))
        resolution = payload.get("resolution")
        if (
            not isinstance(resolution, str)
            or not resolution.endswith("p")
            or not resolution[:-1].isdigit()
            or int(resolution[:-1]) <= 0
        ):
            raise ValueError(
                "MiniMax-H3 resolution must be a positive short-edge label such as '480p'"
            )
        options: dict[str, Any] = {
            "duration": duration,
            # FL2VA derives its canvas from the supplied endpoint image.
            "aspect_ratio": "auto",
            "fps": self.fps,
            "num_inference_steps": self.inference_steps,
            # H3 always creates synchronized audio. Ego2Act's canonical proxy
            # still strips it when generation.protocol.generate_audio is false.
            "generate_audio": True,
            "timeout_seconds": self.job_timeout_seconds,
            "provider_options": {
                "minimax_h3": {
                    # H3 0.5.18 only accepts its native 768px canvas. Ego2Act
                    # applies the requested output resolution during canonicalization.
                    "short_edge": 768,
                    "frame_index": 0,
                    "flow_shift": self.flow_shift,
                    "audio_flow_shift": self.audio_flow_shift,
                }
            },
        }
        seed = payload.get("seed")
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError("MiniMax-H3 seed must be a non-negative integer")
            options["seed"] = seed

        return {
            "provider": "sglang",
            "model": self.model_id,
            "mode": "image_to_video",
            "prompt": prompt.strip(),
            "image_base64": _center_crop_first_frame(payload, aspect_ratio),
            "options": options,
            "metadata": {
                "client": "ego2act",
                "model_alias": self.alias,
                "minimax_h3_task": "fl2va",
            },
        }


def _aspect_ratio(value: Any) -> float:
    if not isinstance(value, str):
        raise ValueError("MiniMax-H3 aspect ratio must use a positive W:H label")
    parts = value.split(":")
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ValueError("MiniMax-H3 aspect ratio must use a positive W:H label")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("MiniMax-H3 aspect ratio must use a positive W:H label")
    return width / height


def _center_crop_first_frame(payload: dict[str, Any], target_ratio: float) -> str:
    data_url = first_frame_data_url(payload)
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.endswith(";base64"):
        raise ValueError("MiniMax-H3 first-frame image must use a base64 data URL")
    raw = base64.b64decode(encoded, validate=True)
    image = _decode_first_frame(raw, "MiniMax-H3 first-frame image")

    height, width = image.shape[:2]
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_width = max(1, round(height * target_ratio))
        left = (width - crop_width) // 2
        image = image[:, left : left + crop_width]
    elif source_ratio < target_ratio:
        crop_height = max(1, round(width / target_ratio))
        top = (height - crop_height) // 2
        image = image[top : top + crop_height, :]

    encoded_ok, buffer = cv2.imencode(".png", image)
    if not encoded_ok:
        raise RuntimeError("MiniMax-H3 first-frame crop could not be encoded")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


def validate_start_image(path: Path) -> None:
    """Fail before server startup when H3 cannot decode a conditioning image."""
    _decode_first_frame(path.read_bytes(), f"MiniMax-H3 cannot decode start image: {path}")


def _decode_first_frame(raw: bytes, error_message: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(error_message)
    return image

