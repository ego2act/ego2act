"""Shared video preprocessing for the baseline adapters."""
import hashlib
import json
import os
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
from openai import OpenAI
from baselines.config import get_experiment_config, make_run_id

_CONFIG = get_experiment_config()
_JUDGE = _CONFIG["baselines"]
PROJECT_ROOT = _CONFIG["_meta"]["config_path"].parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_TIMEOUT_SECONDS = float(_JUDGE["decoding"]["timeout_seconds"])
INFERENCE_VIDEO_MAX_LONG_SIDE = _JUDGE["video"]["max_long_side"]
INFERENCE_VIDEO_MAX_SHORT_SIDE = _JUDGE["video"]["max_short_side"]
INFERENCE_VIDEO_MAX_FPS = float(_JUDGE["video"]["max_fps"])
INFERENCE_VIDEO_MAX_BYTES = _JUDGE["video"]["max_bytes"]
INFERENCE_VIDEO_CRF_ATTEMPTS = tuple(_JUDGE["video"]["crf_attempts"])
INFERENCE_VIDEO_CODEC = _JUDGE["video"]["codec"]
INFERENCE_VIDEO_PIXEL_FORMAT = _JUDGE["video"]["pixel_format"]


def get_video_metadata(video_path: str | Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count < 1 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
        "size_bytes": Path(video_path).stat().st_size,
    }


def _inference_proxy_dimensions(metadata: dict) -> tuple[int, int]:
    width, height = metadata["width"], metadata["height"]
    if width >= height:
        max_width, max_height = (
            INFERENCE_VIDEO_MAX_LONG_SIDE, INFERENCE_VIDEO_MAX_SHORT_SIDE,
        )
    else:
        max_width, max_height = (
            INFERENCE_VIDEO_MAX_SHORT_SIDE, INFERENCE_VIDEO_MAX_LONG_SIDE,
        )
    scale = min(1.0, max_width / width, max_height / height)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    return target_width, target_height


def _estimated_base64_bytes(binary_bytes: int) -> int:
    return 4 * ((int(binary_bytes) + 2) // 3)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_video_for_inference(
    video_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, dict]:
    source_path = Path(video_path).resolve()
    if source_path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError(
            f"Expected an MP4 or MOV video, got: {source_path.suffix}"
        )
    source_sha256 = _sha256_file(source_path)
    source = get_video_metadata(source_path)
    target_width, target_height = _inference_proxy_dimensions(source)
    target_fps = min(source["fps"], INFERENCE_VIDEO_MAX_FPS)

    if output_path is None:
        run_dir = (
            PROJECT_ROOT
            / ".results"
            / make_run_id(source_path.parent.name)
        )
        proxy_path = run_dir / "shared" / "inference_video.mp4"
    else:
        candidate = Path(output_path).expanduser().resolve()
        proxy_path = (
            candidate
            if candidate.suffix.lower() == ".mp4"
            else candidate / "shared" / "inference_video.mp4"
        )
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = proxy_path.with_suffix(proxy_path.suffix + ".source.json")

    policy = {
        "container": "mp4",
        "codec": INFERENCE_VIDEO_CODEC,
        "pixel_format": INFERENCE_VIDEO_PIXEL_FORMAT,
        "width": target_width,
        "height": target_height,
        "fps": round(target_fps, 6),
        "max_bytes": INFERENCE_VIDEO_MAX_BYTES,
        "crf_attempts": list(INFERENCE_VIDEO_CRF_ATTEMPTS),
    }

    def build_report(proxy_metadata: dict, crf: int | None, reused: bool) -> dict:
        return {
            "source": source,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "proxy": proxy_metadata,
            "proxy_path": str(proxy_path),
            "proxy_sha256": _sha256_file(proxy_path),
            "policy": policy,
            "crf": crf,
            "reused": reused,
            "estimated_base64_bytes": _estimated_base64_bytes(
                proxy_metadata["size_bytes"]
            ),
        }

    if proxy_path.exists():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            proxy = get_video_metadata(proxy_path)
            duration_tolerance = max(0.15, 2.0 / max(target_fps, 1.0))
            valid = (
                provenance.get("source_sha256") == source_sha256
                and provenance.get("policy") == policy
                and proxy["width"] == target_width
                and proxy["height"] == target_height
                and proxy["fps"] <= target_fps + 0.05
                and proxy["size_bytes"] <= INFERENCE_VIDEO_MAX_BYTES
                and abs(proxy["duration_seconds"] - source["duration_seconds"])
                <= duration_tolerance
            )
            if valid:
                return proxy_path, build_report(proxy, crf=None, reused=True)
        except Exception:
            pass

    temporary_path = proxy_path.with_name(proxy_path.stem + ".encoding.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    last_size = None
    try:
        for crf in INFERENCE_VIDEO_CRF_ATTEMPTS:
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_path),
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", (
                    f"scale={target_width}:{target_height}:flags=lanczos,"
                    f"fps={target_fps:.6f}"
                ),
                "-c:v", INFERENCE_VIDEO_CODEC, "-preset", "medium",
                "-crf", str(crf), "-pix_fmt", INFERENCE_VIDEO_PIXEL_FORMAT,
                "-movflags", "+faststart", "-map_metadata", "-1",
                str(temporary_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
                raise RuntimeError(f"Could not standardize video: {detail[-1200:]}")
            proxy = get_video_metadata(temporary_path)
            duration_tolerance = max(0.15, 2.0 / max(target_fps, 1.0))
            if (
                proxy["width"] != target_width
                or proxy["height"] != target_height
                or proxy["fps"] > target_fps + 0.05
                or abs(proxy["duration_seconds"] - source["duration_seconds"])
                > duration_tolerance
            ):
                raise RuntimeError(
                    "Standardized proxy did not preserve the requested geometry/timeline: "
                    f"{proxy}"
                )
            last_size = proxy["size_bytes"]
            if last_size <= INFERENCE_VIDEO_MAX_BYTES:
                os.replace(temporary_path, proxy_path)
                provenance_path.write_text(json.dumps(
                    {
                        "source_path": str(source_path),
                        "source_sha256": source_sha256,
                        "policy": policy,
                    },
                    indent=2,
                ), encoding="utf-8")
                proxy = get_video_metadata(proxy_path)
                return proxy_path, build_report(proxy, crf=crf, reused=False)
    finally:
        temporary_path.unlink(missing_ok=True)

    raise ValueError(
        "Standardized video still exceeds the configured inference limit: "
        f"{last_size / (1024 * 1024):.2f} MiB"
    )


def create_openrouter_client(
    api_key: str | None = None,
    base_url: str = OPENROUTER_BASE_URL,
) -> OpenAI:
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise ValueError(
            "An OpenRouter API key is required. Pass api_key=... or set "
            "OPENROUTER_API_KEY."
        )
    return OpenAI(
        base_url=base_url,
        api_key=resolved_key,
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )
