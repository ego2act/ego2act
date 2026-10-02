from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(path: str | Path) -> dict[str, float | int]:
    video = Path(path).resolve()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open generated video: {video}")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frames < 1 or fps <= 0 or width < 1 or height < 1:
        raise RuntimeError(f"Invalid generated video metadata: {video}")
    return {
        "frame_count": frames,
        "fps": fps,
        "duration_seconds": frames / fps,
        "width": width,
        "height": height,
        "size_bytes": video.stat().st_size,
    }


def proxy_dimensions(metadata: dict, policy: dict) -> tuple[int, int]:
    width, height = int(metadata["width"]), int(metadata["height"])
    if width >= height:
        max_width, max_height = policy["max_long_side"], policy["max_short_side"]
    else:
        max_width, max_height = policy["max_short_side"], policy["max_long_side"]
    scale = min(1.0, max_width / width, max_height / height)
    return (
        max(2, int(width * scale) // 2 * 2),
        max(2, int(height * scale) // 2 * 2),
    )


def standardize_video(
    source_path: str | Path,
    output_path: str | Path,
    policy: dict,
) -> dict:
    """Encode one provider-native MP4 to the root config's canonical contract."""
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if source.suffix.lower() != ".mp4" or output.suffix.lower() != ".mp4":
        raise ValueError("Generation inputs and outputs must use the MP4 container")
    if source == output:
        raise ValueError("Native and standardized generation paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_metadata = video_metadata(source)
    width, height = proxy_dimensions(source_metadata, policy)
    fps = min(float(source_metadata["fps"]), float(policy["max_fps"]))
    temporary = output.with_suffix(".encoding.mp4")
    last_size = None
    try:
        for crf in policy["crf_attempts"]:
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"scale={width}:{height}:flags=lanczos,fps={fps:.6f}",
                "-c:v",
                policy["codec"],
                "-crf",
                str(crf),
                "-pix_fmt",
                policy["pixel_format"],
                "-movflags",
                "+faststart",
                str(temporary),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip()[-1200:])
            encoded = video_metadata(temporary)
            last_size = int(encoded["size_bytes"])
            if last_size <= policy["max_bytes"]:
                os.replace(temporary, output)
                standardized = video_metadata(output)
                return {
                    "native_path": str(source),
                    "native_sha256": file_sha256(source),
                    "native": source_metadata,
                    "output_path": str(output),
                    "output_sha256": file_sha256(output),
                    "output": standardized,
                    "codec": policy["codec"],
                    "pixel_format": policy["pixel_format"],
                    "crf": crf,
                }
    finally:
        temporary.unlink(missing_ok=True)
    raise ValueError(
        "Generated video exceeds the canonical byte limit after compression: "
        f"{last_size}"
    )
