"""VideoScore inference.

Requires torch, transformers, mantis-vl and av, so this module is imported only
on the scoring host (a CUDA runtime). The local analysis path in run.py does not
import it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from mantis.models.idefics2 import Idefics2ForSequenceClassification

from baselines.videoscore.mapping import ASPECTS


MODEL_REPOSITORY = "TIGER-Lab/VideoScore-v1.1"
MODEL_REVISION = "0731e98e15f8c06a3e3b4dc6c7a4b8d866f22a89"
DEFAULT_MODEL = MODEL_REPOSITORY
DEFAULT_MAX_FRAMES = 48
ROUND_DIGITS = 3

# Verbatim from TIGER-AI-Lab/VideoScore benchmark/utils_tools.py. Any edit makes
# the run a prompt-modified adaptation rather than upstream scoring logic.
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""


def prompt_sha256() -> str:
    return hashlib.sha256(REGRESSION_QUERY_PROMPT.encode("utf-8")).hexdigest()


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str | None, device: torch.device) -> torch.dtype:
    """Pick the widest dtype the device actually accelerates.

    The checkpoint ships as bfloat16, but Turing cards (Colab's free T4) have no
    bfloat16 units, and bfloat16 matmuls off CUDA fall back to emulation.
    """
    if name:
        return getattr(torch, name)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def describe_runtime(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    description: dict[str, Any] = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        description["gpu_name"] = properties.name
        description["gpu_total_bytes"] = properties.total_memory
        description["bf16_supported"] = torch.cuda.is_bf16_supported()
    return description


def check_memory(device: torch.device, dtype: torch.dtype, weight_bytes: int = 16_543_408_618) -> None:
    """Fail before a 16.5 GB download when the device plainly cannot hold it."""
    if device.type != "cuda":
        return
    total = torch.cuda.get_device_properties(device).total_memory
    needed = weight_bytes if dtype.itemsize == 2 else weight_bytes * 2
    if needed >= total:
        raise RuntimeError(
            f"VideoScore needs ~{needed / 2**30:.1f} GiB for weights in {dtype} but "
            f"{torch.cuda.get_device_properties(device).name} exposes only "
            f"{total / 2**30:.1f} GiB. Use an L4 or larger, or pass an explicit "
            f"quantised loading path."
        )


def load_model(
    model_name: str = DEFAULT_MODEL,
    model_revision: str | None = MODEL_REVISION,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Idefics2ForSequenceClassification, AutoProcessor, dict[str, Any]]:
    resolved_device = device or resolve_device()
    resolved_dtype = dtype or resolve_dtype(None, resolved_device)
    check_memory(resolved_device, resolved_dtype)

    load_options = {"torch_dtype": resolved_dtype}
    if model_revision is not None:
        load_options["revision"] = model_revision
    processor = AutoProcessor.from_pretrained(model_name, **load_options)
    model = Idefics2ForSequenceClassification.from_pretrained(
        model_name, **load_options
    ).eval()
    model.to(resolved_device)
    return model, processor, describe_runtime(resolved_device, resolved_dtype)


def sample_frames(video_path: str | Path, max_frames: int = DEFAULT_MAX_FRAMES) -> list[Image.Image]:
    """Uniformly sample frames, matching upstream eval_videoscore.py's linspace rule.

    Only the selected frames are retained; a standardized 24 fps clip holds a few
    hundred frames and keeping every decoded frame would cost hundreds of MB.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames <= 0 and stream.duration and stream.average_rate:
            duration = float(stream.duration * stream.time_base)
            total_frames = int(duration * float(stream.average_rate))
        if total_frames <= 0:
            raise ValueError(f"Could not determine the frame count of {video_path}")

        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)
        else:
            indices = np.arange(total_frames)
        wanted = {int(index) for index in indices}
        last_wanted = max(wanted)

        frames = []
        for position, frame in enumerate(container.decode(video=0)):
            if position in wanted:
                frames.append(frame.to_image().convert("RGB"))
            if position >= last_wanted:
                break

    if not frames:
        raise ValueError(f"Decoded no frames from {video_path}")
    return frames


def score_frames(
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    goal: str,
    frames: list[Image.Image],
) -> dict[str, float]:
    prompt = REGRESSION_QUERY_PROMPT.format(text_prompt=goal)
    image_tokens = prompt.count("<image>")
    if image_tokens < len(frames):
        prompt += "<image> " * (len(frames) - image_tokens)

    inputs = processor(text=prompt, images=frames, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    if logits.shape[-1] != len(ASPECTS):
        raise RuntimeError(
            f"Expected {len(ASPECTS)} regression heads, got {logits.shape[-1]}"
        )
    return {
        aspect: round(logits[0, index].item(), ROUND_DIGITS)
        for index, aspect in enumerate(ASPECTS)
    }


def score_video(
    model: Idefics2ForSequenceClassification,
    processor: AutoProcessor,
    goal: str,
    video_path: str | Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> tuple[dict[str, float], int]:
    frames = sample_frames(video_path, max_frames)
    return score_frames(model, processor, goal, frames), len(frames)
