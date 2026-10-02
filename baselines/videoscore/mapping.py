"""VideoScore native aspects and fixed Task/Physics diagnostic proxies.

VideoScore produces five independent 1-4 regression outputs and defines no
official scalar aggregate. The mappings below are fixed semantic proxies for
the current Ego2Act rubrics; they are not upstream VideoScore scores and are
never fitted to Ego2Act annotations.
"""
from __future__ import annotations

from statistics import fmean


# Positional order of the five upstream regression heads. method.py imports
# this tuple so scoring and reporting cannot silently disagree.
ASPECTS = (
    "visual_quality",
    "temporal_consistency",
    "dynamic_degree",
    "text_to_video_alignment",
    "factual_consistency",
)

RUBRIC_PROXY_ASPECTS = {
    "task": ("text_to_video_alignment",),
    "physics": ("temporal_consistency", "factual_consistency"),
}

EXCLUDED_FROM_RUBRIC_PROXY = ("visual_quality", "dynamic_degree")


def rubric_proxy_scores(prediction: dict[str, float]) -> dict[str, float]:
    """Return fixed proxies on the current Task, Physics and unified scales."""
    missing = set(ASPECTS) - prediction.keys()
    if missing:
        raise ValueError(f"Missing VideoScore aspects: {sorted(missing)}")
    task_native = float(prediction["text_to_video_alignment"])
    physics_native = fmean(
        float(prediction[aspect]) for aspect in RUBRIC_PROXY_ASPECTS["physics"]
    )
    task = task_native - 1.0
    physics = (physics_native - 1.0) * 4.0 / 3.0
    return {
        "task_proxy_0_to_3": round(task, 6),
        "physics_proxy_0_to_4": round(physics, 6),
        "unified_proxy_0_to_1": round(0.5 * (task / 3.0 + physics / 4.0), 6),
    }
