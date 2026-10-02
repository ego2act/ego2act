"""Shared Final-score rule for the analysis scripts.

Final = sqrt(Task * Physics). Physics is undefined when no subgoal is attempted
(Task = 0 means no interaction to judge). Such a rollout has failed completely,
so its Final score is 0 rather than missing; otherwise it would silently drop
out of every average. A video with Task > 0 whose attempted interactions are
all unobservable keeps a missing Final score.
"""
from __future__ import annotations


def final_score(task, physics, final=None):
    """Final score on 0-100, applying the unattempted-video rule."""
    if final is not None:
        return final
    if task is not None and physics is not None:
        return (task * physics) ** 0.5
    if task is not None and task == 0:
        return 0.0
    return None
