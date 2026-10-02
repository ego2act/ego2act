"""Controlled Ego2Act wording for RBench Common Manipulation."""

from __future__ import annotations

import hashlib


PROMPT_VERSION = "ego2act-human-manipulation-v1"


def create_prompt(goal: str, view: str = "first-person") -> str:
    """Keep RBench's five native 1-5 aspects without robot-only assumptions."""
    return f"""
You are shown a 3×2 grid of six frames in chronological order (read row by
row). The frames come from a {view} video of a person interacting with objects.

Intended goal: {goal}

Identify the hand and the goal-relevant objects from the goal and visible
frames. Evaluate the complete observed manipulation, without assuming that an
object is present or an action succeeded merely because the goal says so.

Score each aspect from 1 to 5. Be strict: 1 means failed, absent, or severely
implausible; 3 means broadly successful with visible problems; 5 means fully
successful and natural.

1. action_execution
   Did the visible hand perform the required action with coherent motion,
   contact, and ordering?
2. task_completion
   Does the observed final state satisfy the intended goal, including required
   intermediate state changes?
3. object_consistency
   Do the goal-relevant objects retain coherent identity, shape, count, and
   state across the frames?
4. hand_consistency
   Does the visible hand/body remain anatomically and temporally coherent,
   without duplication, disappearance, or impossible articulation?
5. physical_plausibility
   Are contact, support, gravity, containment, deformation, and object motion
   physically plausible, without penetration, floating, teleportation, or
   non-contact attachment?

Use the original RBench final-score convention: if action_execution or
task_completion is 1, total is 1; otherwise total is the mean of all five
aspect scores. Return strict JSON only, with this schema:
{{
  "action_execution": {{"reason": "...", "score": 1}},
  "task_completion": {{"reason": "...", "score": 1}},
  "object_consistency": {{"reason": "...", "score": 1}},
  "hand_consistency": {{"reason": "...", "score": 1}},
  "physical_plausibility": {{"reason": "...", "score": 1}},
  "total": {{"reason": "...", "score": 1.0}}
}}
""".strip()


def prompt_sha256() -> str:
    template = create_prompt("{goal}", "{view}")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()
