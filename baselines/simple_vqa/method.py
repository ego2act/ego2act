"""One-shot graph-free Task/Physics baseline using the current high-level rubric."""
from __future__ import annotations
import base64, hashlib, json, math
from pathlib import Path
from typing import Any
from baselines.config import get_experiment_config

SCORE_KEYS = ("task_0_to_3", "physics_0_to_4", "unified_0_to_1")
PROMPT = """Evaluate this embodied-action video against the supplied goal in one direct pass.
Do not generate a plan, subgoal list, dependency graph, or gate-by-gate answers.
Do not infer hidden events. Return one structured function call containing the
two final rubric scores and concise observable evidence.

TASK COMPLETION (0–3)
Judge only whether the requested task was attempted and completed. Ignore
visual glitches, clipping, malformed hands, floating, teleportation, jitter,
and impossible physics when the macro-intent and requested outcome remain
recognizable. Use the whole goal, including multiple requested outcomes and
their necessary ordering, but report one holistic score.
- 0: no recognizable task-directed attempt, or behavior targets the wrong
  source object, tool, or mechanism.
- 1: a correct attempt begins, but the characteristic high-level action is
  not substantially carried through end to end.
- 2: the high-level action is substantially performed, but the required final
  object relation, quantity, or state is materially missed or incomplete.
- 3: the requested final state/outcomes are visibly achieved. Allow slight
  imprecision that does not materially change the stated goal.
Judge final-state evidence more strictly than motion appearance. A prerequisite
or ordering failure should reduce Task when it prevents the requested sequence,
but do not invent customary ordering requirements not implied by the goal.

PHYSICAL PLAUSIBILITY (0–4 or null)
Judge independently of Task success. Return null only when no relevant object
or embodied movement can be inspected; a static/no-attempt clip is Task 0 and
Physics null, not Physics 4. Otherwise assess the visible interaction as a
causal physical event:
- 0: object identity, count, colour/material, or traceable continuity/path
  breaks without a licensed visible transformation.
- 1: an object moves or changes without visible contact, support, or another
  valid preceding trigger.
- 2: contact/causality exists, but the interaction or material/mechanism
  behaviour is physically impossible while it happens.
- 3: continuity, causality, and interaction are plausible, but the resulting
  state does not hold or change as visible forces allow.
- 4: no visible violation of continuity, causality, interaction, or persistence.
Allow ordinary occlusion, perspective/shadow changes, brief appearance flicker,
and transformations visibly explained by cutting, joining, consuming, adding,
pouring, spraying, erasing, or revealing. Do not penalize omitted task actions
as Physics failures. Do not invent an unseen process to explain a discontinuity.

For each dimension provide a numeric score (or Physics null), brief reasoning
based on visible observations, and up to three approximate original-source time
windows with evidence. Do not expose private chain-of-thought. Emit one
structured function call only."""

def prompt_sha256() -> dict[str, str]:
    value = hashlib.sha256(PROMPT.encode()).hexdigest()
    return {"combined": value, "task": value, "physics": value}

def video_to_data_url(video_path: str | Path) -> str:
    return "data:video/mp4;base64," + base64.b64encode(Path(video_path).read_bytes()).decode()

def _evidence() -> dict[str, Any]:
    return {"type":"array","maxItems":3,"items":{"type":"object","properties":{
        "start_seconds":{"type":"number","minimum":0},"end_seconds":{"type":"number","minimum":0},
        "observation":{"type":"string","minLength":1}},"required":["start_seconds","end_seconds","observation"],"additionalProperties":False}}

def _tool() -> dict[str, Any]:
    task = {"type":"object","properties":{"score":{"type":"number","minimum":0,"maximum":3},"reasoning":{"type":"string","minLength":1,"maxLength":1200},"evidence":_evidence()},"required":["score","reasoning","evidence"],"additionalProperties":False}
    physics = {"type":"object","properties":{"score":{"anyOf":[{"type":"number","minimum":0,"maximum":4},{"type":"null"}]},"reasoning":{"type":"string","minLength":1,"maxLength":1200},"evidence":_evidence()},"required":["score","reasoning","evidence"],"additionalProperties":False}
    return {"type":"function","function":{"name":"emit_simple_vqa","description":"Return one-shot Task and Physics scores with concise observable reasoning.","parameters":{"type":"object","properties":{"task":task,"physics":physics},"required":["task","physics"],"additionalProperties":False}}}

def _usage(response: Any) -> dict[str, Any]:
    usage = response.usage
    return {"prompt_tokens":int(getattr(usage,"prompt_tokens",0) or 0),"completion_tokens":int(getattr(usage,"completion_tokens",0) or 0),"total_tokens":int(getattr(usage,"total_tokens",0) or 0),"cost":float(getattr(usage,"cost",0.0) or 0.0),"model_calls":1}

def score(client: Any, *, video_data_url: str, goal: str, model: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = get_experiment_config()["baselines"]
    generation, selected = config["decoding"], model or config["model"]
    completion = client.chat.completions.create(model=selected, temperature=float(generation["temperature"]), seed=generation["seed"], max_tokens=generation["max_tokens"], timeout=float(generation["timeout_seconds"]), tools=[_tool()], tool_choice={"type":"function","function":{"name":"emit_simple_vqa"}}, extra_body={"provider":{"require_parameters":generation["require_parameters"]},"reasoning":{"effort":generation["reasoning_effort"]}}, messages=[{"role":"system","content":"You are a concise one-shot video evaluator. Apply the rubric and emit only the function call."},{"role":"user","content":[{"type":"text","text":f"{PROMPT}\n\nGoal: {goal}"},{"type":"video_url","video_url":{"url":video_data_url}}]}])
    calls = completion.choices[0].message.tool_calls or []
    call = next((item for item in calls if item.function.name == "emit_simple_vqa"), None)
    if call is None: raise RuntimeError("Simple VQA did not emit the required structured result")
    result = json.loads(call.function.arguments); task, physics = result["task"]["score"], result["physics"]["score"]
    if isinstance(task,bool) or not isinstance(task,(int,float)) or not 0 <= task <= 3: raise ValueError("Invalid Task score")
    if physics is not None and (isinstance(physics,bool) or not isinstance(physics,(int,float)) or not 0 <= physics <= 4): raise ValueError("Invalid Physics score")
    unified = None if physics is None else math.sqrt((task / 3) * (physics / 4))
    return {"task_0_to_3":float(task),"physics_0_to_4":None if physics is None else float(physics),"unified_0_to_1":unified,"reasoning":result}, _usage(completion)
