import math
import pytest
from baselines.simple_vqa import method


def _client(arguments):
    class Completion:
        usage = type("Usage", (), {"prompt_tokens": 10, "completion_tokens": 2,
                                    "total_tokens": 12, "cost": .01})()
        choices = [type("Choice", (), {"message": type("Message", (), {
            "tool_calls": [type("Call", (), {"function": type("Function", (), {
                "name": "emit_simple_vqa", "arguments": arguments})()})]
        })()})]
    return type("Client", (), {"chat": type("Chat", (), {"completions": type("Completions", (), {
        "create": lambda *args, **kwargs: Completion()
    })()})()})()


def test_one_shot_prompt_contains_rubric_without_gate_protocol():
    assert "TASK COMPLETION" in method.PROMPT
    assert "PHYSICAL PLAUSIBILITY" in method.PROMPT
    assert "licensed" in method.PROMPT
    assert "Physics null" in method.PROMPT
    assert "Q1" not in method.PROMPT and "P1" not in method.PROMPT


def test_one_shot_uses_geometric_composite():
    client = _client('{"task":{"score":1.5,"reasoning":"Task.","evidence":[]},"physics":{"score":2.0,"reasoning":"Physics.","evidence":[]}}')
    scores, usage = method.score(client, video_data_url="data:video/mp4;base64,AA==", goal="Move it.", model="test")
    assert scores["unified_0_to_1"] == pytest.approx(math.sqrt((1.5 / 3) * (2 / 4)))
    assert usage["model_calls"] == 1


def test_one_shot_physics_na_propagates():
    client = _client('{"task":{"score":3,"reasoning":"Task.","evidence":[]},"physics":{"score":null,"reasoning":"No movement.","evidence":[]}}')
    scores, _ = method.score(client, video_data_url="data:video/mp4;base64,AA==", goal="Move it.", model="test")
    assert scores["physics_0_to_4"] is None
    assert scores["unified_0_to_1"] is None
