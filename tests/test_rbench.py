import pytest

from baselines.rbench.openrouter_entry import _validate_details
from baselines.rbench.prompt import create_prompt, prompt_sha256
from baselines.rbench.run import _fingerprint


def _details(score=3):
    return {
        field: {"reason": "visible evidence", "score": score}
        for field in (
            "action_execution",
            "task_completion",
            "object_consistency",
            "hand_consistency",
            "physical_plausibility",
            "total",
        )
    }


def test_rbench_prompt_is_human_and_goal_conditioned():
    prompt = create_prompt("Put the cup inside the box.")

    assert "Put the cup inside the box." in prompt
    assert "person interacting with objects" in prompt
    assert "Robotic manipulator" not in prompt
    assert "1 to 5" in prompt
    assert len(prompt_sha256()) == 64


def test_rbench_response_schema_and_fingerprint():
    _validate_details(_details())
    with pytest.raises(ValueError, match="outside 1-5"):
        _validate_details(_details(6))
    assert _fingerprint("model-a", "first-person") != _fingerprint(
        "model-b", "first-person"
    )
