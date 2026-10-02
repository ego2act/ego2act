import json

from baselines.worldmodelbench.protocol import (
    affine_1_to_4,
    enrich_record,
    native_score_report,
    render_report,
    rubric_proxy_report,
    rubric_proxy_scores,
)
from baselines.worldmodelbench.run import (
    _absolute_without_resolving,
    _build_records,
    _input_sample,
)
from baselines.worldmodelbench.upstream_entry import parse_instruction, parse_pass


def upstream_record(
    sample_id="case::video",
    instruction=3,
    physical=5,
    common=2,
    fingerprint="fingerprint",
):
    return {
        "id": sample_id,
        "sample_fingerprint": fingerprint,
        "instruction": {"score": instruction, "prompt": "p", "response": "Score: 3"},
        "physical_laws": [
            {"pass": index < physical, "question": str(index), "prompt": "p", "response": "No"}
            for index in range(5)
        ],
        "common_sense": [
            {"pass": index < common, "question": str(index), "prompt": "p", "response": "No"}
            for index in range(2)
        ],
        "native_scores": {
            "instruction_0_to_3": float(instruction),
            "physical_laws_0_to_5": physical,
            "common_sense_0_to_2": common,
            "official_total_0_to_10": float(instruction + physical + common),
        },
    }


def sample(sample_id="case::video", category="correct", gt=None):
    return {
        "id": sample_id,
        "case": "case",
        "video": "video",
        "category": category,
        "prompt": "put the cup down",
        "objects": ["cup"],
        "video_path": "/tmp/video.mp4",
        "video_sha256": "video-hash",
        "prompt_sha256": "prompt-hash",
        "sample_fingerprint": "fingerprint",
        "gt": gt or [4.0, 4.0, 4.0],
        "human_final_1_to_4": 4.0,
        "description": "must never reach the judge",
    }


def test_official_parser_is_preserved():
    assert parse_instruction("analysis. Score: 2.") == (2.0, None)
    assert parse_instruction("unparseable")[0] == 0.0
    assert parse_pass("No, it does not.") is True
    assert parse_pass("Yes, it does.") is False


def test_native_total_and_fixed_affine_scale_are_retained():
    record = enrich_record(upstream_record(instruction=2, physical=4, common=1))
    assert record["native_scores"]["official_total_0_to_10"] == 7.0
    assert record["affine_scores_1_to_4"]["official_total"] == affine_1_to_4(7, 10)
    assert record["affine_scores_1_to_4"]["instruction"] == 3.0


def test_judge_input_excludes_labels_and_failure_metadata():
    visible = _input_sample(sample())
    assert set(visible) == {
        "id",
        "instruction",
        "video_path",
        "video_sha256",
        "prompt_sha256",
        "sample_fingerprint",
    }
    serialized = json.dumps(visible)
    assert "description" not in serialized
    assert "correct" not in serialized
    assert "gt" not in serialized


def test_virtualenv_interpreter_symlink_is_not_dereferenced(tmp_path):
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to("/usr/bin/python3")
    assert _absolute_without_resolving(venv_python) == venv_python
    assert _absolute_without_resolving(venv_python) != venv_python.resolve()


def test_adapter_builds_native_report_and_current_rubric_proxies():
    samples = [
        sample("case::correct", "correct", [4.0, 4.0, 4.0]),
        sample("case::wrong", "wrong", [1.0, 1.0, 1.0]),
    ]
    raw = {
        "records": [
            upstream_record("case::correct", 3, 5, 2),
            upstream_record("case::wrong", 0, 0, 0),
        ]
    }
    records = _build_records(samples, raw)
    native = native_score_report(records)
    assert native["groups"]["all"]["mean"]["official_total_0_to_10"] == 5.0
    assert native["groups"]["correct_minus_wrong"]["official_total_0_to_10"] == 10.0
    assert records[0]["rubric_proxy_scores"] == {
        "task_proxy_0_to_3": 3.0,
        "physics_proxy_0_to_4": 4.0,
        "unified_proxy_0_to_1": 1.0,
    }
    partial = rubric_proxy_scores(enrich_record(upstream_record(instruction=0, physical=0, common=0)))
    assert partial["unified_proxy_0_to_1"] == 0.0
    proxies = rubric_proxy_report(records)
    assert proxies["groups"]["correct"]["mean"]["unified_proxy_0_to_1"] == 1.0


def test_adapter_rejects_stale_same_name_result():
    current = sample()
    raw = {"records": [upstream_record(fingerprint="old-fingerprint")]}
    records = _build_records([current], raw)
    assert records[0]["error"] == "stale upstream output fingerprint"


def test_readable_report_includes_original_and_rubric_alignment():
    records = [{**sample(), **enrich_record(upstream_record())}]
    payload = {
        "successful_videos": 1,
        "requested_videos": 1,
        "execution": "test",
        "judge": {"repository": "judge", "revision": "rev"},
        "official_commit": "official",
        "vila_commit": "vila",
        "runtime_lock_sha256": "lock",
        "tracked_trees_clean": {"official": True},
        "source_integrity": {"ok": True},
        "artifacts": {"upstream_output": "raw", "manifest": "manifest", "upstream_input": "input"},
        "native_score_report": native_score_report(records),
    }
    text = render_report(payload)
    assert "Original WorldModelBench scores" in text
    assert "Official total (0-10)" in text
    assert "Task/Physics alignment" in text
