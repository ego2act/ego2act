import json
from types import SimpleNamespace

import pytest

from baselines.common import load_samples
from baselines.videoscore.mapping import rubric_proxy_scores
from baselines.videoscore.run import load_scored_records, native_score_report


def _sample_and_scores(tmp_path):
    case = tmp_path / "demo"
    (case / "correct").mkdir(parents=True)
    (case / "start.jpg").write_bytes(b"image")
    (case / "correct" / "correct_1.mp4").write_bytes(b"current-video")
    (case / "metadata.json").write_text(json.dumps({
        "goal": "Place the cup on the table.",
        "correct/correct_1.mp4": {"desc": "Completed."},
    }), encoding="utf-8")
    args = SimpleNamespace(
        data_root=tmp_path,
        case=["demo"],
        case_file=[],
        video=[],
        category=[],
        generator=[],
        exclude_generator=[],
        seed=[],
        limit=None,
    )
    sample = load_samples(args)[0]
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({
        "requested_videos": 1,
        "successful_videos": 1,
        "failed_videos": 0,
        "results": [{
            "id": sample["id"],
            "case": sample["case"],
            "video": sample["video"],
            "category": sample["category"],
            "generator": sample["generator"],
            "sample_fingerprint": sample["sample_fingerprint"],
            "scores": [4.0, 4.0, 1.0, 4.0, 4.0],
        }],
    }), encoding="utf-8")
    return case, scores


def test_videoscore_current_input_and_rubric_mapping(tmp_path):
    _, scores_path = _sample_and_scores(tmp_path)

    _, records, drift = load_scored_records(scores_path, tmp_path, allow_drift=False)

    assert drift == []
    assert records[0]["rubric_proxy_scores"] == {
        "task_proxy_0_to_3": 3.0,
        "physics_proxy_0_to_4": 4.0,
        "unified_proxy_0_to_1": 1.0,
    }
    summary = native_score_report(records)
    assert summary["groups"]["correct"]["native_mean"]["visual_quality"] == 4.0


def test_videoscore_rejects_same_filename_with_new_bytes(tmp_path):
    case, scores_path = _sample_and_scores(tmp_path)
    (case / "correct" / "correct_1.mp4").write_bytes(b"regenerated-video")

    with pytest.raises(RuntimeError, match="fingerprint changed"):
        load_scored_records(scores_path, tmp_path, allow_drift=False)


def test_videoscore_low_proxy_scores_map_to_zero():
    assert rubric_proxy_scores({
        "visual_quality": 4.0,
        "temporal_consistency": 1.0,
        "dynamic_degree": 4.0,
        "text_to_video_alignment": 1.0,
        "factual_consistency": 1.0,
    }) == {
        "task_proxy_0_to_3": 0.0,
        "physics_proxy_0_to_4": 0.0,
        "unified_proxy_0_to_1": 0.0,
    }
