import json

from baselines.compile_native_results import BENCHMARKS, compile_results, render_markdown


def test_compile_keeps_each_native_bracket(tmp_path):
    records = {
        "pqsg": {"native_score_0_to_1": 0.5},
        "wr_arena": {"native_score_0_to_3": 2},
        "rbench": {"native_score_1_to_5": 3.5},
        "worldmodelbench": {"native_scores": {
            "instruction_0_to_3": 2,
            "physical_laws_0_to_5": 4,
            "common_sense_0_to_2": 1,
            "official_total_0_to_10": 7,
        }},
        "videoscore": {"prediction": {
            "visual_quality": 3,
            "temporal_consistency": 3,
            "dynamic_degree": 2,
            "text_to_video_alignment": 4,
            "factual_consistency": 3,
        }},
    }
    for name, spec in BENCHMARKS.items():
        path = tmp_path / spec["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "requested_videos": 1,
            "records": [{
                "id": "case::video",
                "category": "ai",
                "generator": "model",
                **records[name],
            }],
        }), encoding="utf-8")

    compiled = compile_results(tmp_path)

    assert compiled["sample_set_union"] == 1
    assert compiled["sample_set_differences"] == {}
    assert compiled["benchmarks"]["pqsg"]["native_components"]["final"]["range"] == [0.0, 1.0]
    assert compiled["benchmarks"]["videoscore"]["native_components"]["visual_quality"]["range"] == [1.0, 4.0]
    assert "cross-benchmark averaging" in compiled["score_policy"]
    assert "Native range" in render_markdown(compiled)
