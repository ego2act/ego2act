import json
from types import SimpleNamespace

from baselines.common import load_samples


def _write_case(root, case_id, byte):
    case = root / case_id
    (case / "correct").mkdir(parents=True)
    (case / "AI" / "wan_2_7").mkdir(parents=True)
    (case / "start.jpg").write_bytes(b"image")
    (case / "correct" / "correct_1.mp4").write_bytes(byte + b"human")
    (case / "AI" / "wan_2_7" / "wan_2_7.seed_101.mp4").write_bytes(
        byte + b"generated"
    )
    (case / "metadata.json").write_text(json.dumps({
        "goal": f"Complete {case_id}.",
        "action_types": ["place"],
        f"correct/correct_1.mp4": {"desc": "Complete."},
    }), encoding="utf-8")


def test_load_samples_preserves_case_file_order_and_filters_generator(tmp_path):
    _write_case(tmp_path, "first", b"1")
    _write_case(tmp_path, "second", b"2")
    cases = tmp_path / "cases.txt"
    cases.write_text("second  # intentional order\n\nfirst\n", encoding="utf-8")
    args = SimpleNamespace(
        data_root=tmp_path,
        case=[],
        case_file=[cases],
        video=[],
        category=["ai"],
        generator=["wan_2_7"],
        exclude_generator=[],
        seed=[101],
        limit=None,
    )

    samples = load_samples(args)

    assert [sample["case"] for sample in samples] == ["second", "first"]
    assert all(sample["category"] == "ai" for sample in samples)
    assert all(sample["generator"] == "wan_2_7" for sample in samples)
    assert all(sample["sample_fingerprint"] for sample in samples)


def test_exclude_generator_retains_human_controls(tmp_path):
    _write_case(tmp_path, "first", b"1")
    args = SimpleNamespace(
        data_root=tmp_path,
        case=[],
        case_file=[],
        video=[],
        category=[],
        generator=[],
        exclude_generator=["wan_2_7"],
        seed=[],
        limit=None,
    )

    samples = load_samples(args)

    assert [sample["video"] for sample in samples] == ["correct_1"]


def test_goal_falls_back_to_prompt_txt(tmp_path):
    from ego2act.data import case_metadata
    case = tmp_path / "c"
    case.mkdir()
    assert case_metadata(case) is None
    (case / "prompt.txt").write_text("Scene.\nGoal: Lid open.\n", encoding="utf-8")
    assert case_metadata(case) == {"goal": "Lid open."}
    (case / "metadata.json").write_text('{"goal": "From metadata"}', encoding="utf-8")
    assert case_metadata(case)["goal"] == "From metadata"
