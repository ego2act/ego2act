import json

from baselines.pqsg.run import _prepare_resume


def test_pqsg_resume_reorders_only_fingerprint_matching_records(tmp_path):
    output = tmp_path / "upstream_output.json"
    output.write_text(json.dumps([
        {"id": "b", "sample_fingerprint": "fingerprint-b", "answers": {"x": 1}},
        {"id": "a", "sample_fingerprint": "stale-a", "answers": {"x": 1}},
    ]), encoding="utf-8")
    samples = [
        {"id": "a", "sample_fingerprint": "fingerprint-a"},
        {"id": "b", "sample_fingerprint": "fingerprint-b"},
    ]

    _prepare_resume(output, samples, enabled=True)

    resumed = json.loads(output.read_text(encoding="utf-8"))
    assert resumed[0] == {}
    assert resumed[1]["id"] == "b"


def test_pqsg_without_resume_removes_position_based_native_cache(tmp_path):
    output = tmp_path / "upstream_output.json"
    output.write_text("[]", encoding="utf-8")

    _prepare_resume(output, [], enabled=False)

    assert not output.exists()
