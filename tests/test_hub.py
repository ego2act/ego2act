import sys
import types
import zipfile

from ego2act import hub


def _fake_hub(tmp_path):
    """Two fake snapshots laid out like ego2act-bench and ego2act-vidgen."""
    bench, vidgen = tmp_path / "bench", tmp_path / "vidgen"
    (bench / "cases" / "c1").mkdir(parents=True)
    (bench / "cases" / "c1" / "prompt.txt").write_text("Goal: x\n")
    (bench / "human" / "correct" / "c1").mkdir(parents=True)
    (bench / "human" / "correct" / "c1" / "correct_1.mp4").write_bytes(b"human")
    with zipfile.ZipFile(bench / "prompt_expansion.zip", "w") as zf:
        zf.writestr("c1_expanded1/prompt.txt", "Goal: y\n")
    (vidgen / "videos" / "wan_2_7" / "c1").mkdir(parents=True)
    (vidgen / "videos" / "wan_2_7" / "c1" / "wan_2_7__seed_101.mp4").write_bytes(b"video")
    (vidgen / "videos" / "kling_v3_pro" / "c1").mkdir(parents=True)
    (vidgen / "videos" / "kling_v3_pro" / "c1" / "kling_v3_pro__seed_101.mp4").write_bytes(b"other")
    (vidgen / "traces" / "ego2act_judge").mkdir(parents=True)
    (vidgen / "traces" / "ego2act_judge" / "traces.jsonl").write_text("{}\n")
    (vidgen / "traces" / "ego2act_judge" / "plans.jsonl").write_text("{}\n")
    (vidgen / "traces" / "baselines" / "pqsg").mkdir(parents=True)
    (vidgen / "traces" / "baselines" / "pqsg" / "scores.csv").write_text("a\n")
    return {"org/bench": bench, "org/vidgen": vidgen}


def _patch(monkeypatch, snaps, seen):
    def fake_snapshot_download(**kwargs):
        seen.append(kwargs)
        return str(snaps[kwargs["repo_id"]])
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))


def test_pull_places_every_subset_where_the_commands_expect_it(tmp_path, monkeypatch):
    seen = []
    _patch(monkeypatch, _fake_hub(tmp_path), seen)
    root, csv_root = tmp_path / "data", tmp_path / "csv"
    kw = dict(bench_repo="org/bench", vidgen_repo="org/vidgen", csv_root=csv_root)
    hub.pull("all", root, ["wan_2_7"], **kw)
    assert (root / "c1" / "prompt.txt").read_text() == "Goal: x\n"
    assert (root / "c1" / "correct" / "correct_1.mp4").read_bytes() == b"human"
    assert (root / "c1" / "AI" / "wan_2_7" / "wan_2_7__seed_101.mp4").read_bytes() == b"video"
    assert not (root / "c1" / "AI" / "kling_v3_pro").exists()          # only the requested model
    assert (root / "c1_expanded1" / "prompt.txt").is_file()
    assert {k["repo_id"] for k in seen} == {"org/bench", "org/vidgen"}
    hub.pull("cases", root, **kw)                                      # idempotent
    hub.pull("traces", root, **kw)
    assert (csv_root / "traces.jsonl").is_file() and (csv_root / "judge_plans.jsonl").is_file()
    assert (csv_root / "baselines" / "pqsg" / "scores.csv").is_file()
