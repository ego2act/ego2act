import sys
from pathlib import Path

from ego2act import cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import reproduce as paper  # noqa: E402


def test_help_lists_every_command(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in cli.COMMANDS)
    assert cli.main(["unknown"]) == 2


def test_verify_compares_numbers_not_formatting():
    assert paper.same("a.json", '{"x": 1.0, "y": [2]}', '{"y": [2], "x": 1.0000000001}')
    assert not paper.same("a.json", '{"x": 1.0}', '{"x": 1.1}')
    assert paper.same("a.tex", "r = 0.5 \\\\", "r = 0.5 \\\\")
    assert not paper.same("a.csv", "m,0.51\n", "m,0.52\n")


def test_simple_vqa_needs_no_upstream_checkout():
    from baselines.runner import UPSTREAM, main
    assert set(UPSTREAM) == {"wr_arena", "pqsg", "rbench", "worldmodelbench", "videoscore", "simple_vqa"}
    assert main(["simple_vqa", "--setup-only"]) == 0
