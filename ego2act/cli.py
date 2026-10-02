"""Ego2Act: generate, judge and run baselines (paper tables: python analysis/reproduce.py).

    ego2act generate         image + goal -> video (six models)
    ego2act judge            video + goal -> Task, Physics, Final (0-100)
    ego2act baseline         run one of the six baseline evaluators
    ego2act data             pull the benchmark from the Hugging Face Hub

Run `ego2act <command> --help` for the options of each command.
"""
from __future__ import annotations

import sys


def _generate(argv):
    from ego2act.generate import main
    return main(argv)


def _judge(argv):
    from ego2act.evaluate import main
    return main(argv)


def _baseline(argv):
    from baselines.runner import main
    return main(argv)


def _data(argv):
    from ego2act.hub import main
    return main(argv)


COMMANDS = {"generate": _generate, "judge": _judge, "baseline": _baseline, "data": _data}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    return COMMANDS[argv[0]](argv[1:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
