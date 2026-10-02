"""Appendix prompt listings: judge_prompts.tex and baseline_prompts.tex.

Reads the prompt modules (ego2act/judge/prompts.py, baselines/simple_vqa/method.py,
baselines/rbench/prompt.py) and writes analysis/tables/judge_prompts.tex and
analysis/tables/baseline_prompts.tex. No API calls. Typographic characters
(en dash, multiplication sign) are mapped to ASCII, as in the paper listings.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "tables"

from ego2act.judge import prompts as J  # noqa: E402
from baselines.rbench.prompt import create_prompt as rbench_prompt  # noqa: E402
from baselines.simple_vqa import method as SV  # noqa: E402

ASCII = {"\u2013": "-", "\u2014": "-", "\u00d7": "x"}


def ascii_text(text: str) -> str:
    for src, dst in ASCII.items():
        text = text.replace(src, dst)
    return text


def listing(title: str, body: str) -> str:
    body = ascii_text(body).rstrip("\n")
    return f"\\paragraph{{{title}}}\n\\begin{{lstlisting}}[style=promptstyle]\n{body}\n\\end{{lstlisting}}\n"


def judge_prompts() -> str:
    gate = lambda q: J.TASK_QUESTIONS[q] + J.TASK_COMMON + J.TASK_EXAMPLES[q]
    parts = [
        "% generated from ego2act/judge/prompts.py; do not edit by hand\n"
        "\\subsection{\\judge{} Prompts}\n"
        "\\label{app:judge-prompts}\n"
        "We reproduce the prompts verbatim. Braced fields are filled at run time, and every call also "
        "receives the response protocol at the end of this subsection together with a JSON schema for the "
        "stage's function call.\n",
        listing("Task subgoal decomposition (from $G$ and $f_1$).", J.TASK_REQUIREMENTS),
        listing("Physics window plan (from $G$ and $f_1$).", J.PLAN),
        listing("Physics dependency review.", J.DEPENDENCIES),
        listing("Task gate T1.", gate("q1")),
        listing("Task gate T2.", gate("q2")),
        listing("Task gate T3.", gate("q3")),
        listing("Physics gate template (\\{gate\\_question\\} is one of P1--P4 below).", J.PHYSICS_GATE),
        listing("Physics gate question P1.", J.P1),
        listing("Physics gate question P2.", J.P2),
        listing("Physics gate question P3.", J.P3),
        listing("Physics gate question P4.", J.P4),
        listing("Response protocol (appended to every gate prompt).", J.RESPONSE),
    ]
    return "\n".join(parts)


def baseline_prompts() -> str:
    source = (ROOT / "baselines" / "simple_vqa" / "method.py").read_text(encoding="utf-8")
    system = re.search(r'"role":"system","content":"([^"]+)"', source).group(1)
    simple = f"System: {system}\n\nUser: {SV.PROMPT}\n\nGoal: <goal G>   [followed by the full video]"
    rbench = rbench_prompt("<goal G>", "first-person")
    return ("% generated from baselines/simple_vqa/method.py and baselines/rbench/prompt.py\n"
            + listing("Simple VQA prompt.", simple)
            + listing("RBench first-person prompt (replaces the robot-specific wording).", rbench))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "judge_prompts.tex").write_text(judge_prompts(), encoding="utf-8")
    (OUT / "baseline_prompts.tex").write_text(baseline_prompts(), encoding="utf-8")


if __name__ == "__main__":
    main()
