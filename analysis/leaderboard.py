#!/usr/bin/env python
"""Cross-benchmark leaderboard (Table 2): human panel, baselines and Ego2ActJudge,
each over every video it scored.

Outputs: analysis/results/leaderboard.csv and leaderboard.tex.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from scoring import final_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
OUT = HERE / "results"
GROUPS = ["human_reference", "human_wrong", "grok_imagine_video_1_5", "seedance_2_0",
          "kling_v3_pro", "wan_2_7", "minimax_h3", "cosmos_3"]
MODELS = GROUPS[2:]
LABELS = {"human_reference": r"\textit{Human (+)}", "human_wrong": r"\textit{Human ($-$)}",
          "grok_imagine_video_1_5": "Grok-1.5", "seedance_2_0": "Seedance-2.0",
          "kling_v3_pro": "Kling-v3-Pro", "wan_2_7": "Wan-2.7", "minimax_h3": "MiniMax-H3",
          "cosmos_3": "Cosmos-3"}
BASELINES = ["wr-arena", "pqsg", "rbench", "worldmodelbench", "videoscore"]


def read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def num(value):
    return None if value in ("", None) else float(value)


def mean(values):
    values = [float(v) for v in values if v not in ("", None)]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def group_of(model: str, source_type: str) -> str:
    return {"ground_truth_reference": "human_reference"}.get(source_type, model if source_type == "ai" else source_type)


def build() -> list[dict]:
    rows = []

    def add(evaluator, group, task, physics, overall, kind):
        rows.append({"evaluator": evaluator, "group": group,
                     "task": task[0], "physics": physics[0], "overall": overall,
                     "overall_kind": kind, "n_task": task[1], "n_physics": physics[1]})

    human = defaultdict(list)
    for row in read(DATA / "scores_human.csv"):
        human[group_of(row["model"], row["source_type"])].append(row)
    for group in GROUPS:
        if group == "human_reference":
            add("human", group, (100.0, 0), (100.0, 0), 100.0, "verified_successful")
            continue
        task = mean(r["human_task"] for r in human[group])
        physics = mean(r["human_physics"] for r in human[group])
        finals = [final_score(num(r["human_task"]), num(r["human_physics"]), num(r["human_final_score"]))
                  for r in human[group]]
        add("human", group, task, physics, mean(finals)[0], "mean_of_per_video_final")

    baselines = defaultdict(lambda: defaultdict(list))
    for row in read(DATA / "scores_baselines.csv"):
        baselines[row["evaluator"]][row["model"]].append(row)
    for evaluator in BASELINES:
        for group in GROUPS:
            records = baselines[evaluator][group]
            add(evaluator, group, mean(r["task"] for r in records), mean(r["physics"] for r in records),
                mean(r["native_0_100"] for r in records)[0], "native_mean")

    judge = defaultdict(list)
    for row in read(DATA / "scores_ego2act.csv"):
        judge[row["model"]].append(row)
    for group in GROUPS:
        records = judge[group]
        add("ego2act", group, mean(r["ego2act_task"] for r in records),
            mean(r["ego2act_physics"] for r in records),
            mean(final_score(num(r["ego2act_task"]), num(r["ego2act_physics"]), num(r["ego2act_score"]))
                 for r in records)[0], "mean_of_per_video_final")
    return rows


def ranks(values: dict) -> dict:
    """Bold/underline markers for the top two generated models."""
    ordered = sorted({v for g, v in values.items() if g in MODELS and v is not None}, reverse=True)
    marks = {}
    for group, value in values.items():
        if group in MODELS and value is not None:
            if value == ordered[0]:
                marks[group] = "textbf"
            elif len(ordered) > 1 and value == ordered[1]:
                marks[group] = "underline"
    return marks


def cell(value, mark=None) -> str:
    if value is None:
        return "--"
    text = f"{value:.1f}"
    return f"\\{mark}{{{text}}}" if mark else text


def latex(rows: list[dict]) -> str:
    table = {(r["evaluator"], r["group"]): r for r in rows}
    order = ["human", *BASELINES, "ego2act"]
    marked = {("human", "task"), ("human", "physics"), ("human", "overall"),
              ("ego2act", "task"), ("ego2act", "physics"), ("ego2act", "overall"),
              *((e, "overall") for e in BASELINES)}
    marks = {(e, a): ranks({g: table[e, g][a] for g in GROUPS}) for e, a in marked}
    lines = []
    for group in GROUPS:
        if group == "grok_imagine_video_1_5" or group == "minimax_h3":
            lines.append(r"\midrule")
        gray = group in ("human_reference", "human_wrong")
        overall = (lambda text: r"\color{black!55}" + text) if gray else (lambda text: text)
        cells = []
        for evaluator in order:
            r = table[evaluator, group]
            mark = lambda axis: marks.get((evaluator, axis), {}).get(group)
            if evaluator != "wr-arena":
                pair = rf"\evalscore{{{cell(r['task'], mark('task'))}}}{{{cell(r['physics'], mark('physics'))}}}"
                cells.append(pair)
            cells.append(overall(cell(r["overall"], mark("overall"))))
        label = (r"\color{black!55}" if gray else "") + LABELS[group]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main():
    rows = build()
    OUT.mkdir(exist_ok=True)
    with (OUT / "leaderboard.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "leaderboard.tex").write_text(latex(rows))
    print({"rows": len(rows), "csv": str(OUT / "leaderboard.csv"), "tex": str(OUT / "leaderboard.tex")})


if __name__ == "__main__":
    main()
