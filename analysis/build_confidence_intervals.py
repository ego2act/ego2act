#!/usr/bin/env python
"""Case-bootstrap 95% intervals for the leaderboard Task and Physics means.

Uses the same videos as leaderboard.py. Each of 1,000 resamples draws cases
with replacement and keeps all videos of every drawn case; the interval is
the 2.5th-97.5th percentile of the resampled video-level means.

Outputs analysis/results/confidence_intervals.csv and analysis/results/confidence_intervals.tex
(Task and Physics table rows in the paper's \\ci{low}{high} format).
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict

from leaderboard import BASELINES, DATA, GROUPS, LABELS, OUT, group_of, read

RESAMPLES = 1000
SEED = 0
EVALUATORS = ["human", *BASELINES, "ego2act"]


def number(value):
    return None if value in ("", None) else float(value)


def by_case() -> dict:
    """{(evaluator, group, axis): {case_id: [scores]}}"""
    data = defaultdict(lambda: defaultdict(list))

    def put(evaluator, group, case, axis, value):
        if value is not None:
            data[evaluator, group, axis][case].append(value)

    for row in read(DATA / "scores_human.csv"):
        group = group_of(row["model"], row["source_type"])
        put("human", group, row["case_id"], "task", number(row["human_task"]))
        put("human", group, row["case_id"], "physics", number(row["human_physics"]))
    for row in read(DATA / "scores_baselines.csv"):
        for axis in ("task", "physics"):
            put(row["evaluator"], row["model"], row["case_id"], axis, number(row[axis]))
    for row in read(DATA / "scores_ego2act.csv"):
        put("ego2act", row["model"], row["case_id"], "task", number(row["ego2act_task"]))
        put("ego2act", row["model"], row["case_id"], "physics", number(row["ego2act_physics"]))
    return data


def bootstrap(cases: dict, rng: random.Random):
    keys = sorted(cases)
    values = [v for k in keys for v in cases[k]]
    estimate = sum(values) / len(values)
    means = []
    for _ in range(RESAMPLES):
        drawn = [v for k in rng.choices(keys, k=len(keys)) for v in cases[k]]
        means.append(sum(drawn) / len(drawn))
    means.sort()
    return estimate, means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1], len(values), len(keys)


def main():
    rng = random.Random(SEED)
    data = by_case()
    rows = []
    for evaluator in EVALUATORS:
        for group in GROUPS:
            for axis in ("task", "physics"):
                cases = data.get((evaluator, group, axis))
                if evaluator == "human" and group == "human_reference":
                    rows.append({"evaluator": evaluator, "group": group, "axis": axis, "mean": 100.0,
                                 "low": None, "high": None, "videos": 0, "cases": 0})
                elif cases:
                    mean, low, high, videos, n_cases = bootstrap(cases, rng)
                    rows.append({"evaluator": evaluator, "group": group, "axis": axis, "mean": mean,
                                 "low": low, "high": high, "videos": videos, "cases": n_cases})
    OUT.mkdir(exist_ok=True)
    with (OUT / "confidence_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = {(r["evaluator"], r["group"], r["axis"]): r for r in rows}
    lines = []
    for axis in ("task", "physics"):
        lines.append(f"% {axis.capitalize()} rows")
        for group in GROUPS:
            if group == "grok_imagine_video_1_5":
                lines.append(r"\midrule")
            cells = []
            for evaluator in EVALUATORS:
                r = table.get((evaluator, group, axis))
                if r is None:
                    cells.append("--")
                elif r["low"] is None:
                    cells.append(f"{r['mean']:.1f}")
                else:
                    cells.append(rf"{r['mean']:.1f}\ci{{{r['low']:.1f}}}{{{r['high']:.1f}}}")
            label = LABELS[group].replace(r"\color{black!55}", "")
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    (OUT / "confidence_intervals.tex").write_text("\n".join(lines) + "\n")
    print({"rows": len(rows), "csv": str(OUT / "confidence_intervals.csv")})


if __name__ == "__main__":
    main()
