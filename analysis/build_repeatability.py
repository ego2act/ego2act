#!/usr/bin/env python
"""Judge repeatability (two runs on the same videos) and generation variability
(seed-to-seed SD per generator).

Inputs: analysis/csv/pilot_scores.json, scores_ego2act.csv.
Output: analysis/results/repeatability.json.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scoring import final_score

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "analysis/csv/pilot_scores.json"


def number(value):
    return None if value in ("", None) else float(value)


def pearson(pairs):
    mx, my = mean(a for a, _ in pairs), mean(b for _, b in pairs)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    return num / math.sqrt(sum((a - mx) ** 2 for a, _ in pairs) * sum((b - my) ** 2 for _, b in pairs))


def judge_repeatability(production):
    pilot = {r["id"]: r for r in json.loads(PILOT.read_text())}
    out = {}
    for axis in ("task", "physics", "final"):
        pairs, by_source = [], defaultdict(list)
        for video, run in pilot.items():
            row = production.get(video)
            if row is None:
                continue
            if axis == "final":
                a = final_score(number(row["ego2act_task"]), number(row["ego2act_physics"]), number(row["ego2act_score"]))
                b = final_score(run.get("judge_task"), run.get("judge_physics"), run.get("judge_score"))
            else:
                a, b = number(row[f"ego2act_{axis}"]), run.get(f"judge_{axis}")
            if a is not None and b is not None:
                pairs.append((a, b))
                by_source[row["source_type"]].append(a - b)
        diffs = [a - b for a, b in pairs]
        sd = lambda d: math.sqrt(sum(x * x for x in d) / len(d) / 2)
        out[axis] = {"videos": len(pairs), "single_run_sd": sd(diffs), "retest_r": pearson(pairs),
                     "mean_abs_diff": mean(abs(x) for x in diffs),
                     "by_source": {k: {"videos": len(v), "single_run_sd": sd(v)} for k, v in by_source.items()}}
    return out


def generation_variability(production):
    seeds = defaultdict(list)
    for row in production.values():
        if row["source_type"] != "ai":
            continue
        s = final_score(number(row["ego2act_task"]), number(row["ego2act_physics"]), number(row["ego2act_score"]))
        if s is not None:
            seeds[row["model"], row["case_id"]].append(s)
    per_model = defaultdict(list)
    for (model, _), values in seeds.items():
        if len(values) >= 2:
            per_model[model].append(stdev(values))
    return {model: {"cases": len(v), "mean_seed_sd": mean(v)} for model, v in per_model.items()}


def main():
    with (HERE.parent / "analysis/csv/scores_ego2act.csv").open(newline="") as stream:
        production = {r["video_id"]: r for r in csv.DictReader(stream)}
    result = {"judge": judge_repeatability(production), "generation": generation_variability(production)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/repeatability.json").write_text(json.dumps(result, indent=2) + "\n")
    for axis, r in result["judge"].items():
        print(axis, r["videos"], round(r["single_run_sd"], 2), round(r["retest_r"], 3),
              {k: round(v["single_run_sd"], 2) for k, v in r["by_source"].items()})
    print({m: round(v["mean_seed_sd"], 2) for m, v in result["generation"].items()})


if __name__ == "__main__":
    main()
