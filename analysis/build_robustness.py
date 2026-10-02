#!/usr/bin/env python
"""Robustness checks for the evaluator and benchmark claims.

(a) Final-score alignment with the human consensus for every evaluator after
    removing Cosmos-3, whose very low scores widen the score range, and
    Ego2ActJudge alignment within each generator.
(b) Task horizon: Spearman correlation between each case's median successful
    human-recording duration and its mean Ego2ActJudge score over generated
    videos, across all cases with metadata, plus mean Task scores for cases
    that humans complete within the 15 s generation budget and for longer ones.

Outputs analysis/results/robustness.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import build_human_alignment as alignment
from scoring import final_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE.parent / "analysis" / "csv"
BUDGET_SECONDS = 15
CASE_METADATA = json.loads((DATA / "case_metadata.json").read_text())


def number(value):
    return None if value in ("", None) else float(value)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def without_cosmos() -> dict:
    human = {r["id"]: r for r in alignment.read(DATA / "scores_human.csv")}
    scores = alignment.evaluator_scores()
    out = {}
    for label, keep in (("all", lambda m: True), ("without_cosmos_3", lambda m: m != "cosmos_3")):
        out[label] = {}
        for evaluator in alignment.EVALUATORS:
            axis = "task" if evaluator == "wr-arena" else "final"
            pairs = []
            for video, s in scores[evaluator].items():
                h = human.get(video)
                if h is None or not keep(h["model"]):
                    continue
                target = (number(h["human_task"]) if axis == "task" else
                          final_score(number(h["human_task"]), number(h["human_physics"]),
                                      number(h["human_final_score"])))
                if target is not None and s[axis] is not None:
                    pairs.append((target, s[axis]))
            out[label][evaluator] = alignment.metrics(pairs, with_tau=False)
    within = {}
    for model in sorted({h["model"] for h in human.values() if h["source_type"] == "ai"}):
        pairs = []
        for video, s in scores["ego2act"].items():
            h = human.get(video)
            if h is None or h["model"] != model:
                continue
            target = final_score(number(h["human_task"]), number(h["human_physics"]), number(h["human_final_score"]))
            if target is not None and s["final"] is not None:
                pairs.append((target, s["final"]))
        within[model] = alignment.metrics(pairs, with_tau=False)
    out["ego2act_within_generator"] = within
    return out


def ranking_agreement() -> dict:
    """Model ranking by mean Final score, human vs Ego2ActJudge, on the same videos."""
    human = {r["id"]: r for r in alignment.read(DATA / "scores_human.csv")}
    judge = {r["video_id"]: r for r in alignment.read(DATA / "scores_ego2act.csv")}
    means = {"human": {}, "judge": {}}
    for model in sorted({h["model"] for h in human.values() if h["source_type"] == "ai"}):
        pairs = []
        for video, h in human.items():
            if h["model"] != model or video not in judge:
                continue
            e = judge[video]
            hs = final_score(number(h["human_task"]), number(h["human_physics"]), number(h["human_final_score"]))
            js = final_score(number(e["ego2act_task"]), number(e["ego2act_physics"]), number(e["ego2act_score"]))
            if hs is not None and js is not None:
                pairs.append((hs, js))
        means["human"][model] = mean(p[0] for p in pairs)
        means["judge"][model] = mean(p[1] for p in pairs)
    models = sorted(means["human"])
    rh = ranks([-means["human"][m] for m in models])
    rj = ranks([-means["judge"][m] for m in models])
    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    concordant = sum(1 for i, j in pairs if (rh[i] - rh[j]) * (rj[i] - rj[j]) > 0)
    return {"means": means, "spearman": spearman(rh, rj),
            "kendall": (2 * concordant - len(pairs)) / len(pairs)}


def horizon() -> dict:
    with (ROOT / "analysis/csv/metadata.csv").open(newline="") as stream:
        cases = [row["case_id"] for row in csv.DictReader(stream)]
    duration = {}
    for case in cases:
        meta = CASE_METADATA[case]
        values = [v["duration"] for k, v in meta.items()
                  if k.startswith("correct/") and isinstance(v, dict) and isinstance(v.get("duration"), (int, float))]
        if values:
            duration[case] = median(values)
    task, final = defaultdict(list), defaultdict(list)
    for row in alignment.read(DATA / "scores_ego2act.csv"):
        if row["source_type"] != "ai" or row["case_id"] not in duration:
            continue
        t = number(row["ego2act_task"])
        f = final_score(t, number(row["ego2act_physics"]), number(row["ego2act_score"]))
        if t is not None:
            task[row["case_id"]].append(t)
        if f is not None:
            final[row["case_id"]].append(f)
    keys = sorted(k for k in duration if task[k] and final[k])
    within = [k for k in keys if duration[k] <= BUDGET_SECONDS]
    longer = [k for k in keys if duration[k] > BUDGET_SECONDS]
    return {
        "cases": len(keys),
        "spearman_duration_vs_task": spearman([duration[k] for k in keys], [mean(task[k]) for k in keys]),
        "spearman_duration_vs_final": spearman([duration[k] for k in keys], [mean(final[k]) for k in keys]),
        "within_budget": {"cases": len(within), "mean_task": mean(mean(task[k]) for k in within)},
        "beyond_budget": {"cases": len(longer), "mean_task": mean(mean(task[k]) for k in longer)},
    }


def main():
    result = {"alignment": without_cosmos(), "horizon": horizon(), "ranking": ranking_agreement()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/robustness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["horizon"], indent=2))
    for label in ("all", "without_cosmos_3"):
        print(label, {k: round(v["pearson"], 2) for k, v in result["alignment"][label].items()})
    print("ranking", {k: round(result["ranking"][k], 3) for k in ("spearman", "kendall")})
    print("within generator", {k: round(v["pearson"], 2) for k, v in result["alignment"]["ego2act_within_generator"].items()})


if __name__ == "__main__":
    main()
