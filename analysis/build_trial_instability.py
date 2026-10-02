#!/usr/bin/env python
"""Per-model trial-instability summary (worst seed, average, best of 3)."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from scoring import final_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/results/plots_data"
OUT = DATA / "trial_instability.csv"
MODELS = ["grok_imagine_video_1_5", "seedance_2_0", "kling_v3_pro", "wan_2_7", "minimax_h3", "cosmos_3"]
LABELS = {"grok_imagine_video_1_5": "Grok-1.5", "seedance_2_0": "Seedance-2.0", "kling_v3_pro": "Kling-v3-Pro", "wan_2_7": "Wan-2.7", "minimax_h3": "MiniMax-H3", "cosmos_3": "Cosmos-3"}


def num(value):
    return None if value in ("", None) else float(value)


def summarize(path, score_field, evaluator):
    grouped = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("model") not in MODELS:
                continue
            task_field = "human_task" if score_field == "human_final_score" else "ego2act_task"
            physics_field = "human_physics" if score_field == "human_final_score" else "ego2act_physics"
            value = final_score(num(row.get(task_field)), num(row.get(physics_field)), num(row.get(score_field)))
            if value is None:
                continue
            grouped[(row["model"], row["case_id"])].append(value)
    rows = []
    for model in MODELS:
        cases = [values for (m, _), values in grouped.items() if m == model]
        if not cases:
            continue
        rows.append({
            "evaluator": evaluator, "model": model, "label": LABELS[model], "cases": len(cases),
            "average": np.mean([v for values in cases for v in values]),
            "worst_seed": np.mean([min(values) for values in cases]),
            "best_at_3": np.mean([max(sorted(values, reverse=True)[:3]) for values in cases]),
        })
    return rows


def main():
    rows = summarize(ROOT / "analysis/csv/scores_human.csv", "human_final_score", "Human")
    rows += summarize(ROOT / "analysis/csv/scores_ego2act.csv", "ego2act_score", "Ego2ActJudge")
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["evaluator", "model", "label", "cases", "worst_seed", "average", "best_at_3"])
        writer.writeheader(); writer.writerows(rows)
    print({"rows": len(rows), "csv": str(OUT)})


if __name__ == "__main__": main()
