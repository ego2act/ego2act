#!/usr/bin/env python
"""Build the current matched human/Ego2Act Task--Physics scatter slice."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "analysis/csv/scores_human.csv"
target = ROOT / "analysis/results/plots_data/task_physics_scatter.csv"
fields = ["id", "case_id", "model", "seed", "human_task", "human_physics", "human_final", "judge_task", "judge_physics", "judge_final"]
with source.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
matched = [
    {**{key: row[key] for key in fields if key in row},
     "human_final": row.get("human_final_score", ""),
     "judge_task": row["ego2act_task"], "judge_physics": row["ego2act_physics"],
     "judge_final": row.get("ego2act_score", "")}
    for row in rows
    if all(row.get(key, "") not in ("", None) for key in ("human_task", "human_physics", "ego2act_task", "ego2act_physics"))
]
with target.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(matched)
print({"rows": len(matched), "output": str(target)})
