#!/usr/bin/env python
"""Judge-side gate ablation: sequential vs independent gate scoring on the
human-rated panel.

Inputs: analysis/csv/scores_gate_ablation.csv, gate_ablation_subgoals.csv, scores_human.csv.
Outputs: analysis/results/gate_ablation.json and gate_ablation.tex.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from build_human_alignment import fmt, metrics, number, read
from scoring import final_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
OUT = HERE / "results"
AXES = ["task", "physics", "final"]
RULES = ["seq", "ind"]
SCORES = DATA / "scores_gate_ablation.csv"
SUBGOALS = DATA / "gate_ablation_subgoals.csv"


# ---------------------------------------------------------------- export

def export(run: Path) -> None:
    """Flatten a run directory (scores/traces/failures.jsonl) into data/ CSVs."""
    human = [r for r in read(DATA / "scores_human.csv") if r["human_task"]]
    lines = lambda name: [json.loads(l) for l in (run / name).read_text().splitlines()
                          if l.strip()] if (run / name).exists() else []
    scores = {s["video_id"]: s for s in lines("scores.jsonl")}
    traces = {t["video_id"]: t for t in lines("traces.jsonl")}
    failures = {f["video_id"]: f["error"] for f in lines("failures.jsonl")}
    rows = []
    for h in human:
        video = h["id"]
        s = scores.get(video)
        row = {"video_id": video, "case_id": h["case_id"], "model": h["model"]}
        if s is None:
            error = failures.get(video)
            status = "not_run" if error is None else ("budget" if error.startswith("budget") else "failed")
            rows.append({**row, **{f"{r}_{a}": "" for r in RULES for a in AXES}, "status": status})
            continue
        for rule in RULES:
            t, p = s[f"{rule}_task"], s[f"{rule}_physics"]
            row.update({f"{rule}_task": t, f"{rule}_physics": p, f"{rule}_final": final_score(t, p)})
        status = "ok" if s["task_status"] == "ok" and s["physics_status"] == "ok" else \
            f"task_{s['task_status']}|physics_{s['physics_status']}"
        rows.append({**row, "status": status})
    fields = ["video_id", "case_id", "model"] + [f"{r}_{a}" for r in RULES for a in AXES] + ["status"]
    with SCORES.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with SUBGOALS.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "axis", "subgoal_id", "seq_level", "ind_level"])
        for video in sorted(traces):
            for axis in ("task", "physics"):
                seq = traces[video][axis].get("levels") or {}
                ind = (traces[video][axis].get("independent") or {}).get("levels") or {}
                for sid in seq:
                    writer.writerow([video, axis, sid, seq[sid], ind.get(sid, "")])


# ---------------------------------------------------------------- analysis

def load():
    human = {r["id"]: {"case": r["case_id"], "task": number(r["human_task"]),
                       "physics": number(r["human_physics"]),
                       "final": final_score(number(r["human_task"]), number(r["human_physics"]),
                                            number(r["human_final_score"]))}
             for r in read(DATA / "scores_human.csv")}
    ablation = {}
    for r in read(SCORES):
        if r["status"] in ("failed", "budget", "not_run"):
            continue
        ablation[r["video_id"]] = {rule: {a: final_score(number(r[f"{rule}_task"]), number(r[f"{rule}_physics"]))
                                          if a == "final" else number(r[f"{rule}_{a}"]) for a in AXES}
                                   for rule in RULES}
    return human, ablation


def triples(human, ablation, axis):
    """(case, human, seq, ind) on videos where all three are defined."""
    return [(human[v]["case"], human[v][axis], s["seq"][axis], s["ind"][axis])
            for v, s in sorted(ablation.items())
            if v in human and human[v][axis] is not None
            and s["seq"][axis] is not None and s["ind"][axis] is not None]


def bootstrap(rows, resamples=1000, seed=0):
    """Case-bootstrap 95% CI for independent minus sequential MAE and r."""
    cases = defaultdict(list)
    for case, h, seq, ind in rows:
        cases[case].append((h, seq, ind))
    keys = sorted(cases)
    rng = random.Random(seed)
    draws = {"mae": [], "pearson": []}
    for _ in range(resamples):
        sample = [x for k in rng.choices(keys, k=len(keys)) for x in cases[k]]
        seq = metrics([(h, s) for h, s, _ in sample], with_tau=False)
        ind = metrics([(h, i) for h, _, i in sample], with_tau=False)
        for name in draws:
            if seq[name] is not None and ind[name] is not None:
                draws[name].append(ind[name] - seq[name])
    point_seq = metrics([(h, s) for _, h, s, _ in rows], with_tau=False)
    point_ind = metrics([(h, i) for _, h, _, i in rows], with_tau=False)
    out = {}
    for name, values in draws.items():
        values.sort()
        out[name] = {"difference": point_ind[name] - point_seq[name],
                     "low": values[int(0.025 * len(values))],
                     "high": values[int(0.975 * len(values)) - 1],
                     "resamples": len(values), "cases": len(keys)}
    return out


def subgoal_disagreement():
    counts = defaultdict(lambda: {"subgoals": 0, "ind_gt_seq": 0, "ind_lt_seq": 0, "videos": set(),
                                  "videos_differ": set()})
    for r in read(SUBGOALS):
        if r["seq_level"] in ("NA", "unresolved", "") or r["ind_level"] in ("NA", "unresolved", ""):
            continue
        c = counts[r["axis"]]
        seq, ind = int(r["seq_level"]), int(r["ind_level"])
        c["subgoals"] += 1
        c["ind_gt_seq"] += ind > seq
        c["ind_lt_seq"] += ind < seq
        c["videos"].add(r["video_id"])
        if ind != seq:
            c["videos_differ"].add(r["video_id"])
    return {axis: {"subgoals": c["subgoals"], "ind_gt_seq": c["ind_gt_seq"],
                   "share_ind_gt_seq": c["ind_gt_seq"] / c["subgoals"],
                   "ind_lt_seq": c["ind_lt_seq"], "videos": len(c["videos"]),
                   "share_videos_with_any_difference": len(c["videos_differ"]) / len(c["videos"])}
            for axis, c in counts.items()}


def analyse():
    human, ablation = load()
    result = {"videos_scored": len(ablation), "alignment": [], "bootstrap": {}, "inflation": {}}
    for axis in AXES:
        rows = triples(human, ablation, axis)
        for rule, index in (("sequential", 2), ("independent", 3)):
            result["alignment"].append({"axis": axis, "scoring": rule,
                                        **metrics([(r[1], r[index]) for r in rows])})
        result["bootstrap"][axis] = bootstrap(rows)
        both = [(s["seq"][axis], s["ind"][axis]) for s in ablation.values()
                if s["seq"][axis] is not None and s["ind"][axis] is not None]
        result["inflation"][axis] = {"n": len(both), "mean_ind_minus_seq": sum(i - s for s, i in both) / len(both),
                                     "share_videos_ind_gt_seq": sum(i > s + 1e-9 for s, i in both) / len(both)}
    result["subgoal_disagreement"] = subgoal_disagreement()
    return result


def latex(result) -> str:
    lines = ["% Gate ablation (same videos per axis): Axis & Scoring & r & tau & CCC & MAE"]
    for row in result["alignment"]:
        axis = row["axis"].capitalize() if row["scoring"] == "sequential" else ""
        label = "Sequential" if row["scoring"] == "sequential" else "Independent"
        lines.append(f"% {row['axis']} {row['scoring']}: n={row['n']}")
        lines.append(f"{axis} & {label} & {fmt(row['pearson'])} & {fmt(row['kendall_tau'])} & "
                     f"{fmt(row['ccc'])} & {fmt(row['mae'], 1)} \\\\")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, help="Export a run directory into data/ first.")
    args = parser.parse_args()
    if args.from_run:
        export(args.from_run)
    result = analyse()
    OUT.mkdir(exist_ok=True)
    (OUT / "gate_ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "gate_ablation.tex").write_text(latex(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
