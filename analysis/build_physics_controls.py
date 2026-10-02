#!/usr/bin/env python
"""Physics negative controls: judge scores on human recordings with synthetic
violations (teleport, swap, ghost) against clean reruns of the same recordings.

Input: analysis/csv/scores_physics_controls.csv (or a raw run via --from-run).
Output: analysis/results/physics_controls.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import fmean

from scoring import final_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
OUT = HERE / "results"
SCORES = DATA / "scores_physics_controls.csv"
PRODUCTION = DATA / "scores_ego2act.csv"
CORRUPTIONS = ["teleport", "swap", "ghost"]
GATES = {0: "P1", 1: "P2", 2: "P3", 3: "P4", 4: "pass"}
FIELDS = ["video_id", "case_id", "recording_id", "domain", "variant", "edit_start_s", "edit_end_s",
          "task100", "physics100", "final100", "physics_windows", "physics_first_failed_gate",
          "sampled_frames_in_edit", "inspected_edit", "task_status", "physics_status", "cost_usd", "status"]
PHYSICS_INITIAL_FRAMES = 24  # Session.initial for the Physics axis


def number(value):
    return None if value in (None, "", "None") else float(value)


def edit_diagnostics(v, result):
    """Uniform Physics frames inside the edit, and whether an inspection overlapped it."""
    if v["variant"] == "clean":
        return "", ""
    a, b = v["edit_start_s"], v["edit_end_s"]
    if v["variant"] == "teleport":
        a, b = a - 0.75, a + 0.75
    fps = float(Fraction(v["fps"]))
    times = [i / fps for i in range(v["output_frames"])]
    targets = [times[-1] * k / (PHYSICS_INITIAL_FRAMES - 1) for k in range(PHYSICS_INITIAL_FRAMES)]
    picked = {min(range(len(times)), key=lambda i: abs(times[i] - t)) for t in targets}
    inside = sum(a <= times[i] <= b for i in picked)
    inspections = [t["inspection"] for t in result["physics"].get("trace", []) if "inspection" in t]
    overlap = any(i["start_seconds"] <= b and i["end_seconds"] >= a for i in inspections)
    return inside, int(overlap)


def gate_of(level):
    return GATES.get(level, str(level)) if isinstance(level, int) else str(level)


# ---------------------------------------------------------------- export

def export(run: Path) -> None:
    lines = lambda name: [json.loads(l) for l in (run / name).read_text().splitlines()
                          if l.strip()] if (run / name).exists() else []
    variants = json.loads((run / "variants.json").read_text())
    domains = {r["video_id"]: r["domain_group"] for r in csv.DictReader((run / "selection.csv").open(newline=""))}
    scores = {s["video_id"]: s for s in lines("scores.jsonl")}
    failures = {f["video_id"]: f for f in lines("failures.jsonl")}
    rows = []
    for v in variants:
        vid = v["video_id"]
        row = {"video_id": vid, "case_id": v["case_id"], "recording_id": v["recording_id"],
               "domain": domains.get(v["recording_id"], ""), "variant": v["variant"],
               "edit_start_s": "" if v.get("edit_start_s") is None else round(v["edit_start_s"], 3),
               "edit_end_s": "" if v.get("edit_end_s") is None else round(v["edit_end_s"], 3)}
        s = scores.get(vid)
        if s is None:
            f = failures.get(vid)
            status = "not_run" if f is None else ("budget" if f["error"].startswith("budget") else "failed")
            rows.append({**row, "status": status})
            continue
        levels = s.get("physics_levels") or {}
        judged = json.loads((run / "artifacts" / vid.replace("::", "____") / "judge_result.json").read_text())
        inside, inspected = edit_diagnostics(v, judged)
        t, p = s["task100"], s["physics100"]
        rows.append({**row, "task100": t, "physics100": p, "final100": final_score(t, p),
                     "physics_windows": json.dumps(levels, sort_keys=True),
                     "physics_first_failed_gate": ";".join(f"{k}:{gate_of(levels[k])}" for k in sorted(levels)),
                     "sampled_frames_in_edit": inside, "inspected_edit": inspected,
                     "task_status": s["task_status"], "physics_status": s["physics_status"],
                     "cost_usd": round(s["cost_usd"], 5),
                     "status": "ok" if s["task_status"] == "ok" and s["physics_status"] == "ok" else
                     f"task_{s['task_status']}|physics_{s['physics_status']}"})
    with SCORES.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- analysis

def bootstrap_mean(values, resamples=1000, seed=0):
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = random.Random(seed)
    draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    return {"mean": fmean(values), "low": draws[int(0.025 * resamples)],
            "high": draws[int(0.975 * resamples) - 1], "n": len(values)}


def failing_windows(row):
    levels = json.loads(row["physics_windows"]) if row["physics_windows"] else {}
    return [lv for lv in levels.values() if isinstance(lv, int) and lv < 4]


def sampling(corrupted):
    """Detection split by whether the judge could have seen the edit at all."""
    seen = lambda x: int(x["sampled_frames_in_edit"] or 0) > 0 or x["inspected_edit"] == "1"
    groups = {"edit_visible_in_evidence": [x for x in corrupted if seen(x)],
              "edit_not_in_evidence": [x for x in corrupted if not seen(x)]}
    out = {name: {"videos": len(g), "any_window_failing": fmean(bool(failing_windows(x)) for x in g) if g else None}
           for name, g in groups.items()}
    out["mean_sampled_frames_in_edit"] = fmean(int(x["sampled_frames_in_edit"] or 0) for x in corrupted) if corrupted else None
    out["inspected_edit_share"] = fmean(x["inspected_edit"] == "1" for x in corrupted) if corrupted else None
    return out


def analyse():
    rows = list(csv.DictReader(SCORES.open(newline="")))
    by = {(r["recording_id"], r["variant"]): r for r in rows}
    recordings = sorted({r["recording_id"] for r in rows})
    ran = lambda r: r is not None and r["status"] not in ("budget", "failed", "not_run")
    coverage = Counter(r["status"] if r["status"] in ("budget", "failed", "not_run") else "judged" for r in rows)

    clean = [by.get((rec, "clean")) for rec in recordings]
    clean_scored = [r for r in clean if ran(r) and number(r["physics100"]) is not None]
    result = {"videos": len(rows), "recordings": len(recordings), "coverage": dict(coverage),
              "spend_usd": round(sum(number(r["cost_usd"]) or 0 for r in rows), 4),
              "clean_false_alarm": {
                  "videos": len(clean_scored),
                  "any_window_failing": (fmean(bool(failing_windows(r)) for r in clean_scored)
                                         if clean_scored else None),
                  "failing_gates": dict(Counter(GATES[lv] for r in clean_scored for lv in failing_windows(r)))},
              "corruptions": {}}

    for kind in CORRUPTIONS:
        pairs = []
        for rec in recordings:
            c, x = by.get((rec, "clean")), by.get((rec, kind))
            if ran(c) and ran(x):
                pairs.append((c, x))
        phys = [(number(c["physics100"]), number(x["physics100"])) for c, x in pairs
                if number(c["physics100"]) is not None and number(x["physics100"]) is not None]
        task = [(number(c["task100"]), number(x["task100"])) for c, x in pairs
                if number(c["task100"]) is not None and number(x["task100"]) is not None]
        final = [(number(c["final100"]), number(x["final100"])) for c, x in pairs
                 if number(c["final100"]) is not None and number(x["final100"]) is not None]
        corrupted_scored = [x for c, x in pairs if number(x["physics100"]) is not None]
        gates = Counter(GATES[lv] for x in corrupted_scored for lv in failing_windows(x))
        clean_anyfail = [bool(failing_windows(c)) for c, x in pairs
                         if number(c["physics100"]) is not None and number(x["physics100"]) is not None]
        corr_anyfail = [bool(failing_windows(x)) for c, x in pairs
                        if number(c["physics100"]) is not None and number(x["physics100"]) is not None]
        result["corruptions"][kind] = {
            "pairs_judged": len(pairs),
            "pairs_with_physics": len(phys),
            "physics_unscored_corrupted": sum(1 for c, x in pairs if number(x["physics100"]) is None),
            "detection_physics_below_clean": fmean(x < c for c, x in phys) if phys else None,
            "physics_above_clean": fmean(x > c for c, x in phys) if phys else None,
            "detection_any_window_failing": fmean(corr_anyfail) if corr_anyfail else None,
            "clean_any_window_failing_same_pairs": fmean(clean_anyfail) if clean_anyfail else None,
            "physics_drop": bootstrap_mean([c - x for c, x in phys]),
            "task_change": bootstrap_mean([x - c for c, x in task]),
            "final_change": bootstrap_mean([x - c for c, x in final]),
            "failing_gate_counts": dict(gates),
            "most_common_failing_gate": gates.most_common(1)[0][0] if gates else None,
            "sampling": sampling([x for c, x in pairs if number(x["physics100"]) is not None]),
            "mean_physics_clean": fmean(c for c, _ in phys) if phys else None,
            "mean_physics_corrupted": fmean(x for _, x in phys) if phys else None,
        }

    production = {r["video_id"]: r for r in csv.DictReader(PRODUCTION.open(newline=""))}
    sanity = [(number(production[r["recording_id"]]["ego2act_physics"]), number(r["physics100"]),
               number(production[r["recording_id"]]["ego2act_task"]), number(r["task100"]))
              for r in clean if ran(r) and r["recording_id"] in production]
    ok = [s for s in sanity if s[1] is not None]
    result["clean_vs_production"] = {
        "videos": len(sanity), "physics_scored": len(ok),
        "production_physics_mean": fmean(s[0] for s in ok) if ok else None,
        "clean_rerun_physics_mean": fmean(s[1] for s in ok) if ok else None,
        "clean_rerun_physics_100_share": fmean(s[1] == 100 for s in ok) if ok else None,
        "physics_mean_abs_diff": fmean(abs(s[0] - s[1]) for s in ok) if ok else None,
        "production_task_mean": fmean(s[2] for s in sanity if s[3] is not None) if sanity else None,
        "clean_rerun_task_mean": fmean(s[3] for s in sanity if s[3] is not None) if sanity else None,
        "task_mean_abs_diff": fmean(abs(s[2] - s[3]) for s in sanity if s[3] is not None) if sanity else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path,
                        help="Re-export the cached CSV from a raw run directory first.")
    args = parser.parse_args()
    if args.from_run:
        export(args.from_run)
    result = analyse()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "physics_controls.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
