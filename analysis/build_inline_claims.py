"""Numbers quoted in the paper text that no other builder emits -> analysis/results/inline_claims.json.

Each entry has 'values' (unrounded) and a 'source' note naming the files it is computed from.
Reads only analysis/csv/ and analysis/results/. No API calls.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
RESULTS = HERE / "results"
GENERATORS = ["grok_imagine_video_1_5", "seedance_2_0", "kling_v3_pro", "wan_2_7", "minimax_h3", "cosmos_3"]


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b))
            / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)))


def sequential_vs_production() -> dict:
    production = {r["video_id"]: r for r in read(DATA / "scores_ego2act.csv")}
    values = {}
    for axis, column in (("task", "ego2act_task"), ("physics", "ego2act_physics")):
        pairs = [(float(r[f"seq_{axis}"]), float(production[r["video_id"]][column]))
                 for r in read(DATA / "scores_gate_ablation.csv")
                 if r["video_id"] in production and r[f"seq_{axis}"] != "" and production[r["video_id"]][column] != ""]
        values[f"{axis}_r"] = pearson(*zip(*pairs))
        values[f"{axis}_videos"] = len(pairs)
    return {"values": values, "source": "analysis/csv/scores_gate_ablation.csv (seq_task, seq_physics) vs "
            "analysis/csv/scores_ego2act.csv (ego2act_task, ego2act_physics), Pearson r over videos scored in both. "
            "Paper App. G: Task r 0.91, Physics r 0.77."}


def physics_control_inspection() -> dict:
    rows = read(DATA / "scores_physics_controls.csv")
    clean = {r["recording_id"]: r for r in rows if r["variant"] == "clean"}
    values = {}
    for variant in ("teleport", "swap", "ghost"):
        edited = [r for r in rows if r["variant"] == variant]
        inspected = [r for r in edited if r["inspected_edit"] == "1"]
        detected = [r for r in inspected if r["physics100"] and clean[r["recording_id"]]["physics100"]
                    and float(r["physics100"]) < float(clean[r["recording_id"]]["physics100"])]
        values[variant] = {"videos": len(edited), "inspected_edit": len(inspected),
                           "inspected_share": len(inspected) / len(edited), "detected_of_inspected": len(detected)}
    return {"values": values, "source": "analysis/csv/scores_physics_controls.csv: edited interval inspected "
            "(inspected_edit = 1), detected = Physics below the unedited rerun of the same recording. "
            "Paper App. G: inspected 7% / 13%, ghost detected in 10 of 13."}


def human_control_gap() -> dict:
    rows = {(r["evaluator"], r["group"]): r for r in read(RESULTS / "leaderboard.csv")}
    positive, negative = (float(rows[("ego2act", g)]["overall"]) for g in ("human_reference", "human_wrong"))
    return {"values": {"successful": positive, "unsuccessful": negative, "gap": positive - negative,
                       "gap_of_rounded_values": round(positive, 1) - round(negative, 1)},
            "source": "results/leaderboard.csv, ego2act rows human_reference and human_wrong (overall). The paper's "
            "'25.3 points above' subtracts the rounded 95.4 and 70.1; the unrounded gap is 25.24 (Mismatch M4)."}


def leaderboard_quotes() -> dict:
    rows = {(r["evaluator"], r["group"]): r for r in read(RESULTS / "leaderboard.csv")}
    number = lambda e, g, c: float(rows[(e, g)][c])
    return {"values": {
        "videoscore_human_reference_overall": number("videoscore", "human_reference", "overall"),
        "videoscore_human_wrong_overall": number("videoscore", "human_wrong", "overall"),
        "videoscore_cosmos_task": number("videoscore", "cosmos_3", "task"),
        "videoscore_cosmos_physics": number("videoscore", "cosmos_3", "physics"),
        "human_cosmos_task": number("human", "cosmos_3", "task"),
        "human_cosmos_physics": number("human", "cosmos_3", "physics"),
        "judge_grok_task": number("ego2act", "grok_imagine_video_1_5", "task"),
        "judge_wan_task": number("ego2act", "wan_2_7", "task")},
        "source": "results/leaderboard.csv (paper App. E control checks: VideoScore 87.5 vs 88.2, Cosmos-3 84.1/93.1 "
        "vs human 16.9/5.2, Grok and Wan Task 67.8)."}


def attachment_gaps() -> dict:
    by_model = load("subgoal_action_types.json")["by_model"]
    gaps = {}
    for model in GENERATORS:
        fam = by_model[model]
        gaps[model] = {"task_completion_points": 100 * (fam["A1"]["task"]["complete"] - fam["A4"]["task"]["complete"]),
                       "physics_valid_points": 100 * (fam["A1"]["physics"]["valid"] - fam["A4"]["physics"]["valid"])}
    return {"values": gaps, "source": "results/subgoal_action_types.json by_model: relocation/arrangement (A1) minus "
            "attachment/connection (A4), Task completion and Physics validity rates in points. Paper App. E: 1.6-21.3 "
            "(Task); 2.4-19.2 in five of six generators (Physics; Cosmos-3 is negative)."}


def task_length() -> dict:
    horizon = load("robustness.json")["horizon"]
    return {"values": {"cases": horizon["cases"], "spearman_duration_vs_task": horizon["spearman_duration_vs_task"],
                       "within_budget_cases": horizon["within_budget"]["cases"],
                       "within_budget_task": horizon["within_budget"]["mean_task"],
                       "beyond_budget_cases": horizon["beyond_budget"]["cases"],
                       "beyond_budget_task": horizon["beyond_budget"]["mean_task"]},
            "source": "results/robustness.json horizon (paper Sec. 5 / App. E: rho -0.29; 68.6% within the 15 s "
            "budget vs 54.8% beyond)."}


def judge_noise_of_model_means() -> dict:
    repeat, coverage = load("repeatability.json")["judge"], load("validation_checks.json")["coverage"]
    values = {}
    for axis in ("final", "physics", "task"):
        sd = repeat[axis]["by_source"]["ai"]["single_run_sd"]
        values[axis] = {"single_run_sd_generated": sd,
                        "standard_error_of_model_mean": {m: sd / math.sqrt(coverage[m]["videos"]) for m in GENERATORS}}
        values[axis]["max_standard_error"] = max(values[axis]["standard_error_of_model_mean"].values())
    return {"values": values, "source": "results/repeatability.json (single-run SD on generated videos) divided by "
            "sqrt(videos per generator, results/validation_checks.json coverage), assuming independent judge noise. "
            "Paper App. G: 'under one point' for model means (Final)."}


def main() -> None:
    claims = {"sequential_vs_production": sequential_vs_production(),
              "physics_control_inspection": physics_control_inspection(),
              "human_control_gap": human_control_gap(),
              "leaderboard_quotes": leaderboard_quotes(),
              "attachment_gaps_by_generator": attachment_gaps(),
              "task_length": task_length(),
              "judge_noise_of_model_means": judge_noise_of_model_means()}
    (RESULTS / "inline_claims.json").write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
