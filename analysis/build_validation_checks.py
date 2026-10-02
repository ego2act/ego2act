#!/usr/bin/env python
"""Held-out backbone validation, score coverage, and judge vs human scores on the panel.

Output: analysis/results/validation_checks.json.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from build_human_alignment import metrics
from scoring import final_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
FOLDS = 5
SEED = 0
MODELS = ["grok_imagine_video_1_5", "seedance_2_0", "kling_v3_pro", "wan_2_7", "minimax_h3", "cosmos_3"]


def number(value):
    return None if value in ("", None) else float(value)


def read(path):
    with path.open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if not r.get("video_id", r.get("id", "")).startswith("Aggregation")]


def heldout():
    human = {r["id"]: (r["case_id"], final_score(number(r["human_task"]), number(r["human_physics"]),
                                                 number(r["human_final_score"])))
             for r in read(DATA / "scores_human.csv")}
    flash = {r["video_id"]: final_score(number(r["ego2act_task"]), number(r["ego2act_physics"]), number(r["ego2act_score"]))
             for r in read(DATA / "scores_ego2act.csv")}
    terra = {r["video_id"]: final_score(number(r["task"]), number(r["physics"]), number(r["final"]))
             for r in read(DATA / "scores_terra.csv")}
    videos = [v for v in terra if v in human and human[v][1] is not None and terra[v] is not None and flash.get(v) is not None]
    cases = sorted({human[v][0] for v in videos})
    rng = random.Random(SEED)
    rng.shuffle(cases)
    folds = [cases[i::FOLDS] for i in range(FOLDS)]
    chosen, pooled, per_fold = [], {"selected": [], "flash": [], "terra": []}, []
    for k, test in enumerate(folds):
        train = [v for v in videos if human[v][0] not in test]
        held = [v for v in videos if human[v][0] in test]
        r_train = {name: metrics([(human[v][1], s[v]) for v in train], with_tau=False)["pearson"]
                   for name, s in (("flash", flash), ("terra", terra))}
        pick = max(r_train, key=r_train.get)
        chosen.append(pick)
        source = flash if pick == "flash" else terra
        pooled["selected"] += [(human[v][1], source[v]) for v in held]
        pooled["flash"] += [(human[v][1], flash[v]) for v in held]
        pooled["terra"] += [(human[v][1], terra[v]) for v in held]
        per_fold.append({"fold": k, "test_cases": len(test), "train_r": r_train, "selected": pick,
                         "heldout_r_selected": metrics([(human[v][1], source[v]) for v in held], with_tau=False)["pearson"]})
    return {"videos": len(videos), "cases": len(cases), "selected_per_fold": chosen, "folds": per_fold,
            "pooled_heldout": {k: metrics(v, with_tau=False) for k, v in pooled.items()}}


def coverage():
    rows = read(DATA / "scores_ego2act.csv")
    out = {}
    for group in MODELS + ["human_reference", "human_wrong"]:
        sub = [r for r in rows if r["model"] == group]
        n = len(sub)
        task = sum(number(r["ego2act_task"]) is not None for r in sub)
        physics = sum(number(r["ego2act_physics"]) is not None for r in sub)
        final = sum(final_score(number(r["ego2act_task"]), number(r["ego2act_physics"]), number(r["ego2act_score"])) is not None
                    for r in sub)
        unattempted = sum(number(r["ego2act_task"]) == 0 and number(r["ego2act_physics"]) is None for r in sub)
        out[group] = {"videos": n, "task": task / n, "physics": physics / n, "final": final / n,
                      "final_missing": 1 - final / n, "unattempted_zero": unattempted / n,
                      "task_status": dict(Counter(r["task_status"] or "ok" for r in sub)),
                      "physics_status": dict(Counter(r["physics_status"] or "ok" for r in sub))}
    total = len(rows)
    finals = sum(final_score(number(r["ego2act_task"]), number(r["ego2act_physics"]), number(r["ego2act_score"])) is not None
                 for r in rows)
    out["all"] = {"videos": total, "final": finals / total, "final_missing": 1 - finals / total}
    return out


def panel_scores():
    human = {r["id"]: r for r in read(DATA / "scores_human.csv")}
    judge = {r["video_id"]: r for r in read(DATA / "scores_ego2act.csv")}
    out = {}
    for model in MODELS:
        j_all, pairs = [], []
        for v, h in human.items():
            if h["model"] != model:
                continue
            hs = final_score(number(h["human_task"]), number(h["human_physics"]), number(h["human_final_score"]))
            e = judge.get(v)
            js = None if e is None else final_score(number(e["ego2act_task"]), number(e["ego2act_physics"]),
                                                    number(e["ego2act_score"]))
            if js is not None:
                j_all.append(js)
            if hs is not None and js is not None:
                pairs.append((hs, js))
        out[model] = {"judge_panel_all": mean(j_all), "judge_panel_matched": mean(p[1] for p in pairs),
                      "human_panel_matched": mean(p[0] for p in pairs), "matched_videos": len(pairs)}
    return out


def main():
    result = {"heldout": heldout(), "coverage": coverage(), "panel": panel_scores()}
    (HERE / "results/validation_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    h = result["heldout"]
    print("selected", h["selected_per_fold"], {k: round(v["pearson"], 3) for k, v in h["pooled_heldout"].items()},
          {k: round(v["mae"], 2) for k, v in h["pooled_heldout"].items()})
    for g, c in result["coverage"].items():
        print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
    for m, p in result["panel"].items():
        print(m, {k: round(v, 1) if isinstance(v, float) else v for k, v in p.items()})


if __name__ == "__main__":
    main()
