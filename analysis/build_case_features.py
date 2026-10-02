#!/usr/bin/env python
"""Domain profiles and case-feature diagnostics on the human-panel cases (Appendix E).

Inputs: analysis/csv/scores_human.csv, scores_ego2act.csv, case_features.csv, action_taxonomy.json.
Outputs: analysis/results/case_features.json, domain_models.tex, feature_detail.tex,
feature_qvalues.tex.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from build_robustness import spearman
from scoring import final_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
OUT = HERE / "results"
MODELS = ["grok_imagine_video_1_5", "seedance_2_0", "kling_v3_pro", "wan_2_7", "minimax_h3", "cosmos_3"]
MODEL_LABELS = ["Grok-1.5", "Seedance-2.0", "Kling-v3-Pro", "Wan-2.7", "MiniMax-H3", "Cosmos-3"]
DOMAINS = [("Household Org/Storage", "Household"), ("Kitchen/Food Prep", "Kitchen"),
           ("Office/Study Workspace", "Office"), ("Personal Care/Utilities", "Personal care"), ("Others", "Other")]
FAMILIES = {"A1": "Relocation/arrangement", "A2": "Containment/retrieval", "A3": "Opening/closing",
            "A4": "Attachment/connection", "A5": "Shape/orientation change", "A6": "Material transfer/mixing",
            "A7": "Surface treatment", "A8": "Cutting/separation", "A9": "Device activation"}
CLUTTER = {"None": 0, "Minimal": 1, "Heavy": 2}
RESAMPLES = 2000
PERMUTATIONS = 5000
SEED = 0
BLUE, PURPLE, ORANGE = "#2B729A", "#9d4edd", "#ea580c"


def number(value):
    return None if value in ("", None) else float(value)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load():
    panel = {r["case_id"] for r in read(DATA / "scores_human.csv")}
    features = {r["case_id"]: r for r in read(DATA / "case_features.csv") if r["case_id"] in panel}
    taxonomy = json.loads((DATA / "action_taxonomy.json").read_text())
    family_of = {a: k for k, v in taxonomy.items() for a in v["action_types"]}
    for f in features.values():
        f["families"] = sorted({family_of[a] for a in json.loads(f["action_types"])})

    judge = defaultdict(lambda: defaultdict(list))  # case -> model -> [final]
    judge_by_video = {}
    for r in read(DATA / "scores_ego2act.csv"):
        if r["case_id"] not in panel or r["model"] not in MODELS:
            continue
        s = final_score(number(r["ego2act_task"]), number(r["ego2act_physics"]), number(r["ego2act_score"]))
        if s is not None:
            judge[r["case_id"]][r["model"]].append(s)
            judge_by_video[r["video_id"]] = s

    error = defaultdict(list)
    for r in read(DATA / "scores_human.csv"):
        if r["case_id"] not in panel or r["model"] not in MODELS or r["id"] not in judge_by_video:
            continue
        h = final_score(number(r["human_task"]), number(r["human_physics"]), number(r["human_final_score"]))
        if h is not None:
            error[r["case_id"]].append(abs(judge_by_video[r["id"]] - h))
    return features, judge, {c: mean(v) for c, v in error.items()}


def case_score(judge, case, models=MODELS):
    values = [s for m in models for s in judge[case][m]]
    return mean(values) if values else None


def bh(pvalues):
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    q, running = [0.0] * n, 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * n / rank)
        q[i] = running
    return q


def rho_test(x, y, rng):
    keys = list(range(len(x)))
    estimate = spearman(x, y)
    boots = []
    for _ in range(RESAMPLES):
        idx = rng.choices(keys, k=len(keys))
        xs, ys = [x[i] for i in idx], [y[i] for i in idx]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            boots.append(spearman(xs, ys))
    boots.sort()
    shuffled, extreme = list(y), 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        extreme += abs(spearman(x, shuffled)) >= abs(estimate) - 1e-12
    return {"estimate": estimate, "low": boots[int(0.025 * len(boots))], "high": boots[int(0.975 * len(boots)) - 1],
            "p": (extreme + 1) / (PERMUTATIONS + 1), "n": len(x)}


def difference_test(present, absent, rng):
    estimate = mean(present) - mean(absent)
    boots = sorted(mean(rng.choices(present, k=len(present))) - mean(rng.choices(absent, k=len(absent)))
                   for _ in range(RESAMPLES))
    pooled, extreme = present + absent, 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        extreme += abs(mean(pooled[:len(present)]) - mean(pooled[len(present):])) >= abs(estimate) - 1e-12
    return {"estimate": estimate, "low": boots[int(0.025 * RESAMPLES)], "high": boots[int(0.975 * RESAMPLES) - 1],
            "p": (extreme + 1) / (PERMUTATIONS + 1), "present": len(present), "absent": len(absent)}


def continuous(features):
    return {
        "Observed operations": {c: float(f["mean_observed_steps"]) for c, f in features.items()},
        "Involved objects": {c: float(f["involved_object_count"]) for c, f in features.items()},
        "Reference duration": {c: float(f["mean_human_duration_seconds"]) for c, f in features.items()},
        "Action-family diversity": {c: float(len(f["families"])) for c, f in features.items()},
        "Clutter (ordinal)": {c: float(CLUTTER[f["clutter_level"]]) for c, f in features.items()
                              if f["clutter_level"] in CLUTTER},
    }


def analyse():
    rng = random.Random(SEED)
    features, judge, error = load()
    cases = sorted(features)
    score = {c: case_score(judge, c) for c in cases}

    domains = {}
    for model in MODELS:
        domains[model] = {}
        for key, _ in DOMAINS:
            members = [c for c in cases if features[c]["domain"] == key and judge[c][model]]
            domains[model][key] = {"score": mean(mean(judge[c][model]) for c in members), "cases": len(members)}

    families = {}
    for code, name in FAMILIES.items():
        members = [c for c in cases if code in features[c]["families"]]
        families[code] = {"name": name, "cases": len(members),
                          "score": mean(score[c] for c in members) if members else None}
    clutter = {}
    for level in ("Heavy", "Minimal", "None"):
        members = [c for c in cases if features[c]["clutter_level"] == level]
        clutter[level] = {"cases": len(members), "score": mean(score[c] for c in members) if members else None}

    tests = {"generator": {}, "interaction": {}, "judge_error": {}}
    for name, values in continuous(features).items():
        keys = [c for c in cases if c in values]
        tests["generator"][name] = rho_test([values[c] for c in keys], [score[c] for c in keys], rng)
        if name != "Clutter (ordinal)":
            keys = [c for c in keys if c in error]
            tests["judge_error"][name] = rho_test([values[c] for c in keys], [error[c] for c in keys], rng)
    for code, name in FAMILIES.items():
        present = [score[c] for c in cases if code in features[c]["families"]]
        absent = [score[c] for c in cases if code not in features[c]["families"]]
        if len(present) >= 3 and len(absent) >= 3:
            tests["interaction"][name] = difference_test(present, absent, rng)
    for group in tests.values():
        names = list(group)
        for name, q in zip(names, bh([group[n]["p"] for n in names])):
            group[name]["q"] = q

    points = [{"case_id": c, "observed_operations": float(features[c]["mean_observed_steps"]),
               "reference_duration": float(features[c]["mean_human_duration_seconds"]), "score": score[c],
               "domain": features[c]["domain"], "families": features[c]["families"]} for c in cases]
    return {"cases": len(cases), "domains": domains, "families": families, "clutter": clutter,
            "tests": tests, "points": points, "judge_error_cases": len(error)}


def write_tables(result):
    lines = []
    for model, label in zip(MODELS, MODEL_LABELS):
        cells = []
        for key, _ in DOMAINS:
            value = result["domains"][model][key]["score"]
            best = max(result["domains"][m][key]["score"] for m in MODELS)
            cells.append(rf"\textbf{{{value:.1f}}}" if value == best else f"{value:.1f}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    (OUT / "domain_models.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for f in result["families"].values():
        score = "--" if f["score"] is None else f"{f['score']:.1f}"
        lines.append(f"Action & {f['name']} & {f['cases']} & {score} \\\\")
    lines.append(r"\midrule")
    for level, c in result["clutter"].items():
        lines.append(f"Clutter & {level} & {c['cases']} & {c['score']:.1f} \\\\")
    (OUT / "feature_detail.tex").write_text("\n".join(lines) + "\n")

    t = result["tests"]
    lines = []
    for name, g in t["generator"].items():
        e = t["judge_error"].get(name)
        lines.append(f"{name} & {g['q']:.3f} & {'--' if e is None else format(e['q'], '.3f')} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{@{}l}{\textit{Interaction-associated score differences}} \\")
    for name, g in t["interaction"].items():
        lines.append(f"{name} & {g['q']:.3f} & -- \\\\")
    (OUT / "feature_qvalues.tex").write_text("\n".join(lines) + "\n")


def main():
    result = analyse()
    OUT.mkdir(exist_ok=True)
    (OUT / "case_features.json").write_text(json.dumps(result, indent=2) + "\n")
    write_tables(result)
    print("cases", result["cases"], "judge-error cases", result["judge_error_cases"])
    for model in MODELS:
        print(model, {k[:9]: round(v["score"], 1) for k, v in result["domains"][model].items()},
              [v["cases"] for v in result["domains"][model].values()])
    for f in result["families"].values():
        print(f["name"], f["cases"], None if f["score"] is None else round(f["score"], 1))
    print({k: (v["cases"], round(v["score"], 1)) for k, v in result["clutter"].items()})
    for key, group in result["tests"].items():
        print(key, {n: (round(g["estimate"], 3), round(g["p"], 3), round(g["q"], 3)) for n, g in group.items()})


if __name__ == "__main__":
    main()
