#!/usr/bin/env python
"""Subgoal-level Task and Physics outcomes by action family, from the judge traces
and plans; subgoals are assigned to families by their leading verb.

Inputs: analysis/csv/traces.jsonl, judge_plans.jsonl, action_taxonomy.json.
Output: analysis/results/subgoal_action_types.json.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
OUT = HERE / "results"
MODELS = ["grok_imagine_video_1_5", "seedance_2_0", "kling_v3_pro", "wan_2_7", "minimax_h3", "cosmos_3"]
MODEL_LABELS = ["Grok-1.5", "Seedance-2.0", "Kling-v3-Pro", "Wan-2.7", "MiniMax-H3", "Cosmos-3"]
FAMILY_LABELS = {"A1": "Relocation/arrangement", "A2": "Containment/retrieval", "A3": "Opening/closing",
                 "A4": "Attachment/connection", "A5": "Shape/orientation change", "A6": "Material transfer/mixing",
                 "A7": "Surface treatment", "A8": "Cutting/separation", "A9": "Device activation"}
SYNONYMS = {
    "put": "A1", "take": "A1", "pick": "A1", "transfer": "A1", "arrange": "A1", "reposition": "A1",
    "position": "A1", "push": "A1", "slide": "A1", "lift": "A1", "set": "A1", "lay": "A1", "carry": "A1",
    "store": "A2", "load": "A2", "empty": "A2", "unpack": "A2", "extract": "A2",
    "cover": "A3", "secure": "A3", "screw": "A3", "unscrew": "A3", "seal": "A3", "lock": "A3", "unlock": "A3",
    "connect": "A4", "disconnect": "A4", "fit": "A4", "install": "A4", "plug": "A4", "staple": "A4",
    "clip": "A4", "mount": "A4", "thread": "A4", "tie": "A4",
    "stand": "A5", "orient": "A5", "rotate": "A5", "wrap": "A5", "bend": "A5", "straighten": "A5",
    "measure": "A6", "add": "A6", "squeeze": "A6", "spread": "A6", "sprinkle": "A6",
    "wash": "A7", "clean": "A7", "spray": "A7", "write": "A7", "brush": "A7", "polish": "A7",
    "slice": "A8", "split": "A8", "peel": "A8", "separate": "A8",
    "light": "A9", "switch": "A9", "activate": "A9", "start": "A9", "charge": "A9",
}
RESAMPLES = 1000
SEED = 0


def lexicon():
    taxonomy = json.loads((DATA / "action_taxonomy.json").read_text())
    words = {}
    for family, spec in taxonomy.items():
        for verb in spec["action_types"]:
            words.setdefault(verb.split("_")[0], family)
    words.update(SYNONYMS)
    return words


def forms(word):
    out = {word, word + "s", word + "es", word + "ed", word + "d", word + "ing"}
    if word.endswith("e"):
        out |= {word[:-1] + "ing"}
    if re.fullmatch(r".*[^aeiou][aeiou][^aeiouwy]", word):
        out |= {word + word[-1] + "ed", word + word[-1] + "ing"}
    return out


def classifier():
    table = {}
    for word, family in lexicon().items():
        for form in forms(word):
            table.setdefault(form, family)

    def classify(action):
        tokens = re.findall(r"[a-z]+", action.lower())
        for i, token in enumerate(tokens[:3]):
            if token == "turn" or token in forms("turn"):
                return "A9" if set(tokens[i + 1:i + 4]) & {"on", "off"} else "A5"
            if token in table:
                return table[token]
        return None
    return classify


def load():
    classify = classifier()
    units = defaultdict(dict)  # (case, seed) -> model -> {"task": [(fam, level)], "physics": [...]}
    counts = Counter()
    plans = {}
    with (DATA / "judge_plans.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["run"] == "production":
                plans[row["video_id"], row["axis"]] = row["plan"]
    with (DATA / "traces.jsonl").open() as stream:
        for line in stream:
            trace = json.loads(line)
            video = trace["video_id"]
            case, rest = video.split("::")
            model = next((m for m in MODELS if rest.startswith(m)), None)
            if model is None:
                continue
            seed = rest.split("__")[-1]
            record = {}
            for axis in ("task", "physics"):
                levels = (trace.get(axis) or {}).get("levels") or {}
                plan = plans.get((video, axis))
                if not levels or not plan:
                    record = None
                    break
                actions = {s["id"]: s.get("action", "") for s in plan}
                rows = []
                for sid, level in levels.items():
                    family = classify(actions.get(sid, ""))
                    counts[axis, family is not None] += 1
                    if family:
                        rows.append((family, level))
                record[axis] = rows
            if record is not None:
                units[case, seed][model] = record
    complete = {k: v for k, v in units.items() if all(m in v for m in MODELS)}
    return complete, counts


def rates(units, keys, family, axis, models=MODELS):
    task = Counter()
    for key in keys:
        for model in models:
            for fam, level in units[key][model][axis]:
                if fam == family:
                    task[level] += 1
    if axis == "task":
        n = sum(task.values())
        return None if n == 0 else {"n": n, "complete": task[3] / n, "skipped": task[0] / n, "incomplete": task[2] / n,
                                    "levels": {str(k): task[k] / n for k in range(4)}}
    judgeable = sum(v for k, v in task.items() if k != "NA")
    n = sum(task.values())
    return None if judgeable == 0 else {"n": n, "judgeable": judgeable, "valid": task[4] / judgeable,
                                        "state_inconsistency": task[0] / judgeable,
                                        "invalid_interaction": task[2] / judgeable, "na": task["NA"] / n,
                                        "levels": {str(k): task[k] / judgeable for k in range(5)}}


def interval(units, keys, family, axis, field, rng, models=MODELS):
    values = []
    for _ in range(RESAMPLES):
        r = rates(units, rng.choices(keys, k=len(keys)), family, axis, models)
        if r is not None:
            values.append(r[field])
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def analyse():
    rng = random.Random(SEED)
    units, counts = load()
    keys = sorted(units)
    families = {}
    for family in FAMILY_LABELS:
        t, p = rates(units, keys, family, "task"), rates(units, keys, family, "physics")
        entry = {"name": FAMILY_LABELS[family], "task": t, "physics": p}
        if t and t["n"] >= 100:
            entry["task_ci"] = interval(units, keys, family, "task", "complete", rng)
        if p and p["judgeable"] >= 100:
            entry["physics_ci"] = interval(units, keys, family, "physics", "valid", rng)
        families[family] = entry
    by_model = {m: {f: {"task": rates(units, keys, f, "task", [m]), "physics": rates(units, keys, f, "physics", [m])}
                    for f in FAMILY_LABELS} for m in MODELS}
    task_fail = defaultdict(Counter)
    phys_fail = defaultdict(Counter)
    for key in keys:
        for model in MODELS:
            for _, level in units[key][model]["task"]:
                if level != 3:
                    task_fail[model][level] += 1
            for _, level in units[key][model]["physics"]:
                if level != 4:
                    phys_fail[model][level] += 1
    failure = {m: {"task_level0_or_2": (task_fail[m][0] + task_fail[m][2]) / sum(task_fail[m].values()),
                   "physics_na_0_or_2": (phys_fail[m]["NA"] + phys_fail[m][0] + phys_fail[m][2]) / sum(phys_fail[m].values())}
               for m in MODELS}
    pooled_t = sum((task_fail[m] for m in MODELS), Counter())
    pooled_p = sum((phys_fail[m] for m in MODELS), Counter())
    failure["pooled"] = {"task_level0_or_2": (pooled_t[0] + pooled_t[2]) / sum(pooled_t.values()),
                         "physics_na_0_or_2": (pooled_p["NA"] + pooled_p[0] + pooled_p[2]) / sum(pooled_p.values())}
    level_share = {}
    for model in MODELS:
        t, p = Counter(), Counter()
        for key in keys:
            t.update(level for _, level in units[key][model]["task"])
            p.update(level for _, level in units[key][model]["physics"])
        level_share[model] = {"task": {str(k): t[k] / sum(t.values()) for k in range(4)},
                              "physics": {str(k): p[k] / sum(p.values()) for k in [0, 1, 2, 3, 4, "NA"]}}
    return {"units": len(keys), "cases": len({c for c, _ in keys}), "level_share": level_share,
            "classified_share": {axis: counts[axis, True] / (counts[axis, True] + counts[axis, False])
                                 for axis in ("task", "physics")},
            "families": families, "by_model": by_model, "failure_concentration": failure}


def main():
    result = analyse()
    OUT.mkdir(exist_ok=True)
    (OUT / "subgoal_action_types.json").write_text(json.dumps(result, indent=2) + "\n")
    print("units", result["units"], "cases", result["cases"], "classified", result["classified_share"])
    for f, e in result["families"].items():
        t, p = e["task"], e["physics"]
        if t:
            print(f, e["name"], t["n"], f"complete {100*t['complete']:.1f} skip {100*t['skipped']:.1f} incompl {100*t['incomplete']:.1f}",
                  e.get("task_ci"), "|", p and f"valid {100*p['valid']:.1f} P0 {100*p['state_inconsistency']:.1f}", e.get("physics_ci"))
    print(json.dumps(result["failure_concentration"], indent=1))


if __name__ == "__main__":
    main()
