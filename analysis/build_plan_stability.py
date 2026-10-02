#!/usr/bin/env python
"""Stability of Ego2ActJudge's Task and Physics plans across repeated generations
for the same case, and whether plan variation moves the scores.

Inputs: analysis/csv/traces.jsonl, judge_plans.jsonl, pilot_scores.json, scores_ego2act.csv.
Outputs: analysis/results/plan_stability.json and plan_stability_cases.csv.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

HERE = Path(__file__).resolve().parent
TRACES = HERE.parent / "analysis/csv/traces.jsonl"
SCORES = HERE.parent / "analysis/csv/scores_ego2act.csv"
PILOT = HERE.parent / "analysis/csv/pilot_scores.json"
PLANS = HERE.parent / "analysis/csv/judge_plans.jsonl"
OUT_JSON = HERE / "results/plan_stability.json"
OUT_CSV = HERE / "results/plan_stability_cases.csv"

AXES = ("task", "physics")
MAX_PAIRS = 50
JACCARD = 0.5
HIGH_F1 = 0.8
BOOT = 1000
# Coarse verb classes for the synonym variant only (primary matcher uses the raw verb).
VERB_CLASS = {"put": "plac", "set": "plac", "position": "plac", "lay": "plac", "rest": "plac", "insert": "plac",
              "mov": "plac", "transfer": "plac", "hang": "plac", "return": "plac", "stand": "plac",
              "pick": "grasp", "grab": "grasp", "take": "grasp", "lift": "grasp", "grasp": "grasp", "remov": "grasp",
              "shut": "clos", "unscrew": "open", "twist": "turn", "rotat": "turn", "push": "press"}
STOP = {"the", "a", "an", "of", "on", "in", "into", "onto", "to", "from", "at", "with",
        "and", "its", "their", "it", "s", "inside", "top", "for", "by", "up"}


# ---------------------------------------------------------------- loading
def load_plans():
    """(run, video_id, axis) -> (plan, provider response id), extracted from the
    raw judge artifacts into analysis/csv/judge_plans.jsonl."""
    plans = {}
    for line in PLANS.open():
        r = json.loads(line)
        plans[r["run"], r["video_id"], r["axis"]] = (r["plan"], r["response_id"])
    return plans


# ---------------------------------------------------------------- matching
def lemma(tok: str) -> str:
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            tok = tok[: -len(suf)] + rep
            break
    if len(tok) > 3 and tok[-1] == tok[-2] and tok[-1] not in "aeiou":  # "placing"->"plac", "dropped"->"dropp"
        tok = tok[:-1]
    return tok.rstrip("e") if len(tok) > 3 else tok


def tokens(text) -> list[str]:
    text = str(text or "").lower().replace("'s", " ").replace("’s", " ")
    return [lemma(t) for t in re.findall(r"[a-z0-9]+", text) if t not in STOP]


def norm(sub, verb_class=False):
    act = tokens(sub.get("action"))
    verb = act[0] if act else ""
    if verb_class:
        verb = VERB_CLASS.get(verb, verb)
    return {"id": sub.get("id"), "verb": verb,
            "objs": set(tokens(sub.get("source"))) | set(tokens(sub.get("target"))),
            "requires": list(sub.get("requires") or [])}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def match(pa, pb, use_verb=True):
    """Greedy one-to-one matching; returns dict id_a -> id_b.
    use_verb: True (same lemmatized verb), "class" (coarse verb class), False (objects only)."""
    A = [norm(s, use_verb == "class") for s in pa]
    B = [norm(s, use_verb == "class") for s in pb]
    cand = []
    for x in A:
        for y in B:
            if use_verb and x["verb"] != y["verb"]:
                continue
            j = jaccard(x["objs"], y["objs"])
            if j >= JACCARD:
                cand.append((j, x["id"], y["id"]))
    cand.sort(key=lambda c: -c[0])
    used_a, used_b, m = set(), set(), {}
    for _, a, b in cand:
        if a not in used_a and b not in used_b:
            m[a] = b
            used_a.add(a)
            used_b.add(b)
    return m


def compare(pa, pb, use_verb=True):
    m = match(pa, pb, use_verb)
    n = len(pa) + len(pb)
    f1 = 2 * len(m) / n if n else 1.0
    req_a = {s["id"]: set(s.get("requires") or []) for s in pa}
    req_b = {s["id"]: set(s.get("requires") or []) for s in pb}
    dep_ok, dep_nontriv = [], []
    for a, b in m.items():
        mapped = {m.get(r, f"__unmatched_{r}") for r in req_a.get(a, set())}
        ok = mapped == req_b.get(b, set())
        dep_ok.append(ok)
        if req_a.get(a) or req_b.get(b):
            dep_nontriv.append(ok)
    return {"f1": f1, "matches": len(m), "dep_ok": dep_ok, "dep_nontrivial": dep_nontriv}


# ---------------------------------------------------------------- stats
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": mean(xs), "median": median(xs),
            "q1": quantile(xs, 0.25), "q3": quantile(xs, 0.75)}


def single_run_sd(diffs):
    return math.sqrt(sum(d * d for d in diffs) / len(diffs) / 2) if diffs else None


def bootstrap(cases, stat, seed=0, n=BOOT):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [rng.choice(cases) for _ in cases]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    return [quantile(vals, 0.025), quantile(vals, 0.975)] if vals else None


def _dcount_gap(st):
    """Mean |diff| for pairs whose plan counts differ minus pairs with equal counts."""
    groups = st["by_dcount"]
    pos = [v for k, v in groups.items() if k != "0"]
    if "0" not in groups or not pos:
        return None
    return sum(v["mean_absdiff"] * v["videos"] for v in pos) / sum(v["videos"] for v in pos) - groups["0"]["mean_absdiff"]


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return g


def number(v):
    return None if v in ("", None) else float(v)


# ---------------------------------------------------------------- main
def main():
    meta = {r["video_id"]: r for r in csv.DictReader(SCORES.open())}
    cached = load_plans()
    prod = {}
    for line in TRACES.open():
        t = json.loads(line)
        if not t.get("artifact"):
            continue
        rec = {"video": t["video_id"], "case": t["video_id"].split("::")[0],
               "task": t.get("task_score"), "physics": t.get("physics_score"),
               "model": meta.get(t["video_id"], {}).get("model")}
        for axis in AXES:
            rec[f"{axis}_plan"], rec[f"{axis}_resp"] = cached.get(("production", t["video_id"], axis), (None, None))
        prod[t["video_id"]] = rec

    by_case = defaultdict(list)
    for r in prod.values():
        by_case[r["case"]].append(r)

    # ---- data coverage / caching check
    coverage = {"videos": len(prod), "cases": len(by_case)}
    for axis in AXES:
        plans = [r for r in prod.values() if r[f"{axis}_plan"]]
        per_case_distinct, per_case_n = [], []
        dup_resp = 0
        for rows in by_case.values():
            ps = [json.dumps(r[f"{axis}_plan"], sort_keys=True) for r in rows if r[f"{axis}_plan"]]
            per_case_n.append(len(ps))
            per_case_distinct.append(len(set(ps)) / len(ps) if ps else None)
        resp = Counter(r[f"{axis}_resp"] for r in plans if r[f"{axis}_resp"])
        dup_resp = sum(c for c in resp.values() if c > 1)
        coverage[axis] = {
            "videos_with_plan": len(plans),
            "plans_per_case": summary(per_case_n),
            "share_distinct_plan_json_per_case": summary(per_case_distinct),
            "videos_with_response_id": sum(resp.values()),
            "videos_sharing_a_response_id": dup_resp,
        }

    rng = random.Random(0)
    case_rows = {c: {"case": c} for c in sorted(by_case)}
    results = {"coverage": coverage}

    # ---- 1 & 2
    for axis in AXES:
        stab, f1_case, f1_obj_case, f1_cls_case, dep_all = {}, {}, {}, {}, {"ok": 0, "n": 0, "nt_ok": 0, "nt_n": 0}
        for case, rows in sorted(by_case.items()):
            plans = [r[f"{axis}_plan"] for r in rows if r[f"{axis}_plan"]]
            if len(plans) < 2:
                continue
            counts = [len(p) for p in plans]
            mode, mode_n = Counter(counts).most_common(1)[0]
            sd = pstdev(counts)
            stab[case] = {"n_plans": len(plans), "mean": mean(counts), "sd": sd,
                          "cv": sd / mean(counts) if mean(counts) else None,
                          "modal": mode, "modal_share": mode_n / len(plans),
                          "min": min(counts), "max": max(counts)}
            pairs = list(itertools.combinations(range(len(plans)), 2))
            if len(pairs) > MAX_PAIRS:
                pairs = rng.sample(pairs, MAX_PAIRS)
            f1s, f1o, f1c, dep_ok, dep_nt = [], [], [], [], []
            for i, j in pairs:
                c = compare(plans[i], plans[j])
                f1s.append(c["f1"])
                dep_ok += c["dep_ok"]
                dep_nt += c["dep_nontrivial"]
                f1o.append(compare(plans[i], plans[j], use_verb=False)["f1"])
                f1c.append(compare(plans[i], plans[j], use_verb="class")["f1"])
            f1_case[case], f1_obj_case[case], f1_cls_case[case] = mean(f1s), mean(f1o), mean(f1c)
            dep_all["ok"] += sum(dep_ok)
            dep_all["n"] += len(dep_ok)
            dep_all["nt_ok"] += sum(dep_nt)
            dep_all["nt_n"] += len(dep_nt)
            cr = case_rows[case]
            cr.update({f"{axis}_n_plans": len(plans), f"{axis}_count_mean": round(mean(counts), 3),
                       f"{axis}_count_sd": round(sd, 3), f"{axis}_count_modal": mode,
                       f"{axis}_modal_share": round(mode_n / len(plans), 3),
                       f"{axis}_count_min": min(counts), f"{axis}_count_max": max(counts),
                       f"{axis}_pair_f1": round(mean(f1s), 3), f"{axis}_pair_f1_objects_only": round(mean(f1o), 3),
                       f"{axis}_pair_f1_verb_class": round(mean(f1c), 3),
                       f"{axis}_dep_agreement": round(sum(dep_ok) / len(dep_ok), 3) if dep_ok else ""})
        cases = sorted(f1_case)
        results[axis] = {
            "count_stability": {
                "cases": len(stab),
                "mean_count": summary([s["mean"] for s in stab.values()]),
                "sd_count": summary([s["sd"] for s in stab.values()]),
                "cv_count": summary([s["cv"] for s in stab.values()]),
                "modal_share": summary([s["modal_share"] for s in stab.values()]),
                "share_cases_constant_count": mean(s["sd"] == 0 for s in stab.values()),
                "share_cases_range_le_1": mean(s["max"] - s["min"] <= 1 for s in stab.values()),
                "pooled_share_plans_at_modal_count": sum(s["modal_share"] * s["n_plans"] for s in stab.values())
                / sum(s["n_plans"] for s in stab.values()),
            },
            "content_agreement": {
                "matcher": f"same leading action verb + Jaccard(source+target tokens) >= {JACCARD}, greedy 1:1",
                "pairs_per_case_max": MAX_PAIRS,
                "pair_f1_case_mean": summary(list(f1_case.values())),
                "pair_f1_median_ci95": bootstrap(cases, lambda s: median(f1_case[c] for c in s)),
                "pair_f1_verb_class": summary(list(f1_cls_case.values())),
                "pair_f1_verb_class_median_ci95": bootstrap(cases, lambda s: median(f1_cls_case[c] for c in s)),
                "pair_f1_objects_only": summary(list(f1_obj_case.values())),
                "pair_f1_objects_only_median_ci95": bootstrap(cases, lambda s: median(f1_obj_case[c] for c in s)),
                "dependency_agreement_matched_pairs": dep_all["ok"] / dep_all["n"] if dep_all["n"] else None,
                "dependency_agreement_nontrivial": dep_all["nt_ok"] / dep_all["nt_n"] if dep_all["nt_n"] else None,
                "matched_pairs": dep_all["n"],
                "matched_pairs_with_any_requires": dep_all["nt_n"],
            },
        }
        results[axis]["_stab"] = stab

    # ---- 3a: plan count vs Task score within case
    tstab = results["task"].pop("_stab")
    results["physics"].pop("_stab")
    impact = {}
    for label, keep in (("human_positive", lambda r: r["model"] == "human_reference"),
                        ("all_videos", lambda r: True)):
        xs, ys, dev_groups, per_case = [], [], defaultdict(list), {}
        for case, rows in by_case.items():
            rows = [r for r in rows if keep(r) and r["task_plan"] and r["task"] is not None]
            if len(rows) < 2 or case not in tstab:
                continue
            mc = mean(len(r["task_plan"]) for r in rows)
            mt = mean(r["task"] for r in rows)
            per_case[case] = [(len(r["task_plan"]) - mc, r["task"] - mt) for r in rows]
            for r in rows:
                xs.append(len(r["task_plan"]) - mc)
                ys.append(r["task"] - mt)
                d = len(r["task_plan"]) - tstab[case]["modal"]
                dev_groups["below_modal" if d < 0 else "modal" if d == 0 else "above_modal"].append(r["task"])
        impact[label] = {
            "videos": len(xs),
            "within_case_spearman_count_vs_task": spearman(xs, ys),
            "within_case_spearman_ci95": bootstrap(
                sorted(per_case), lambda smp: spearman([x for c in smp for x, _ in per_case[c]],
                                                       [y for c in smp for _, y in per_case[c]])),
            "cases_with_count_variation": sum(any(x != 0 for x, _ in v) for v in per_case.values()),
            "within_case_pearson_count_vs_task": pearson(xs, ys),
            "by_count_deviation_from_case_modal": {
                k: {"videos": len(v), "mean_task": mean(v), "share_task_100": mean(t >= 99.999 for t in v)}
                for k, v in sorted(dev_groups.items())},
        }
    results["3a_plan_count_vs_task"] = impact

    # ---- 3c: per case SD of Human(+) Task vs SD of their plan counts
    xs, ys = [], []
    for case, rows in by_case.items():
        rows = [r for r in rows if r["model"] == "human_reference" and r["task_plan"] and r["task"] is not None]
        if len(rows) >= 2:
            xs.append(pstdev([len(r["task_plan"]) for r in rows]))
            ys.append(pstdev([r["task"] for r in rows]))
            case_rows[case]["humanpos_task_sd"] = round(ys[-1], 3)
            case_rows[case]["humanpos_count_sd"] = round(xs[-1], 3)
    results["3c_humanpos_task_sd_vs_count_sd"] = {
        "cases": len(xs), "spearman": spearman(xs, ys),
        "mean_task_sd_cases_constant_count": mean([y for x, y in zip(xs, ys) if x == 0]) if any(x == 0 for x in xs) else None,
        "mean_task_sd_cases_varying_count": mean([y for x, y in zip(xs, ys) if x > 0]) if any(x > 0 for x in xs) else None,
        "cases_constant_count": sum(x == 0 for x in xs)}

    # ---- 3b: production vs pilot
    pilot = json.loads(PILOT.read_text())
    pairs = []
    for p in pilot:
        r = prod.get(p["id"])
        if r is None or not any(("pilot", p["id"], axis) in cached for axis in AXES):
            continue
        rec = {"video": p["id"], "case": r["case"], "model": r["model"]}
        for axis in AXES:
            plan_b, _ = cached.get(("pilot", p["id"], axis), (None, None))
            plan_a = r[f"{axis}_plan"]
            a, b = r[axis], p.get(f"judge_{axis}")
            rec[f"{axis}_diff"] = None if a is None or b is None else a - b
            if plan_a and plan_b:
                rec[f"{axis}_dcount"] = abs(len(plan_a) - len(plan_b))
                rec[f"{axis}_f1"] = compare(plan_a, plan_b)["f1"]
                rec[f"{axis}_f1_obj"] = compare(plan_a, plan_b, use_verb=False)["f1"]
                rec[f"{axis}_f1_cls"] = compare(plan_a, plan_b, use_verb="class")["f1"]
                rec[f"{axis}_same_plan_json"] = json.dumps(plan_a, sort_keys=True) == json.dumps(plan_b, sort_keys=True)
            else:
                rec[f"{axis}_dcount"] = rec[f"{axis}_f1"] = rec[f"{axis}_f1_obj"] = None
        pairs.append(rec)

    def rb_stats(rows, axis):
        rows = [r for r in rows if r[f"{axis}_diff"] is not None and r[f"{axis}_f1"] is not None]
        if len(rows) < 3:
            return None
        ad = [abs(r[f"{axis}_diff"]) for r in rows]
        high = [r for r in rows if r[f"{axis}_dcount"] == 0 and r[f"{axis}_f1"] >= HIGH_F1]
        rest = [r for r in rows if not (r[f"{axis}_dcount"] == 0 and r[f"{axis}_f1"] >= HIGH_F1)]
        sd_all = single_run_sd([r[f"{axis}_diff"] for r in rows])
        sd_high = single_run_sd([r[f"{axis}_diff"] for r in high])
        sd_rest = single_run_sd([r[f"{axis}_diff"] for r in rest])
        return {
            "videos": len(rows),
            "same_plan_json_share": mean(r[f"{axis}_same_plan_json"] for r in rows),
            "same_count_share": mean(r[f"{axis}_dcount"] == 0 for r in rows),
            "plan_f1": summary([r[f"{axis}_f1"] for r in rows]),
            "plan_f1_objects_only": summary([r[f"{axis}_f1_obj"] for r in rows]),
            "spearman_absdiff_vs_dcount": spearman(ad, [r[f"{axis}_dcount"] for r in rows]),
            "spearman_absdiff_vs_f1": spearman(ad, [r[f"{axis}_f1"] for r in rows]),
            "spearman_absdiff_vs_f1_objects_only": spearman(ad, [r[f"{axis}_f1_obj"] for r in rows]),
            "high_agreement": {"videos": len(high), "mean_absdiff": mean(abs(r[f"{axis}_diff"]) for r in high) if high else None,
                               "single_run_sd": sd_high,
                               "share_zero_diff": mean(r[f"{axis}_diff"] == 0 for r in high) if high else None},
            "rest": {"videos": len(rest), "mean_absdiff": mean(abs(r[f"{axis}_diff"]) for r in rest) if rest else None,
                     "single_run_sd": sd_rest,
                     "share_zero_diff": mean(r[f"{axis}_diff"] == 0 for r in rest) if rest else None},
            "all": {"mean_absdiff": mean(ad), "single_run_sd": sd_all},
            "spearman_absdiff_vs_f1_verb_class": spearman(ad, [r[f"{axis}_f1_cls"] for r in rows]),
            "by_dcount": {str(k): {"videos": len(g), "mean_absdiff": mean(abs(r[f"{axis}_diff"]) for r in g),
                                   "single_run_sd": single_run_sd([r[f"{axis}_diff"] for r in g]),
                                   "share_zero_diff": mean(r[f"{axis}_diff"] == 0 for r in g)}
                          for k, g in sorted(_group(rows, lambda r: min(r[f"{axis}_dcount"], 2)).items())},
            "variance_share_excess_over_same_count":
                (1 - single_run_sd([r[f"{axis}_diff"] for r in rows if r[f"{axis}_dcount"] == 0]) ** 2 / sd_all ** 2)
                if sd_all and any(r[f"{axis}_dcount"] == 0 for r in rows) else None,
            "variance_share_excess_over_high_agreement":
                (1 - sd_high ** 2 / sd_all ** 2) if sd_high is not None and sd_all else None,
        }

    rb = {"definition_high_agreement": f"same subgoal count and matched F1 >= {HIGH_F1} (same axis plan)"}
    pcases = sorted({r["case"] for r in pairs})
    for axis in AXES:
        s = rb_stats(pairs, axis)
        if s is None:
            continue

        idx = defaultdict(list)
        for r in pairs:
            idx[r["case"]].append(r)

        def bstat(sample, fn, axis=axis):
            rows = [r for c in sample for r in idx[c]]
            st = rb_stats(rows, axis)
            return fn(st) if st else None

        s["spearman_absdiff_vs_f1_ci95"] = bootstrap(pcases, lambda smp: bstat(smp, lambda st: st["spearman_absdiff_vs_f1"]))
        s["spearman_absdiff_vs_dcount_ci95"] = bootstrap(pcases, lambda smp: bstat(smp, lambda st: st["spearman_absdiff_vs_dcount"]))
        s["mean_absdiff_dcount_pos_minus_zero_ci95"] = bootstrap(
            pcases, lambda smp: bstat(smp, _dcount_gap))
        s["mean_absdiff_rest_minus_high_ci95"] = bootstrap(
            pcases, lambda smp: bstat(smp, lambda st: (st["rest"]["mean_absdiff"] - st["high_agreement"]["mean_absdiff"])
                                      if st["rest"]["mean_absdiff"] is not None and st["high_agreement"]["mean_absdiff"] is not None else None))
        rb[axis] = s
    rows = [r for r in pairs if r["physics_diff"] is not None and r["task_f1"] is not None]
    rb["cross"] = {"spearman_abs_dphysics_vs_task_plan_f1": spearman([abs(r["physics_diff"]) for r in rows], [r["task_f1"] for r in rows])}
    rb["pairs"] = len(pairs)
    rb["cases"] = len(pcases)
    results["3b_production_vs_pilot"] = rb

    OUT_JSON.write_text(json.dumps(results, indent=2))
    fields = sorted({k for r in case_rows.values() for k in r}, key=lambda k: (k != "case", k))
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in case_rows.values():
            w.writerow(r)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
