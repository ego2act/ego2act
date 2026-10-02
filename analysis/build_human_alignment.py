#!/usr/bin/env python
"""Inter-annotator agreement and evaluator-human alignment (Table 3), and the
backbone ablation.

Inputs: analysis/csv/human_annot_original.csv, scores_human.csv, scores_baselines.csv, scores_terra.csv.
Outputs: analysis/results/human_alignment.{csv,json,tex}, human_alignment_figure.csv,
backbone_ablation.tex.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from scoring import final_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "csv"
OUT = HERE / "results"
EVALUATORS = ["wr-arena", "pqsg", "rbench", "worldmodelbench", "videoscore", "simple_vqa", "ego2act"]
AXES = ["task", "physics", "final"]


def read(path: Path, skip_note: bool = False) -> list[dict]:
    with path.open(newline="") as stream:
        if skip_note:
            next(stream)
        return list(csv.DictReader(stream))


def number(value):
    return None if value in ("", None) else float(value)


# ---------------------------------------------------------------- agreement

def krippendorff_interval(groups: list[list[float]]) -> float:
    values = [v for g in groups for v in g]
    mean = sum(values) / len(values)
    expected = 2 * sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    observed = sum(2 * sum((a - b) ** 2 for a, b in itertools.combinations(g, 2)) / (len(g) - 1)
                   for g in groups) / len(values)
    return 1 - observed / expected


def oneway_reml_icc(groups: list[list[float]]) -> float:
    """REML fit of y = mu + video + error; returns average-rater reliability."""
    n_total = sum(len(g) for g in groups)
    stats = [(len(g), sum(g) / len(g), sum((v - sum(g) / len(g)) ** 2 for v in g)) for g in groups]

    def criterion(log_ratio):  # -2 * restricted log-likelihood, profiled over sigma_error^2
        ratio = math.exp(log_ratio)
        weights = [n / (1 + n * ratio) for n, _, _ in stats]
        mu = sum(w * m for w, (_, m, _) in zip(weights, stats)) / sum(weights)
        q = sum(within + w * (m - mu) ** 2 for w, (_, m, within) in zip(weights, stats))
        return ((n_total - 1) * math.log(q) + sum(math.log(1 + n * ratio) for n, _, _ in stats)
                + math.log(sum(weights)))

    low, high = -20.0, 10.0  # golden-section search on log(sigma_video^2 / sigma_error^2)
    golden = (math.sqrt(5) - 1) / 2
    for _ in range(200):
        a, b = high - golden * (high - low), low + golden * (high - low)
        if criterion(a) < criterion(b):
            high = b
        else:
            low = a
    ratio = math.exp((low + high) / 2)
    k = len(groups) / sum(1 / len(g) for g in groups)
    return ratio / (ratio + 1 / k)


def agreement() -> list[dict]:
    ratings = defaultdict(lambda: defaultdict(list))
    for row in read(DATA / "human_annot_original.csv", skip_note=True):
        for i in (1, 2, 3):
            task, physics = number(row[f"rater_{i}_task"]), number(row[f"rater_{i}_phy"])
            if task is not None:
                ratings["task"][row["video_id"]].append(task)
            if physics is not None:
                ratings["physics"][row["video_id"]].append(physics)
            final = final_score(None if task is None else task * 100 / 3,
                                None if physics is None else physics * 25)
            if final is not None:
                ratings["final"][row["video_id"]].append(final / 100)
    rows = []
    for axis in AXES:
        groups = [g for g in ratings[axis].values() if len(g) >= 2]
        rows.append({"axis": axis, "videos": len(groups), "ratings": sum(len(g) for g in groups),
                     "icc": oneway_reml_icc(groups), "krippendorff_alpha": krippendorff_interval(groups)})
    return rows


# ---------------------------------------------------------------- alignment

def tau_b(x, y):
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return None if not denominator else (concordant - discordant) / denominator


def metrics(pairs, with_tau=True):
    """pairs: (human, evaluator). Bias is evaluator minus human."""
    n = len(pairs)
    if n < 2:
        return {"n": n, "pearson": None, "kendall_tau": None, "ccc": None, "mae": None, "bias": None}
    x, y = [a for a, _ in pairs], [b for _, b in pairs]
    mx, my = sum(x) / n, sum(y) / n
    sx, sy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    cross = sum((a - mx) * (b - my) for a, b in pairs)
    return {
        "n": n,
        "pearson": cross / math.sqrt(sx * sy) if sx and sy else None,
        "kendall_tau": tau_b(x, y) if with_tau else None,
        "ccc": 2 * (cross / n) / (sx / n + sy / n + (mx - my) ** 2),
        "mae": sum(abs(a - b) for a, b in pairs) / n,
        "bias": my - mx,
    }


def evaluator_scores() -> dict:
    scores = defaultdict(dict)
    for row in read(DATA / "scores_baselines.csv"):
        scores[row["evaluator"]][row["video_id"]] = {
            "task": number(row["task"]), "physics": number(row["physics"]),
            "final": None if row["evaluator"] == "wr-arena" else number(row["native_0_100"])}
    for row in read(DATA / "scores_simple_vqa.csv"):
        task, physics = number(row["task"]), number(row["physics"])
        scores["simple_vqa"][row["video_id"]] = {"task": task, "physics": physics,
                                                 "final": final_score(task, physics, number(row["final"]))}
    for row in read(DATA / "scores_ego2act.csv"):
        scores["ego2act"][row["video_id"]] = {
            "task": number(row["ego2act_task"]), "physics": number(row["ego2act_physics"]),
            "final": final_score(number(row["ego2act_task"]), number(row["ego2act_physics"]),
                                 number(row["ego2act_score"]))}
    return scores


def alignment() -> list[dict]:
    human = {row["id"]: {"task": number(row["human_task"]), "physics": number(row["human_physics"]),
                         "final": final_score(number(row["human_task"]), number(row["human_physics"]),
                                              number(row["human_final_score"]))}
             for row in read(DATA / "scores_human.csv")}
    scores = evaluator_scores()
    rows = []
    for evaluator in EVALUATORS:
        for axis in AXES:
            pairs = [(human[v][axis], s[axis]) for v, s in scores[evaluator].items()
                     if v in human and human[v][axis] is not None and s[axis] is not None]
            rows.append({"evaluator": evaluator, "axis": axis, **metrics(pairs)})
    # Backbone ablation: the judge with GPT-5.6-Terra instead of Gemini Flash,
    # on the videos Terra scored successfully within the ego2act alignment set.
    terra = {row["video_id"]: {"task": number(row["task"]), "physics": number(row["physics"]),
                               "final": final_score(number(row["task"]), number(row["physics"]), number(row["final"]))}
             for row in read(DATA / "scores_terra.csv")}
    for axis in AXES:
        pairs = [(human[v][axis], t[axis]) for v, t in terra.items()
                 if v in human and human[v][axis] is not None and t[axis] is not None
                 and scores["ego2act"].get(v, {}).get(axis) is not None]
        rows.append({"evaluator": "terra", "axis": axis, **metrics(pairs)})
        # Flash on exactly the same videos, so the backbone comparison is matched.
        matched = [(human[v][axis], scores["ego2act"][v][axis]) for v, t in terra.items()
                   if v in human and human[v][axis] is not None and t[axis] is not None
                   and scores["ego2act"].get(v, {}).get(axis) is not None]
        rows.append({"evaluator": "flash_matched_to_terra", "axis": axis, **metrics(matched)})
    # Human reference: each complete rating against the other raters' consensus.
    rows.append({"evaluator": "human_leave_one_out", "axis": "final",
                 **metrics([(h, e) for _, h, e in human_leave_one_out()])})
    return rows


# ---------------------------------------------------------------- figure data

def human_leave_one_out() -> list[tuple]:
    """(case, rater Final, Final of the other raters' axis means) per complete rating."""
    triples = []
    for row in read(DATA / "human_annot_original.csv", skip_note=True):
        raters = [(number(row[f"rater_{i}_task"]), number(row[f"rater_{i}_phy"])) for i in (1, 2, 3)]
        raters = [(t * 100 / 3, None if p is None else p * 25) for t, p in raters if t is not None]
        for i, (task, physics) in enumerate(raters):
            others = raters[:i] + raters[i + 1:]
            mine = final_score(task, physics)
            if not others or mine is None:
                continue
            mean_task = sum(t for t, _ in others) / len(others)
            phys = [p for _, p in others if p is not None]
            consensus = final_score(mean_task, sum(phys) / len(phys) if phys else None)
            if consensus is not None:
                triples.append((row["case_id"], consensus, mine))
    return triples


def figure_data(resamples: int = 1000, seed: int = 0) -> list[dict]:
    """Final-score MAE, bias, CCC and r with case-bootstrap 95% intervals."""
    import random

    human = {row["id"]: (row["case_id"],
                         final_score(number(row["human_task"]), number(row["human_physics"]),
                                     number(row["human_final_score"])),
                         number(row["human_task"]))
             for row in read(DATA / "scores_human.csv")}
    scores = evaluator_scores()
    sets = {}
    for evaluator in EVALUATORS:
        axis = "task" if evaluator == "wr-arena" else "final"
        sets[evaluator] = [(human[v][0], human[v][2 if axis == "task" else 1], s[axis])
                           for v, s in scores[evaluator].items()
                           if v in human and human[v][2 if axis == "task" else 1] is not None and s[axis] is not None]
    sets["human"] = human_leave_one_out()
    rng = random.Random(seed)
    names = ("mae", "bias", "ccc", "pearson")
    rows = []
    for evaluator, triples in sets.items():
        cases = defaultdict(list)
        for case, h, e in triples:
            cases[case].append((h, e))
        keys = sorted(cases)
        point = metrics([p for k in keys for p in cases[k]])
        draws = defaultdict(list)
        for _ in range(resamples):
            m = metrics([p for k in rng.choices(keys, k=len(keys)) for p in cases[k]], with_tau=False)
            for name in names:
                draws[name].append(m[name])
        for name in names:
            values = sorted(draws[name])
            rows.append({"evaluator": evaluator, "metric": name, "value": point[name], "n": point["n"],
                         "low": values[int(0.025 * resamples)], "high": values[int(0.975 * resamples) - 1]})
    return rows


# ---------------------------------------------------------------- latex

def fmt(value, digits=2):
    if value is None:
        return "--"
    text = f"{value:.{digits}f}"
    return text.replace("-", "$-$") if value < 0 else text


def latex(agree: list[dict], align: list[dict]) -> str:
    a = {r["axis"]: r for r in agree}
    m = {(r["evaluator"], r["axis"]): r for r in align}
    labels = {"wr-arena": "WRA", "pqsg": "PQSG", "rbench": "RBench", "worldmodelbench": "WMB",
              "videoscore": "VScore", "simple_vqa": r"\textit{Simple VQA}", "ego2act": r"\textbf{Ours}"}
    lines = ["% (a) Rubric agreement: ICC and Krippendorff's alpha (Final, Task, Physics)",
             rf"ICC               & \multicolumn{{2}}{{c}}{{{a['final']['icc']:.3f}}} & {a['task']['icc']:.3f} & {a['physics']['icc']:.3f} \\",
             rf"Kripp.'s $\alpha$ & \multicolumn{{2}}{{c}}{{{a['final']['krippendorff_alpha']:.3f}}} & {a['task']['krippendorff_alpha']:.3f} & {a['physics']['krippendorff_alpha']:.3f} \\",
             "% (b) Final-score alignment (WR-Arena: Task): r, tau, CCC, MAE"]
    for evaluator in EVALUATORS:
        r = m[evaluator, "task" if evaluator == "wr-arena" else "final"]
        cells = [fmt(r["pearson"]), fmt(r["kendall_tau"]), fmt(r["ccc"]), fmt(r["mae"], 1)]
        if evaluator == "ego2act":
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(f"{labels[evaluator]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def backbone_latex(align: list[dict]) -> str:
    """Table rows: Terra vs Flash, both on the videos Terra scored."""
    m = {(r["evaluator"], r["axis"]): r for r in align}
    lines = []
    for axis in AXES:
        terra, flash = m["terra", axis], m["flash_matched_to_terra", axis]
        lines.append(f"% {axis.capitalize()}: Terra n={terra['n']}, Flash n={flash['n']}")
        lines.append(f"      & Terra        & {terra['pearson']:.3f} & {terra['mae']:.2f} \\\\")
        lines.append(r"      & \cellcolor{lightgreen}\textbf{Flash} & "
                     rf"\cellcolor{{lightgreen}}\textbf{{{flash['pearson']:.3f}}} & "
                     rf"\cellcolor{{lightgreen}}\textbf{{{flash['mae']:.2f}}} \\")
    return "\n".join(lines) + "\n"


def main():
    agree, align = agreement(), alignment()
    OUT.mkdir(exist_ok=True)
    with (OUT / "human_alignment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(align[0]))
        writer.writeheader()
        writer.writerows(align)
    figure = figure_data()
    with (OUT / "human_alignment_figure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(figure[0]))
        writer.writeheader()
        writer.writerows(figure)
    (OUT / "human_alignment.json").write_text(json.dumps({"agreement": agree, "alignment": align}, indent=2) + "\n")
    (OUT / "human_alignment.tex").write_text(latex(agree, align))
    (OUT / "backbone_ablation.tex").write_text(backbone_latex(align))
    print(json.dumps(agree, indent=2))


if __name__ == "__main__":
    main()
