"""Every paper table as a compile-ready .tex file:  python analysis/build_paper_tables.py

Reads only analysis/results/ (written by the other builders) and writes one file per table to
analysis/tables/<name>.tex. Each file holds the complete table environment(s): caption, label,
column specification, colour macros and booktabs rules, with best / second-best marking recomputed
or copied from the row fragments of the other builders. Captions are kept below, verbatim from the
paper, so a fresh compile needs no hand edits.

Not generated: tab:human-rubric-ablation (needs the pilot gate annotations, not in the repository).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
RESULTS = ANALYSIS / "results"
OUT = ANALYSIS / "tables"

GENERATORS = [("grok_imagine_video_1_5", "Grok-1.5"), ("seedance_2_0", "Seedance-2.0"),
              ("kling_v3_pro", "Kling-v3-Pro"), ("wan_2_7", "Wan-2.7"),
              ("minimax_h3", "MiniMax-H3"), ("cosmos_3", "Cosmos-3")]

# ----------------------------------------------------------------------------- captions (paper text)
CAPTIONS = {
    "main": r"""\textbf{Cross-Benchmark Performance Leaderboard on \method{}.} Sub-component \textcolor{taskorange}{\textbf{Task}} / \textcolor{physicsblue}{\textbf{Physics}} splits are reported alongside average of per video geometric mean or benchmark official score, with human controls excluded from \textbf{best} and \underline{second-best} model rankings.""",
    "combined_metrics": r"""\centering Inter-annotator Agreement and Alignment.""",
    "backbone_ablation": r"""\centering \textbf{Backbone ablation} across evaluation axes. Both backbones are compared on the same videos.""",
    "task_ci": r"""\textcolor{taskorange}{\textbf{Task}} scores (0--100) with 95\% confidence intervals. Values match Table~\ref{tab:main_table}, and subscripts give 95\% case-level bootstrap intervals (1,000 resamples of cases, keeping all videos of a sampled case). Human uses the 600-video panel, while automated evaluators use every benchmark video they scored. Human (+) recordings are verified as successful executions during dataset construction (Section~\ref{sec:data-collection}) and receive the rubric maximum.""",
    "physics_ci": r"""\textcolor{physicsblue}{\textbf{Physics}} scores (0--100) with 95\% confidence intervals. Values match Table~\ref{tab:main_table}, and subscripts give 95\% case-level bootstrap intervals (1,000 resamples of cases, keeping all videos of a sampled case). Human uses the 600-video panel, while automated evaluators use every benchmark video they scored. Human (+) recordings are verified as successful executions during dataset construction (Section~\ref{sec:data-collection}) and receive the rubric maximum. WR-Arena has no Physics component.""",
    "domain_models": r"""Domain profiles of generated rollouts. Values are case-balanced Final scores under \judge{}, and each domain contains five cases.""",
    "feature_detail": r"""Lower scores in some interaction categories motivate targeted failure analysis, but the groups are small and overlap. Scores are descriptive case-balanced Final scores under \judge{}. Clutter labels cover every panel case.""",
    "feature_qvalues": r"""Adjusted $q$-values for the feature diagnostics. No displayed association meets $q<0.05$. The interaction tests shown here are those in Figure~\ref{fig:feature-significance}b, which cover every action family with at least three cases on each side. Values are Benjamini--Hochberg adjusted within each panel of Figure~\ref{fig:feature-significance}.""",
    "judge_gate_ablation": r"""Judge-side gating ablation. \judge{} is rerun with every gate asked independently, and both rules are scored from the same answers. Rows compare each rule with the human consensus on the same videos. $\Delta$ is independent minus sequential with 95\% case-bootstrap intervals.""",
    "prompt_expansion": r"""Prompt-expansion diagnostic. Mean \judge{} Final score (three seeds) with the original prompt and with each of three expanded prompts, scored against the original goal. Unattempted videos count as 0.""",
    "repeatability": r"""Judge repeatability and generation variability. Judge rows compare two independent runs of \judge{} on the same videos, with subgoals re-derived in each run. Single-run SD is $\sqrt{\overline{d^2}/2}$ for run differences $d$. Generation variability is the SD of the \judge{} Final score across the three seeds of a case, averaged over cases, and it includes judge noise. Both are per-video quantities, not uncertainties of model means.""",
    "evaluator_alignment": r"""Task and Physics alignment with the human consensus on the 600-video panel. Bold marks the highest automated point estimate. A dash indicates an unsupported axis.""",
    "unified_evaluator_alignment": r"""Final-score alignment with the human consensus on the 600-video panel. Values match Table~\ref{tab:combined_metrics}. MAE and signed bias (evaluator minus human) are in $0$--$100$ units. Bold marks the best automated point estimate, which does not establish statistical significance. $\dagger$ denotes VideoScore's five-aspect mean diagnostic. $^{\S}$WR-Arena has no Final score and is compared on Task. $^{\ddagger}$Human reference: each rating compared with the consensus of the other raters on the same video.""",
    "coverage": r"""Score coverage of \judge{} over all benchmark videos (\%). Physics is missing when no attempted interaction can be inspected. Unattempted videos (Task 0, no Physics) receive $S=0$ and count as scored. The remaining missing Final scores come from unresolved gate decisions or from videos with $T>0$ but no inspectable interaction.""",
    "panel_vs_benchmark": r"""Mean Final score (rank) per generator. The first two columns use the same panel videos, and the last column is the \judge{} Agg.\ of Table~\ref{tab:main_table} over all benchmark videos.""",
    "physics_controls": r"""Physics negative controls on 30 real recordings. Detection is the share of pairs whose Physics score falls below the unedited rerun (29 pairs with a Physics score). Changes are corrupted minus unedited, with 95\% bootstrap intervals over recordings. False alarms on unedited videos: 0 of 29.""",
}


# ----------------------------------------------------------------------------- helpers
def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_csv(name: str) -> list[dict]:
    with open(RESULTS / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fragment(name: str) -> dict[str, list[str]]:
    """Row fragment file -> {section: rows}. A '% <name>' comment line starts a new section."""
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("%"):
            current = line.lstrip("% ").strip()
            sections[current] = []
        elif line.strip():
            sections[current].append(line.strip())
    return sections


def fixed(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def minus_text(text: str) -> str:
    """Text-mode minus sign, as used in the paper's alignment tables."""
    return "$-$" + text[1:] if text.startswith("-") else text


def signed_text(value: float, digits: int = 1) -> str:
    text = fixed(value, digits)
    return minus_text(text) if text.startswith("-") else ("+" + text)


def signed_math(value: float, digits: int) -> str:
    """Math-mode signed number; the sign follows the unrounded value (so -0.0004 prints $-0.00$)."""
    return f"${'-' if value < 0 else '+'}{fixed(abs(value), digits)}$"


def plain_math(value: float, digits: int = 1) -> str:
    """Negative values in math mode, zero and positive values bare (physics-controls table)."""
    text = fixed(value, digits)
    return f"$-{text[1:]}$" if text.startswith("-") and float(text) != 0 else text.lstrip("-")


def alignment_index() -> dict[tuple[str, str], dict]:
    return {(r["evaluator"], r["axis"]): r for r in load_json("human_alignment.json")["alignment"]}


def emit(name: str, body: str, source: str) -> None:
    OUT.mkdir(exist_ok=True)
    text = f"% generated by analysis/build_paper_tables.py from {source}; do not edit by hand\n{body.rstrip()}\n"
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")


def table(env: str, caption: str, label: str, settings: str, tabular: str, placement: str = "htbp",
          centering: str = r"\centering\small", before_caption: str = "") -> str:
    """Table environment. `settings` follows the label (as in the paper); `before_caption` precedes the caption."""
    return (f"\\begin{{{env}}}[{placement}]\n{centering}\n{before_caption}"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n{settings}{tabular}\\end{{{env}}}")


# ----------------------------------------------------------------------------- main table
def build_main() -> None:
    rows = "\n".join(fragment("leaderboard.tex")[""])
    body = r"""\begin{table*}[t]
\caption{""" + CAPTIONS["main"] + r"""}
\label{tab:main_table}
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.1}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l | c >{\columncolor{gray!12}}c | >{\columncolor{gray!12}}c | c >{\columncolor{gray!12}}c | c >{\columncolor{gray!12}}c | c >{\columncolor{gray!12}}c | c >{\columncolor{gray!12}}c | c >{\columncolor{blue!10}}c}
\toprule
 & \multicolumn{2}{c|}{Human Judge$^{\dagger}$} & WRA & \multicolumn{2}{c|}{PQSG} & \multicolumn{2}{c|}{RBench} & \multicolumn{2}{c|}{WMBench} & \multicolumn{2}{c|}{VideoScore} & \multicolumn{2}{c}{\textbf{\judge{}}} \\
\textbf{Gen. / Eval.} & \textbf{\textcolor{taskorange}{T}/\textcolor{physicsblue}{P}} & \textbf{Agg.} & \textbf{Ori.} & \textbf{\textcolor{taskorange}{T}/\textcolor{physicsblue}{P}} & \textbf{Ori.} & \textbf{\textcolor{taskorange}{T}/\textcolor{physicsblue}{P}} & \textbf{Ori.} & \textbf{\textcolor{taskorange}{T}/\textcolor{physicsblue}{P}} & \textbf{Ori.} & \textbf{\textcolor{taskorange}{T}/\textcolor{physicsblue}{P}} & \textbf{Ori.} & \textbf{\textcolor{taskorange}{T}/\textcolor{physicsblue}{P}} & \textbf{Agg.} \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}%
}
\par\vspace{0.2em}
\begin{minipage}{\linewidth}\tiny
$^{\dagger}$Human Judge scores come from the 600-video alignment panel (App.~\ref{app:human-alignment}). Human (+) and Human ($-$) are human recordings of correct and incorrect task executions, and human (+) recordings are verified as successful executions during dataset construction (Section~\ref{sec:data-collection}) and receive the rubric maximum. Baseline \textcolor{taskorange}{\textbf{T}}/\textcolor{physicsblue}{\textbf{P}} mappings and Ori.\ are defined in App.~\ref{app:baseline-map}, and Agg.\ in App.~\ref{app:score-aggregation}.
\end{minipage}
\end{table*}"""
    emit("main", body, "results/leaderboard.tex")


# ----------------------------------------------------------------------------- combined metrics
def build_combined_metrics() -> None:
    data = load_json("human_alignment.json")
    agreement = {a["axis"]: a for a in data["agreement"]}
    align = alignment_index()
    columns = [("WRA", "wr-arena", "task"), ("PQSG", "pqsg", "final"), ("RBench", "rbench", "final"),
               ("WMB", "worldmodelbench", "final"), ("VScore", "videoscore", "final"),
               ("Simple VQA", "simple_vqa", "final"), ("Ours", "ego2act", "final")]

    def cell(key: str, digits: int, evaluator: str, axis: str) -> str:
        text = fixed(align[(evaluator, axis)][key], digits)
        return r"\textbf{" + text + "}" if evaluator == "ego2act" else text

    def row(label: str, key: str, digits: int) -> str:
        cells = " & ".join(cell(key, digits, e, a) for _, e, a in columns)
        return f"        {label:<21} & {cells} \\\\"

    icc = " & ".join(fixed(agreement[a]["icc"], 3) for a in ("final", "task", "physics"))
    alpha = " & ".join(fixed(agreement[a]["krippendorff_alpha"], 3) for a in ("final", "task", "physics"))
    body = r"""\begin{table}[t]
    \centering
    \small
    \caption{""" + CAPTIONS["combined_metrics"] + r"""}
    \label{tab:combined_metrics}
    \setlength{\tabcolsep}{3.5pt}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}[t]{@{}l c c c@{}}
        \toprule
        \multicolumn{4}{c}{\textbf{(a) Rubric Agreement}} \\
        \textbf{Metric} $\uparrow$ & \textbf{Final} & \textbf{Task} & \textbf{Physics} \\
        \midrule
        ICC               & """ + icc + r""" \\
        Kripp.'s $\alpha$ & """ + alpha + r""" \\
        \bottomrule
    \end{tabular}\hspace{14pt}%
    \begin{tabular}[t]{@{}l c c c c c | c >{\columncolor{lightgreen}}c@{}}
        \toprule
        \multicolumn{8}{c}{\textbf{(b) \judge{} Alignment}} \\
        \textbf{Metric} & \textbf{WRA} & \textbf{PQSG} & \textbf{RBench} & \textbf{WMB} & \textbf{VScore} & \textit{Simple VQA} & \textbf{Ours} \\
        \midrule
""" + "\n".join([row(r"$r$ $\uparrow$", "pearson", 2), row(r"$\tau$ $\uparrow$", "kendall_tau", 2),
                 row(r"CCC $\uparrow$", "ccc", 2), row(r"MAE $\downarrow$", "mae", 1)]) + r"""
        \bottomrule
    \end{tabular}}
\end{table}"""
    emit("combined_metrics", body, "results/human_alignment.json")


# ----------------------------------------------------------------------------- backbone ablation
def build_backbone_ablation() -> None:
    align = alignment_index()
    lines = []
    for i, axis in enumerate(("task", "physics", "final")):
        terra, flash = align[("terra", axis)], align[("flash_matched_to_terra", axis)]
        if i:
            lines.append(r"    \midrule")
        lines += [f"    \\multirow{{2}}{{*}}{{{axis.capitalize()}}}",
                  f"      & Terra        & {fixed(terra['pearson'], 3)} & {fixed(terra['mae'], 2)} \\\\",
                  f"      & \\cellcolor{{lightgreen}}\\textbf{{Flash}} & \\cellcolor{{lightgreen}}\\textbf{{{fixed(flash['pearson'], 3)}}}"
                  f" & \\cellcolor{{lightgreen}}\\textbf{{{fixed(flash['mae'], 2)}}} \\\\"]
    tabular = (r"""\begin{tabular}{@{}l l c c@{}}
    \toprule
    \textbf{Axis} & \textbf{Model} & \textbf{$r$} $\uparrow$ & \textbf{MAE} $\downarrow$ \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
\end{tabular}
""")
    body = table("table", CAPTIONS["backbone_ablation"], "tab:backbone_ablation",
                 "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}%\n\\renewcommand{\\arraystretch}{1.05}%\n",
                 tabular, placement="t", centering=r"\centering")
    emit("backbone_ablation", body, "results/human_alignment.json")


# ----------------------------------------------------------------------------- Task / Physics CI
def build_ci_task_physics() -> None:
    rows = fragment("confidence_intervals.tex")
    ci_head = (r"""\resizebox{\textwidth}{!}{%
\begin{tabular}{l!{\color{gray!60}\vrule width 0.5pt}>{\columncolor{gray!10}}c c c c c c >{\columncolor{blue!10}}c}
\toprule
Gen.\ / Eval. & Human & WR-Arena & PQSG & RBench & WMBench & VideoScore & \judge{} \\
\midrule
""")
    ci_foot = "\\bottomrule\n\\end{tabular}%\n}\n"

    def one(caption: str, label: str, key: str) -> str:
        return table("table*", caption, label, "", ci_head + "\n".join(rows[key]) + "\n" + ci_foot,
                     before_caption="\\setlength{\\tabcolsep}{3pt}\n")
    body = (one(CAPTIONS["task_ci"], "tab:task-ci", "Task rows") + "\n\n"
            + one(CAPTIONS["physics_ci"], "tab:physics-ci", "Physics rows"))
    emit("ci_task_physics", body, "results/confidence_intervals.tex")


# ----------------------------------------------------------------------------- case-feature tables
def build_feature_tables() -> None:
    rows = "\n".join(fragment("domain_models.tex")[""])
    emit("domain_models", table("table", CAPTIONS["domain_models"], "tab:domain-models",
         "", "\\setlength{\\tabcolsep}{5pt}\n"
         "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nGenerator & Household & Kitchen & Office & Personal care & Other \\\\\n"
         "\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n"), "results/domain_models.tex")
    rows = "\n".join(fragment("feature_detail.tex")[""])
    emit("feature_detail", table("table", CAPTIONS["feature_detail"], "tab:feature-detail", "",
         "\\begin{tabular}{@{}llrr@{}}\n\\toprule\nFeature & Group & Cases & Score \\\\\n\\midrule\n"
         + rows + "\n\\bottomrule\n\\end{tabular}\n"), "results/feature_detail.tex")
    rows = "\n".join(fragment("feature_qvalues.tex")[""])
    emit("feature_qvalues", table("table", CAPTIONS["feature_qvalues"], "tab:feature-qvalues", "",
         "\\setlength{\\tabcolsep}{6pt}\n\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
         "Case feature & Generator-score $q$ & Judge-error $q$ \\\\\n\\midrule\n"
         + rows + "\n\\bottomrule\n\\end{tabular}\n"), "results/feature_qvalues.tex")


# ----------------------------------------------------------------------------- judge gate ablation
def build_judge_gate_ablation() -> None:
    data = load_json("gate_ablation.json")
    rows = {(r["axis"], r["scoring"]): r for r in data["alignment"]}
    lines = []
    for i, axis in enumerate(("task", "physics", "final")):
        if i:
            lines.append(r"\midrule")
        for scoring in ("sequential", "independent"):
            r = rows[(axis, scoring)]
            label = axis.capitalize() if scoring == "sequential" else ""
            lines.append(f"{label} & {scoring.capitalize()} & {fixed(r['pearson'], 2)} & {fixed(r['kendall_tau'], 2)}"
                         f" & {fixed(r['ccc'], 2)} & {fixed(r['mae'], 1)} & {signed_text(r['bias'])} \\\\")
        b = data["bootstrap"][axis]
        r_ci, m_ci = b["pearson"], b["mae"]
        lines.append(f" & $\\Delta$ & \\multicolumn{{3}}{{l}}{{$r$: {signed_math(r_ci['difference'], 2)} "
                     f"[{signed_math(r_ci['low'], 2)}, {signed_math(r_ci['high'], 2)}]}} & "
                     f"\\multicolumn{{2}}{{l}}{{{signed_math(m_ci['difference'], 1)} "
                     f"[{signed_math(m_ci['low'], 1)}, {signed_math(m_ci['high'], 1)}]}} \\\\")
    tabular = ("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
               "Axis & Scoring & $r$ $\\uparrow$ & $\\tau$ $\\uparrow$ & CCC $\\uparrow$ & MAE $\\downarrow$ & Bias \\\\\n"
               "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    emit("judge_gate_ablation", table("table", CAPTIONS["judge_gate_ablation"], "tab:judge-gate-ablation",
         "\\setlength{\\tabcolsep}{4pt}\n", tabular), "results/gate_ablation.json")


# ----------------------------------------------------------------------------- prompt expansion
def build_prompt_expansion() -> None:
    data = load_json("prompt_expansion.json")
    rows = data["rows"]
    lines = []
    for case in data["cases"]:
        cells = []
        for model in ("cosmos_3", "wan_2_7"):
            sel = sorted((r for r in rows if r["case"] == case and r["model"] == model), key=lambda r: r["variant"])
            cells += [fixed(sel[0]["original"], 1), " / ".join(fixed(r["expanded"], 1) for r in sel)]
        lines.append(f"\\texttt{{{case.replace('_', chr(92) + '_')}}} & " + " & ".join(cells) + " \\\\")
    s = data["summary"]
    mean = (f"Mean & {fixed(s['cosmos_3']['original_mean'], 1)} & {fixed(s['cosmos_3']['expanded_mean'], 1)}"
            f" & {fixed(s['wan_2_7']['original_mean'], 1)} & {fixed(s['wan_2_7']['expanded_mean'], 1)} \\\\")
    tabular = (r"""\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Cosmos-3} & \multicolumn{2}{c}{Wan-2.7} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Case & Original & Expanded 1 / 2 / 3 & Original & Expanded 1 / 2 / 3 \\
\midrule
""" + "\n".join(lines) + "\n\\midrule\n" + mean + "\n\\bottomrule\n\\end{tabular}}\n")
    emit("prompt_expansion", table("table", CAPTIONS["prompt_expansion"], "tab:prompt-expansion",
                                   "\\setlength{\\tabcolsep}{4pt}\n", tabular), "results/prompt_expansion.json")


# ----------------------------------------------------------------------------- repeatability
def build_repeatability() -> None:
    data = load_json("repeatability.json")
    judge, sds = data["judge"], [g["mean_seed_sd"] for g in data["generation"].values()]
    lines = []
    for i, axis in enumerate(("task", "physics", "final")):
        label = "Judge repeatability" if i == 0 else ""
        lines.append(f"{label} & {axis.capitalize()} & {fixed(judge[axis]['single_run_sd'], 1)}"
                     f" & {fixed(judge[axis]['retest_r'], 2)} \\\\")
    tabular = ("\\begin{tabular}{@{}llrr@{}}\n\\toprule\nQuantity & Axis & SD & Test--retest $r$ \\\\\n\\midrule\n"
               + "\n".join(lines) + "\n\\midrule\n"
               + f"Generation variability & Final & {fixed(min(sds), 1)}--{fixed(max(sds), 1)} & -- \\\\\n"
               + "\\bottomrule\n\\end{tabular}\n")
    emit("repeatability", table("table", CAPTIONS["repeatability"], "tab:repeatability",
                                "\\setlength{\\tabcolsep}{5pt}\n", tabular), "results/repeatability.json")


# ----------------------------------------------------------------------------- evaluator alignment
EVALUATORS = [("wr-arena", "WR-Arena"), ("pqsg", "PQSG"), ("rbench", "RBench"), ("worldmodelbench", "WMBench"),
              ("videoscore", "VideoScore"), ("simple_vqa", "Simple VQA"), ("ego2act", r"\judge{}")]


def build_evaluator_alignment() -> None:
    align = alignment_index()
    def value(evaluator, axis, key):
        row = align[(evaluator, axis)]
        return row[key] if row["n"] else None
    best = {(axis, key): max(v for e, _ in EVALUATORS if (v := value(e, axis, key)) is not None)
            for axis in ("task", "physics") for key in ("pearson", "kendall_tau")}
    lines = []
    for evaluator, label in EVALUATORS:
        cells = []
        for axis in ("task", "physics"):
            for key in ("pearson", "kendall_tau"):
                v = value(evaluator, axis, key)
                if v is None:
                    cells.append("--")
                    continue
                text = minus_text(fixed(v, 2))
                cells.append(r"\textbf{" + text + "}" if v == best[(axis, key)] else text)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    tabular = (r"""\begin{tabular}{@{}lcccc@{}}
\toprule
& \multicolumn{2}{c}{Task} & \multicolumn{2}{c}{Physics} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Evaluator & $r\uparrow$ & $\tau_b\uparrow$ & $r\uparrow$ & $\tau_b\uparrow$ \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    emit("evaluator_alignment", table("table", CAPTIONS["evaluator_alignment"], "tab:evaluator-alignment", "", tabular),
         "results/human_alignment.json")


def build_unified_evaluator_alignment() -> None:
    align = alignment_index()
    labels = {"wr-arena": r"WR-Arena$^{\S}$", "videoscore": r"VideoScore$^\dagger$"}
    stats = {}
    for evaluator, label in EVALUATORS:
        row = align[(evaluator, "task" if evaluator == "wr-arena" else "final")]
        stats[evaluator] = row
    best = {"mae": min(r["mae"] for r in stats.values()), "bias": min(abs(r["bias"]) for r in stats.values()),
            "ccc": max(r["ccc"] for r in stats.values()), "pearson": max(r["pearson"] for r in stats.values()),
            "kendall_tau": max(r["kendall_tau"] for r in stats.values())}

    def cell(row: dict, key: str) -> str:
        if key == "bias":
            text, top = signed_text(row["bias"]), abs(row["bias"]) == best["bias"]
        else:
            text = minus_text(fixed(row[key], 1 if key == "mae" else 2))
            top = row[key] == best[key]
        return r"\textbf{" + text + "}" if top else text

    lines = []
    for evaluator, label in EVALUATORS:
        cells = " & ".join(cell(stats[evaluator], k) for k in ("mae", "bias", "ccc", "pearson", "kendall_tau"))
        lines.append(f"{labels.get(evaluator, label)} & {cells} \\\\")
    human = align[("human_leave_one_out", "final")]
    human_cells = " & ".join([fixed(human["mae"], 1), signed_text(human["bias"])]
                             + [fixed(human[k], 2) for k in ("ccc", "pearson", "kendall_tau")])
    tabular = (r"""\begin{tabular}{@{}lrrrrr@{}}
\toprule
Evaluator & MAE $\downarrow$ & Bias $\to0$ & CCC $\uparrow$ & $r\uparrow$ & $\tau\uparrow$ \\
\midrule
""" + "\n".join(lines) + "\n\\midrule\n" + f"Human$^\\ddagger$ & {human_cells} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    emit("unified_evaluator_alignment", table("table", CAPTIONS["unified_evaluator_alignment"],
         "tab:unified-evaluator-alignment", "\\setlength{\\tabcolsep}{4pt}\n", tabular), "results/human_alignment.json")


# ----------------------------------------------------------------------------- appendix D tables
def build_coverage() -> None:
    cov = load_json("validation_checks.json")["coverage"]
    groups = [(k, n) for k, n in GENERATORS] + [("human_reference", "Human (+)"), ("human_wrong", r"Human ($-$)")]
    lines = [f"{name} & " + " & ".join(fixed(100 * cov[key][field], 1)
             for field in ("task", "physics", "unattempted_zero", "final")) + " \\\\" for key, name in groups]
    tabular = (r"""\begin{tabular}{@{}lcccc@{}}
\toprule
Group & Task & Physics & Unattempted ($S=0$) & Final \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    emit("coverage", table("table", CAPTIONS["coverage"], "tab:coverage", "", tabular), "results/validation_checks.json")


def build_panel_vs_benchmark() -> None:
    rank = load_json("ranking_agreement.json")
    bench = {r["group"]: float(r["overall"]) for r in load_csv("leaderboard.csv") if r["evaluator"] == "ego2act"}
    columns = [rank["human"], rank["judge"], {k: bench[k] for k, _ in GENERATORS}]
    ranks = [{k: 1 + sum(v > col[k] for v in col.values()) for k in col} for col in columns]
    lines = [f"{name} & " + " & ".join(f"{fixed(col[key], 1)} ({rk[key]})" for col, rk in zip(columns, ranks)) + " \\\\"
             for key, name in GENERATORS]
    tabular = (r"""\begin{tabular}{@{}lccc@{}}
\toprule
Generator & Human, panel & \judge{}, panel & \judge{}, benchmark \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    emit("panel_vs_benchmark", table("table", CAPTIONS["panel_vs_benchmark"], "tab:panel-vs-benchmark", "", tabular),
         "results/ranking_agreement.json, results/leaderboard.csv")


# ----------------------------------------------------------------------------- physics controls
def build_physics_controls() -> None:
    data = load_json("physics_controls.json")["corruptions"]
    order = ("teleport", "swap", "ghost")

    def interval(mean: float, low: float, high: float) -> str:
        return f"{plain_math(mean)} [{plain_math(low)}, {plain_math(high)}]"
    detection = " & ".join(fixed(100 * data[c]["detection_physics_below_clean"], 1) + r"\%" for c in order)
    physics = " & ".join(interval(-d["mean"], -d["high"], -d["low"]) for d in (data[c]["physics_drop"] for c in order))
    task = " & ".join(interval(d["mean"], d["low"], d["high"]) for d in (data[c]["task_change"] for c in order))
    gate = " & ".join(data[c]["most_common_failing_gate"] for c in order)
    tabular = (r"""\begin{tabular}{@{}lccc@{}}
\toprule
& Teleport & Swap & Ghost \\
\midrule
""" + f"Detection & {detection} \\\\\nPhysics change & {physics} \\\\\nTask change & {task} \\\\\n"
            f"Main failing gate & {gate} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    emit("physics_controls", table("table", CAPTIONS["physics_controls"], "tab:physics-controls",
                                   "\\setlength{\\tabcolsep}{4pt}\n", tabular), "results/physics_controls.json")


BUILDERS = [build_main, build_combined_metrics, build_backbone_ablation, build_ci_task_physics,
            build_feature_tables, build_judge_gate_ablation, build_prompt_expansion, build_repeatability,
            build_evaluator_alignment, build_unified_evaluator_alignment, build_coverage,
            build_panel_vs_benchmark, build_physics_controls]


def main() -> None:
    for build in BUILDERS:
        build()


if __name__ == "__main__":
    main()
