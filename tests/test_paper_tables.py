"""Compare analysis/tables/*.tex with the paper's LaTeX source (skipped when the source is absent).

Set EGO2ACT_PAPER_SRC to the unpacked paper directory (the one with main.tex). Run directly for a
per-table report:  python tests/test_paper_tables.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "analysis" / "tables"
SRC = Path(os.environ.get("EGO2ACT_PAPER_SRC", "paper_src"))

APP = "appendix"
# generated file -> (paper file, label of the table inside it, or None for the whole file)
PAPER = {
    "main": ("tables/main.tex", None),
    "combined_metrics": ("sections/05_experiments_and_results/02_main_results/index.tex", "tab:combined_metrics"),
    "backbone_ablation": ("sections/05_experiments_and_results/03_analysis_ablations/02_rubric_design_significance/index.tex",
                          "tab:backbone_ablation"),
    "ci_task_physics": (f"{APP}/05_expanded_quantitative_results/tables/tab_ci_task_physics.tex", None),
    "domain_models": (f"{APP}/05_expanded_quantitative_results/tables/tab_domain_models.tex", None),
    "feature_detail": (f"{APP}/05_expanded_quantitative_results/tables/tab_feature_detail.tex", None),
    "feature_qvalues": (f"{APP}/05_expanded_quantitative_results/tables/tab_feature_qvalues.tex", None),
    "judge_gate_ablation": (f"{APP}/07_rubric_ablations_and_repeatability/tables/tab_judge_gate_ablation.tex", None),
    "prompt_expansion": (f"{APP}/05_expanded_quantitative_results/tables/tab_prompt_expansion.tex", None),
    "repeatability": (f"{APP}/07_rubric_ablations_and_repeatability/tables/tab_repeatability.tex", None),
    "evaluator_alignment": (f"{APP}/07_rubric_ablations_and_repeatability/tables/tab_evaluator_alignment.tex", None),
    "unified_evaluator_alignment": (f"{APP}/07_rubric_ablations_and_repeatability/tables/tab_unified_evaluator_alignment.tex", None),
    "coverage": (f"{APP}/04_human_alignment_study_protocol/tab_coverage.tex", None),
    "panel_vs_benchmark": (f"{APP}/04_human_alignment_study_protocol/tab_panel_vs_benchmark.tex", None),
    "physics_controls": (f"{APP}/07_rubric_ablations_and_repeatability/content.tex", "tab:physics-controls"),
    "judge_prompts": (f"{APP}/02_video_scoring_rubric/judge_prompts.tex", None),
    "baseline_prompts": (f"{APP}/03_detailed_baseline_setup/baseline_prompts.tex", None),
}
TOKEN = re.compile(r"\\[A-Za-z]+|\d+(?:\.\d+)?|[A-Za-z]+|[^\sA-Za-z0-9]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _group(text: str, start: int) -> tuple[str, int]:
    """Content of the brace group opening at text[start] == '{' and the index after it."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def normalise(text: str) -> str:
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    return text.replace("$-$", "-")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(normalise(text))


def parts(text: str) -> dict[str, list]:
    """Captions, labels and tabular bodies of a file (structure-independent comparison)."""
    clean = normalise(text)
    caps = []
    for m in re.finditer(r"\\caption(?:of\{table\})?\{", clean):
        caps.append(TOKEN.findall(_group(clean, m.end() - 1)[0]))
    return {"caption": caps, "label": re.findall(r"\\label\{([^}]*)\}", clean),
            "tabular": [TOKEN.findall(b) for b in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", clean, re.S)]}


def paper_text(name: str) -> str:
    path, label = PAPER[name]
    text = (SRC / path).read_text(encoding="utf-8")
    if label is None:
        return text
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S):
        if f"\\label{{{label}}}" in block:
            return block
    # backbone table: box + caption live in different places
    box = re.search(r"\\sbox\{\\backboneablationbox\}\{%(.*?)\\end\{tabular\}\}", text, re.S).group(1)
    cap = re.search(r"\\captionof\{table\}\{.*?\}\n\\label\{" + re.escape(label) + r"\}", text, re.S).group(0)
    return cap + "\n" + box + "\\end{tabular}"


def compare(name: str) -> tuple[str, str]:
    generated = (TABLES / f"{name}.tex").read_text(encoding="utf-8")
    paper = paper_text(name)
    if tokens(generated) == tokens(paper):
        return "identical", ""
    if parts(generated) == parts(paper):
        return "identical content (caption, label, tabular); environment wrapper differs", ""
    g, p = [NUMBER.findall(normalise(t)) for t in (generated, paper)]
    if g == p:
        return "numerically identical, text differs", ""
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(g, p)) if a != b][:12]
    return "MISMATCH", f"generated has {len(g)} numbers, paper {len(p)}; first differences (idx, repo, paper): {diff}"


@pytest.mark.skipif(not (SRC / "main.tex").exists(), reason="paper source not available")
@pytest.mark.parametrize("name", sorted(PAPER))
def test_table_matches_paper(name):
    if not (TABLES / f"{name}.tex").exists():
        pytest.skip("run analysis/build_paper_tables.py first")
    status, detail = compare(name)
    assert status != "MISMATCH", detail


if __name__ == "__main__":
    for table_name in PAPER:
        result, note = compare(table_name)
        print(f"{table_name:30s} {result} {note}")
