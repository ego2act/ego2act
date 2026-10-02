"""Every paper table, and the data behind each figure, from analysis/csv/:  python analysis/reproduce.py [--verify]

No API calls. Each builder in this folder writes its outputs to analysis/results/; the last one
(build_paper_tables.py, with build_prompt_appendix.py) writes the compile-ready paper tables
to analysis/tables/*.tex. With --verify, every number in the regenerated analysis/results/*.json,
*.csv and *.tex and in analysis/tables/*.tex is compared with the files present before the run
(the committed ones in a clean checkout); any difference fails the command.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
RESULTS = ANALYSIS / "results"
TABLES = ANALYSIS / "tables"
TOLERANCE = 1e-6

# (builder, paper output)
BUILDERS = [
    ("leaderboard", "Table 2 (leaderboard)"),
    ("build_human_alignment", "Table 3, backbone ablation, alignment figure data"),
    ("build_confidence_intervals", "confidence intervals"),
    ("build_trial_instability", "Figure 4b data (3-seed stability)"),
    ("build_task_physics_scatter", "Figure 4a data (Task vs Physics)"),
    ("build_benchmark_statistics", "Figure 2 statistics"),
    ("build_case_features", "task-feature tables"),
    ("build_robustness", "robustness and ranking agreement"),
    ("build_validation_checks", "score coverage and validation checks"),
    ("build_gate_ablation", "judge-side gate ablation"),
    ("build_prompt_expansion", "prompt-expansion diagnostic"),
    ("build_subgoal_action_types", "subgoal action-type analysis"),
    ("build_physics_controls", "physics negative controls"),
    ("build_plan_stability", "plan stability"),
    ("build_repeatability", "judge repeatability"),
    ("build_inline_claims", "numbers quoted in the text (inline_claims.json)"),
    ("build_prompt_appendix", "appendix prompt listings (tables/judge_prompts.tex, baseline_prompts.tex)"),
    ("build_paper_tables", "every paper table as a compile-ready .tex (tables/*.tex)"),
]
# Builders that read the judge traces and plans, which live on the Hugging Face Hub (not in git).
TRACE_FILES = [ANALYSIS / "csv" / "traces.jsonl", ANALYSIS / "csv" / "judge_plans.jsonl"]
NEEDS_TRACES = {"build_gate_ablation", "build_subgoal_action_types", "build_plan_stability"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def snapshot() -> dict[str, str]:
    """Text of every checked file, keyed by its path below analysis/ (results/... and tables/*.tex)."""
    files = [p for p in sorted(RESULTS.rglob("*")) if p.suffix in {".json", ".csv", ".tex"}]
    files += sorted(TABLES.glob("*.tex"))
    return {str(p.relative_to(ANALYSIS)): p.read_text(encoding="utf-8") for p in files}


def _close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def _text_close(a: str, b: str) -> bool:
    """Same text once numbers are replaced, and the numbers agree."""
    if NUMBER.sub("#", a) != NUMBER.sub("#", b):
        return False
    return _close([float(x) for x in NUMBER.findall(a)], [float(x) for x in NUMBER.findall(b)])


def same(name: str, before: str, after: str) -> bool:
    if before == after:
        return True
    if name.endswith(".json"):
        return _close(json.loads(before), json.loads(after))
    if name.endswith(".csv"):
        rows = lambda text: list(csv.reader(io.StringIO(text)))
        a, b = rows(before), rows(after)
        return len(a) == len(b) and all(len(x) == len(y) and all(map(_text_close, x, y)) for x, y in zip(a, b))
    return _text_close(before, after)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python analysis/reproduce.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="fail if any regenerated number differs from the existing analysis/results/")
    args = parser.parse_args(argv)
    before = snapshot() if args.verify else {}
    env = dict(os.environ)
    failed, skipped = [], []
    have_traces = all(f.is_file() for f in TRACE_FILES)
    for name, output in BUILDERS:
        if name in NEEDS_TRACES and not have_traces:
            skipped.append(name)
            print(f"skip  {name:32s} needs the judge traces: run `ego2act data pull --subset traces`", flush=True)
            continue
        result = subprocess.run([sys.executable, f"{name}.py"], cwd=ANALYSIS, env=env,
                                capture_output=True, text=True)
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"{status:5s} {name:32s} {output}", flush=True)
        if result.returncode:
            failed.append(name)
            lines = result.stderr.strip().splitlines()
            print("      " + (lines[-1] if lines else "no error output"))
    done = len(BUILDERS) - len(failed) - len(skipped)
    print(f"\n{done} of {len(BUILDERS)} builders finished"
          + (f", {len(skipped)} skipped (traces missing)" if skipped else "") + ". Outputs in analysis/results/.")
    if args.verify:
        after = snapshot()
        changed = sorted(n for n in before.keys() | after.keys()
                         if n not in before or n not in after or not same(n, before[n], after[n]))
        for name in changed:
            print(f"differs: analysis/{name}")
        counts = {folder: sum(n.startswith(folder + "/") for n in before) for folder in ("results", "tables")}
        matched = {folder: sum(n.startswith(folder + "/") and n not in changed for n in before) for folder in counts}
        print("verify: " + (f"{len(changed)} files differ; " if changed else "")
              + ", ".join(f"{matched[f]} of {counts[f]} {f} files match" for f in counts))
        if skipped:
            print(f"warning: {', '.join(skipped)} were not re-run, so their results were not verified")
        failed += changed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
