#!/usr/bin/env python
"""Benchmark composition statistics for the dataset-statistics figure.

Reads analysis/csv/case_metadata.json (the per-case metadata.json files) for the benchmark cases in
analysis/csv/metadata.csv (expanded variants are excluded). Structured fields are
step and object annotations of the human-correct recordings; durations are
the cached recording lengths in seconds. Domain counts come from the domain column of
analysis/csv/metadata.csv.

Outputs analysis/results/benchmark_statistics.json.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FAMILIES = json.loads((HERE.parent / "analysis/csv/action_taxonomy.json").read_text())
CASE_METADATA = json.loads((HERE.parent / "analysis/csv/case_metadata.json").read_text())


def quantile(values, p):
    values = sorted(values)
    i = (len(values) - 1) * p
    a = int(i)
    b = min(a + 1, len(values) - 1)
    return values[a] + (values[b] - values[a]) * (i - a)


def summary(values):
    return {"n": len(values), "min": min(values), "q1": quantile(values, .25), "median": median(values),
            "mean": mean(values), "q3": quantile(values, .75), "max": max(values)}


def main():
    with (ROOT / "analysis/csv/metadata.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    benchmark = [row["case_id"] for row in rows]
    domains = Counter(row["domain"] for row in rows)
    cases, durations = [], {"correct": [], "wrong": []}
    for case in sorted(benchmark):
        metadata = CASE_METADATA[case]
        for key, value in metadata.items():
            group = key.split("/")[0]
            if "/" in key and group in durations and isinstance(value, dict):
                seconds = value.get("duration")
                if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0:
                    durations[group].append(seconds)
        structured = metadata.get("human_correct_summary")
        if not structured:
            continue
        steps = [v["num_steps"] for k, v in metadata.items()
                 if k.startswith("correct/") and isinstance(v, dict) and isinstance(v.get("num_steps"), (int, float))]
        cases.append({"case": case, "objects": structured["object_count"],
                      "families": sorted(set(structured["action_categories"])),
                      "mean_steps": mean(steps) if steps else None})


    families = list(FAMILIES)
    result = {
        "benchmark_cases": len(benchmark),
        "structured_cases": len(cases),
        "domains": dict(domains.most_common()),
        "families": {k: v["name"] for k, v in FAMILIES.items()},
        "family_coverage": {f: sum(f in c["families"] for c in cases) for f in families},
        "cooccurrence": {a: {b: sum(a in c["families"] and b in c["families"] for c in cases) for b in families}
                         for a in families},
        "action_diversity": dict(sorted(Counter(len(c["families"]) for c in cases).items())),
        "objects": {"mean": mean(c["objects"] for c in cases),
                    "counts": dict(sorted(Counter(c["objects"] for c in cases).items()))},
        "observed_operations": {"case_means": [c["mean_steps"] for c in cases],
                                "mean": mean(c["mean_steps"] for c in cases)},
        "durations": {group: summary(values) for group, values in durations.items()},
        "case_rows": cases,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "benchmark_statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ("benchmark_cases", "structured_cases", "family_coverage", "action_diversity")})


if __name__ == "__main__":
    main()
