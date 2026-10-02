# Analysis

Each builder reads `analysis/csv/` and writes to `analysis/results/`. Run them all with

```bash
python analysis/reproduce.py            # add --verify to check every number against analysis/results/
```

| Builder | Paper output |
|---|---|
| `leaderboard.py` | Table 2 leaderboard |
| `build_human_alignment.py` | Tables 3 and 4 (agreement, alignment, backbone ablation), alignment figure data |
| `build_confidence_intervals.py` | Task and Physics confidence intervals |
| `build_task_physics_scatter.py`, `build_trial_instability.py` | Figure 4 data (`results/plots_data/`) |
| `build_benchmark_statistics.py` | Figure 2 data (`results/benchmark_statistics.json`) |
| `build_case_features.py` | Task-feature tables |
| `build_robustness.py`, `build_validation_checks.py` | Robustness, ranking agreement, score coverage |
| `build_gate_ablation.py` | Judge-side gate ablation |
| `build_prompt_expansion.py` | Prompt-expansion diagnostic |
| `build_subgoal_action_types.py` | Subgoal action-type analysis |
| `build_physics_controls.py` | Physics negative controls |
| `build_plan_stability.py`, `build_repeatability.py` | Plan stability and judge repeatability |
| `build_inline_claims.py` | Numbers quoted in the text that no other builder emits (`results/inline_claims.json`) |
| `build_prompt_appendix.py` | Appendix prompt listings `tables/judge_prompts.tex`, `tables/baseline_prompts.tex` |
| `build_paper_tables.py` | Every paper table as a compile-ready file in `tables/` (Tables 1-3, appendix D-G; last builder) |

Paper tables land in `analysis/tables/*.tex` and are checked by `--verify` as well. With the paper
source unpacked (set `EGO2ACT_PAPER_SRC`), `python tests/test_paper_tables.py` compares each one with
the paper's own table file. `tab:human-rubric-ablation` is not generated (it needs the pilot gate annotations); its values, also
plotted in Figure 5, are in `csv/rubric_ablation_pilot.json`.

`scoring.py` holds the one scoring rule shared by all builders.
