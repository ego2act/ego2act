# Scores behind the paper

Every table and figure in the paper is rebuilt from these files by `python analysis/reproduce.py`.
Nothing here needs the videos.

| File | Content |
|---|---|
| `metadata.csv` | one row per case: goal, action types, domain, video counts |
| `scores_ego2act.csv` | Ego2ActJudge Task, Physics and Final scores for every video |
| `traces.jsonl` | Per-video Task and Physics plans and evidence |
| `scores_human.csv` | Human scores on the 600-video panel, on 0–100 |
| `human_annot_original.csv` | Raw human ratings: up to three raters, Task 0–3, Physics 0–4 |
| `scores_baselines.csv` | WR-Arena, PQSG, RBench, WorldModelBench and VideoScore scores |
| `scores_simple_vqa.csv` | Simple VQA scores on the human-rated videos |
| `scores_terra.csv` | Alternative-backbone judge scores (backbone ablation) |
| `scores_gate_ablation.csv`, `gate_ablation_subgoals.csv` | Judge-side gate ablation |
| `scores_physics_controls.csv` | Physics negative controls |
| `prompt_expansion_scores.jsonl` | Prompt-expansion diagnostic |
| `case_features.csv` | Case-level features |
| `action_taxonomy.json` | Action families A1–A9 and their verbs |
| `case_metadata.json` | `metadata.json` of every benchmark case (durations, steps, objects) |
| `judge_plans.jsonl` | Ego2ActJudge Task and Physics plans of the production run and the repeatability pilot |
| `pilot_scores.json` | Second judge run on a representative subset (repeatability) |
| `rubric_ablation_pilot.json` | Rubric-ablation MAE from the pilot annotation study (Figure 5) |
| `baselines/<name>/` | Raw baseline outputs (`scores.csv`, `traces.jsonl`) |

Scores follow the paper: Task T = 100·mean/3, Physics P = 100·mean/4, and
Final S = √(T·P). A video with no attempted subgoal scores S = 0. Missing
values stay blank, and zero is a real score. Load with explicit missing-value
handling so the clutter label `None` survives:

```python
import pandas as pd
human = pd.read_csv("analysis/csv/scores_human.csv", keep_default_na=False, na_values=[""])
```

`native_0_100` in `scores_baselines.csv` maps each baseline's own final score
to 0–100: PQSG ×100, RBench (s−1)×25, WR-Arena s×100/3, WorldModelBench ×10,
and VideoScore the mean of its five aspects (s−1)×100/3.
