# Baselines

Six evaluators, each run on its pinned official code with its own output and
scoring rule:

```bash
ego2act baseline <method> [--output-dir DIR] [selection options]
```

The first run clones the upstream repositories at the audited commit into
`baselines/<method>/external/` (ignored by Git); later runs verify the checkout
and refuse to use a modified one. `--setup-only` prepares without running.
Results go to `.results/baselines/<method>/` by default.

| Method | Upstream (commit) | Model | Native output |
|---|---|---|---|
| `wr_arena` | [WR-Arena](https://github.com/MBZUAI-IFM/WR-Arena) `7b3e4af` | `google/gemini-3.7-flash` | Action Simulation Fidelity, 0–3 |
| `pqsg` | [PQSG](https://github.com/atinpothiraj/pqsg) `03afdb0` | `google/gemini-3.7-flash` | dependency-weighted score, 0–1 |
| `rbench` | [ReVidgen](https://github.com/DAGroup-PKU/ReVidgen) `b03df27` | `google/gemini-3.7-flash` | five aspects 1–5, and total |
| `worldmodelbench` | [WorldModelBench](https://github.com/WorldModelBench-Team/WorldModelBench) `00b7aa1`, [VILA](https://github.com/NVlabs/VILA) `361c931` | `vila-ewm-qwen2-1.5b` (GPU, ~4 GB) | instruction 0–3 + physics 0–5 + common sense 0–2 = 0–10 |
| `videoscore` | [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore) `194736f`, [Mantis](https://github.com/TIGER-AI-Lab/Mantis) `62b7438` | `VideoScore-v1.1` (GPU, ~16.5 GB) | five aspects, 1–4 |
| `simple_vqa` | ours (`simple_vqa/method.py`) | `google/gemini-3.7-flash` | Task 0–3, Physics 0–4 |

The OpenRouter methods need `OPENROUTER_API_KEY` in `.env`. WorldModelBench and
VideoScore need an NVIDIA GPU and [uv](https://docs.astral.sh/uv/): the first
run builds a private environment from `<method>/runtime-lock.txt` and downloads
the released judge weights at a fixed revision.

## Selection options

Every method accepts `--data-root` (default `data/ego2act`), repeated `--case`,
`--case-file` (one ID per line, order kept), `--video`, `--category
{correct,wrong,ai}`, `--generator`, `--exclude-generator`, `--seed`, `--limit`,
and `--prepare-only` (writes the input manifest without model calls). `--resume`
reuses a result only when the video bytes, goal and protocol fingerprints match.

```bash
ego2act baseline pqsg --case laptop_open --category ai --generator wan_2_7 --seed 101 --prepare-only
```

## Changes from the upstream protocols

- **WR-Arena, PQSG:** only the model client is swapped for OpenRouter.
- **RBench:** keeps the six-frame grid and 1–5 post-processing; its
  robot-specific wording is replaced by the first-person prompt in
  `rbench/prompt.py` (hash recorded in each report).
- **WorldModelBench:** official judge and 0–10 total, applied to Ego2Act videos
  (not a reproduction of its 350-prompt leaderboard).
- **VideoScore:** official prompt, frame sampling, model and five heads.
- **Simple VQA:** our single-call control, the human Task and Physics rubrics
  condensed into one prompt.

Every evaluator sees the same 480p/24 fps inference copy of each video
(`video.py`, settings in `config.yaml`). The mapping of native outputs to Task
and Physics (0–100) is in Appendix C of the paper. To collect all native scores
into one table:

```bash
python -m baselines.compile_native_results --results-root .results/baselines
```
