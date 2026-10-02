# Data

```text
data/
└── ego2act/          the benchmark cases, pulled from the Hugging Face Hub (not tracked in git)
```

Everything the paper reproduction reads (scores, ratings, `metadata.csv`) is in
[`analysis/csv/`](../analysis/csv/README.md), so the tables and figures need no download.

## Benchmark cases: `data/ego2act/`

Each case is one folder:

```text
<case>/
├── start.jpg         initial scene, given to every video model and to Ego2ActJudge
├── prompt.txt        generation prompt (the goal)
├── metadata.json     goal, objects, domain, clutter, action types
├── correct/          three successful human recordings
├── wrong/            three unsuccessful human recordings
└── AI/<model>/<model>__seed_<seed>.mp4    generated videos, seeds 101, 202, 303
```

The folder is git-ignored because of its size. Download the released benchmark with

```bash
ego2act data pull --subset cases                       # start images, prompts, metadata
ego2act data pull --subset human                       # human correct and wrong recordings, 480p
ego2act data pull --subset generated --model wan_2_7   # one generator's videos
ego2act data pull --subset all                         # cases, human, generated and prompt_expansion
```

Cases and human recordings come from [`ego2act/ego2act-bench`](https://huggingface.co/datasets/ego2act/ego2act-bench)
and generated videos from [`ego2act/ego2act-vidgen`](https://huggingface.co/datasets/ego2act/ego2act-vidgen).
Set `HF_TOKEN` only if you use a private mirror. To use another location, pass `--data` (`ego2act judge`) or
`--data-root` (`ego2act baseline`, `ego2act generate`). Folders named `<case>_expanded<N>` hold the expanded
prompts of the prompt-expansion diagnostic (`--subset prompt_expansion`).
