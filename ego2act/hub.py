"""`ego2act data pull`: download the released benchmark from the Hugging Face Hub.

    ego2act data pull --subset cases                 start images, prompts, metadata (small)
    ego2act data pull --subset human                 human correct and wrong recordings, 480p
    ego2act data pull --subset generated --model wan_2_7
    ego2act data pull --subset traces                judge traces for analysis/reproduce.py (about 40 MB)
    ego2act data pull --subset all                   cases, human, generated and prompt_expansion

Cases and human recordings come from ego2act/ego2act-bench, generated videos and traces from
ego2act/ego2act-vidgen. Files land in data/ego2act/<case>/... exactly as the judge, baselines and
generators expect. Set HF_TOKEN if a repository is private.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from ego2act.data import DEFAULT_DATASET_ROOT

BENCH_REPO = "ego2act/ego2act-bench"
VIDGEN_REPO = "ego2act/ego2act-vidgen"
CSV_ROOT = Path(__file__).resolve().parents[1] / "analysis" / "csv"
TRACE_RENAMES = {"traces/ego2act_judge/traces.jsonl": "traces.jsonl",
                 "traces/ego2act_judge/plans.jsonl": "judge_plans.jsonl"}
SUBSETS = ["cases", "human", "generated", "traces", "prompt_expansion", "all"]


def _download(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns))


def _copy(src: Path, dst: Path) -> int:
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return 1


def _unzip(archive: Path, root: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = root / info.filename
            if info.is_dir() or (target.is_file() and target.stat().st_size == info.file_size):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def pull(subset: str, root: Path, models: list[str] | None = None, *, bench_repo: str = BENCH_REPO,
         vidgen_repo: str = VIDGEN_REPO, csv_root: Path = CSV_ROOT) -> None:
    if subset not in SUBSETS:
        raise SystemExit(f"unknown subset {subset!r}; choose from {', '.join(SUBSETS)}")
    wanted = {"cases", "human", "generated", "prompt_expansion"} if subset == "all" else {subset}
    root.mkdir(parents=True, exist_ok=True)

    if wanted & {"cases", "human", "prompt_expansion"}:
        patterns = (["cases/*/*"] if "cases" in wanted else []) + (["human/*/*/*.mp4"] if "human" in wanted else []) \
            + (["prompt_expansion.zip"] if "prompt_expansion" in wanted else [])
        local = _download(bench_repo, patterns)
        n = sum(_copy(f, root / f.parent.name / f.name) for f in sorted(local.glob("cases/*/*")))
        n += sum(_copy(f, root / f.parent.name / f.parent.parent.name / f.name)   # human/<label>/<case>/<file>
                 for f in sorted(local.glob("human/*/*/*.mp4")))
        for archive in sorted(local.glob("prompt_expansion.zip")):
            n += _unzip(archive, root)
        print(f"{bench_repo}: {n} new files in {root}", flush=True)

    if "generated" in wanted:
        folders = models or ["*"]
        local = _download(vidgen_repo, [f"videos/{m}/*/*.mp4" for m in folders])
        n = sum(_copy(f, root / f.parent.name / "AI" / f.parent.parent.name / f.name)   # videos/<model>/<case>/<file>
                for m in folders for f in sorted(local.glob(f"videos/{m}/*/*.mp4")))
        print(f"{vidgen_repo}: {n} new generated videos in {root}", flush=True)

    if "traces" in wanted:
        local = _download(vidgen_repo, ["traces/*"])
        for src in sorted(p for p in (local / "traces").rglob("*") if p.is_file()):
            rel = src.relative_to(local).as_posix()
            target = csv_root / TRACE_RENAMES.get(rel, rel.removeprefix("traces/"))
            _copy(src, target)
            print(f"  {target.relative_to(csv_root.parent.parent)}", flush=True)
        print(f"traces ready in {csv_root}")
    print(f"data ready in {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ego2act data", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pull", help="download the benchmark from the Hub")
    p.add_argument("--subset", choices=SUBSETS, default="cases")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--model", action="append", dest="models", help="generated videos of this model only; repeatable")
    p.add_argument("--bench-repo", default=BENCH_REPO)
    p.add_argument("--vidgen-repo", default=VIDGEN_REPO)
    p.add_argument("--csv-root", type=Path, default=CSV_ROOT, help="where --subset traces writes (default analysis/csv)")
    args = parser.parse_args(argv)
    pull(args.subset, args.data_root.expanduser().resolve(), args.models, bench_repo=args.bench_repo,
         vidgen_repo=args.vidgen_repo, csv_root=args.csv_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
