"""`ego2act generate`: image + goal -> video with the six evaluated models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ego2act.vidgen.config import load_generation_config
from ego2act.vidgen.pipeline import (
    PROJECT_ROOT,
    build_generation_plan,
    execute_generation,
    plan_summary,
    select_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ego2act generate",
        description="Generate videos through the configured model-runner registry."
    )
    parser.add_argument(
        "cases", nargs="*", help="Case IDs (space/comma separated), or 'all'."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Configured model alias; repeat, comma-separate, or use 'all'.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print configured model routes, then exit.",
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate capabilities and print cost without writing or submitting (default).",
    )
    execution.add_argument(
        "--run",
        action="store_true",
        help="Submit paid jobs and download their outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Folder of case folders (default: generation.data_root in config.yaml).",
    )
    parser.add_argument("--show-jobs", action="store_true")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument(
        "--num-seeds",
        type=int,
        help=(
            "Number of trial IDs per case/model. "
            "Seeds are 101, 202, 303, and so on."
        ),
    )
    seeds.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        type=int,
        help="Explicit non-negative trial ID/provider seed; may be repeated.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        help="Override the configured video duration for this run.",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--poll-timeout", type=int, default=1800)
    parser.add_argument(
        "--keep-native",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain provider-native MP4s after canonical encoding.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show generation progress (enabled by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.workers < 1
        or args.poll_timeout < 1
        or (args.num_seeds is not None and args.num_seeds < 1)
        or (
            args.sample_ids is not None
            and any(seed < 0 for seed in args.sample_ids)
        )
        or (args.duration_seconds is not None and args.duration_seconds < 1)
    ):
        raise SystemExit(
            "--workers, --poll-timeout, --num-seeds, and --duration-seconds must "
            "be positive; --sample-id must be non-negative"
        )
    load_dotenv(Path(PROJECT_ROOT) / ".env")
    config = load_generation_config()
    if args.data_root is not None:
        config["_meta"]["data_root"] = args.data_root.expanduser().resolve()
    if args.list_models:
        print(json.dumps(config["generation"]["models"], indent=2))
        return 0
    if not args.cases or not args.models:
        raise SystemExit("Select at least one case and pass at least one --model")

    selected_models = select_models(config["generation"]["models"], args.models)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if (
        any(spec["runner"] == "openrouter" for spec in selected_models.values())
        and not api_key
    ):
        raise SystemExit("OPENROUTER_API_KEY is not configured")
    api_key = api_key or "unused-for-local-runner"

    try:
        plan = build_generation_plan(
            args.cases,
            args.models,
            api_key,
            config=config,
            num_seeds=args.num_seeds,
            sample_ids=args.sample_ids,
            duration_seconds=args.duration_seconds,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(plan_summary(plan, include_jobs=args.show_jobs), indent=2))
    if not args.run:
        return 0

    summary = execute_generation(
        plan,
        api_key,
        workers=args.workers,
        poll_timeout=args.poll_timeout,
        keep_native=args.keep_native,
        show_progress=args.progress,
    )
    print(json.dumps(summary, indent=2))
    return 0 if set(summary["statuses"]) == {"downloaded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
