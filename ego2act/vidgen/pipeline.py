from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any, Iterable

from tqdm import tqdm

from ego2act.data import prompt_goal
from ego2act.vidgen.config import load_generation_config
from ego2act.vidgen.utils import standardize_video
from ego2act.vidgen.runners import (
    GenerationRunner,
    build_runner,
    live_price_per_second,
    model_catalog,
    unavailable_model_report,
    validate_openrouter_model as validate_model,
    ego2act_model_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
START_NAMES = {"start.jpg", "start.jpeg", "start.png", "start.heic"}
LOCAL_RESOLUTION_FALLBACK = "720p"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def video_probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        process = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_name,pix_fmt,width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(process.stdout)
        probe.update({"available": True, "backend": "ffprobe"})
        return probe
    return {"available": False, "reason": "ffprobe_unavailable"}


def _split_values(values: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for value in values
            for item in value.split(",")
            if item.strip()
        )
    )



def discover_generation_inventory(
    data_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    cases: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    case_dirs = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and CASE_RE.fullmatch(path.name)
    )
    for case_dir in case_dirs:
        case_id = case_dir.name
        starts = sorted(
            path
            for path in case_dir.iterdir()
            if path.is_file() and path.name.lower() in START_NAMES
        )
        if len(starts) != 1:
            skipped[case_id] = f"expected one start image, found {len(starts)}"
            continue
        prompt_path = case_dir / "prompt.txt"
        if not prompt_path.is_file():
            skipped[case_id] = "prompt.txt is missing"
            continue
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped[case_id] = "prompt.txt is not valid UTF-8"
            continue
        goal = prompt_goal(prompt)
        if goal is None:
            skipped[case_id] = "prompt.txt must contain exactly one non-empty Goal: line"
            continue
        cases[case_id] = {
            "id": case_id,
            "goal": goal,
            "prompt": prompt,
            "case_dir": case_dir.resolve(),
            "start_path": starts[0].resolve(),
        }
    return cases, skipped


def discover_generation_cases(data_root: Path) -> dict[str, dict[str, Any]]:
    cases, _ = discover_generation_inventory(data_root)
    return cases


def select_cases(
    available: dict[str, dict[str, Any]],
    requested: Iterable[str],
    skipped: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    selected = _split_values(requested)
    if selected == ["all"]:
        return list(available.values())
    if "all" in selected:
        raise ValueError("Use either 'all' or explicit case IDs, not both")
    incomplete = {
        case_id: skipped[case_id]
        for case_id in selected
        if skipped and case_id in skipped
    }
    if incomplete:
        detail = "; ".join(
            f"{case_id}: {reason}" for case_id, reason in incomplete.items()
        )
        raise ValueError(f"Incomplete generation case(s): {detail}")
    unknown = [case_id for case_id in selected if case_id not in available]
    if unknown:
        raise ValueError(f"Unknown generation case(s): {unknown}")
    if not selected:
        raise ValueError("Select at least one generation case")
    return [available[case_id] for case_id in selected]


def select_models(
    configured: dict[str, dict[str, Any]], requested: Iterable[str]
) -> dict[str, dict[str, Any]]:
    selected = _split_values(requested)
    if selected == ["all"]:
        return dict(configured)
    if "all" in selected:
        raise ValueError("Use either 'all' or explicit model aliases, not both")
    unknown = [alias for alias in selected if alias not in configured]
    if unknown:
        raise ValueError(f"Unknown configured video model(s): {unknown}")
    if not selected:
        raise ValueError("Select at least one configured video model")
    return {alias: configured[alias] for alias in selected}


def _generation_digest(generation: dict[str, Any]) -> str:
    normalized = json.dumps(generation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sample_ids_for_count(count: int) -> list[int]:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("num_seeds must be a positive integer")
    return [101 * index for index in range(1, count + 1)]


def _validated_sample_ids(sample_ids: Iterable[int]) -> list[int]:
    selected = list(sample_ids)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(
            not isinstance(sample_id, int)
            or isinstance(sample_id, bool)
            or sample_id < 0
            for sample_id in selected
        )
    ):
        raise ValueError("sample_ids must be unique non-negative integers")
    return selected


def _protocol_without_sample_ids(protocol: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in protocol.items() if key != "sample_ids"}


def _resolution_rank(resolution: str) -> int:
    progressive = re.fullmatch(r"([0-9]+)p", resolution)
    if progressive:
        return int(progressive.group(1))
    cinematic = re.fullmatch(r"([0-9]+)[kK]", resolution)
    if cinematic:
        return int(cinematic.group(1)) * 1000
    raise ValueError(f"Unsupported resolution label in model catalog: {resolution!r}")


def lowest_supported_resolution(live: dict[str, Any]) -> str:
    resolutions = live.get("supported_resolutions") or []
    if not isinstance(resolutions, list) or not resolutions or any(
        not isinstance(value, str) for value in resolutions
    ):
        raise ValueError("OpenRouter model does not advertise usable resolutions")
    return min(resolutions, key=_resolution_rank)


def _model_protocol(
    base: dict[str, Any],
    spec: dict[str, Any],
    live: dict[str, Any] | None,
    duration_override: int | None,
) -> dict[str, Any]:
    protocol = dict(base)
    protocol.update(spec.get("protocol") or {})
    if duration_override is not None:
        protocol["duration_seconds"] = duration_override
    if protocol["resolution"] != "lowest":
        return protocol
    protocol["resolution"] = (
        lowest_supported_resolution(live or {})
        if spec["runner"] == "openrouter"
        else LOCAL_RESOLUTION_FALLBACK
    )
    return protocol


def build_generation_plan(
    case_ids: Iterable[str],
    model_aliases: Iterable[str],
    api_key: str,
    *,
    config: dict[str, Any] | None = None,
    num_seeds: int | None = None,
    sample_ids: Iterable[int] | None = None,
    duration_seconds: int | None = None,
) -> dict[str, Any]:
    effective = config or load_generation_config()
    generation = effective["generation"]
    protocol = dict(generation["protocol"])
    if num_seeds is not None and sample_ids is not None:
        raise ValueError("Use either num_seeds or explicit sample_ids, not both")
    if num_seeds is not None:
        protocol["sample_ids"] = sample_ids_for_count(num_seeds)
    elif sample_ids is not None:
        protocol["sample_ids"] = _validated_sample_ids(sample_ids)
    if duration_seconds is not None:
        if (
            not isinstance(duration_seconds, int)
            or isinstance(duration_seconds, bool)
            or duration_seconds < 1
        ):
            raise ValueError("duration_seconds must be a positive integer")
        protocol["duration_seconds"] = duration_seconds
    data_root = Path(effective["_meta"]["data_root"])
    available, skipped = discover_generation_inventory(data_root)
    requested_cases = list(case_ids)
    cases = select_cases(available, requested_cases, skipped)
    selected_case_ids = _split_values(requested_cases)
    reported_skips = skipped if selected_case_ids == ["all"] else {}
    models = select_models(generation["models"], model_aliases)
    api_root = generation["api_root"].rstrip("/")
    catalog = (
        model_catalog(api_key, api_root)
        if any(spec["runner"] == "openrouter" for spec in models.values())
        else {}
    )

    model_reports = {}
    for alias, spec in models.items():
        model_id = spec["model_id"]
        live = catalog.get(model_id)
        if spec["runner"] == "openrouter" and live is None:
            raise ValueError(
                f"Configured OpenRouter video model is unavailable: {model_id}"
            )
        model_protocol = _model_protocol(
            protocol, spec, live, duration_override=duration_seconds
        )
        if spec["runner"] == "openrouter":
            report = validate_model(alias, model_id, live, model_protocol)
        elif spec["runner"] in {"cosmos3", "minimax_h3"}:
            report = ego2act_model_report(alias, spec)
        else:
            report = unavailable_model_report(alias, spec)
        report["protocol"] = model_protocol
        model_reports[alias] = report

    jobs = []
    outputs = set()
    for case in cases:
        prompt = case["prompt"]
        start_hash = sha256(case["start_path"])
        for alias, model in model_reports.items():
            model_protocol = model["protocol"]
            for sample_id in protocol["sample_ids"]:
                relative_output = Path(
                    generation["output_template"].format(
                        model_name=alias,
                        sample_id=sample_id,
                    )
                )
                if (
                    relative_output.is_absolute()
                    or ".." in relative_output.parts
                    or not relative_output.parts
                    or relative_output.parts[0] not in {"AI", "ai"}
                ):
                    raise ValueError(
                        "generation.output_template must stay under the case's AI/ folder"
                    )
                output = case["case_dir"] / relative_output
                if output in outputs:
                    raise ValueError(f"Duplicate generation output: {output}")
                outputs.add(output)
                jobs.append(
                    {
                        "case": case["id"],
                        "goal": case["goal"],
                        "prompt": prompt,
                        "start_image": str(case["start_path"]),
                        "start_image_sha256": start_hash,
                        "model_alias": alias,
                        "model_id": model["model_id"],
                        "runner": model["runner"],
                        "runner_reason": model.get("reason"),
                        "sample_id": sample_id,
                        "provider_seed": (
                            sample_id if model["seed_support"] else None
                        ),
                        "protocol": dict(model_protocol),
                        "output": str(output),
                        "estimated_cost_usd": (
                            round(
                                model["usd_per_second"]
                                * model_protocol["duration_seconds"],
                                6,
                            )
                            if (
                                model["available"]
                                and model["usd_per_second"] is not None
                            )
                            else None
                        ),
                    }
                )

    generation_digest_source = {
        **generation,
        "protocol": _protocol_without_sample_ids(protocol),
    }
    return {
        "planned_at": utc_now(),
        "generation_config_sha256": _generation_digest(generation_digest_source),
        "_generation_digest_source": generation_digest_source,
        "api_root": api_root,
        "protocol": dict(protocol),
        "video_policy": dict(generation["video"]),
        "case_count": len(cases),
        "skipped_case_count": len(reported_skips),
        "skipped_cases": reported_skips,
        "model_count": len(models),
        "unavailable_model_count": sum(
            not report["available"] for report in model_reports.values()
        ),
        "unpriced_job_count": sum(
            job["runner"] == "openrouter" and job["estimated_cost_usd"] is None
            for job in jobs
        ),
        "job_count": len(jobs),
        "estimated_cost_usd": round(
            sum(
                job["estimated_cost_usd"]
                for job in jobs
                if job["estimated_cost_usd"] is not None
            ),
            6,
        ),
        "models": model_reports,
        "jobs": jobs,
    }


def plan_summary(plan: dict[str, Any], *, include_jobs: bool = False) -> dict[str, Any]:
    summary = {
        key: plan[key]
        for key in (
            "generation_config_sha256",
            "case_count",
            "skipped_case_count",
            "skipped_cases",
            "model_count",
            "unavailable_model_count",
            "unpriced_job_count",
            "job_count",
            "estimated_cost_usd",
            "models",
        )
    }
    if include_jobs:
        summary["jobs"] = plan["jobs"]
    return summary


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sanitized_request(payload: dict[str, Any], start_path: Path) -> dict[str, Any]:
    result = dict(payload)
    result["frame_images"] = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"local:{start_path.name}#sha256={sha256(start_path)}"
            },
            "frame_type": "first_frame",
        }
    ]
    return result


def _prepare_manifests(plan: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for job in plan["jobs"]:
        output = Path(job["output"])
        grouped.setdefault(output.parent / "manifest.json", []).append(job)

    manifests = {}
    for path, jobs in grouped.items():
        first = jobs[0]
        model = plan["models"][first["model_alias"]]
        relative_start = os.path.relpath(first["start_image"], path.parent)
        stable = {
            "generation_config_sha256": plan["generation_config_sha256"],
            "case": first["case"],
            "goal": first["goal"],
            "prompt": first["prompt"],
            "start_image": {
                "path": relative_start,
                "sha256": first["start_image_sha256"],
            },
            "model_alias": first["model_alias"],
            "model_id": first["model_id"],
            "runner": first["runner"],
            "runner_reason": first["runner_reason"],
            "protocol": _protocol_without_sample_ids(first["protocol"]),
            "video_policy": plan["video_policy"],
            "catalog_snapshot": model["catalog_snapshot"],
        }
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            legacy_sample_ids = (manifest.get("protocol") or {}).get("sample_ids")
            if legacy_sample_ids is not None:
                digest_source = plan["_generation_digest_source"]
                legacy_digest = _generation_digest(
                    {
                        **digest_source,
                        "protocol": {
                            **digest_source["protocol"],
                            "sample_ids": legacy_sample_ids,
                        },
                    }
                )
                if manifest.get("generation_config_sha256") != legacy_digest:
                    raise ValueError(
                        f"Existing manifest mismatch: {path} "
                        "(generation_config_sha256)"
                    )
                manifest["generation_config_sha256"] = stable[
                    "generation_config_sha256"
                ]
                manifest["protocol"] = _protocol_without_sample_ids(
                    manifest["protocol"]
                )
            for key, expected in stable.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"Existing manifest mismatch: {path} ({key})")
        else:
            manifest = {"created_at": utc_now(), **stable, "jobs": {}}

        for job in jobs:
            key = str(job["sample_id"])
            expected = {
                "sample_id": job["sample_id"],
                "provider_seed": job["provider_seed"],
                "output": Path(job["output"]).name,
                "estimated_cost_usd": job["estimated_cost_usd"],
            }
            record = manifest["jobs"].get(key)
            if record is None:
                output = Path(job["output"])
                if output.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite untracked generation output: {output}"
                    )
                manifest["jobs"][key] = {**expected, "status": "not_submitted"}
            else:
                for field, value in expected.items():
                    if record.get(field) != value:
                        raise ValueError(
                            f"Existing manifest mismatch: {path} ({key}.{field})"
                        )
                if record.get("status") == "downloaded":
                    output = path.parent / record["output"]
                    if not output.is_file() or (
                        record.get("video_sha256")
                        and sha256(output) != record["video_sha256"]
                    ):
                        raise ValueError(f"Downloaded artifact failed validation: {output}")
        _save_manifest(path, manifest)
        manifests[path] = manifest
    return manifests


def _runner_for_manifest(
    manifest: dict[str, Any], api_key: str, api_root: str
) -> GenerationRunner:
    spec = {
        "runner": manifest["runner"],
        "model_id": manifest["model_id"],
    }
    if manifest["runner"] != "openrouter":
        spec["reason"] = manifest["runner_reason"]
    return build_runner(manifest["model_alias"], spec, api_key, api_root)


def _commit_record(
    path: Path,
    key: str,
    manifest: dict[str, Any],
    record: dict[str, Any],
    lock: Lock,
) -> None:
    with lock:
        manifest["jobs"][key] = record
        _save_manifest(path, manifest)


def _submit_record(
    path: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    runner: GenerationRunner,
) -> None:
    start_path = (path.parent / manifest["start_image"]["path"]).resolve()
    payload: dict[str, Any] = {
        "model": manifest["model_id"],
        "prompt": manifest["prompt"],
        "duration": manifest["protocol"]["duration_seconds"],
        "resolution": manifest["protocol"]["resolution"],
        "aspect_ratio": manifest["protocol"]["aspect_ratio"],
        "generate_audio": manifest["protocol"]["generate_audio"],
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(start_path)},
                "frame_type": "first_frame",
            }
        ],
    }
    if record["provider_seed"] is not None:
        payload["seed"] = record["provider_seed"]
    record["request"] = _sanitized_request(payload, start_path)
    record["submitted_at"] = utc_now()
    try:
        response = runner.submit(payload)
    except NotImplementedError as error:
        record["status"] = "not_implemented"
        record["error"] = str(error)
    except Exception as error:
        record["status"] = "submission_error"
        record["error"] = f"{type(error).__name__}: {error}"
    else:
        record["submit_response"] = response
        record["job_id"] = response["id"]
        record["polling_url"] = response.get(
            "polling_url", f"/api/v1/videos/{response['id']}"
        )
        record["status"] = response.get("status", "pending")


def _poll_one(
    record: dict[str, Any], runner: GenerationRunner, timeout_seconds: int
) -> dict[str, Any]:
    polling_url = record["polling_url"]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = runner.poll(polling_url)
        record["poll_response"] = response
        record["status"] = response.get("status", record["status"])
        if record["status"] in TERMINAL_STATUSES:
            record["finished_at"] = utc_now()
            return record
        time.sleep(20)
    record["status"] = "poll_timeout"
    record["error"] = f"No terminal status after {timeout_seconds}s"
    return record


def _download_record(
    path: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    runner: GenerationRunner,
    keep_native: bool,
) -> None:
    output = path.parent / record["output"]
    native = output.with_name(f"{output.stem}.native.mp4")
    try:
        if not output.is_file():
            if not native.is_file():
                runner.download(record["job_id"], native)
            standardization = standardize_video(
                native, output, manifest["video_policy"]
            )
            if not keep_native:
                native.unlink()
            standardization["native_retained"] = keep_native
            record["media_standardization"] = standardization
        record["video_sha256"] = sha256(output)
        record["video_bytes"] = output.stat().st_size
        record["media_probe"] = video_probe(output)
    except Exception as error:
        record["status"] = "download_error"
        record["error"] = f"{type(error).__name__}: {error}"
    else:
        record.pop("error", None)
        record["status"] = "downloaded"
        record["downloaded_at"] = utc_now()


def _execute_record(
    path: Path,
    key: str,
    manifest: dict[str, Any],
    api_key: str,
    api_root: str,
    timeout_seconds: int,
    keep_native: bool,
    lock: Lock,
    halt_submissions: Event,
) -> str:
    record = dict(manifest["jobs"][key])
    runner = _runner_for_manifest(manifest, api_key, api_root)

    if record["status"] == "not_submitted":
        if halt_submissions.is_set():
            return "not_submitted"
        _submit_record(path, manifest, record, runner)
        if record["status"] == "submission_error":
            halt_submissions.set()
        _commit_record(path, key, manifest, record, lock)

    if record["status"] not in TERMINAL_STATUSES | {
        "submission_error",
        "not_implemented",
        "download_error",
    }:
        try:
            _poll_one(record, runner, timeout_seconds)
        except Exception as error:
            record["status"] = "poll_error"
            record["error"] = f"{type(error).__name__}: {error}"
        if record["status"] in {"poll_error", "poll_timeout"}:
            halt_submissions.set()
        _commit_record(path, key, manifest, record, lock)

    if record["status"] in {"completed", "download_error"}:
        _download_record(path, manifest, record, runner, keep_native)
        _commit_record(path, key, manifest, record, lock)
    return record["status"]


def _execute_records(
    manifests: dict[Path, dict[str, Any]],
    api_key: str,
    api_root: str,
    workers: int,
    timeout_seconds: int,
    keep_native: bool,
    show_progress: bool,
) -> None:
    skipped = {
        "downloaded",
        "failed",
        "cancelled",
        "expired",
        "submission_error",
        "not_implemented",
    }
    candidates = [
        (path, key)
        for path, manifest in manifests.items()
        for key, record in manifest["jobs"].items()
        if record["status"] not in skipped
    ]
    candidates.sort(
        key=lambda item: (
            manifests[item[0]]["jobs"][item[1]]["status"] == "not_submitted",
            str(item[0]),
            item[1],
        )
    )
    locks = {path: Lock() for path in manifests}
    halt_submissions = Event()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _execute_record,
                path,
                key,
                manifests[path],
                api_key,
                api_root,
                timeout_seconds,
                keep_native,
                locks[path],
                halt_submissions,
            ): (path, key)
            for path, key in candidates
        }
        progress = tqdm(
            total=len(futures),
            desc="Generating videos",
            unit="video",
            disable=not show_progress,
        )
        try:
            for future in as_completed(futures):
                path, key = futures[future]
                try:
                    status = future.result()
                except Exception as error:
                    halt_submissions.set()
                    record = dict(manifests[path]["jobs"][key])
                    record["status"] = "pipeline_error"
                    record["error"] = f"{type(error).__name__}: {error}"
                    _commit_record(
                        path, key, manifests[path], record, locks[path]
                    )
                    status = "pipeline_error"
                progress.set_postfix_str(status, refresh=False)
                progress.update()
        finally:
            progress.close()


def _summarize_manifests(
    manifests: dict[Path, dict[str, Any]]
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    costs = []
    for path, manifest in manifests.items():
        local_statuses: dict[str, int] = {}
        local_costs = []
        for record in manifest["jobs"].values():
            status = record["status"]
            statuses[status] = statuses.get(status, 0) + 1
            local_statuses[status] = local_statuses.get(status, 0) + 1
            cost = (record.get("poll_response") or {}).get("usage", {}).get("cost")
            if isinstance(cost, (int, float)):
                costs.append(float(cost))
                local_costs.append(float(cost))
        manifest["summary"] = {
            "statuses": local_statuses,
            "reported_cost_usd": round(sum(local_costs), 6),
            "updated_at": utc_now(),
        }
        _save_manifest(path, manifest)
    return {
        "statuses": statuses,
        "reported_cost_usd": round(sum(costs), 6),
        "reported_cost_jobs": len(costs),
        "manifest_count": len(manifests),
    }


def execute_generation(
    plan: dict[str, Any],
    api_key: str,
    *,
    workers: int = 5,
    poll_timeout: int = 1800,
    keep_native: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    manifests = _prepare_manifests(plan)
    _execute_records(
        manifests,
        api_key,
        plan["api_root"],
        workers,
        poll_timeout,
        keep_native,
        show_progress,
    )
    return _summarize_manifests(manifests)
