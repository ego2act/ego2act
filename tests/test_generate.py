from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import pytest

import ego2act.vidgen.pipeline as generation_module
from ego2act.generate import build_parser
from ego2act.vidgen.pipeline import (
    build_generation_plan,
    discover_generation_cases,
    discover_generation_inventory,
    execute_generation,
    live_price_per_second,
    lowest_supported_resolution,
    sample_ids_for_count,
    select_cases,
    select_models,
    validate_model,
)
from ego2act.vidgen.config import load_generation_config
from ego2act.vidgen.runners import StubGenerationRunner


def catalog_item(**overrides):
    item = {
        "id": "provider/model",
        "name": "Provider Model",
        "seed": True,
        "generate_audio": False,
        "supported_frame_images": ["first_frame"],
        "supported_durations": [10],
        "supported_resolutions": ["720p"],
        "supported_aspect_ratios": ["16:9"],
        "pricing_skus": {"image_to_video_duration_seconds_720p": "0.03"},
    }
    item.update(overrides)
    return item


def protocol(**overrides):
    value = {
        "duration_seconds": 10,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "generate_audio": False,
        "sample_ids": [101],
    }
    value.update(overrides)
    return value


def test_generation_cli_is_safe_by_default_and_exposes_runtime_controls():
    defaults = build_parser().parse_args(["demo_case", "--model", "demo"])
    assert defaults.run is False
    assert defaults.keep_native is False
    assert defaults.progress is True

    live = build_parser().parse_args(
        [
            "demo_case",
            "--model",
            "demo",
            "--run",
            "--keep-native",
            "--no-progress",
        ]
    )
    assert live.run is True
    assert live.keep_native is True
    assert live.progress is False

    explicit_seed = build_parser().parse_args(
        ["demo_case", "--model", "demo", "--sample-id", "202"]
    )
    assert explicit_seed.sample_ids == [202]
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "demo_case",
                "--model",
                "demo",
                "--num-seeds",
                "2",
                "--sample-id",
                "202",
            ]
        )


def test_generation_discovery_uses_prompt_verbatim(tmp_path: Path):
    case = tmp_path / "demo_case"
    case.mkdir()
    prompt = "Action simulator.\n\nGoal: Move the cup.\n\nKeep every object visible.\n"
    (case / "prompt.txt").write_text(prompt, encoding="utf-8")
    (case / "start.jpg").write_bytes(b"image")
    (case / "metadata.json").write_text('{"goal": "ignored"}\n', encoding="utf-8")

    discovered = discover_generation_cases(tmp_path)["demo_case"]

    assert discovered["prompt"] == prompt
    assert discovered["goal"] == "Move the cup."


def test_video_generation_protocol_has_model_aliases():
    config = load_generation_config()
    assert set(config) == {"generation", "_meta"}
    generation = config["generation"]
    assert generation["protocol"]["resolution"] == "lowest"
    assert generation["protocol"]["sample_ids"] == [101, 202, 303]
    assert generation["models"]["wan_2_7"] == {
        "runner": "openrouter",
        "model_id": "alibaba/wan-2.7",
        "protocol": {"duration_seconds": 10},
    }
    assert set(generation["models"]) == {
        "grok_imagine_video_1_5", "seedance_2_0", "kling_v3_pro",
        "wan_2_7", "cosmos_3", "minimax_h3",
    }
    assert generation["models"]["minimax_h3"]["protocol"] == {
        "duration_seconds": 15,
        "resolution": "480p",
        "aspect_ratio": "4:3",
    }
    assert generation["video"] == {
        "codec": "libx264",
        "pixel_format": "yuv420p",
        "max_long_side": 854,
        "max_short_side": 480,
        "max_fps": 24.0,
        "max_bytes": 15728640,
        "crf_attempts": [23, 27, 30],
    }


def test_model_validation_records_unsupported_seed_capability():
    result = validate_model(
        "kling_v3_pro",
        "kwaivgi/kling-v3.0-pro",
        catalog_item(id="kwaivgi/kling-v3.0-pro", seed=False),
        protocol(),
    )
    assert result["seed_support"] is False


def test_model_validation_accepts_provider_with_undeclared_seed_support():
    result = validate_model(
        "grok",
        "x-ai/grok-imagine-video-1.5",
        catalog_item(id="x-ai/grok-imagine-video-1.5", seed=None),
        protocol(),
    )
    assert result["checks"]["seed_declared"] is False
    assert result["seed_support"] is False


def test_model_validation_rejects_incompatible_protocol():
    with pytest.raises(ValueError, match="duration"):
        validate_model(
            "demo",
            "provider/model",
            catalog_item(),
            protocol(duration_seconds=8),
        )


def test_model_selection_uses_only_configured_aliases():
    configured = {
        "wan": {"runner": "openrouter", "model_id": "provider/wan"},
        "veo": {"runner": "openrouter", "model_id": "provider/veo"},
    }
    assert select_models(configured, ["wan"]) == {"wan": configured["wan"]}
    assert select_models(configured, ["wan,veo"]) == configured
    assert select_models(configured, ["all"]) == configured
    with pytest.raises(ValueError, match="Unknown configured"):
        select_models(configured, ["raw/provider-id"])


def test_generation_case_discovery_needs_only_prompt_and_start_image(tmp_path: Path):
    case = tmp_path / "demo_case"
    case.mkdir()
    (case / "prompt.txt").write_text("Goal: Move the cup.\n", encoding="utf-8")
    (case / "start.png").write_bytes(b"image")
    available = discover_generation_cases(tmp_path)
    selected = select_cases(available, ["demo_case"])
    assert selected[0]["goal"] == "Move the cup."
    assert selected[0]["start_path"] == (case / "start.png").resolve()


def test_incomplete_prompt_is_skipped_for_all_and_explained_when_selected(
    tmp_path: Path,
):
    case = tmp_path / "blank_goal"
    case.mkdir()
    (case / "prompt.txt").write_text("Goal: \n", encoding="utf-8")
    (case / "start.jpg").write_bytes(b"image")

    available, skipped = discover_generation_inventory(tmp_path)

    assert select_cases(available, ["all"]) == []
    with pytest.raises(ValueError, match="Incomplete generation case.*Goal:"):
        select_cases(available, ["blank_goal"], skipped)


def test_seed_count_uses_101_multiples():
    assert sample_ids_for_count(4) == [101, 202, 303, 404]
    with pytest.raises(ValueError, match="positive integer"):
        sample_ids_for_count(0)


def test_lowest_resolution_orders_progressive_and_cinematic_labels():
    assert lowest_supported_resolution(
        {"supported_resolutions": ["4K", "1080p", "480p", "720p"]}
    ) == "480p"
    with pytest.raises(ValueError, match="does not advertise"):
        lowest_supported_resolution({"supported_resolutions": []})


def test_plan_uses_selected_cases_models_and_configured_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "demo_case"
    case.mkdir()
    prompt = "Use the image.\n\nGoal: Move the cup.\n"
    (case / "prompt.txt").write_text(prompt, encoding="utf-8")
    (case / "start.png").write_bytes(b"image")
    configured = {
        "generation": {
            "api_root": "https://openrouter.ai/api/v1",
            "protocol": protocol(
                duration_seconds=8,
                sample_ids=[101, 202],
                resolution="lowest",
            ),
            "output_template": "ai/{model_name}/{sample_id}.mp4",
            "video": {
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "max_long_side": 854,
                "max_short_side": 480,
                "max_fps": 24.0,
                "max_bytes": 15728640,
                "crf_attempts": [23, 27, 30],
            },
            "models": {
                "demo": {
                    "runner": "openrouter",
                    "model_id": "provider/model",
                }
            },
        },
        "_meta": {"data_root": tmp_path},
    }
    monkeypatch.setattr(
        generation_module,
        "model_catalog",
        lambda api_key, api_root: {
            "provider/model": catalog_item(
                supported_resolutions=["1080p", "480p", "720p"],
                pricing_skus={"image_to_video_duration_seconds_480p": "0.03"},
            )
        },
    )
    plan = build_generation_plan(
        ["demo_case"],
        ["demo"],
        "test-key",
        config=configured,
        num_seeds=3,
        duration_seconds=10,
    )
    assert plan["case_count"] == 1
    assert plan["model_count"] == 1
    assert plan["protocol"]["resolution"] == "lowest"
    assert plan["protocol"]["duration_seconds"] == 10
    assert plan["models"]["demo"]["protocol"]["resolution"] == "480p"
    assert plan["models"]["demo"]["protocol"]["duration_seconds"] == 10
    assert [job["sample_id"] for job in plan["jobs"]] == [101, 202, 303]
    assert {job["protocol"]["resolution"] for job in plan["jobs"]} == {"480p"}
    assert all(job["prompt"] == prompt for job in plan["jobs"])
    assert plan["estimated_cost_usd"] == pytest.approx(0.9)
    assert all("ai/demo" in job["output"] for job in plan["jobs"])


def test_unavailable_runner_is_an_explicit_not_implemented_stub():
    runner = StubGenerationRunner(
        "cosmos_predict_2_5",
        "nvidia/cosmos-predict2.5",
        "No configured hosted route.",
    )
    with pytest.raises(NotImplementedError, match="No configured hosted route"):
        runner.submit({})


def test_local_model_protocol_seed_resume_and_cli_duration_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "demo_case"
    case.mkdir()
    (case / "prompt.txt").write_text("Goal: Move the cup.\n", encoding="utf-8")
    (case / "start.png").write_bytes(b"image")
    configured = {
        "generation": {
            "api_root": "https://openrouter.ai/api/v1",
            "protocol": protocol(resolution="lowest"),
            "output_template": "ai/{model_name}/{sample_id}.mp4",
            "video": {
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "max_long_side": 854,
                "max_short_side": 480,
                "max_fps": 24.0,
                "max_bytes": 15728640,
                "crf_attempts": [23, 27, 30],
            },
            "models": {
                "minimax_h3": {
                    "runner": "minimax_h3",
                    "model_id": "MiniMaxAI/MiniMax-H3",
                    "reason": "Local H3.",
                    "protocol": {
                        "duration_seconds": 15,
                        "resolution": "480p",
                        "aspect_ratio": "4:3",
                    },
                }
            },
        },
        "_meta": {"data_root": tmp_path},
    }
    monkeypatch.setattr(
        generation_module,
        "model_catalog",
        lambda *args: (_ for _ in ()).throw(AssertionError("catalog called")),
    )

    default_plan = build_generation_plan(
        ["demo_case"], ["minimax_h3"], "unused", config=configured
    )
    override_plan = build_generation_plan(
        ["demo_case"],
        ["minimax_h3"],
        "unused",
        config=configured,
        duration_seconds=12,
    )
    additional_seeds_plan = build_generation_plan(
        ["demo_case"],
        ["minimax_h3"],
        "unused",
        config=configured,
        sample_ids=[202, 303],
    )

    assert default_plan["models"]["minimax_h3"]["protocol"] == {
        "duration_seconds": 15,
        "resolution": "480p",
        "aspect_ratio": "4:3",
        "generate_audio": False,
        "sample_ids": [101],
    }
    assert default_plan["jobs"][0]["protocol"]["duration_seconds"] == 15
    assert override_plan["jobs"][0]["protocol"]["duration_seconds"] == 12
    assert [job["sample_id"] for job in additional_seeds_plan["jobs"]] == [
        202,
        303,
    ]

    first_manifests = generation_module._prepare_manifests(default_plan)
    manifest_path, legacy_manifest = next(iter(first_manifests.items()))
    legacy_manifest["protocol"]["sample_ids"] = [101]
    digest_source = default_plan["_generation_digest_source"]
    legacy_manifest["generation_config_sha256"] = generation_module._generation_digest(
        {
            **digest_source,
            "protocol": {**digest_source["protocol"], "sample_ids": [101]},
        }
    )
    manifest_path.write_text(
        json.dumps(legacy_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resumed = generation_module._prepare_manifests(additional_seeds_plan)[
        manifest_path
    ]

    assert set(resumed["jobs"]) == {"101", "202", "303"}
    assert "sample_ids" not in resumed["protocol"]
    assert resumed["generation_config_sha256"] == additional_seeds_plan[
        "generation_config_sha256"
    ]


def test_stub_model_can_be_planned_without_openrouter_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case = tmp_path / "demo_case"
    case.mkdir()
    (case / "prompt.txt").write_text("Goal: Move the cup.\n", encoding="utf-8")
    (case / "start.png").write_bytes(b"image")
    configured = {
        "generation": {
            "api_root": "https://openrouter.ai/api/v1",
            "protocol": protocol(),
            "output_template": "ai/{model_name}/{sample_id}.mp4",
            "video": {
                "codec": "libx264",
                "pixel_format": "yuv420p",
                "max_long_side": 854,
                "max_short_side": 480,
                "max_fps": 24.0,
                "max_bytes": 15728640,
                "crf_attempts": [23, 27, 30],
            },
            "models": {
                "cosmos": {
                    "runner": "stub",
                    "model_id": "nvidia/cosmos-predict2.5",
                    "reason": "No configured hosted route.",
                }
            },
        },
        "_meta": {"data_root": tmp_path},
    }
    monkeypatch.setattr(
        generation_module,
        "model_catalog",
        lambda *args: (_ for _ in ()).throw(AssertionError("catalog called")),
    )
    plan = build_generation_plan(
        ["demo_case"], ["cosmos"], "unused", config=configured
    )
    assert plan["unavailable_model_count"] == 1
    assert plan["estimated_cost_usd"] == 0
    assert plan["jobs"][0]["runner"] == "stub"
    assert plan["jobs"][0]["estimated_cost_usd"] is None
    summary = execute_generation(plan, "unused", show_progress=False)
    assert summary["statuses"] == {"not_implemented": 1}
    manifest = tmp_path / "demo_case" / "ai" / "cosmos" / "manifest.json"
    assert manifest.is_file()
    assert "No configured hosted route" in manifest.read_text(encoding="utf-8")


def test_runway_cents_price_is_normalized_to_dollars():
    model = {
        "id": "runway/gen-4.5",
        "pricing_skus": {"cents_per_second_output": "12"},
    }
    assert live_price_per_second(model, "720p") == pytest.approx(0.12)


@pytest.mark.parametrize("keep_native", [False, True])
def test_execution_bounds_concurrency_and_applies_native_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keep_native: bool,
):
    case_dir = tmp_path / "demo_case"
    case_dir.mkdir()
    start = case_dir / "start.jpg"
    start.write_bytes(b"image")
    samples = [101, 202, 303, 404]
    jobs = [
        {
            "case": "demo_case",
            "goal": "Move the cup.",
            "prompt": "Goal: Move the cup.\n",
            "start_image": str(start),
            "start_image_sha256": generation_module.sha256(start),
            "model_alias": "demo",
            "model_id": "provider/demo",
            "runner": "openrouter",
            "runner_reason": None,
            "sample_id": sample,
            "provider_seed": sample,
            "protocol": protocol(),
            "output": str(
                case_dir / "AI" / "demo" / f"demo__seed_{sample}.mp4"
            ),
            "estimated_cost_usd": 1.0,
        }
        for sample in samples
    ]
    plan = {
        "generation_config_sha256": "test-config",
        "api_root": "https://example.invalid/api/v1",
        "protocol": protocol(sample_ids=samples),
        "video_policy": {},
        "models": {"demo": {"catalog_snapshot": {"id": "provider/demo"}}},
        "jobs": jobs,
    }
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    class FakeRunner:
        def submit(self, payload):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            return {"id": str(payload["seed"]), "status": "pending"}

        def poll(self, url):
            time.sleep(0.01)
            with state_lock:
                state["active"] -= 1
            return {"status": "completed"}

        def download(self, job_id, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"native-{job_id}".encode())

    def fake_standardize(source, output, policy):
        shutil.copyfile(source, output)
        return {"native_path": str(source), "output_path": str(output)}

    monkeypatch.setattr(
        generation_module, "_runner_for_manifest", lambda *args: FakeRunner()
    )
    monkeypatch.setattr(generation_module, "standardize_video", fake_standardize)
    monkeypatch.setattr(
        generation_module,
        "video_probe",
        lambda output: {"available": True, "path": str(output)},
    )

    summary = execute_generation(
        plan,
        "unused",
        workers=2,
        keep_native=keep_native,
        show_progress=False,
    )

    assert state["maximum"] == 2
    assert summary["statuses"] == {"downloaded": 4}
    for job in jobs:
        output = Path(job["output"])
        native = output.with_name(f"{output.stem}.native.mp4")
        assert output.is_file()
        assert native.is_file() is keep_native


def test_local_runners_read_positive_env_settings(monkeypatch):
    from ego2act.vidgen.cosmos3 import Cosmos3GenerationRunner
    from ego2act.vidgen.minimax_h3 import MiniMaxH3GenerationRunner

    cosmos = Cosmos3GenerationRunner("cosmos_3", "nvidia/Cosmos3-Nano")
    assert (cosmos.width, cosmos.fps, cosmos.inference_steps, cosmos.guidance_scale) == (None, 24, 35, 4.0)
    monkeypatch.setenv("COSMOS3_WIDTH", "640")
    with pytest.raises(ValueError, match="set together"):
        Cosmos3GenerationRunner("cosmos_3", "nvidia/Cosmos3-Nano")
    monkeypatch.setenv("COSMOS3_HEIGHT", "360")
    assert Cosmos3GenerationRunner("cosmos_3", "x").height == 360
    monkeypatch.setenv("COSMOS3_FPS", "abc")
    with pytest.raises(ValueError, match="COSMOS3_FPS must be a positive integer"):
        Cosmos3GenerationRunner("cosmos_3", "x")
    monkeypatch.setenv("MINIMAX_H3_FLOW_SHIFT", "-1")
    with pytest.raises(ValueError, match="positive number"):
        MiniMaxH3GenerationRunner("minimax_h3", "x")


@pytest.mark.parametrize("model", ["cosmos_3", "minimax_h3"])
def test_local_model_dry_run_matches_recorded_jobs(model, capsys):
    """Requests planned for the local backends are byte-identical to the recorded ones
    (the OpenRouter models depend on the live catalog, so they are not recorded)."""
    from pathlib import Path
    from ego2act.generate import main

    root = Path(__file__).resolve().parents[1]
    assert main(["toothbrush_case", "--model", model,
                 "--data-root", str(root / "tests" / "fixtures" / "cases"), "--show-jobs"]) == 0
    got = capsys.readouterr().out.replace(str(root), "<ROOT>")
    assert json.loads(got) == json.loads((Path(__file__).parent / "golden" / f"generate_{model}.json").read_text())
