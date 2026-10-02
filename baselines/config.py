"""The `baselines:` section of config.yaml, shared by every baseline adapter."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATHS = (
    Path(__file__).resolve().parents[1] / "config.yaml",
    Path(sys.prefix) / "share/ego2act/config.yaml",  # installed wheel
)


@lru_cache(maxsize=1)
def get_experiment_config() -> dict:
    path = next((p.resolve() for p in CONFIG_PATHS if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("config.yaml not found at the repository root")
    section = yaml.safe_load(path.read_text(encoding="utf-8"))["baselines"]
    missing = {"api_root", "model", "decoding", "video"} - set(section)
    if missing:
        raise ValueError(f"config.yaml baselines section is missing {sorted(missing)}")
    digest = hashlib.sha256(json.dumps({"baselines": section}, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"baselines": section, "_meta": {"config_path": path, "config_sha256": digest}}


def make_run_id(case_folder: str) -> str:
    return f"{case_folder}_{datetime.now().astimezone().strftime('%H-%M_%d%m%y')}"
