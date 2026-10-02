"""Read Ego2Act cases: start image, goal (<case>/metadata.json, or the "Goal:" line of
prompt.txt), and videos in correct/, wrong/ and AI/<model>/."""
from __future__ import annotations

import json
import re
from pathlib import Path


VIDEO_GROUPS = ("correct", "wrong", "ai")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "ego2act"

# Accepted spellings match the benchmark data layout; see START_NAMES and the
# AI filename patterns below.
START_IMAGE_NAMES = ("start.png", "start.jpg", "start.jpeg", "start.heic")
VIDEO_SUFFIXES = (".mp4", ".mov")
_AI_SEEDED = re.compile(r"^.*__seed_(?P<seed>[0-9]+)$")
_AI_MODEL_SEED = re.compile(r"^.+\.seed_(?P<seed>[0-9]+)$")
_AI_NUMERIC = re.compile(r"^(?P<seed>[0-9]+)$")


def prompt_goal(prompt: str) -> str | None:
    """The single non-empty `Goal:` line of a prompt.txt, or None."""
    goals = [line.removeprefix("Goal:").strip() for line in prompt.splitlines()
             if line.startswith("Goal:")]
    return goals[0] if len(goals) == 1 and goals[0] else None


def case_metadata(case_dir: Path) -> dict | None:
    """metadata.json if present, else {"goal": ...} from prompt.txt, else None."""
    metadata_path, prompt_path = case_dir / "metadata.json", case_dir / "prompt.txt"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if prompt_path.is_file():
        goal = prompt_goal(prompt_path.read_text(encoding="utf-8"))
        return {"goal": goal} if goal else None
    return None


def _group_dir(case_dir: Path, group: str) -> Path | None:
    """Resolve a video-group folder, tolerating `AI` as well as `ai`."""
    direct = case_dir / group
    if direct.is_dir():
        return direct
    for path in sorted(case_dir.iterdir()):
        if path.is_dir() and path.name.lower() == group:
            return path
    return None


def _video_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def _start_image(case_dir: Path) -> Path | None:
    """Pick the start frame, preferring PNG and keeping the on-disk spelling."""
    available = {
        f"start{path.suffix.lower()}": path
        for path in sorted(case_dir.glob("start.*"))
        if path.is_file()
    }
    for name in START_IMAGE_NAMES:
        match = available.get(name)
        if match is not None:
            return match
    return None


def _ai_seed(stem: str) -> int | None:
    for pattern in (_AI_SEEDED, _AI_MODEL_SEED, _AI_NUMERIC):
        match = pattern.fullmatch(stem)
        if not match:
            continue
        seed = match.group("seed")
        if str(int(seed)) != seed:
            return None
        return int(seed)
    return None


def load_dataset_case(
    case_id: str,
    data_root: str | Path = DEFAULT_DATASET_ROOT,
) -> dict:
    case_dir = Path(data_root).expanduser().resolve() / case_id
    start_path = _start_image(case_dir)
    metadata = case_metadata(case_dir)
    if metadata is None or start_path is None:
        raise FileNotFoundError(
            f"{case_id} must contain metadata.json (or a prompt.txt with one "
            f"'Goal:' line) and one of {', '.join(START_IMAGE_NAMES)}"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"{case_id}.metadata.json must contain a JSON object")
    goal = metadata.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError(f"{case_id}.goal must be a non-empty string")
    objects = metadata.get("objects", [])
    if not isinstance(objects, list) or any(not isinstance(item, str) for item in objects):
        raise ValueError(f"{case_id}.objects must be a list of strings when present")

    unexpected = [
        path.name for path in case_dir.iterdir()
        if path.is_dir() and path.name.lower() not in VIDEO_GROUPS
    ]
    if unexpected:
        raise ValueError(f"{case_id} has non-canonical video folders: {unexpected}")

    annotated_videos = []
    metadata_keys = {
        str(key).casefold(): key
        for key, value in metadata.items()
        if isinstance(value, dict)
    }
    for category in ("correct", "wrong"):
        category_dir = _group_dir(case_dir, category)
        if category_dir is None:
            continue
        for video_path in _video_files(category_dir):
            key = video_path.stem
            if not re.fullmatch(rf"{category}_\d+", key):
                raise ValueError(f"Invalid {category} video name: {video_path.name}")
            relative_key = f"{category}/{video_path.name}"
            try:
                stored_key = metadata_keys[relative_key.casefold()]
                item = metadata[stored_key]
            except KeyError as error:
                raise ValueError(
                    f"Missing metadata for {case_id}/{relative_key}"
                ) from error
            annotated_videos.append({
                "key": key,
                "description": str(item.get("desc", "")),
                "gt": None,
                "video_path": video_path.resolve(),
                "category": category,
                "generation": None,
            })

    videos = annotated_videos + _generated_videos(case_dir)
    # Some case folders contain the same control recording twice with only
    # extension-case differences (for example `.MOV` and `.mov`). Keep the
    # first deterministic entry; they represent one logical sample.
    unique_videos = []
    seen_keys = set()
    for video in videos:
        if video["key"] in seen_keys:
            continue
        seen_keys.add(video["key"])
        unique_videos.append(video)
    videos = unique_videos
    keys = [video["key"] for video in videos]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"{case_id} has duplicate video keys: {duplicates}")
    if not videos:
        raise ValueError(f"{case_id} contains no evaluable MP4 videos")

    return {
        "id": case_id,
        "goal": goal,
        "objects": list(objects),
        "action_types": list(metadata.get("action_types", [])),
        "domain": metadata.get("category") or metadata.get("object_family"),
        "case_dir": case_dir,
        "metadata_path": case_dir / "metadata.json",
        "start_path": start_path.resolve(),
        "videos": videos,
    }


def discover_dataset_cases(
    data_root: str | Path = DEFAULT_DATASET_ROOT,
) -> list[dict]:
    root = Path(data_root).expanduser().resolve()
    cases = []
    for path in sorted(root.glob("*/metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not str(metadata.get("goal", "")).strip():
            continue
        try:
            cases.append(load_dataset_case(path.parent.name, root))
        except ValueError as error:
            # Some metadata-only expansion folders remain in the repository.
            # They are not evaluable datasets and should not block a filtered
            # run; preserve all other validation errors.
            if "contains no evaluable MP4 videos" not in str(error):
                raise
    return cases


def evaluable_videos(dataset_case: dict) -> list[dict]:
    """Return every locally available human or generated candidate video."""
    return list(dataset_case["videos"])


def _generated_videos(case_dir: Path) -> list[dict]:
    """Discover standardized ``ai/<model>/<seed>.mp4`` artifacts."""
    ai_dir = _group_dir(case_dir, "ai")
    if ai_dir is None:
        return []
    videos = []
    for model_dir in sorted(ai_dir.glob("*")):
        if not model_dir.is_dir():
            continue
        model_alias = model_dir.name
        if not re.fullmatch(r"[a-z0-9_]+", model_alias):
            raise ValueError(f"Invalid generated-video model alias: {model_dir}")
        manifest_path = model_dir / "manifest.json"

        by_seed: dict[int, Path] = {}
        for video_path in _video_files(model_dir):
            if video_path.stem.endswith(".native"):
                continue
            seed = _ai_seed(video_path.stem)
            if seed is None:
                raise ValueError(f"Generated video must carry a numeric seed: {video_path}")
            # A bare `<seed>.mp4` predates the current generation output template.
            # Where both spellings exist, the template-named file is the one the
            # sibling manifest registers, so it wins.
            incumbent = by_seed.get(seed)
            if incumbent is None or (
                incumbent.stem.isdigit() and not video_path.stem.isdigit()
            ):
                by_seed[seed] = video_path

        for sample_id in sorted(by_seed):
            video_path = by_seed[sample_id]
            key = f"{model_alias}__{sample_id}"
            videos.append({
                "key": key,
                "description": f"Generated by {model_alias}, sample {sample_id}.",
                "gt": None,
                "video_path": video_path.resolve(),
                "category": "ai",
                "generation": {
                    "model_alias": model_alias,
                    "sample_id": sample_id,
                    "manifest_path": (
                        manifest_path.resolve() if manifest_path.is_file() else None
                    ),
                },
            })
    return videos
