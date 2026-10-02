"""`ego2act baseline <method>`: set up the pinned upstream code, then run it.

On first use each baseline clones its upstream repositories at the audited
commit into baselines/<method>/external/ (ignored by Git). VideoScore and
WorldModelBench also build a private GPU environment from runtime-lock.txt and
download their released judge weights at a fixed revision. Later runs only
verify the checkout. Remaining arguments go to baselines/<method>/run.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

# method -> upstream repositories (folder, URL, commit) and optional GPU runtime
UPSTREAM = {
    "wr_arena": {"repos": [("WR-Arena", "https://github.com/MBZUAI-IFM/WR-Arena.git",
                            "7b3e4af4d9346bc58fe2c6c75626f15c3a464fca")]},
    "pqsg": {"repos": [("pqsg", "https://github.com/atinpothiraj/pqsg.git",
                        "03afdb0e6f9ed68d363a68eea5ca4fd90d53848e")]},
    "rbench": {"repos": [("ReVidgen", "https://github.com/DAGroup-PKU/ReVidgen.git",
                          "b03df27f0376faa148dcd8cd620a1989a32ca979")]},
    "worldmodelbench": {
        "repos": [("WorldModelBench", "https://github.com/WorldModelBench-Team/WorldModelBench.git",
                   "00b7aa17a05f9fd1ab5c8f66bcf476d04c9c33bf"),
                  ("VILA", "https://github.com/NVlabs/VILA.git",
                   "361c9317a2787ebf1814e115de6a4cf804ffaf51")],
        # VILA's own environment_setup.sh overrides triton after torch==2.3.0.
        "runtime": {"python": "3.10", "editable": "VILA", "after": ["triton==3.1.0"]},
        "model": ("Efficient-Large-Model/vila-ewm-qwen2-1.5b", "7a269c562f83e902f5ff1ef57015bd72f25f461a",
                  "vila-ewm-qwen2-1.5b", ".worldmodelbench_revision.json"),
    },
    "videoscore": {
        "repos": [("VideoScore", "https://github.com/TIGER-AI-Lab/VideoScore.git",
                   "194736f913018188105a6e0c2b0667eff5bd3be5"),
                  ("Mantis", "https://github.com/TIGER-AI-Lab/Mantis.git",
                   "62b7438199fb86aca2b2ab1340cc1d646d44be58")],
        "runtime": {"python": "3.12", "editable": "Mantis", "after": [],
                    "install_flags": ["--index-strategy", "unsafe-best-match"]},
        "model": ("TIGER-Lab/VideoScore-v1.1", "0731e98e15f8c06a3e3b4dc6c7a4b8d866f22a89",
                  "VideoScore-v1.1", ".videoscore_revision.json"),
    },
    "simple_vqa": {"repos": []},
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def checkout(target: Path, url: str, commit: str) -> None:
    """Clone `url` into `target` at `commit`, or verify an existing checkout."""
    fresh = not (target / ".git").exists()
    if fresh:
        if target.exists():
            raise RuntimeError(f"{target} exists but is not a Git checkout")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)], check=True)
    origin = _git(target, "remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    if origin != url.removesuffix(".git"):
        raise RuntimeError(f"{target} points to {origin}, expected {url}")
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "-q", "--verify", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if fresh or head != commit:
        if subprocess.run(["git", "-C", str(target), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode:
            subprocess.run(["git", "-C", str(target), "fetch", "--no-tags", "--depth", "1", "origin", commit],
                           check=True)
        # -f only on a fresh clone, whose empty index would otherwise look modified
        subprocess.run(["git", "-C", str(target), "checkout", "-q", *(["-f"] if fresh else []),
                        "--detach", commit], check=True)
    if _git(target, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{target} has local modifications to the upstream code")
    if _git(target, "rev-parse", "HEAD") != commit:
        raise RuntimeError(f"{target} is not at the audited commit {commit}")
    print(f"{target.relative_to(ROOT)} at {commit[:12]}")


def runtime(method: str, spec: dict) -> Path:
    """Private GPU environment for VideoScore / WorldModelBench; returns its python."""
    external = HERE / method / "external"
    venv, lock = external / ".venv", HERE / method / "runtime-lock.txt"
    python = venv / "bin" / "python"
    if shutil.which("uv") is None:
        raise RuntimeError(f"{method} needs uv to build its runtime (https://docs.astral.sh/uv/)")
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", spec["python"], str(venv)], check=True)
    marker = venv / f".{method}_runtime_lock.sha256"
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if not marker.exists() or marker.read_text().split()[0] != digest:
        pip = ["uv", "pip", "install", "--python", str(python)]
        subprocess.run([*pip, *spec.get("install_flags", []), "-r", str(lock)], check=True)
        subprocess.run([*pip, "--no-deps", "-e", str(external / spec["editable"])], check=True)
        if spec["after"]:
            subprocess.run([*pip, "--no-deps", *spec["after"]], check=True)
        marker.write_text(digest + "\n")
    return python


def weights(method: str, python: Path, model: tuple[str, str, str, str]) -> None:
    repository, revision, folder, marker_name = model
    target = HERE / method / "external" / "models" / folder
    expected = {"repository": repository, "revision": revision}
    marker = target / marker_name
    if marker.exists() and json.loads(marker.read_text()) == expected:
        return
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_ENABLE_HF_TRANSFER"}
    subprocess.run([str(python), "-c",
                    "import sys; from huggingface_hub import snapshot_download; "
                    "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])",
                    repository, revision, str(target)], check=True, env=env)
    marker.write_text(json.dumps(expected, indent=2))


def setup(method: str) -> Path:
    """Prepare `method` and return the python interpreter that runs its model."""
    spec = UPSTREAM[method]
    for folder, url, commit in spec["repos"]:
        checkout(HERE / method / "external" / folder, url, commit)
    python = Path(sys.executable)
    if "runtime" in spec:
        python = runtime(method, spec["runtime"])
        weights(method, python, spec["model"])
    return python


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ego2act baseline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=sorted(UPSTREAM))
    parser.add_argument("--output-dir", type=Path, help="default: .results/baselines/<method>")
    parser.add_argument("--setup-only", action="store_true", help="prepare the upstream code and exit")
    args, rest = parser.parse_known_args(argv)
    python = setup(args.method)
    if args.setup_only:
        return 0
    out = (args.output_dir or ROOT / ".results" / "baselines" / args.method).resolve()
    module = f"baselines.{args.method}"
    here = [sys.executable, "-m"]
    if args.method in ("pqsg", "rbench", "worldmodelbench"):
        commands = [[*here, f"{module}.run", "--output-dir", str(out), *rest]]
    elif args.method in ("wr_arena", "simple_vqa"):
        commands = [[*here, f"{module}.run", "--output", str(out / "report.json"), *rest]]
    else:  # videoscore: score on the GPU runtime, then summarize natively
        scores = str(out / "scores.json")
        commands = [[str(python), "-m", f"{module}.score", "--output", scores, *rest]]
        if "--prepare-only" not in rest:
            data_root = [a for i, a in enumerate(rest) if a.startswith("--data-root")
                         or (i and rest[i - 1] == "--data-root")]
            commands.append([*here, f"{module}.run", "--scores", scores,
                             "--output", str(out / "report.json"), *data_root])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
