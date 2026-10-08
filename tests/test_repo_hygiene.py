"""Nothing large or private may be tracked: data, checkpoints, notebooks outputs."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tracked_files():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(  # noqa: S603
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True  # noqa: S607
    )
    return out.stdout.splitlines()


def test_no_model_binaries_are_tracked():
    assert not [f for f in _tracked_files() if f.endswith((".pt", ".pth", ".safetensors", ".ckpt"))]


def test_no_dataset_is_tracked():
    tracked = _tracked_files()
    assert not [f for f in tracked if f.startswith(("data/", "files/", "checkpoints/"))]


def test_no_hardcoded_local_paths_in_code():
    offenders = []
    for path in ROOT.glob("*.py"):
        if "C:\\Users" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"hard-coded user paths in: {offenders}"


def test_no_unsafe_checkpoint_loading_in_code():
    offenders = []
    for path in [*ROOT.glob("*.py"), *(ROOT / "src").glob("*.py")]:
        if "weights_only=False" in path.read_text(encoding="utf-8"):
            offenders.append(path.name)
    assert not offenders, f"weights_only=False found in: {offenders}"
