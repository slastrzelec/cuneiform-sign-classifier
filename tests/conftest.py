"""Shared fixtures. Everything here is synthetic: no dataset, no trained checkpoint, no network."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model  # noqa: E402

CLASS_NAMES = ["AA", "BB", "CC", "DD"]


def make_image(path: Path, shade: int, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (shade, shade, shade)).save(path)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A checkpoint of the real architecture with random weights (4 classes)."""
    torch.manual_seed(0)
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "epoch": 3,
            "val_f1": 0.5,
            "val_acc": 0.5,
        },
        path,
    )
    return path


@pytest.fixture(scope="session")
def synthetic_gallery(tmp_path_factory) -> Path:
    """gallery/<class>/<image>.png with a few tiny images per class."""
    root = tmp_path_factory.mktemp("gallery")
    for i, name in enumerate(CLASS_NAMES):
        for j in range(2):
            make_image(root / name / f"{name.lower()}_{j}.png", shade=40 + 50 * i + 10 * j)
    return root


@pytest.fixture()
def image_folder_dataset(tmp_path) -> Path:
    """train/val/test image folders with unequal class sizes (for class-weight tests)."""
    counts = {"train": {"AA": 6, "BB": 3, "CC": 2}, "val": {"AA": 1, "BB": 1, "CC": 1},
              "test": {"AA": 1, "BB": 1, "CC": 1}}
    for split, per_class in counts.items():
        for cls, n in per_class.items():
            for k in range(n):
                make_image(tmp_path / split / cls / f"{cls}_{k}.png", shade=30 * (k + 1))
    return tmp_path
