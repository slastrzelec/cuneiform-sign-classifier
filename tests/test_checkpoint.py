"""Checkpoint loading: round trip, validation, and rejection of malicious files."""

import pickle
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.inference import load_checkpoint, predict
from src.model import build_model
from tests.conftest import CLASS_NAMES, make_image


def test_round_trip_gives_identical_logits(tiny_checkpoint):
    model, class_names, checkpoint = load_checkpoint(tiny_checkpoint)
    reference = build_model(num_classes=len(CLASS_NAMES), pretrained=False).eval()
    reference.load_state_dict(torch.load(tiny_checkpoint, weights_only=True)["model_state_dict"])
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), reference(x))
    assert class_names == CLASS_NAMES
    assert checkpoint["epoch"] == 3
    assert not model.training


def test_missing_required_key_is_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}, "class_names": CLASS_NAMES}, path)
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path)


def test_wrong_class_count_is_rejected(tmp_path):
    model = build_model(num_classes=7, pretrained=False)
    path = tmp_path / "mismatch.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "epoch": 1, "val_f1": 0.1},
        path,
    )
    with pytest.raises(RuntimeError):
        load_checkpoint(path)


class _Payload:
    """Pickle that would create a marker file if it were executed on load."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


def test_malicious_checkpoint_is_rejected_and_not_executed(tmp_path):
    marker = tmp_path / "pwned.txt"
    path = tmp_path / "evil.pt"
    torch.save({"model_state_dict": _Payload(marker)}, path)

    # control: the unrestricted loader WOULD run the payload, so the test is meaningful
    torch.load(path, weights_only=False)
    assert marker.exists()
    marker.unlink()

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
    assert not marker.exists()


def test_predict_returns_sorted_top3_and_cam(tiny_checkpoint, tmp_path):
    model, _, _ = load_checkpoint(tiny_checkpoint)
    make_image(tmp_path / "x.png", 120)
    idx, probs, cam = predict(model, Image.open(tmp_path / "x.png"))
    assert len(idx) == len(probs) == 3
    assert probs == sorted(probs, reverse=True)
    assert 0.0 < sum(probs) <= 1.0 + 1e-6
    assert cam.shape == (224, 224)


def test_predict_does_not_accumulate_hooks(tiny_checkpoint, tmp_path):
    model, _, _ = load_checkpoint(tiny_checkpoint)
    make_image(tmp_path / "x.png", 120)
    image = Image.open(tmp_path / "x.png")
    layer = model.layer4[-1]
    before = len(layer._forward_hooks), len(layer._backward_hooks)
    for _ in range(3):
        predict(model, image)
    assert (len(layer._forward_hooks), len(layer._backward_hooks)) == before
