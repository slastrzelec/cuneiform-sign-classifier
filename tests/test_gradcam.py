"""Grad-CAM on a random-weight model: contract only (shape, range, hooks), not visual quality."""

import numpy as np
import torch
from PIL import Image

from src.gradcam import GradCAM, overlay_heatmap
from src.model import build_model


def _model():
    torch.manual_seed(0)
    return build_model(num_classes=4, pretrained=False).eval()


def test_map_has_expected_shape_and_range():
    model = _model()
    cam = GradCAM(model, model.layer4[-1], image_size=224)
    out = cam.generate(torch.randn(1, 3, 224, 224).requires_grad_(True), class_idx=1)
    cam.remove()
    assert out.shape == (224, 224)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6
    assert np.isfinite(out).all()


def test_different_target_classes_give_different_maps():
    model = _model()
    x = torch.randn(1, 3, 224, 224)
    cam = GradCAM(model, model.layer4[-1], image_size=224)
    a = cam.generate(x.clone().requires_grad_(True), class_idx=0)
    b = cam.generate(x.clone().requires_grad_(True), class_idx=3)
    cam.remove()
    assert not np.allclose(a, b)


def test_remove_detaches_hooks():
    model = _model()
    layer = model.layer4[-1]
    before = len(layer._forward_hooks), len(layer._backward_hooks)
    cam = GradCAM(model, layer)
    assert len(layer._forward_hooks) == before[0] + 1
    cam.remove()
    assert (len(layer._forward_hooks), len(layer._backward_hooks)) == before


def test_overlay_returns_rgb_image_of_map_size():
    cam = np.zeros((224, 224))
    overlay = overlay_heatmap(Image.new("L", (80, 60), 100), cam)
    assert overlay.mode == "RGB"
    assert overlay.size == (224, 224)
