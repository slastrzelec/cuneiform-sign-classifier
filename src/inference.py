"""Checkpoint loading and single-image prediction (no Streamlit, so it is testable)."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.data import get_transforms
from src.gradcam import GradCAM
from src.model import build_model

REQUIRED_KEYS = ("model_state_dict", "class_names", "epoch", "val_f1")


def load_checkpoint(path: Path):
    """Load a trained checkpoint and return (model in eval mode, class_names, checkpoint).

    Security: weights_only=True makes torch use a restricted unpickler that only accepts
    tensors and plain Python containers. A tampered file that tries to run code is rejected
    (UnpicklingError) instead of being executed.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    class_names = list(checkpoint["class_names"])
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])  # raises on shape / key mismatch
    model.eval()
    return model, class_names, checkpoint


def predict(model, image: Image.Image, image_size: int = 224):
    """Returns (top3 class indices, top3 probabilities, Grad-CAM map for the top-1 class)."""
    transform = get_transforms(image_size, train=False)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1)[0]
    top_probs, top_indices = probs.topk(min(3, probs.numel()))

    gradcam = GradCAM(model, model.layer4[-1], image_size)
    try:
        cam = gradcam.generate(input_tensor.clone().requires_grad_(True), top_indices[0].item())
    finally:
        gradcam.remove()

    return top_indices.tolist(), top_probs.tolist(), cam
