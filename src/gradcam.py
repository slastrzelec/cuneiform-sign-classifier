"""Grad-CAM for ResNet (hooks on the last convolutional block) and a heat-map overlay."""

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Shows which image regions influenced the score of a chosen class."""

    def __init__(self, model, target_layer, image_size: int = 224):
        self.model = model
        self.image_size = image_size
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        """Detach the hooks (otherwise every call would stack another pair on the model)."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a (H, W) map scaled to [0, 1]."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay_heatmap(original_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend the Grad-CAM map ('jet' colormap) with the original image."""
    size = cam.shape[0]
    original = np.array(original_img.resize((size, size)).convert("RGB")).astype(float) / 255.0
    heatmap = cm.jet(cam)[:, :, :3]
    blended = np.clip((1 - alpha) * original + alpha * heatmap, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))
