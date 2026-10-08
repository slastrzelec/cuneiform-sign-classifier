"""ResNet18 model definition (kept separate from train.py so it can be imported cheaply)."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# Layers whose parameters are trained; everything else (conv1, bn1, layer1, layer2) is frozen.
TRAINABLE_PREFIXES = ("layer3", "layer4", "fc")


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet18 with a new classification head.

    pretrained=True downloads the ImageNet weights (needed for training).
    pretrained=False builds the bare architecture (used for inference, where the
    weights are loaded from a checkpoint anyway, and in tests, which must not need a network).

    "Frozen" means: no gradient updates. BatchNorm running statistics of the frozen layers
    are still updated while the model is in train mode.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(TRAINABLE_PREFIXES)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
