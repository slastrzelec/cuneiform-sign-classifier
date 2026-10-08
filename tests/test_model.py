"""Architecture and the transfer-learning freeze policy."""

import torch
import torch.nn as nn

from src.model import build_model


def test_output_shape():
    model = build_model(num_classes=30, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 30)


def test_only_late_layers_are_trainable():
    model = build_model(num_classes=5, pretrained=False)
    for name, param in model.named_parameters():
        should_train = name.startswith(("layer3", "layer4", "fc"))
        assert param.requires_grad == should_train, name


def test_frozen_layers_do_not_change_after_an_optimizer_step():
    torch.manual_seed(0)
    model = build_model(num_classes=3, pretrained=False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    optimizer.step()

    for name, param in model.named_parameters():
        changed = not torch.equal(before[name], param.detach())
        assert changed == name.startswith(("layer3", "layer4", "fc")), name
