"""Transforms and loaders: the domain rule (no mirror flips) and class weighting."""

import pytest
import torch
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD, get_dataloaders, get_transforms
from tests.conftest import make_image

FLIPS = (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip)


def test_train_pipeline_has_no_mirror_flips():
    """Mirroring a cuneiform sign changes its identity, so flips must never be used."""
    pipeline = get_transforms(224, train=True).transforms
    assert not [t for t in pipeline if isinstance(t, FLIPS)]


def test_train_pipeline_uses_small_rotation_only():
    rotations = [t for t in get_transforms(224, train=True).transforms
                 if isinstance(t, transforms.RandomRotation)]
    assert len(rotations) == 1
    assert max(abs(d) for d in rotations[0].degrees) <= 10


def test_eval_pipeline_is_deterministic(tmp_path):
    image = make_image_and_open(tmp_path / "x.png")
    tf = get_transforms(224, train=False)
    assert torch.equal(tf(image), tf(image))


def test_output_shape_and_normalisation(tmp_path):
    image = make_image_and_open(tmp_path / "x.png", shade=255)
    out = get_transforms(224, train=False)(image)
    assert out.shape == (3, 224, 224)
    expected_white = torch.tensor([(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)])
    assert torch.allclose(out[:, 0, 0], expected_white, atol=1e-4)


def make_image_and_open(path, shade=128):
    from PIL import Image

    make_image(path, shade)
    return Image.open(path).convert("RGB")


def test_class_weights_are_inverse_frequency(image_folder_dataset):
    *_, class_names, weights = get_dataloaders(image_folder_dataset, batch_size=2, image_size=64)
    assert class_names == ["AA", "BB", "CC"]
    # train counts: AA=6, BB=3, CC=2  -> rarest class gets the largest weight
    assert weights[2] > weights[1] > weights[0]
    n_total, n_classes = 11, 3
    assert weights[0].item() == pytest.approx(n_total / (n_classes * 6))
    assert weights[2].item() == pytest.approx(n_total / (n_classes * 2))


def test_loaders_share_class_to_index_mapping(image_folder_dataset):
    train, val, test, class_names, _ = get_dataloaders(image_folder_dataset, batch_size=2, image_size=64)
    assert train.dataset.classes == val.dataset.classes == test.dataset.classes == class_names


def test_loaders_refuse_inconsistent_class_sets(image_folder_dataset):
    make_image(image_folder_dataset / "test" / "ZZ" / "extra.png", 50)
    with pytest.raises(AssertionError):
        get_dataloaders(image_folder_dataset, batch_size=2, image_size=64)
