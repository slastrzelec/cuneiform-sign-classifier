"""
src/data.py

Buduje PyTorch DataLoadery na podstawie struktury folderow wygenerowanej
przez build_dataset.py:
    data/processed/train/<charname>/*.png
    data/processed/val/<charname>/*.png
    data/processed/test/<charname>/*.png

Uzywamy torchvision.datasets.ImageFolder, bo dane sa juz poukladane
w podfoldery per klasa - to standardowy, sprawdzony sposob i nie
wymaga wlasnej klasy Dataset.

WAZNA DECYZJA DOMENOWA: BRAK random horizontal/vertical flip w augmentacji.
Znaki klinowe nie sa symetryczne - odbicie lustrzane zmienia sens znaku
(albo tworzy cos, co nie jest zadnym prawdziwym znakiem). Uzywamy tylko
malej rotacji i lekkich zmian jasnosci/kontrastu, ktore odpowiadaja
naturalnej zmiennosci warunkow renderowania/oswietlenia tabliczki.
"""

from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_dataloaders(data_dir: Path, batch_size: int = 32, image_size: int = 224,
                     num_workers: int = 0):
    """
    Zwraca (train_loader, val_loader, test_loader, class_names, class_weights).

    class_weights to tensor wag odwrotnie proporcjonalnych do liczebnosci
    klasy w train - do uzycia w nn.CrossEntropyLoss(weight=...), bo mamy
    umiarkowany class imbalance (~5x roznicy miedzy najczestsza a najrzadsza
    z top-30 klas).
    """
    data_dir = Path(data_dir)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=get_transforms(image_size, train=True))
    val_ds = datasets.ImageFolder(data_dir / "val", transform=get_transforms(image_size, train=False))
    test_ds = datasets.ImageFolder(data_dir / "test", transform=get_transforms(image_size, train=False))

    # ImageFolder sortuje klasy alfabetycznie i przypisuje indeksy - upewniamy
    # sie, ze train/val/test maja IDENTYCZNE mapowanie klasa->indeks
    assert train_ds.classes == val_ds.classes == test_ds.classes, (
        "Niespojne zbiory klas miedzy train/val/test! "
        "Sprawdz test_every_class_present_in_train w tests/test_no_leakage.py"
    )
    class_names = train_ds.classes

    # Wagi klas na podstawie liczebnosci w train
    train_counts = Counter(label for _, label in train_ds.samples)
    n_classes = len(class_names)
    n_total = len(train_ds.samples)
    weights = torch.zeros(n_classes)
    for class_idx, count in train_counts.items():
        weights[class_idx] = n_total / (n_classes * count)  # inverse frequency

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names, weights


if __name__ == "__main__":
    # Szybki test modulu - uruchom `python src/data.py` z katalogu glownego
    # projektu, zeby sprawdzic, ze wszystko sie laduje poprawnie.
    DATA_DIR = Path("data/processed")
    train_loader, val_loader, test_loader, class_names, weights = get_dataloaders(DATA_DIR)
    print(f"Klasy ({len(class_names)}): {class_names}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    print(f"Wagi klas (min/max): {weights.min():.2f} / {weights.max():.2f}")

    batch_x, batch_y = next(iter(train_loader))
    print(f"Ksztalt batcha: {batch_x.shape}, etykiety: {batch_y[:8]}")
