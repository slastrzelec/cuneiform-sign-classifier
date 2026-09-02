"""
train.py

Trening klasyfikatora znakow klinowych: transfer learning na ResNet18
(wagi ImageNet), dostosowany do treningu na CPU.

Strategia transfer learningu:
- Zamrazamy wczesne warstwy konwolucyjne (layer1, layer2) - wykrywaja
  ogolne cechy niskiego poziomu (krawedzie, tekstury), uniwersalne
  niezaleznie od domeny
- Odmrazamy layer3, layer4 i finalna warstwe fc - te uczymy od nowa,
  zeby dostosowac sie do specyfiki wizualnej znakow klinowych
- To znaczaco redukuje liczbe trenowanych parametrow i przyspiesza
  trening na CPU, przy zachowaniu dobrej zdolnosci adaptacji

Model wybierany na podstawie macro-F1 na val (nie accuracy!), bo mamy
umiarkowany class imbalance - accuracy moglaby byc zdominowana przez
czeste klasy.

Uzycie:
    python train.py
"""

from pathlib import Path
import time

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from src.data import get_dataloaders

# ============== CONFIG ==============
DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15  # CALKOWITA docelowa liczba epok (liczona od poczatku, nie "ile dodatkowo")
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
DEVICE = torch.device("cpu")

# Wznowienie treningu: ustaw na sciezke checkpointu, zeby kontynuowac
# zamiast trenowac od zera. Ustaw na None, zeby zaczac od poczatku.
# Preferujemy last_model.pt (stan po OSTATNIEJ wykonanej epoce), zeby
# nie powtarzac epok. Jesli jeszcze nie istnieje (np. masz tylko stary
# best_model.pt sprzed tej wersji skryptu), uzywamy go jako fallback.
_last_ckpt = CHECKPOINT_DIR / "last_model.pt"
_best_ckpt = CHECKPOINT_DIR / "best_model.pt"
RESUME_FROM = _last_ckpt if _last_ckpt.exists() else _best_ckpt
# =====================================


def build_model(num_classes: int):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Zamroz wczesne warstwy (layer1, layer2, conv1, bn1)
    for name, param in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Podmien ostatnia warstwe pod nasza liczbe klas
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    """Jedna epoka treningu (optimizer podany) lub ewaluacji (optimizer=None)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def main():
    torch.manual_seed(RANDOM_SEED)

    print("Ladowanie danych...")
    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    print(f"Klasy ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}\n")

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trenowane parametry: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.1f}%)\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_f1 = -1.0
    start_epoch = 1
    history = []

    # --- Wznowienie treningu z checkpointu, jesli istnieje ---
    if RESUME_FROM is not None and Path(RESUME_FROM).exists():
        print(f"Wznawiam trening z checkpointu: {RESUME_FROM}")
        checkpoint = torch.load(RESUME_FROM, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_f1 = checkpoint.get("val_f1", -1.0)
        start_epoch = checkpoint.get("epoch", 0) + 1

        # Stan optymalizatora jest dostepny tylko jesli checkpoint byl
        # zapisany JUZ z ta wersja skryptu (ponizej zapisujemy go zawsze
        # od teraz). Stare checkpointy (jak Twoj obecny po 6 epokach)
        # go nie maja - trening ruszy z "swiezym" optymalizatorem,
        # co jest lekkim, ale akceptowalnym kompromisem.
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("  -> Wczytano rowniez stan optymalizatora (dokladne wznowienie)")
        else:
            print("  -> Brak zapisanego stanu optymalizatora w tym checkpoincie "
                  "(pochodzi z wczesniejszej wersji skryptu) - optymalizator "
                  "startuje na swiezo, co jest niegrozne")

        print(f"  -> Kontynuacja od epoki {start_epoch}, "
              f"najlepszy dotychczasowy val_f1={best_val_f1:.3f}\n")

        if start_epoch > NUM_EPOCHS:
            print(f"UWAGA: checkpoint jest juz na epoce {start_epoch - 1}, "
                  f"a NUM_EPOCHS={NUM_EPOCHS}. Zwieksz NUM_EPOCHS w CONFIG, "
                  f"zeby trenowac dalej.")
            return

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, val_preds, val_labels = run_epoch(model, val_loader, criterion)
        elapsed = time.time() - t0

        print(f"Epoka {epoch}/{NUM_EPOCHS} ({elapsed:.0f}s) | "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} train_f1={train_f1:.3f} | "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "train_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        # Zawsze zapisujemy checkpoint "ostatniej epoki" - dzieki temu
        # wznowienie po awaryjnym wylaczeniu komputera zaczyna sie od
        # OSTATNIO WYKONANEJ epoki, a nie od ostatniej NAJLEPSZEJ (ktora
        # moglaby byc kilka epok wczesniej).
        last_checkpoint_path = CHECKPOINT_DIR / "last_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "epoch": epoch,
            "val_f1": val_f1,
            "val_acc": val_acc,
        }, last_checkpoint_path)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, best_checkpoint_path)
            print(f"  -> Nowy najlepszy model zapisany (val_f1={val_f1:.3f})")

    print("\n" + "=" * 60)
    print("Trening zakonczony. Ladowanie najlepszego checkpointu do ewaluacji na TEST...")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, test_preds, test_labels = run_epoch(model, test_loader, criterion)
    print(f"\nWYNIK NA TEST: loss={test_loss:.3f} accuracy={test_acc:.3f} macro_f1={test_f1:.3f}\n")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()