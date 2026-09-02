"""
evaluate.py

Poglebiona ewaluacja najlepszego modelu na zbiorze TEST:
1. Confusion matrix (pelna, 30x30) - zapisana jako PNG
2. Top-10 najczesciej mylonych par klas
3. Weryfikacja hipotezy z EDA: czy accuracy dla znakow LICZBOWYCH
   (U, ASZ, DISZ_(1), MIN_(2)) roznie sie miedzy okresami historycznymi
   (widzielismy w eda_period_comparison.png, ze np. znak U wyglada
   fizycznie inaczej w ED IIIa/b vs pozniejsze okresy)

Uzycie:
    python evaluate.py
"""

from pathlib import Path
from collections import defaultdict, Counter
import re

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data import get_dataloaders
from train import build_model, DATA_DIR, IMAGE_SIZE, BATCH_SIZE, DEVICE, CHECKPOINT_DIR

# ============== CONFIG ==============
TRANSLIT_CSV = Path(r"C:\Users\slast\PYTHON\0_projekty do portfolio\20_cuneiform-sign-classifier\files\translitmetadata.csv")
OUTPUT_DIR = Path("eda_outputs")
NUMERIC_SIGNS_OF_INTEREST = ["U", "ASZ", "DISZ_(1)", "MIN_(2)"]  # znaki z wątpliwości EDA
# =====================================


def build_tablet_period_map(translit_csv_path):
    """Ta sama logika co w eda.py - wyciaga HS_#### z Filename i mapuje na okres."""
    pattern = re.compile(r"HS_\d+")
    tablet_periods = defaultdict(Counter)
    with open(translit_csv_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 15:
                continue
            m = pattern.search(parts[1])
            if m:
                tablet_periods[m.group(0)][parts[5]] += 1
    return {t: c.most_common(1)[0][0] for t, c in tablet_periods.items()}


def get_test_filepaths_and_tablets(test_dataset):
    """
    Wyciaga z ImageFolder liste (filepath, true_label_idx) i odpowiadajaca
    nazwe tabliczki (parsowana z nazwy pliku, tak jak w build_dataset.py).
    """
    results = []
    for filepath, label_idx in test_dataset.samples:
        filename = Path(filepath).name
        stem = filename.rsplit(".", 1)[0]
        parts = stem.split("_")
        tablet = f"{parts[-4]}_{parts[-3]}" if len(parts) >= 4 else None
        results.append((filepath, label_idx, tablet))
    return results


def main():
    print("Ladowanie danych i modelu...")
    train_loader, val_loader, test_loader, class_names, _ = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Wczytano model z epoki {checkpoint['epoch']}, val_f1={checkpoint['val_f1']:.3f}\n")

    # --- Inference na test z zachowaniem sciezek plikow ---
    test_info = get_test_filepaths_and_tablets(test_loader.dataset)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    assert len(all_preds) == len(test_info), "Niezgodna liczba predykcji i plikow"

    # ========== 1. CONFUSION MATRIX ==========
    print("Generuje confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_title("Confusion Matrix - test set")
    plt.colorbar(im)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {cm_path}")

    # ========== 2. TOP-10 NAJCZESCIEJ MYLONYCH PAR ==========
    print("\n" + "=" * 60)
    print("TOP-10 NAJCZESCIEJ MYLONYCH PAR KLAS")
    print("=" * 60)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    for count, true_cls, pred_cls in confusions[:10]:
        print(f"  {true_cls:<25} -> mylony z -> {pred_cls:<25} ({count}x)")

    # ========== 3. ACCURACY PER OKRES DLA ZNAKOW LICZBOWYCH ==========
    print("\n" + "=" * 60)
    print("ACCURACY PER OKRES - ZNAKI LICZBOWE (hipoteza z EDA)")
    print("=" * 60)

    tablet_period = build_tablet_period_map(TRANSLIT_CSV)
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    for sign in NUMERIC_SIGNS_OF_INTEREST:
        if sign not in class_names:
            continue
        sign_idx = class_names.index(sign)

        period_correct = defaultdict(int)
        period_total = defaultdict(int)
        for (filepath, true_label, tablet), pred_label in zip(test_info, all_preds):
            if true_label != sign_idx:
                continue
            period = tablet_period.get(tablet, "nieznany")
            period_total[period] += 1
            if pred_label == true_label:
                period_correct[period] += 1

        print(f"\n  Znak: {sign} (n={sum(period_total.values())} w test)")
        for period in sorted(period_total.keys(), key=lambda p: -period_total[p]):
            acc = period_correct[period] / period_total[period]
            print(f"    {period:<45} {period_correct[period]:>3}/{period_total[period]:<3} "
                  f"({acc*100:.0f}%)")

    print("\nGotowe. Zobacz eda_outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
