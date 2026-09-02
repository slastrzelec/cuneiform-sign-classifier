
"""
embeddings.py
 
Wizualizacja przestrzeni cech (embeddingow) wyuczonych przez model.
Wyciagamy wektory z ostatniej warstwy przed klasyfikacja (avgpool ResNet18,
512-wymiarowy wektor na obraz) i rzutujemy je na 2D metoda t-SNE.
 
Dwa wykresy:
1. eda_outputs/embeddings_all_classes.png
   Ogolna mapa wszystkich 30 klas - sprawdza, czy pary czesto mylone
   przez model (LUGAL/LU2, A/MIN_(2)) faktycznie leza blisko siebie
   w przestrzeni cech (geometryczny dowod na sensownosc bledow modelu).
 
2. eda_outputs/embeddings_period_drift.png
   Fokus na znaki liczbowe (U, ASZ, DISZ_(1), MIN_(2)) pokolorowane
   wg okresu historycznego - test hipotezy, czy znak U faktycznie
   tworzy DWA oddzielne skupiska w przestrzeni cech (stary okragly
   wariant vs pozniejszy klinowy), co bylby ilosciowym potwierdzeniem
   odkrycia z evaluate.py (spadek accuracy dla U w ED IIIb).
 
Uzycie:
    python embeddings.py
"""
 
from pathlib import Path
from collections import defaultdict, Counter
import re
 
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
 
from src.data import get_dataloaders
from train import build_model, DATA_DIR, IMAGE_SIZE, BATCH_SIZE, DEVICE, CHECKPOINT_DIR
 
# ============== CONFIG ==============
TRANSLIT_CSV = Path(r"C:\Users\slast\PYTHON\0_projekty do portfolio\20_cuneiform-sign-classifier\files\translitmetadata.csv")
OUTPUT_DIR = Path("eda_outputs")
NUMERIC_SIGNS_OF_INTEREST = ["U", "ASZ", "DISZ_(1)", "MIN_(2)"]
RANDOM_SEED = 42
# =====================================
 
 
def build_tablet_period_map(translit_csv_path):
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
 
 
def extract_tablet_from_filepath(filepath: str):
    stem = Path(filepath).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return f"{parts[-4]}_{parts[-3]}"
 
 
def extract_features(model, loader):
    """Rejestruje hook na avgpool i zbiera 512-wym. wektory cech dla kazdego obrazu."""
    features_batch = {}
 
    def hook(module, input, output):
        features_batch["feat"] = output.flatten(1).detach().numpy()
 
    handle = model.avgpool.register_forward_hook(hook)
 
    all_features, all_labels, all_filepaths = [], [], []
    filepaths = [fp for fp, _ in loader.dataset.samples]
    idx = 0
 
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            _ = model(images)
            feats = features_batch["feat"]
            all_features.append(feats)
            all_labels.extend(labels.tolist())
            batch_size = images.size(0)
            all_filepaths.extend(filepaths[idx:idx + batch_size])
            idx += batch_size
 
    handle.remove()
    return np.vstack(all_features), all_labels, all_filepaths
 
 
def main():
    print("Ladowanie danych i modelu...")
    _, _, test_loader, class_names, _ = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model wczytany (epoka {checkpoint['epoch']})\n")
 
    print("Wyciagam wektory cech (avgpool, 512-wym.) dla test setu...")
    features, labels, filepaths = extract_features(model, test_loader)
    print(f"  -> {features.shape[0]} wektorow, wymiar {features.shape[1]}\n")
 
    print("Licze t-SNE (moze potrwac chwile na CPU)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, init="pca")
    embedding_2d = tsne.fit_transform(features)
    print("  Gotowe.\n")
 
    OUTPUT_DIR.mkdir(exist_ok=True)
 
    # ========== WYKRES 1: wszystkie 30 klas ==========
    print("Generuje mape wszystkich klas...")
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = plt.get_cmap("gist_ncar", len(class_names))
 
    for class_idx, class_name in enumerate(class_names):
        mask = np.array(labels) == class_idx
        points = embedding_2d[mask]
        ax.scatter(points[:, 0], points[:, 1], s=15, alpha=0.6,
                   color=cmap(class_idx), label=class_name)
        if len(points) > 0:
            cx, cy = points[:, 0].mean(), points[:, 1].mean()
            ax.annotate(class_name, (cx, cy), fontsize=9, fontweight="bold",
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
 
    ax.set_title("Przestrzen cech modelu (t-SNE) - wszystkie 30 klas\n"
                 "(pary czesto mylone przez model powinny lezec blisko siebie)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    out1 = OUTPUT_DIR / "embeddings_all_classes.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"  Zapisano: {out1}")
 
    # ========== WYKRES 2: dryf paleograficzny znakow liczbowych ==========
    print("\nGeneruje analize dryfu paleograficznego...")
    tablet_period = build_tablet_period_map(TRANSLIT_CSV)
    tablets = [extract_tablet_from_filepath(fp) for fp in filepaths]
    periods = [tablet_period.get(t, "nieznany") for t in tablets]
 
    all_periods_present = sorted(set(
        p for i, p in enumerate(periods)
        if class_names[labels[i]] in NUMERIC_SIGNS_OF_INTEREST
    ))
    period_colors = {p: plt.cm.tab10(i / max(len(all_periods_present) - 1, 1))
                     for i, p in enumerate(all_periods_present)}
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
 
    for i, sign in enumerate(NUMERIC_SIGNS_OF_INTEREST):
        ax = axes[i]
        if sign not in class_names:
            ax.set_visible(False)
            continue
        sign_idx = class_names.index(sign)
        mask = np.array(labels) == sign_idx
        sign_points = embedding_2d[mask]
        sign_periods = [periods[j] for j in range(len(labels)) if labels[j] == sign_idx]
 
        for period in sorted(set(sign_periods)):
            period_mask = np.array([p == period for p in sign_periods])
            pts = sign_points[period_mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=40, alpha=0.8,
                      color=period_colors.get(period, "gray"), label=period)
 
        ax.set_title(f"Znak: {sign} (n={mask.sum()})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="best")
 
    plt.suptitle("Dryf paleograficzny znakow liczbowych w przestrzeni cech modelu\n"
                "(oddzielne skupiska = model 'widzi' rozne warianty graficzne tego samego znaku)",
                fontsize=12)
    plt.tight_layout()
    out2 = OUTPUT_DIR / "embeddings_period_drift.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"  Zapisano: {out2}")
 
    print("\nGotowe. Zobacz oba pliki w eda_outputs/.")
 
 
if __name__ == "__main__":
    main()
 
