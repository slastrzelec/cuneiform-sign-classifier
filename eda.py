"""
eda.py

Eksploracyjna analiza danych (EDA) na zbudowanym datasecie top-30 znakow.

Co robi:
1. Wczytuje manifest.csv (wynik build_dataset.py)
2. Odtwarza mape tablet -> Time_period / Language z translitmetadata.csv
   (przez wyciagniecie tokenu HS_#### z pola Filename - to pole jest
   niespojne co do pelnej nazwy pliku, ale sam identyfikator tabliczki
   w nim jest wiarygodny)
3. Drukuje:
   - rozklad klas w train/val/test (czy proporcje sa zblizone)
   - rozklad okresow (Time_period) w przefiltrowanym datasecie top-30
   - statystyki wymiarow obrazow (min/max/srednia szerokosc i wysokosc)
4. Zapisuje dwa obrazy PNG do wizualnej inspekcji:
   - eda_class_samples.png: po jednym przykladzie z kazdej z 30 klas
   - eda_period_comparison.png: ten sam znak (jesli dostepny) z roznych
     okresow obok siebie - do oceny, czy warto zawezic do jednego okresu

Uzycie:
    python eda.py
"""

import csv
import re
import random
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
import matplotlib.pyplot as plt

# ============== CONFIG ==============
PROJECT_ROOT = Path(r"C:\Users\slast\PYTHON\0_projekty do portfolio\20_cuneiform-sign-classifier")
TRANSLIT_CSV = PROJECT_ROOT / "files" / "translitmetadata.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"

N_SAMPLE_FOR_DIMENSIONS = 500  # ile plikow losowo sprawdzic pod katem wymiarow
RANDOM_SEED = 42
# =====================================


def load_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_tablet_period_map(translit_csv_path):
    """
    Wyciaga token HS_#### z pola Filename w translitmetadata.csv i mapuje
    go na najczestszy Time_period/Language dla tej tabliczki.
    Pole Filename ma niespojny pelny schemat, ale sam token HS_#### jest
    wiarygodny (sprawdzone wczesniej: 100% wierszy go zawiera).
    """
    pattern = re.compile(r"HS_\d+")
    tablet_periods = defaultdict(Counter)
    tablet_languages = defaultdict(Counter)

    with open(translit_csv_path, encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(";")
            if len(parts) < 15:
                continue
            filename = parts[1]
            time_period = parts[5]
            language = parts[6]
            m = pattern.search(filename)
            if not m:
                continue
            tablet = m.group(0)
            tablet_periods[tablet][time_period] += 1
            tablet_languages[tablet][language] += 1

    resolved_period = {t: c.most_common(1)[0][0] for t, c in tablet_periods.items()}
    resolved_language = {t: c.most_common(1)[0][0] for t, c in tablet_languages.items()}
    return resolved_period, resolved_language


def print_class_distribution(manifest):
    print("=" * 70)
    print("ROZKLAD KLAS W SPLITACH")
    print("=" * 70)

    counts = defaultdict(lambda: defaultdict(int))
    for row in manifest:
        counts[row["charname"]][row["split"]] += 1

    print(f"{'Klasa':<20} {'train':>8} {'val':>8} {'test':>8} {'razem':>8}")
    for charname in sorted(counts.keys(), key=lambda c: -sum(counts[c].values())):
        c = counts[charname]
        total = c["train"] + c["val"] + c["test"]
        print(f"{charname:<20} {c['train']:>8} {c['val']:>8} {c['test']:>8} {total:>8}")


def print_period_distribution(manifest, tablet_period):
    print("\n" + "=" * 70)
    print("ROZKLAD OKRESOW (Time_period) W PRZEFILTROWANYM DATASECIE (top-30)")
    print("=" * 70)

    period_counts = Counter()
    unknown = 0
    for row in manifest:
        period = tablet_period.get(row["tablet"])
        if period is None:
            unknown += 1
        else:
            period_counts[period] += 1

    total = sum(period_counts.values()) + unknown
    for period, count in period_counts.most_common():
        pct = count / total * 100
        print(f"  {period:<45} {count:>6} ({pct:.1f}%)")
    if unknown:
        print(f"  {'(nieznany - tabliczka nie znaleziona w CSV)':<45} {unknown:>6}")


def print_image_dimension_stats(manifest):
    print("\n" + "=" * 70)
    print(f"STATYSTYKI WYMIAROW OBRAZOW (probka {N_SAMPLE_FOR_DIMENSIONS})")
    print("=" * 70)

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(manifest, min(N_SAMPLE_FOR_DIMENSIONS, len(manifest)))

    widths, heights = [], []
    for row in sample:
        fpath = PROCESSED_DIR / row["split"] / row["charname"] / row["filename"]
        try:
            with Image.open(fpath) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            print(f"  Nie mozna otworzyc {fpath}: {e}")

    if widths:
        print(f"  Szerokosc: min={min(widths)}, max={max(widths)}, "
              f"srednia={sum(widths)/len(widths):.1f}")
        print(f"  Wysokosc:  min={min(heights)}, max={max(heights)}, "
              f"srednia={sum(heights)/len(heights):.1f}")


def save_class_samples_grid(manifest):
    """Zapisuje siatke z jednym przykladowym obrazem na kazda z 30 klas."""
    rng = random.Random(RANDOM_SEED)
    by_class = defaultdict(list)
    for row in manifest:
        by_class[row["charname"]].append(row)

    classes = sorted(by_class.keys())
    n = len(classes)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i, charname in enumerate(classes):
        row = rng.choice(by_class[charname])
        fpath = PROCESSED_DIR / row["split"] / row["charname"] / row["filename"]
        try:
            img = Image.open(fpath)
            axes[i].imshow(img, cmap="gray")
        except Exception:
            pass
        axes[i].set_title(charname, fontsize=8)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "eda_class_samples.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nZapisano: {out_path}")


def save_period_comparison(manifest, tablet_period):
    """
    Dla kilku najczestszych klas pokazuje przyklady z roznych okresow
    obok siebie (jesli dana klasa wystepuje w wiecej niz jednym okresie)
    - pomaga ocenic, czy warto zawezic dataset do jednego okresu.
    """
    by_class_period = defaultdict(lambda: defaultdict(list))
    for row in manifest:
        period = tablet_period.get(row["tablet"], "unknown")
        by_class_period[row["charname"]][period].append(row)

    # wybierz klasy, ktore maja przyklady z >= 2 roznych okresow
    multi_period_classes = [
        c for c, periods in by_class_period.items() if len(periods) >= 2
    ]
    if not multi_period_classes:
        print("\nBrak klas z przykladami z wiecej niz jednego okresu - "
              "pomijam eda_period_comparison.png")
        return

    rng = random.Random(RANDOM_SEED)
    selected_classes = multi_period_classes[:5]

    max_periods = max(len(by_class_period[c]) for c in selected_classes)
    fig, axes = plt.subplots(len(selected_classes), max_periods,
                              figsize=(max_periods * 2, len(selected_classes) * 2))
    if len(selected_classes) == 1:
        axes = axes.reshape(1, -1)

    for i, charname in enumerate(selected_classes):
        periods = sorted(by_class_period[charname].keys())
        for j in range(max_periods):
            ax = axes[i, j]
            if j < len(periods):
                period = periods[j]
                row = rng.choice(by_class_period[charname][period])
                fpath = PROCESSED_DIR / row["split"] / row["charname"] / row["filename"]
                try:
                    img = Image.open(fpath)
                    ax.imshow(img, cmap="gray")
                except Exception:
                    pass
                short_period = period[:20]
                ax.set_title(f"{charname}\n{short_period}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "eda_period_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Zapisano: {out_path}")


def main():
    manifest = load_manifest()
    print(f"Wczytano manifest: {len(manifest)} wierszy\n")

    tablet_period, tablet_language = build_tablet_period_map(TRANSLIT_CSV)

    print_class_distribution(manifest)
    print_period_distribution(manifest, tablet_period)
    print_image_dimension_stats(manifest)
    save_class_samples_grid(manifest)
    save_period_comparison(manifest, tablet_period)

    print("\nGotowe. Obejrzyj pliki w folderze eda_outputs/.")


if __name__ == "__main__":
    main()
