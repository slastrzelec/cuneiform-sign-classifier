"""
build_dataset.py

Buduje przefiltrowany dataset do klasyfikacji znakow klinowych z MaiCuBeDa.

WAZNA UWAGA O JAKOSCI DANYCH:
Pole 'Filename' w translitmetadata.csv NIE odpowiada rzeczywistym nazwom
plikow w opublikowanym zipie z obrazami (inny schemat - CSV ma dodatkowe
pole liczbowe, prawdopodobnie wordindex, ktorego nie ma w nazwach na dysku).
Dlatego NIE dopasowujemy po pelnej nazwie pliku z CSV, tylko:
  1. Parsujemy nazwy plikow BEZPOSREDNIO z dysku (transliteracja + tabliczka)
  2. Budujemy mape transliteracja -> charname na podstawie CSV (923 unikalne
     odczyty, tylko 18 niejednoznacznych - rozwiazywane wiekszoscia glosow)
  3. Kazdemu plikowi na dysku przypisujemy charname przez ta mape

Co robi caly skrypt:
1. Wczytuje translitmetadata.csv, buduje mape transliteracja -> charname
2. Skanuje folder char/, parsuje kazda nazwe pliku (reading + tablet)
3. Przypisuje charname kazdemu plikowi przez mape z kroku 1
4. Wybiera top-N najczestszych klas (na podstawie tego co REALNIE jest na dysku)
5. Dzieli dane na train/val/test GRUPOWO po nazwie tabliczki (bez wycieku)
6. Kopiuje pliki do docelowej struktury:
     data/processed/train/<charname>/<plik>
     data/processed/val/<charname>/<plik>
     data/processed/test/<charname>/<plik>
7. Zapisuje manifest.csv

Uzycie:
    python build_dataset.py
"""

import csv
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

# ============== CONFIG - dostosuj do swojego srodowiska ==============
TRANSLIT_CSV = Path(r"C:\Users\slast\PYTHON\0_projekty do portfolio\20_cuneiform-sign-classifier\files\translitmetadata.csv")
CHAR_IMAGES_DIR = Path(r"C:\Users\slast\Downloads\MaiCuBeDa_Annotations_MSII\char")
OUTPUT_DIR = Path(r"C:\Users\slast\PYTHON\0_projekty do portfolio\20_cuneiform-sign-classifier\data\processed")

TOP_N_CLASSES = 30
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42
# =======================================================================


def parse_translit_csv(path: Path):
    """
    Parsuje translitmetadata.csv recznie, bo pole 'charname' czasem
    zawiera dodatkowe srodniki (np. 'DAR_(GN;_?U-gun__SI-gun)'),
    co psuje naiwne pd.read_csv(sep=';').
    """
    cols = ["ID", "Filename", "CDLI_Number", "IRI", "IIIF", "Time_period",
            "Language", "Genre", "side", "bbox", "column", "line",
            "charindex", "charname", "transliteration"]

    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()  # pomijamy naglowek
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 15:
                continue
            first13 = parts[:13]
            charname = ";".join(parts[13:-1])
            transliteration = parts[-1]
            row = dict(zip(cols, first13 + [charname, transliteration]))
            rows.append(row)
    return rows


def build_reading_to_charname_map(rows):
    """Mapuje transliteracje (reading) -> najczestszy charname dla tego reading."""
    reading_charnames = defaultdict(Counter)
    for r in rows:
        reading_charnames[r["transliteration"]][r["charname"]] += 1

    resolved = {}
    ambiguous = 0
    for reading, counter in reading_charnames.items():
        top_charname, _ = counter.most_common(1)[0]
        resolved[reading] = top_charname
        if len(counter) > 1:
            ambiguous += 1
    print(f"Mapa transliteracja->charname: {len(resolved)} odczytow "
          f"({ambiguous} niejednoznacznych, rozwiazanych wiekszoscia glosow)")
    return resolved


def parse_disk_filename(filename: str):
    """
    Parsuje nazwe pliku BEZPOSREDNIO z dysku wg schematu:
    TRANSLITERATION_COLUMN_LINE_CHARINDEX_TABLETNAME_SURFACE.ext

    Odporne na readingi zawierajace wewnetrzne podkreslniki (np. gdy
    slash w oryginalnej transliteracji zostal zamieniony na '_').
    Zaklada, ze ostatnie 4 tokeny to zawsze:
    TABLET_PREFIX, TABLET_NUM, SURFACE_CODE, SURFACE_NAME
    a wszystko przed nimi minus ostatnie 3 liczbowe pola to reading.

    Zwraca (reading, tablet_name) albo (None, None) jesli nie da sie sparsowac.
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 8:
        return None, None

    tablet = f"{parts[-4]}_{parts[-3]}"
    front_parts = parts[:-4]  # wszystko przed tablet+surface
    if len(front_parts) < 4:
        return None, None

    reading = "_".join(front_parts[:-3])  # reading moze miec wewnetrzne "_"
    return reading, tablet


def group_split(tablets: list, train_frac, val_frac, test_frac, seed):
    """Losowy podzial UNIKALNYCH tabliczek na train/val/test."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = random.Random(seed)
    tablets = sorted(set(tablets))
    rng.shuffle(tablets)

    n = len(tablets)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_tablets = set(tablets[:n_train])
    val_tablets = set(tablets[n_train:n_train + n_val])
    test_tablets = set(tablets[n_train + n_val:])
    return train_tablets, val_tablets, test_tablets


def main():
    print(f"Wczytuje {TRANSLIT_CSV} ...")
    rows = parse_translit_csv(TRANSLIT_CSV)
    print(f"  -> {len(rows)} wierszy wczytanych\n")

    reading_to_charname = build_reading_to_charname_map(rows)

    # --- Skanuj folder char/ i parsuj nazwy plikow BEZPOSREDNIO Z DYSKU ---
    print(f"\nSkanuje {CHAR_IMAGES_DIR} ...")
    if not CHAR_IMAGES_DIR.exists():
        print(f"BLAD: folder nie istnieje: {CHAR_IMAGES_DIR}")
        return

    disk_files = list(CHAR_IMAGES_DIR.iterdir())
    print(f"  -> {len(disk_files)} plikow na dysku")

    parsed = []  # lista (filepath, reading, tablet, charname)
    unparsed = 0
    unmapped = 0
    for fpath in disk_files:
        reading, tablet = parse_disk_filename(fpath.name)
        if reading is None:
            unparsed += 1
            continue
        charname = reading_to_charname.get(reading)
        if charname is None:
            unmapped += 1
            continue
        parsed.append((fpath, reading, tablet, charname))

    print(f"  Sparsowano poprawnie: {len(parsed)}")
    print(f"  Nie dalo sie sparsowac nazwy: {unparsed}")
    print(f"  Sparsowano, ale reading nie znaleziony w CSV: {unmapped}")

    # --- Wybor top-N klas na podstawie tego co REALNIE jest na dysku ---
    class_counts = Counter(charname for _, _, _, charname in parsed)
    top_classes = set(c for c, _ in class_counts.most_common(TOP_N_CLASSES))
    min_count = class_counts.most_common(TOP_N_CLASSES)[-1][1]
    print(f"\nTop {TOP_N_CLASSES} klas (wg realnych plikow na dysku), "
          f"min. liczba przykladow: {min_count}")

    filtered = [(fpath, reading, tablet, charname)
                for fpath, reading, tablet, charname in parsed
                if charname in top_classes]
    print(f"Po filtrowaniu do top-{TOP_N_CLASSES}: {len(filtered)} plikow")

    # --- Group split po tabliczce ---
    all_tablets = [tablet for _, _, tablet, _ in filtered]
    train_tablets, val_tablets, test_tablets = group_split(
        all_tablets, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED
    )
    print(f"\nPodzial tabliczek: train={len(train_tablets)}, "
          f"val={len(val_tablets)}, test={len(test_tablets)}")

    def get_split(tablet):
        if tablet in train_tablets:
            return "train"
        elif tablet in val_tablets:
            return "val"
        elif tablet in test_tablets:
            return "test"
        return None

    # --- Kopiowanie ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (OUTPUT_DIR / split).mkdir(exist_ok=True)

    print("\nKopiowanie plikow...")
    split_counts = Counter()
    manifest_rows = []
    for fpath, reading, tablet, charname in filtered:
        split = get_split(tablet)
        if split is None:
            continue
        dest_dir = OUTPUT_DIR / split / charname
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / fpath.name
        shutil.copy2(fpath, dst)

        split_counts[split] += 1
        manifest_rows.append({
            "filename": fpath.name,
            "charname": charname,
            "reading": reading,
            "tablet": tablet,
            "split": split,
        })

    print(f"Skopiowano: {dict(split_counts)}")

    if not manifest_rows:
        print("\nBLAD: Nic nie zostalo skopiowane.")
        return

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest zapisany: {manifest_path}")

    # --- Sanity check: brak wycieku tabliczek miedzy splitami ---
    tablets_per_split = defaultdict(set)
    for r in manifest_rows:
        tablets_per_split[r["split"]].add(r["tablet"])
    overlap_tv = tablets_per_split["train"] & tablets_per_split["val"]
    overlap_tt = tablets_per_split["train"] & tablets_per_split["test"]
    overlap_vt = tablets_per_split["val"] & tablets_per_split["test"]
    assert not overlap_tv, f"WYCIEK train/val: {overlap_tv}"
    assert not overlap_tt, f"WYCIEK train/test: {overlap_tt}"
    assert not overlap_vt, f"WYCIEK val/test: {overlap_vt}"
    print("Sanity check OK: brak wspolnych tabliczek miedzy splitami.")


if __name__ == "__main__":
    main()