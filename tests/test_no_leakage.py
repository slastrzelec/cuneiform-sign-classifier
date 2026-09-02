"""
tests/test_no_leakage.py

Testy jednostkowe integralnosci datasetu zbudowanego przez build_dataset.py.
Najwazniejszy test: zadna tabliczka (tablet) nie moze wystapic w wiecej
niz jednym splicie (train/val/test) - to gwarancja braku wycieku danych.

Uruchomienie (z katalogu glownego projektu):
    pytest tests/test_no_leakage.py -v
"""

import csv
from pathlib import Path
from collections import defaultdict

import pytest

# Zakladana struktura: <project_root>/tests/test_no_leakage.py
#                       <project_root>/data/processed/manifest.csv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"


@pytest.fixture(scope="module")
def manifest_rows():
    if not MANIFEST_PATH.exists():
        pytest.skip(f"manifest.csv nie znaleziony pod {MANIFEST_PATH} - "
                     f"uruchom najpierw build_dataset.py")
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def tablets_per_split(manifest_rows):
    result = defaultdict(set)
    for row in manifest_rows:
        result[row["split"]].add(row["tablet"])
    return result


def test_manifest_not_empty(manifest_rows):
    assert len(manifest_rows) > 0, "manifest.csv jest pusty"


def test_all_three_splits_present(tablets_per_split):
    for split in ("train", "val", "test"):
        assert split in tablets_per_split, f"Brak splitu '{split}' w manifescie"
        assert len(tablets_per_split[split]) > 0, f"Split '{split}' nie ma zadnej tabliczki"


def test_no_tablet_overlap_train_val(tablets_per_split):
    overlap = tablets_per_split["train"] & tablets_per_split["val"]
    assert not overlap, (
        f"WYCIEK DANYCH: {len(overlap)} tabliczek wystepuje jednoczesnie "
        f"w train i val: {sorted(overlap)[:10]}..."
    )


def test_no_tablet_overlap_train_test(tablets_per_split):
    overlap = tablets_per_split["train"] & tablets_per_split["test"]
    assert not overlap, (
        f"WYCIEK DANYCH: {len(overlap)} tabliczek wystepuje jednoczesnie "
        f"w train i test: {sorted(overlap)[:10]}..."
    )


def test_no_tablet_overlap_val_test(tablets_per_split):
    overlap = tablets_per_split["val"] & tablets_per_split["test"]
    assert not overlap, (
        f"WYCIEK DANYCH: {len(overlap)} tabliczek wystepuje jednoczesnie "
        f"w val i test: {sorted(overlap)[:10]}..."
    )


def test_no_duplicate_filenames(manifest_rows):
    filenames = [row["filename"] for row in manifest_rows]
    duplicates = {f for f in filenames if filenames.count(f) > 1}
    assert not duplicates, f"Zduplikowane nazwy plikow w manifescie: {duplicates}"


def test_every_class_present_in_train(manifest_rows):
    """Kazda klasa, ktora wystepuje w val lub test, musi tez wystepowac w train
    (inaczej model nigdy sie jej nie nauczy, a bedziemy go na niej ewaluowac)."""
    train_classes = {r["charname"] for r in manifest_rows if r["split"] == "train"}
    other_classes = {r["charname"] for r in manifest_rows if r["split"] in ("val", "test")}
    missing = other_classes - train_classes
    assert not missing, (
        f"Klasy obecne w val/test ale nieobecne w train: {missing} - "
        f"model nie mial szansy sie ich nauczyc"
    )


def test_reasonable_split_proportions(manifest_rows):
    """Sprawdza, ze proporcje train/val/test sa w rozsadnym zakresie
    (grupowy split po tabliczce moze dac lekkie odchylenie od 70/15/15,
    ale nie powinien byc drastycznie inny)."""
    total = len(manifest_rows)
    counts = defaultdict(int)
    for r in manifest_rows:
        counts[r["split"]] += 1

    train_frac = counts["train"] / total
    val_frac = counts["val"] / total
    test_frac = counts["test"] / total

    assert 0.55 <= train_frac <= 0.85, f"train_frac poza oczekiwanym zakresem: {train_frac:.2f}"
    assert 0.05 <= val_frac <= 0.30, f"val_frac poza oczekiwanym zakresem: {val_frac:.2f}"
    assert 0.05 <= test_frac <= 0.30, f"test_frac poza oczekiwanym zakresem: {test_frac:.2f}"
