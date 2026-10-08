"""The code that builds the train/val/test split: grouped by tablet, deterministic, no overlap."""

import pytest

from build_dataset import group_split, parse_disk_filename

TABLETS = [f"HS_{n}" for n in range(1, 201)]


def test_no_tablet_in_two_splits():
    train, val, test = group_split(TABLETS, 0.70, 0.15, 0.15, seed=42)
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)


def test_every_tablet_assigned_exactly_once():
    train, val, test = group_split(TABLETS, 0.70, 0.15, 0.15, seed=42)
    assert train | val | test == set(TABLETS)
    assert len(train) + len(val) + len(test) == len(TABLETS)


def test_proportions_are_close_to_requested():
    train, val, test = group_split(TABLETS, 0.70, 0.15, 0.15, seed=42)
    assert abs(len(train) / len(TABLETS) - 0.70) < 0.02
    assert abs(len(val) / len(TABLETS) - 0.15) < 0.02
    assert abs(len(test) / len(TABLETS) - 0.15) < 0.03


def test_same_seed_same_split_different_seed_different_split():
    a = group_split(TABLETS, 0.70, 0.15, 0.15, seed=1)
    b = group_split(TABLETS, 0.70, 0.15, 0.15, seed=1)
    c = group_split(TABLETS, 0.70, 0.15, 0.15, seed=2)
    assert a == b
    assert a != c


def test_duplicates_in_input_do_not_create_duplicates_in_output():
    train, val, test = group_split(TABLETS + TABLETS, 0.70, 0.15, 0.15, seed=42)
    assert len(train) + len(val) + len(test) == len(TABLETS)


def test_fractions_must_sum_to_one():
    with pytest.raises(AssertionError):
        group_split(TABLETS, 0.5, 0.2, 0.2, seed=42)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a_1_10_3_HS_785_03_front.png", ("a", "HS_785")),
        ("1(asz)_1_1_1_HS_1194_03_front.png", ("1(asz)", "HS_1194")),
        ("lugal_1_4_2_HS_1098_06_back.png", ("lugal", "HS_1098")),
        # readings may contain underscores themselves
        ("sze3_(esz2_gi7_zi3)_1_2_3_HS_77_03_front.png", ("sze3_(esz2_gi7_zi3)", "HS_77")),
    ],
)
def test_parse_disk_filename(filename, expected):
    assert parse_disk_filename(filename) == expected


def test_parse_disk_filename_rejects_malformed_names():
    assert parse_disk_filename("garbage.png") == (None, None)
    assert parse_disk_filename("a_b_c.png") == (None, None)
