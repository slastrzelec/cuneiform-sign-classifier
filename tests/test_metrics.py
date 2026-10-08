import pytest

from src.metrics import wilson_interval


def test_known_value_for_three_of_seven():
    low, high = wilson_interval(3, 7)
    assert low == pytest.approx(0.158, abs=0.002)
    assert high == pytest.approx(0.750, abs=0.002)


def test_interval_contains_the_point_estimate_and_stays_in_unit_range():
    for successes, n in [(0, 10), (10, 10), (45, 50), (1, 2000)]:
        low, high = wilson_interval(successes, n)
        assert 0.0 <= low <= successes / n <= high <= 1.0


def test_more_data_gives_a_narrower_interval():
    small = wilson_interval(9, 10)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


@pytest.mark.parametrize(("successes", "n"), [(1, 0), (-1, 5), (6, 5)])
def test_invalid_input_is_rejected(successes, n):
    with pytest.raises(ValueError):
        wilson_interval(successes, n)
