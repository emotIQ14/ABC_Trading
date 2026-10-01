"""Tests for abc_trading.backtest.metrics (pure statistics helpers).

Every expected number is worked out by hand in a comment.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from abc_trading.backtest import metrics
from abc_trading.backtest.metrics import Drawdown
from abc_trading.types import Outcome

UP, DOWN = Outcome.UP, Outcome.DOWN


# --------------------------------------------------------------------------- mean / median / std


def test_mean_exact() -> None:
    # (1 + 2 + 3 + 4) / 4 = 2.5
    assert metrics.mean([1.0, 2.0, 3.0, 4.0]) == 2.5
    # naive summation depends on order (0.1+0.2+0.3 = 0.6000000000000001, 0.3+0.2+0.1 = 0.6);
    # the fsum-based mean must not
    assert metrics.mean([0.1, 0.2, 0.3]) == metrics.mean([0.3, 0.2, 0.1])


def test_median_odd_even_unsorted() -> None:
    assert metrics.median([5.0, 1.0, 3.0]) == 3.0  # sorted 1, 3, 5
    assert metrics.median([4.0, 1.0, 3.0, 2.0]) == 2.5  # sorted 1, 2, 3, 4 -> (2 + 3) / 2
    assert metrics.median([7.0]) == 7.0


def test_std_known_data_set() -> None:
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # mean 5; squared deviations 9, 1, 1, 1, 0, 0, 4, 16 sum to 32
    assert metrics.std(data, ddof=0) == pytest.approx(2.0)  # sqrt(32 / 8) = sqrt(4)
    assert metrics.std(data) == pytest.approx(math.sqrt(32.0 / 7.0))  # 2.1380899...


def test_std_constant_series_is_zero() -> None:
    assert metrics.std([3.0, 3.0, 3.0]) == 0.0


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (metrics.mean, ([],)),
        (metrics.median, ([],)),
        (metrics.std, ([1.0],)),  # sample std needs at least 2 values
        (metrics.max_drawdown, ([],)),
    ],
)
def test_empty_or_too_short_input_raises(func: object, args: tuple[list[float]]) -> None:
    with pytest.raises(ValueError):
        func(*args)  # type: ignore[operator]


def test_std_rejects_negative_ddof_and_ddof_ge_n() -> None:
    with pytest.raises(ValueError, match="ddof"):
        metrics.std([1.0, 2.0], ddof=-1)
    with pytest.raises(ValueError, match="more than ddof"):
        metrics.std([1.0, 2.0], ddof=2)
    assert metrics.std([4.0], ddof=0) == 0.0  # one value, population std


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_raise_everywhere(bad: float) -> None:
    for func in (metrics.mean, metrics.median, metrics.std, metrics.max_drawdown):
        with pytest.raises(ValueError, match="finite"):
            func([1.0, bad, 2.0])
    with pytest.raises(ValueError, match="finite"):
        metrics.sharpe_like([1.0, bad])
    with pytest.raises(ValueError, match="finite"):
        metrics.sharpe_like([bad])  # the n < 2 early exit validates too


# --------------------------------------------------------------------------- sharpe-like


def test_sharpe_like_hand_computed() -> None:
    # [1, 3]: mean 2, sample std sqrt(((1-2)^2 + (3-2)^2) / 1) = sqrt(2), n = 2
    # sharpe = 2 / sqrt(2) * sqrt(2) = 2
    assert metrics.sharpe_like([1.0, 3.0]) == pytest.approx(2.0)
    assert metrics.sharpe_like([-1.0, -3.0]) == pytest.approx(-2.0)
    # [10, -4, 2]: mean 8/3, sum of squared deviations 296/3, var = 148/3
    # sharpe = (8/3) / sqrt(148/3) * sqrt(3) = 8 / sqrt(148) = 4 / sqrt(37)
    assert metrics.sharpe_like([10.0, -4.0, 2.0]) == pytest.approx(4.0 / math.sqrt(37.0))


@pytest.mark.parametrize("values", [[], [3.0], [5.0, 5.0, 5.0], [1e-14, 2e-14]])
def test_sharpe_like_is_zero_when_undefined(values: list[float]) -> None:
    # fewer than 2 values, or no dispersion (std <= 1e-12)
    assert metrics.sharpe_like(values) == 0.0


def test_sharpe_like_scale_invariant_and_sign_symmetric_random() -> None:
    rng = random.Random(7)
    for _ in range(30):
        values = [rng.gauss(0.3, 2.0) for _ in range(rng.randint(2, 40))]
        base = metrics.sharpe_like(values)
        assert metrics.sharpe_like([v * 17.5 for v in values]) == pytest.approx(base)
        assert metrics.sharpe_like([-v for v in values]) == pytest.approx(-base)


def test_sharpe_like_matches_statistics_module_random() -> None:
    rng = random.Random(11)
    for _ in range(30):
        n = rng.randint(2, 50)
        values = [rng.uniform(-5, 5) for _ in range(n)]
        expected = statistics.fmean(values) / statistics.stdev(values) * math.sqrt(n)
        assert metrics.sharpe_like(values) == pytest.approx(expected)


# --------------------------------------------------------------------------- ratio


def test_ratio() -> None:
    assert metrics.ratio(1.0, 4.0) == 0.25
    assert metrics.ratio(-3.0, 2.0) == -1.5
    assert metrics.ratio(0.0, 5.0) == 0.0
    assert metrics.ratio(1.0, 0.0) is None
    with pytest.raises(ValueError, match="finite"):
        metrics.ratio(math.nan, 1.0)
    with pytest.raises(ValueError, match="finite"):
        metrics.ratio(1.0, math.inf)


# --------------------------------------------------------------------------- max drawdown


def test_max_drawdown_hand_computed() -> None:
    # 100, 110 (peak), 90, 95, 80 (trough), 120: deepest fall is 110 -> 80 = 30 = 27.27% of 110
    dd = metrics.max_drawdown([100.0, 110.0, 90.0, 95.0, 80.0, 120.0])
    assert dd == Drawdown(depth=30.0, fraction=30.0 / 110.0, peak_index=1, trough_index=4)


def test_max_drawdown_monotone_and_single_point_are_zero() -> None:
    assert metrics.max_drawdown([1.0, 2.0, 3.0]) == Drawdown(0.0, 0.0, 0, 0)
    assert metrics.max_drawdown([5.0]) == Drawdown(0.0, 0.0, 0, 0)
    assert metrics.max_drawdown([4.0, 4.0, 4.0]) == Drawdown(0.0, 0.0, 0, 0)


def test_max_drawdown_first_deepest_episode_wins_ties() -> None:
    # 10 -> 8 (depth 2) then 10 -> 8 again (depth 2): the first episode is reported
    assert metrics.max_drawdown([10.0, 8.0, 10.0, 8.0]) == Drawdown(2.0, 0.2, 0, 1)


def test_max_drawdown_measures_from_running_peak_not_global_peak() -> None:
    # global peak 100 comes AFTER the trough 50, so it cannot create a drawdown: 60 -> 50 = 10
    dd = metrics.max_drawdown([60.0, 50.0, 100.0])
    assert dd == Drawdown(10.0, 10.0 / 60.0, 0, 1)


def test_max_drawdown_non_positive_peak_has_zero_fraction() -> None:
    assert metrics.max_drawdown([-5.0, -10.0]) == Drawdown(5.0, 0.0, 0, 1)


def test_max_drawdown_matches_brute_force_on_random_walks() -> None:
    rng = random.Random(3)
    for _ in range(40):
        level, series = 10_000.0, []
        for _ in range(rng.randint(1, 80)):
            level += rng.gauss(0.0, 25.0)
            series.append(level)
        brute = max(
            (series[i] - series[j] for j in range(len(series)) for i in range(j + 1)),
            default=0.0,
        )
        dd = metrics.max_drawdown(series)
        assert dd.depth == pytest.approx(max(brute, 0.0), abs=1e-9)
        assert dd.depth >= 0.0
        assert dd.peak_index <= dd.trough_index
        assert series[dd.peak_index] - series[dd.trough_index] == pytest.approx(dd.depth, abs=1e-9)


# --------------------------------------------------------------------------- brier


def test_brier_up_hand_computed() -> None:
    # forecast 0.8 and UP won (y=1): (0.8-1)^2 = 0.04; forecast 0.3 and DOWN won (y=0): 0.09
    # mean = (0.04 + 0.09) / 2 = 0.065
    assert metrics.brier_up([0.8, 0.3], [UP, DOWN]) == pytest.approx(0.065)
    assert metrics.brier_up([0.5, 0.5], [UP, DOWN]) == pytest.approx(0.25)  # coin flip
    assert metrics.brier_up([1.0, 0.0], [UP, DOWN]) == 0.0  # perfect
    assert metrics.brier_up([0.0], [UP]) == 1.0  # confidently wrong


def test_brier_up_validates_input() -> None:
    with pytest.raises(ValueError, match="forecasts but"):
        metrics.brier_up([0.5], [UP, DOWN])
    with pytest.raises(ValueError):
        metrics.brier_up([], [])
    with pytest.raises(ValueError):
        metrics.brier_up([1.5], [UP])


def test_brier_skill() -> None:
    assert metrics.brier_skill(0.1, 0.25) == pytest.approx(0.6)  # 1 - 0.1 / 0.25
    assert metrics.brier_skill(0.25, 0.25) == pytest.approx(0.0)
    assert metrics.brier_skill(0.3, 0.25) == pytest.approx(-0.2)  # worse than the reference
    assert metrics.brier_skill(0.1, 0.0) is None  # no reference skill defined
    with pytest.raises(ValueError, match="finite"):
        metrics.brier_skill(math.nan, 0.25)
