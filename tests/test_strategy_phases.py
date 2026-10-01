"""Tests for abc_trading.strategy.phases.phase_for (DESIGN 2.6).

Market used throughout: start 1000, end 1900 (a 900 s window) with the default TimingConfig
(warmup 15, wind_down 90, flatten 25), so the phase boundaries fall at
    ts < 1015                WARMUP        (since < 15)
    1015 <= ts < 1810        ACCUMULATE    (left > 90)
    1810 <= ts < 1875        WIND_DOWN     (90 >= left > 25)
    1875 <= ts < 1900        FLATTEN       (left <= 25)
    ts >= 1900               DONE
All timestamps are exactly representable floats so the boundary comparisons are exact.
"""

from __future__ import annotations

import math
import random

import pytest

from abc_trading.config import TimingConfig
from abc_trading.strategy.phases import phase_for
from abc_trading.types import MarketSpec, Phase

MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)
CFG = TimingConfig()  # warmup 15, wind_down 90, flatten 25

W, A, WD, F, D = Phase.WARMUP, Phase.ACCUMULATE, Phase.WIND_DOWN, Phase.FLATTEN, Phase.DONE


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        (990.0, W),  # before the window opens
        (999.5, W),
        (1000.0, W),  # since = 0
        (1014.5, W),
        (1014.999, W),  # since = 14.999 < 15
        (1015.0, A),  # since == warmup_seconds -> ACCUMULATE
        (1015.001, A),
        (1500.0, A),
        (1809.5, A),  # left = 90.5 > 90
        (1809.999, A),
        (1810.0, WD),  # left == wind_down_seconds -> WIND_DOWN
        (1810.001, WD),
        (1850.0, WD),
        (1874.5, WD),  # left = 25.5 > 25
        (1874.999, WD),
        (1875.0, F),  # left == flatten_seconds -> FLATTEN
        (1875.001, F),
        (1890.0, F),
        (1899.999, F),
        (1900.0, D),  # ts >= end -> DONE
        (1900.001, D),
        (5000.0, D),
    ],
)
def test_default_boundaries(ts: float, expected: Phase) -> None:
    assert phase_for(ts, MKT, CFG) is expected


def test_custom_thresholds() -> None:
    cfg = TimingConfig(warmup_seconds=60.0, wind_down_seconds=200.0, flatten_seconds=50.0)
    # warmup ends at 1060; wind-down starts at left <= 200 (ts 1700); flatten at left <= 50 (1850).
    assert phase_for(1059.5, MKT, cfg) is W
    assert phase_for(1060.0, MKT, cfg) is A
    assert phase_for(1699.5, MKT, cfg) is A
    assert phase_for(1700.0, MKT, cfg) is WD
    assert phase_for(1849.5, MKT, cfg) is WD
    assert phase_for(1850.0, MKT, cfg) is F
    assert phase_for(1900.0, MKT, cfg) is D


def test_zero_warmup_starts_accumulating_at_the_open() -> None:
    cfg = TimingConfig(warmup_seconds=0.0)
    assert phase_for(1000.0, MKT, cfg) is A  # since = 0 is not < 0
    assert phase_for(999.5, MKT, cfg) is W  # still before the window


def test_zero_flatten_never_flattens() -> None:
    cfg = TimingConfig(flatten_seconds=0.0)
    # left <= 0 only happens at ts >= end, which is DONE.
    assert phase_for(1899.999, MKT, cfg) is WD
    assert phase_for(1900.0, MKT, cfg) is D


def test_flatten_equal_to_wind_down_has_no_wind_down_phase() -> None:
    cfg = TimingConfig(wind_down_seconds=60.0, flatten_seconds=60.0)
    assert phase_for(1839.5, MKT, cfg) is A  # left = 60.5
    assert phase_for(1840.0, MKT, cfg) is F  # left = 60
    seen = {phase_for(1000.0 + i * 0.5, MKT, cfg) for i in range(0, 1800)}
    assert WD not in seen


def test_done_wins_over_warmup_for_a_window_shorter_than_the_warmup() -> None:
    short = MarketSpec("s", "BTC", start_ts=1000.0, end_ts=1010.0)  # 10 s window, 15 s warmup
    assert phase_for(1005.0, short, CFG) is W
    assert phase_for(1009.999, short, CFG) is W
    assert phase_for(1010.0, short, CFG) is D
    assert phase_for(1020.0, short, CFG) is D


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_ts_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        phase_for(bad, MKT, CFG)


def test_phase_sequence_is_monotone_and_complete_on_a_second_grid() -> None:
    order = [W, A, WD, F, D]
    phases = [phase_for(1000.0 + s, MKT, CFG) for s in range(-20, 950)]
    ranks = [order.index(p) for p in phases]
    assert ranks == sorted(ranks)
    assert set(phases) == set(order)
    # The grid covers ts 980..1949 (970 points). Dwell times: 20 before the open + 15 warmup,
    # 795 ACCUMULATE, 65 WIND_DOWN, 25 FLATTEN, 50 DONE.
    assert phases.count(W) == 20 + 15
    assert phases.count(A) == 795  # ts 1015..1809
    assert phases.count(WD) == 65  # ts 1810..1874
    assert phases.count(F) == 25  # ts 1875..1899
    assert phases.count(D) == 50  # ts 1900..1949
    assert len(phases) == 35 + 795 + 65 + 25 + 50


def test_random_configs_stay_ordered() -> None:
    rng = random.Random(7)
    order = [W, A, WD, F, D]
    for _ in range(300):
        wd = rng.choice([0.0, 30.0, 60.0, 120.0, 400.0])
        fl = rng.choice([0.0, 10.0, 25.0, 60.0]) if wd >= 60.0 else 0.0
        cfg = TimingConfig(
            warmup_seconds=rng.choice([0.0, 5.0, 15.0, 120.0]),
            wind_down_seconds=wd,
            flatten_seconds=min(fl, wd),
        )
        phases = [phase_for(1000.0 + 0.5 * i, MKT, cfg) for i in range(-10, 1850)]
        ranks = [order.index(p) for p in phases]
        assert ranks == sorted(ranks), cfg
        assert phases[-1] is D
        assert phase_for(MKT.end_ts, MKT, cfg) is D
