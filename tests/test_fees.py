"""Tests for abc_trading.fees (fee curve, maker fee/rebate, validation)."""

from __future__ import annotations

import math
import random

import pytest

from abc_trading.config import FeeConfig
from abc_trading.fees import FeeModel

DEFAULT = FeeConfig()  # taker rate 0.25, exponent 2, maker fee/rebate 0
REL = 1e-12


def approx(x: float) -> object:
    return pytest.approx(x, rel=REL, abs=1e-15)


@pytest.fixture
def model() -> FeeModel:
    return FeeModel(DEFAULT)


# --------------------------------------------------------------------------- worked numbers


def test_taker_fee_at_half(model: FeeModel) -> None:
    # 100 * 0.5 * 0.25 * (0.5 * 0.5) ** 2 = 12.5 * 0.0625 = 0.78125
    assert model.fee(0.5, 100.0, False) == approx(0.78125)


def test_taker_fee_at_47c(model: FeeModel) -> None:
    # p(1-p) = 0.47 * 0.53 = 0.2491; squared = 0.06205081
    # fee = 100 * 0.47 * 0.25 * 0.06205081 = 11.75 * 0.06205081 = 0.7290970175
    assert model.fee(0.47, 100.0, False) == approx(0.7290970175)


def test_taker_fee_at_52c(model: FeeModel) -> None:
    # p(1-p) = 0.52 * 0.48 = 0.2496; squared = 0.06230016
    # fee = 100 * 0.52 * 0.25 * 0.06230016 = 13 * 0.06230016 = 0.80990208
    assert model.fee(0.52, 100.0, False) == approx(0.80990208)


def test_taker_fee_per_share_matches_fee_of_one_share(model: FeeModel) -> None:
    # 1 * 0.5 * 0.25 * 0.0625 = 0.0078125
    assert model.taker_fee_per_share(0.5) == approx(0.0078125)
    for p in (0.0, 0.01, 0.3, 0.47, 0.52, 0.99, 1.0):
        assert model.taker_fee_per_share(p) == model.fee(p, 1.0, False)


def test_all_in_buy_cost_taker(model: FeeModel) -> None:
    # 0.5 + 0.0078125
    assert model.all_in_buy_cost(0.5, False) == approx(0.5078125)


def test_all_in_buy_cost_maker_zero_fee_is_the_price(model: FeeModel) -> None:
    for p in (0.0, 0.01, 0.47, 0.52, 1.0):
        assert model.all_in_buy_cost(p, True) == p


def test_exponent_one() -> None:
    m = FeeModel(FeeConfig(taker_fee_rate=0.1, taker_fee_exponent=1.0))
    # 10 * 0.4 * 0.1 * (0.4 * 0.6) ** 1 = 0.4 * 0.24 = 0.096
    assert m.fee(0.4, 10.0, False) == approx(0.096)


def test_exponent_fractional() -> None:
    m = FeeModel(FeeConfig(taker_fee_rate=0.25, taker_fee_exponent=0.5))
    # 100 * 0.5 * 0.25 * sqrt(0.25) = 12.5 * 0.5 = 6.25
    assert m.fee(0.5, 100.0, False) == approx(6.25)


def test_exponent_zero_is_flat_rate_on_notional() -> None:
    m = FeeModel(FeeConfig(taker_fee_rate=0.1, taker_fee_exponent=0.0))
    # fee = rate * notional: 0.1 * 0.4 * 10 = 0.4; at p=1: 0.1 * 1 * 10 = 1.0; at p=0: 0
    assert m.fee(0.4, 10.0, False) == approx(0.4)
    assert m.fee(1.0, 10.0, False) == approx(1.0)
    assert m.fee(0.0, 10.0, False) == 0.0


# --------------------------------------------------------------------------- maker


def test_maker_fee_zero_by_default(model: FeeModel) -> None:
    assert model.fee(0.5, 100.0, True) == 0.0


def test_maker_fee_net_of_rebate() -> None:
    m = FeeModel(FeeConfig(maker_fee_rate=0.02, maker_rebate_rate=0.005))
    # (0.02 - 0.005) * 0.5 * 100 = 0.75
    assert m.fee(0.5, 100.0, True) == approx(0.75)
    # taker leg unaffected by maker parameters: still 0.78125
    assert m.fee(0.5, 100.0, False) == approx(0.78125)


def test_maker_rebate_is_negative_fee_and_lowers_all_in_cost() -> None:
    m = FeeModel(FeeConfig(maker_rebate_rate=0.01))
    # -0.01 * 0.5 * 100 = -0.5 ; per share -0.005 -> all-in 0.495
    assert m.fee(0.5, 100.0, True) == approx(-0.5)
    assert m.all_in_buy_cost(0.5, True) == approx(0.495)


def test_maker_fee_linear_in_notional() -> None:
    m = FeeModel(FeeConfig(maker_fee_rate=0.03))
    # 0.03 * 0.25 * 40 = 0.3 and 0.03 * 0.75 * 40 = 0.9
    assert m.fee(0.25, 40.0, True) == approx(0.3)
    assert m.fee(0.75, 40.0, True) == approx(0.9)


# --------------------------------------------------------------------------- curve properties


def test_taker_fee_zero_at_zero_and_one(model: FeeModel) -> None:
    assert model.fee(0.0, 100.0, False) == 0.0
    assert model.fee(1.0, 100.0, False) == 0.0
    assert model.taker_fee_per_share(0.0) == 0.0
    assert model.taker_fee_per_share(1.0) == 0.0
    assert model.all_in_buy_cost(1.0, False) == 1.0


def test_taker_fee_never_negative_and_positive_inside() -> None:
    m = FeeModel(DEFAULT)
    for i in range(0, 1001):
        p = i / 1000
        f = m.fee(p, 10.0, False)
        assert f >= 0.0
        if 0 < i < 1000:
            assert f > 0.0


def test_effective_rate_on_notional_is_symmetric_and_max_at_half(model: FeeModel) -> None:
    # effective rate = fee / (price * size) = 0.25 * (p(1-p))**2 ; at 0.5 -> 0.25 * 0.0625
    peak = 0.25 * 0.0625
    best_p, best_rate = 0.0, -1.0
    for i in range(1, 1000):
        p = i / 1000
        rate = model.fee(p, 1.0, False) / p
        assert rate == pytest.approx(model.fee(1 - p, 1.0, False) / (1 - p), rel=1e-9)
        assert rate <= peak + 1e-15
        if rate > best_rate:
            best_p, best_rate = p, rate
    assert best_p == 0.5
    assert best_rate == approx(peak)


def test_usdc_fee_per_share_peaks_at_e_plus_one_over_two_e_plus_one() -> None:
    # The USDC fee per share is rate * p**(e+1) * (1-p)**e, whose maximum is at
    # p* = (e+1)/(2e+1): 0.6 for e=2 (d/dp[p^3 (1-p)^2] = p^2 (1-p) (3 - 5p) = 0 at p=0.6).
    m = FeeModel(DEFAULT)
    grid = [i / 1000 for i in range(1001)]
    best = max(grid, key=m.taker_fee_per_share)
    assert best == 0.6
    # value at the peak: 0.25 * 0.6**3 * 0.4**2 = 0.25 * 0.216 * 0.16 = 0.00864
    assert m.taker_fee_per_share(0.6) == approx(0.00864)
    # exponent 1 -> p* = 2/3 (nearest grid point 0.667)
    m1 = FeeModel(FeeConfig(taker_fee_exponent=1.0))
    assert max(grid, key=m1.taker_fee_per_share) == 0.667


def test_fee_linear_in_size(model: FeeModel) -> None:
    base = model.fee(0.47, 7.0, False)
    assert model.fee(0.47, 14.0, False) == 2 * base  # doubling is exact in binary floats
    assert model.fee(0.47, 21.0, False) == pytest.approx(3 * base, rel=1e-12)
    assert model.fee(0.47, 0.0, False) == 0.0
    assert model.fee(0.47, 0.0, True) == 0.0


def test_fee_additive_in_size_seeded() -> None:
    rng = random.Random(2024)
    for cfg in (
        DEFAULT,
        FeeConfig(maker_fee_rate=0.02, maker_rebate_rate=0.005),
        FeeConfig(taker_fee_rate=0.07, taker_fee_exponent=1.5),
    ):
        m = FeeModel(cfg)
        for _ in range(200):
            p = rng.random()
            a, b = rng.uniform(0, 500), rng.uniform(0, 500)
            for is_maker in (True, False):
                whole = m.fee(p, a + b, is_maker)
                parts = m.fee(p, a, is_maker) + m.fee(p, b, is_maker)
                assert whole == pytest.approx(parts, rel=1e-9, abs=1e-12)


def test_fee_is_deterministic_and_does_not_mutate_config() -> None:
    cfg = FeeConfig(maker_fee_rate=0.01)
    m = FeeModel(cfg)
    first = m.fee(0.33, 12.5, False)
    assert m.fee(0.33, 12.5, False) == first
    assert m.cfg is cfg
    assert cfg == FeeConfig(maker_fee_rate=0.01)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "price", [-0.01, -1e-9, 1.0000001, 1.01, 2.0, math.nan, math.inf, -math.inf]
)
def test_invalid_price_raises(model: FeeModel, price: float) -> None:
    for is_maker in (True, False):
        with pytest.raises(ValueError, match="price"):
            model.fee(price, 1.0, is_maker)
        with pytest.raises(ValueError, match="price"):
            model.all_in_buy_cost(price, is_maker)
    with pytest.raises(ValueError, match="price"):
        model.taker_fee_per_share(price)


@pytest.mark.parametrize("size", [-1.0, -1e-12, math.nan, math.inf])
def test_invalid_size_raises(model: FeeModel, size: float) -> None:
    for is_maker in (True, False):
        with pytest.raises(ValueError, match="size"):
            model.fee(0.5, size, is_maker)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maker_fee_rate": -0.01},
        {"maker_rebate_rate": -0.01},
        {"taker_fee_rate": -0.25},
        {"taker_fee_exponent": -1.0},
        {"taker_fee_rate": math.nan},
        {"maker_fee_rate": math.inf},
    ],
)
def test_invalid_config_raises(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="FeeConfig"):
        FeeModel(FeeConfig(**kwargs))


def test_boundary_prices_and_zero_size_are_valid(model: FeeModel) -> None:
    assert model.fee(0.0, 0.0, False) == 0.0
    assert model.fee(1.0, 0.0, True) == 0.0
    assert model.all_in_buy_cost(0.0, False) == 0.0
