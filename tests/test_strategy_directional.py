"""Tests for abc_trading.strategy.directional (binary_kelly, directional_target, effective_target).

Default fee curve: fee/share(p) = p * 0.25 * (p * (1 - p))**2, e.g. fee(0.5) = 0.5*0.25*0.0625 =
1/128 = 0.0078125. Defaults: min_edge 0.03, kelly_fraction 0.25, max_directional_shares 200,
max_inventory_per_side_shares 1500.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import pytest

from abc_trading.config import BotConfig, DirectionalConfig, FeeConfig, PairConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.directional import binary_kelly, directional_target, effective_target
from abc_trading.types import (
    BookSnapshot,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    Outcome,
    Side,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)
CFG = BotConfig()
FEES = FeeModel(CFG.fees)


def fv(p_up: float, *, valid: bool = True) -> FairValue:
    return FairValue(p_up, p_up, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, valid)


def book(token: Outcome, bid: float | None, ask: float | None) -> BookSnapshot:
    return BookSnapshot(
        token,
        () if bid is None else (Level(bid, 400.0),),
        () if ask is None else (Level(ask, 400.0),),
    )


def snap(
    up: tuple[float | None, float | None] = (0.49, 0.50),
    down: tuple[float | None, float | None] = (0.50, 0.51),
) -> MarketSnapshot:
    return MarketSnapshot(
        1100.0, MKT, book(UP, *up), book(DOWN, *down), spot=100.0, spot_ts=1100.0, ref_price=100.0
    )


def inv_with(up: float = 0.0, down: float = 0.0) -> MarketInventory:
    inv = MarketInventory("m1")
    for token, qty in ((UP, up), (DOWN, down)):
        if qty > 0:
            inv.apply_fill(Fill("f", "o", "m1", token, Side.BUY, 0.5, qty, 0.0, True, 1.0))
    return inv


def target(
    s: MarketSnapshot,
    fair: FairValue,
    *,
    cfg: BotConfig = CFG,
    inv: MarketInventory | None = None,
    equity: float = 1000.0,
) -> float:
    return directional_target(
        snap=s,
        fair=fair,
        inv=inv or inv_with(),
        cfg=cfg,
        fees=FeeModel(cfg.fees),
        equity=equity,
    )


def no_fee_cfg(min_edge: float) -> BotConfig:
    return replace(
        CFG,
        fees=FeeConfig(taker_fee_rate=0.0),
        directional=replace(CFG.directional, min_edge=min_edge),
    )


# --------------------------------------------------------------------------- binary_kelly


@pytest.mark.parametrize(
    ("p", "price", "expected"),
    [
        (0.6, 0.5, 0.2),  # (0.6 - 0.5) / (1 - 0.5)
        (0.55, 0.45, 0.10 / 0.55),  # 0.181818...
        (0.7, 0.0, 0.7),  # free shares: f* = p
        (1.0, 0.5, 1.0),  # certain win
        (0.5, 0.5, 0.0),  # no edge
        (0.4, 0.5, 0.0),  # negative edge is floored at 0
        (0.0, 0.3, 0.0),
        (0.9, 1.0, 0.0),  # price 1 offers no upside
        (0.3, 0.1, 0.2 / 0.9),
    ],
)
def test_binary_kelly_values(p: float, price: float, expected: float) -> None:
    assert binary_kelly(p, price) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("p", "price"),
    [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.5), (math.nan, 0.5), (0.5, math.inf)],
)
def test_binary_kelly_rejects_bad_inputs(p: float, price: float) -> None:
    with pytest.raises(ValueError):
        binary_kelly(p, price)


# --------------------------------------------------------------------------- effective_target


def test_effective_target_clamps_and_gates() -> None:
    assert effective_target(fair=fv(0.6), cfg=CFG, target_net=120.0) == 120.0
    assert effective_target(fair=fv(0.6), cfg=CFG, target_net=999.0) == 200.0
    assert effective_target(fair=fv(0.6), cfg=CFG, target_net=-999.0) == -200.0
    assert effective_target(fair=fv(0.6, valid=False), cfg=CFG, target_net=120.0) == 0.0
    off = replace(CFG, directional=DirectionalConfig(enabled=False))
    assert effective_target(fair=fv(0.6), cfg=off, target_net=120.0) == 0.0
    with pytest.raises(ValueError, match="finite"):
        effective_target(fair=fv(0.6), cfg=CFG, target_net=math.nan)


# --------------------------------------------------------------------------- directional_target


def test_kelly_target_hand_computed_up() -> None:
    # UP ask 0.50: fee = 0.5*0.25*(0.25)^2 = 1/128, all-in c = 65/128 = 0.5078125.
    # p = 0.60: edge = 0.0921875 = 59/640 >= 0.03; f* = (59/640) / (1 - 65/128) = 59/315.
    # shares = 0.25 * f* * equity / c = (1/4)(59/315)(1000)(128/65) = 75520/819 = 92.2100122...
    got = target(snap(), fv(0.60), equity=1000.0)
    assert got == pytest.approx(float(Fraction(75520, 819)), abs=1e-9)
    assert got == pytest.approx(92.2100122, abs=1e-6)


def test_target_is_clipped_to_max_directional_shares() -> None:
    # equity 10000 -> uncapped 922.1 shares; max_directional_shares = 200 binds.
    assert target(snap(), fv(0.60), equity=10_000.0) == 200.0
    small = replace(CFG, directional=replace(CFG.directional, max_directional_shares=50.0))
    assert target(snap(), fv(0.60), cfg=small, equity=10_000.0) == 50.0
    assert target(snap(), fv(0.60), cfg=small, equity=100.0) == pytest.approx(9.2210012, abs=1e-6)


def test_kelly_target_hand_computed_down_is_negative() -> None:
    # p_up = 0.40 -> p_down = 0.60. DOWN ask 0.51: 0.51*0.49 = 0.2499, squared 0.06245001,
    # fee = 0.51 * 0.25 * 0.06245001 = 0.00796237..., c = 0.51796237...
    # edge = 0.6 - c = 0.0820376; f* = edge / (1 - c) = 0.17018925; shares = 0.25 f* 1000 / c.
    c = 0.51 + 0.51 * 0.25 * (0.51 * 0.49) ** 2
    expected = 0.25 * ((0.6 - c) / (1 - c)) * 1000.0 / c
    assert expected == pytest.approx(82.1436373, abs=1e-6)
    got = target(snap(), fv(0.40), equity=1000.0)
    assert got == pytest.approx(-expected, abs=1e-9)


def test_edge_threshold_boundary_without_fees() -> None:
    cfg = no_fee_cfg(min_edge=0.03)
    # UP ask 0.50, p = 0.53: edge = 0.03 == min_edge -> qualifies. f* = 0.03/0.5 = 0.06,
    # shares = 0.25 * 0.06 * 1000 / 0.5 = 30.
    assert target(snap(), fv(0.53), cfg=cfg) == pytest.approx(30.0, abs=1e-9)
    # edge = 0.0299 < min_edge -> no target.
    assert target(snap(), fv(0.5299), cfg=cfg) == 0.0


def test_default_fee_makes_the_same_probability_not_enough() -> None:
    # With the default fee c = 0.5078125, p = 0.53 gives edge 0.0221875 < 0.03.
    assert target(snap(), fv(0.53)) == 0.0


def test_zero_min_edge_still_needs_a_positive_edge() -> None:
    cfg = no_fee_cfg(min_edge=0.0)
    assert target(snap(), fv(0.50), cfg=cfg) == 0.0  # edge exactly 0 -> Kelly 0
    assert target(snap(), fv(0.51), cfg=cfg) > 0.0


def test_zero_when_disabled_invalid_or_no_equity() -> None:
    off = replace(CFG, directional=replace(CFG.directional, enabled=False))
    assert target(snap(), fv(0.9), cfg=off) == 0.0
    assert target(snap(), fv(0.9, valid=False)) == 0.0
    assert target(snap(), fv(0.9), equity=0.0) == 0.0
    assert target(snap(), fv(0.9), equity=-50.0) == 0.0
    zero_kelly = replace(CFG, directional=replace(CFG.directional, kelly_fraction=0.0))
    assert target(snap(), fv(0.9), cfg=zero_kelly) == 0.0
    zero_cap = replace(CFG, directional=replace(CFG.directional, max_directional_shares=0.0))
    assert target(snap(), fv(0.9), cfg=zero_cap) == 0.0
    with pytest.raises(ValueError, match="finite"):
        target(snap(), fv(0.9), equity=math.nan)


def test_missing_or_degenerate_asks_are_skipped() -> None:
    # No UP ask: UP cannot be targeted, and DOWN has no edge at p_up = 0.9.
    assert target(snap(up=(0.49, None)), fv(0.90)) == 0.0
    # No UP ask but DOWN is the favoured side: still works through the DOWN book.
    assert target(snap(up=(0.49, None)), fv(0.10)) < 0.0
    # An ask of 1.0 (or 0.0) offers nothing / is unusable.
    assert target(snap(up=(0.98, 1.0)), fv(0.99)) == 0.0
    assert target(snap(up=(0.0, 0.0)), fv(0.99)) == 0.0


def test_larger_edge_wins_and_ties_go_to_up() -> None:
    cfg = no_fee_cfg(min_edge=0.03)
    # p_up = 0.5, both asks 0.40: both edges 0.10 -> tie -> UP.
    tie = snap(up=(0.39, 0.40), down=(0.39, 0.40))
    assert target(tie, fv(0.50), cfg=cfg) > 0.0
    # DOWN ask 0.38 -> DOWN edge 0.12 > UP edge 0.10 -> DOWN, negative.
    # shares = 0.25 * (0.12/0.62) * 1000 / 0.38 = 127.3...
    down_better = snap(up=(0.39, 0.40), down=(0.37, 0.38))
    got = target(down_better, fv(0.50), cfg=cfg)
    assert got == pytest.approx(-0.25 * (0.12 / 0.62) * 1000.0 / 0.38, abs=1e-9)


def test_target_respects_the_per_side_inventory_cap() -> None:
    # Holding 1450 DOWN: a +target would need UP = DOWN + target <= 1500 -> room 50.
    held = inv_with(down=1450.0)
    assert target(snap(), fv(0.60), inv=held, equity=10_000.0) == 50.0
    # Holding 1500 DOWN: no room at all.
    assert target(snap(), fv(0.60), inv=inv_with(down=1500.0), equity=10_000.0) == 0.0
    # Mirror image for a DOWN target.
    assert target(snap(), fv(0.40), inv=inv_with(up=1450.0), equity=10_000.0) == -50.0
    # A custom side cap.
    cfg = replace(CFG, pair=PairConfig(max_inventory_per_side_shares=600.0))
    assert target(snap(), fv(0.60), cfg=cfg, inv=inv_with(down=550.0), equity=10_000.0) == 50.0


def test_target_never_exceeds_caps_on_a_grid() -> None:
    for p in (0.35, 0.5, 0.55, 0.6, 0.75, 0.9, 0.99):
        for equity in (10.0, 100.0, 1000.0, 50_000.0):
            t = target(snap(), fv(p), equity=equity)
            assert abs(t) <= CFG.directional.max_directional_shares + 1e-12
            if abs(t) > 0:
                assert (t > 0) == (p > 0.5)


def test_target_grows_with_equity_and_probability() -> None:
    assert target(snap(), fv(0.62), equity=500.0) < target(snap(), fv(0.62), equity=1000.0)
    assert target(snap(), fv(0.60), equity=1000.0) < target(snap(), fv(0.65), equity=1000.0)
