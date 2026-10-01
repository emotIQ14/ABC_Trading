"""Tests for abc_trading.inventory (MarketInventory, Portfolio, PnLBreakdown).

Expected numbers are worked out by hand in the comments. Randomised tests are seeded and
compare the float implementation against an independent exact-arithmetic (``Fraction``)
reference model.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections import Counter
from fractions import Fraction

import pytest

from abc_trading.config import FeeConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory, PnLBreakdown, Portfolio
from abc_trading.types import (
    EPS,
    Fill,
    MergeResult,
    OpenOrder,
    Outcome,
    Settlement,
    Side,
    TimeInForce,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
M = "mkt-1"


def approx(x: float, abs_tol: float = 1e-9) -> object:
    return pytest.approx(x, rel=1e-12, abs=abs_tol)


def mkfill(
    token: Outcome,
    side: Side,
    price: float,
    size: float,
    fee: float = 0.0,
    *,
    market: str = M,
    maker: bool = True,
) -> Fill:
    return Fill(
        fill_id="f",
        order_id="o",
        market_id=market,
        token=token,
        side=side,
        price=price,
        size=size,
        fee=fee,
        is_maker=maker,
        ts=0.0,
    )


def buy(
    token: Outcome,
    price: float,
    size: float,
    fee: float = 0.0,
    *,
    market: str = M,
    maker: bool = True,
) -> Fill:
    return mkfill(token, Side.BUY, price, size, fee, market=market, maker=maker)


def sell(
    token: Outcome,
    price: float,
    size: float,
    fee: float = 0.0,
    *,
    market: str = M,
    maker: bool = True,
) -> Fill:
    return mkfill(token, Side.SELL, price, size, fee, market=market, maker=maker)


def pair_inv(
    up: tuple[float, float] = (0.47, 100.0), down: tuple[float, float] = (0.52, 100.0)
) -> MarketInventory:
    """UP@0.47 x100 + DOWN@0.52 x100, maker, zero fees (the DESIGN worked example)."""
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, *up))
    inv.apply_fill(buy(DOWN, *down))
    return inv


def state(inv: MarketInventory) -> tuple[object, ...]:
    return (
        dict(inv.qty),
        dict(inv.cost),
        inv.fees_paid,
        inv.sell_pnl,
        inv.merge_pnl,
        inv.settle_pnl,
        inv.settled,
    )


# =========================================================================== MarketInventory


def test_empty_inventory() -> None:
    inv = MarketInventory(M)
    assert inv.market_id == M
    assert inv.qty == {UP: 0.0, DOWN: 0.0}
    assert inv.cost == {UP: 0.0, DOWN: 0.0}
    assert (inv.fees_paid, inv.sell_pnl, inv.merge_pnl, inv.settled) == (0.0, 0.0, 0.0, False)
    assert inv.avg_cost(UP) is None and inv.avg_cost(DOWN) is None
    assert inv.paired_qty == 0.0 and inv.net_shares == 0.0
    assert inv.unpaired() == (None, 0.0)
    assert inv.locked_profit() == 0.0
    assert inv.capital_at_risk() == 0.0
    assert inv.value_at(0.3) == 0.0
    assert inv.realised_pnl == 0.0


def test_buys_accumulate_all_in_cost() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.47, 100.0))
    inv.apply_fill(buy(UP, 0.50, 50.0, fee=0.1))
    # qty = 150 ; cost = 0.47*100 + (0.50*50 + 0.1) = 47 + 25.1 = 72.1 ; avg = 72.1/150
    assert inv.qty[UP] == 150.0
    assert inv.cost[UP] == approx(72.1)
    assert inv.avg_cost(UP) == approx(72.1 / 150.0)
    assert inv.fees_paid == approx(0.1)
    assert inv.qty[DOWN] == 0.0 and inv.avg_cost(DOWN) is None
    assert inv.capital_at_risk() == approx(72.1)


def test_maker_rebate_lowers_cost_basis() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(DOWN, 0.5, 10.0, fee=-0.05))
    # cost = 5.0 - 0.05 = 4.95 ; fees_paid = -0.05
    assert inv.cost[DOWN] == approx(4.95)
    assert inv.fees_paid == approx(-0.05)
    assert inv.avg_cost(DOWN) == approx(0.495)


def test_worked_example_locks_one_dollar_and_merge_agrees() -> None:
    inv = pair_inv()
    # cost = 47 + 52 = 99 for 100 pairs worth 100 -> 1.00 locked
    assert inv.capital_at_risk() == approx(99.0)
    assert inv.paired_qty == 100.0
    assert inv.net_shares == 0.0
    assert inv.unpaired() == (None, 0.0)
    assert inv.locked_profit() == approx(1.0)  # 100 * (1 - 0.47 - 0.52)
    pnl = inv.apply_merge(100.0)  # 100 - (47 + 52) = 1
    assert pnl == approx(1.0)
    assert inv.merge_pnl == approx(1.0)
    assert inv.qty == {UP: 0.0, DOWN: 0.0}
    assert inv.cost == {UP: 0.0, DOWN: 0.0}
    assert inv.locked_profit() == 0.0


@pytest.mark.parametrize("winner", [UP, DOWN])
def test_worked_example_settlement_agrees_with_merge(winner: Outcome) -> None:
    merged = pair_inv()
    merge_pnl = merged.apply_merge(100.0)
    settled = pair_inv()
    out = settled.apply_settlement(winner)
    # payout 100 ; cost 99 ; pair part = 100*(1-0.47-0.52) = 1 ; directional = (100-99) - 1 = 0
    assert out.settle_pair_pnl == approx(1.0)
    assert out.settle_directional_pnl == approx(0.0)
    assert out.total == approx(1.0)
    assert out.total == approx(merge_pnl)


def test_worked_example_with_taker_fee_on_one_leg() -> None:
    fees = FeeModel(FeeConfig())
    down_fee = fees.fee(0.52, 100.0, False)  # 13 * 0.06230016 = 0.80990208 (see test_fees)
    assert down_fee == approx(0.80990208)
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.47, 100.0))
    inv.apply_fill(buy(DOWN, 0.52, 100.0, fee=down_fee, maker=False))
    # DOWN cost = 52 + 0.80990208 = 52.80990208 ; avg_down = 0.5280990208
    assert inv.cost[DOWN] == approx(52.80990208)
    assert inv.avg_cost(DOWN) == approx(0.5280990208)
    # locked = 100 * (1 - 0.47 - 0.5280990208) = 100 * 0.0019009792 = 0.19009792
    assert inv.locked_profit() == approx(0.19009792)
    assert inv.fees_paid == approx(0.80990208)
    again = MarketInventory(M)
    for f in (buy(UP, 0.47, 100.0), buy(DOWN, 0.52, 100.0, fee=down_fee, maker=False)):
        again.apply_fill(f)
    # merge: 100 - (47 + 52.80990208) = 0.19009792 ; settlement pair part is the same number
    assert inv.apply_merge(100.0) == approx(0.19009792)
    out = again.apply_settlement(UP)
    assert out.settle_pair_pnl == approx(0.19009792)
    assert out.settle_directional_pnl == approx(0.0)
    assert out.fees_paid == approx(0.80990208)
    assert out.total == approx(0.19009792)


def test_worked_example_with_taker_fee_on_up_leg() -> None:
    fees = FeeModel(FeeConfig())
    up_fee = fees.fee(0.47, 100.0, False)  # 0.7290970175
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.47, 100.0, fee=up_fee, maker=False))
    inv.apply_fill(buy(DOWN, 0.52, 100.0))
    # pnl = 100 - (47 + 0.7290970175 + 52) = 0.2709029825
    assert inv.apply_merge(100.0) == approx(0.2709029825)


def test_locked_profit_with_unpaired_remainder() -> None:
    inv = pair_inv(up=(0.45, 150.0), down=(0.50, 100.0))
    # paired = 100 ; avg_up = 0.45 ; avg_down = 0.50 -> 100 * (1 - 0.45 - 0.50) = 5.0
    assert inv.paired_qty == 100.0
    assert inv.locked_profit() == approx(5.0)
    assert inv.unpaired() == (UP, 50.0)


def test_locked_profit_zero_when_a_side_is_empty() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.4, 10.0))
    assert inv.locked_profit() == 0.0


def test_unpaired_heavy_token_and_balance_tolerance() -> None:
    inv = pair_inv(up=(0.45, 30.0), down=(0.5, 80.0))
    assert inv.net_shares == -50.0
    assert inv.unpaired() == (DOWN, 50.0)
    balanced = pair_inv(up=(0.45, 100.0), down=(0.5, 100.0 + 5e-10))
    assert balanced.unpaired() == (None, 0.0)  # |net| <= EPS counts as balanced
    off = pair_inv(up=(0.45, 100.0), down=(0.5, 100.0 + 5e-9))
    assert off.unpaired()[0] is DOWN


def test_value_at_marks_both_tokens() -> None:
    inv = pair_inv(up=(0.45, 100.0), down=(0.5, 40.0))
    assert inv.value_at(0.7) == approx(82.0)  # 100*0.7 + 40*0.3
    assert inv.value_at(0.0) == approx(40.0)
    assert inv.value_at(1.0) == approx(100.0)
    assert inv.value_at(1.0 + 5e-10) == approx(100.0, 1e-6)  # EPS slack on model output


@pytest.mark.parametrize("p", [-0.1, 1.1, math.nan, math.inf])
def test_value_at_rejects_bad_mark(p: float) -> None:
    with pytest.raises(ValueError, match="p_up"):
        pair_inv().value_at(p)


# --------------------------------------------------------------------------- sells


def test_sell_realises_at_average_cost() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.40, 100.0))
    inv.apply_fill(buy(UP, 0.50, 100.0))
    # qty 200, cost 40 + 50 = 90, avg 0.45
    inv.apply_fill(sell(UP, 0.60, 50.0, fee=0.3))
    # removed = 90 * 50 / 200 = 22.5 ; realised = 30 - 0.3 - 22.5 = 7.2
    assert inv.sell_pnl == approx(7.2)
    assert inv.qty[UP] == 150.0
    assert inv.cost[UP] == approx(67.5)
    assert inv.avg_cost(UP) == approx(0.45)  # average cost is unchanged by a sell
    assert inv.fees_paid == approx(0.3)
    inv.apply_fill(sell(UP, 0.40, 150.0))
    # proceeds 60 ; removed 67.5 ; realised -7.5 ; sell_pnl = 7.2 - 7.5 = -0.3
    assert inv.sell_pnl == approx(-0.3)
    assert inv.qty[UP] == 0.0 and inv.cost[UP] == 0.0  # exactly zero, no dust
    assert inv.avg_cost(UP) is None


def test_sell_down_token_with_rebate_fee() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(DOWN, 0.30, 100.0))  # cost 30
    inv.apply_fill(sell(DOWN, 0.35, 40.0, fee=-0.02))
    # removed = 30 * 40 / 100 = 12 ; realised = 14 + 0.02 - 12 = 2.02 (rebate adds)
    assert inv.sell_pnl == approx(2.02)
    assert inv.qty[DOWN] == 60.0 and inv.cost[DOWN] == approx(18.0)
    assert inv.fees_paid == approx(-0.02)
    assert inv.qty[UP] == 0.0


def test_sell_does_not_touch_the_other_token() -> None:
    inv = pair_inv()
    inv.apply_fill(sell(UP, 0.5, 100.0))
    assert inv.qty == {UP: 0.0, DOWN: 100.0}
    assert inv.cost[DOWN] == approx(52.0)


def test_float_dust_is_snapped_to_zero() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.5, 0.3))
    for _ in range(3):
        inv.apply_fill(sell(UP, 0.5, 0.1))  # 0.3 - 0.1 - 0.1 - 0.1 leaves -2.8e-17 in floats
    assert inv.qty[UP] == 0.0
    assert inv.cost[UP] == 0.0
    # cost 0.15 ; proceeds 3 * 0.05 = 0.15 ; realised 0
    assert inv.sell_pnl == approx(0.0)


def test_sub_eps_residual_is_snapped_and_its_cost_realised() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.40, 100.0))  # cost 40
    inv.apply_fill(sell(UP, 0.50, 100.0 - 5e-10))
    # residual 5e-10 < EPS -> qty exactly 0 and all 40 of cost removed ; proceeds ~50
    assert inv.qty[UP] == 0.0 and inv.cost[UP] == 0.0
    assert inv.sell_pnl == approx(10.0)


def test_sell_within_eps_above_holdings_is_allowed() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.40, 100.0))
    inv.apply_fill(sell(UP, 0.50, 100.0 + 5e-10))
    assert inv.qty[UP] == 0.0 and inv.cost[UP] == 0.0
    # whole 40 of cost removed; proceeds 0.5 * (100 + 5e-10)
    assert inv.sell_pnl == approx(10.0)


def test_residual_above_eps_is_kept() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.40, 100.0))
    inv.apply_fill(sell(UP, 0.50, 100.0 - 5e-9))
    assert inv.qty[UP] == approx(5e-9, 1e-12)
    assert inv.cost[UP] > 0.0
    assert inv.avg_cost(UP) == approx(0.40)


# --------------------------------------------------------------------------- fill errors


@pytest.mark.parametrize(
    ("make_fill", "message"),
    [
        (lambda: buy(UP, 0.5, 10.0, market="other"), "market"),
        (lambda: sell(DOWN, 0.5, 10.0, market="other"), "market"),
        (lambda: sell(UP, 0.5, 10.0), "no UP inventory"),  # nothing held at all
        (lambda: sell(DOWN, 0.5, 100.0), "cannot remove"),  # only 40 DOWN held
        (lambda: sell(UP, 0.5, 100.0 + 2e-9), "cannot remove"),  # beyond EPS slack
        (lambda: buy(UP, 0.5, 0.0), "size"),
        (lambda: buy(UP, 0.5, -1.0), "size"),
        (lambda: sell(UP, 0.5, -1.0), "size"),
        (lambda: buy(UP, 0.5, math.nan), "size"),
        (lambda: buy(UP, 0.5, math.inf), "size"),
        (lambda: buy(UP, -0.01, 10.0), "price"),
        (lambda: buy(UP, 1.01, 10.0), "price"),
        (lambda: sell(UP, math.nan, 10.0), "price"),
        (lambda: buy(UP, 0.5, 10.0, fee=math.nan), "fee"),
        (lambda: sell(UP, 0.5, 10.0, fee=math.inf), "fee"),
    ],
)
def test_fill_errors_raise_and_leave_state_unchanged(make_fill: object, message: str) -> None:
    inv = MarketInventory(M)
    if message != "no UP inventory":
        inv.apply_fill(buy(UP, 0.45, 100.0))
        inv.apply_fill(buy(DOWN, 0.5, 40.0))
    before = state(inv)
    with pytest.raises(ValueError, match=message):
        inv.apply_fill(make_fill())  # type: ignore[operator]
    assert state(inv) == before


def test_prices_at_the_boundaries_are_valid() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.0, 10.0))  # free shares: cost 0
    inv.apply_fill(buy(DOWN, 1.0, 10.0))  # cost 10
    assert inv.cost == {UP: 0.0, DOWN: 10.0}
    assert inv.avg_cost(UP) == 0.0
    inv.apply_fill(sell(DOWN, 1.0, 10.0))
    assert inv.sell_pnl == approx(0.0)  # bought and sold at $1


def test_settled_inventory_rejects_everything() -> None:
    inv = pair_inv()
    inv.apply_settlement(UP)
    for call in (
        lambda: inv.apply_fill(buy(UP, 0.5, 1.0)),
        lambda: inv.apply_fill(sell(UP, 0.5, 1.0)),
        lambda: inv.apply_merge(1.0),
        lambda: inv.apply_settlement(DOWN),
    ):
        with pytest.raises(ValueError, match="already settled"):
            call()


# --------------------------------------------------------------------------- merges


def test_partial_merge_removes_average_cost_from_each_side() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.47, 100.0))
    inv.apply_fill(buy(UP, 0.49, 100.0))  # UP: 200 sh, cost 47 + 49 = 96, avg 0.48
    inv.apply_fill(buy(DOWN, 0.50, 100.0))  # DOWN: 100 sh, cost 50, avg 0.50
    assert inv.paired_qty == 100.0
    pnl = inv.apply_merge(60.0)
    # removed_up = 96*60/200 = 28.8 ; removed_down = 50*60/100 = 30 ; pnl = 60 - 58.8 = 1.2
    assert pnl == approx(1.2)
    assert inv.merge_pnl == approx(1.2)
    assert inv.qty == {UP: 140.0, DOWN: 40.0}
    assert inv.cost[UP] == approx(67.2) and inv.cost[DOWN] == approx(20.0)
    assert inv.avg_cost(UP) == approx(0.48) and inv.avg_cost(DOWN) == approx(0.50)
    # merge the remaining 40 pairs: removed_up = 67.2*40/140 = 19.2 ; removed_down = 20
    # pnl = 40 - 39.2 = 0.8 ; merge_pnl = 2.0 ; UP keeps 100 sh at cost 48 (avg 0.48)
    assert inv.apply_merge(40.0) == approx(0.8)
    assert inv.merge_pnl == approx(2.0)
    assert inv.qty == {UP: 100.0, DOWN: 0.0}
    assert inv.cost[UP] == approx(48.0)
    assert inv.cost[DOWN] == 0.0
    assert inv.unpaired() == (UP, 100.0)


def test_merge_within_eps_of_paired_snaps_to_zero() -> None:
    inv = pair_inv()
    pnl = inv.apply_merge(100.0 + 5e-10)
    assert pnl == approx(1.0)  # credits $size = 100.0000000005 against 99 of cost
    assert inv.qty == {UP: 0.0, DOWN: 0.0}
    assert inv.cost == {UP: 0.0, DOWN: 0.0}


@pytest.mark.parametrize("size", [0.0, -1.0, -1e-12, math.nan, math.inf])
def test_merge_rejects_nonpositive_or_nonfinite_size(size: float) -> None:
    inv = pair_inv()
    before = state(inv)
    with pytest.raises(ValueError, match="merge size"):
        inv.apply_merge(size)
    assert state(inv) == before


def test_merge_more_than_paired_raises_and_leaves_state() -> None:
    inv = pair_inv(up=(0.45, 150.0), down=(0.5, 100.0))
    before = state(inv)
    with pytest.raises(ValueError, match="cannot merge"):
        inv.apply_merge(100.0 + 2e-9)  # paired = 100, only EPS of slack
    assert state(inv) == before


def test_merge_with_nothing_paired_raises() -> None:
    inv = MarketInventory(M)
    with pytest.raises(ValueError, match="cannot merge"):
        inv.apply_merge(1.0)
    inv.apply_fill(buy(UP, 0.5, 10.0))  # one side only
    with pytest.raises(ValueError, match="cannot merge"):
        inv.apply_merge(1.0)
    with pytest.raises(ValueError, match="cannot merge"):
        inv.apply_merge(1e-10)  # within EPS of paired == 0 is still nothing to merge
    assert inv.qty == {UP: 10.0, DOWN: 0.0}


# --------------------------------------------------------------------------- settlement


def test_settlement_splits_pair_and_directional_parts() -> None:
    # UP 150 @ 0.45 (cost 67.5) ; DOWN 100 @ 0.50 (cost 50) ; total cost 117.5
    up = pair_inv(up=(0.45, 150.0), down=(0.5, 100.0))
    out = up.apply_settlement(UP)
    # pair = 100*(1 - 0.45 - 0.50) = 5 ; payout 150 -> directional = 150 - 117.5 - 5 = 27.5
    # (unpaired 50 UP: payout 50 - cost 50*0.45 = 22.5 -> 27.5)
    assert out.settle_pair_pnl == approx(5.0)
    assert out.settle_directional_pnl == approx(27.5)
    assert out.total == approx(32.5)
    assert out.market_id == M

    down = pair_inv(up=(0.45, 150.0), down=(0.5, 100.0))
    out2 = down.apply_settlement(DOWN)
    # payout 100 -> directional = 100 - 117.5 - 5 = -22.5 (the unpaired UP is lost: -50*0.45)
    assert out2.settle_pair_pnl == approx(5.0)
    assert out2.settle_directional_pnl == approx(-22.5)
    assert out2.total == approx(-17.5)


def test_settlement_includes_prior_sell_and_merge_pnl() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.47, 100.0))
    inv.apply_fill(buy(UP, 0.49, 100.0))
    inv.apply_fill(buy(DOWN, 0.50, 100.0))
    inv.apply_merge(60.0)  # merge_pnl 1.2 ; UP 140 @ cost 67.2 ; DOWN 40 @ cost 20
    inv.apply_fill(sell(UP, 0.60, 20.0))  # removed 67.2*20/140 = 9.6 ; realised 12 - 9.6 = 2.4
    assert inv.sell_pnl == approx(2.4)
    assert inv.qty[UP] == 120.0 and inv.cost[UP] == approx(57.6)
    out = inv.apply_settlement(UP)
    # paired 40 -> 40*(1 - 0.48 - 0.50) = 0.8 ; total cost 77.6 ; payout 120
    # directional = 120 - 77.6 - 0.8 = 41.6 ; total = 2.4 + 1.2 + 0.8 + 41.6 = 46.0
    assert out.sell_pnl == approx(2.4) and out.merge_pnl == approx(1.2)
    assert out.settle_pair_pnl == approx(0.8)
    assert out.settle_directional_pnl == approx(41.6)
    assert out.total == approx(46.0)
    assert out.total == approx(out.sell_pnl + out.merge_pnl + out.settle_pair_pnl
                               + out.settle_directional_pnl)  # fmt: skip


@pytest.mark.parametrize(("winner", "expected"), [(UP, 60.0), (DOWN, -40.0)])
def test_settlement_one_sided_position_is_all_directional(winner: Outcome, expected: float) -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.40, 100.0))  # cost 40 ; payout 100 if UP wins else 0
    out = inv.apply_settlement(winner)
    assert out.settle_pair_pnl == 0.0
    assert out.settle_directional_pnl == approx(expected)
    assert out.total == approx(expected)


def test_settlement_of_empty_inventory() -> None:
    inv = MarketInventory(M)
    out = inv.apply_settlement(UP)
    assert out == PnLBreakdown(M, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert inv.settled


def test_settlement_zeroes_positions_and_sets_flags() -> None:
    inv = pair_inv(up=(0.45, 150.0), down=(0.5, 100.0))
    out = inv.apply_settlement(UP)
    assert inv.settled
    assert inv.qty == {UP: 0.0, DOWN: 0.0} and inv.cost == {UP: 0.0, DOWN: 0.0}
    assert inv.avg_cost(UP) is None
    assert inv.locked_profit() == 0.0 and inv.capital_at_risk() == 0.0
    assert inv.value_at(0.5) == 0.0
    assert inv.settle_pnl == approx(out.settle_pair_pnl + out.settle_directional_pnl)
    assert inv.realised_pnl == approx(out.total)


def test_settlement_reports_fees_paid() -> None:
    inv = MarketInventory(M)
    inv.apply_fill(buy(UP, 0.5, 10.0, fee=0.2, maker=False))
    inv.apply_fill(sell(UP, 0.5, 4.0, fee=0.1, maker=False))
    assert inv.apply_settlement(UP).fees_paid == approx(0.3)


def test_pnl_breakdown_is_frozen() -> None:
    out = MarketInventory(M).apply_settlement(UP)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.total = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- property-style


def _random_inventory(rng: random.Random) -> MarketInventory:
    inv = MarketInventory(M)
    for _ in range(rng.randint(1, 8)):
        token = rng.choice([UP, DOWN])
        inv.apply_fill(
            buy(token, rng.randint(1, 99) / 100, rng.uniform(1, 100), rng.uniform(0, 0.5))
        )
    return inv


@pytest.mark.parametrize("seed", range(5))
def test_settlement_pair_part_is_locked_profit_and_directional_differs_by_net(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(100):
        a, b = _random_inventory(rng), MarketInventory(M)
        for token in Outcome:
            b.qty[token], b.cost[token] = a.qty[token], a.cost[token]
        locked, net, cost = a.locked_profit(), a.net_shares, a.capital_at_risk()
        payout_up, payout_down = a.qty[UP], a.qty[DOWN]
        up, down = a.apply_settlement(UP), b.apply_settlement(DOWN)
        assert up.settle_pair_pnl == pytest.approx(locked, abs=1e-9)
        assert down.settle_pair_pnl == pytest.approx(locked, abs=1e-9)
        # directional(UP wins) - directional(DOWN wins) = payout difference = qty_up - qty_down
        assert up.settle_directional_pnl - down.settle_directional_pnl == pytest.approx(
            net, abs=1e-9
        )
        # pair + directional == payout - remaining total cost, for either winner
        assert up.settle_pair_pnl + up.settle_directional_pnl == pytest.approx(
            payout_up - cost, abs=1e-9
        )
        assert down.settle_pair_pnl + down.settle_directional_pnl == pytest.approx(
            payout_down - cost, abs=1e-9
        )


@pytest.mark.parametrize("seed", range(5))
def test_merging_a_fully_paired_inventory_realises_the_locked_profit(seed: int) -> None:
    rng = random.Random(100 + seed)
    for _ in range(100):
        inv = MarketInventory(M)
        up_sizes = [rng.uniform(1, 50) for _ in range(rng.randint(1, 5))]
        for size in up_sizes:
            inv.apply_fill(buy(UP, rng.randint(1, 60) / 100, size, rng.uniform(0, 0.2)))
        down_sizes = [rng.uniform(1, 50) for _ in range(rng.randint(0, 4))]
        down_sizes.append(sum(up_sizes) - sum(down_sizes))  # balance the sides (to float dust)
        if down_sizes[-1] < 0.1:
            continue
        for size in down_sizes:
            inv.apply_fill(buy(DOWN, rng.randint(1, 60) / 100, size, rng.uniform(0, 0.2)))
        assert abs(inv.net_shares) < 1e-9
        locked = inv.locked_profit()
        assert inv.apply_merge(inv.paired_qty) == pytest.approx(locked, abs=1e-9)
        assert inv.qty == {UP: 0.0, DOWN: 0.0} and inv.cost == {UP: 0.0, DOWN: 0.0}


class RefInventory:
    """Exact-arithmetic reference for one market (same rules, Fractions instead of floats)."""

    def __init__(self) -> None:
        self.q = {UP: Fraction(0), DOWN: Fraction(0)}
        self.c = {UP: Fraction(0), DOWN: Fraction(0)}
        self.sell_pnl = Fraction(0)
        self.merge_pnl = Fraction(0)
        self.settle_pnl = Fraction(0)

    def buy(self, token: Outcome, price: float, size: float, fee: float) -> None:
        self.q[token] += Fraction(size)
        self.c[token] += Fraction(price) * Fraction(size) + Fraction(fee)

    def remove(self, token: Outcome, size: Fraction) -> Fraction:
        q, c = self.q[token], self.c[token]
        rem = q - size
        if rem < Fraction(EPS):
            self.q[token] = Fraction(0)
            self.c[token] = Fraction(0)
            return c
        self.q[token] = rem
        self.c[token] = c * rem / q
        return c - self.c[token]

    def sell(self, token: Outcome, price: float, size: float, fee: float) -> None:
        removed = self.remove(token, Fraction(size))
        self.sell_pnl += Fraction(price) * Fraction(size) - Fraction(fee) - removed

    def merge(self, size: float) -> None:
        removed = self.remove(UP, Fraction(size)) + self.remove(DOWN, Fraction(size))
        self.merge_pnl += Fraction(size) - removed

    def settle(self, winner: Outcome) -> Fraction:
        payout = self.q[winner]
        self.settle_pnl = payout - self.c[UP] - self.c[DOWN]
        self.q = {UP: Fraction(0), DOWN: Fraction(0)}
        self.c = {UP: Fraction(0), DOWN: Fraction(0)}
        return payout


def test_inventory_matches_exact_reference_under_random_trading() -> None:
    rng = random.Random(7)
    inv, ref = MarketInventory(M), RefInventory()
    cash = Fraction(0)  # per-market cash flow: identity says cash + cost == sell_pnl + merge_pnl
    for _ in range(400):
        r = rng.random()
        if r < 0.45:
            token, price = rng.choice([UP, DOWN]), rng.randint(1, 99) / 100
            size, fee = rng.uniform(1, 80), rng.uniform(-0.05, 0.4)
            inv.apply_fill(buy(token, price, size, fee))
            ref.buy(token, price, size, fee)
            cash -= Fraction(price) * Fraction(size) + Fraction(fee)
        elif r < 0.75:
            held = [t for t in Outcome if inv.qty[t] > 0]
            if not held:
                continue
            token = rng.choice(held)
            size = inv.qty[token] * rng.choice([1.0, 0.5, rng.uniform(0.05, 0.95)])
            price, fee = rng.randint(1, 99) / 100, rng.uniform(-0.05, 0.4)
            inv.apply_fill(sell(token, price, size, fee))
            ref.sell(token, price, size, fee)
            cash += Fraction(price) * Fraction(size) - Fraction(fee)
        else:
            if inv.paired_qty <= 0:
                continue
            size = inv.paired_qty * rng.choice([1.0, 0.3, rng.uniform(0.05, 0.95)])
            inv.apply_merge(size)
            ref.merge(size)
            cash += Fraction(size)
        for t in Outcome:
            assert inv.qty[t] >= 0.0 and inv.cost[t] >= 0.0
            assert inv.qty[t] == pytest.approx(float(ref.q[t]), abs=1e-7)
            assert inv.cost[t] == pytest.approx(float(ref.c[t]), abs=1e-7)
        assert inv.sell_pnl == pytest.approx(float(ref.sell_pnl), abs=1e-7)
        assert inv.merge_pnl == pytest.approx(float(ref.merge_pnl), abs=1e-7)
        # per-market identity: cash flows + cost basis == realised pnl
        assert float(cash) + inv.capital_at_risk() == pytest.approx(
            inv.sell_pnl + inv.merge_pnl, abs=1e-7
        )
    winner = rng.choice([UP, DOWN])
    payout = ref.settle(winner)
    out = inv.apply_settlement(winner)
    assert out.settle_pair_pnl + out.settle_directional_pnl == pytest.approx(
        float(ref.settle_pnl), abs=1e-7
    )
    cash += payout
    assert float(cash) == pytest.approx(out.total, abs=1e-7)  # all cash flow is now realised


# =========================================================================== Portfolio


def make_order(
    side: Side, price: float, remaining: float, *, live: bool = True, oid: str = "o"
) -> OpenOrder:
    return OpenOrder(
        order_id=oid,
        client_id=oid,
        market_id=M,
        token=UP,
        side=side,
        price=price,
        size=remaining,
        remaining=remaining,
        tif=TimeInForce.POST_ONLY,
        created_ts=0.0,
        live=live,
    )


def test_portfolio_init_and_get_or_create() -> None:
    pf = Portfolio(1000.0)
    assert pf.initial_cash == 1000.0 and pf.cash == 1000.0 and pf.inventories == {}
    inv = pf.inventory(M)
    assert pf.inventory(M) is inv
    assert pf.inventories == {M: inv}
    assert pf.capital_at_risk() == 0.0 and pf.realised_pnl() == 0.0
    assert Portfolio(0.0).cash == 0.0


@pytest.mark.parametrize("cash", [-1.0, math.nan, math.inf])
def test_portfolio_rejects_bad_initial_cash(cash: float) -> None:
    with pytest.raises(ValueError, match="initial_cash"):
        Portfolio(cash)


def test_fills_move_cash_and_inventory() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.47, 100.0, fee=0.5))
    # cash = 1000 - (47 + 0.5) = 952.5
    assert pf.cash == approx(952.5)
    assert pf.inventories[M].qty[UP] == 100.0
    assert pf.capital_at_risk() == approx(47.5)
    pf.apply_fill(sell(UP, 0.60, 40.0, fee=0.3))
    # cash += 24 - 0.3 -> 976.2 ; removed cost = 47.5*40/100 = 19 ; realised = 24 - 0.3 - 19 = 4.7
    assert pf.cash == approx(976.2)
    assert pf.inventories[M].sell_pnl == approx(4.7)
    assert pf.capital_at_risk() == approx(28.5)
    # realised = 976.2 + 28.5 - 1000 = 4.7
    assert pf.realised_pnl() == approx(4.7)


def test_maker_rebate_fill_costs_less_cash() -> None:
    pf = Portfolio(100.0)
    pf.apply_fill(buy(DOWN, 0.5, 10.0, fee=-0.05))
    assert pf.cash == approx(100.0 - 4.95)


def test_buy_that_overdraws_cash_is_rejected_atomically() -> None:
    pf = Portfolio(100.0)
    with pytest.raises(ValueError, match="cash"):
        pf.apply_fill(buy(UP, 0.5, 201.0))  # costs 100.5 > 100
    assert pf.cash == 100.0
    assert pf.inventories == {}  # no stray empty inventory
    pf.apply_fill(buy(UP, 0.5, 40.0))  # fine: 20
    with pytest.raises(ValueError, match="cash"):
        pf.apply_fill(buy(UP, 0.5, 200.0, fee=0.0))  # 100 > 80
    assert pf.cash == approx(80.0)
    assert pf.inventories[M].qty[UP] == 40.0


def test_buy_cash_boundary_and_tolerance() -> None:
    pf = Portfolio(100.0)
    pf.apply_fill(buy(UP, 1.0, 100.0))  # spends exactly everything
    assert pf.cash == 0.0
    pf2 = Portfolio(100.0)
    pf2.apply_fill(buy(UP, 1.0, 100.0 + 5e-10))  # cash -5e-10 is inside the 1e-9 tolerance
    assert pf2.cash == pytest.approx(-5e-10, abs=1e-12)
    pf3 = Portfolio(100.0)
    with pytest.raises(ValueError, match="cash"):
        pf3.apply_fill(buy(UP, 1.0, 100.0 + 2e-9))
    assert pf3.cash == 100.0


def test_failed_sell_leaves_cash_and_inventory_untouched() -> None:
    pf = Portfolio(100.0)
    with pytest.raises(ValueError, match="no UP inventory"):
        pf.apply_fill(sell(UP, 0.5, 1.0))
    assert pf.cash == 100.0 and pf.inventories == {}
    pf.apply_fill(buy(UP, 0.5, 10.0))
    before = (pf.cash, dict(pf.inventories[M].qty))
    with pytest.raises(ValueError, match="cannot remove"):
        pf.apply_fill(sell(UP, 0.5, 11.0))
    assert (pf.cash, dict(pf.inventories[M].qty)) == before


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_portfolio_rejects_nonfinite_fill_fields(bad: float) -> None:
    pf = Portfolio(100.0)
    for f in (buy(UP, 0.5, bad), buy(UP, bad, 1.0), buy(UP, 0.5, 1.0, fee=bad)):
        with pytest.raises(ValueError):
            pf.apply_fill(f)
    assert pf.cash == 100.0 and pf.inventories == {}


def test_fill_into_settled_market_raises_and_keeps_cash() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.5, 10.0))
    pf.apply_settlement(Settlement(M, UP, 1.0, 10.0))
    cash = pf.cash
    for f in (buy(UP, 0.5, 1.0), sell(UP, 0.5, 1.0)):
        with pytest.raises(ValueError, match="already settled"):
            pf.apply_fill(f)
    assert pf.cash == cash


def test_portfolio_worked_example_merge_path() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.47, 100.0))
    pf.apply_fill(buy(DOWN, 0.52, 100.0))
    # cash 1000 - 47 - 52 = 901 ; at risk 99 ; realised 901 + 99 - 1000 = 0
    assert pf.cash == approx(901.0)
    assert pf.capital_at_risk() == approx(99.0)
    assert pf.realised_pnl() == approx(0.0)
    pnl = pf.apply_merge(MergeResult(M, 60.0, 60.0, 1.0))
    # pnl = 60 - 99*0.6 = 0.6 ; cash 961 ; at risk 39.6 ; realised 961 + 39.6 - 1000 = 0.6
    assert pnl == approx(0.6)
    assert pf.cash == approx(961.0)
    assert pf.capital_at_risk() == approx(39.6)
    assert pf.realised_pnl() == approx(0.6)
    pf.apply_merge(MergeResult(M, 40.0, 40.0, 2.0))
    assert pf.cash == approx(1001.0) and pf.capital_at_risk() == 0.0
    assert pf.realised_pnl() == approx(1.0)


@pytest.mark.parametrize("winner", [UP, DOWN])
def test_portfolio_worked_example_settlement_path(winner: Outcome) -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.47, 100.0))
    pf.apply_fill(buy(DOWN, 0.52, 100.0))
    out = pf.apply_settlement(Settlement(M, winner, 5.0, 100.0))
    # cash 901 + 100 = 1001 ; realised 1.0 ; same as merging
    assert pf.cash == approx(1001.0)
    assert pf.capital_at_risk() == 0.0
    assert pf.realised_pnl() == approx(1.0)
    assert out.total == approx(1.0)
    assert out.settle_pair_pnl == approx(1.0)
    assert pf.inventories[M].settled


def test_compounding_cycle_of_pairs_and_merges() -> None:
    pf = Portfolio(1000.0)
    for k in range(1, 6):
        pf.apply_fill(buy(UP, 0.47, 100.0))
        pf.apply_fill(buy(DOWN, 0.52, 100.0))
        pf.apply_merge(MergeResult(M, 100.0, 100.0, float(k)))
        # each cycle nets exactly 100 - 99 = 1.00
        assert pf.cash == approx(1000.0 + k)
        assert pf.realised_pnl() == approx(float(k))
    assert pf.capital_at_risk() == 0.0


def test_merge_errors_leave_portfolio_untouched() -> None:
    pf = Portfolio(1000.0)
    with pytest.raises(ValueError, match="unknown market"):
        pf.apply_merge(MergeResult("nope", 1.0, 1.0, 0.0))
    assert pf.inventories == {}
    pf.apply_fill(buy(UP, 0.47, 100.0))
    pf.apply_fill(buy(DOWN, 0.52, 50.0))
    cash, inv_state = pf.cash, state(pf.inventories[M])
    for result in (
        MergeResult(M, 60.0, 60.0, 0.0),  # more than the 50 paired
        MergeResult(M, 10.0, 9.0, 0.0),  # cash != size
        MergeResult(M, 10.0, math.nan, 0.0),
        MergeResult(M, 0.0, 0.0, 0.0),
        MergeResult(M, -5.0, -5.0, 0.0),
    ):
        with pytest.raises(ValueError):
            pf.apply_merge(result)
        assert pf.cash == cash and state(pf.inventories[M]) == inv_state


def test_settlement_payout_must_match_winning_quantity() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.47, 100.0))
    pf.apply_fill(buy(DOWN, 0.52, 60.0))
    cash, inv_state = pf.cash, state(pf.inventories[M])
    for s in (
        Settlement(M, UP, 1.0, 60.0),  # wrong side's quantity
        Settlement(M, UP, 1.0, 99.0),
        Settlement(M, DOWN, 1.0, 100.0),
        Settlement(M, UP, 1.0, 100.0 + 2e-9),
        Settlement(M, UP, 1.0, math.nan),
    ):
        with pytest.raises(ValueError, match="payout"):
            pf.apply_settlement(s)
        assert pf.cash == cash and state(pf.inventories[M]) == inv_state
    out = pf.apply_settlement(Settlement(M, UP, 1.0, 100.0 + 5e-10))  # inside tolerance
    # cash is credited with the reported payout, not the (within-tolerance) quantity held
    assert pf.cash == pytest.approx(cash + 100.0 + 5e-10, rel=0, abs=1e-11)
    assert out.total == approx(100.0 - 47.0 - 31.2)  # 60 DOWN cost 0.52*60 = 31.2


def test_double_settlement_raises() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.5, 10.0))
    pf.apply_settlement(Settlement(M, UP, 1.0, 10.0))
    cash = pf.cash
    for payout in (0.0, 10.0):  # even a (bogus) non-zero payout reports the real problem
        with pytest.raises(ValueError, match="already settled"):
            pf.apply_settlement(Settlement(M, UP, 2.0, payout))
    assert pf.cash == cash


def test_settlement_of_unknown_market() -> None:
    pf = Portfolio(1000.0)
    with pytest.raises(ValueError, match="payout"):
        pf.apply_settlement(Settlement("ghost", UP, 1.0, 5.0))
    assert pf.inventories == {} and pf.cash == 1000.0
    out = pf.apply_settlement(Settlement("ghost", UP, 1.0, 0.0))
    assert out.total == 0.0 and pf.cash == 1000.0
    assert pf.inventories["ghost"].settled


def test_realised_pnl_excludes_unrealised_and_includes_losses() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.60, 100.0))  # cost 60
    assert pf.realised_pnl() == approx(0.0)  # nothing realised yet, whatever the mark
    pf.apply_fill(sell(UP, 0.50, 100.0))  # realise -10
    assert pf.realised_pnl() == approx(-10.0)
    assert pf.cash == approx(990.0)


# --------------------------------------------------------------------------- equity


def test_equity_values_open_positions_at_marks_or_cost() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.47, 100.0, market="A"))  # cost 47
    pf.apply_fill(buy(DOWN, 0.30, 50.0, market="B"))  # cost 15 ; cash 938
    assert pf.equity({}) == approx(1000.0)  # all at cost: 938 + 47 + 15
    # A marked 0.6 -> 60 ; B unmarked -> at cost 15 ; 938 + 60 + 15 = 1013
    assert pf.equity({"A": 0.6}) == approx(1013.0)
    # B marked 0.2 -> DOWN worth 0.8 -> 50*0.8 = 40 ; 938 + 60 + 40 = 1038
    assert pf.equity({"A": 0.6, "B": 0.2}) == approx(1038.0)
    # marks for unknown markets are ignored
    assert pf.equity({"A": 0.6, "B": 0.2, "Z": 0.9}) == approx(1038.0)


def test_equity_ignores_marks_of_settled_markets() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.47, 100.0, market="A"))
    pf.apply_fill(buy(DOWN, 0.30, 50.0, market="B"))
    pf.apply_settlement(Settlement("A", UP, 1.0, 100.0))  # cash 938 + 100 = 1038
    # B at 0.2 -> 40 ; A's mark must not double count: equity = 1038 + 40 = 1078
    assert pf.equity({"A": 0.0, "B": 0.2}) == approx(1078.0)
    # = initial + realised(A: 100 - 47 = 53) + unrealised(B: 40 - 15 = 25)
    assert pf.equity({"B": 0.2}) == approx(1000.0 + 53.0 + 25.0)


def test_equity_rejects_invalid_mark_for_held_market() -> None:
    pf = Portfolio(1000.0)
    pf.apply_fill(buy(UP, 0.5, 10.0))
    with pytest.raises(ValueError, match="p_up"):
        pf.equity({M: 1.5})


def test_equity_of_empty_portfolio_is_cash() -> None:
    assert Portfolio(123.0).equity({"x": 0.5}) == 123.0


# --------------------------------------------------------------------------- reserved cash


def test_reserved_and_available_cash() -> None:
    pf = Portfolio(1000.0)
    orders = [
        make_order(Side.BUY, 0.40, 100.0, oid="a"),  # 40
        make_order(Side.BUY, 0.55, 30.0, oid="b"),  # partially filled: remaining 30 -> 16.5
        make_order(Side.SELL, 0.60, 80.0, oid="c"),  # SELLs reserve no cash
        make_order(Side.BUY, 0.25, 20.0, live=False, oid="d"),  # in flight still reserves: 5
    ]
    assert pf.reserved_cash(orders) == approx(61.5)
    assert pf.available_cash(orders) == approx(938.5)
    assert pf.reserved_cash(iter(orders)) == approx(61.5)  # any single-pass iterable
    assert pf.reserved_cash([]) == 0.0
    assert pf.available_cash([]) == 1000.0
    assert pf.reserved_cash([make_order(Side.SELL, 0.6, 10.0)]) == 0.0


def test_reserved_cash_uses_remaining_not_original_size() -> None:
    o = make_order(Side.BUY, 0.5, 100.0)
    o.remaining = 25.0
    assert Portfolio(10.0).reserved_cash([o]) == approx(12.5)
    o.remaining = 0.0
    assert Portfolio(10.0).reserved_cash([o]) == 0.0


def test_available_cash_is_not_clamped() -> None:
    pf = Portfolio(10.0)
    assert pf.available_cash([make_order(Side.BUY, 0.5, 100.0)]) == approx(-40.0)


def test_reserved_cash_validation() -> None:
    pf = Portfolio(100.0)
    with pytest.raises(ValueError, match="negative remaining"):
        pf.reserved_cash([make_order(Side.BUY, 0.5, -1.0)])
    with pytest.raises(ValueError, match="price"):
        pf.reserved_cash([make_order(Side.BUY, 1.5, 1.0)])
    # float dust below zero is tolerated and reserves nothing
    assert pf.reserved_cash([make_order(Side.BUY, 0.5, -1e-12)]) == 0.0


# --------------------------------------------------------------------------- randomised identities


def _fee_cfgs() -> list[FeeConfig]:
    return [
        FeeConfig(),
        FeeConfig(maker_fee_rate=0.01, maker_rebate_rate=0.004),
        FeeConfig(maker_rebate_rate=0.01, taker_fee_rate=0.1, taker_fee_exponent=1.0),
    ]


def _check_state(
    pf: Portfolio, refs: dict[str, RefInventory], ref_cash: Fraction, initial: float
) -> None:
    assert pf.cash >= -1e-9
    assert float(ref_cash) == pytest.approx(pf.cash, abs=1e-7)
    realised = 0.0
    for mid, inv in pf.inventories.items():
        ref = refs[mid]
        for t in Outcome:
            assert inv.qty[t] >= 0.0 and inv.cost[t] >= 0.0
            if inv.qty[t] == 0.0:
                assert inv.cost[t] == 0.0
            assert inv.qty[t] == pytest.approx(float(ref.q[t]), abs=1e-7)
            assert inv.cost[t] == pytest.approx(float(ref.c[t]), abs=1e-7)
        assert inv.sell_pnl == pytest.approx(float(ref.sell_pnl), abs=1e-7)
        assert inv.merge_pnl == pytest.approx(float(ref.merge_pnl), abs=1e-7)
        assert inv.settle_pnl == pytest.approx(float(ref.settle_pnl), abs=1e-7)
        realised += inv.sell_pnl + inv.merge_pnl + (inv.settle_pnl if inv.settled else 0.0)
    # DESIGN 7.2: cash + capital_at_risk - initial_cash == realised pnl
    assert pf.realised_pnl() == pytest.approx(realised, abs=1e-8)
    # valued entirely at cost, equity is initial cash plus realised pnl
    assert pf.equity({}) == pytest.approx(initial + realised, abs=1e-8)


def _run_random_portfolio(seed: int, steps: int = 300) -> tuple[Counter[str], list[PnLBreakdown]]:
    rng = random.Random(seed)
    fees = FeeModel(_fee_cfgs()[seed % 3])
    initial = 5000.0
    pf = Portfolio(initial)
    ref_cash = Fraction(initial)
    refs: dict[str, RefInventory] = {}
    next_id = 5
    open_markets = [f"m{i}" for i in range(next_id)]
    counts: Counter[str] = Counter()
    breakdowns: list[PnLBreakdown] = []

    def ref_of(mid: str) -> RefInventory:
        return refs.setdefault(mid, RefInventory())

    for step in range(steps):
        r = rng.random()
        if r < 0.50:
            mid, token = rng.choice(open_markets), rng.choice([UP, DOWN])
            price, size, maker = rng.randint(1, 99) / 100, rng.uniform(1, 80), rng.random() < 0.6
            fee = fees.fee(price, size, maker)
            if price * size + fee > pf.cash:
                counts["skipped"] += 1
                continue
            pf.apply_fill(buy(token, price, size, fee, market=mid, maker=maker))
            ref_of(mid).buy(token, price, size, fee)
            ref_cash -= Fraction(price) * Fraction(size) + Fraction(fee)
            counts["buy"] += 1
        elif r < 0.75:
            held = [
                (m, t)
                for m in open_markets
                if m in pf.inventories
                for t in Outcome
                if pf.inventories[m].qty[t] > 0
            ]
            if not held:
                counts["skipped"] += 1
                continue
            mid, token = rng.choice(held)
            qty = pf.inventories[mid].qty[token]
            mode = rng.choice(["all", "frac", "frac", "dust"])
            if mode == "all":
                size = qty
            elif mode == "dust" and qty > 1.0:
                size = qty - 1e-10  # leaves a sub-EPS residual that must be snapped
                counts["dust"] += 1
            else:
                size = qty * rng.uniform(0.05, 0.95)
            price, maker = rng.randint(1, 99) / 100, rng.random() < 0.5
            fee = fees.fee(price, size, maker)
            pf.apply_fill(sell(token, price, size, fee, market=mid, maker=maker))
            ref_of(mid).sell(token, price, size, fee)
            ref_cash += Fraction(price) * Fraction(size) - Fraction(fee)
            counts["sell"] += 1
        elif r < 0.90:
            paired = [
                m for m in open_markets if m in pf.inventories and pf.inventories[m].paired_qty > 0
            ]
            if not paired:
                counts["skipped"] += 1
                continue
            mid = rng.choice(paired)
            pq = pf.inventories[mid].paired_qty
            size = rng.choice([pq, pq * rng.uniform(0.1, 0.9), pq - 1e-10 if pq > 1 else pq])
            pf.apply_merge(MergeResult(mid, size, size, float(step)))
            ref_of(mid).merge(size)
            ref_cash += Fraction(size)
            counts["merge"] += 1
        elif r < 0.91:
            mid, winner = rng.choice(open_markets), rng.choice([UP, DOWN])
            payout = pf.inventories[mid].qty[winner] if mid in pf.inventories else 0.0
            breakdowns.append(pf.apply_settlement(Settlement(mid, winner, float(step), payout)))
            ref_cash += ref_of(mid).settle(winner)
            open_markets.remove(mid)
            open_markets.append(f"m{next_id}")  # windows roll over: a new market opens
            next_id += 1
            counts["settle"] += 1
        else:
            counts["idle"] += 1
            continue
        _check_state(pf, refs, ref_cash, initial)

    # settle everything that is still open: DESIGN 7.4
    for mid in list(open_markets):
        winner = rng.choice([UP, DOWN])
        payout = pf.inventories[mid].qty[winner] if mid in pf.inventories else 0.0
        breakdowns.append(pf.apply_settlement(Settlement(mid, winner, float(steps), payout)))
        ref_cash += ref_of(mid).settle(winner)
        counts["settle"] += 1
    _check_state(pf, refs, ref_cash, initial)
    assert pf.capital_at_risk() == 0.0
    assert pf.equity({}) == pytest.approx(pf.cash, abs=1e-12)
    total = sum(b.total for b in breakdowns)
    assert total == pytest.approx(pf.cash - initial, abs=1e-7)
    assert pf.realised_pnl() == pytest.approx(total, abs=1e-7)
    assert all(inv.settled for inv in pf.inventories.values())
    return counts, breakdowns


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_portfolio_identities_hold_under_random_trading(seed: int) -> None:
    counts, breakdowns = _run_random_portfolio(seed)
    executed = counts["buy"] + counts["sell"] + counts["merge"] + counts["settle"]
    assert executed >= 200  # at least 200 real random steps per seed
    assert counts["buy"] > 0 and counts["sell"] > 0 and counts["merge"] > 0
    assert len(breakdowns) >= 5  # the 5 markets open at the end are always settled
    for b in breakdowns:
        assert b.total == pytest.approx(
            b.sell_pnl + b.merge_pnl + b.settle_pair_pnl + b.settle_directional_pnl, abs=1e-9
        )


def test_random_runs_exercise_dust_snapping_and_mid_run_settlement() -> None:
    totals: Counter[str] = Counter()
    for seed in range(1, 7):
        counts, _ = _run_random_portfolio(seed)
        totals.update(counts)
    assert totals["dust"] > 0
    assert totals["settle"] > 6 * 5  # some markets settled mid-run, not only at the end


def test_random_run_is_deterministic() -> None:
    a, _ = _run_random_portfolio(3)
    b, _ = _run_random_portfolio(3)
    assert a == b
