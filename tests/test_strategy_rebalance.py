"""Tests for abc_trading.strategy.rebalance (pairs_to_merge, plan_completion, plan_flatten).

Default fee curve: fee/share(p) = p * 0.25 * (p * (1 - p))**2. Hand-worked values used below:
    fee(0.51) = 0.51 * 0.25 * 0.2499^2 = 0.007962376275,  all-in buy cost 0.517962376275
    fee(0.49) = 0.49 * 0.25 * 0.2499^2 = 0.007650126225,  net sale proceeds 0.482349873775
Defaults: rebalance_trigger_shares 100, taker_lock_margin 0.005, rebalance_max_loss_per_pair 0,
merge_min_pairs 50, min_order_size 5, hold_margin 0.02, max_directional_shares 200.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from abc_trading.config import BotConfig, DirectionalConfig, FeeConfig, PairConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.rebalance import (
    REASON_BELOW_MIN,
    REASON_EDGE,
    REASON_FLAT,
    REASON_NO_BID,
    REASON_SELL,
    CompletionOrder,
    FlattenPlan,
    SellOrder,
    pairs_to_merge,
    plan_completion,
    plan_flatten,
)
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
MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)  # tick 0.01, min size 5
CFG = BotConfig()
FEES = FeeModel(CFG.fees)


def fv(p_up: float = 0.5, *, valid: bool = True) -> FairValue:
    return FairValue(p_up, p_up, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, valid)


def lvl(price: float, size: float = 400.0) -> tuple[Level, ...]:
    return (Level(price, size),)


def book(
    token: Outcome,
    bid: float | None,
    ask: float | None,
    bid_size: float = 400.0,
    ask_size: float = 400.0,
) -> BookSnapshot:
    return BookSnapshot(
        token,
        () if bid is None else lvl(bid, bid_size),
        () if ask is None else lvl(ask, ask_size),
    )


def snap(up: BookSnapshot | None = None, down: BookSnapshot | None = None) -> MarketSnapshot:
    return MarketSnapshot(
        1850.0,
        MKT,
        up or book(UP, 0.49, 0.51),
        down or book(DOWN, 0.49, 0.51),
        spot=100.0,
        spot_ts=1850.0,
        ref_price=100.0,
    )


def inv_with(
    up: float = 0.0, down: float = 0.0, up_px: float = 0.47, down_px: float = 0.52
) -> MarketInventory:
    inv = MarketInventory("m1")
    for token, qty, px in ((UP, up, up_px), (DOWN, down, down_px)):
        if qty > 0:
            inv.apply_fill(Fill("f", "o", "m1", token, Side.BUY, px, qty, 0.0, True, 1.0))
    return inv


def completion(
    s: MarketSnapshot,
    inv: MarketInventory,
    *,
    cfg: BotConfig = CFG,
    fair: FairValue | None = None,
    budget: float = 10_000.0,
    target: float = 0.0,
) -> list[CompletionOrder]:
    return plan_completion(
        snap=s,
        inv=inv,
        fair=fair or fv(),
        cfg=cfg,
        fees=FeeModel(cfg.fees),
        cash_budget=budget,
        target_net=target,
    )


# --------------------------------------------------------------------------- pairs_to_merge


def test_merge_threshold_outside_flatten() -> None:
    assert pairs_to_merge(inv_with(49, 49), CFG, flatten=False) == 0.0  # below merge_min_pairs
    assert pairs_to_merge(inv_with(50, 50), CFG, flatten=False) == 50.0
    assert pairs_to_merge(inv_with(120, 70), CFG, flatten=False) == 70.0  # paired = 70
    assert pairs_to_merge(inv_with(50.7, 52.1), CFG, flatten=False) == 50.0  # floor(50.7)
    assert pairs_to_merge(inv_with(100, 0), CFG, flatten=False) == 0.0  # no pairs at all


def test_merge_in_flatten_takes_any_whole_pair() -> None:
    assert pairs_to_merge(inv_with(3.9, 8.0), CFG, flatten=True) == 3.0
    assert pairs_to_merge(inv_with(1, 1), CFG, flatten=True) == 1.0
    assert pairs_to_merge(inv_with(0.9, 5.0), CFG, flatten=True) == 0.0  # less than one pair
    assert pairs_to_merge(inv_with(0, 0), CFG, flatten=True) == 0.0


def test_merge_disabled() -> None:
    off = replace(CFG, pair=PairConfig(merge_enabled=False))
    assert pairs_to_merge(inv_with(100, 100), off, flatten=False) == 0.0
    assert pairs_to_merge(inv_with(100, 100), off, flatten=True) == 0.0


def test_merge_never_exceeds_what_is_held_with_float_dust() -> None:
    inv = inv_with(99.9999999995, 100.0)  # a hair under 100 paired
    size = pairs_to_merge(inv, CFG, flatten=False)
    assert size <= inv.paired_qty  # the exchange is never asked for more than is held
    assert size == pytest.approx(100.0, abs=1e-9)


# --------------------------------------------------------------------------- plan_completion


def test_completion_hand_computed() -> None:
    # Heavy UP: 150 @ 0.47. DOWN ask 0.51 (size 400): unit cost 0.517962376275,
    # net profit/share = 1 - 0.47 - 0.517962376275 = 0.012037623725 >= taker_lock_margin 0.005.
    # size = min(excess 150, displayed 400, 10000 / 0.51796 = 19305) = 150.
    plans = completion(snap(), inv_with(up=150))
    assert plans == [CompletionOrder(DOWN, 0.51, 150.0, plans[0].net_profit_per_share)]
    assert plans[0].net_profit_per_share == pytest.approx(0.012037623725, abs=1e-12)


def test_completion_mirror_for_a_heavy_down() -> None:
    # Heavy DOWN 150 @ 0.47 -> buy UP at its ask 0.51: profit = 1 - 0.47 - 0.517962376275.
    plans = completion(snap(), inv_with(down=150, down_px=0.47))
    assert [(p.token, p.price, p.size) for p in plans] == [(UP, 0.51, 150.0)]
    assert plans[0].net_profit_per_share == pytest.approx(1 - 0.47 - 0.517962376275, abs=1e-12)
    # At an average cost of 0.52 the same completion would lose 0.0380 per pair: no order.
    assert completion(snap(), inv_with(down=150, down_px=0.52)) == []


def test_completion_trigger_is_inclusive() -> None:
    assert completion(snap(), inv_with(up=99)) == []
    assert [p.size for p in completion(snap(), inv_with(up=100))] == [100.0]
    # Only the unpaired part counts: 300 UP vs 250 DOWN is 50 unpaired.
    assert completion(snap(), inv_with(up=300, down=250)) == []
    assert [p.size for p in completion(snap(), inv_with(up=350, down=250))] == [100.0]


def test_completion_profit_test_with_fees() -> None:
    # DOWN ask 0.53: fee = 0.53*0.25*(0.53*0.47)^2 = 0.00822, so profit = 1 - 0.47 - 0.5382 < 0.
    s = snap(down=book(DOWN, 0.51, 0.53))
    assert completion(s, inv_with(up=150)) == []


def test_completion_profit_boundary_without_fees() -> None:
    cfg = replace(CFG, fees=FeeConfig(taker_fee_rate=0.0))
    inv = inv_with(up=150, up_px=0.47)
    # profit = 1 - 0.47 - ask. ask 0.525 -> exactly 0.005 == taker_lock_margin -> allowed.
    assert len(completion(snap(down=book(DOWN, 0.51, 0.525)), inv, cfg=cfg)) == 1
    assert completion(snap(down=book(DOWN, 0.51, 0.526)), inv, cfg=cfg) == []
    # rebalance_max_loss_per_pair lowers the bar to 0.005 - 0.01 = -0.005.
    loose = replace(cfg, pair=PairConfig(rebalance_max_loss_per_pair=0.01))
    assert len(completion(snap(down=book(DOWN, 0.51, 0.535)), inv, cfg=loose)) == 1  # profit -0.005
    assert completion(snap(down=book(DOWN, 0.51, 0.536)), inv, cfg=loose) == []


def test_completion_size_is_limited_by_displayed_ask() -> None:
    s = snap(down=book(DOWN, 0.49, 0.51, ask_size=40.0))
    assert [p.size for p in completion(s, inv_with(up=150))] == [40.0]


def test_completion_size_is_limited_by_cash_budget() -> None:
    # 25 / 0.517962376275 = 48.2660...; floored to 0.01 -> 48.26 (cost 24.99686 <= 25).
    plans = completion(snap(), inv_with(up=150), budget=25.0)
    assert [p.size for p in plans] == [48.26]
    assert plans[0].size * 0.517962376275 <= 25.0


def test_completion_below_min_order_size_is_dropped() -> None:
    # 2 / 0.517962 = 3.86 shares < min_order_size 5.
    assert completion(snap(), inv_with(up=150), budget=2.0) == []
    assert completion(snap(), inv_with(up=150), budget=0.0) == []


def test_completion_directional_allowance_is_not_rebalanced() -> None:
    long_up = fv(0.7)
    # Target +100 UP: of 150 unpaired UP, 100 is deliberate -> excess 50 < trigger 100 -> nothing.
    assert completion(snap(), inv_with(up=150), fair=long_up, target=100.0) == []
    # 250 unpaired: excess 150 >= 100 -> complete only the excess.
    assert [p.size for p in completion(snap(), inv_with(up=250), fair=long_up, target=100.0)] == [
        150.0
    ]
    # A target in the OTHER direction gives no allowance: the full 150 is unintended.
    assert [p.size for p in completion(snap(), inv_with(up=150), fair=long_up, target=-100.0)] == [
        150.0
    ]
    # An invalid model or a disabled overlay ignores the target entirely.
    assert [
        p.size
        for p in completion(snap(), inv_with(up=150), fair=fv(0.7, valid=False), target=100.0)
    ] == [150.0]
    off = replace(CFG, directional=DirectionalConfig(enabled=False))
    assert [p.size for p in completion(snap(), inv_with(up=150), cfg=off, target=100.0)] == [150.0]
    # The allowance is clamped to max_directional_shares (200): target 900 acts like 200.
    assert completion(snap(), inv_with(up=250), fair=long_up, target=900.0) == []  # excess 50
    assert [p.size for p in completion(snap(), inv_with(up=350), fair=long_up, target=900.0)] == [
        150.0
    ]


def test_completion_needs_a_usable_ask_and_an_unpaired_inventory() -> None:
    assert completion(snap(down=book(DOWN, 0.49, None)), inv_with(up=150)) == []
    assert completion(snap(down=BookSnapshot(DOWN)), inv_with(up=150)) == []
    assert completion(snap(down=book(DOWN, 0.99, 1.0)), inv_with(up=150)) == []  # ask at 1
    assert completion(snap(), inv_with(150, 150)) == []  # fully paired
    assert completion(snap(), inv_with()) == []  # empty


def test_completion_input_validation() -> None:
    with pytest.raises(ValueError, match="cash_budget"):
        completion(snap(), inv_with(up=150), budget=-1.0)
    with pytest.raises(ValueError, match="cash_budget"):
        completion(snap(), inv_with(up=150), budget=math.nan)
    with pytest.raises(ValueError, match="target_net"):
        completion(snap(), inv_with(up=150), target=math.nan)


# --------------------------------------------------------------------------- plan_flatten


def flatten(
    s: MarketSnapshot,
    inv: MarketInventory,
    fair: FairValue,
    *,
    cfg: BotConfig = CFG,
    pending: dict[Outcome, float] | None = None,
) -> FlattenPlan:
    return plan_flatten(
        snap=s, inv=inv, fair=fair, cfg=cfg, fees=FeeModel(cfg.fees), pending_sells=pending
    )


def test_flatten_merges_pairs_and_holds_when_the_model_has_edge() -> None:
    # UP 100 @ 0.47, DOWN 60 @ 0.52: merge 60 pairs; 40 unpaired UP.
    # UP bid 0.49: net sale = 0.49 - 0.007650126225 = 0.482349873775.
    # p_up = 0.55: 0.55 - 0.482349873775 = 0.0676 >= hold_margin 0.02 -> hold all 40.
    plan = flatten(snap(), inv_with(100, 60), fv(0.55))
    assert plan.merge_size == 60.0
    assert plan.sells == ()
    assert (plan.hold_token, plan.hold_size, plan.reason) == (UP, 40.0, REASON_EDGE)


def test_flatten_sells_when_the_edge_is_below_hold_margin() -> None:
    # p_up = 0.50: 0.50 - 0.482349873775 = 0.01765 < 0.02 -> sell all 40 at the bid 0.49.
    plan = flatten(snap(), inv_with(100, 60), fv(0.50))
    assert plan.merge_size == 60.0
    assert plan.sells == (SellOrder(UP, 0.49, 40.0),)
    assert (plan.hold_token, plan.hold_size, plan.reason) == (UP, 0.0, REASON_SELL)


def test_flatten_hold_margin_boundary() -> None:
    cfg = replace(CFG, fees=FeeConfig(taker_fee_rate=0.0))  # exit value = bid exactly
    inv = inv_with(40, 0)
    # bid 0.49, hold_margin 0.02: p = 0.51 -> 0.51 - 0.49 = 0.02 -> hold; p = 0.5099 -> sell.
    assert flatten(snap(), inv, fv(0.51), cfg=cfg).sells == ()
    assert flatten(snap(), inv, fv(0.5099), cfg=cfg).sells == (SellOrder(UP, 0.49, 40.0),)


def test_flatten_hold_is_capped_by_max_directional_shares() -> None:
    # 300 unpaired UP, strong model: hold up to 200, sell the 100 excess at the bid.
    plan = flatten(snap(), inv_with(300, 0), fv(0.95))
    assert plan.sells == (SellOrder(UP, 0.49, 100.0),)
    assert (plan.hold_size, plan.reason) == (200.0, REASON_EDGE)


def test_flatten_sells_everything_when_model_invalid_or_overlay_off() -> None:
    inv = inv_with(40, 0)
    assert flatten(snap(), inv, fv(0.95, valid=False)).sells == (SellOrder(UP, 0.49, 40.0),)
    off = replace(CFG, directional=DirectionalConfig(enabled=False))
    assert flatten(snap(), inv, fv(0.95), cfg=off).sells == (SellOrder(UP, 0.49, 40.0),)


def test_flatten_mirror_for_a_heavy_down() -> None:
    # Heavy DOWN 40; the DOWN bid is what matters. p_up = 0.45 -> p_down = 0.55.
    s = snap(down=book(DOWN, 0.49, 0.51))
    hold = flatten(s, inv_with(0, 40), fv(0.45))
    assert (hold.sells, hold.hold_token, hold.hold_size) == ((), DOWN, 40.0)
    sell = flatten(s, inv_with(0, 40), fv(0.50))  # p_down = 0.5: 0.5 - 0.48235 < 0.02
    assert sell.sells == (SellOrder(DOWN, 0.49, 40.0),)


def test_flatten_skips_selling_without_a_usable_bid() -> None:
    inv = inv_with(40, 0)
    for up_book in (
        book(UP, None, 0.51),  # empty bids
        book(UP, 0.0, 0.01),  # bid at 0
        book(UP, 0.52, 0.51),  # crossed
    ):
        plan = flatten(snap(up=up_book), inv, fv(0.10))  # model says sell, but no bid
        assert plan.sells == ()
        assert (plan.hold_token, plan.hold_size, plan.reason) == (UP, 40.0, REASON_NO_BID)
    # Pairs are still merged.
    assert flatten(snap(up=book(UP, None, 0.51)), inv_with(100, 60), fv(0.1)).merge_size == 60.0


def test_flatten_dust_below_min_order_size_is_held() -> None:
    plan = flatten(snap(), inv_with(3, 0), fv(0.50))
    assert plan.sells == ()
    assert (plan.hold_size, plan.reason) == (3.0, REASON_BELOW_MIN)


def test_flatten_floors_sell_size_to_the_lot() -> None:
    plan = flatten(snap(), inv_with(40.567, 0), fv(0.50))
    assert plan.sells == (SellOrder(UP, 0.49, 40.56),)
    assert plan.hold_size == pytest.approx(0.007, abs=1e-9)  # the sub-lot dust


def test_flatten_accounts_for_pending_sells() -> None:
    inv = inv_with(40, 0)
    # 25 already offered by an in-flight IOC: only 15 more are planned.
    plan = flatten(snap(), inv, fv(0.50), pending={UP: 25.0})
    assert plan.sells == (SellOrder(UP, 0.49, 15.0),)
    assert plan.hold_size == 0.0
    # Everything already offered: nothing further to send.
    done = flatten(snap(), inv, fv(0.50), pending={UP: 40.0})
    assert done.sells == ()
    assert done.hold_size == 0.0
    # Pending on the other token is irrelevant.
    other = flatten(snap(), inv, fv(0.50), pending={DOWN: 40.0})
    assert other.sells == (SellOrder(UP, 0.49, 40.0),)


def test_flatten_flat_and_pairs_only() -> None:
    flat = flatten(snap(), inv_with(), fv())
    assert (flat.merge_size, flat.sells, flat.hold_token, flat.hold_size, flat.reason) == (
        0.0,
        (),
        None,
        0.0,
        REASON_FLAT,
    )
    paired = flatten(snap(), inv_with(75, 75), fv())
    assert paired.merge_size == 75.0
    assert (paired.sells, paired.hold_token, paired.reason) == ((), None, REASON_FLAT)


def test_flatten_merge_disabled_still_resolves_the_remainder() -> None:
    off = replace(CFG, pair=PairConfig(merge_enabled=False))
    plan = flatten(snap(), inv_with(100, 60), fv(0.5), cfg=off)
    assert plan.merge_size == 0.0
    assert plan.sells == (SellOrder(UP, 0.49, 40.0),)


def test_flatten_never_plans_more_than_held() -> None:
    for up in (5.0, 5.01, 17.3, 99.99, 300.0, 1234.5):
        for p in (0.1, 0.5, 0.97):
            plan = flatten(snap(), inv_with(up, 0), fv(p))
            assert sum(s.size for s in plan.sells) <= up + 1e-9
            assert all(s.size >= 5.0 for s in plan.sells)
            assert plan.hold_size + sum(s.size for s in plan.sells) == pytest.approx(up, abs=1e-9)
