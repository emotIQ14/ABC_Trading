"""Tests for abc_trading.strategy.quoter (compute_quotes, max_bid_for_token, plan_tokens).

Default config used unless stated: target_margin 0.01, clip passed explicitly (100), ladder 3 levels
spacing 1 tick decay 0.7 (so level sizes are 100, 70, 49), skew 1 tick per 100 shares of
imbalance, max_net_imbalance 300, max_inventory_per_side 1500, directional min_edge 0.03,
max_directional_shares 200, max_spread_ticks_to_quote 6, market tick 0.01 and min order size 5.
Inventories are built from fills at an average cost of 0.47 (UP) / 0.52 (DOWN) unless stated, with
zero fees, so avg_cost is exactly the fill price.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from abc_trading.config import (
    BotConfig,
    DirectionalConfig,
    FeeConfig,
    PairConfig,
    RiskConfig,
    SizingConfig,
)
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.quoter import (
    QuoteLevel,
    compute_quotes,
    floor_size,
    max_bid_for_token,
    plan_tokens,
)
from abc_trading.strategy.risk import books_quotable
from abc_trading.types import (
    EPS,
    BookSnapshot,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    Outcome,
    Phase,
    Side,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)  # tick 0.01, min size 5
CFG = BotConfig()
ACC, WD = Phase.ACCUMULATE, Phase.WIND_DOWN


def fv(p_up: float = 0.5, *, valid: bool = True) -> FairValue:
    return FairValue(p_up, p_up, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, valid)


def book(token: Outcome, bid: float | None, ask: float | None, size: float = 400.0) -> BookSnapshot:
    return BookSnapshot(
        token,
        () if bid is None else (Level(bid, size),),
        () if ask is None else (Level(ask, size),),
    )


def snap(
    up: tuple[float | None, float | None] = (0.49, 0.51),
    down: tuple[float | None, float | None] = (0.49, 0.51),
    market: MarketSpec = MKT,
) -> MarketSnapshot:
    return MarketSnapshot(
        1100.0,
        market,
        book(UP, *up),
        book(DOWN, *down),
        spot=100.0,
        spot_ts=1100.0,
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


def quotes(
    s: MarketSnapshot,
    fair: FairValue,
    inv: MarketInventory,
    *,
    phase: Phase = ACC,
    cfg: BotConfig = CFG,
    clip: float = 100.0,
    budget: float = 1e6,
    target: float = 0.0,
    skip: tuple[Outcome, ...] = (),
) -> list[QuoteLevel]:
    return compute_quotes(
        snap=s,
        fair=fair,
        inv=inv,
        phase=phase,
        cfg=cfg,
        fees=FeeModel(cfg.fees),
        clip_shares=clip,
        cash_budget=budget,
        target_net=target,
        skip_tokens=skip,
    )


def view(qs: list[QuoteLevel]) -> list[tuple[str, float, float]]:
    return [(q.token.value, q.price, q.size) for q in qs]


def cap(token: Outcome, fair: FairValue, inv: MarketInventory, **kw: object) -> float | None:
    cfg = kw.get("cfg", CFG)
    assert isinstance(cfg, BotConfig)
    target = kw.get("target", 0.0)
    assert isinstance(target, float)
    return max_bid_for_token(token=token, fair=fair, inv=inv, cfg=cfg, target_net=target)


# --------------------------------------------------------------------------- floor_size


def test_floor_size() -> None:
    assert floor_size(70.0) == 70.0
    assert floor_size(100 * 0.7) == 70.0  # 69.99999999999999 survives the lot rounding
    assert floor_size(100 * 0.7**2) == 49.0  # 48.99999999999999
    assert floor_size(24.5) == 24.5
    assert floor_size(0.019) == 0.01
    assert floor_size(22.9166) == 22.91
    assert floor_size(0.0) == 0.0


# --------------------------------------------------------------------------- balanced ladder


def test_flat_inventory_quotes_a_symmetric_two_sided_ladder() -> None:
    # fair 0.5: cap = 0.5 - 0.005 = 0.495 on both tokens (caps sum to 0.99 = 1 - margin).
    # Level k price = min(0.495, 0.49 - 0.01k) floored: 0.49, 0.48, 0.47; sizes 100, 70, 49.
    qs = quotes(snap(), fv(0.5), inv_with())
    assert view(qs) == [
        ("UP", 0.49, 100.0),
        ("UP", 0.48, 70.0),
        ("UP", 0.47, 49.0),
        ("DOWN", 0.49, 100.0),
        ("DOWN", 0.48, 70.0),
        ("DOWN", 0.47, 49.0),
    ]
    assert cap(UP, fv(0.5), inv_with()) == pytest.approx(0.495)
    assert cap(DOWN, fv(0.5), inv_with()) == pytest.approx(0.495)


def test_caps_sum_to_one_minus_margin_when_flat() -> None:
    for p_up in (0.2, 0.35, 0.5, 0.64, 0.9):
        up = cap(UP, fv(p_up), inv_with())
        down = cap(DOWN, fv(p_up), inv_with())
        assert up is not None
        assert down is not None
        assert up + down == pytest.approx(1 - CFG.pair.target_margin, abs=1e-12)
        assert up == pytest.approx(p_up - 0.005)


def test_ladder_levels_never_duplicate_a_price() -> None:
    # fair_UP 0.55 -> cap 0.545 floors to 0.54. The touch is 0.55, so the raw level prices are
    # min(0.545, 0.55) = 0.545 -> 0.54, min(0.545, 0.54) = 0.54 (duplicate: skipped, its size 70 is
    # NOT merged), min(0.545, 0.53) = 0.53 (size 49).
    s = snap(up=(0.55, 0.57), down=(0.43, 0.45))
    qs = quotes(s, fv(0.55), inv_with())
    assert [(q.price, q.size) for q in qs if q.token is UP] == [(0.54, 100.0), (0.53, 49.0)]
    prices = [(q.token, q.price) for q in qs]
    assert len(prices) == len(set(prices))


def test_cap_below_touch_collapses_to_one_level() -> None:
    # fair 0.40: cap_UP = 0.395 -> 0.39 on a book whose bid is 0.49 -> all three levels clamp to
    # 0.39, so exactly one level (the first, size 100) survives.
    qs = quotes(snap(), fv(0.40), inv_with())
    assert [(q.price, q.size) for q in qs if q.token is UP] == [(0.39, 100.0)]
    # DOWN: p = 0.60, cap 0.595; the touch is 0.49 so the ladder hangs from the touch as usual.
    assert [(q.price, q.size) for q in qs if q.token is DOWN] == [
        (0.49, 100.0),
        (0.48, 70.0),
        (0.47, 49.0),
    ]


def test_ladder_spacing_levels_and_decay() -> None:
    cfg = replace(
        CFG,
        sizing=SizingConfig(ladder_levels=2, ladder_spacing_ticks=2, ladder_size_decay=0.5),
    )
    qs = quotes(snap(), fv(0.5), inv_with(), cfg=cfg)
    # prices 0.49 and 0.49 - 2*0.01 = 0.47; sizes 100 and 50.
    assert view(qs) == [
        ("UP", 0.49, 100.0),
        ("UP", 0.47, 50.0),
        ("DOWN", 0.49, 100.0),
        ("DOWN", 0.47, 50.0),
    ]
    one = replace(CFG, sizing=SizingConfig(ladder_levels=1))
    assert len(quotes(snap(), fv(0.5), inv_with(), cfg=one)) == 2


def test_post_only_never_crosses_and_never_improves_the_touch() -> None:
    # cap 0.975 would allow bidding at the ask; the ladder is capped at best_ask - tick and
    # anchored on the best bid.
    s = snap(up=(0.97, 0.98), down=(0.02, 0.03))
    qs = quotes(s, fv(0.98), inv_with())
    up = [q for q in qs if q.token is UP]
    assert [q.price for q in up] == [0.97, 0.96, 0.95]
    for q in qs:
        assert q.price < s.book(q.token).best_ask  # type: ignore[operator]
    # A one-tick spread whose bid sits at ask - tick: price = bid.
    tight = quotes(snap(up=(0.49, 0.50), down=(0.50, 0.51)), fv(0.6), inv_with())
    assert [q.price for q in tight if q.token is UP][0] == 0.49


def test_price_never_exceeds_best_ask_minus_one_tick_even_on_an_unaligned_book() -> None:
    # An odd book (bid 0.504, ask 0.505, not on the 0.01 grid) with a high cap (0.545): the spec
    # limit is best_ask - tick = 0.495 -> 0.49. Flooring the bid alone would give 0.50.
    s = snap(up=(0.504, 0.505), down=(0.495, 0.496))
    qs = quotes(s, fv(0.55), inv_with())
    assert max(q.price for q in qs if q.token is UP) == 0.49
    plans = plan_tokens(snap=s, fair=fv(0.55), inv=inv_with(), phase=ACC, cfg=CFG, target_net=0.0)
    assert plans[UP].ceiling == 0.49


def test_never_quotes_price_zero_or_one() -> None:
    # fair 0.99 on a book with ask 1.00: the ceiling is min(cap, ask - tick, 1 - tick) = 0.98.
    s = snap(up=(0.98, 1.0), down=(0.0, 0.02))
    qs = quotes(s, fv(0.99), inv_with())
    assert all(0.0 < q.price < 1.0 for q in qs)
    assert max(q.price for q in qs if q.token is UP) == 0.98
    # fair 0.01: cap_UP = 0.005 floors to 0 -> no UP bids at all.
    low = quotes(snap(up=(0.01, 0.02), down=(0.98, 0.99)), fv(0.01), inv_with())
    assert [q for q in low if q.token is UP] == []
    assert all(q.price >= 0.01 for q in low)


def test_finer_tick_size_is_respected() -> None:
    fine = MarketSpec("fine", "BTC", 1000.0, 1900.0, tick_size=0.001, min_order_size=5.0)
    # 4-tick-wide books (0.004 <= 6 * 0.001). fair 0.52 -> cap 0.515 sits above the touch 0.498,
    # so the ladder hangs from the touch in 0.001 steps.
    s = snap(up=(0.498, 0.502), down=(0.498, 0.502), market=fine)
    qs = quotes(s, fv(0.52), inv_with())
    assert [q.price for q in qs if q.token is UP] == [0.498, 0.497, 0.496]
    for q in qs:
        assert abs(q.price / 0.001 - round(q.price / 0.001)) < 1e-6
    # The default 10-tick-wide 0.495 / 0.505 book exceeds the 6-tick limit: no quotes.
    wide = snap(up=(0.495, 0.505), down=(0.495, 0.505), market=fine)
    assert quotes(wide, fv(0.5), inv_with()) == []


# --------------------------------------------------------------------------- pair cap & skew


def test_after_an_up_fill_the_down_cap_is_one_minus_margin_minus_avg_cost() -> None:
    inv = inv_with(up=100, up_px=0.47)
    assert cap(DOWN, fv(0.5), inv) == pytest.approx(1 - 0.01 - 0.47)  # 0.52
    # The UP token itself is not completing: balanced cap fair - margin/2.
    assert cap(UP, fv(0.5), inv) == pytest.approx(0.495)


def test_down_bid_is_pair_capped_even_when_the_book_is_higher() -> None:
    # DOWN touch is 0.55 but we hold unpaired UP @ 0.47: cap 0.52 binds, one level at 0.52.
    # UP is heavy by 100: skew 1 tick down from min(0.495, 0.43) = 0.43 -> 0.42; sizes limited by
    # the net room 300 - 100 = 200: 100 + 70 + 30.
    s = snap(up=(0.43, 0.45), down=(0.55, 0.57))
    qs = quotes(s, fv(0.5), inv_with(up=100, up_px=0.47))
    assert view(qs) == [
        ("UP", 0.42, 100.0),
        ("UP", 0.41, 70.0),
        ("UP", 0.40, 30.0),
        ("DOWN", 0.52, 100.0),
    ]
    # invariant 8: bid + avg_cost[opposite] <= 1 - margin
    assert 1 - 0.01 + EPS >= 0.52 + 0.47


def test_skew_shifts_the_heavy_side_down_by_whole_and_fractional_ticks() -> None:
    # net +100: 1 tick. UP bids 0.48, 0.47, 0.46 (sizes 100, 70, 30 by the net room 200).
    qs = quotes(snap(), fv(0.5), inv_with(up=100))
    assert [(q.price, q.size) for q in qs if q.token is UP] == [
        (0.48, 100.0),
        (0.47, 70.0),
        (0.46, 30.0),
    ]
    # net +150: 1.5 ticks. L0: 0.49 - 0.015 = 0.475 -> 0.47; L1: 0.465 -> 0.46; room 150 -> 100, 50.
    qs = quotes(snap(), fv(0.5), inv_with(up=150))
    assert [(q.price, q.size) for q in qs if q.token is UP] == [(0.47, 100.0), (0.46, 50.0)]
    # DOWN (the completing, light side) is not skewed.
    assert [(q.price, q.size) for q in qs if q.token is DOWN] == [
        (0.49, 100.0),
        (0.48, 70.0),
        (0.47, 49.0),
    ]


def test_skew_is_symmetric_for_a_heavy_down() -> None:
    qs = quotes(snap(), fv(0.5), inv_with(down=100, down_px=0.52))
    assert [(q.price, q.size) for q in qs if q.token is DOWN] == [
        (0.48, 100.0),
        (0.47, 70.0),
        (0.46, 30.0),
    ]
    # UP completes: cap = 1 - 0.01 - 0.52 = 0.47 binds below the touch (0.49): UP rests at 0.47.
    assert [(q.price, q.size) for q in qs if q.token is UP] == [(0.47, 100.0)]


def test_skew_scales_with_skew_ticks_per_100_shares() -> None:
    cfg = replace(CFG, pair=PairConfig(skew_ticks_per_100_shares=2.0))
    # net 100 -> 2 ticks: 0.49 -> 0.47.
    qs = quotes(snap(), fv(0.5), inv_with(up=100), cfg=cfg)
    assert [q.price for q in qs if q.token is UP][0] == 0.47
    off = replace(CFG, pair=PairConfig(skew_ticks_per_100_shares=0.0))
    qs = quotes(snap(), fv(0.5), inv_with(up=100), cfg=off)
    assert [q.price for q in qs if q.token is UP][0] == 0.49


def test_hard_stop_on_the_heavy_side_at_max_net_imbalance() -> None:
    # net +300 == max_net_imbalance_shares: no UP bids; DOWN (completing) still quoted.
    qs = quotes(snap(), fv(0.5), inv_with(up=300))
    assert {q.token for q in qs} == {DOWN}
    # net +250: 50 shares of room, a single (reduced) UP level at 0.49 - 2.5 ticks = 0.465 -> 0.46.
    qs = quotes(snap(), fv(0.5), inv_with(up=250))
    assert [(q.price, q.size) for q in qs if q.token is UP] == [(0.46, 50.0)]
    # net +299.99: only 0.01 shares of room -> below min order size -> dropped.
    qs = quotes(snap(), fv(0.5), inv_with(up=299.99))
    assert [q for q in qs if q.token is UP] == []


def test_per_side_inventory_cap() -> None:
    # 1500 UP and 1400 DOWN (net +100): UP is at max_inventory_per_side -> no UP bids at all.
    qs = quotes(snap(), fv(0.5), inv_with(up=1500, down=1400))
    assert {q.token for q in qs} == {DOWN}
    # 1450 UP / 1400 DOWN: 50 shares of UP room left.
    qs = quotes(snap(), fv(0.5), inv_with(up=1450, down=1400))
    assert sum(q.size for q in qs if q.token is UP) == 50.0
    # A heavy position that is paired is still capped by the side limit.
    big = replace(
        CFG, pair=PairConfig(max_inventory_per_side_shares=520.0, max_net_imbalance_shares=300.0)
    )
    qs = quotes(snap(), fv(0.5), inv_with(up=500, down=500), cfg=big)
    assert sum(q.size for q in qs if q.token is UP) == 20.0
    assert sum(q.size for q in qs if q.token is DOWN) == 20.0


def test_paired_inventory_balanced_quotes_as_flat() -> None:
    qs = quotes(snap(), fv(0.5), inv_with(up=80, down=80))
    assert view(qs) == view(quotes(snap(), fv(0.5), inv_with()))


# --------------------------------------------------------------------------- phases, model, books


@pytest.mark.parametrize("phase", [Phase.WARMUP, Phase.FLATTEN, Phase.DONE])
def test_no_quotes_outside_quoting_phases(phase: Phase) -> None:
    assert quotes(snap(), fv(0.5), inv_with(), phase=phase) == []
    assert quotes(snap(), fv(0.5), inv_with(up=200), phase=phase) == []


def test_wind_down_quotes_only_the_completing_token() -> None:
    # Unpaired UP 80 @ 0.47: only DOWN, whose total size is limited to the 80 unpaired shares.
    qs = quotes(snap(), fv(0.5), inv_with(up=80), phase=WD)
    assert view(qs) == [("DOWN", 0.49, 80.0)]
    # Mirror: unpaired DOWN 30.5 -> UP only, 30.5 shares (cap 1 - 0.01 - 0.52 = 0.47 < touch 0.49).
    qs = quotes(snap(), fv(0.5), inv_with(down=30.5), phase=WD)
    assert view(qs) == [("UP", 0.47, 30.5)]
    # Balanced or flat inventory: nothing to complete, no directional view -> no quotes.
    assert quotes(snap(), fv(0.5), inv_with(up=80, down=80), phase=WD) == []
    assert quotes(snap(), fv(0.5), inv_with(), phase=WD) == []


def test_wind_down_quotes_the_favoured_token_within_the_directional_budget() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0))
    # Flat, target +100: UP is favoured (cap max(0.495, 0.5 - 0) = 0.5 -> touch 0.49 binds? the
    # book bid is 0.50 so the top level is 0.50), room limited to the 100 shares still wanted.
    s = snap(up=(0.50, 0.52), down=(0.48, 0.50))
    qs = quotes(s, fv(0.5), inv_with(), phase=WD, cfg=cfg, target=100.0)
    assert [(q.token, q.price) for q in qs] == [(UP, 0.50)]
    assert sum(q.size for q in qs) == 100.0  # 100 + 70 + 49 would be 219: room = 100
    # Once the target is reached the favoured token is no longer quoted in wind-down.
    assert quotes(s, fv(0.5), inv_with(up=100), phase=WD, cfg=cfg, target=100.0) == []
    # Without a (valid, enabled) directional view there is no favoured token.
    assert quotes(s, fv(0.5, valid=False), inv_with(), phase=WD, cfg=cfg, target=100.0) == []


def test_invalid_model_quotes_only_completing_tokens() -> None:
    invalid = fv(0.5, valid=False)
    assert quotes(snap(), invalid, inv_with()) == []  # flat: nothing to complete
    qs = quotes(snap(), invalid, inv_with(up=100))  # unpaired UP: DOWN at the pair cap
    assert {q.token for q in qs} == {DOWN}
    assert cap(UP, invalid, inv_with(up=100)) is None
    assert cap(DOWN, invalid, inv_with(up=100)) == pytest.approx(0.52)


@pytest.mark.parametrize(
    ("up", "down"),
    [
        ((None, 0.51), (0.49, 0.51)),  # empty UP bids
        ((0.49, None), (0.49, 0.51)),  # empty UP asks
        ((0.49, 0.51), (None, None)),  # empty DOWN book
        ((0.51, 0.49), (0.49, 0.51)),  # crossed UP
        ((0.50, 0.50), (0.49, 0.51)),  # locked UP
        ((0.46, 0.53), (0.47, 0.54)),  # 7 ticks wide > 6
    ],
)
def test_no_quotes_on_bad_books(
    up: tuple[float | None, float | None], down: tuple[float | None, float | None]
) -> None:
    assert quotes(snap(up=up, down=down), fv(0.5), inv_with()) == []


def test_book_exactly_at_the_spread_limit_still_quotes() -> None:
    qs = quotes(snap(up=(0.47, 0.53), down=(0.47, 0.53)), fv(0.5), inv_with())
    assert qs  # 6 ticks wide == max_spread_ticks_to_quote
    tight = replace(CFG, risk=RiskConfig(max_spread_ticks_to_quote=1))
    assert quotes(snap(up=(0.47, 0.53), down=(0.47, 0.53)), fv(0.5), inv_with(), cfg=tight) == []


# --------------------------------------------------------------------------- budget & min size


def test_cash_budget_is_allocated_sequentially_across_levels() -> None:
    # Budget 60: UP L0 100 @ 0.49 costs 49.0 -> 11.0 left; UP L1 at 0.48: 11 / 0.48 = 22.9166
    # -> 22.91 (cost 10.9968); 0.0032 left: nothing else fits (and DOWN gets nothing).
    qs = quotes(snap(), fv(0.5), inv_with(), budget=60.0)
    assert view(qs) == [("UP", 0.49, 100.0), ("UP", 0.48, 22.91)]
    assert sum(q.price * q.size for q in qs) <= 60.0
    # Budget 49 buys exactly UP L0.
    assert view(quotes(snap(), fv(0.5), inv_with(), budget=49.0)) == [("UP", 0.49, 100.0)]
    # Budget 100 = 49 + 70*0.48 (33.6) + 49*0.47 (23.03) = 105.63 would be needed for all UP:
    # UP L0 49, L1 33.6, L2: 17.4 / 0.47 = 37.02 -> 37.02. Nothing is left for DOWN.
    qs = quotes(snap(), fv(0.5), inv_with(), budget=100.0)
    assert view(qs) == [("UP", 0.49, 100.0), ("UP", 0.48, 70.0), ("UP", 0.47, 37.02)]


def test_zero_or_tiny_budget_yields_no_quotes() -> None:
    assert quotes(snap(), fv(0.5), inv_with(), budget=0.0) == []
    # min_order_size is 5: 5 shares cost 2.45 at 0.49, 2.40 at 0.48 and 2.35 at 0.47, and a
    # cheaper deeper level can afford the minimum when the touch level cannot.
    assert quotes(snap(), fv(0.5), inv_with(), budget=2.3) == []
    assert view(quotes(snap(), fv(0.5), inv_with(), budget=2.35)) == [("UP", 0.47, 5.0)]
    assert view(quotes(snap(), fv(0.5), inv_with(), budget=2.4)) == [("UP", 0.48, 5.0)]
    assert view(quotes(snap(), fv(0.5), inv_with(), budget=2.45)) == [("UP", 0.49, 5.0)]


def test_orders_below_min_order_size_are_dropped() -> None:
    big_min = MarketSpec("m", "BTC", 1000.0, 1900.0, min_order_size=60.0)
    qs = quotes(snap(market=big_min), fv(0.5), inv_with())
    # Level sizes 100 / 70 / 49: the 49 is below 60 and dropped.
    assert [q.size for q in qs if q.token is UP] == [100.0, 70.0]
    tiny_clip = quotes(snap(), fv(0.5), inv_with(), clip=6.0)
    # 6, 4.2, 2.94 -> only the first reaches min size 5.
    assert [q.size for q in tiny_clip if q.token is UP] == [6.0]


def test_maker_fee_reduces_the_cap_and_is_charged_to_the_budget() -> None:
    cfg = replace(CFG, fees=FeeConfig(maker_fee_rate=0.02))
    # cap = (0.5 - 0.005) / 1.02 = 0.485294 -> top level floor(0.4853) = 0.48; level 1 would be
    # min(0.4853, 0.48) = 0.48 again (duplicate, skipped); level 2 is 0.47.
    assert cap(UP, fv(0.5), inv_with(), cfg=cfg) == pytest.approx(0.495 / 1.02)
    qs = quotes(snap(), fv(0.5), inv_with(), cfg=cfg)
    assert [q.price for q in qs if q.token is UP] == [0.48, 0.47]
    # The pair cap shrinks too: (1 - 0.01 - 0.47) / 1.02.
    assert cap(DOWN, fv(0.5), inv_with(up=100), cfg=cfg) == pytest.approx(0.52 / 1.02)
    # The cash needed per level includes the fee: 100 shares @ 0.48 cost 48 * 1.02 = 48.96.
    qs = quotes(snap(), fv(0.5), inv_with(), cfg=cfg, budget=48.96)
    assert view(qs) == [("UP", 0.48, 100.0)]
    # 48.95 / (0.48 * 1.02 = 0.4896 per share) = 99.9796 -> 99.97 shares (cost 48.9451).
    qs = quotes(snap(), fv(0.5), inv_with(), cfg=cfg, budget=48.95)
    assert view(qs) == [("UP", 0.48, 99.97)]


def test_maker_rebate_never_raises_the_cap() -> None:
    cfg = replace(CFG, fees=FeeConfig(maker_rebate_rate=0.01))
    assert cap(UP, fv(0.5), inv_with(), cfg=cfg) == pytest.approx(0.495)
    assert cap(DOWN, fv(0.5), inv_with(up=100), cfg=cfg) == pytest.approx(0.52)


# --------------------------------------------------------------------------- directional tilt


def test_favoured_cap_rises_to_fair_minus_min_edge_when_that_is_higher() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0))
    # Flat, target +100, fair 0.5: UP cap = max(0.495, 0.5 - 0.0) = 0.5.
    assert cap(UP, fv(0.5), inv_with(), cfg=cfg, target=100.0) == pytest.approx(0.5)
    # The unfavoured token keeps the balanced cap.
    assert cap(DOWN, fv(0.5), inv_with(), cfg=cfg, target=100.0) == pytest.approx(0.495)
    # With the default min_edge 0.03 the "raise" would be a cut, so the balanced cap stays.
    assert cap(UP, fv(0.5), inv_with(), target=100.0) == pytest.approx(0.495)
    # Symmetric for a negative target.
    assert cap(DOWN, fv(0.5), inv_with(), cfg=cfg, target=-100.0) == pytest.approx(0.5)
    assert cap(UP, fv(0.5), inv_with(), cfg=cfg, target=-100.0) == pytest.approx(0.495)


def test_tilt_moves_the_ladder_and_skews_the_unfavoured_side() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0))
    s = snap(up=(0.50, 0.52), down=(0.48, 0.50))
    # UP favoured: top level reaches the 0.50 touch (cap 0.5). DOWN is "heavy relative to the
    # target" by 100 shares -> 1 tick down: min(0.495, 0.48) - 0.01 = 0.47, 0.46, 0.45.
    qs = quotes(s, fv(0.5), inv_with(), cfg=cfg, target=100.0)
    assert view(qs) == [
        ("UP", 0.50, 100.0),
        ("UP", 0.49, 70.0),
        ("UP", 0.48, 49.0),
        ("DOWN", 0.47, 100.0),
        ("DOWN", 0.46, 70.0),
        ("DOWN", 0.45, 49.0),
    ]
    # Same book without a tilt: UP top is the cap floor(0.495) = 0.49 (L1 duplicates, dropped).
    plain = quotes(s, fv(0.5), inv_with(), cfg=cfg, target=0.0)
    assert view(plain)[:2] == [("UP", 0.49, 100.0), ("UP", 0.48, 49.0)]


def test_unfavoured_size_scales_down_with_progress_and_vanishes_at_target() -> None:
    # Target +100. Holding UP 50 @ 0.47 (net +50): progress 0.5 -> DOWN sizes halve: 50, 35, 24.5.
    # DOWN completes (unpaired UP) at cap 0.52; skew = (target - net)/100 = 0.5 tick:
    # L0 min(0.52, 0.49) - 0.005 = 0.485 -> 0.48, then 0.475 -> 0.47, 0.465 -> 0.46.
    qs = quotes(snap(), fv(0.5), inv_with(up=50), target=100.0)
    assert [(q.price, q.size) for q in qs if q.token is DOWN] == [
        (0.48, 50.0),
        (0.47, 35.0),
        (0.46, 24.5),
    ]
    # UP is still favoured (net 50 < 100): not skewed (excess = net - target = -50).
    assert [(q.price, q.size) for q in qs if q.token is UP] == [
        (0.49, 100.0),
        (0.48, 70.0),
        (0.47, 49.0),
    ]
    # Target reached (net +100 == target): DOWN removed entirely, UP no longer favoured.
    reached = quotes(snap(), fv(0.5), inv_with(up=100), target=100.0)
    assert {q.token for q in reached} == {UP}
    assert [(q.price, q.size) for q in reached] == [(0.49, 100.0), (0.48, 70.0), (0.47, 49.0)]


def test_tilt_is_mirrored_for_a_down_target() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0))
    s = snap(up=(0.48, 0.50), down=(0.50, 0.52))
    qs = quotes(s, fv(0.5), inv_with(), cfg=cfg, target=-100.0)
    assert view(qs) == [
        ("UP", 0.47, 100.0),
        ("UP", 0.46, 70.0),
        ("UP", 0.45, 49.0),
        ("DOWN", 0.50, 100.0),
        ("DOWN", 0.49, 70.0),
        ("DOWN", 0.48, 49.0),
    ]
    reached = quotes(s, fv(0.5), inv_with(down=100, down_px=0.52), cfg=cfg, target=-100.0)
    assert {q.token for q in reached} == {DOWN}


def test_directional_allowance_extends_the_hard_stop_in_the_favoured_direction() -> None:
    # Holding 300 UP (net +300 == max_net): without a target UP is stopped.
    assert {q.token for q in quotes(snap(), fv(0.5), inv_with(up=300))} == {DOWN}
    # With target +200 the allowance is 200: UP room = 300 + 200 - 300 = 200 and no skew for the
    # 200 deliberate shares: excess = 300 - 200 = 100 -> 1 tick.
    qs = quotes(snap(), fv(0.5), inv_with(up=300), target=200.0)
    ups = [q for q in qs if q.token is UP]
    assert sum(q.size for q in ups) == 200.0
    assert ups[0].price == 0.48  # 0.49 - 1 tick
    # The allowance never shrinks the limit for the OTHER direction: target -200 and net +250
    # leaves UP room 50 (not 300 - 200 - 250).
    qs = quotes(snap(), fv(0.5), inv_with(up=250), target=-200.0)
    assert sum(q.size for q in qs if q.token is UP) == 50.0


def test_target_is_clamped_and_ignored_when_disabled_or_invalid() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0, max_directional_shares=100.0))
    s = snap(up=(0.50, 0.52), down=(0.48, 0.50))
    # target 5000 behaves like 100 (clamped to max_directional_shares).
    assert quotes(s, fv(0.5), inv_with(), cfg=cfg, target=5000.0) == quotes(
        s, fv(0.5), inv_with(), cfg=cfg, target=100.0
    )
    off = replace(CFG, directional=DirectionalConfig(enabled=False, min_edge=0.0))
    base = quotes(s, fv(0.5), inv_with(), cfg=off, target=0.0)
    assert quotes(s, fv(0.5), inv_with(), cfg=off, target=150.0) == base
    invalid = fv(0.5, valid=False)
    assert quotes(s, invalid, inv_with(), cfg=cfg, target=100.0) == []


def test_completing_token_keeps_the_strict_pair_cap_under_a_tilt() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0))
    # Hold unpaired DOWN 100 @ 0.52; a (misguided) UP target must not lift the UP cap above the
    # pair cap 1 - 0.01 - 0.52 = 0.47, although fair_UP - min_edge = 0.5.
    assert cap(UP, fv(0.5), inv_with(down=100, down_px=0.52), cfg=cfg, target=100.0) == (
        pytest.approx(0.47)
    )


# ------------------------------------------------------------------------- skip, validation, plans


def test_skip_tokens_are_not_quoted_and_use_no_budget() -> None:
    qs = quotes(snap(), fv(0.5), inv_with(), skip=(UP,))
    assert {q.token for q in qs} == {DOWN}
    # With a budget that only covers one ladder level, skipping UP lets DOWN use it.
    assert view(quotes(snap(), fv(0.5), inv_with(), budget=49.0, skip=(UP,))) == [
        ("DOWN", 0.49, 100.0)
    ]
    assert quotes(snap(), fv(0.5), inv_with(), skip=(UP, DOWN)) == []


def test_compute_quotes_validates_its_inputs() -> None:
    inv = inv_with()
    with pytest.raises(ValueError, match="clip_shares"):
        quotes(snap(), fv(), inv, clip=0.0)
    with pytest.raises(ValueError, match="clip_shares"):
        quotes(snap(), fv(), inv, clip=math.nan)
    with pytest.raises(ValueError, match="cash_budget"):
        quotes(snap(), fv(), inv, budget=-1.0)
    with pytest.raises(ValueError, match="cash_budget"):
        quotes(snap(), fv(), inv, budget=math.inf)
    with pytest.raises(ValueError, match="target_net"):
        quotes(snap(), fv(), inv, target=math.nan)
    with pytest.raises(ValueError, match="target_net"):
        quotes(snap(), fv(), inv, phase=Phase.DONE, target=math.inf)
    with pytest.raises(ValueError, match="target_net"):
        cap(UP, fv(), inv, target=math.nan)


def test_plan_tokens_exposes_ceiling_room_and_scale() -> None:
    plans = plan_tokens(
        snap=snap(), fair=fv(0.5), inv=inv_with(up=150), phase=ACC, cfg=CFG, target_net=0.0
    )
    up, down = plans[UP], plans[DOWN]
    assert up.cap == pytest.approx(0.495)
    assert up.ceiling == 0.48  # floor(min(0.495, 0.51 - 0.01) - 1.5 ticks) = floor(0.48)
    assert up.skew_ticks == pytest.approx(1.5)
    assert up.room == pytest.approx(150.0)  # 300 - 150 net room
    assert up.size_scale == 1.0
    assert down.cap == pytest.approx(0.52)
    assert down.ceiling == 0.50  # floor(min(0.52, 0.51 - 0.01)) = 0.50
    assert down.skew_ticks == 0.0
    assert down.room == pytest.approx(450.0)  # 300 - (0 - 150)
    # In DONE every plan is blocked.
    done = plan_tokens(
        snap=snap(), fair=fv(0.5), inv=inv_with(), phase=Phase.DONE, cfg=CFG, target_net=0.0
    )
    assert all(p.ceiling is None and p.room == 0.0 for p in done.values())


def test_quotes_are_deterministic_and_ordered() -> None:
    inv = inv_with(up=120, down=40)
    a = quotes(snap(), fv(0.55), inv, target=30.0)
    b = quotes(snap(), fv(0.55), inv, target=30.0)
    assert a == b
    tokens = [q.token for q in a]
    assert tokens == sorted(tokens, key=lambda t: 0 if t is UP else 1)
    for token in (UP, DOWN):
        prices = [q.price for q in a if q.token is token]
        assert prices == sorted(prices, reverse=True)


# --------------------------------------------------------------------------- seeded property test


def _random_cfg(rng: random.Random) -> BotConfig:
    max_net = rng.choice([0.0, 100.0, 300.0, 300.0, 600.0])
    return BotConfig(
        fees=FeeConfig(maker_fee_rate=rng.choice([0.0, 0.0, 0.01, 0.03])),
        sizing=SizingConfig(
            ladder_levels=rng.randint(1, 5),
            ladder_spacing_ticks=rng.randint(1, 3),
            ladder_size_decay=rng.choice([0.3, 0.5, 0.7, 1.0]),
        ),
        pair=PairConfig(
            target_margin=rng.choice([0.0, 0.005, 0.01, 0.03]),
            max_net_imbalance_shares=max_net,
            max_inventory_per_side_shares=max_net + rng.choice([0.0, 200.0, 1200.0]),
            skew_ticks_per_100_shares=rng.choice([0.0, 0.5, 1.0, 3.0]),
        ),
        directional=DirectionalConfig(
            enabled=rng.random() < 0.7,
            min_edge=rng.choice([0.0, 0.01, 0.03]),
            max_directional_shares=rng.choice([0.0, 100.0, 200.0]),
        ),
        risk=RiskConfig(max_spread_ticks_to_quote=rng.choice([2, 6, 6])),
    )


def _random_snapshot(rng: random.Random) -> MarketSnapshot:
    tick = rng.choice([0.01, 0.01, 0.001])
    market = MarketSpec(
        "m1", "BTC", 1000.0, 1900.0, tick_size=tick, min_order_size=rng.choice([0.5, 5.0, 25.0])
    )
    n_ticks = round(1.0 / tick)
    mid_i = rng.randint(2, n_ticks - 2)
    half_lo = rng.choice([0, 0, 1])  # bid sits this many ticks below mid_i
    half_hi = rng.choice([1, 1, 2, 3, 6])  # ask sits this many ticks above
    bid_i, ask_i = max(0, mid_i - half_lo), min(n_ticks, mid_i + half_hi)
    # Occasionally corrupt the book: crossed, empty or independent DOWN prices.
    roll = rng.random()
    up_bid: float | None = round(bid_i * tick, 6)
    up_ask: float | None = round(ask_i * tick, 6)
    if roll < 0.04:
        up_bid, up_ask = up_ask, up_bid
    elif roll < 0.07:
        up_bid = None
    elif roll < 0.10:
        up_ask = None
    dn_bid = None if up_ask is None else round(1.0 - up_ask, 6)
    dn_ask = None if up_bid is None else round(1.0 - up_bid, 6)
    if rng.random() < 0.05:
        dn_bid, dn_ask = round(rng.uniform(0.02, 0.5), 2), round(rng.uniform(0.5, 0.98), 2)
    return snap(up=(up_bid, up_ask), down=(dn_bid, dn_ask), market=market)


def _random_inventory(rng: random.Random, max_side: float) -> MarketInventory:
    inv = MarketInventory("m1")
    kind = rng.random()
    if kind < 0.3:
        up = down = 0.0  # flat
    elif kind < 0.45:
        up = down = rng.uniform(1.0, max_side)  # balanced
    else:
        up = rng.choice([0.0, rng.uniform(0.0, max_side * 1.05)])
        down = rng.choice([0.0, rng.uniform(0.0, max_side * 1.05)])
    for token, qty in ((UP, up), (DOWN, down)):
        if qty > 0.005:
            px = round(rng.uniform(0.05, 0.95), 2)
            inv.apply_fill(Fill("f", "o", "m1", token, Side.BUY, px, qty, 0.0, True, 1.0))
    return inv


def _effective_target(cfg: BotConfig, fair: FairValue, target: float) -> float:
    if not cfg.directional.enabled or not fair.valid:
        return 0.0
    lim = cfg.directional.max_directional_shares
    return max(-lim, min(lim, target))


def test_randomised_states_respect_every_quoting_invariant() -> None:
    rng = random.Random(20240607)
    counts = {
        "states": 0,
        "with_quotes": 0,
        "completing_quotes": 0,
        "wind_down_quotes": 0,
        "budget_limited": 0,
        "favoured_quotes": 0,
        "empty_blocked": 0,
    }
    for _ in range(3000):
        cfg = _random_cfg(rng)
        s = _random_snapshot(rng)
        inv = _random_inventory(rng, cfg.pair.max_inventory_per_side_shares)
        fair = fv(rng.uniform(0.02, 0.98), valid=rng.random() < 0.85)
        phase = rng.choice([ACC] * 5 + [WD] * 3 + [Phase.WARMUP, Phase.FLATTEN, Phase.DONE])
        clip = rng.choice([5.0, 20.0, 100.0, 400.0])
        budget = rng.choice([0.0, rng.uniform(0.0, 120.0), rng.uniform(0.0, 120.0), 1e5, 1e5])
        target = rng.choice([0.0, 0.0, rng.uniform(-400.0, 400.0)])
        counts["states"] += 1

        fees = FeeModel(cfg.fees)
        out = compute_quotes(
            snap=s,
            fair=fair,
            inv=inv,
            phase=phase,
            cfg=cfg,
            fees=fees,
            clip_shares=clip,
            cash_budget=budget,
            target_net=target,
        )
        assert out == compute_quotes(  # deterministic
            snap=s, fair=fair, inv=inv, phase=phase, cfg=cfg, fees=fees,
            clip_shares=clip, cash_budget=budget, target_net=target,
        )  # fmt: skip

        quotable = books_quotable(s, cfg.risk.max_spread_ticks_to_quote)
        if phase not in (ACC, WD) or not quotable:
            assert out == []  # no quotes in WARMUP/FLATTEN/DONE or on a bad book
            counts["empty_blocked"] += 1
            continue

        tick, min_size = s.market.tick_size, s.market.min_order_size
        eff = _effective_target(cfg, fair, target)
        if not (cfg.directional.enabled and fair.valid):
            ref = compute_quotes(
                snap=s, fair=fair, inv=inv, phase=phase, cfg=cfg, fees=fees,
                clip_shares=clip, cash_budget=budget, target_net=0.0,
            )  # fmt: skip
            assert out == ref  # the tilt is off when disabled / invalid
        if out:
            counts["with_quotes"] += 1
            if phase is WD:
                counts["wind_down_quotes"] += 1

        # ---- per-quote invariants
        rate = cfg.fees.maker_fee_rate
        spent = 0.0
        for q in out:
            best_ask = s.book(q.token).best_ask
            assert best_ask is not None
            assert 0.0 < q.price < 1.0
            assert abs(q.price / tick - round(q.price / tick)) < 1e-6  # tick aligned
            assert q.price < best_ask  # post-only: never crosses
            assert q.price <= best_ask - tick + 1e-9  # ... not even to the touch of the ask
            assert abs(q.size * 100 - round(q.size * 100)) < 1e-6  # 0.01 lot
            assert min_size - 1e-9 <= q.size <= clip + 1e-9
            spent += q.price * q.size * (1.0 + rate)
        assert spent <= budget + 1e-9  # the budget is never exceeded (fees included)
        if out and spent > budget - 1.0 and budget < 120.0:
            counts["budget_limited"] += 1
        order = [0 if q.token is UP else 1 for q in out]
        assert order == sorted(order)  # UP before DOWN

        plans = plan_tokens(snap=s, fair=fair, inv=inv, phase=phase, cfg=cfg, target_net=target)
        for token in (UP, DOWN):
            lst = [q for q in out if q.token is token]
            prices = [q.price for q in lst]
            assert prices == sorted(prices, reverse=True)  # level 0 first
            assert len(set(prices)) == len(prices)  # no duplicate ladder price
            opp = token.opposite
            sign = 1.0 if token is UP else -1.0
            held, other = inv.qty[token], inv.qty[opp]
            completing = other > held + EPS
            avg_other = inv.avg_cost(opp)
            plan = plans[token]
            bid_cap = max_bid_for_token(token=token, fair=fair, inv=inv, cfg=cfg, target_net=target)
            if not lst:
                continue
            counts["completing_quotes"] += int(completing)
            assert bid_cap is not None
            assert plan.ceiling is not None
            for q in lst:
                assert q.price <= bid_cap + 1e-9  # never above the cap
                assert q.price <= plan.ceiling + 1e-9
                if completing:
                    # DESIGN invariant 8: the pair cap.
                    assert avg_other is not None
                    assert q.price + avg_other <= 1.0 - cfg.pair.target_margin + EPS
            if not fair.valid:
                assert completing  # invalid model quotes only to complete pairs
            # ---- room: a fully filled ladder never breaches the inventory limits
            total = sum(q.size for q in lst)
            allowance = max(0.0, sign * eff)
            assert held + total <= cfg.pair.max_inventory_per_side_shares + 1e-9
            assert (held - other) + total <= cfg.pair.max_net_imbalance_shares + allowance + 1e-9
            net_dir = held - other  # net exposure in this token's direction
            favoured = eff * sign > EPS and net_dir < sign * eff - EPS
            counts["favoured_quotes"] += int(favoured)
            if phase is WD:
                assert completing or favoured  # wind-down: only completing / favoured tokens
                if completing and not favoured:
                    assert total <= (other - held) + 1e-9  # no more than completes the pairs
    # The property test must not be vacuous.
    assert counts["with_quotes"] > 500
    assert counts["completing_quotes"] > 250
    assert counts["wind_down_quotes"] > 100
    assert counts["budget_limited"] > 80
    assert counts["favoured_quotes"] > 50
    assert counts["empty_blocked"] > 500
