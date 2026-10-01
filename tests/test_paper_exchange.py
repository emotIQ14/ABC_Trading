"""Tests for abc_trading.exchange (Exchange protocol, PaperExchange fill model and accounting).

Every expected number is worked out by hand in a comment. Default test fees are all zero so the
arithmetic stays visible; the tests that exercise fees say so explicitly.
"""

from __future__ import annotations

import math
import random
from typing import TypedDict

import pytest

from abc_trading.config import BotConfig, FeeConfig, PaperExchangeConfig
from abc_trading.exchange import Exchange, PaperExchange
from abc_trading.inventory import Portfolio
from abc_trading.types import (
    BookSnapshot,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    OrderRequest,
    Outcome,
    Side,
    TimeInForce,
    Trade,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
BUY, SELL = Side.BUY, Side.SELL
POST, IOC = TimeInForce.POST_ONLY, TimeInForce.IOC
M1, M2 = "m1", "m2"
INITIAL = 10_000.0


def approx(x: float, abs_tol: float = 1e-9) -> object:
    return pytest.approx(x, abs=abs_tol, rel=1e-12)


# --------------------------------------------------------------------------- builders


def cfg_with(
    *,
    latency: int = 0,
    queue: float = 1.0,
    fill: float = 1.0,
    slip: int = 0,
    enforce_min: bool = True,
    cash: float = INITIAL,
    maker_fee: float = 0.0,
    rebate: float = 0.0,
    taker_rate: float = 0.0,
) -> BotConfig:
    return BotConfig(
        fees=FeeConfig(
            maker_fee_rate=maker_fee, maker_rebate_rate=rebate, taker_fee_rate=taker_rate
        ),
        exchange=PaperExchangeConfig(
            initial_cash=cash,
            latency_ticks=latency,
            queue_ahead_fraction=queue,
            trade_fill_fraction=fill,
            taker_slippage_ticks=slip,
            enforce_min_order_size=enforce_min,
        ),
    )


class Book(TypedDict):
    """Keyword arguments of ``snap`` that describe the UP book."""

    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


def levels(pairs: tuple[tuple[float, float], ...]) -> tuple[Level, ...]:
    return tuple(Level(p, s) for p, s in pairs)


def mirrored(pairs: tuple[tuple[float, float], ...]) -> tuple[Level, ...]:
    """The other token's ladder: same sizes at ``1 - price`` (best-first order is preserved)."""
    return tuple(Level(round(1.0 - p, 6), s) for p, s in pairs)


def snap(
    ts: float,
    *,
    bids: tuple[tuple[float, float], ...] = ((0.46, 100.0),),
    asks: tuple[tuple[float, float], ...] = ((0.52, 100.0),),
    trades: tuple[Trade, ...] = (),
    mid: str = M1,
    tick: float = 0.01,
    min_size: float = 5.0,
) -> MarketSnapshot:
    """UP book from ``bids``/``asks`` (best first); the DOWN book is its exact mirror."""
    return MarketSnapshot(
        ts=ts,
        market=MarketSpec(
            market_id=mid,
            asset="BTC",
            start_ts=0.0,
            end_ts=900.0,
            tick_size=tick,
            min_order_size=min_size,
        ),
        up_book=BookSnapshot(UP, bids=levels(bids), asks=levels(asks)),
        down_book=BookSnapshot(DOWN, bids=mirrored(asks), asks=mirrored(bids)),
        trades=trades,
    )


def tr(ts: float, token: Outcome, price: float, size: float, aggressor: Side) -> Trade:
    return Trade(ts=ts, token=token, price=price, size=size, aggressor=aggressor)


def prints(ts: float, up_price: float, size: float, up_aggressor: Side) -> tuple[Trade, Trade]:
    """A UP print and its mirrored DOWN print (opposite aggressor at ``1 - price``)."""
    return (
        tr(ts, UP, up_price, size, up_aggressor),
        tr(ts, DOWN, round(1.0 - up_price, 6), size, up_aggressor.opposite),
    )


def req(
    side: Side,
    price: float,
    size: float,
    token: Outcome = UP,
    tif: TimeInForce = POST,
    mid: str = M1,
) -> OrderRequest:
    return OrderRequest(
        client_id="c", market_id=mid, token=token, side=side, price=price, size=size, tif=tif
    )


def submit_ok(ex: PaperExchange, r: OrderRequest, ts: float = 0.0) -> str:
    res = ex.submit(r, ts)
    assert res.ok, res.reason
    assert res.order_id is not None
    return res.order_id


def buy_now(ex: PaperExchange, token: Outcome, size: float, price: float, ts: float = 0.0) -> None:
    """Acquire a position with a latency-0 IOC buy (needs the last known book to offer it)."""
    submit_ok(ex, req(BUY, price, size, token, IOC), ts)
    fills = ex.drain_fills()
    assert sum(f.size for f in fills) == approx(size)


# --------------------------------------------------------------------------- protocol / config


def test_paper_exchange_implements_protocol() -> None:
    ex = PaperExchange(cfg_with())
    assert isinstance(ex, Exchange)
    assert ex.balance() == INITIAL
    assert ex.position(M1, UP) == 0.0
    assert ex.open_orders() == []
    assert ex.reserved_cash == 0.0


def test_initial_stats_are_zero() -> None:
    ex = PaperExchange(cfg_with())
    for key in (
        "submitted",
        "accepted",
        "cancelled",
        "filled_maker",
        "filled_taker",
        "post_only_cancels",
        "merges",
        "merge_rejects",
        "settled",
        "volume_maker",
        "volume_taker",
    ):
        assert ex.stats[key] == 0


@pytest.mark.parametrize(
    "bad",
    [
        PaperExchangeConfig(latency_ticks=-1),
        PaperExchangeConfig(trade_fill_fraction=1.5),
        PaperExchangeConfig(queue_ahead_fraction=-0.1),
        PaperExchangeConfig(taker_slippage_ticks=-1),
        PaperExchangeConfig(initial_cash=float("nan")),
    ],
)
def test_invalid_exchange_config_is_rejected(bad: PaperExchangeConfig) -> None:
    with pytest.raises(ValueError):
        PaperExchange(BotConfig(exchange=bad))


def test_fee_rates_above_one_are_rejected() -> None:
    with pytest.raises(ValueError):
        PaperExchange(BotConfig(fees=FeeConfig(taker_fee_rate=1.5)))
    with pytest.raises(ValueError):
        PaperExchange(BotConfig(fees=FeeConfig(maker_fee_rate=1.5)))


# --------------------------------------------------------------------------- submit validation


@pytest.mark.parametrize(
    ("price", "size", "reason"),
    [
        (0.0, 10.0, "invalid_price"),
        (1.0, 10.0, "invalid_price"),
        (-0.1, 10.0, "invalid_price"),
        (1.2, 10.0, "invalid_price"),
        (float("nan"), 10.0, "invalid_price"),
        (float("inf"), 10.0, "invalid_price"),
        (0.47, 0.0, "invalid_size"),
        (0.47, -5.0, "invalid_size"),
        (0.47, float("nan"), "invalid_size"),
        (0.47, float("inf"), "invalid_size"),
        (0.475, 10.0, "bad_tick"),  # half a tick
        (0.4701, 10.0, "bad_tick"),
        (0.47, 4.99, "min_size"),  # fallback min size is 5
    ],
)
def test_submit_rejects_invalid_orders(price: float, size: float, reason: str) -> None:
    ex = PaperExchange(cfg_with())
    res = ex.submit(req(BUY, price, size), 0.0)
    assert not res.ok
    assert res.order_id is None
    assert res.reason == reason
    assert ex.open_orders() == []
    assert ex.stats["submitted"] == 1
    assert ex.stats["accepted"] == 0
    assert ex.stats[f"rejected_{reason}"] == 1
    assert ex.reserved_cash == 0.0
    # a rejected order does not consume an id
    assert submit_ok(ex, req(BUY, 0.47, 10.0)) == "o1"


@pytest.mark.parametrize("price", [0.01, 0.99, 0.5])
def test_submit_accepts_prices_strictly_inside_0_1(price: float) -> None:
    ex = PaperExchange(cfg_with())
    assert ex.submit(req(BUY, price, 5.0), 0.0).ok  # exactly the min size is fine


def test_tick_and_min_size_come_from_the_market_spec() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, tick=0.001, min_size=20.0, bids=((0.460, 100.0),), asks=((0.520, 100.0),)))
    assert ex.submit(req(BUY, 0.475, 20.0), 0.0).ok  # tick 0.001 allows 0.475
    assert ex.submit(req(BUY, 0.4755, 20.0), 0.0).reason == "bad_tick"
    assert ex.submit(req(BUY, 0.475, 19.0), 0.0).reason == "min_size"


def test_min_size_not_enforced_when_disabled() -> None:
    ex = PaperExchange(cfg_with(enforce_min=False))
    assert ex.submit(req(BUY, 0.47, 1.0), 0.0).ok
    assert ex.submit(req(BUY, 0.47, 1e-3), 0.0).ok  # size > 0 is still required
    assert ex.submit(req(BUY, 0.47, 0.0), 0.0).reason == "invalid_size"


def test_ids_are_deterministic_and_per_exchange() -> None:
    a, b = PaperExchange(cfg_with()), PaperExchange(cfg_with())
    assert [submit_ok(a, req(BUY, 0.40, 5.0)) for _ in range(3)] == ["o1", "o2", "o3"]
    assert submit_ok(b, req(BUY, 0.40, 5.0)) == "o1"


# --------------------------------------------------------------------------- reservations


def test_reserved_cash_is_sum_of_buy_notional_including_in_flight() -> None:
    ex = PaperExchange(cfg_with(latency=1, cash=1000.0))
    ex.process(snap(0.0))
    o1 = submit_ok(ex, req(BUY, 0.47, 100.0))  # 0.47 * 100 = 47
    submit_ok(ex, req(BUY, 0.40, 50.0, DOWN))  # 0.40 * 50 = 20
    assert ex.reserved_cash == approx(67.0)
    assert ex.balance() == 1000.0  # reservations are not deducted from cash
    assert all(not o.live for o in ex.open_orders())  # in flight orders reserve too
    assert ex.cancel(o1, 0.0)
    assert ex.reserved_cash == approx(20.0)


def test_ioc_buy_in_flight_reserves_notional_only() -> None:
    ex = PaperExchange(cfg_with(latency=1, taker_rate=0.25))
    ex.process(snap(0.0))
    submit_ok(ex, req(BUY, 0.50, 100.0, UP, IOC))
    assert ex.reserved_cash == approx(50.0)  # price * remaining, no fee


def test_cash_limits_resting_bids() -> None:
    ex = PaperExchange(cfg_with(cash=100.0))
    submit_ok(ex, req(BUY, 0.50, 100.0))  # reserves 50, free 50
    res = ex.submit(req(BUY, 0.51, 100.0), 0.0)  # needs 51 > 50 free
    assert (res.ok, res.reason) == (False, "insufficient_cash")
    submit_ok(ex, req(BUY, 0.50, 100.0))  # needs exactly 50: fine
    assert ex.reserved_cash == approx(100.0)
    assert ex.submit(req(BUY, 0.40, 5.0), 0.0).reason == "insufficient_cash"  # needs 2 > 0
    assert ex.stats["rejected_insufficient_cash"] == 2


def test_positive_maker_fee_is_committed_so_cash_never_drops_below_reservations() -> None:
    ex = PaperExchange(cfg_with(cash=100.0, maker_fee=0.01))
    ex.process(snap(0.0, bids=((0.30, 100.0),), asks=((0.60, 100.0),)))
    # 200 @ 0.50 needs 100 + 0.01 * 100 = 101 > 100
    assert ex.submit(req(BUY, 0.50, 200.0), 0.0).reason == "insufficient_cash"
    # 198 @ 0.50 needs 99 + 0.99 = 99.99 <= 100
    submit_ok(ex, req(BUY, 0.50, 198.0))
    assert ex.reserved_cash == approx(99.0)
    fills = ex.process(
        snap(
            1.0,
            bids=((0.30, 100.0),),
            asks=((0.60, 100.0),),
            trades=(tr(1.0, UP, 0.45, 500.0, SELL),),
        )
    )
    assert len(fills) == 1 and fills[0].size == approx(198.0) and fills[0].fee == approx(0.99)
    assert ex.balance() == approx(0.01)  # 100 - 99 - 0.99
    assert ex.balance() >= ex.reserved_cash == 0.0


def test_maker_fee_commitment_of_open_bids_limits_later_bids() -> None:
    # 100 @ 0.50 with a 1% maker fee commits 50 + 0.5 = 50.5. With 100.7 cash the first bid is
    # fine (100.7 >= 50.5) but the second sees only 100.7 - 50.5 = 50.2 < 50.5 and is refused;
    # ignoring the first bid's fee would wrongly accept it (50.7 >= 50.5) and then overdraw
    # cash when both fill (100.7 - 50.5 - 50.5 < 0).
    ex = PaperExchange(cfg_with(cash=100.7, maker_fee=0.01))
    ex.process(snap(0.0, bids=((0.30, 100.0),), asks=((0.60, 100.0),)))
    submit_ok(ex, req(BUY, 0.50, 100.0))
    assert ex.submit(req(BUY, 0.50, 100.0), 0.0).reason == "insufficient_cash"
    # 20 more shares fit: 50.2 >= 0.5 * 20 * 1.01 = 10.1
    submit_ok(ex, req(BUY, 0.50, 20.0))
    assert ex.reserved_cash == approx(60.0)  # reserved_cash itself stays the plain notional


def test_no_overselling() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.50, 300.0),)))
    assert ex.submit(req(BUY, 0.55, 5.0, UP, TimeInForce.POST_ONLY), 0.0).reason == (
        "post_only_crosses"
    )
    assert ex.submit(req(SELL, 0.60, 10.0), 0.0).reason == "insufficient_position"  # nothing held
    buy_now(ex, UP, 100.0, 0.50)
    s1 = submit_ok(ex, req(SELL, 0.60, 60.0))
    assert ex.submit(req(SELL, 0.61, 50.0), 0.0).reason == "insufficient_position"  # 40 free
    assert ex.submit(req(SELL, 0.60, 50.0, DOWN), 0.0).reason == "insufficient_position"
    submit_ok(ex, req(SELL, 0.62, 40.0))  # exactly the 40 left
    assert ex.submit(req(SELL, 0.62, 5.0), 0.0).reason == "insufficient_position"
    assert ex.cancel(s1, 0.0)
    submit_ok(ex, req(SELL, 0.62, 60.0))  # the freed 60
    assert ex.position(M1, UP) == 100.0  # reservations do not move the position


def test_ioc_sell_needs_free_position() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.50, 300.0),)))
    assert ex.submit(req(SELL, 0.40, 10.0, UP, IOC), 0.0).reason == "insufficient_position"


def test_ioc_buy_needs_worst_case_taker_fee() -> None:
    # 100 @ 0.50: notional 50, taker fee = 100 * 0.5 * 0.25 * (0.25) ** 2 = 0.78125
    ex = PaperExchange(cfg_with(latency=1, cash=50.5, taker_rate=0.25))
    ex.process(snap(0.0))
    res = ex.submit(req(BUY, 0.50, 100.0, UP, IOC), 0.0)
    assert (res.ok, res.reason) == (False, "insufficient_cash")  # 50.78125 > 50.5
    ex2 = PaperExchange(cfg_with(latency=1, cash=50.8, taker_rate=0.25))
    ex2.process(snap(0.0))
    assert ex2.submit(req(BUY, 0.50, 100.0, UP, IOC), 0.0).ok  # 50.78125 <= 50.8


# --------------------------------------------------------------------------- queue priority


def test_queue_position_drains_before_we_fill() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.47, 100.0), (0.46, 50.0)), "asks": ((0.49, 80.0),)}
    ex.process(snap(0.0, **book))
    oid = submit_ok(ex, req(BUY, 0.47, 50.0))  # queue_ahead = 1.0 * 100 shares at 0.47
    assert ex.open_orders()[0].live is True

    # print of 60 < 100 ahead: queue drains to 40, no fill
    assert ex.process(snap(1.0, trades=(tr(1.0, UP, 0.47, 60.0, SELL),), **book)) == []
    assert ex.open_orders()[0].remaining == 50.0
    # another 60: 40 drain the queue, the remaining 20 fill us at our price
    fills = ex.process(snap(2.0, trades=(tr(2.0, UP, 0.47, 60.0, SELL),), **book))
    assert len(fills) == 1
    f = fills[0]
    assert (f.order_id, f.market_id, f.token, f.side) == (oid, M1, UP, BUY)
    assert f.price == 0.47 and f.size == approx(20.0) and f.fee == 0.0 and f.is_maker
    assert f.fill_id == "f1" and f.ts == 2.0
    assert ex.open_orders()[0].remaining == approx(30.0)  # partial fill keeps it resting
    assert ex.balance() == approx(INITIAL - 9.4)  # 0.47 * 20
    assert ex.position(M1, UP) == approx(20.0)
    # queue is empty now: a 100 print fills the remaining 30 and the order vanishes
    fills = ex.process(snap(3.0, trades=(tr(3.0, UP, 0.47, 100.0, SELL),), **book))
    assert [(x.size, x.fill_id) for x in fills] == [(pytest.approx(30.0), "f2")]
    assert ex.open_orders() == []
    assert ex.balance() == approx(INITIAL - 23.5)  # 0.47 * 50
    assert ex.position(M1, UP) == approx(50.0)
    assert ex.reserved_cash == 0.0


def test_queue_ahead_fraction() -> None:
    ex = PaperExchange(cfg_with(queue=0.5))
    book: Book = {"bids": ((0.47, 100.0),), "asks": ((0.49, 80.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.47, 50.0))  # queue_ahead = 0.5 * 100 = 50
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.47, 60.0, SELL),), **book))
    assert len(fills) == 1 and fills[0].size == approx(10.0)  # 60 - 50 ahead


def test_improving_the_touch_or_resting_on_an_empty_level_has_no_queue() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.47, 100.0),), "asks": ((0.49, 80.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.48, 50.0))  # improves the touch: queue 0
    submit_ok(ex, req(BUY, 0.45, 50.0))  # level 0.45 absent from the book: queue 0
    fills = ex.process(
        snap(1.0, trades=(tr(1.0, UP, 0.48, 30.0, SELL), tr(1.0, UP, 0.45, 10.0, SELL)), **book)
    )
    # print 1 @0.48 x30: o1 (0.48) fills 30; o2 (0.45) is below the print: untouched.
    # print 2 @0.45 x10: o1 (0.48) is through-filled for 10; o2 at its price fills 0 (o1 first).
    assert [(f.order_id, f.price, f.size) for f in fills] == [
        ("o1", 0.48, pytest.approx(30.0)),
        ("o1", 0.48, pytest.approx(10.0)),
    ]


def test_through_fill_at_our_price_and_queue_is_not_reset() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.47, 100.0),), "asks": ((0.49, 80.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.47, 50.0))  # queue 100
    # a SELL print at 0.45 < 0.47 through-fills us at OUR price 0.47, not at 0.45
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 30.0, SELL),), **book))
    assert [(f.price, f.size) for f in fills] == [(0.47, pytest.approx(30.0))]
    assert ex.balance() == approx(INITIAL - 14.1)  # 0.47 * 30
    # a print above our bid never fills us
    assert ex.process(snap(2.0, trades=(tr(2.0, UP, 0.48, 500.0, SELL),), **book)) == []
    # conservative: the through-print did not clear the queue, so 100 at 0.47 only drains it
    assert ex.process(snap(3.0, trades=(tr(3.0, UP, 0.47, 100.0, SELL),), **book)) == []
    fills = ex.process(snap(4.0, trades=(tr(4.0, UP, 0.47, 15.0, SELL),), **book))
    assert fills[0].size == approx(15.0) and ex.open_orders()[0].remaining == approx(5.0)


def test_buy_aggressor_prints_never_fill_a_resting_bid() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0))
    submit_ok(ex, req(BUY, 0.50, 50.0))
    assert ex.process(snap(1.0, trades=(tr(1.0, UP, 0.40, 500.0, BUY),))) == []
    # a print on the other token does not fill a UP order either
    assert ex.process(snap(2.0, trades=(tr(2.0, DOWN, 0.40, 500.0, SELL),))) == []


def test_resting_sell_queue_through_fill_and_symmetry() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.45, 100.0),), "asks": ((0.50, 100.0),)}
    ex.process(snap(0.0, **book))
    buy_now(ex, UP, 100.0, 0.50)  # 100 UP for 50.00
    assert ex.balance() == approx(9950.0)
    submit_ok(ex, req(SELL, 0.50, 40.0))  # joins the ask level: queue_ahead = 100
    # BUY print @0.50 x120: 100 drain the queue, 20 fill us at 0.50 (proceeds 10)
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.50, 120.0, BUY),), **book))
    assert [(f.side, f.price, f.size) for f in fills] == [(SELL, 0.50, pytest.approx(20.0))]
    # BUY print @0.52 > 0.50 is through us: fills the remaining 20 at our 0.50 (proceeds 10)
    fills = ex.process(snap(2.0, trades=(tr(2.0, UP, 0.52, 25.0, BUY),), **book))
    assert [(f.price, f.size) for f in fills] == [(0.50, pytest.approx(20.0))]
    assert ex.open_orders() == []
    assert ex.position(M1, UP) == approx(60.0)
    assert ex.balance() == approx(9970.0)  # 9950 + 10 + 10
    # a BUY print below our ask price does not hit a resting sell
    submit_ok(ex, req(SELL, 0.55, 10.0))
    assert ex.process(snap(3.0, trades=(tr(3.0, UP, 0.54, 100.0, BUY),), **book)) == []


def test_shared_print_capacity_better_price_then_lower_id() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.46, 100.0),), "asks": ((0.52, 100.0),)}  # no queue at 0.47 / 0.48
    ex.process(snap(0.0, **book))
    o1 = submit_ok(ex, req(BUY, 0.48, 30.0))
    o2 = submit_ok(ex, req(BUY, 0.47, 30.0))
    o3 = submit_ok(ex, req(BUY, 0.48, 30.0))
    # print 1: SELL x50 @0.45 through all three. priority: o1(0.48), o3(0.48), o2(0.47)
    #   o1 takes 30, o3 takes the remaining 20, o2 gets nothing.
    # print 2: SELL x100 @0.47: o3 (0.48, through) takes its last 10, o2 (0.47, queue 0) 30.
    fills = ex.process(
        snap(1.0, trades=(tr(1.0, UP, 0.45, 50.0, SELL), tr(1.0, UP, 0.47, 100.0, SELL)), **book)
    )
    got = [(f.order_id, f.price, f.size) for f in fills]
    assert got == [
        (o1, 0.48, pytest.approx(30.0)),
        (o3, 0.48, pytest.approx(20.0)),
        (o3, 0.48, pytest.approx(10.0)),
        (o2, 0.47, pytest.approx(30.0)),
    ]
    assert ex.open_orders() == []
    assert ex.balance() == approx(INITIAL - (0.48 * 60 + 0.47 * 30))  # 28.8 + 14.1 = 42.9
    assert [f.fill_id for f in fills] == ["f1", "f2", "f3", "f4"]


def test_sell_priority_is_lower_price_first() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0, bids=((0.45, 100.0),), asks=((0.50, 100.0),)))
    buy_now(ex, UP, 100.0, 0.50)
    book: Book = {"bids": ((0.45, 100.0),), "asks": ((0.50, 100.0),)}
    a = submit_ok(ex, req(SELL, 0.52, 30.0))  # older but worse price
    b = submit_ok(ex, req(SELL, 0.51, 30.0))
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.53, 50.0, BUY),), **book))
    assert [(f.order_id, f.price, f.size) for f in fills] == [
        (b, 0.51, pytest.approx(30.0)),
        (a, 0.52, pytest.approx(20.0)),
    ]


def test_trade_fill_fraction_scales_print_availability() -> None:
    ex = PaperExchange(cfg_with(fill=0.5))
    book: Book = {"bids": ((0.46, 100.0),), "asks": ((0.52, 100.0),)}
    ex.process(snap(0.0, **book))
    o1 = submit_ok(ex, req(BUY, 0.48, 30.0))
    o2 = submit_ok(ex, req(BUY, 0.47, 30.0))
    # print x80 -> availability 40: o1 takes 30, o2 takes 10
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 80.0, SELL),), **book))
    assert [(f.order_id, f.size) for f in fills] == [
        (o1, pytest.approx(30.0)),
        (o2, pytest.approx(10.0)),
    ]


def test_trade_fill_fraction_applies_before_the_queue_drains() -> None:
    ex = PaperExchange(cfg_with(fill=0.5))
    book: Book = {"bids": ((0.47, 100.0),), "asks": ((0.49, 100.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.47, 80.0))  # queue 100
    # print x300 -> availability 150: 100 drain the queue, 50 fill us
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.47, 300.0, SELL),), **book))
    assert fills[0].size == approx(50.0)
    assert ex.open_orders()[0].remaining == approx(30.0)


def test_float_dust_left_on_an_order_counts_as_filled() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.46, 100.0),), "asks": ((0.52, 100.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.48, 10.0))
    # the print leaves 1e-10 shares (< EPS) behind: the order must not linger as an open order
    fills = ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 10.0 - 1e-10, SELL),), **book))
    assert fills[0].size == approx(10.0, 1e-9)
    assert ex.open_orders() == []
    assert ex.reserved_cash == 0.0


def test_zero_size_prints_are_ignored_and_dust_orders_vanish() -> None:
    ex = PaperExchange(cfg_with(enforce_min=False))
    book: Book = {"bids": ((0.46, 100.0),), "asks": ((0.52, 100.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.48, 0.3))
    assert ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 0.0, SELL),), **book)) == []
    # 0.3 - 0.1 - 0.2 leaves float dust (< EPS): the order must be considered filled
    fills = ex.process(
        snap(2.0, trades=(tr(2.0, UP, 0.45, 0.1, SELL), tr(2.0, UP, 0.45, 0.2, SELL)), **book)
    )
    assert sum(f.size for f in fills) == approx(0.3)
    assert ex.open_orders() == []
    assert ex.reserved_cash == 0.0


# --------------------------------------------------------------------------- no look-ahead


def test_latency_one_order_cannot_fill_from_the_activation_snapshot() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0))
    submit_ok(ex, req(BUY, 0.47, 50.0))
    assert ex.open_orders()[0].live is False
    # snapshot 1 activates the order; its print happened before the order was live: no fill
    assert ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 100.0, SELL),))) == []
    assert ex.open_orders()[0].live is True
    # snapshot 2's print fills it (queue: bids at 0.46 only, so 0.47 improves the touch)
    fills = ex.process(snap(2.0, trades=(tr(2.0, UP, 0.45, 100.0, SELL),)))
    assert [(f.price, f.size) for f in fills] == [(0.47, pytest.approx(50.0))]


def test_latency_two_goes_live_on_the_second_snapshot() -> None:
    ex = PaperExchange(cfg_with(latency=2))
    ex.process(snap(0.0))
    submit_ok(ex, req(BUY, 0.47, 50.0))

    def sells(t: float) -> tuple[Trade, ...]:
        return (tr(t, UP, 0.45, 100.0, SELL),)

    assert ex.process(snap(1.0, trades=sells(1.0))) == []
    assert ex.open_orders()[0].live is False
    assert ex.process(snap(2.0, trades=sells(2.0))) == []  # activates here; this print is too early
    assert ex.open_orders()[0].live is True
    assert len(ex.process(snap(3.0, trades=sells(3.0)))) == 1
    # a print dated before the activation snapshot is invisible even if delivered later
    ex2 = PaperExchange(cfg_with(latency=1))
    ex2.process(snap(0.0))
    submit_ok(ex2, req(BUY, 0.47, 50.0))
    ex2.process(snap(1.0))  # live from ts 1.0
    assert ex2.process(snap(2.0, trades=sells(0.5))) == []
    assert len(ex2.process(snap(3.0, trades=sells(1.0)))) == 1  # dated exactly when it went live


def test_latency_zero_is_live_immediately_but_ignores_older_prints() -> None:
    ex = PaperExchange(cfg_with(latency=0))
    ex.process(snap(0.0))
    submit_ok(ex, req(BUY, 0.47, 50.0), ts=5.0)
    assert ex.open_orders()[0].live is True
    assert ex.open_orders()[0].created_ts == 5.0
    # one print dated 4.5 (before our order existed) and one dated 5.5
    fills = ex.process(
        snap(6.0, trades=(tr(4.5, UP, 0.45, 20.0, SELL), tr(5.5, UP, 0.45, 30.0, SELL)))
    )
    assert [f.size for f in fills] == [pytest.approx(30.0)]
    assert ex.open_orders()[0].remaining == approx(20.0)


def test_latency_counts_snapshots_of_the_orders_own_market() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, mid=M1))
    ex.process(snap(0.0, mid=M2))
    submit_ok(ex, req(BUY, 0.47, 50.0, mid=M1))
    ex.process(snap(1.0, mid=M2))
    ex.process(snap(2.0, mid=M2))
    assert ex.open_orders(M1)[0].live is False
    ex.process(snap(1.0, mid=M1))
    assert ex.open_orders(M1)[0].live is True


def test_ioc_latency_one_executes_against_the_activating_snapshot() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, asks=((0.50, 100.0),)))
    oid = submit_ok(ex, req(BUY, 0.52, 10.0, UP, IOC))
    assert ex.open_orders()[0].live is False
    fills = ex.process(snap(1.0, bids=((0.45, 100.0),), asks=((0.51, 100.0),)))
    assert [(f.order_id, f.price, f.size, f.is_maker) for f in fills] == [
        (oid, 0.51, pytest.approx(10.0), False)
    ]
    assert ex.open_orders() == []
    assert ex.balance() == approx(INITIAL - 5.1)
    assert ex.position(M1, UP) == approx(10.0)


def test_ioc_whose_limit_is_no_longer_reachable_just_expires() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, asks=((0.50, 100.0),)))
    submit_ok(ex, req(BUY, 0.52, 10.0, UP, IOC))
    assert ex.process(snap(1.0, bids=((0.50, 100.0),), asks=((0.53, 100.0),))) == []
    assert ex.open_orders() == [] and ex.reserved_cash == 0.0 and ex.balance() == INITIAL


def test_ioc_latency_zero_needs_a_known_book_and_executes_at_submit() -> None:
    ex = PaperExchange(cfg_with(latency=0))
    res = ex.submit(req(BUY, 0.52, 10.0, UP, IOC), 0.0)
    assert (res.ok, res.reason) == (False, "no_book")
    assert ex.stats["rejected_no_book"] == 1
    ex.process(snap(0.0, asks=((0.50, 100.0),)))
    submit_ok(ex, req(BUY, 0.52, 10.0, UP, IOC), ts=0.5)
    # state moved at once; the fill is queued for delivery
    assert ex.position(M1, UP) == approx(10.0)
    assert ex.balance() == approx(INITIAL - 5.0)
    assert ex.open_orders() == []
    (fill,) = ex.drain_fills()
    assert (fill.price, fill.size, fill.ts, fill.is_maker) == (
        0.50,
        pytest.approx(10.0),
        0.5,
        False,
    )
    assert ex.drain_fills() == []


def test_submit_time_fills_are_delivered_first_by_the_next_process() -> None:
    ex = PaperExchange(cfg_with(latency=0))
    ex.process(snap(0.0, asks=((0.50, 100.0),)))
    submit_ok(ex, req(BUY, 0.50, 10.0, UP, IOC), ts=0.5)
    submit_ok(ex, req(BUY, 0.40, 10.0, UP, POST), ts=0.5)  # resting bid (no queue at 0.40)
    fills = ex.process(snap(1.0, asks=((0.50, 100.0),), trades=(tr(1.0, UP, 0.39, 10.0, SELL),)))
    assert [(f.fill_id, f.is_maker) for f in fills] == [("f1", False), ("f2", True)]


def test_ioc_latency_one_with_no_known_book_is_accepted() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    assert ex.submit(req(BUY, 0.52, 10.0, UP, IOC), 0.0).ok
    fills = ex.process(snap(1.0, asks=((0.50, 100.0),)))
    assert [(f.price, f.size) for f in fills] == [(0.50, pytest.approx(10.0))]


# --------------------------------------------------------------------------- post-only


def test_post_only_crossing_is_rejected_at_submit() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, bids=((0.47, 100.0),), asks=((0.49, 100.0),)))
    for price in (0.49, 0.50):  # at or through the best ask
        res = ex.submit(req(BUY, price, 10.0), 0.0)
        assert (res.ok, res.reason) == (False, "post_only_crosses")
    assert ex.submit(req(BUY, 0.48, 10.0), 0.0).ok
    # DOWN book is the mirror: bid 1 - 0.49 = 0.51, ask 1 - 0.47 = 0.53
    assert ex.submit(req(BUY, 0.53, 10.0, DOWN), 0.0).reason == "post_only_crosses"
    assert ex.submit(req(BUY, 0.52, 10.0, DOWN), 0.0).ok
    assert ex.stats["rejected_post_only_crosses"] == 3


def test_post_only_sell_crossing_the_bid_is_rejected_at_submit() -> None:
    ex = PaperExchange(cfg_with(latency=0))
    ex.process(snap(0.0, bids=((0.47, 100.0),), asks=((0.49, 300.0),)))
    buy_now(ex, UP, 50.0, 0.49)
    assert ex.submit(req(SELL, 0.47, 10.0), 0.0).reason == "post_only_crosses"
    assert ex.submit(req(SELL, 0.46, 10.0), 0.0).reason == "post_only_crosses"
    assert ex.submit(req(SELL, 0.48, 10.0), 0.0).ok


def test_post_only_without_a_known_book_is_accepted_and_rechecked_at_activation() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    assert ex.submit(req(BUY, 0.50, 10.0), 0.0).ok  # no book known yet
    ex.process(snap(1.0, bids=((0.45, 100.0),), asks=((0.50, 100.0),)))  # BUY 0.50 == ask
    assert ex.open_orders() == []
    assert ex.stats["post_only_cancels"] == 1
    assert ex.reserved_cash == 0.0


def test_post_only_crossing_at_activation_cancels_without_fill() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, bids=((0.45, 100.0),), asks=((0.50, 100.0),)))
    submit_ok(ex, req(BUY, 0.48, 10.0))  # fine against the book it was submitted on
    assert ex.reserved_cash == approx(4.8)
    # the ask drops to 0.48: our bid would now cross -> cancelled, never filled as taker
    fills = ex.process(
        snap(
            1.0,
            bids=((0.45, 100.0),),
            asks=((0.48, 100.0),),
            trades=(tr(1.0, UP, 0.45, 100.0, SELL),),
        )
    )
    assert fills == []
    assert ex.open_orders() == []
    assert ex.stats["post_only_cancels"] == 1
    assert ex.reserved_cash == 0.0 and ex.balance() == INITIAL and ex.position(M1, UP) == 0.0


def test_post_only_sell_crossing_at_activation_cancels() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0, bids=((0.45, 100.0),), asks=((0.50, 300.0),)))
    submit_ok(ex, req(BUY, 0.49, 20.0, UP, IOC))  # latency 1: executes on snapshot 1
    ex.process(snap(1.0, bids=((0.45, 100.0),), asks=((0.49, 300.0),)))
    assert ex.position(M1, UP) == approx(20.0)
    submit_ok(ex, req(SELL, 0.47, 10.0))  # fine against bid 0.45
    ex.process(snap(2.0, bids=((0.47, 100.0),), asks=((0.50, 100.0),)))  # bid 0.47 >= our ask
    assert ex.open_orders() == []
    assert ex.stats["post_only_cancels"] == 1
    assert ex.position(M1, UP) == approx(20.0)  # reservation released, nothing sold


# --------------------------------------------------------------------------- IOC execution


def test_ioc_walks_two_levels_and_charges_the_taker_fee() -> None:
    ex = PaperExchange(cfg_with(taker_rate=0.25))
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.50, 60.0), (0.52, 100.0))))
    submit_ok(ex, req(BUY, 0.52, 100.0, UP, IOC))
    f1, f2 = ex.drain_fills()
    # level 0.50 x60: fee = 60 * 0.50 * 0.25 * (0.25) ** 2 = 7.5 * 0.0625 = 0.46875
    assert (f1.price, f1.size, f1.is_maker) == (0.50, pytest.approx(60.0), False)
    assert f1.fee == approx(0.46875)
    # level 0.52 x40: fee = 40 * 0.52 * 0.25 * (0.52 * 0.48) ** 2 = 5.2 * 0.06230016 = 0.323960832
    assert (f2.price, f2.size) == (0.52, pytest.approx(40.0))
    assert f2.fee == approx(0.323960832)
    # cash = 10000 - (30 + 0.46875) - (20.8 + 0.323960832) = 9948.407289168
    assert ex.balance() == approx(9948.407289168)
    assert ex.position(M1, UP) == approx(100.0)
    assert ex.stats["filled_taker"] == 2
    assert ex.stats["volume_taker"] == approx(50.8)  # 30 + 20.8
    assert ex.stats["volume_maker"] == 0
    assert ex.open_orders() == []


def test_ioc_never_walks_beyond_its_limit() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.50, 60.0), (0.52, 100.0))))
    submit_ok(ex, req(BUY, 0.51, 100.0, UP, IOC))
    (f,) = ex.drain_fills()
    assert (f.price, f.size) == (0.50, pytest.approx(60.0))
    assert ex.position(M1, UP) == approx(60.0)  # the other 40 are cancelled, not resting
    assert ex.open_orders() == []


def test_ioc_sell_walks_bids_and_charges_fees() -> None:
    ex = PaperExchange(cfg_with(taker_rate=0.25))
    ex.process(snap(0.0, bids=((0.45, 100.0),), asks=((0.50, 100.0),)))
    buy_now(ex, UP, 100.0, 0.50)  # 50 + 0.78125 -> cash 9949.21875
    assert ex.balance() == approx(9949.21875)
    ex.process(snap(1.0, bids=((0.50, 60.0), (0.49, 100.0)), asks=((0.55, 100.0),)))
    submit_ok(ex, req(SELL, 0.49, 100.0, UP, IOC), ts=1.0)
    f1, f2 = ex.drain_fills()
    # 60 @0.50: fee 0.46875; 40 @0.49: fee = 40 * 0.49 * 0.25 * (0.49 * 0.51) ** 2
    #   = 4.9 * 0.06245001 = 0.306005049
    assert (f1.side, f1.price, f1.size, f1.fee) == (
        SELL,
        0.50,
        pytest.approx(60.0),
        approx(0.46875),
    )
    assert (f2.price, f2.size, f2.fee) == (0.49, pytest.approx(40.0), approx(0.306005049))
    # proceeds 30 + 19.6 = 49.6 ; cash = 9949.21875 + 49.6 - 0.46875 - 0.306005049
    assert ex.balance() == approx(9998.043994951)
    assert ex.position(M1, UP) == 0.0


def test_ioc_sell_partial_when_the_book_is_shallow() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0, bids=((0.45, 30.0),), asks=((0.50, 300.0),)))
    buy_now(ex, UP, 100.0, 0.50)
    submit_ok(ex, req(SELL, 0.40, 100.0, UP, IOC))
    (f,) = ex.drain_fills()
    assert (f.price, f.size) == (0.45, pytest.approx(30.0))
    assert ex.position(M1, UP) == approx(70.0)
    assert ex.open_orders() == []


def test_ioc_depth_is_shared_within_a_snapshot_and_refreshes_on_the_next() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.40, 100.0),), "asks": ((0.50, 100.0),)}
    ex.process(snap(0.0, **book))
    submit_ok(ex, req(BUY, 0.50, 60.0, UP, IOC))
    submit_ok(ex, req(BUY, 0.50, 60.0, UP, IOC))  # only 100 - 60 = 40 left
    submit_ok(ex, req(BUY, 0.50, 10.0, UP, IOC))  # nothing left
    assert [f.size for f in ex.drain_fills()] == [pytest.approx(60.0), pytest.approx(40.0)]
    ex.process(snap(1.0, **book))  # a fresh snapshot: full depth again
    submit_ok(ex, req(BUY, 0.50, 60.0, UP, IOC))
    assert [f.size for f in ex.drain_fills()] == [pytest.approx(60.0)]
    assert ex.position(M1, UP) == approx(160.0)


def test_ioc_activations_in_the_same_snapshot_share_depth_in_id_order() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0))
    a = submit_ok(ex, req(BUY, 0.50, 70.0, UP, IOC))
    b = submit_ok(ex, req(BUY, 0.50, 70.0, UP, IOC))
    fills = ex.process(snap(1.0, bids=((0.40, 100.0),), asks=((0.50, 100.0),)))
    assert [(f.order_id, f.size) for f in fills] == [
        (a, pytest.approx(70.0)),
        (b, pytest.approx(30.0)),
    ]


def test_ioc_slippage_is_charged_but_capped_by_the_limit() -> None:
    # 2 ticks of slippage = 0.02
    ex = PaperExchange(cfg_with(slip=2))
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.50, 300.0),)))
    submit_ok(ex, req(BUY, 0.55, 10.0, UP, IOC))  # 0.50 + 0.02 = 0.52 <= 0.55
    submit_ok(ex, req(BUY, 0.51, 10.0, UP, IOC))  # 0.52 would exceed the limit: capped at 0.51
    submit_ok(ex, req(BUY, 0.50, 10.0, UP, IOC))  # limit == level: no worse price allowed
    assert [f.price for f in ex.drain_fills()] == [0.52, 0.51, 0.50]
    assert ex.balance() == approx(INITIAL - (5.2 + 5.1 + 5.0))
    ex.process(snap(1.0, bids=((0.50, 300.0),), asks=((0.60, 100.0),)))
    submit_ok(ex, req(SELL, 0.45, 10.0, UP, IOC))  # 0.50 - 0.02 = 0.48 >= 0.45
    submit_ok(ex, req(SELL, 0.49, 10.0, UP, IOC))  # 0.48 would breach the limit: capped at 0.49
    assert [f.price for f in ex.drain_fills()] == [0.48, 0.49]


def test_ioc_slippage_fee_uses_the_slipped_price() -> None:
    ex = PaperExchange(cfg_with(slip=1, taker_rate=0.25))
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.49, 300.0),)))
    submit_ok(ex, req(BUY, 0.55, 100.0, UP, IOC))
    (f,) = ex.drain_fills()
    # price 0.49 + 0.01 = 0.50; fee = 100 * 0.5 * 0.25 * 0.0625 = 0.78125
    assert (f.price, f.fee) == (0.50, approx(0.78125))
    assert ex.balance() == approx(INITIAL - 50.78125)


def test_ioc_is_partial_when_cash_runs_out_at_execution() -> None:
    # Both IOCs pass the submit check: o1 needs 50.78125 of 100.78125; o2 needs 50.78125 of the
    # 50.78125 left once o1's 50 is reserved. At execution o1 spends 50.78125 (notional + fee),
    # leaving cash 50.0 for o2, which can buy 50 / (0.5 + 0.0078125) = 6400 / 65 shares.
    ex = PaperExchange(cfg_with(latency=1, cash=100.78125, taker_rate=0.25))
    ex.process(snap(0.0))
    a = submit_ok(ex, req(BUY, 0.50, 100.0, UP, IOC))
    b = submit_ok(ex, req(BUY, 0.50, 100.0, UP, IOC))
    assert ex.reserved_cash == approx(100.0)
    fills = ex.process(snap(1.0, bids=((0.40, 100.0),), asks=((0.50, 500.0),)))
    assert [f.order_id for f in fills] == [a, b]
    assert fills[0].size == approx(100.0) and fills[0].fee == approx(0.78125)
    assert fills[1].size == approx(6400.0 / 65.0)  # 98.4615384615...
    assert fills[1].fee == approx(6400.0 / 65.0 * 0.0078125)  # 0.7692307692...
    assert ex.balance() == approx(0.0)
    assert ex.balance() >= 0.0
    assert ex.open_orders() == []


def test_ioc_buy_cash_check_excludes_the_cash_other_bids_committed() -> None:
    ex = PaperExchange(cfg_with(latency=1, cash=100.0))
    ex.process(snap(0.0, bids=((0.40, 100.0),), asks=((0.50, 500.0),)))
    submit_ok(ex, req(BUY, 0.45, 100.0, UP, POST))  # commits 45 of the 100
    submit_ok(ex, req(BUY, 0.50, 100.0, UP, IOC))  # needs 50 <= 55 free: accepted
    fills = ex.process(snap(1.0, bids=((0.40, 100.0),), asks=((0.50, 500.0),)))
    assert sum(f.size for f in fills) == approx(100.0)
    assert ex.balance() == approx(50.0)  # 100 - 50
    assert ex.balance() - ex.reserved_cash >= 0.0  # the resting 0.45 bid is still funded (45)


# --------------------------------------------------------------------------- cancel


def test_cancel_semantics() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(0.0))
    live_id = submit_ok(ex, req(BUY, 0.47, 50.0))
    assert ex.cancel(live_id, 0.0) is True  # in flight orders can be cancelled
    assert ex.stats["cancelled"] == 1
    assert ex.cancel(live_id, 0.0) is False  # already gone
    assert ex.cancel("o999", 0.0) is False  # unknown
    assert ex.reserved_cash == 0.0
    # a cancelled in-flight order never activates or fills
    assert ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 100.0, SELL),))) == []
    assert ex.process(snap(2.0, trades=(tr(2.0, UP, 0.45, 100.0, SELL),))) == []


def test_cancel_of_a_filled_order_returns_false_and_partial_can_be_cancelled() -> None:
    ex = PaperExchange(cfg_with())
    book: Book = {"bids": ((0.46, 100.0),), "asks": ((0.52, 100.0),)}
    ex.process(snap(0.0, **book))
    full = submit_ok(ex, req(BUY, 0.48, 10.0))
    part = submit_ok(ex, req(BUY, 0.47, 50.0))
    ex.process(snap(1.0, trades=(tr(1.0, UP, 0.45, 30.0, SELL),), **book))  # 10 + 20
    assert ex.cancel(full, 1.0) is False  # filled
    assert ex.open_orders()[0].remaining == approx(30.0)
    assert ex.reserved_cash == approx(14.1)  # 0.47 * 30
    assert ex.cancel(part, 1.0) is True
    assert ex.reserved_cash == 0.0
    assert ex.position(M1, UP) == approx(30.0)  # fills are kept


def test_open_orders_are_copies_sorted_by_numeric_id() -> None:
    ex = PaperExchange(cfg_with())
    for i in range(11):
        submit_ok(ex, req(BUY, 0.40, 5.0, UP, POST, M1 if i % 2 == 0 else M2))
    ids = [o.order_id for o in ex.open_orders()]
    assert ids == [f"o{i}" for i in range(1, 12)]  # o10 sorts after o9, not after o1
    assert [o.order_id for o in ex.open_orders(M2)] == ["o2", "o4", "o6", "o8", "o10"]
    assert ex.open_orders("nope") == []
    victim = ex.open_orders()[0]
    victim.remaining = 0.0
    victim.price = 0.99
    victim.live = False
    again = ex.open_orders()[0]
    assert (again.remaining, again.price, again.live) == (5.0, 0.40, True)
    assert ex.reserved_cash == approx(11 * 0.40 * 5.0)


# --------------------------------------------------------------------------- merge / settle


def _hold_pairs(latency: int = 0) -> PaperExchange:
    """100 UP bought at 0.50 and 100 DOWN at 0.45 (cash 10000 - 50 - 45 = 9905)."""
    ex = PaperExchange(cfg_with(latency=latency))
    ex.process(snap(0.0, bids=((0.55, 300.0),), asks=((0.50, 300.0),)))  # UP ask .50, DOWN ask .45
    buy_now(ex, UP, 100.0, 0.50)
    buy_now(ex, DOWN, 100.0, 0.45)
    assert ex.balance() == approx(9905.0)
    return ex


def test_merge_moves_positions_and_cash() -> None:
    ex = _hold_pairs()
    res = ex.merge(M1, 60.0, 7.0)
    assert res is not None
    assert (res.market_id, res.size, res.cash, res.ts) == (M1, 60.0, 60.0, 7.0)
    assert ex.position(M1, UP) == approx(40.0) and ex.position(M1, DOWN) == approx(40.0)
    assert ex.balance() == approx(9965.0)
    assert ex.stats["merges"] == 1


def test_merge_rejections_leave_state_untouched() -> None:
    ex = _hold_pairs()
    assert ex.merge(M1, 100.5, 0.0) is None  # only 100 pairs
    assert ex.merge(M1, 0.0, 0.0) is None
    assert ex.merge(M1, -3.0, 0.0) is None
    assert ex.merge(M1, float("nan"), 0.0) is None
    assert ex.merge("other", 1.0, 0.0) is None
    assert ex.stats["merge_rejects"] == 5 and ex.stats["merges"] == 0
    assert ex.balance() == approx(9905.0) and ex.position(M1, UP) == approx(100.0)
    # after merging everything there is nothing left to merge
    assert ex.merge(M1, 100.0, 0.0) is not None
    assert ex.position(M1, UP) == 0.0 and ex.position(M1, DOWN) == 0.0
    assert ex.merge(M1, 1.0, 0.0) is None


def test_merge_is_limited_by_the_smaller_token_and_tolerates_float_dust() -> None:
    ex = PaperExchange(cfg_with())
    ex.process(snap(0.0, bids=((0.55, 300.0),), asks=((0.50, 300.0),)))
    buy_now(ex, UP, 100.0, 0.50)
    buy_now(ex, DOWN, 30.0, 0.45)
    assert ex.merge(M1, 31.0, 0.0) is None
    assert ex.merge(M1, 30.0 + 5e-10, 0.0) is not None  # within EPS of the 30 held
    assert ex.position(M1, DOWN) == 0.0  # dust snapped away
    assert ex.position(M1, UP) == approx(70.0, abs_tol=1e-6)


def test_merge_only_uses_shares_not_reserved_by_open_sells() -> None:
    ex = _hold_pairs()
    submit_ok(ex, req(SELL, 0.60, 30.0, UP))  # reserves 30 UP
    assert ex.merge(M1, 100.0, 0.0) is None  # only 70 UP are free
    assert ex.merge(M1, 70.0, 0.0) is not None
    assert ex.position(M1, UP) == approx(30.0)  # exactly the reserved shares remain


def test_settle_pays_winner_and_cancels_open_orders() -> None:
    ex = PaperExchange(cfg_with(latency=0))
    ex.process(snap(0.0, bids=((0.55, 300.0),), asks=((0.50, 300.0),)))
    ex.process(snap(0.0, mid=M2, bids=((0.55, 300.0),), asks=((0.50, 300.0),)))
    buy_now(ex, UP, 100.0, 0.50)  # 50
    buy_now(ex, DOWN, 30.0, 0.45)  # 13.5
    submit_ok(ex, req(BUY, 0.40, 50.0, UP, POST, M1))  # reserves 20
    submit_ok(ex, req(BUY, 0.30, 10.0, UP, POST, M2))  # reserves 3
    assert ex.balance() == approx(10000 - 63.5)
    s = ex.settle(M1, UP, 900.0)
    assert (s.market_id, s.winner, s.ts) == (M1, UP, 900.0)
    assert s.payout == approx(100.0)  # 100 winning UP shares, DOWN pays nothing
    assert ex.balance() == approx(10000 - 63.5 + 100.0)
    assert ex.position(M1, UP) == 0.0 and ex.position(M1, DOWN) == 0.0
    assert [o.market_id for o in ex.open_orders()] == [M2]  # only M1's orders are cancelled
    assert ex.reserved_cash == approx(3.0)
    assert ex.stats["settled"] == 1


def test_settle_the_other_winner() -> None:
    ex = _hold_pairs()
    s = ex.settle(M1, DOWN, 900.0)
    assert s.payout == approx(100.0)
    assert ex.balance() == approx(9905.0 + 100.0)


def test_settle_twice_raises_and_a_settled_market_is_closed() -> None:
    ex = _hold_pairs()
    ex.settle(M1, UP, 900.0)
    with pytest.raises(ValueError):
        ex.settle(M1, DOWN, 901.0)
    assert ex.balance() == approx(10005.0)  # the failed call changed nothing
    assert ex.submit(req(BUY, 0.40, 10.0), 902.0).reason == "market_settled"
    assert ex.merge(M1, 1.0, 902.0) is None
    with pytest.raises(ValueError):
        ex.process(snap(903.0))


def test_settle_of_a_market_with_no_activity_pays_nothing() -> None:
    ex = PaperExchange(cfg_with())
    s = ex.settle("never-seen", UP, 1.0)
    assert s.payout == 0.0 and ex.balance() == INITIAL


def test_settle_refuses_while_submit_time_fills_are_undelivered() -> None:
    ex = PaperExchange(cfg_with(latency=0))
    ex.process(snap(0.0, asks=((0.50, 100.0),)))
    submit_ok(ex, req(BUY, 0.50, 10.0, UP, IOC))
    with pytest.raises(ValueError):
        ex.settle(M1, UP, 1.0)
    assert len(ex.drain_fills()) == 1
    assert ex.settle(M1, UP, 1.0).payout == approx(10.0)


# --------------------------------------------------------------------------- process validation


def test_process_rejects_bad_data_without_changing_state() -> None:
    ex = PaperExchange(cfg_with(latency=1))
    ex.process(snap(5.0))
    submit_ok(ex, req(BUY, 0.47, 50.0))
    good = snap(6.0)
    bad_snaps = [
        snap(4.0),  # older than the previous snapshot of this market
        snap(6.0, trades=(tr(6.0, UP, 1.5, 10.0, SELL),)),  # price outside [0, 1]
        snap(6.0, trades=(tr(6.0, UP, 0.5, -1.0, SELL),)),  # negative size
        snap(6.0, trades=(tr(6.0, UP, 0.5, float("nan"), SELL),)),
        snap(float("nan")),
        MarketSnapshot(  # book slots swapped
            ts=6.0, market=good.market, up_book=good.down_book, down_book=good.up_book
        ),
        snap(6.0, bids=((1.2, 10.0),)),  # level price outside [0, 1]
        snap(6.0, bids=((0.4, -10.0),)),  # negative level size
        snap(6.0, tick=0.0),
    ]
    for bad in bad_snaps:
        with pytest.raises(ValueError):
            ex.process(bad)
    assert ex.open_orders()[0].live is False  # the failed calls did not advance the latency clock
    ex.process(good)
    assert ex.open_orders()[0].live is True
    ex.process(snap(6.0))  # equal timestamps are allowed


def test_submit_rejects_a_non_finite_clock() -> None:
    ex = PaperExchange(cfg_with())
    with pytest.raises(ValueError):
        ex.submit(req(BUY, 0.47, 10.0), float("nan"))
    with pytest.raises(ValueError):
        ex.cancel("o1", float("inf"))


# --------------------------------------------------------------------------- determinism


def _script(cfg: BotConfig) -> tuple[list[Fill], dict[str, float], list[object], float]:
    ex = PaperExchange(cfg)
    fills: list[Fill] = []
    book: Book = {"bids": ((0.46, 100.0),), "asks": ((0.52, 100.0),)}
    fills += ex.process(snap(0.0, **book))
    for price, size in ((0.47, 40.0), (0.48, 30.0), (0.45, 25.0)):
        ex.submit(req(BUY, price, size), 0.0)
    ex.submit(req(BUY, 0.52, 20.0, UP, IOC), 0.0)
    for t in (1.0, 2.0, 3.0):
        fills += ex.process(snap(t, trades=(tr(t, UP, 0.44 + 0.01 * t, 35.0, SELL),), **book))
        fills += ex.drain_fills()
    ex.cancel("o3", 3.0)
    return fills, dict(ex.stats), list(ex.open_orders()), ex.balance()


@pytest.mark.parametrize("latency", [0, 1, 2])
def test_same_inputs_give_identical_results(latency: int) -> None:
    cfg = cfg_with(latency=latency, taker_rate=0.25, maker_fee=0.001)
    first, second = _script(cfg), _script(cfg)
    assert first == second
    assert first[0], "the script must actually produce fills"


# --------------------------------------------------------------------------- scripted scenario


def _pair_scenario(cfg: BotConfig) -> tuple[PaperExchange, Portfolio, list[Fill]]:
    """Bid UP@0.47 and DOWN@0.52 (x100 each), fill both via prints, merge 100 pairs."""
    ex = PaperExchange(cfg)
    pf = Portfolio(cfg.exchange.initial_cash)
    # UP bid 0.46 / ask 0.49 -> DOWN bid 0.51 / ask 0.54: both our bids improve the touch (no queue)
    book: Book = {"bids": ((0.46, 300.0),), "asks": ((0.49, 300.0),)}
    fills = ex.process(snap(0.0, **book))
    assert ex.submit(req(BUY, 0.47, 100.0, UP), 0.0).ok
    assert ex.submit(req(BUY, 0.52, 100.0, DOWN), 0.0).ok
    assert ex.reserved_cash == approx(99.0)  # 47 + 52
    fills += ex.process(snap(1.0, **book))  # activation, post-only re-check passes
    assert ex.reserved_cash == approx(99.0)
    # UP SELL x100 @0.47 hits our UP bid; its mirror (DOWN BUY @0.53) cannot hit a bid
    fills += ex.process(snap(2.0, trades=prints(2.0, 0.47, 100.0, SELL), **book))
    assert ex.position(M1, UP) == approx(100.0) and ex.position(M1, DOWN) == 0.0
    # UP BUY x100 @0.48 cannot hit a bid; its mirror (DOWN SELL @0.52) fills our DOWN bid
    fills += ex.process(snap(3.0, trades=prints(3.0, 0.48, 100.0, BUY), **book))
    assert ex.position(M1, DOWN) == approx(100.0)
    assert ex.open_orders() == [] and ex.reserved_cash == 0.0
    for f in fills:
        pf.apply_fill(f)
    merged = ex.merge(M1, 100.0, 3.0)
    assert merged is not None
    pf.apply_merge(merged)
    return ex, pf, fills


def test_scripted_pair_cycle_locks_exactly_one_cent_per_pair() -> None:
    ex, pf, fills = _pair_scenario(cfg_with())
    assert [(f.token, f.side, f.price, f.size, f.fee, f.is_maker) for f in fills] == [
        (UP, BUY, 0.47, pytest.approx(100.0), 0.0, True),
        (DOWN, BUY, 0.52, pytest.approx(100.0), 0.0, True),
    ]
    # cash = 10000 - 47 - 52 + 100 = 10001 exactly: 100 pairs * (1 - 0.47 - 0.52)
    assert ex.balance() == approx(INITIAL + 1.0)
    assert ex.position(M1, UP) == 0.0 and ex.position(M1, DOWN) == 0.0
    assert pf.cash == approx(ex.balance())
    assert pf.realised_pnl() == approx(1.0)
    assert pf.inventory(M1).merge_pnl == approx(1.0)
    assert ex.stats["submitted"] == 2 and ex.stats["accepted"] == 2
    assert ex.stats["filled_maker"] == 2 and ex.stats["filled_taker"] == 0
    assert ex.stats["volume_maker"] == approx(99.0)  # 47 + 52
    assert ex.stats["merges"] == 1 and ex.stats["merge_rejects"] == 0


def test_scripted_pair_cycle_with_a_maker_fee() -> None:
    # fees: 0.01 * 47 = 0.47 and 0.01 * 52 = 0.52 -> cash = 10001 - 0.99 = 10000.01
    ex, pf, fills = _pair_scenario(cfg_with(maker_fee=0.01))
    assert [f.fee for f in fills] == [approx(0.47), approx(0.52)]
    assert ex.balance() == approx(INITIAL + 0.01)
    assert pf.cash == approx(ex.balance())
    assert pf.realised_pnl() == approx(0.01)


def test_scripted_pair_cycle_with_a_maker_rebate() -> None:
    # rebates: 0.005 * 47 = 0.235 and 0.005 * 52 = 0.26 (negative fees) -> 10001 + 0.495
    ex, pf, fills = _pair_scenario(cfg_with(rebate=0.005))
    assert [f.fee for f in fills] == [approx(-0.235), approx(-0.26)]
    assert ex.balance() == approx(INITIAL + 1.495)
    assert pf.cash == approx(ex.balance())
    assert pf.realised_pnl() == approx(1.495)


def test_scripted_pair_cycle_held_to_settlement_pays_the_same() -> None:
    cfg = cfg_with()
    ex = PaperExchange(cfg)
    book: Book = {"bids": ((0.46, 300.0),), "asks": ((0.49, 300.0),)}
    ex.process(snap(0.0, **book))
    ex.submit(req(BUY, 0.47, 100.0, UP), 0.0)
    ex.submit(req(BUY, 0.52, 100.0, DOWN), 0.0)
    ex.process(snap(1.0, **book))
    ex.process(snap(2.0, trades=prints(2.0, 0.47, 100.0, SELL), **book))
    ex.process(snap(3.0, trades=prints(3.0, 0.48, 100.0, BUY), **book))
    for winner in (UP, DOWN):
        ex2 = PaperExchange(cfg)
        ex2.process(snap(0.0, **book))
        ex2.submit(req(BUY, 0.47, 100.0, UP), 0.0)
        ex2.submit(req(BUY, 0.52, 100.0, DOWN), 0.0)
        ex2.process(snap(1.0, **book))
        ex2.process(snap(2.0, trades=prints(2.0, 0.47, 100.0, SELL), **book))
        ex2.process(snap(3.0, trades=prints(3.0, 0.48, 100.0, BUY), **book))
        assert ex2.settle(M1, winner, 900.0).payout == approx(100.0)
        assert ex2.balance() == approx(INITIAL + 1.0)  # -99 + 100 whoever wins


# --------------------------------------------------------------------------- randomised invariants


def _tick_price(k: int) -> float:
    return round(k * 0.01, 2)


def _random_snapshot(rng: random.Random, mid: str, ts: float, center: int) -> MarketSnapshot:
    bid_t = center
    ask_t = center + rng.randint(1, 3)
    bids = tuple(
        (_tick_price(bid_t - i), float(rng.choice([10, 25, 50, 100, 200]))) for i in range(3)
    )
    asks = tuple(
        (_tick_price(ask_t + i), float(rng.choice([10, 25, 50, 100, 200]))) for i in range(3)
    )
    trades: list[Trade] = []
    for _ in range(rng.randint(0, 3)):
        up_tick = min(99, max(1, rng.randint(bid_t - 2, ask_t + 2)))
        side = rng.choice([BUY, SELL])
        size = float(rng.choice([5, 10, 20, 50, 100]))
        trades.extend(prints(ts, _tick_price(up_tick), size, side))
    return snap(ts, bids=bids, asks=asks, trades=tuple(trades), mid=mid)


def _random_request(rng: random.Random, mid: str, center: int) -> OrderRequest:
    token = rng.choice([UP, DOWN])
    ref = center if token is UP else 100 - center
    price = _tick_price(min(99, max(1, rng.randint(ref - 5, ref + 5))))
    return req(
        rng.choice([BUY, BUY, SELL]),
        price,
        float(rng.choice([5, 10, 20, 40, 80])),
        token,
        IOC if rng.random() < 0.25 else POST,
        mid,
    )


def _check_state(ex: PaperExchange, pf: Portfolio, markets: tuple[str, ...]) -> None:
    orders = ex.open_orders()
    # reserved_cash == sum of BUY price * remaining, at all times
    assert ex.reserved_cash == approx(
        sum(o.price * o.remaining for o in orders if o.side is BUY), 1e-9
    )
    assert ex.balance() >= -1e-9
    assert ex.balance() - ex.reserved_cash >= -1e-9
    sold: dict[tuple[str, Outcome], float] = {}
    for o in orders:
        assert 0.0 < o.remaining <= o.size + 1e-9
        if o.side is SELL:
            sold[(o.market_id, o.token)] = sold.get((o.market_id, o.token), 0.0) + o.remaining
    for m in markets:
        for t in Outcome:
            assert ex.position(m, t) >= 0.0
            assert sold.get((m, t), 0.0) <= ex.position(m, t) + 1e-9  # no overselling
            # reconciliation with the independent accounting in inventory.py
            assert pf.inventory(m).qty[t] == approx(ex.position(m, t), 1e-6)
    assert pf.cash == approx(ex.balance(), 1e-6)
    realised = sum(inv.realised_pnl for inv in pf.inventories.values())
    assert pf.realised_pnl() == approx(realised, 1e-6)  # DESIGN invariant 2


@pytest.mark.parametrize(
    "seed, latency, maker_fee, rebate, slip, queue, fill_fraction, cash0",
    [
        (11, 0, 0.0, 0.0, 0, 1.0, 1.0, 3_000.0),
        (12, 1, 0.01, 0.0, 0, 1.0, 1.0, 3_000.0),
        (13, 2, 0.0, 0.005, 1, 0.5, 1.0, 3_000.0),
        (14, 1, 0.002, 0.0, 0, 0.5, 0.5, 3_000.0),
        (15, 3, 0.0, 0.0, 2, 0.0, 1.0, 3_000.0),
        # tight cash: reservations, fee commitments and IOC cash caps are the binding constraint
        (16, 0, 0.0, 0.0, 0, 1.0, 1.0, 120.0),
        (17, 1, 0.02, 0.0, 0, 0.5, 1.0, 120.0),
        (18, 2, 0.01, 0.0, 1, 0.0, 1.0, 80.0),
        (19, 1, 0.0, 0.003, 0, 1.0, 0.5, 100.0),
    ],
)
def test_random_walk_preserves_exchange_invariants(
    seed: int,
    latency: int,
    maker_fee: float,
    rebate: float,
    slip: int,
    queue: float,
    fill_fraction: float,
    cash0: float,
) -> None:
    rng = random.Random(seed)
    cfg = cfg_with(
        latency=latency,
        cash=cash0,
        maker_fee=maker_fee,
        rebate=rebate,
        slip=slip,
        queue=queue,
        fill=fill_fraction,
        taker_rate=0.25,
    )
    ex, pf = PaperExchange(cfg), Portfolio(cash0)
    markets = (M1, M2)
    center = {M1: 50, M2: 40}
    clock = {M1: 0.0, M2: 0.0}
    order_info: dict[str, OrderRequest] = {}
    filled: dict[str, float] = {}
    next_fill = 1
    saw_reject = False

    def absorb(new_fills: list[Fill]) -> None:
        nonlocal next_fill
        for f in new_fills:
            assert f.fill_id == f"f{next_fill}"  # sequential, deterministic, delivered in order
            next_fill += 1
            r = order_info[f.order_id]
            assert f.size > 0.0 and 0.0 < f.price < 1.0
            assert f.is_maker == (r.tif is POST)  # post-only never fills as taker, IOC never maker
            if f.is_maker:
                assert f.price == r.price  # a resting order fills at its own limit
            elif r.side is BUY:
                assert f.price <= r.price + 1e-9
            else:
                assert f.price >= r.price - 1e-9
            filled[f.order_id] = filled.get(f.order_id, 0.0) + f.size
            assert filled[f.order_id] <= r.size + 1e-9  # fills never exceed the order size
            pf.apply_fill(f)

    for m in markets:
        absorb(ex.process(_random_snapshot(rng, m, 0.0, center[m])))
    for _ in range(400):
        m = rng.choice(markets)
        roll = rng.random()
        if roll < 0.45:
            center[m] = min(80, max(20, center[m] + rng.randint(-2, 2)))
            clock[m] += 1.0
            absorb(ex.process(_random_snapshot(rng, m, clock[m], center[m])))
        elif roll < 0.85:
            r = _random_request(rng, m, center[m])
            res = ex.submit(r, clock[m])
            if res.ok:
                assert res.order_id is not None and res.order_id not in order_info
                order_info[res.order_id] = r
                absorb(ex.drain_fills())
            else:
                saw_reject = True
                assert res.reason and res.order_id is None
        elif roll < 0.93:
            open_ids = [o.order_id for o in ex.open_orders()]
            if open_ids:
                assert ex.cancel(rng.choice(open_ids), clock[m])
        else:
            size = float(rng.choice([1, 5, 10, 25, 50]))
            merged = ex.merge(m, size, clock[m])
            if merged is not None:
                pf.apply_merge(merged)
        _check_state(ex, pf, markets)

    assert saw_reject
    assert ex.stats["filled_maker"] > 0 and ex.stats["filled_taker"] > 0
    if cash0 < 1_000.0:
        assert ex.stats["rejected_insufficient_cash"] > 0  # cash really was the constraint
    assert ex.stats["post_only_cancels"] + ex.stats["rejected_post_only_crosses"] > 0
    # settle everything: no orders left, positions zero, per-market pnl adds up to total pnl
    absorb(ex.drain_fills())
    total = 0.0
    for m in markets:
        s = ex.settle(m, rng.choice([UP, DOWN]), 900.0)
        total += pf.apply_settlement(s).total
    assert ex.open_orders() == [] and ex.reserved_cash == 0.0
    assert all(ex.position(m, t) == 0.0 for m in markets for t in Outcome)
    assert ex.balance() == approx(pf.cash, 1e-6)
    assert total == approx(ex.balance() - cash0, 1e-6)  # DESIGN invariant 4 (exchange side)
    assert math.isfinite(ex.balance())
