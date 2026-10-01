"""Tests for abc_trading.strategy.risk (RiskManager, books_quotable), DESIGN 2.7.

Default RiskConfig: per-market cap 1500, total cap 5000, daily loss 1000, spot staleness 5 s,
20 orders / s, spread limit 6 ticks. Tick size 0.01 throughout.
"""

from __future__ import annotations

import math
import random

import pytest

from abc_trading.config import RiskConfig
from abc_trading.strategy.risk import RiskManager, books_quotable
from abc_trading.types import (
    BookSnapshot,
    Level,
    MarketSnapshot,
    MarketSpec,
    Outcome,
)

MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)  # tick 0.01
UP, DOWN = Outcome.UP, Outcome.DOWN


def book(token: Outcome, bid: float | None, ask: float | None, size: float = 100.0) -> BookSnapshot:
    return BookSnapshot(
        token,
        () if bid is None else (Level(bid, size),),
        () if ask is None else (Level(ask, size),),
    )


def snap(
    *,
    ts: float = 1100.0,
    spot: float | None = 100.0,
    spot_ts: float | None = 1100.0,
    up: tuple[float | None, float | None] = (0.49, 0.51),
    down: tuple[float | None, float | None] = (0.49, 0.51),
) -> MarketSnapshot:
    return MarketSnapshot(
        ts=ts,
        market=MKT,
        up_book=book(UP, *up),
        down_book=book(DOWN, *down),
        spot=spot,
        spot_ts=spot_ts,
        ref_price=100.0,
    )


def mgr(**kw: float) -> RiskManager:
    return RiskManager(RiskConfig(**kw), starting_equity=10_000.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- kill switch


def test_kill_switch_trips_exactly_at_the_loss_limit() -> None:
    r = mgr(max_daily_loss_usd=1000.0)
    assert not r.kill_switch
    r.update_equity(10_000.0)
    r.update_equity(9_000.5)  # loss 999.5 < 1000
    assert not r.kill_switch
    assert r.kill_equity is None
    r.update_equity(9_000.0)  # equity - start = -1000 <= -1000
    assert r.kill_switch
    assert r.kill_equity == 9_000.0


def test_kill_switch_ignores_profits_and_small_losses() -> None:
    r = mgr(max_daily_loss_usd=500.0)
    for equity in (10_000.0, 12_345.6, 9_900.0, 9_500.01, 11_000.0):
        r.update_equity(equity)
    assert not r.kill_switch


def test_kill_switch_latches_and_never_resets() -> None:
    r = mgr(max_daily_loss_usd=100.0)
    r.update_equity(9_800.0)  # -200 -> tripped
    assert r.kill_switch
    first_equity = r.kill_equity
    r.update_equity(20_000.0)  # a huge recovery does not un-trip it
    assert r.kill_switch
    r.update_equity(9_000.0)  # and a worse equity does not overwrite the recorded one
    assert r.kill_switch
    assert r.kill_equity == first_equity == 9_800.0


def test_kill_switch_rejects_non_finite_equity() -> None:
    r = mgr()
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            r.update_equity(bad)
    assert not r.kill_switch


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="starting_equity"):
        RiskManager(RiskConfig(), starting_equity=math.nan)
    with pytest.raises(ValueError, match="max_orders_per_second"):
        RiskManager(RiskConfig(max_orders_per_second=0.5), starting_equity=1.0)
    assert RiskManager(RiskConfig(), starting_equity=123.0).starting_equity == 123.0


# --------------------------------------------------------------------------- spot staleness


def test_spot_ok_age_boundary() -> None:
    r = mgr(max_spot_staleness_seconds=5.0)
    assert r.spot_ok(snap(ts=1100.0, spot_ts=1100.0))  # age 0
    assert r.spot_ok(snap(ts=1100.0, spot_ts=1095.0))  # age 5 == max -> ok
    assert not r.spot_ok(snap(ts=1100.0, spot_ts=1094.99))  # age 5.01 > 5 -> stale
    assert not r.spot_ok(snap(ts=1100.0, spot_ts=1000.0))


def test_spot_ok_requires_spot_and_timestamp() -> None:
    r = mgr()
    assert not r.spot_ok(snap(spot=None))
    assert not r.spot_ok(snap(spot_ts=None))  # age unknown -> fail closed
    assert not r.spot_ok(snap(spot=0.0))
    assert not r.spot_ok(snap(spot=-5.0))
    assert not r.spot_ok(snap(spot=math.nan))
    assert not r.spot_ok(snap(spot=math.inf))
    assert not r.spot_ok(snap(spot_ts=math.nan))


def test_spot_from_the_future_is_not_ok() -> None:
    r = mgr()
    assert not r.spot_ok(snap(ts=1100.0, spot_ts=1101.0))  # look-ahead
    assert r.spot_ok(snap(ts=1100.0, spot_ts=1100.0 + 1e-12))  # float noise is tolerated


# --------------------------------------------------------------------------- book sanity


def test_book_ok_spread_limit_in_ticks() -> None:
    r = mgr()
    assert r.book_ok(snap(up=(0.49, 0.51), down=(0.49, 0.51)), 6)
    assert r.book_ok(snap(up=(0.47, 0.53), down=(0.47, 0.53)), 6)  # 6 ticks: 0.06 (with fp noise)
    assert not r.book_ok(snap(up=(0.46, 0.53), down=(0.47, 0.54)), 6)  # 7 ticks
    assert r.book_ok(snap(up=(0.46, 0.53), down=(0.47, 0.54)), 7)
    assert not r.book_ok(snap(up=(0.49, 0.51)), 1)  # 2 ticks > 1


def test_book_ok_needs_both_books_two_sided() -> None:
    r = mgr()
    assert not r.book_ok(snap(up=(None, 0.51)), 6)
    assert not r.book_ok(snap(up=(0.49, None)), 6)
    assert not r.book_ok(snap(down=(None, 0.51)), 6)
    assert not r.book_ok(snap(down=(0.49, None)), 6)
    assert not r.book_ok(snap(up=(None, None), down=(None, None)), 6)


def test_book_ok_rejects_crossed_and_locked_books() -> None:
    r = mgr()
    assert not r.book_ok(snap(up=(0.52, 0.50)), 6)  # crossed
    assert not r.book_ok(snap(up=(0.50, 0.50)), 6)  # locked
    assert not r.book_ok(snap(down=(0.55, 0.50)), 6)  # the DOWN book alone is crossed


def test_book_ok_rejects_prices_outside_unit_interval() -> None:
    r = mgr()
    assert not r.book_ok(snap(up=(0.98, 1.2)), 50)
    assert not r.book_ok(snap(up=(-0.2, 0.02)), 50)
    assert not r.book_ok(snap(up=(math.nan, 0.5)), 50)
    assert r.book_ok(snap(up=(0.0, 0.02), down=(0.98, 1.0)), 6)  # the extremes themselves are fine


def test_books_quotable_matches_book_ok() -> None:
    r = mgr()
    rng = random.Random(3)
    for _ in range(300):
        mid = rng.uniform(0.0, 1.0)
        half = rng.choice([0.005, 0.01, 0.02, 0.04, -0.01, 0.0])
        s = snap(
            up=(round(mid - half, 2), round(mid + half, 2)),
            down=(round(1 - mid - half, 2), round(1 - mid + half, 2)),
        )
        assert r.book_ok(s, 4) == books_quotable(s, 4)


# --------------------------------------------------------------------------- capital


def test_capital_ok_boundaries() -> None:
    r = mgr()  # caps 1500 / 5000
    assert r.capital_ok(1000.0, 3000.0, 500.0)  # market capital hits 1500 exactly
    assert not r.capital_ok(1000.0, 3000.0, 500.01)
    assert r.capital_ok(100.0, 4800.0, 200.0)  # total hits 5000 exactly
    assert not r.capital_ok(100.0, 4800.0, 200.5)
    assert r.capital_ok(0.0, 0.0, 0.0)
    assert not r.capital_ok(1600.0, 1600.0, 0.0)  # already over the market cap
    assert not r.capital_ok(100.0, 5100.0, 0.0)  # already over the total cap


def test_capital_headroom() -> None:
    r = mgr()
    assert r.capital_headroom(1000.0, 3000.0) == 500.0  # min(1500-1000, 5000-3000)
    assert r.capital_headroom(100.0, 4900.0) == 100.0  # total binds
    assert r.capital_headroom(1600.0, 1600.0) == 0.0  # over a cap -> clipped at 0
    assert r.capital_headroom(0.0, 0.0) == 1500.0


def test_capital_inputs_are_validated() -> None:
    r = mgr()
    for args in [(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (math.nan, 0.0, 0.0)]:
        with pytest.raises(ValueError):
            r.capital_ok(*args)
    with pytest.raises(ValueError):
        r.capital_headroom(-1.0, 0.0)


def test_capital_headroom_is_consistent_with_capital_ok() -> None:
    r = mgr(max_capital_per_market_usd=800.0, max_total_capital_usd=1200.0)
    rng = random.Random(11)
    for _ in range(500):
        m, t = rng.uniform(0, 900), rng.uniform(0, 1400)
        h = r.capital_headroom(m, t)
        assert h >= 0.0
        assert r.capital_ok(m, t, h) or max(m - 800.0, t - 1200.0) > 0.0
        assert not r.capital_ok(m, t, h + 0.01)


# --------------------------------------------------------------------------- rate limiter


def test_rate_limiter_grants_up_to_the_cap_per_window() -> None:
    r = mgr(max_orders_per_second=3.0)
    assert r.allow_orders(0.0, 2) == 2
    assert r.allow_orders(0.0, 2) == 1  # only one slot left
    assert r.allow_orders(0.5, 5) == 0
    assert r.allow_orders(0.999, 1) == 0  # the events at t=0 are 0.999 s old: still inside
    assert r.allow_orders(1.0, 5) == 3  # events at t=0 are exactly 1 s old: expired
    assert r.allow_orders(1.5, 1) == 0


def test_rate_limiter_window_slides() -> None:
    r = mgr(max_orders_per_second=4.0)
    assert r.allow_orders(0.0, 2) == 2
    assert r.allow_orders(0.6, 2) == 2  # window now holds 4
    assert r.allow_orders(0.9, 1) == 0
    # t=1.0: the 2 events at 0.0 expired, the 2 at 0.6 remain -> capacity 2
    assert r.allow_orders(1.0, 4) == 2
    # t=1.6: the 2 at 0.6 expired, the 2 at 1.0 remain -> capacity 2
    assert r.allow_orders(1.6, 4) == 2
    # t=2.0: the 2 at 1.0 expired, the 2 at 1.6 remain
    assert r.orders_in_window(2.0) == 2


def test_cancels_are_never_blocked_but_count_toward_the_limit() -> None:
    r = mgr(max_orders_per_second=3.0)
    r.record_cancels(0.0, 100)  # never refused, even far above the cap
    assert r.orders_in_window(0.0) == 100
    assert r.allow_orders(0.5, 1) == 0  # the cancels fill the window
    assert r.allow_orders(0.999, 1) == 0
    assert r.allow_orders(1.0, 3) == 3  # and expire after a second
    # Cancels also count when orders came first: 2 orders + cancels leave room for none.
    r2 = mgr(max_orders_per_second=3.0)
    assert r2.allow_orders(0.0, 2) == 2
    r2.record_cancels(0.1, 1)
    assert r2.allow_orders(0.2, 1) == 0


def test_fractional_rate_is_floored() -> None:
    r = mgr(max_orders_per_second=2.9)
    assert r.allow_orders(0.0, 10) == 2


def test_float_noise_at_the_window_edge() -> None:
    r = mgr(max_orders_per_second=1.0)
    assert r.allow_orders(0.1, 1) == 1
    assert r.allow_orders(1.1, 1) == 1  # 1.1 - 1.0 = 0.10000000000000009 > 0.1: expired
    assert r.allow_orders(2.2, 1) == 1  # 2.2 - 1.0 = 1.2000000000000002; event at 1.1 expired
    r2 = mgr(max_orders_per_second=1.0)
    assert r2.allow_orders(0.2, 1) == 1
    assert r2.allow_orders(1.2, 1) == 1  # 1.2 - 1.0 = 0.19999999999999996 < 0.2, still expired


def test_rate_limiter_validation() -> None:
    r = mgr()
    with pytest.raises(ValueError):
        r.allow_orders(0.0, -1)
    with pytest.raises(ValueError):
        r.record_cancels(0.0, -1)
    with pytest.raises(ValueError, match="finite"):
        r.allow_orders(math.nan, 1)
    r.allow_orders(10.0, 1)
    with pytest.raises(ValueError, match="backwards"):
        r.allow_orders(9.0, 1)
    with pytest.raises(ValueError, match="backwards"):
        r.record_cancels(9.0, 1)
    assert r.allow_orders(10.0, 0) == 0  # n = 0 is fine and records nothing
    assert r.orders_in_window(10.0) == 1


def test_rate_limiter_matches_a_brute_force_window_count() -> None:
    rng = random.Random(5)
    cap = 7
    r = mgr(max_orders_per_second=float(cap))
    events: list[float] = []  # timestamps of every recorded order or cancel
    t = 0.0
    for _ in range(2000):
        t += rng.choice([0.0, 0.01, 0.1, 0.25, 0.5, 1.3])
        n = rng.randint(0, 6)
        # Brute force: events in the window (t - 1, t]; an event exactly 1 s old has expired.
        before = sum(1 for e in events if e > t - 1.0 + 1e-9)
        assert r.orders_in_window(t) == before
        if rng.random() < 0.3:
            r.record_cancels(t, n)  # never refused, whatever the window holds
            events.extend([t] * n)
        else:
            granted = r.allow_orders(t, n)
            # The grant is exactly what fits under the cap, and the window never rises above
            # max(cap, what cancels already put there).
            assert granted == min(n, max(0, cap - before))
            assert before + granted <= max(cap, before)
            events.extend([t] * granted)
