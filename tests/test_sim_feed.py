"""Tests for abc_trading.sim.feed (synthetic Up/Down market feed, DESIGN section 6).

Hand-worked numbers are shown in comments. Statistical tests use fixed seeds with loose bounds
(several standard errors) so they are deterministic and not brittle.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

import abc_trading
from abc_trading.config import BotConfig, SimConfig
from abc_trading.sim import SyntheticFeed
from abc_trading.sim import feed as feed_mod
from abc_trading.sim.feed import (
    _geometric,
    _poisson,
    _quote_ticks,
    _swept_levels,
    _up_probability,
    _winner_of,
)
from abc_trading.types import (
    FeedEvent,
    MarketResolved,
    MarketSnapshot,
    Outcome,
    Side,
    Trade,
)

TICK = 0.01
SECONDS_PER_YEAR = 31_557_600.0


# --------------------------------------------------------------------------- helpers


def make_cfg(**kw: Any) -> BotConfig:
    base: dict[str, Any] = {"seed": 11, "window_seconds": 60, "n_windows": 3, "tick_seconds": 1.0}
    base.update(kw)
    return BotConfig(sim=dataclasses.replace(SimConfig(), **base))


def run(**kw: Any) -> list[FeedEvent]:
    return list(SyntheticFeed(make_cfg(**kw)))


def snaps_of(events: list[FeedEvent]) -> list[MarketSnapshot]:
    return [e for e in events if isinstance(e, MarketSnapshot)]


def resolved_of(events: list[FeedEvent]) -> list[MarketResolved]:
    return [e for e in events if isinstance(e, MarketResolved)]


def by_market(events: list[FeedEvent]) -> dict[str, list[MarketSnapshot]]:
    out: dict[str, list[MarketSnapshot]] = defaultdict(list)
    for s in snaps_of(events):
        out[s.market.market_id].append(s)
    return dict(out)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9


def on_grid(price: float, tick: float) -> bool:
    ticks = price / tick
    return abs(ticks - round(ticks)) < 1e-6


@pytest.fixture(scope="module")
def long_snaps() -> list[MarketSnapshot]:
    """10 windows x 300 one-second ticks x BTC+ETH, shared by the statistical tests."""
    return snaps_of(run(window_seconds=300, n_windows=10))


def check_book_invariants(snap: MarketSnapshot, tick: float, n_levels: int) -> None:
    for book in (snap.up_book, snap.down_book):
        assert book.bids and book.asks
        assert len(book.bids) <= n_levels and len(book.asks) <= n_levels
        assert not book.is_crossed
        assert book.bids[0].price < book.asks[0].price
        for ladder in (book.bids, book.asks):
            for lv in ladder:
                assert 0.0 < lv.price < 1.0
                assert lv.size > 0.0
                assert on_grid(lv.price, tick)
        for hi, lo in zip(book.bids, book.bids[1:], strict=False):
            assert close(hi.price - lo.price, tick)
        for lo, hi in zip(book.asks, book.asks[1:], strict=False):
            assert close(hi.price - lo.price, tick)
    up, down = snap.up_book, snap.down_book
    assert len(down.bids) == len(up.asks) and len(down.asks) == len(up.bids)
    for u, d in zip(up.asks, down.bids, strict=True):
        assert close(d.price, 1.0 - u.price) and d.size == u.size
    for u, d in zip(up.bids, down.asks, strict=True):
        assert close(d.price, 1.0 - u.price) and d.size == u.size


def check_print_mirrors(snap: MarketSnapshot) -> None:
    trades = snap.trades
    assert len(trades) % 2 == 0
    for up, down in zip(trades[0::2], trades[1::2], strict=True):
        assert up.token is Outcome.UP and down.token is Outcome.DOWN
        assert up.size == down.size and up.size > 0.0
        assert down.aggressor is up.aggressor.opposite
        assert close(down.price, 1.0 - up.price)
    for t in trades:
        assert t.ts == snap.ts
        assert 0.0 < t.price < 1.0


def check_event_stream(
    events: list[FeedEvent], feed: SyntheticFeed, window_s: int, tick_s: float
) -> None:
    """Cadence, ordering and resolution invariants that hold for every configuration."""
    ts = [e.ts for e in events]
    assert ts == sorted(ts)
    per_market = by_market(events)
    k_per_window = round(window_s / tick_s)
    assert set(per_market) == {m.market_id for m in feed.markets}
    resolutions = {r.market_id: i for i, r in enumerate(events) if isinstance(r, MarketResolved)}
    assert len(resolutions) == len(feed.markets)  # exactly one per market
    for spec in feed.markets:
        snaps = per_market[spec.market_id]
        assert len(snaps) == k_per_window
        for k, s in enumerate(snaps):
            assert s.ts == spec.start_ts + k * tick_s
            assert s.market == spec
            assert s.ts < spec.end_ts
        idx = resolutions[spec.market_id]
        res = events[idx]
        assert isinstance(res, MarketResolved)
        assert res.ts == spec.end_ts
        assert res.winner is feed.winner(spec.market_id)
        assert events.index(snaps[-1]) < idx
    # at equal ts: all resolutions first (sorted by market_id), then snapshots (sorted)
    groups: dict[float, list[FeedEvent]] = defaultdict(list)
    for e in events:
        groups[e.ts].append(e)
    for group in groups.values():
        kinds = [isinstance(e, MarketResolved) for e in group]
        assert kinds == sorted(kinds, reverse=True)
        r_ids = [e.market_id for e in group if isinstance(e, MarketResolved)]
        s_ids = [e.market.market_id for e in group if isinstance(e, MarketSnapshot)]
        assert r_ids == sorted(r_ids) and s_ids == sorted(s_ids)


# --------------------------------------------------------------------------- markets list


def test_markets_layout_and_ids() -> None:
    feed = SyntheticFeed(make_cfg(window_seconds=60, n_windows=3))
    # ts0: 1_700_000_000 / 60 = 28_333_333.33 -> nearest multiple 28_333_333 * 60 = 1_699_999_980
    starts = [1_699_999_980.0, 1_700_000_040.0, 1_700_000_100.0]
    expected = [(f"{a}-{int(s)}-60s", a, s) for s in starts for a in ("BTC", "ETH")]
    got = [(m.market_id, m.asset, m.start_ts) for m in feed.markets]
    assert got == expected
    assert got[0][0] == "BTC-1699999980-60s"
    for m in feed.markets:
        assert m.end_ts == m.start_ts + 60.0 and m.duration == 60.0
        assert m.tick_size == 0.01 and m.min_order_size == 5.0


def test_markets_carry_tick_and_min_size_from_config() -> None:
    feed = SyntheticFeed(make_cfg(tick_size=0.005, min_order_size=7.0, n_windows=1))
    assert [(m.tick_size, m.min_order_size) for m in feed.markets] == [(0.005, 7.0)] * 2


@pytest.mark.parametrize(
    ("window", "first_start"),
    [
        # 1_700_000_000 / 900 = 1_888_888.89 -> 1_888_889 * 900 = 1_700_000_100
        (900, 1_700_000_100.0),
        # 1_700_000_000 / 300 = 5_666_666.67 -> 5_666_667 * 300 = 1_700_000_100
        (300, 1_700_000_100.0),
        # 1_700_000_000 / 10 is exact
        (10, 1_700_000_000.0),
    ],
)
def test_epoch_is_rounded_to_a_window_multiple(window: int, first_start: float) -> None:
    feed = SyntheticFeed(make_cfg(window_seconds=window, n_windows=2))
    assert feed.markets[0].start_ts == first_start
    assert first_start % window == 0
    assert feed.markets[2].start_ts == first_start + window  # back-to-back windows


def test_markets_property_returns_a_copy() -> None:
    feed = SyntheticFeed(make_cfg())
    first = feed.markets
    first.clear()
    assert len(feed.markets) == 6


def test_assets_order_follows_config_for_markets() -> None:
    feed = SyntheticFeed(make_cfg(assets=("ETH", "BTC"), n_windows=1))
    assert [m.asset for m in feed.markets] == ["ETH", "BTC"]


# --------------------------------------------------------------------------- determinism


def test_two_iterations_are_identical() -> None:
    feed = SyntheticFeed(make_cfg())
    assert list(feed) == list(feed)


def test_two_instances_with_same_config_are_identical() -> None:
    assert run() == run()


def test_different_seeds_differ() -> None:
    a, b = run(seed=1), run(seed=2)
    assert a != b
    assert [s.spot for s in snaps_of(a)] != [s.spot for s in snaps_of(b)]


def test_partially_consumed_iteration_does_not_affect_the_next() -> None:
    feed = SyntheticFeed(make_cfg())
    it = iter(feed)
    for _ in range(25):
        next(it)
    assert list(feed) == run()


def test_feed_does_not_touch_global_rng_or_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> float:
        raise AssertionError("wall clock used")

    monkeypatch.setattr(time, "time", boom)
    monkeypatch.setattr(time, "monotonic", boom)
    random.seed(123)
    state = random.getstate()
    first = run()
    assert random.getstate() == state  # global RNG untouched
    random.seed(999)
    assert run() == first  # and not influenced by it either


def test_events_identical_across_processes_with_different_hash_seeds() -> None:
    code = (
        "import dataclasses, hashlib\n"
        "from abc_trading.config import BotConfig, SimConfig\n"
        "from abc_trading.sim import SyntheticFeed\n"
        "sim = dataclasses.replace(SimConfig(), seed=5, window_seconds=20, n_windows=2)\n"
        "ev = list(SyntheticFeed(BotConfig(sim=sim)))\n"
        "print(hashlib.sha256(repr(ev).encode()).hexdigest())\n"
    )
    src = str(Path(abc_trading.__file__).resolve().parent.parent)
    digests = set()
    for hash_seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": src}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    sim = dataclasses.replace(SimConfig(), seed=5, window_seconds=20, n_windows=2)
    local = hashlib.sha256(repr(list(SyntheticFeed(BotConfig(sim=sim)))).encode()).hexdigest()
    assert digests == {local}


def test_first_windows_unchanged_when_more_windows_are_requested() -> None:
    short, long_ = run(n_windows=2), run(n_windows=4)
    assert long_[: len(short)] == short
    assert SyntheticFeed(make_cfg(n_windows=2)).winner("BTC-1699999980-60s") is SyntheticFeed(
        make_cfg(n_windows=4)
    ).winner("BTC-1699999980-60s")


def test_one_asset_is_unaffected_by_adding_another() -> None:
    btc_only = run(assets=("BTC",))
    both = [e for e in run(assets=("BTC", "ETH")) if _asset_of(e) == "BTC"]
    assert btc_only == both


def _asset_of(e: FeedEvent) -> str:
    market_id = e.market_id if isinstance(e, MarketResolved) else e.market.market_id
    return market_id.split("-")[0]


def test_knobs_leave_unrelated_randomness_alone() -> None:
    base = snaps_of(run(informed_flow=0.0, uninformed_trades_per_sec=1.0))
    # more informed flow only adds sweeps in front of the identical uninformed prints
    swept = snaps_of(run(informed_flow=1.0, uninformed_trades_per_sec=1.0))
    assert [s.spot for s in base] == [s.spot for s in swept]
    assert [s.up_book for s in base] == [s.up_book for s in swept]
    assert [s.down_book for s in base] == [s.down_book for s in swept]
    n_extra = 0
    for b, s in zip(base, swept, strict=True):
        assert s.trades[len(s.trades) - len(b.trades) :] == b.trades
        n_extra += len(s.trades) - len(b.trades)
    assert n_extra > 0
    # the uninformed rate and the depth do not move prices or spreads
    quiet = snaps_of(run(uninformed_trades_per_sec=0.0))
    busy = snaps_of(run(uninformed_trades_per_sec=3.0))
    assert [s.up_book for s in quiet] == [s.up_book for s in busy]
    shallow = snaps_of(run(depth_mean_shares=50.0))
    deep = snaps_of(run(depth_mean_shares=900.0))
    for a, b in zip(shallow, deep, strict=True):
        assert a.spot == b.spot
        assert [lv.price for lv in a.up_book.bids] == [lv.price for lv in b.up_book.bids]
        assert [lv.price for lv in a.up_book.asks] == [lv.price for lv in b.up_book.asks]
        # same draws, scaled sizes: sizes are linear in depth_mean_shares
        for la, lb in zip(a.up_book.bids, b.up_book.bids, strict=True):
            assert lb.size == pytest.approx(la.size * 18.0, rel=1e-9)


# --------------------------------------------------------------------------- event stream


def test_event_stream_cadence_order_and_resolution() -> None:
    cfg = make_cfg(window_seconds=20, n_windows=3)
    check_event_stream(list(SyntheticFeed(cfg)), SyntheticFeed(cfg), 20, 1.0)


@pytest.mark.parametrize("tick_seconds", [0.5, 2.0, 5.0])
def test_event_stream_with_other_tick_lengths(tick_seconds: float) -> None:
    cfg = make_cfg(window_seconds=20, n_windows=2, tick_seconds=tick_seconds)
    check_event_stream(list(SyntheticFeed(cfg)), SyntheticFeed(cfg), 20, tick_seconds)


def test_resolutions_precede_next_window_snapshots_and_sort_by_market_id() -> None:
    # reversed asset order must not change the id-sorted order of simultaneous events
    events = run(window_seconds=20, n_windows=3, assets=("ETH", "BTC"))
    boundary = 1_700_000_000.0 + 20.0  # 1_700_000_000 is a multiple of 20: window 1 starts here
    at = [e for e in events if e.ts == boundary]
    assert [type(e).__name__ for e in at] == [
        "MarketResolved",
        "MarketResolved",
        "MarketSnapshot",
        "MarketSnapshot",
    ]
    assert [e.market_id for e in at[:2] if isinstance(e, MarketResolved)] == [
        "BTC-1700000000-20s",
        "ETH-1700000000-20s",
    ]
    assert [e.market.market_id for e in at[2:] if isinstance(e, MarketSnapshot)] == [
        "BTC-1700000020-20s",
        "ETH-1700000020-20s",
    ]
    assert [type(e).__name__ for e in events[-2:]] == ["MarketResolved", "MarketResolved"]


def test_snapshot_fields_and_first_snapshot() -> None:
    cfg = make_cfg(window_seconds=20, n_windows=3)
    feed = SyntheticFeed(cfg)
    for spec in feed.markets:
        snaps = by_market(list(feed))[spec.market_id]
        first = snaps[0]
        assert first.ts == spec.start_ts
        assert first.ref_price == first.spot  # ref is the spot at window start
        assert first.trades == ()  # nothing has happened before the first snapshot
        for s in snaps:
            assert s.spot is not None and s.spot > 0.0
            assert s.spot_ts == s.ts
            assert s.ref_price == first.ref_price
    first_windows = [s for s in snaps_of(run(window_seconds=20)) if s.ts == 1_700_000_000.0]
    assert {s.market.asset: s.spot for s in first_windows} == {"BTC": 60_000.0, "ETH": 3_000.0}


# --------------------------------------------------------------------------- resolution


def test_winner_is_up_iff_spot_end_at_or_above_ref() -> None:
    assert _winner_of([100.0, 101.0, 100.0]) is Outcome.UP  # tie resolves UP
    assert _winner_of([100.0, 50.0, 100.01]) is Outcome.UP
    assert _winner_of([100.0, 150.0, 99.99]) is Outcome.DOWN


def test_resolved_winner_matches_the_spot_path() -> None:
    feed = SyntheticFeed(make_cfg(window_seconds=30, n_windows=6))
    events = list(feed)
    per_market = by_market(events)
    winners = {r.market_id: r.winner for r in resolved_of(events)}
    seen_both = set()
    for spec in feed.markets:
        # spot at window end is the next window's first sample (and ref) of the same asset
        nxt = next(
            (m for m in feed.markets if m.asset == spec.asset and m.start_ts == spec.end_ts), None
        )
        assert winners[spec.market_id] is feed.winner(spec.market_id)
        if nxt is None:
            continue
        ref = per_market[spec.market_id][0].ref_price
        spot_end = per_market[nxt.market_id][0].spot
        assert ref is not None and spot_end is not None
        assert per_market[nxt.market_id][0].ref_price == spot_end
        expected = Outcome.UP if spot_end >= ref else Outcome.DOWN
        assert winners[spec.market_id] is expected
        seen_both.add(expected)
    assert seen_both == {Outcome.UP, Outcome.DOWN}


def test_winner_is_available_before_iterating_and_rejects_unknown_ids() -> None:
    feed = SyntheticFeed(make_cfg())
    ids = [m.market_id for m in feed.markets]
    before = {i: feed.winner(i) for i in ids}
    assert {r.market_id: r.winner for r in resolved_of(list(feed))} == before
    with pytest.raises(ValueError, match="unknown market_id"):
        feed.winner("BTC-1-60s")


def test_last_window_winner_agrees_with_a_longer_run() -> None:
    # In the longer run the short run's last window has a successor, whose first sample is its
    # end spot: so this cross-checks the winner of a final window against an independent view.
    short = SyntheticFeed(make_cfg(window_seconds=20, n_windows=12))
    long_ = SyntheticFeed(make_cfg(window_seconds=20, n_windows=13))
    per_market = by_market(list(long_))
    for asset in ("BTC", "ETH"):
        last = next(
            m
            for m in short.markets
            if m.asset == asset and m.start_ts == short.markets[-1].start_ts
        )
        nxt = next(m for m in long_.markets if m.asset == asset and m.start_ts == last.end_ts)
        ref = per_market[last.market_id][0].ref_price
        spot_end = per_market[nxt.market_id][0].spot
        assert ref is not None and spot_end is not None
        assert short.winner(last.market_id) is (Outcome.UP if spot_end >= ref else Outcome.DOWN)


# --------------------------------------------------------------------------- books


def test_first_snapshot_book_with_no_noise_and_unit_spread() -> None:
    # spot == ref -> ln(1) = 0 -> p = Phi(0) = 0.5 -> mid = 50 ticks (no noise, so no shift).
    # spread = 1 tick: bid = 50 - ceil(1/2) = 49, ask = 49 + 1 = 50. DOWN is the mirror:
    # down bid = 1 - 0.50 = 0.50, down ask = 1 - 0.49 = 0.51.
    events = run(pricing_noise_ticks=0.0, mean_spread_ticks=1.0, n_windows=2)
    firsts = [s for s in snaps_of(events) if s.ts == s.market.start_ts]
    assert len(firsts) == 4  # 2 windows x 2 assets; every window starts with spot == ref
    for s in firsts:
        assert [lv.price for lv in s.up_book.bids] == [0.49, 0.48, 0.47, 0.46, 0.45]
        assert [lv.price for lv in s.up_book.asks] == [0.50, 0.51, 0.52, 0.53, 0.54]
        assert [lv.price for lv in s.down_book.bids] == [0.50, 0.49, 0.48, 0.47, 0.46]
        assert [lv.price for lv in s.down_book.asks] == [0.51, 0.52, 0.53, 0.54, 0.55]
        assert s.up_book.spread == pytest.approx(0.01)


def test_up_probability_hand_worked() -> None:
    # sigma = 0.55 / sqrt(31_557_600) = 0.55 / 5617.615 = 9.7906e-5 per sqrt(second)
    # spot 60_060 vs ref 60_000: ln(1.001) = 9.9950e-4; tau = 100 s -> sigma*sqrt(tau) = 9.7906e-4
    # z = 9.9950e-4 / 9.7906e-4 = 1.02087 and Phi(1.02087) = 0.84634
    sigma = 0.55 / math.sqrt(SECONDS_PER_YEAR)
    assert _up_probability(60_060.0, 60_000.0, sigma, 100.0) == pytest.approx(0.84634, abs=2e-5)
    assert _up_probability(60_000.0, 60_000.0, sigma, 100.0) == 0.5
    below = _up_probability(60_000.0 / 1.001, 60_000.0, sigma, 100.0)
    assert below == pytest.approx(1.0 - _up_probability(60_060.0, 60_000.0, sigma, 100.0), abs=1e-6)
    # a larger market vol multiplier flattens the response toward 0.5
    assert _up_probability(60_060.0, 60_000.0, 2 * sigma, 100.0) < 0.84634


def _independent_mid_ticks(
    spot: float, ref: float, ts: float, end: float, vol: float, mult: float
) -> int:
    """Re-derivation of the market mid (no noise, no lag) for comparison with the feed."""
    sigma = vol / math.sqrt(SECONDS_PER_YEAR) * mult
    z = math.log(spot / ref) / (sigma * math.sqrt(end - ts))
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return min(max(math.floor(p / TICK + 0.5), 1), 99)


def _expected_unit_spread_quotes(mid: int) -> tuple[float, float]:
    # spread 1: bid = mid - 1 tick, ask = mid; at mid == 1 tick the bid would be 0 so both shift up
    return (0.01, 0.02) if mid == 1 else (round((mid - 1) * TICK, 6), round(mid * TICK, 6))


def test_up_quotes_follow_the_market_model_exactly_without_noise_or_lag() -> None:
    mult = 0.7
    events = run(
        window_seconds=120,
        n_windows=2,
        pricing_noise_ticks=0.0,
        market_lag_seconds=0.0,
        mean_spread_ticks=1.0,
        market_vol_multiplier=mult,
    )
    vols = {"BTC": 0.55, "ETH": 0.75}
    mids: set[int] = set()
    for s in snaps_of(events):
        assert s.spot is not None and s.ref_price is not None
        mid = _independent_mid_ticks(
            s.spot, s.ref_price, s.ts, s.market.end_ts, vols[s.market.asset], mult
        )
        mids.add(mid)
        bid, ask = _expected_unit_spread_quotes(mid)
        assert s.up_book.bids[0].price == bid
        assert s.up_book.asks[0].price == ask
    assert len(mids) > 15  # the mid actually moves around, so the test is not vacuous


@pytest.mark.parametrize(("lag", "lag_ticks"), [(2.0, 2), (2.5, 3), (1.0, 1)])
def test_market_prices_off_stale_spot_but_never_before_window_start(
    lag: float, lag_ticks: int
) -> None:
    # latest sample at or before ts - lag, i.e. k - ceil(lag/tick), floored at the window start
    mult = 0.3
    events = run(
        window_seconds=60,
        n_windows=2,
        pricing_noise_ticks=0.0,
        market_lag_seconds=lag,
        mean_spread_ticks=1.0,
        market_vol_multiplier=mult,
        assets=("BTC",),
    )
    n_differs_from_unlagged = 0
    for snaps in by_market(events).values():
        for k, s in enumerate(snaps):
            lagged = snaps[max(0, k - lag_ticks)].spot
            assert lagged is not None and s.ref_price is not None and s.spot is not None
            args = (s.ts, s.market.end_ts, 0.55, mult)
            mid = _independent_mid_ticks(lagged, s.ref_price, *args)
            assert (
                s.up_book.bids[0].price,
                s.up_book.asks[0].price,
            ) == _expected_unit_spread_quotes(mid)
            if k < lag_ticks:
                assert lagged == s.ref_price and s.up_book.asks[0].price == 0.5  # sees ref
            unlagged = _independent_mid_ticks(s.spot, s.ref_price, *args)
            n_differs_from_unlagged += unlagged != mid
    assert n_differs_from_unlagged > 10  # the lag is visible in the prices


def test_pricing_noise_is_a_stationary_ar1_with_the_configured_std() -> None:
    # A huge market vol multiplier pins p = Phi(~0) = 0.5, so with spread 1 the UP ask in ticks
    # is round(50 + noise / tick): the noise is observable. Noise std = 3 ticks, AR coef 0.9.
    events = run(
        window_seconds=600,
        n_windows=10,
        assets=("BTC",),
        market_vol_multiplier=1e6,
        market_lag_seconds=0.0,
        pricing_noise_ticks=3.0,
        mean_spread_ticks=1.0,
    )
    series = {
        mid: [round(s.up_book.asks[0].price / TICK) - 50 for s in snaps]
        for mid, snaps in by_market(events).items()
    }
    flat = [x for xs in series.values() for x in xs]
    assert len(flat) == 6000
    # n_eff = n (1 - phi^2) / (1 + phi^2) ~ 630 -> se of the std ~ 3%, of the mean ~ 0.12
    assert statistics.fmean(flat) == pytest.approx(0.0, abs=0.6)
    var = statistics.fmean(x * x for x in flat)
    assert math.sqrt(var) == pytest.approx(3.0, rel=0.1)  # rounding adds only 1/12 to var 9

    def autocorr(lag: int) -> float:
        pairs = [(xs[i], xs[i + lag]) for xs in series.values() for i in range(len(xs) - lag)]
        return statistics.fmean(a * b for a, b in pairs) / var

    assert autocorr(1) == pytest.approx(0.9, abs=0.03)
    assert autocorr(5) == pytest.approx(0.9**5, abs=0.08)  # 0.59
    assert autocorr(20) == pytest.approx(0.9**20, abs=0.12)  # 0.12


def test_no_noise_means_a_smooth_stationary_market_at_one_half() -> None:
    events = run(
        window_seconds=60,
        n_windows=2,
        assets=("BTC",),
        market_vol_multiplier=1e6,
        pricing_noise_ticks=0.0,
        mean_spread_ticks=1.0,
    )
    assert {s.up_book.asks[0].price for s in snaps_of(events)} == {0.5}


def test_quote_ticks_hand_table() -> None:
    n = 100
    assert _quote_ticks(50, 1, n) == (49, 50)  # bid = 50 - ceil(1/2)
    assert _quote_ticks(50, 2, n) == (49, 51)
    assert _quote_ticks(50, 3, n) == (48, 51)  # bid = 50 - ceil(3/2) = 48
    assert _quote_ticks(50, 4, n) == (48, 52)
    assert _quote_ticks(1, 1, n) == (1, 2)  # bid 0 -> shifted up by 1
    assert _quote_ticks(1, 2, n) == (1, 3)  # bid 0 -> shifted up by 1
    assert _quote_ticks(1, 3, n) == (1, 4)  # bid -1 -> shifted up by 2
    assert _quote_ticks(99, 1, n) == (98, 99)  # fits, no shift
    assert _quote_ticks(99, 2, n) == (97, 99)  # bid 98, ask 100 -> both shifted down by 1
    assert _quote_ticks(99, 4, n) == (95, 99)  # bid 97, ask 101 -> shifted down by 2
    assert _quote_ticks(50, 500, n) == (1, 99)  # spread capped at n - 2 = 98
    for mid in range(1, 100):
        for spread in (1, 2, 3, 5, 40, 98, 99, 1000):
            bid, ask = _quote_ticks(mid, spread, n)
            assert 1 <= bid < ask <= 99
            assert ask - bid == min(spread, 98)


def test_quote_ticks_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        _quote_ticks(0, 1, 100)
    with pytest.raises(ValueError):
        _quote_ticks(100, 1, 100)
    with pytest.raises(ValueError):
        _quote_ticks(50, 0, 100)


def test_book_invariants_on_default_style_config() -> None:
    events = run(window_seconds=120, n_windows=3)
    snaps = snaps_of(events)
    assert len(snaps) == 720
    for s in snaps:
        check_book_invariants(s, TICK, 5)
        check_print_mirrors(s)


def test_book_depth_and_spread_statistics(long_snaps: list[MarketSnapshot]) -> None:
    # defaults: mean_spread_ticks = 1.5, depth_mean_shares = 400
    snaps = long_snaps
    spreads = [round((s.up_book.asks[0].price - s.up_book.bids[0].price) / TICK) for s in snaps]
    assert min(spreads) == 1
    assert max(spreads) >= 4  # geometric tail
    # extra ticks ~ Geometric(mean 0.5): P(spread == 1) = 1 - q = 1 / (1 + 0.5) = 2/3 (se 0.006)
    mean_spread = statistics.fmean(spreads)
    assert mean_spread == pytest.approx(1.5, abs=0.08)
    assert spreads.count(1) / len(spreads) == pytest.approx(2 / 3, abs=0.04)
    sizes = [lv.size for s in snaps for lv in (*s.up_book.bids, *s.up_book.asks)]
    # lognormal with arithmetic mean 400, log-sd 0.5: CV 0.53, n ~ 60k -> se 0.2%
    assert statistics.fmean(sizes) == pytest.approx(400.0, rel=0.03)
    assert min(sizes) > 0.0


@pytest.mark.parametrize("mean_spread", [1.0, 3.0])
def test_mean_spread_knob(mean_spread: float) -> None:
    snaps = snaps_of(run(window_seconds=200, n_windows=4, mean_spread_ticks=mean_spread))
    spreads = [round((s.up_book.asks[0].price - s.up_book.bids[0].price) / TICK) for s in snaps]
    if mean_spread == 1.0:
        assert set(spreads) == {1}
    else:
        # Geometric extra mean 2.0: sd = sqrt(q) / (1 - q) = sqrt(2/3) / (1/3) = 2.45; n = 1600
        assert statistics.fmean(spreads) == pytest.approx(3.0, abs=0.35)


def test_books_at_the_price_extremes_shift_inward_and_drop_levels() -> None:
    # a tiny market vol multiplier saturates the market mid at 1 tick or 99 ticks
    events = run(
        window_seconds=60,
        n_windows=4,
        pricing_noise_ticks=0.0,
        mean_spread_ticks=1.0,
        market_vol_multiplier=0.01,
        assets=("BTC",),
    )
    high = low = 0
    for s in snaps_of(events):
        check_book_invariants(s, TICK, 5)
        up_bid, up_ask = s.up_book.bids[0].price, s.up_book.asks[0].price
        if up_ask == 0.99:  # mid 99: bid 98, ask 99; no ask level at 1.00
            high += 1
            assert [lv.price for lv in s.up_book.bids] == [0.98, 0.97, 0.96, 0.95, 0.94]
            assert [lv.price for lv in s.up_book.asks] == [0.99]
            assert [lv.price for lv in s.down_book.bids] == [0.01]
            assert [lv.price for lv in s.down_book.asks] == [0.02, 0.03, 0.04, 0.05, 0.06]
        if up_bid == 0.01:  # mid 1: bid would be 0 -> shifted to bid 1 / ask 2
            low += 1
            assert up_ask == 0.02
            assert [lv.price for lv in s.up_book.bids] == [0.01]
            assert [lv.price for lv in s.up_book.asks] == [0.02, 0.03, 0.04, 0.05, 0.06]
            assert [lv.price for lv in s.down_book.bids] == [0.98, 0.97, 0.96, 0.95, 0.94]
            assert [lv.price for lv in s.down_book.asks] == [0.99]
    assert high > 10 and low > 10


# --------------------------------------------------------------------------- trades


def test_every_print_has_its_mirror_and_first_snapshot_has_none() -> None:
    for kw in (
        {"informed_flow": 1.0, "uninformed_trades_per_sec": 1.0},
        {"informed_flow": 0.0, "uninformed_trades_per_sec": 3.0},
        {"informed_flow": 0.5, "uninformed_trades_per_sec": 0.0, "market_vol_multiplier": 0.2},
    ):
        events = run(window_seconds=60, **kw)
        n_prints = 0
        for snaps in by_market(events).values():
            assert snaps[0].trades == ()
            for s in snaps:
                check_print_mirrors(s)
                n_prints += len(s.trades)
        assert n_prints > 100


def _touch_price(snap: MarketSnapshot, t: Trade) -> float:
    book = snap.book(t.token)
    price = book.best_ask if t.aggressor is Side.BUY else book.best_bid
    assert price is not None
    return price


def test_without_informed_flow_every_print_is_at_the_touch() -> None:
    events = run(window_seconds=120, informed_flow=0.0, uninformed_trades_per_sec=2.0)
    combos: Counter[tuple[Outcome, Side]] = Counter()
    for s in snaps_of(events):
        for t in s.trades:
            assert close(t.price, _touch_price(s, t))
            combos[(t.token, t.aggressor)] += 1
    total = sum(combos.values())
    assert total > 500
    assert len(combos) == 4  # both tokens, both aggressor sides
    for n in combos.values():  # each (token, side) gets a quarter of the prints
        assert n / total == pytest.approx(0.25, abs=0.03)


def test_no_flow_at_all_means_no_prints() -> None:
    events = run(informed_flow=0.0, uninformed_trades_per_sec=0.0)
    assert all(s.trades == () for s in snaps_of(events))


def test_uninformed_arrival_rate_scales_with_rate_and_tick() -> None:
    # each Poisson arrival is two prints (UP + DOWN mirror); expected arrivals per tick =
    # rate * tick_seconds = 2.0 * 1.0 = 2.0, and 1.0 * 2.0 = 2.0 for a 2 s tick.
    for rate, tick_s in ((2.0, 1.0), (1.0, 2.0)):
        events = run(
            window_seconds=200,
            n_windows=5,
            assets=("BTC",),
            informed_flow=0.0,
            uninformed_trades_per_sec=rate,
            tick_seconds=tick_s,
        )
        snaps = snaps_of(events)
        arrivals = [len(s.trades) // 2 for s in snaps if s.ts != s.market.start_ts]
        # Poisson(2): sd 1.41, n = 495..995 -> se <= 0.064; bound is 4 se
        assert statistics.fmean(arrivals) == pytest.approx(2.0, abs=0.26)
        assert statistics.pvariance(arrivals) == pytest.approx(2.0, abs=0.6)  # variance == mean
        sizes = [t.size for s in snaps for t in s.trades[0::2]]
        assert min(sizes) >= 1.0
        # lognormal, arithmetic mean 30, CV 0.66, n ~ 1000+ -> se ~ 2%
        assert statistics.fmean(sizes) == pytest.approx(30.0, rel=0.1)


def test_uninformed_sizes_are_floored_at_one_share(monkeypatch: pytest.MonkeyPatch) -> None:
    # shrink the size distribution so that the floor binds often (it practically never does at 30)
    monkeypatch.setattr(feed_mod, "UNINFORMED_SIZE_MEAN", 0.5)
    snaps = snaps_of(
        run(informed_flow=0.0, uninformed_trades_per_sec=2.0, window_seconds=100, assets=("BTC",))
    )
    sizes = [t.size for s in snaps for t in s.trades]
    assert len(sizes) > 300
    assert min(sizes) == 1.0
    assert sizes.count(1.0) > 100  # lognormal(mean 0.5, log-sd 0.6): P(size < 1) ~ 0.9


def _expected_sweeps(
    old: MarketSnapshot, new: MarketSnapshot
) -> dict[tuple[Outcome, Side, float], float]:
    """(token, aggressor, price) -> OLD displayed size of the level the feed should sweep."""
    out: dict[tuple[Outcome, Side, float], float] = {}
    new_bid, new_ask = new.up_book.bids[0].price, new.up_book.asks[0].price
    for lv in old.up_book.bids:
        if lv.price > new_bid + 1e-9:  # old best bid down to new best bid + 1 tick
            out[(Outcome.UP, Side.SELL, lv.price)] = lv.size
            out[(Outcome.DOWN, Side.BUY, round(1.0 - lv.price, 6))] = lv.size
    for lv in old.up_book.asks:
        if lv.price < new_ask - 1e-9:  # old best ask up to new best ask - 1 tick
            out[(Outcome.UP, Side.BUY, lv.price)] = lv.size
            out[(Outcome.DOWN, Side.SELL, round(1.0 - lv.price, 6))] = lv.size
    return out


@pytest.mark.parametrize("flow", [1.0, 0.5])
def test_informed_sweeps_hit_exactly_the_consumed_old_levels(flow: float) -> None:
    n_levels = 3
    events = run(
        window_seconds=120,
        n_windows=2,
        n_levels=n_levels,
        informed_flow=flow,
        uninformed_trades_per_sec=0.0,
        market_vol_multiplier=0.2,  # fast market -> multi-tick moves
        pricing_noise_ticks=2.0,
        annual_vol={"BTC": 3.0, "ETH": 4.0},
    )
    multi_level = capped = checked = 0
    for snaps in by_market(events).values():
        for old, new in zip(snaps, snaps[1:], strict=False):
            expected = _expected_sweeps(old, new)
            actual: dict[tuple[Outcome, Side, float], float] = {}
            for t in new.trades:
                key = (t.token, t.aggressor, round(t.price, 6))
                assert key not in actual  # one print per level
                actual[key] = t.size
            assert actual.keys() == expected.keys()
            for key, size in actual.items():
                # size = informed_flow * old displayed size * U(0.3, 1.0)
                assert flow * 0.3 * expected[key] - 1e-9 <= size <= flow * expected[key] + 1e-9
            checked += 1
            n_sell_up = sum(1 for k in actual if k[:2] == (Outcome.UP, Side.SELL))
            multi_level += n_sell_up >= 2
            drop = round((old.up_book.bids[0].price - new.up_book.bids[0].price) / TICK)
            if drop > n_levels:  # deeper than displayed: only the displayed levels can be swept
                capped += 1
                assert n_sell_up == n_levels
    assert checked > 200
    assert multi_level > 20
    assert capped > 0


def test_informed_sweep_prints_are_all_below_old_best_bid_or_above_old_best_ask() -> None:
    events = run(window_seconds=120, informed_flow=1.0, uninformed_trades_per_sec=0.0)
    for snaps in by_market(events).values():
        for old, new in zip(snaps, snaps[1:], strict=False):
            for t in new.trades:
                if t.token is Outcome.UP and t.aggressor is Side.SELL:
                    assert new.up_book.bids[0].price < t.price <= old.up_book.bids[0].price + 1e-9
                if t.token is Outcome.UP and t.aggressor is Side.BUY:
                    assert old.up_book.asks[0].price - 1e-9 <= t.price < new.up_book.asks[0].price


def test_sweeps_are_gated_by_informed_flow_and_precede_uninformed_prints() -> None:
    quiet = snaps_of(run(informed_flow=0.0, uninformed_trades_per_sec=0.0))
    assert sum(len(s.trades) for s in quiet) == 0
    mixed = snaps_of(run(informed_flow=1.0, uninformed_trades_per_sec=2.0, window_seconds=120))
    base = snaps_of(run(informed_flow=0.0, uninformed_trades_per_sec=2.0, window_seconds=120))
    assert sum(len(m.trades) for m in mixed) > sum(len(b.trades) for b in base)


# --------------------------------------------------------------------------- spot statistics


def test_per_tick_vol_and_zero_drift_match_the_annual_vol(long_snaps: list[MarketSnapshot]) -> None:
    snaps = long_snaps
    for asset, vol in (("BTC", 0.55), ("ETH", 0.75)):
        spots = [s.spot for s in snaps if s.market.asset == asset and s.spot is not None]
        assert len(spots) == 3000
        rets = [math.log(b / a) for a, b in zip(spots, spots[1:], strict=False)]
        target = vol * math.sqrt(1.0 / SECONDS_PER_YEAR)  # 0.55 * 1.78e-4 = 9.79e-5 per tick
        # n = 2999: relative se of a sample sd is 1/sqrt(2n) = 1.3%; bound is ~4 se
        assert statistics.pstdev(rets) == pytest.approx(target, rel=0.06)
        # exact lognormal step with zero drift: E[log return] = -sigma^2 / 2 ~ 0 (se = 1.8e-6)
        assert abs(statistics.fmean(rets)) < 6 * target / math.sqrt(len(rets))


def test_spot_step_is_a_zero_drift_gbm_with_the_exact_lognormal_step() -> None:
    # Drift is invisible at realistic vols (sigma^2/2 ~ 5e-9 per tick), so use a huge one:
    # vol = 2000 -> sigma = 2000 / sqrt(31_557_600) = 0.356 per 1 s tick, so the exact step
    # S' = S * exp(-sigma^2/2 + sigma z) has E[ln(S'/S)] = -sigma^2/2 = -0.0634 and E[S'/S] = 1.
    sigma = 2000.0 / math.sqrt(SECONDS_PER_YEAR)
    events = run(
        window_seconds=10,
        n_windows=400,
        assets=("BTC",),
        annual_vol={"BTC": 2000.0, "ETH": 0.75},
    )
    spots = [s.spot for s in snaps_of(events) if s.spot is not None]
    assert len(spots) == 4000 and all(x > 0.0 for x in spots)
    ratios = [b / a for a, b in zip(spots, spots[1:], strict=False)]
    logs = [math.log(r) for r in ratios]
    # n = 3999: se of the mean log-return = 0.356 / 63 = 0.0056; se of the mean ratio ~ 0.0056
    assert statistics.fmean(logs) == pytest.approx(-sigma * sigma / 2.0, abs=0.02)
    assert statistics.pstdev(logs) == pytest.approx(sigma, rel=0.05)
    assert statistics.fmean(ratios) == pytest.approx(1.0, abs=0.02)  # martingale, not log-drift


def test_vol_scales_with_tick_length() -> None:
    snaps = snaps_of(run(window_seconds=600, n_windows=5, tick_seconds=5.0, assets=("BTC",)))
    spots = [s.spot for s in snaps if s.spot is not None]
    rets = [math.log(b / a) for a, b in zip(spots, spots[1:], strict=False)]
    target = 0.55 * math.sqrt(5.0 / SECONDS_PER_YEAR)
    assert statistics.pstdev(rets) == pytest.approx(target, rel=0.1)  # n = 599, se 2.9%


def test_btc_eth_log_returns_are_correlated_about_point_eight(
    long_snaps: list[MarketSnapshot],
) -> None:
    snaps = long_snaps
    btc = [s.spot for s in snaps if s.market.asset == "BTC" and s.spot is not None]
    eth = [s.spot for s in snaps if s.market.asset == "ETH" and s.spot is not None]
    r_btc = [math.log(b / a) for a, b in zip(btc, btc[1:], strict=False)]
    r_eth = [math.log(b / a) for a, b in zip(eth, eth[1:], strict=False)]
    # n = 2999 -> se of the correlation = (1 - 0.64) / sqrt(n) = 0.0066
    assert statistics.correlation(r_btc, r_eth) == pytest.approx(0.8, abs=0.04)


def test_spot_stays_positive_and_continuous_across_windows() -> None:
    events = run(window_seconds=10, n_windows=30, annual_vol={"BTC": 30.0, "ETH": 40.0})
    for asset in ("BTC", "ETH"):
        spots = [s.spot for s in snaps_of(events) if s.market.asset == asset and s.spot is not None]
        assert len(spots) == 300
        assert all(x > 0.0 and math.isfinite(x) for x in spots)
    feed = SyntheticFeed(make_cfg(window_seconds=10, n_windows=30))
    per_market = by_market(list(feed))
    for spec in feed.markets:
        snaps = per_market[spec.market_id]
        assert all(s.ref_price == snaps[0].spot for s in snaps)


def test_up_win_fraction_over_many_windows_is_balanced() -> None:
    feed = SyntheticFeed(make_cfg(seed=3, window_seconds=10, n_windows=200))
    outcomes = [feed.winner(m.market_id) for m in feed.markets if m.asset == "BTC"]
    assert len(outcomes) == 200
    # Binomial(200, 1/2): sd of the fraction = 0.035, so [0.4, 0.6] is +-2.8 sd
    assert 0.4 <= outcomes.count(Outcome.UP) / 200 <= 0.6


def test_up_win_fraction_with_other_seeds() -> None:
    fractions = []
    for seed in (1, 2, 4, 5, 6):
        feed = SyntheticFeed(make_cfg(seed=seed, window_seconds=10, n_windows=200))
        ups = sum(feed.winner(m.market_id) is Outcome.UP for m in feed.markets if m.asset == "ETH")
        fractions.append(ups / 200)
        assert 0.4 <= fractions[-1] <= 0.6
    assert 0.45 <= statistics.fmean(fractions) <= 0.55  # 1000 windows: se 1.6%


# --------------------------------------------------------------------------- helper distributions


def test_geometric_matches_target_mean_and_tail() -> None:
    rng = random.Random(4)
    draws = [_geometric(rng, 0.5) for _ in range(20_000)]
    assert min(draws) == 0
    assert statistics.fmean(draws) == pytest.approx(0.5, abs=0.025)  # sd 0.87, se 0.006
    # q = 0.5 / 1.5 = 1/3: P(X = 0) = 2/3, P(X >= 2) = q^2 = 1/9
    assert draws.count(0) / len(draws) == pytest.approx(2 / 3, abs=0.012)
    assert sum(d >= 2 for d in draws) / len(draws) == pytest.approx(1 / 9, abs=0.01)
    rng = random.Random(4)
    assert {_geometric(rng, 0.0) for _ in range(100)} == {0}


def test_geometric_consumes_exactly_one_uniform_whatever_the_mean() -> None:
    for mean in (0.0, 0.5, 7.0):
        used, fresh = random.Random(8), random.Random(8)
        _geometric(used, mean)
        fresh.random()
        assert used.random() == fresh.random()


def test_poisson_matches_target_mean_and_variance() -> None:
    rng = random.Random(6)
    small = [_poisson(rng, 0.3) for _ in range(20_000)]
    assert statistics.fmean(small) == pytest.approx(0.3, abs=0.012)  # se 0.004
    assert small.count(0) / len(small) == pytest.approx(math.exp(-0.3), abs=0.01)  # 0.7408
    big = [_poisson(rng, 5.0) for _ in range(20_000)]
    assert statistics.fmean(big) == pytest.approx(5.0, abs=0.08)  # se 0.016
    assert statistics.pvariance(big) == pytest.approx(5.0, abs=0.25)
    assert _poisson(rng, 0.0) == 0
    assert _poisson(random.Random(1), 2_000.0) > 1_800  # no exp(-lambda) underflow


def test_swept_levels_hand_table() -> None:
    bids = [(50, 400.0), (49, 300.0), (48, 200.0), (47, 100.0)]
    asks = [(51, 10.0), (52, 20.0), (53, 30.0)]
    # bid falls 50 -> 48: old levels 50 and 49 (old best bid down to new best bid + 1)
    assert _swept_levels(bids, 48, Side.SELL) == [(50, 400.0), (49, 300.0)]
    assert _swept_levels(bids, 50, Side.SELL) == []  # unchanged
    assert _swept_levels(bids, 51, Side.SELL) == []  # rose
    assert _swept_levels(bids, 40, Side.SELL) == bids  # fell past the displayed depth
    # ask rises 51 -> 53: old levels 51 and 52 (old best ask up to new best ask - 1)
    assert _swept_levels(asks, 53, Side.BUY) == [(51, 10.0), (52, 20.0)]
    assert _swept_levels(asks, 51, Side.BUY) == []
    assert _swept_levels(asks, 50, Side.BUY) == []
    assert _swept_levels(asks, 99, Side.BUY) == asks


# --------------------------------------------------------------------------- randomised properties


def _random_sim_kwargs(rng: random.Random) -> dict[str, Any]:
    tick_seconds = rng.choice([0.5, 1.0, 2.0, 5.0])
    return {
        "seed": rng.randrange(10_000),
        "assets": rng.choice([("BTC",), ("BTC", "ETH"), ("ETH", "BTC")]),
        "window_seconds": rng.choice([10, 20, 30, 60]),
        "n_windows": rng.randint(1, 3),
        "tick_seconds": tick_seconds,
        "tick_size": rng.choice([0.005, 0.01, 0.02, 0.025, 0.05]),
        "annual_vol": {"BTC": rng.uniform(0.2, 3.0), "ETH": rng.uniform(0.2, 3.0)},
        "market_vol_multiplier": rng.choice([0.02, 0.2, 1.0, 2.0]),
        "market_lag_seconds": rng.choice([0.0, 1.0, 2.5, 6.0]),
        "pricing_noise_ticks": rng.choice([0.0, 0.7, 4.0]),
        "mean_spread_ticks": rng.choice([1.0, 1.5, 4.0]),
        "depth_mean_shares": rng.choice([5.0, 400.0, 2_000.0]),
        "n_levels": rng.randint(1, 8),
        "uninformed_trades_per_sec": rng.choice([0.0, 0.3, 3.0]),
        "informed_flow": rng.choice([0.0, 0.5, 1.0]),
    }


def test_randomised_configs_hold_all_invariants() -> None:
    rng = random.Random(2024)
    for _ in range(14):
        kw = _random_sim_kwargs(rng)
        cfg = make_cfg(**kw)
        feed = SyntheticFeed(cfg)
        events = list(feed)
        check_event_stream(events, feed, kw["window_seconds"], kw["tick_seconds"])
        for s in snaps_of(events):
            check_book_invariants(s, kw["tick_size"], kw["n_levels"])
            check_print_mirrors(s)
            assert s.spot is not None and s.spot > 0.0 and s.spot_ts == s.ts
            if kw["informed_flow"] == 0.0:
                for t in s.trades:
                    assert close(t.price, _touch_price(s, t))
        assert list(feed) == events  # determinism per configuration
        assert list(SyntheticFeed(cfg)) == events


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kw",
    [
        {"window_seconds": 0},
        {"window_seconds": -60},
        {"window_seconds": 10.5},
        {"window_seconds": 10, "tick_seconds": 3.0},  # not a multiple
        {"tick_seconds": 0.0},
        {"tick_seconds": -1.0},
        {"tick_seconds": math.nan},
        {"n_windows": 0},
        {"tick_size": 0.0},
        {"tick_size": 0.03},  # 1 / tick is not an integer: the mirror would be off-grid
        {"tick_size": 0.6},
        {"tick_size": 1e-5},
        {"min_order_size": 0.0},
        {"assets": ()},
        {"assets": ("BTC", "BTC")},
        {"assets": ("BTC", "SOL")},
        {"spot0": {"BTC": 0.0, "ETH": 3_000.0}},
        {"spot0": {"BTC": math.nan, "ETH": 3_000.0}},
        {"annual_vol": {"BTC": 0.0, "ETH": 0.75}},
        {"annual_vol": {"BTC": -0.5, "ETH": 0.75}},
        {"market_vol_multiplier": 0.0},
        {"market_lag_seconds": -1.0},
        {"pricing_noise_ticks": -0.1},
        {"mean_spread_ticks": 0.5},
        {"depth_mean_shares": 0.0},
        {"n_levels": 0},
        {"uninformed_trades_per_sec": -1.0},
        {"informed_flow": -0.1},
        {"informed_flow": 1.5},
    ],
)
def test_invalid_sim_config_is_rejected(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="sim config"):
        SyntheticFeed(make_cfg(**kw))


def test_absurd_volatility_fails_loudly_instead_of_producing_infinite_prices() -> None:
    with pytest.raises(ValueError, match="vol too high"):
        SyntheticFeed(make_cfg(annual_vol={"BTC": 1e9, "ETH": 0.75}))


def test_sim_package_exports_the_feed() -> None:
    from abc_trading.sim.feed import SyntheticFeed as Direct

    assert SyntheticFeed is Direct
