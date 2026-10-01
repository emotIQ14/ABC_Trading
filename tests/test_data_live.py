"""Tests for abc_trading.data.live (LiveFeed polling, resolution inference, record tee).

Offline and deterministic: a ``FakeClock`` replaces time.time/sleep and a ``World`` object plays
the role of every endpoint (it is the injected ``Transport``). The World is synthetic and
mirrors the shapes that public_api.py *assumes*; it says nothing about the real services.

Time layout used throughout: T0 = 1_700_000_100 is an exact multiple of 300, window = 300 s,
poll = 100 s, so ticks fall at T0, T0+100, T0+200, T0+300 (= next window start), ...
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from abc_trading.config import BotConfig, SimConfig
from abc_trading.data.events import event_to_dict, read_jsonl
from abc_trading.data.live import STAT_KEYS, LiveFeed, record
from abc_trading.data.public_api import DataError, PolymarketPublicClient, SpotClient
from abc_trading.types import (
    BookSnapshot,
    FeedEvent,
    Level,
    MarketResolved,
    MarketSnapshot,
    MarketSpec,
    Outcome,
    Side,
    Trade,
)

T0 = 1_700_000_100
W = 300
CFG = BotConfig(sim=dataclasses.replace(SimConfig(), min_order_size=7.0))

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com/book"
DATA_URL = "https://data-api.polymarket.com/trades"
BINANCE_URL = "https://api.binance.com/api/v3/ticker/price"
COINBASE_PREFIX = "https://api.coinbase.com/v2/prices/"

BTC_PRICES = {
    0: 60000.0, 100: 60010.0, 200: 59990.0, 300: 60050.0,
    400: 60020.0, 500: 60040.0, 600: 60000.0, 700: 60010.0,
}  # fmt: skip

Predicate = Callable[[float, str, dict[str, str]], bool]


class FakeClock:
    def __init__(self, t: float) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class ScriptedClock(FakeClock):
    """``sleep`` jumps to the next scripted time (can go backwards) instead of adding."""

    def __init__(self, times: list[float]) -> None:
        super().__init__(times[0])
        self.script = list(times[1:])

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t = self.script.pop(0)


class World:
    """Fake Polymarket + spot endpoints driven by the fake clock; injected as the Transport.

    Books: UP bid 0.48 / ask 0.50, DOWN bid 0.50 / ask 0.52, every size = 100 + (t - T0).
    """

    def __init__(self, clock: FakeClock, spot: Mapping[str, Callable[[float], float]]) -> None:
        self.clock = clock
        self.spot = dict(spot)
        self.trades: list[dict[str, Any]] = []
        self.failures: list[Predicate] = []
        self.calls: list[tuple[float, str, dict[str, str]]] = []

    def fail_when(self, predicate: Predicate) -> None:
        self.failures.append(predicate)

    def count(self, url: str) -> int:
        return sum(1 for _, u, _p in self.calls if u == url or u.startswith(url))

    def get_json(
        self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
    ) -> Any:
        p = dict(params or {})
        t = self.clock.t
        self.calls.append((t, url, p))
        if any(f(t, url, p) for f in self.failures):
            raise DataError(f"injected failure for {url}")
        if url == GAMMA_URL:
            slug = p["slug"]
            return [
                {
                    "slug": slug,
                    "conditionId": f"cond-{slug}",
                    "outcomes": '["Up", "Down"]',
                    "clobTokenIds": json.dumps([f"up-{slug}", f"down-{slug}"]),
                    "orderPriceMinTickSize": 0.005,
                }
            ]
        if url == CLOB_URL:
            return self._book(p["token_id"], t)
        if url == DATA_URL:
            return [r for r in self.trades if r["timestamp"] <= t]
        if url == BINANCE_URL:
            asset = p["symbol"].removesuffix("USDT")
            return {"price": repr(self.spot[asset](t))}
        if url.startswith(COINBASE_PREFIX):
            asset = url.removeprefix(COINBASE_PREFIX).split("-")[0]
            return {"data": {"amount": repr(self.spot[asset](t))}}
        raise AssertionError(f"unexpected URL {url}")

    @staticmethod
    def _book(token_id: str, t: float) -> dict[str, Any]:
        size = str(100.0 + (t - T0))
        if token_id.startswith("up-"):
            return {
                "bids": [{"price": "0.48", "size": size}],
                "asks": [{"price": "0.50", "size": size}],
            }
        return {
            "bids": [{"price": "0.50", "size": size}],
            "asks": [{"price": "0.52", "size": size}],
        }


def table_spot(table: Mapping[int, float]) -> Callable[[float], float]:
    return lambda t: table[int(t - T0)]


def is_spot(url: str) -> bool:
    return url == BINANCE_URL or url.startswith(COINBASE_PREFIX)


def is_spot_of(asset: str, url: str, params: Mapping[str, str]) -> bool:
    return params.get("symbol") == asset + "USDT" or url == f"{COINBASE_PREFIX}{asset}-USD/spot"


def make_world(
    prices: Callable[[float], float] | None = None, *, start: float = T0
) -> tuple[World, FakeClock]:
    clock = FakeClock(start)
    return World(clock, {"BTC": prices or table_spot(BTC_PRICES)}), clock


def make_feed(world: World, clock: FakeClock, **kw: Any) -> LiveFeed:
    opts: dict[str, Any] = {
        "assets": ("BTC",),
        "window_seconds": W,
        "poll_seconds": 100.0,
        "clock": clock.now,
        "sleep": clock.sleep,
    }
    opts.update(kw)
    return LiveFeed(CFG, PolymarketPublicClient(world), SpotClient(world), **opts)


def slug_for(start: int, asset: str = "btc") -> str:
    return f"{asset}-updown-5m-{start}"


def snap(ev: FeedEvent) -> MarketSnapshot:
    assert isinstance(ev, MarketSnapshot)
    return ev


def kinds(events: list[FeedEvent]) -> str:
    return "".join("R" if isinstance(e, MarketResolved) else "S" for e in events)


def zero_stats(**overrides: int) -> dict[str, int]:
    out = dict.fromkeys(STAT_KEYS, 0)
    out.update(overrides)
    return out


# --------------------------------------------------------------------------- rollover & fields


def test_window_rollover_resolution_and_snapshot_fields() -> None:
    world, clock = make_world()
    feed = make_feed(world, clock, max_events=9)
    events = list(feed)

    assert kinds(events) == "SSSRSSSRS"
    assert [e.ts for e in events] == [T0, T0 + 100, T0 + 200, T0 + 300, T0 + 300,
                                      T0 + 400, T0 + 500, T0 + 600, T0 + 600]  # fmt: skip

    s1 = slug_for(T0)  # window 1: [T0, T0+300)
    s2 = slug_for(T0 + 300)
    s3 = slug_for(T0 + 600)
    spec1 = MarketSpec(
        market_id=f"cond-{s1}", asset="BTC", start_ts=float(T0), end_ts=float(T0 + 300),
        tick_size=0.005,  # reported by the market lookup
        min_order_size=7.0,  # not reported: falls back to cfg.sim.min_order_size
        up_token_id=f"up-{s1}", down_token_id=f"down-{s1}", slug=s1,
    )  # fmt: skip
    assert events[0] == MarketSnapshot(
        ts=T0,
        market=spec1,
        up_book=BookSnapshot(Outcome.UP, (Level(0.48, 100.0),), (Level(0.5, 100.0),)),
        down_book=BookSnapshot(Outcome.DOWN, (Level(0.5, 100.0),), (Level(0.52, 100.0),)),
        spot=60000.0,
        spot_ts=T0,
        ref_price=60000.0,  # first reading of the window
        trades=(),
    )
    # tick 2: spot 60010, ref still 60000, book sizes = 100 + 100
    s_1 = snap(events[1])
    assert (s_1.spot, s_1.spot_ts, s_1.ref_price) == (60010.0, T0 + 100, 60000.0)
    assert s_1.up_book.bids == (Level(0.48, 200.0),) and s_1.market == spec1

    # window 1 ends: spot 60050 >= ref 60000 -> UP; resolution carries the window end time
    assert events[3] == MarketResolved(ts=T0 + 300, market_id=f"cond-{s1}", winner=Outcome.UP)
    # the same reading (60050) is the new window's price to beat
    s_4 = snap(events[4])
    assert s_4.market.slug == s2 and s_4.market.start_ts == T0 + 300
    assert s_4.market.end_ts == T0 + 600 and s_4.market.market_id == f"cond-{s2}"
    assert (s_4.spot, s_4.ref_price) == (60050.0, 60050.0)
    assert snap(events[5]).ref_price == 60050.0

    # window 2 ends at spot 60000 < ref 60050 -> DOWN
    assert events[7] == MarketResolved(ts=T0 + 600, market_id=f"cond-{s2}", winner=Outcome.DOWN)
    s_8 = snap(events[8])
    assert s_8.market.slug == s3 and s_8.ref_price == 60000.0

    assert feed.stats == zero_stats(ticks=7, snapshots=7, resolutions=2, windows_opened=3)
    assert clock.sleeps == [100.0] * 6  # no sleep after the last event
    assert world.count(GAMMA_URL) == 3  # market looked up once per window
    assert world.count(BINANCE_URL) == 7
    assert world.count(CLOB_URL) == 14 and world.count(DATA_URL) == 7
    assert feed.last_error is None


def test_tie_resolves_up() -> None:
    world, clock = make_world(table_spot({0: 60000.0, 100: 60000.0, 200: 60000.0, 300: 60000.0}))
    events = list(make_feed(world, clock, max_events=4))
    assert events[3] == MarketResolved(T0 + 300, f"cond-{slug_for(T0)}", Outcome.UP)


def test_long_run_invariants_and_determinism() -> None:
    def prices(t: float) -> float:
        return 60000.0 + 10.0 * (int(t - T0) // 100 % 7 - 3)

    runs = []
    for _ in range(2):
        world, clock = make_world(prices)
        events = list(make_feed(world, clock, max_events=40))
        runs.append((events, world.count(GAMMA_URL)))
    events, gamma_calls = runs[0]
    assert runs[1][0] == events  # same inputs, same events

    # 31 ticks: 30 snapshots, and the 10th window's resolution is the 40th (last) event
    assert len(events) == 40
    assert Counter(type(e) for e in events) == {MarketSnapshot: 30, MarketResolved: 10}
    assert isinstance(events[-1], MarketResolved)
    ts = [e.ts for e in events]
    assert ts == sorted(ts)

    resolved = [e for e in events if isinstance(e, MarketResolved)]
    assert len({r.market_id for r in resolved}) == 10  # at most once per market
    snaps = [snap(e) for e in events if isinstance(e, MarketSnapshot)]
    for s in snaps:
        assert s.market.start_ts <= s.ts < s.market.end_ts
    for i, r in enumerate(resolved):
        mine = [s for s in snaps if s.market.market_id == r.market_id]
        assert len(mine) == 3
        assert r.ts == mine[0].market.end_ts == T0 + 300 * (i + 1)
        assert max(s.ts for s in mine) < r.ts
    # the 31st tick resolves window 10 and already opens window 11 (its snapshot is built but
    # the feed stops before yielding it), so there are 11 market lookups
    assert gamma_calls == 11


def test_trades_are_attached_once_and_mirrored() -> None:
    world, clock = make_world()
    s1 = slug_for(T0)
    world.trades = [
        {"side": "BUY", "asset": f"up-{s1}", "conditionId": f"cond-{s1}", "size": 10,
         "price": 0.45, "timestamp": T0 + 50, "transactionHash": "h1"},
        {"side": "SELL", "asset": f"down-{s1}", "conditionId": f"cond-{s1}", "size": 25.5,
         "price": 0.3, "timestamp": T0 + 150, "transactionHash": "h2"},
    ]  # fmt: skip
    events = list(make_feed(world, clock, max_events=3))
    assert [snap(e).trades for e in events] == [
        (),  # nothing printed yet at T0
        (  # since T0: r1 (a BUY of UP at 0.45) and its mirror, a SELL of DOWN at 1 - 0.45
            Trade(T0 + 50, Outcome.UP, 0.45, 10.0, Side.BUY),
            Trade(T0 + 50, Outcome.DOWN, 0.55, 10.0, Side.SELL),
        ),
        (  # r1 is not repeated; r2 (SELL DOWN 0.30) and its mirror (BUY UP at 0.70)
            Trade(T0 + 150, Outcome.DOWN, 0.3, 25.5, Side.SELL),
            Trade(T0 + 150, Outcome.UP, 0.7, 25.5, Side.BUY),
        ),
    ]


# --------------------------------------------------------------------------- errors


def test_book_error_skips_the_tick_and_is_counted() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: url == CLOB_URL and t == T0 + 100)
    feed = make_feed(world, clock, max_events=2)
    events = list(feed)
    assert [e.ts for e in events] == [T0, T0 + 200]  # the T0+100 tick produced nothing
    assert feed.stats == zero_stats(ticks=3, snapshots=2, windows_opened=1, errors=1, errors_book=1)
    assert feed.last_error is not None and feed.last_error.startswith("book: injected failure")
    assert world.count(CLOB_URL) == 5  # 2 + 1 (failed on the first book) + 2


def test_market_lookup_error_retries_and_keeps_the_reference_price() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: url == GAMMA_URL and t == T0)
    feed = make_feed(world, clock, max_events=1)
    (first,) = [snap(e) for e in feed]
    assert first.ts == T0 + 100
    # the price to beat was read at the window start (60000), not at the successful lookup
    assert (first.spot, first.ref_price) == (60010.0, 60000.0)
    assert first.market.slug == slug_for(T0)
    assert feed.stats == zero_stats(
        ticks=2, snapshots=1, windows_opened=1, errors=1, errors_market=1
    )


def test_spot_error_skips_the_tick_and_is_counted_once() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: is_spot(url) and t == T0 + 100)  # Binance AND Coinbase
    feed = make_feed(world, clock, max_events=2)
    events = list(feed)
    assert [e.ts for e in events] == [T0, T0 + 200]
    assert feed.stats == zero_stats(ticks=3, snapshots=2, windows_opened=1, errors=1, errors_spot=1)
    assert feed.last_error is not None and feed.last_error.startswith("spot: spot unavailable")
    assert clock.sleeps == [100.0, 100.0]


def test_spot_falls_back_to_coinbase_without_counting_an_error() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: url == BINANCE_URL)
    feed = make_feed(world, clock, max_events=2)
    events = [snap(e) for e in feed]
    assert [e.spot for e in events] == [60000.0, 60010.0]
    assert feed.stats["errors"] == 0


def test_trades_error_does_not_skip_the_tick_and_nothing_is_lost() -> None:
    world, clock = make_world()
    s1 = slug_for(T0)
    world.trades = [
        {"side": "BUY", "asset": f"up-{s1}", "conditionId": f"cond-{s1}", "size": 10,
         "price": 0.45, "timestamp": T0 + 50, "transactionHash": "h1"},
    ]  # fmt: skip
    world.fail_when(lambda t, url, p: url == DATA_URL and t == T0 + 100)
    feed = make_feed(world, clock, max_events=3)
    events = [snap(e) for e in feed]
    assert [e.ts for e in events] == [T0, T0 + 100, T0 + 200]  # snapshot still emitted
    assert events[1].trades == ()
    # the missed print is delivered with the next successful trades call
    assert events[2].trades == (
        Trade(T0 + 50, Outcome.UP, 0.45, 10.0, Side.BUY),
        Trade(T0 + 50, Outcome.DOWN, 0.55, 10.0, Side.SELL),
    )
    assert feed.stats["errors_trades"] == 1 and feed.stats["errors"] == 1


def test_non_data_errors_propagate() -> None:
    world, clock = make_world()

    class Boom(World):
        def get_json(
            self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
        ) -> Any:
            if url == CLOB_URL:
                raise RuntimeError("bug")
            return super().get_json(url, params, timeout)

    boom = Boom(clock, world.spot)
    feed = LiveFeed(
        CFG, PolymarketPublicClient(boom), SpotClient(boom), assets=("BTC",), window_seconds=W,
        poll_seconds=100.0, clock=clock.now, sleep=clock.sleep,
    )  # fmt: skip
    with pytest.raises(RuntimeError, match="bug"):
        next(feed)
    assert feed.stats["errors"] == 0


def test_spot_outage_at_rollover_delays_resolution() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: is_spot(url) and t == T0 + 300)
    feed = make_feed(world, clock, max_events=5)
    events = list(feed)
    assert kinds(events) == "SSSRS"
    # the T0+300 tick yields nothing; at T0+400 the window is resolved from that reading
    # (60020 >= 60000 -> UP) but stamped with the window end, which is not before the last event
    assert events[3] == MarketResolved(T0 + 300, f"cond-{slug_for(T0)}", Outcome.UP)
    last = snap(events[4])
    assert last.ts == T0 + 400
    # the new window started at T0+300; its first reading came 100 s late but within the
    # default tolerance max(5, 2 * poll) = 200 s, so it is accepted
    assert last.market.start_ts == T0 + 300 and last.ref_price == 60020.0
    assert feed.stats == zero_stats(
        ticks=5, snapshots=4, resolutions=1, windows_opened=2, errors=1, errors_spot=1
    )


def test_late_first_reading_skips_the_window() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: is_spot(url) and t == T0 + 300)
    feed = make_feed(world, clock, max_events=5, max_ref_lag_seconds=50.0)
    events = list(feed)
    assert kinds(events) == "SSSRS"
    # the reading at T0+400 is 100 s after the window start > 50 s: that window is skipped (no
    # events, no spot reads at T0+500); the next window starts on time at T0+600
    last = snap(events[4])
    assert last.ts == T0 + 600 and last.market.slug == slug_for(T0 + 600)
    assert last.ref_price == 60000.0
    assert feed.stats == zero_stats(
        ticks=7, snapshots=4, resolutions=1, windows_opened=2, windows_skipped_late=1,
        errors=1, errors_spot=1,
    )  # fmt: skip
    # spot reads: T0, +100, +200, +300 (fails), +400, [+500 skipped], +600
    assert world.count(BINANCE_URL) == 6


def test_feed_started_mid_window_waits_for_the_next_window() -> None:
    world, clock = make_world(lambda t: 60000.0 + (t - T0), start=T0 + 30)
    feed = make_feed(world, clock, poll_seconds=10.0, max_events=1)
    (first,) = [snap(e) for e in feed]
    # default tolerance = max(5, 2 * 10) = 20 s < 30 s late: the partial window is skipped
    assert first.ts == T0 + 300
    assert first.market.slug == slug_for(T0 + 300)
    assert first.ref_price == 60300.0 and first.spot == 60300.0
    # ticks T0+30 ... T0+300 step 10 = 28; spot read at the first tick and at the boundary only
    assert feed.stats == zero_stats(ticks=28, snapshots=1, windows_opened=1, windows_skipped_late=1)
    assert world.count(BINANCE_URL) == 2
    assert world.count(GAMMA_URL) == 1  # no market lookup for the skipped window


def test_skipped_window_never_emits_a_resolution() -> None:
    world, clock = make_world(lambda t: 60000.0 + (t - T0), start=T0 + 30)
    events = list(make_feed(world, clock, poll_seconds=10.0, max_events=3))
    assert kinds(events) == "SSS"  # no MarketResolved for the window that was skipped


# --------------------------------------------------------------------------- iteration control


def test_max_events_stops_iteration() -> None:
    world, clock = make_world()
    feed = make_feed(world, clock, max_events=2)
    assert iter(feed) is feed
    assert next(feed).ts == T0 and next(feed).ts == T0 + 100
    with pytest.raises(StopIteration):
        next(feed)
    assert list(feed) == []  # stays exhausted
    assert feed.stats["ticks"] == 2 and clock.sleeps == [100.0]
    assert world.count(BINANCE_URL) == 2  # no polling after the limit


def test_max_events_zero_does_nothing() -> None:
    world, clock = make_world()
    feed = make_feed(world, clock, max_events=0)
    assert list(feed) == []
    assert world.calls == [] and clock.sleeps == [] and feed.stats == zero_stats()


def test_max_events_can_cut_a_tick_in_the_middle() -> None:
    clock = FakeClock(T0)
    eth = table_spot({0: 3000.0, 100: 3000.0, 200: 3000.0, 300: 2990.0})
    world = World(clock, {"BTC": table_spot(BTC_PRICES), "ETH": eth})
    feed = make_feed(world, clock, assets=("BTC", "ETH"), max_events=7)
    events = list(feed)
    assert kinds(events) == "SSSSSSR"  # stops after the first resolution of the rollover tick


def test_two_assets_resolutions_before_snapshots_in_a_tick() -> None:
    clock = FakeClock(T0)
    eth = table_spot({0: 3000.0, 100: 3000.0, 200: 3000.0, 300: 2990.0})
    world = World(clock, {"BTC": table_spot(BTC_PRICES), "ETH": eth})
    feed = make_feed(world, clock, assets=("BTC", "ETH"), max_events=10)
    events = list(feed)
    assert kinds(events) == "SSSSSSRRSS"
    assert [e.ts for e in events] == [T0] * 2 + [T0 + 100] * 2 + [T0 + 200] * 2 + [T0 + 300] * 4
    assert [snap(e).market.asset for e in events[:2]] == ["BTC", "ETH"]
    assert events[2:4] != events[:2]
    assert events[6] == MarketResolved(T0 + 300, f"cond-{slug_for(T0, 'btc')}", Outcome.UP)
    assert events[7] == MarketResolved(T0 + 300, f"cond-{slug_for(T0, 'eth')}", Outcome.DOWN)
    assert snap(events[8]).market.asset == "BTC" and snap(events[9]).market.asset == "ETH"
    assert snap(events[9]).ref_price == 2990.0


def test_delayed_resolution_of_one_asset_keeps_the_tick_sorted_by_time() -> None:
    clock = FakeClock(T0)
    eth = table_spot({0: 3000.0, 100: 3000.0, 200: 3000.0, 600: 3010.0})
    world = World(clock, {"BTC": table_spot(BTC_PRICES), "ETH": eth})
    # ETH has no spot from T0+300 to T0+500: its first window stays open, then resolves late
    world.fail_when(lambda t, url, p: is_spot_of("ETH", url, p) and T0 + 300 <= t <= T0 + 500)
    feed = make_feed(world, clock, assets=("BTC", "ETH"), max_events=14)
    events = list(feed)
    assert [e.ts for e in events] == sorted(e.ts for e in events)
    tail = events[10:]
    assert kinds(tail) == "RRSS"
    # ETH window 1 (ref 3000, end reading 3010 -> UP) was due at T0+300 but could only be
    # resolved at T0+600; it is stamped max(window end, last event) = T0+500 and listed before
    # the BTC window-2 resolution (ref 60050, end 60000 -> DOWN) stamped T0+600.
    assert tail[0] == MarketResolved(T0 + 500, f"cond-{slug_for(T0, 'eth')}", Outcome.UP)
    assert tail[1] == MarketResolved(T0 + 600, f"cond-{slug_for(T0 + 300, 'btc')}", Outcome.DOWN)
    assert snap(tail[2]).market.asset == "BTC" and snap(tail[3]).market.asset == "ETH"
    assert snap(tail[3]).ref_price == 3010.0
    assert feed.stats["errors_spot"] == 3  # ETH at T0+300, +400, +500


def test_window_without_any_snapshot_is_not_resolved() -> None:
    world, clock = make_world()
    world.fail_when(lambda t, url, p: url == CLOB_URL and t < T0 + 300)  # no books all window
    feed = make_feed(world, clock, max_events=1)
    (first,) = [snap(e) for e in feed]
    # the market was looked up but never snapshotted, so no MarketResolved(window 1) is emitted
    assert first.ts == T0 + 300 and first.market.slug == slug_for(T0 + 300)
    assert feed.stats == zero_stats(ticks=4, snapshots=1, windows_opened=2, errors=3, errors_book=3)


def test_backwards_clock_is_clamped_to_keep_timestamps_monotone() -> None:
    clock = ScriptedClock([T0, T0 + 100, T0 + 50, T0 + 200])
    world = World(clock, {"BTC": lambda t: 60000.0})
    feed = make_feed(world, clock, max_events=4)
    assert [e.ts for e in feed] == [T0, T0 + 100, T0 + 100, T0 + 200]


def test_non_finite_clock_is_an_error() -> None:
    world, clock = make_world(start=math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        next(make_feed(world, clock))


@pytest.mark.parametrize(
    ("poll", "sleeps", "second_ts"),
    [
        (2.0, [0.75], T0 + 2.0),  # tick 1 took 5 * 0.25 = 1.25 s: sleep the remaining 0.75 s
        (1.0, [], T0 + 1.25),  # tick took longer than the poll interval: no sleep at all
    ],
)
def test_sleep_accounts_for_time_spent_polling(
    poll: float, sleeps: list[float], second_ts: float
) -> None:
    class SlowWorld(World):
        def get_json(
            self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
        ) -> Any:
            self.clock.t += 0.25  # every request takes 0.25 s
            return super().get_json(url, params, timeout)

    clock = FakeClock(T0)
    world = SlowWorld(clock, {"BTC": lambda t: 60000.0})
    feed = make_feed(world, clock, poll_seconds=poll, max_events=2)
    events = list(feed)
    # tick 1 = spot + lookup + 2 books + trades = 5 requests
    assert clock.sleeps == sleeps
    assert [e.ts for e in events] == [T0, second_ts]


# --------------------------------------------------------------------------- templates & validation


def test_custom_slug_template_placeholders() -> None:
    world, clock = make_world()
    feed = make_feed(
        world, clock, max_events=1,
        slug_template="{asset_upper}_{asset}_{minutes}_{window_seconds}_{start_ts}_{end_ts}",
    )  # fmt: skip
    (first,) = [snap(e) for e in feed]
    assert first.market.slug == f"BTC_BTC_5_300_{T0}_{T0 + 300}"
    assert [p["slug"] for _, u, p in world.calls if u == GAMMA_URL] == [first.market.slug]


@pytest.mark.parametrize("template", ["{nope}", "{0}", "{asset", "{asset!x}"])
def test_bad_slug_templates_fail_at_construction(template: str) -> None:
    world, clock = make_world()
    with pytest.raises(ValueError, match="slug_template"):
        make_feed(world, clock, slug_template=template)


def test_minutes_placeholder_needs_whole_minutes() -> None:
    world, clock = make_world()
    with pytest.raises(ValueError, match="multiple of 60"):
        make_feed(world, clock, window_seconds=90)
    make_feed(world, clock, window_seconds=90, slug_template="{asset_lower}-{start_ts}")  # fine


@pytest.mark.parametrize(
    "kw",
    [
        {"assets": ()},
        {"assets": ("BTC", "BTC")},
        {"assets": ("",)},
        {"window_seconds": 0},
        {"window_seconds": -300},
        {"window_seconds": 300.0},
        {"window_seconds": True},
        {"poll_seconds": 0.0},
        {"poll_seconds": -1.0},
        {"poll_seconds": math.nan},
        {"poll_seconds": math.inf},
        {"max_events": -1},
        {"max_ref_lag_seconds": -1.0},
        {"max_ref_lag_seconds": math.nan},
    ],
)
def test_invalid_constructor_arguments(kw: dict[str, Any]) -> None:
    world, clock = make_world()
    with pytest.raises(ValueError):
        make_feed(world, clock, **kw)


def test_constructor_makes_no_requests() -> None:
    world, clock = make_world()
    make_feed(world, clock)
    assert world.calls == [] and clock.sleeps == []


# --------------------------------------------------------------------------- record


def test_record_tees_events_into_jsonl_and_passes_them_through(tmp_path: Path) -> None:
    world, clock = make_world()
    feed = make_feed(world, clock, max_events=9)
    path = tmp_path / "runs" / "paper.jsonl"
    seen = list(record(feed, path))
    assert len(seen) == 9 and kinds(seen) == "SSSRSSSRS"
    assert list(read_jsonl(path)) == seen  # lossless
    assert len(path.read_text().splitlines()) == 9


def test_record_writes_before_yielding_and_closes_on_early_exit(tmp_path: Path) -> None:
    world, clock = make_world()
    path = tmp_path / "early.jsonl"
    stream = record(make_feed(world, clock), path)
    assert not path.exists()  # lazy: nothing happens until the first next()
    first = next(stream)
    # the event is already on disk although the stream is still open
    assert json.loads(path.read_text().splitlines()[0]) == event_to_dict(first)
    second = next(stream)
    assert len(path.read_text().splitlines()) == 2
    stream.close()
    assert list(read_jsonl(path)) == [first, second]
    assert world.count(BINANCE_URL) == 2  # the feed was not polled beyond what was consumed


def test_record_truncates_an_existing_file_and_handles_empty_streams(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("stale\n")
    assert list(record([], path)) == []
    assert path.read_text() == ""
    ev = MarketResolved(5.0, "m", Outcome.UP)
    assert list(record(iter([ev, ev]), path)) == [ev, ev]
    assert list(read_jsonl(path)) == [ev, ev]


def test_record_rejects_non_finite_events_without_yielding_them(tmp_path: Path) -> None:
    bad = MarketResolved(math.nan, "m", Outcome.UP)
    stream = record([bad], tmp_path / "bad.jsonl")
    with pytest.raises(ValueError):
        next(stream)
