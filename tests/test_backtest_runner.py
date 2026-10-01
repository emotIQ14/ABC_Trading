"""Tests for abc_trading.backtest.runner.

The runner is exercised with a SCRIPTED engine (it returns canned actions, so every number below
is independent of the strategy's tuning) and the real PaperExchange. Expected values are worked
out by hand in the comments.

Market "m1": BTC, window 1000..1900, tick 0.01, min order size 5. All books have mid 0.50: UP bids
0.49/0.48/0.47/0.46 (size 400) and 0.45 (size 300), UP asks 0.51/0.52/0.53 (size 400); the DOWN
book is the exact mirror.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from abc_trading.backtest import runner as runner_mod
from abc_trading.backtest.runner import (
    BacktestResult,
    BacktestRunner,
    CalibSample,
    InvariantError,
    MarketRecord,
    activity_stats,
    calibration_stats,
    fills_digest,
    mark_to_mid,
    pair_stats,
    pnl_stats,
    run_backtest,
)
from abc_trading.config import BotConfig
from abc_trading.exchange.paper import PaperExchange
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.types import (
    Action,
    BookSnapshot,
    CancelOrder,
    FeedEvent,
    Fill,
    Level,
    MarketResolved,
    MarketSnapshot,
    MarketSpec,
    MergePairs,
    MergeResult,
    OpenOrder,
    OrderRequest,
    Outcome,
    Phase,
    PlaceOrder,
    Settlement,
    Side,
    TimeInForce,
    Trade,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
BUY, SELL = Side.BUY, Side.SELL
POST, IOC = TimeInForce.POST_ONLY, TimeInForce.IOC
M1 = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)
M2 = MarketSpec("m2", "ETH", start_ts=1000.0, end_ts=1900.0)

UP_BIDS = ((0.49, 400.0), (0.48, 400.0), (0.47, 400.0), (0.46, 400.0), (0.45, 300.0))
UP_ASKS = ((0.51, 400.0), (0.52, 400.0), (0.53, 400.0))


# --------------------------------------------------------------------------- builders


def _levels(pairs: Sequence[tuple[float, float]]) -> tuple[Level, ...]:
    return tuple(Level(p, s) for p, s in pairs)


def mirrored_books() -> tuple[BookSnapshot, BookSnapshot]:
    up = BookSnapshot(UP, _levels(UP_BIDS), _levels(UP_ASKS))
    down_bids = [(round(1.0 - p, 6), s) for p, s in UP_ASKS]
    down_asks = [(round(1.0 - p, 6), s) for p, s in UP_BIDS]
    return up, BookSnapshot(DOWN, _levels(down_bids), _levels(down_asks))


def snap(ts: float, market: MarketSpec = M1, trades: Sequence[Trade] = ()) -> MarketSnapshot:
    up, down = mirrored_books()
    return MarketSnapshot(
        ts=ts, market=market, up_book=up, down_book=down,
        spot=60_000.0, spot_ts=ts, ref_price=60_000.0, trades=tuple(trades),
    )  # fmt: skip


def place(
    cid: str, token: Outcome, side: Side, price: float, size: float, tif: TimeInForce,
    market_id: str = "m1",
) -> PlaceOrder:  # fmt: skip
    return PlaceOrder(OrderRequest(cid, market_id, token, side, price, size, tif))


def print_(ts: float, token: Outcome, price: float, size: float) -> Trade:
    return Trade(ts, token, price, size, SELL)


class ScriptedEngine(MarketMakerEngine):
    """Real portfolio/fee bookkeeping, but ``on_snapshot`` returns the canned actions for that
    snapshot ts (keyed by (market_id, ts)) and logs every call, so the strategy plays no part."""

    def __init__(self, cfg: BotConfig, script: dict[tuple[str, float], list[Action]]) -> None:
        super().__init__(cfg)
        self.script = script
        self.log: list[tuple[str, object]] = []

    def on_snapshot(self, snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]:
        self.log.append(("snapshot", snap.ts))
        return list(self.script.get((snap.market.market_id, snap.ts), []))

    def on_fill(self, fill: Fill) -> None:
        self.log.append(("fill", fill.fill_id))
        super().on_fill(fill)

    def on_merge(self, result: MergeResult) -> None:
        self.log.append(("merge", result.size))
        super().on_merge(result)

    def on_settlement(self, s: Settlement):  # type: ignore[no-untyped-def]
        self.log.append(("settle", s.winner))
        return super().on_settlement(s)


class SpyExchange(PaperExchange):
    """Logs every call the runner makes, in order."""

    def __init__(self, cfg: BotConfig) -> None:
        super().__init__(cfg)
        self.calls: list[tuple[str, object]] = []

    def submit(self, req, ts):  # type: ignore[no-untyped-def]
        self.calls.append(("submit", req.client_id))
        return super().submit(req, ts)

    def cancel(self, order_id, ts):  # type: ignore[no-untyped-def]
        self.calls.append(("cancel", order_id))
        return super().cancel(order_id, ts)

    def merge(self, market_id, size, ts):  # type: ignore[no-untyped-def]
        self.calls.append(("merge", size))
        return super().merge(market_id, size, ts)

    def process(self, snap):  # type: ignore[no-untyped-def]
        self.calls.append(("process", snap.ts))
        return super().process(snap)


def make_runner(
    script: dict[tuple[str, float], list[Action]],
    cfg: BotConfig | None = None,
    **kwargs: Any,
) -> BacktestRunner:
    cfg = cfg or BotConfig()
    runner = BacktestRunner(cfg, **kwargs)
    runner.engine = ScriptedEngine(cfg, script)
    runner.exchange = SpyExchange(cfg)
    return runner


# Scenario S1 (latency 1): two bids fill, merge, one more fill held to resolution.
#   1100: place BUY UP 100@0.49 and BUY DOWN 100@0.49 (post-only; live at the 1101 snapshot)
#   1102: prints of 500 at 0.49 on both tokens: each drains the queue ahead (400, the displayed
#         size at 0.49 when the order went live) and fills the remaining 500 - 400 = 100
#         -> both orders fill 100@0.49 (maker, fee 0): cash 10000 - 49 - 49 = 9902; the engine
#         then merges the 100 pairs: cash 9902 + 100 = 10002, merge pnl = 100 - (49 + 49) = +2
#   1103: place BUY UP 20@0.45 (live at 1104, queue ahead 300), 1105: a print of 5000 at 0.45
#         drains 300 and fills 20@0.45: cash 10002 - 9 = 9993
#   1106: place BUY DOWN 10@0.30 that is still resting when the market resolves
S1_SCRIPT: dict[tuple[str, float], list[Action]] = {
    ("m1", 1100.0): [
        place("c1", UP, BUY, 0.49, 100.0, POST),
        place("c2", DOWN, BUY, 0.49, 100.0, POST),
    ],
    ("m1", 1102.0): [MergePairs("m1", 100.0)],
    ("m1", 1103.0): [place("c3", UP, BUY, 0.45, 20.0, POST)],
    ("m1", 1106.0): [place("c4", DOWN, BUY, 0.30, 10.0, POST)],
}


def s1_events(winner: Outcome = UP, *, resolve: bool = True) -> list[FeedEvent]:
    events: list[FeedEvent] = [
        snap(1100.0),
        snap(1101.0),
        snap(1102.0, trades=[print_(1102.0, UP, 0.49, 500.0), print_(1102.0, DOWN, 0.49, 500.0)]),
        snap(1103.0),
        snap(1104.0),
        snap(1105.0, trades=[print_(1105.0, UP, 0.45, 5000.0)]),
        snap(1106.0),
        snap(1107.0),
    ]
    if resolve:
        events.append(MarketResolved(1900.0, "m1", winner))
    return events


def run_s1(winner: Outcome = UP, **kwargs: Any) -> tuple[BacktestResult, ScriptedEngine]:
    runner = make_runner(S1_SCRIPT, **kwargs)
    for ev in s1_events(winner):
        runner.feed(ev)
    assert isinstance(runner.engine, ScriptedEngine)
    return runner.finish(), runner.engine


# --------------------------------------------------------------------------- scenario S1


def test_s1_money_and_pnl_by_hand() -> None:
    result, _ = run_s1(UP, equity_sample_every=1, source_label="synthetic:test")
    # cash 10000 -> 9902 (two fills) -> 10002 (merge) -> 9993 (third fill) -> 10013 (20 UP pay $20)
    assert result.initial_cash == 10_000.0
    assert result.final_equity == pytest.approx(10_013.0)
    assert result.total_pnl == pytest.approx(13.0)
    assert result.pnl_pct_of_initial_cash == pytest.approx(0.13)  # 13 / 10000
    assert (result.n_events, result.n_snapshots, result.n_resolutions) == (9, 8, 1)
    assert (result.first_ts, result.last_ts) == (1100.0, 1900.0)
    assert result.unresolved_markets == []
    assert result.kill_switch is False

    (m,) = result.markets
    assert (m.market_id, m.asset, m.resolved, m.winner) == ("m1", "BTC", True, "UP")
    assert m.merge_pnl == pytest.approx(2.0)  # 100 - (49 + 49)
    assert m.sell_pnl == 0.0
    assert m.settle_pair_pnl == 0.0  # no DOWN left, so no pair part
    assert m.settle_directional_pnl == pytest.approx(11.0)  # payout 20 - cost 9
    assert m.pnl == pytest.approx(13.0)
    assert m.fees_paid == 0.0


def test_s1_market_record_by_hand() -> None:
    result, _ = run_s1(UP)
    (m,) = result.markets
    assert m.n_fills == 3
    assert m.buy_shares == pytest.approx(220.0)  # 100 + 100 + 20
    assert m.sell_shares == 0.0
    assert m.volume_maker == pytest.approx(107.0)  # 49 + 49 + 9
    assert m.volume_taker == 0.0
    assert m.max_inventory_shares == pytest.approx(200.0)  # 100 UP + 100 DOWN before the merge
    assert m.max_net_shares == pytest.approx(100.0)  # after the first of the two fills
    assert m.max_capital_at_risk == pytest.approx(98.0)  # 49 + 49 all-in cost basis
    assert m.merged_pairs == pytest.approx(100.0)
    assert m.paired_at_close == 0.0
    assert m.avg_pair_cost_at_close is None  # one side empty at close
    assert m.avg_merge_pair_cost == pytest.approx(0.98)  # 1 - 2 / 100
    assert (m.unpaired_token, m.unpaired_at_close, m.ended_unhedged) == ("UP", 20.0, True)
    assert m.pair_completion_rate == pytest.approx(200.0 / 220.0)  # 2 * 100 pairs / 220 bought
    assert m.p_up_last_accumulate is None  # the scripted engine never reports a fair value


def test_s1_aggregates_by_hand() -> None:
    result, _ = run_s1(UP)
    a, p, s = result.activity, result.pairs, result.pnl
    assert (a.n_fills, a.n_buy_fills, a.n_sell_fills) == (3, 3, 0)
    assert (a.n_maker_fills, a.n_taker_fills) == (3, 0)
    assert a.volume_maker == pytest.approx(107.0)
    assert a.avg_fill_notional == pytest.approx(107.0 / 3.0)  # (49 + 49 + 9) / 3
    assert a.median_fill_notional == pytest.approx(49.0)  # sorted 9, 49, 49
    assert p.shares_bought == pytest.approx(220.0)
    assert (p.merged_pairs, p.paired_at_close, p.pairs_formed) == (100.0, 0.0, 100.0)
    assert p.pair_completion_rate == pytest.approx(200.0 / 220.0)
    assert p.avg_merge_pair_cost == pytest.approx(0.98)
    assert p.avg_close_pair_cost is None
    assert (p.n_markets, p.n_markets_traded, p.n_markets_unhedged) == (1, 1, 1)
    assert (p.frac_markets_unhedged, p.frac_traded_markets_unhedged) == (1.0, 1.0)
    assert s.n_markets_resolved == 1
    assert s.mean_market_pnl == pytest.approx(13.0)
    assert s.std_market_pnl == 0.0  # one market: undefined, reported as 0
    assert s.sharpe_like == 0.0
    assert (s.best_market_pnl, s.worst_market_pnl, s.win_rate_traded) == (
        pytest.approx(13.0),
        pytest.approx(13.0),
        1.0,
    )
    assert s.pnl_merge == pytest.approx(2.0)
    assert s.pnl_settle_directional == pytest.approx(11.0)
    # no accumulate snapshot was ever reported by the scripted engine
    assert result.calibration.n_resolved == 1
    assert (result.calibration.n_no_accumulate, result.calibration.n_scored) == (1, 0)
    assert result.calibration.brier_model is None


def test_s1_equity_curve_marks_at_snapshot_mids() -> None:
    result, _ = run_s1(UP, equity_sample_every=1)
    curve = result.equity_curve
    assert [p.event_index for p in curve] == list(range(1, 10))
    # mids are 0.50 on both tokens in every snapshot
    #   events 1-2: 10000; event 3 (1102): cash 9902 + 100 UP * 0.5 + 100 DOWN * 0.5 = 10002, and
    #   after the merge cash 10002 with no positions = 10002; events 4-5 the same;
    #   event 6 (1105): cash 9993 + 20 UP * 0.5 = 10003; events 7-8 the same;
    #   event 9 (resolution, UP wins): cash 9993 + 20 = 10013
    assert [p.equity for p in curve] == pytest.approx(
        [10_000.0, 10_000.0, 10_002.0, 10_002.0, 10_002.0, 10_003.0, 10_003.0, 10_003.0, 10_013.0]
    )
    assert [p.cash for p in curve] == pytest.approx(
        [10_000.0, 10_000.0, 10_002.0, 10_002.0, 10_002.0, 9_993.0, 9_993.0, 9_993.0, 10_013.0]
    )
    assert curve[-1].ts == 1900.0
    assert result.pnl.max_drawdown_usd == 0.0  # the curve never falls
    assert result.pnl.peak_equity == pytest.approx(10_013.0)


def test_s1_losing_variant_drawdown_by_hand() -> None:
    result, _ = run_s1(DOWN, equity_sample_every=1)
    # DOWN wins: the 20 UP shares pay 0, so settle directional pnl = 0 - 9 = -9, total 2 - 9 = -7
    (m,) = result.markets
    assert m.settle_directional_pnl == pytest.approx(-9.0)
    assert m.pnl == pytest.approx(-7.0)
    assert result.final_equity == pytest.approx(9_993.0)
    # peak 10003 (events 6-8), then the resolution brings equity to 9993: depth 10
    assert result.pnl.max_drawdown_usd == pytest.approx(10.0)
    assert result.pnl.max_drawdown_frac == pytest.approx(10.0 / 10_003.0)
    assert result.pnl.peak_equity == pytest.approx(10_003.0)
    assert result.pnl.win_rate_traded == 0.0


def test_s1_equity_sampling_schedule() -> None:
    # first event, every 4th event (4, 8) and the last event (9)
    result, _ = run_s1(UP, equity_sample_every=4)
    assert [p.event_index for p in result.equity_curve] == [1, 4, 8, 9]
    # an interval equal to the stream length samples the first and last event only
    result, _ = run_s1(UP, equity_sample_every=9)
    assert [p.event_index for p in result.equity_curve] == [1, 9]
    result, _ = run_s1(UP, equity_sample_every=1000)
    assert [p.event_index for p in result.equity_curve] == [1, 9]


def test_s1_call_order_matches_the_design_loop() -> None:
    runner = make_runner(S1_SCRIPT)
    for ev in s1_events(UP):
        runner.feed(ev)
    runner.finish()
    engine, exchange = runner.engine, runner.exchange
    assert isinstance(engine, ScriptedEngine) and isinstance(exchange, SpyExchange)
    # fills reach the engine before it sees the snapshot that produced them
    assert engine.log[:2] == [("snapshot", 1100.0), ("snapshot", 1101.0)]
    assert sorted(engine.log[2:4], key=repr) == [("fill", "f1"), ("fill", "f2")]
    assert engine.log[4:6] == [("snapshot", 1102.0), ("merge", 100.0)]
    assert engine.log[6:9] == [("snapshot", 1103.0), ("snapshot", 1104.0), ("fill", "f3")]
    assert engine.log[9:] == [
        ("snapshot", 1105.0),
        ("snapshot", 1106.0),
        ("snapshot", 1107.0),
        ("settle", UP),
    ]
    # the exchange sees process(snap) first, then the actions of that snapshot in order
    assert exchange.calls[:5] == [
        ("process", 1100.0),
        ("submit", "c1"),
        ("submit", "c2"),
        ("process", 1101.0),
        ("process", 1102.0),
    ]
    assert exchange.calls[5] == ("merge", 100.0)


def test_actions_execute_in_the_order_given() -> None:
    script: dict[tuple[str, float], list[Action]] = {
        ("m1", 1100.0): [place("c1", UP, BUY, 0.49, 10.0, POST)],
        ("m1", 1102.0): [
            place("c2", DOWN, BUY, 0.49, 10.0, POST),
            CancelOrder("o1"),
            MergePairs("m1", 5.0),
            place("c3", UP, BUY, 0.48, 10.0, POST),
        ],
    }
    runner = make_runner(script)
    for ts in (1100.0, 1101.0, 1102.0):
        runner.feed(snap(ts))
    exchange = runner.exchange
    assert isinstance(exchange, SpyExchange)
    assert exchange.calls[-4:] == [
        ("submit", "c2"),
        ("cancel", "o1"),
        ("merge", 5.0),
        ("submit", "c3"),
    ]


def test_s1_counters_and_digest() -> None:
    result, engine = run_s1(UP)
    assert result.runner_stats == {
        "orders_submitted": 4,
        "order_rejections": {},
        "cancel_misses": 0,
        "merges_executed": 1,
        "merge_failures": 0,
    }
    ex = result.exchange_stats
    assert (ex["submitted"], ex["accepted"], ex["filled_maker"], ex["filled_taker"]) == (4, 4, 3, 0)
    assert (ex["merges"], ex["settled"]) == (1, 1)
    assert ex["volume_maker"] == pytest.approx(107.0)
    assert result.engine_stats["fills"] == 3  # the stats come from the (scripted) engine object
    assert list(result.exchange_stats) == sorted(result.exchange_stats)
    assert len(result.fills_sha256) == 64
    assert result.fills_sha256 == fills_digest(result.fills)
    assert [f.fill_id for f in result.fills] == ["f1", "f2", "f3"]
    assert engine.portfolio.cash == pytest.approx(10_013.0)


def test_open_order_at_resolution_is_cancelled_by_settlement() -> None:
    # c4 (DOWN 10@0.30) is still resting at the resolution; the run must end clean (invariant 4)
    runner = make_runner(S1_SCRIPT)
    for ev in s1_events(UP)[:-1]:
        runner.feed(ev)
    assert [o.client_id for o in runner.exchange.open_orders("m1")] == ["c4"]
    runner.feed(MarketResolved(1900.0, "m1", UP))
    assert runner.exchange.open_orders() == []
    assert runner.finish().total_pnl == pytest.approx(13.0)


# --------------------------------------------------------------------------- latency 0 + IOC


def latency0_cfg() -> BotConfig:
    cfg = BotConfig()
    return replace(cfg, exchange=replace(cfg.exchange, latency_ticks=0))


def test_latency_zero_ioc_fills_reach_the_engine_via_drain_fills() -> None:
    # IOC buy 50 UP @0.51 hits the ask, IOC sell 50 UP @0.49 hits the bid, both at submit time.
    # fee(p, 50) = 50 * p * 0.25 * (p (1 - p))^2 and p(1-p) = 0.2499 for both prices, so
    #   (0.2499)^2 = 0.06245001
    #   buy fee  = 50 * 0.51 * 0.25 * 0.06245001 = 6.375  * 0.06245001 = 0.39811881375
    #   sell fee = 50 * 0.49 * 0.25 * 0.06245001 = 6.125  * 0.06245001 = 0.38250631125
    # buy cost 25.5 + 0.39811881375 = 25.89811881375; sell proceeds 24.5 - 0.38250631125
    # sell pnl = 24.5 - 0.38250631125 - 25.89811881375 = -1.780625125
    script: dict[tuple[str, float], list[Action]] = {
        ("m1", 1100.0): [
            place("c1", UP, BUY, 0.51, 50.0, IOC),
            place("c2", UP, SELL, 0.49, 50.0, IOC),
        ],
    }
    runner = make_runner(script, latency0_cfg())
    runner.feed(snap(1100.0))
    runner.feed(snap(1101.0))
    runner.feed(MarketResolved(1900.0, "m1", DOWN))
    result = runner.finish()
    assert result.total_pnl == pytest.approx(-1.780625125, abs=1e-9)
    (m,) = result.markets
    assert m.n_fills == 2
    assert (m.buy_shares, m.sell_shares) == (50.0, 50.0)
    assert m.sell_pnl == pytest.approx(-1.780625125, abs=1e-9)
    assert m.fees_paid == pytest.approx(0.39811881375 + 0.38250631125, abs=1e-9)
    assert m.volume_taker == pytest.approx(50.0)  # 25.5 + 24.5
    assert m.volume_maker == 0.0
    assert m.max_inventory_shares == pytest.approx(50.0)
    assert m.ended_unhedged is False
    assert result.activity.n_taker_fills == 2
    assert result.activity.avg_fill_notional == pytest.approx(25.0)
    assert result.pairs.pair_completion_rate == 0.0  # 50 bought, no pair ever formed
    assert [f.ts for f in result.fills] == [1100.0, 1100.0]  # stamped with the submit time


# --------------------------------------------------------------------------- rejections


def test_rejected_orders_are_counted_by_reason_and_the_engine_is_not_told() -> None:
    script: dict[tuple[str, float], list[Action]] = {
        ("m1", 1100.0): [
            place("c1", UP, BUY, 0.495, 100.0, POST),  # price off the 0.01 tick grid
            place("c2", UP, BUY, 0.49, 1.0, POST),  # below the 5-share minimum
            place("c3", UP, BUY, 0.51, 100.0, POST),  # would cross the ask at 0.51
            place("c4", UP, BUY, 0.49, 100_000.0, POST),  # 49000 > cash 10000
            place("c5", UP, SELL, 0.52, 10.0, POST),  # nothing to sell
            place("c6", UP, BUY, 0.49, 10.0, POST),  # accepted
        ],
        ("m1", 1101.0): [CancelOrder("o999"), MergePairs("m1", 10.0)],
    }
    runner = make_runner(script, allow_unresolved=True)
    runner.feed(snap(1100.0))
    runner.feed(snap(1101.0))
    stats = runner.finish().runner_stats
    assert stats["orders_submitted"] == 6
    assert stats["order_rejections"] == {
        "bad_tick": 1,
        "insufficient_cash": 1,
        "insufficient_position": 1,
        "min_size": 1,
        "post_only_crosses": 1,
    }
    assert stats["cancel_misses"] == 1  # o999 does not exist
    assert stats["merge_failures"] == 1  # no pairs held
    assert stats["merges_executed"] == 0
    assert runner.engine.stats["quotes_placed"] == 0  # the engine is never told about rejections
    assert [o.client_id for o in runner.exchange.open_orders()] == ["c6"]


# --------------------------------------------------------------------------- several markets


def test_two_interleaved_markets_are_accounted_separately() -> None:
    # m2 (ETH): BUY DOWN 10@0.49 at 1100; a print of 500 at 0.49 at 1102 drains the queue (400)
    # and fills the 10 shares: cost 4.9. m2 resolves UP, so the DOWN shares pay 0: pnl = -4.9
    script = dict(S1_SCRIPT)
    script[("m2", 1100.0)] = [place("d1", DOWN, BUY, 0.49, 10.0, POST, market_id="m2")]
    runner = make_runner(script)
    events: list[FeedEvent] = []
    for ev in s1_events(UP, resolve=False):
        events.append(ev)
        if isinstance(ev, MarketSnapshot) and ev.ts <= 1102.0:
            trades = [print_(ev.ts, DOWN, 0.49, 500.0)] if ev.ts == 1102.0 else []
            events.append(snap(ev.ts, M2, trades=trades))
    events += [MarketResolved(1900.0, "m1", UP), MarketResolved(1900.0, "m2", UP)]
    for ev in events:
        runner.feed(ev)
    result = runner.finish()
    by_id = {m.market_id: m for m in result.markets}
    assert [m.market_id for m in result.markets] == ["m1", "m2"]  # first-seen order
    assert by_id["m1"].pnl == pytest.approx(13.0)
    assert by_id["m2"].pnl == pytest.approx(-4.9)
    assert by_id["m2"].winner == "UP"
    assert by_id["m2"].max_capital_at_risk == pytest.approx(4.9)
    # a DOWN-heavy market: the net is -10 shares but the reported maximum is its magnitude
    assert by_id["m2"].max_net_shares == pytest.approx(10.0)
    assert by_id["m2"].max_inventory_shares == pytest.approx(10.0)
    assert by_id["m2"].unpaired_token == "DOWN"
    assert result.total_pnl == pytest.approx(13.0 - 4.9)
    assert result.pnl.n_markets_resolved == 2
    # mean = (13 - 4.9) / 2 = 4.05; sample std = |13 + 4.9| / sqrt(2) = 12.6572...; sharpe = 4.05 /
    # 12.6572 * sqrt(2) = 0.4525...  -> equivalently (13 - 4.9) / |13 + 4.9| = 8.1 / 17.9
    assert result.pnl.mean_market_pnl == pytest.approx(4.05)
    assert result.pnl.std_market_pnl == pytest.approx(17.9 / math.sqrt(2.0))
    assert result.pnl.sharpe_like == pytest.approx(8.1 / 17.9)
    assert result.pairs.n_markets == 2


# --------------------------------------------------------------------------- dust and calibration


def dust_run(print_size: float) -> MarketRecord:
    # BUY UP 10@0.49 goes live at 1101 behind 400 displayed shares; a print of ``print_size`` at
    # 0.49 drains the queue and fills the rest: print_size - 400 shares
    script: dict[tuple[str, float], list[Action]] = {
        ("m1", 1100.0): [place("c1", UP, BUY, 0.49, 10.0, POST)],
    }
    runner = make_runner(script)
    for ev in (
        snap(1100.0),
        snap(1101.0),
        snap(1102.0, trades=[print_(1102.0, UP, 0.49, print_size)]),
        MarketResolved(1900.0, "m1", UP),
    ):
        runner.feed(ev)
    (m,) = runner.finish().markets
    return m


def test_fractional_leftover_below_one_share_is_dust_not_unhedged() -> None:
    m = dust_run(400.9)  # fills 0.9 shares: merges are whole pairs, so this cannot be merged
    assert m.n_fills == 1 and m.buy_shares == pytest.approx(0.9)
    assert (m.unpaired_token, m.unpaired_at_close) == ("UP", pytest.approx(0.9))
    assert m.ended_unhedged is False  # reported in unpaired_at_close, but not "unhedged"
    assert m.settle_directional_pnl == pytest.approx(0.9 - 0.9 * 0.49)  # payout - cost = 0.459


def test_exactly_one_unpaired_share_counts_as_unhedged() -> None:
    m = dust_run(401.0)  # fills exactly 1.0 share
    assert m.unpaired_at_close == pytest.approx(1.0)
    assert m.ended_unhedged is True  # the threshold is inclusive


class FairScriptedEngine(ScriptedEngine):
    """Also reports a scripted (phase, fair value) per snapshot ts, as the real engine would."""

    def __init__(self, cfg: BotConfig, fairs: dict[float, tuple[Phase, FairValue]]) -> None:
        super().__init__(cfg, {})
        self.fairs = fairs
        self._now: tuple[Phase, FairValue] | None = None

    def on_snapshot(self, snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]:
        self._now = self.fairs[snap.ts]
        return super().on_snapshot(snap, open_orders)

    def phase_of(self, market_id: str) -> Phase | None:
        return None if self._now is None else self._now[0]

    def last_fair(self, market_id: str) -> FairValue | None:
        return None if self._now is None else self._now[1]


def fair(p_up: float, p_model: float, valid: bool = True) -> FairValue:
    return FairValue(p_up, p_model, 0.0, 0.0, 1e-4, 100.0, 0.0, 0.0, 0.0, valid)


def test_brier_uses_the_last_accumulate_snapshot_of_each_market() -> None:
    # m1: forecasts 0.7 (1101) then 0.9 (1102) in ACCUMULATE, then 0.99 in WIND_DOWN and 0.0 in
    #     FLATTEN: only the last ACCUMULATE one (0.9, model 0.8, market mid 0.5) counts; UP wins
    # m2: the LAST ACCUMULATE snapshot has an invalid fair value -> not scored (even though the
    #     snapshot before it was valid); DOWN wins
    # m3: WARMUP only -> never in ACCUMULATE; UP wins
    fairs = {
        1100.0: (Phase.WARMUP, fair(0.5, 0.5, False)),
        1101.0: (Phase.ACCUMULATE, fair(0.7, 0.6)),
        1102.0: (Phase.ACCUMULATE, fair(0.9, 0.8)),
        1103.0: (Phase.WIND_DOWN, fair(0.99, 0.99)),
        1104.0: (Phase.FLATTEN, fair(0.0, 0.0)),
    }
    cfg = BotConfig()
    runner = BacktestRunner(cfg)
    engine = FairScriptedEngine(cfg, {})
    runner.engine = engine
    for ts in (1100.0, 1101.0, 1102.0, 1103.0, 1104.0):
        engine.fairs = {ts: fairs[ts]}
        runner.feed(snap(ts, M1))
    # m2 snapshots interleave at the same timestamps as m1's last two; use fresh ones after m1
    for ts, fv in (
        (1104.0, (Phase.ACCUMULATE, fair(0.2, 0.1))),
        (1105.0, (Phase.ACCUMULATE, fair(0.5, 0.5, False))),
    ):
        engine.fairs = {ts: fv}
        runner.feed(snap(ts, M2))
    engine.fairs = {1105.0: (Phase.WARMUP, fair(0.5, 0.5, False))}
    runner.feed(snap(1105.0, MarketSpec("m3", "BTC", 1000.0, 1900.0)))
    for mid, winner in (("m1", UP), ("m2", DOWN), ("m3", UP)):
        runner.feed(MarketResolved(1900.0, mid, winner))
    result = runner.finish()
    c = result.calibration
    assert (c.n_resolved, c.n_scored, c.n_invalid, c.n_no_accumulate) == (3, 1, 1, 1)
    assert c.brier_model == pytest.approx(0.01)  # (0.9 - 1)^2
    assert c.brier_model_raw == pytest.approx(0.04)  # (0.8 - 1)^2
    assert c.brier_market == pytest.approx(0.25)  # (0.5 - 1)^2
    assert c.skill_vs_market == pytest.approx(1.0 - 0.01 / 0.25)  # 0.96
    by_id = {m.market_id: m for m in result.markets}
    assert by_id["m1"].p_up_last_accumulate == pytest.approx(0.9)
    assert by_id["m2"].p_up_last_accumulate == pytest.approx(0.5)  # recorded, but not scored
    assert by_id["m3"].p_up_last_accumulate is None


# --------------------------------------------------------------------------- unresolved streams


def test_unresolved_market_at_end_violates_invariant_4() -> None:
    runner = make_runner(S1_SCRIPT)
    for ev in s1_events(UP, resolve=False):
        runner.feed(ev)
    with pytest.raises(InvariantError, match=r"invariant 4.*never resolved.*m1"):
        runner.finish()


def test_unresolved_market_allowed_is_marked_to_mid() -> None:
    runner = make_runner(S1_SCRIPT, allow_unresolved=True)
    for ev in s1_events(UP, resolve=False):
        runner.feed(ev)
    result = runner.finish()
    assert result.unresolved_markets == ["m1"]
    (m,) = result.markets
    assert (m.resolved, m.winner) == (False, None)
    assert m.pnl == pytest.approx(2.0)  # realised so far: the merge
    assert m.settle_directional_pnl == 0.0
    assert m.unpaired_at_close == pytest.approx(20.0)
    # cash 9993 + 20 UP at mid 0.5 = 10003: realised 2 + unrealised (0.50 - 0.45) * 20 = 1
    assert result.final_equity == pytest.approx(10_003.0)
    assert result.total_pnl == pytest.approx(3.0)
    assert result.pnl.n_markets_resolved == 0
    assert result.calibration.n_resolved == 0


def test_unresolved_market_without_invariant_checks_is_accepted() -> None:
    runner = make_runner(S1_SCRIPT, check_invariants=False)
    for ev in s1_events(UP, resolve=False):
        runner.feed(ev)
    assert runner.finish().unresolved_markets == ["m1"]


# --------------------------------------------------------------------------- bad event streams


def test_empty_stream() -> None:
    result = run_backtest(BotConfig(), [], source_label="empty")
    assert result.n_events == 0
    assert (result.first_ts, result.last_ts) == (None, None)
    assert result.final_equity == 10_000.0 and result.total_pnl == 0.0
    assert result.equity_curve == [] and result.markets == []
    assert result.pnl.max_drawdown_usd == 0.0 and result.pnl.peak_equity == 10_000.0
    assert result.activity.avg_fill_notional is None
    assert result.pairs.pair_completion_rate is None
    assert result.pairs.frac_markets_unhedged is None
    assert result.fills_sha256 == fills_digest([])
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_trade_stream_leaves_cash_untouched() -> None:
    # a scripted engine that never acts: both markets resolve with nothing traded
    runner = make_runner({})
    for ev in (snap(1100.0), snap(1101.0), MarketResolved(1900.0, "m1", UP)):
        runner.feed(ev)
    result = runner.finish()
    assert result.final_equity == 10_000.0
    (m,) = result.markets
    assert (m.n_fills, m.pnl, m.traded) == (0, 0.0, False)
    assert m.pair_completion_rate is None  # nothing bought
    assert m.ended_unhedged is False
    assert result.pnl.win_rate_traded is None  # no traded market
    assert result.pairs.frac_traded_markets_unhedged is None


def test_timestamps_must_not_go_backwards() -> None:
    runner = make_runner({})
    runner.feed(snap(1101.0))
    with pytest.raises(ValueError, match="goes backwards"):
        runner.feed(snap(1100.0))
    # an equal timestamp is fine (ties are legal)
    runner2 = make_runner({})
    runner2.feed(snap(1100.0))
    runner2.feed(snap(1100.0, M2))


def test_non_finite_timestamp_is_rejected() -> None:
    runner = make_runner({})
    with pytest.raises(ValueError, match="finite"):
        runner.feed(MarketResolved(math.nan, "m1", UP))


def test_resolution_of_unknown_market_is_rejected() -> None:
    runner = make_runner({})
    runner.feed(snap(1100.0))
    with pytest.raises(ValueError, match="never seen"):
        runner.feed(MarketResolved(1900.0, "nope", UP))


def test_snapshot_after_resolution_is_rejected() -> None:
    runner = make_runner({})
    runner.feed(snap(1100.0))
    runner.feed(MarketResolved(1900.0, "m1", UP))
    with pytest.raises(ValueError, match="already resolved"):
        runner.feed(snap(1901.0))


def test_duplicate_resolution_is_rejected() -> None:
    runner = make_runner({})
    runner.feed(snap(1100.0))
    runner.feed(MarketResolved(1900.0, "m1", UP))
    with pytest.raises(ValueError, match="resolved twice"):
        runner.feed(MarketResolved(1900.0, "m1", DOWN))


def test_non_event_is_a_type_error() -> None:
    runner = make_runner({})
    with pytest.raises(TypeError, match="not a FeedEvent"):
        runner.feed("snapshot")  # type: ignore[arg-type]


def test_runner_cannot_be_reused_after_finish() -> None:
    runner = make_runner({})
    runner.finish()
    with pytest.raises(ValueError, match="already finished"):
        runner.feed(snap(1100.0))
    with pytest.raises(ValueError, match="already finished"):
        runner.finish()


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "60"])
def test_equity_sample_every_must_be_a_positive_int(bad: object) -> None:
    with pytest.raises(ValueError, match="equity_sample_every"):
        BacktestRunner(BotConfig(), equity_sample_every=bad)  # type: ignore[arg-type]


def test_invalid_config_is_rejected() -> None:
    cfg = BotConfig()
    bad = replace(cfg, exchange=replace(cfg.exchange, initial_cash=-1.0))
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestRunner(bad)


def test_engine_unknown_action_is_a_type_error() -> None:
    runner = make_runner({("m1", 1100.0): ["nonsense"]})  # type: ignore[list-item]
    with pytest.raises(TypeError, match="unknown action"):
        runner.feed(snap(1100.0))


# --------------------------------------------------------------------------- fault injection


class SkewedExchange(PaperExchange):
    """An exchange whose reported cash/positions/reservations are wrong after ``after`` calls."""

    def __init__(self, cfg: BotConfig, *, cash_skew: float = 0.0, reserved: float = 0.0) -> None:
        super().__init__(cfg)
        self.cash_skew, self.reserved, self.position_skew = cash_skew, reserved, 0.0

    def balance(self) -> float:
        return super().balance() + self.cash_skew

    def position(self, market_id: str, token: Outcome) -> float:
        return super().position(market_id, token) + self.position_skew

    @property
    def reserved_cash(self) -> float:
        return super().reserved_cash + self.reserved


def skewed_runner(**skew: float) -> BacktestRunner:
    cfg = BotConfig()
    runner = BacktestRunner(cfg)
    runner.engine = ScriptedEngine(cfg, S1_SCRIPT)
    runner.exchange = SkewedExchange(cfg, **skew)
    return runner


def test_invariant_1_cash_reconciliation_failure_is_reported_precisely() -> None:
    runner = skewed_runner(cash_skew=0.01)
    with pytest.raises(InvariantError) as err:
        runner.feed(snap(1100.0))
    msg = str(err.value)
    assert "invariant 1 (reconciliation)" in msg
    assert "event #1" in msg and "market='m1'" in msg
    assert "engine cash 10000 != exchange cash 10000.01" in msg


def test_invariant_1_position_reconciliation_failure() -> None:
    runner = skewed_runner()
    runner.feed(snap(1100.0))
    assert isinstance(runner.exchange, SkewedExchange)
    runner.exchange.position_skew = 0.5
    with pytest.raises(InvariantError, match=r"invariant 1.*UP position 0 != exchange 0\.5"):
        runner.feed(snap(1101.0))


def test_invariant_2_realised_pnl_identity_failure() -> None:
    runner = make_runner(S1_SCRIPT)
    for ev in s1_events(UP)[:4]:
        runner.feed(ev)
    # corrupt the per-market realised pnl only (cash still reconciles with the exchange)
    runner.engine.portfolio.inventory("m1").sell_pnl += 1.0
    with pytest.raises(InvariantError, match=r"invariant 2.*realised pnl"):
        runner.feed(snap(1104.0))


def test_invariant_3_negative_cash_and_reserved_cash() -> None:
    runner = make_runner({})
    with pytest.raises(InvariantError, match=r"invariant 3 \(non-negative\).*negative cash"):
        runner._check_nonnegative(-1.0, -1.0, "somewhere")
    runner._check_nonnegative(-1e-9, 5.0, "inside the tolerance")  # within TOL: fine
    runner.exchange = SkewedExchange(BotConfig(), reserved=1e9)
    with pytest.raises(InvariantError, match=r"invariant 3 \(reserved cash\).*exceeds cash 5"):
        runner._check_nonnegative(5.0, 5.0, "somewhere")


def test_exchange_reserving_more_than_its_cash_is_caught_during_a_run() -> None:
    # the exchange's own internal check fires first; the runner reports it as an invariant error
    with pytest.raises(InvariantError, match="reserved cash"):
        skewed_runner(reserved=1e9).feed(snap(1100.0))


def test_invariant_3_negative_quantity() -> None:
    runner = skewed_runner()
    runner.feed(snap(1100.0))
    runner.engine.portfolio.inventory("m1").qty[UP] = -1.0
    assert isinstance(runner.exchange, SkewedExchange)
    runner.exchange.position_skew = -1.0  # reconciles, but a negative quantity is still wrong
    with pytest.raises(InvariantError, match=r"invariant 3.*negative UP quantity"):
        runner.feed(snap(1101.0))


def test_invariant_4_stray_order_in_a_resolved_market_is_detected() -> None:
    runner = make_runner({})
    runner.feed(snap(1100.0))
    runner.feed(MarketResolved(1900.0, "m1", UP))
    stray = OpenOrder("o9", "c9", "m1", UP, BUY, 0.4, 10.0, 10.0, POST, 1900.0)
    runner.exchange.open_orders = lambda market_id=None: [stray]  # type: ignore[method-assign]
    with pytest.raises(InvariantError, match=r"invariant 4.*open order o9 in resolved"):
        runner.finish()


def test_final_accounting_failure_is_detected_at_the_end() -> None:
    runner = make_runner({})
    runner.feed(snap(1100.0))
    runner.feed(MarketResolved(1900.0, "m1", UP))
    # both sides agree on the corrupted cash, so only the end-of-run identities can notice
    runner.engine.portfolio.cash += 0.5
    assert isinstance(runner.exchange, SpyExchange)
    runner.exchange._cash += 0.5  # fault injection
    with pytest.raises(InvariantError, match="invariant (2|4)"):
        runner.finish()


def test_engine_rejecting_an_exchange_fill_is_an_invariant_violation() -> None:
    class RejectingEngine(ScriptedEngine):
        def on_fill(self, fill: Fill) -> None:
            raise ValueError("portfolio says no")

    cfg = BotConfig()
    runner = BacktestRunner(cfg)
    runner.engine = RejectingEngine(cfg, S1_SCRIPT)
    events = s1_events(UP)
    for ev in events[:2]:
        runner.feed(ev)
    with pytest.raises(InvariantError, match=r"invariant 1.*rejected exchange fill f1.*says no"):
        runner.feed(events[2])


def test_component_assertion_errors_become_invariant_errors() -> None:
    class BrokenExchange(PaperExchange):
        def process(self, snap: MarketSnapshot) -> list[Fill]:
            raise AssertionError("PaperExchange invariant broken: boom")

    cfg = BotConfig()
    runner = BacktestRunner(cfg)
    runner.exchange = BrokenExchange(cfg)
    with pytest.raises(InvariantError, match="component check failed: .*boom"):
        runner.feed(snap(1100.0))


def test_component_assertion_at_the_end_of_the_stream_is_wrapped_too() -> None:
    class BrokenAtEnd(PaperExchange):
        def open_orders(self, market_id: str | None = None) -> list[OpenOrder]:
            raise AssertionError("PaperExchange invariant broken: end")

    cfg = BotConfig()
    runner = BacktestRunner(cfg)
    runner.exchange = BrokenAtEnd(cfg)
    with pytest.raises(InvariantError, match="end of stream: component check failed: .*end"):
        runner.finish()


def test_check_invariants_false_skips_the_checks() -> None:
    cfg = BotConfig()
    runner = BacktestRunner(cfg, check_invariants=False)
    runner.engine = ScriptedEngine(cfg, S1_SCRIPT)
    runner.exchange = SkewedExchange(cfg, cash_skew=0.01)
    for ev in s1_events(UP):
        runner.feed(ev)
    result = runner.finish()
    assert result.invariants_checked is False


def test_full_sweep_runs_periodically(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_mod, "FULL_SWEEP_EVERY", 3)
    runner = make_runner(S1_SCRIPT)
    calls: list[int] = []
    original = runner._check_full

    def spying_sweep() -> None:
        calls.append(runner._n)
        original()

    runner._check_full = spying_sweep  # type: ignore[method-assign]
    for ev in s1_events(UP):
        runner.feed(ev)
    runner.finish()
    assert calls == [3, 6, 9, 9]  # every 3rd event, plus the end-of-stream sweep


# --------------------------------------------------------------------------- pure aggregations


def rec(**overrides: object) -> MarketRecord:
    base = MarketRecord(
        market_id="m", asset="BTC", resolved=True, winner="UP", sell_pnl=0.0, merge_pnl=0.0,
        settle_pair_pnl=0.0, settle_directional_pnl=0.0, fees_paid=0.0, pnl=0.0, n_fills=0,
        buy_shares=0.0, sell_shares=0.0, volume_maker=0.0, volume_taker=0.0,
        max_inventory_shares=0.0, max_net_shares=0.0, max_capital_at_risk=0.0, merged_pairs=0.0,
        paired_at_close=0.0, avg_pair_cost_at_close=None, avg_merge_pair_cost=None,
        unpaired_token=None, unpaired_at_close=0.0, ended_unhedged=False,
        pair_completion_rate=None, p_up_last_accumulate=None,
    )  # fmt: skip
    return replace(base, **overrides)  # type: ignore[arg-type]


def fill(
    fid: str, token: Outcome, side: Side, price: float, size: float, maker: bool, fee: float = 0.0
) -> Fill:
    return Fill(fid, "o1", "m1", token, side, price, size, fee, maker, 1.0)


def test_activity_stats_by_hand() -> None:
    fills = [
        fill("f1", UP, BUY, 0.40, 100.0, True),  # notional 40
        fill("f2", DOWN, BUY, 0.55, 20.0, False),  # notional 11
        fill("f3", UP, SELL, 0.30, 10.0, False),  # notional 3
    ]
    a = activity_stats(fills)
    assert (a.n_fills, a.n_buy_fills, a.n_sell_fills) == (3, 2, 1)
    assert (a.n_maker_fills, a.n_taker_fills) == (1, 2)
    assert a.volume_maker == pytest.approx(40.0)
    assert a.volume_taker == pytest.approx(14.0)  # 11 + 3
    assert a.avg_fill_notional == pytest.approx(18.0)  # (40 + 11 + 3) / 3
    assert a.median_fill_notional == pytest.approx(11.0)  # sorted 3, 11, 40
    empty = activity_stats([])
    assert (empty.n_fills, empty.avg_fill_notional, empty.median_fill_notional) == (0, None, None)
    assert empty.volume_maker == empty.volume_taker == 0.0


def test_pair_stats_by_hand() -> None:
    records = [
        # A: bought 200, merged 90 pairs at pnl 4.5 (cost 1 - 4.5/90 = 0.95), 5 pairs held at 0.90
        rec(
            n_fills=4, buy_shares=200.0, merged_pairs=90.0, merge_pnl=4.5, paired_at_close=5.0,
            avg_pair_cost_at_close=0.90,
        ),
        # B: bought 100, nothing paired, 50 unhedged at close
        rec(n_fills=2, buy_shares=100.0, unpaired_token="UP", unpaired_at_close=50.0,
            ended_unhedged=True),
        # C: never traded
        rec(),
    ]  # fmt: skip
    p = pair_stats(records)
    assert (p.n_markets, p.n_markets_traded) == (3, 2)
    assert p.shares_bought == 300.0 and p.merged_pairs == 90.0 and p.paired_at_close == 5.0
    assert p.pairs_formed == 95.0
    assert p.pair_completion_rate == pytest.approx(190.0 / 300.0)  # 2 * 95 / 300
    assert p.avg_merge_pair_cost == pytest.approx(0.95)  # 1 - 4.5 / 90
    assert p.avg_close_pair_cost == pytest.approx(0.90)
    assert p.n_markets_unhedged == 1
    assert p.frac_markets_unhedged == pytest.approx(1.0 / 3.0)
    assert p.frac_traded_markets_unhedged == pytest.approx(0.5)


def test_pair_stats_pools_costs_weighted_by_size() -> None:
    records = [
        rec(n_fills=1, buy_shares=40.0, merged_pairs=10.0, merge_pnl=0.5,  # cost 0.95
            paired_at_close=10.0, avg_pair_cost_at_close=0.90),
        rec(n_fills=1, buy_shares=120.0, merged_pairs=30.0, merge_pnl=0.3,  # cost 0.99
            paired_at_close=30.0, avg_pair_cost_at_close=0.98),
    ]  # fmt: skip
    p = pair_stats(records)
    # pooled merge cost = 1 - (0.5 + 0.3) / 40 = 0.98 (size-weighted, not the mean of 0.95 / 0.99)
    assert p.avg_merge_pair_cost == pytest.approx(0.98)
    # pooled close cost = (0.90 * 10 + 0.98 * 30) / 40 = 38.4 / 40 = 0.96
    assert p.avg_close_pair_cost == pytest.approx(0.96)
    assert p.pair_completion_rate == pytest.approx(1.0)  # 2 * (40 + 40) pairs / 160 bought


def test_pair_stats_empty() -> None:
    p = pair_stats([])
    assert (p.n_markets, p.shares_bought, p.pairs_formed) == (0, 0.0, 0.0)
    assert p.pair_completion_rate is None and p.avg_merge_pair_cost is None
    assert p.frac_markets_unhedged is None


def test_pnl_stats_by_hand() -> None:
    records = [
        rec(n_fills=1, pnl=10.0, merge_pnl=7.0, sell_pnl=1.0, settle_pair_pnl=1.0,
            settle_directional_pnl=1.0, fees_paid=0.5),
        rec(n_fills=1, pnl=-4.0, merge_pnl=1.0, sell_pnl=-6.0, settle_pair_pnl=0.0,
            settle_directional_pnl=1.0, fees_paid=0.25),
        rec(n_fills=1, pnl=2.0, merge_pnl=2.0, fees_paid=0.25),
        rec(n_fills=1, resolved=False, pnl=3.0, merge_pnl=3.0),  # unresolved: excluded
    ]  # fmt: skip
    s = pnl_stats(records, [10_000.0, 10_050.0, 9_950.0, 10_010.0], 10_000.0)
    assert s.n_markets_resolved == 3
    assert s.mean_market_pnl == pytest.approx(8.0 / 3.0)  # (10 - 4 + 2) / 3
    assert s.std_market_pnl == pytest.approx(math.sqrt(148.0 / 3.0))  # sum sq dev 296/3, /2
    assert s.sharpe_like == pytest.approx(4.0 / math.sqrt(37.0))  # 8 / sqrt(148)
    assert (s.best_market_pnl, s.worst_market_pnl) == (10.0, -4.0)
    assert s.win_rate_traded == pytest.approx(2.0 / 3.0)
    assert (s.pnl_merge, s.pnl_sell) == (10.0, -5.0)
    assert (s.pnl_settle_pair, s.pnl_settle_directional) == (1.0, 2.0)
    assert s.fees_paid == pytest.approx(1.0)
    # peak 10050 then 9950: depth 100 = 0.995% of the peak
    assert s.peak_equity == 10_050.0
    assert s.max_drawdown_usd == pytest.approx(100.0)
    assert s.max_drawdown_frac == pytest.approx(100.0 / 10_050.0)


def test_win_rate_counts_only_strictly_profitable_traded_markets() -> None:
    records = [
        rec(n_fills=1, pnl=5.0),
        rec(n_fills=1, pnl=0.0),
        rec(n_fills=1, pnl=-1.0),
        rec(pnl=0.0),
    ]
    s = pnl_stats(records, [10_000.0], 10_000.0)
    assert s.win_rate_traded == pytest.approx(1.0 / 3.0)  # untraded markets are not counted
    assert s.n_markets_resolved == 4


def test_pnl_stats_without_resolved_markets_or_curve() -> None:
    s = pnl_stats([], [], 10_000.0)
    assert (s.mean_market_pnl, s.std_market_pnl, s.sharpe_like) == (0.0, 0.0, 0.0)
    assert (s.best_market_pnl, s.worst_market_pnl, s.win_rate_traded) == (0.0, 0.0, None)
    assert (s.peak_equity, s.max_drawdown_usd, s.max_drawdown_frac) == (10_000.0, 0.0, 0.0)


def test_calibration_stats_by_hand() -> None:
    samples: list[tuple[CalibSample | None, Outcome]] = [
        (CalibSample(0.8, 0.9, 0.6, True), UP),  # scored
        (CalibSample(0.3, 0.2, 0.5, True), DOWN),  # scored
        (None, UP),  # never in ACCUMULATE
        (CalibSample(0.5, 0.5, 0.5, False), UP),  # invalid fair value
        (CalibSample(0.5, 0.5, None, True), DOWN),  # no UP mid
    ]
    c = calibration_stats(samples)
    assert (c.n_resolved, c.n_scored, c.n_no_accumulate, c.n_invalid) == (5, 2, 1, 2)
    # engine p_up:  ((0.8 - 1)^2 + (0.3 - 0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
    # raw model:    ((0.9 - 1)^2 + (0.2 - 0)^2) / 2 = (0.01 + 0.04) / 2 = 0.025
    # market mid:   ((0.6 - 1)^2 + (0.5 - 0)^2) / 2 = (0.16 + 0.25) / 2 = 0.205
    assert c.brier_model == pytest.approx(0.065)
    assert c.brier_model_raw == pytest.approx(0.025)
    assert c.brier_market == pytest.approx(0.205)
    assert c.skill_vs_market == pytest.approx(1.0 - 0.065 / 0.205)


def test_calibration_stats_nothing_scored() -> None:
    c = calibration_stats([(None, UP), (CalibSample(0.5, 0.5, 0.5, False), DOWN)])
    assert (c.n_resolved, c.n_scored, c.n_no_accumulate, c.n_invalid) == (2, 0, 1, 1)
    assert c.brier_model is None and c.brier_market is None and c.skill_vs_market is None
    assert calibration_stats([]).n_resolved == 0


def test_mark_to_mid_by_hand() -> None:
    inv = MarketInventory("m1")
    inv.qty[UP], inv.cost[UP] = 10.0, 4.0
    inv.qty[DOWN], inv.cost[DOWN] = 5.0, 3.0
    assert mark_to_mid(inv, 0.6, 0.4) == pytest.approx(8.0)  # 10 * 0.6 + 5 * 0.4
    assert mark_to_mid(inv, 0.6, 0.45) == pytest.approx(8.25)  # each token at its own mid
    assert mark_to_mid(inv, 0.6, None) == pytest.approx(8.0)  # DOWN at 1 - 0.6: 6 + 2
    assert mark_to_mid(inv, None, 0.3) == pytest.approx(8.5)  # UP at 0.7: 7 + 1.5
    assert mark_to_mid(inv, None, None) == pytest.approx(7.0)  # at cost: 4 + 3
    assert mark_to_mid(MarketInventory("empty"), 0.5, 0.5) == 0.0


def test_fills_digest_is_sensitive_to_every_field_and_the_order() -> None:
    base = [fill("f1", UP, BUY, 0.4, 10.0, True), fill("f2", DOWN, SELL, 0.5, 5.0, False, 0.01)]
    assert fills_digest(base) == fills_digest(list(base))
    assert fills_digest(base) != fills_digest(base[::-1])
    assert fills_digest(base) != fills_digest(base[:1])
    for change in (
        {"fill_id": "fX"}, {"order_id": "oX"}, {"market_id": "mX"}, {"token": DOWN},
        {"side": SELL}, {"price": 0.41}, {"size": 10.5}, {"fee": 0.5}, {"is_maker": False},
        {"ts": 2.0},
    ):  # fmt: skip
        mutated = [replace(base[0], **change), base[1]]
        assert fills_digest(mutated) != fills_digest(base), change


# --------------------------------------------------------------------------- randomised properties


def test_activity_stats_matches_a_reference_on_random_fills() -> None:
    rng = random.Random(5)
    for _ in range(40):
        fills = [
            fill(
                f"f{i}",
                rng.choice([UP, DOWN]),
                rng.choice([BUY, SELL]),
                round(rng.uniform(0.01, 0.99), 2),
                round(rng.uniform(1.0, 200.0), 1),
                rng.random() < 0.6,
            )
            for i in range(rng.randint(1, 30))
        ]
        a = activity_stats(fills)
        notionals = [f.price * f.size for f in fills]
        assert a.n_fills == len(fills)
        assert a.n_buy_fills + a.n_sell_fills == a.n_fills
        assert a.n_maker_fills + a.n_taker_fills == a.n_fills
        assert a.volume_maker + a.volume_taker == pytest.approx(sum(notionals))
        assert a.avg_fill_notional == pytest.approx(statistics.fmean(notionals))
        assert a.median_fill_notional == pytest.approx(statistics.median(notionals))
        assert a.volume_maker == pytest.approx(sum(f.notional for f in fills if f.is_maker))


def test_mark_to_mid_is_consistent_for_mirrored_books_on_random_inventories() -> None:
    rng = random.Random(6)
    for _ in range(50):
        inv = MarketInventory("m")
        inv.qty[UP], inv.cost[UP] = rng.uniform(0, 300), rng.uniform(0, 150)
        inv.qty[DOWN], inv.cost[DOWN] = rng.uniform(0, 300), rng.uniform(0, 150)
        m = rng.uniform(0.02, 0.98)
        expected = inv.qty[UP] * m + inv.qty[DOWN] * (1.0 - m)
        # a mirrored pair of mids, or either mid alone, values the position identically
        assert mark_to_mid(inv, m, 1.0 - m) == pytest.approx(expected)
        assert mark_to_mid(inv, m, None) == pytest.approx(expected)
        assert mark_to_mid(inv, None, 1.0 - m) == pytest.approx(expected)
        assert mark_to_mid(inv, 0.5, 0.5) == pytest.approx(0.5 * (inv.qty[UP] + inv.qty[DOWN]))


def test_pair_and_calibration_stats_match_a_reference_on_random_records() -> None:
    rng = random.Random(8)
    for _ in range(40):
        records = []
        for i in range(rng.randint(1, 12)):
            merged = float(rng.randint(0, 200))
            held = round(rng.uniform(0.0, 20.0), 2)
            extra = round(rng.uniform(0.0, 60.0), 2)
            bought = 2.0 * (merged + held) + extra  # at least what the pairs need
            close_cost = rng.uniform(0.8, 1.0) if held else None
            records.append(
                rec(
                    market_id=f"m{i}", n_fills=1 if bought > 0 else 0, buy_shares=bought,
                    merged_pairs=merged, merge_pnl=merged * rng.uniform(0.0, 0.05),
                    paired_at_close=held, avg_pair_cost_at_close=close_cost,
                    ended_unhedged=extra >= 1.0,
                )
            )  # fmt: skip
        p = pair_stats(records)
        merged_total = sum(r.merged_pairs for r in records)
        held_total = sum(r.paired_at_close for r in records)
        bought_total = sum(r.buy_shares for r in records)
        assert p.pairs_formed == pytest.approx(merged_total + held_total)
        if bought_total > 0:
            assert p.pair_completion_rate == pytest.approx(
                2.0 * (merged_total + held_total) / bought_total
            )
            assert 0.0 <= (p.pair_completion_rate or 0.0) <= 1.0 + 1e-12
        assert p.n_markets_unhedged == sum(r.ended_unhedged for r in records)
        assert p.n_markets_traded == sum(r.n_fills > 0 for r in records)

        samples = [
            (CalibSample(rng.random(), rng.random(), rng.random(), True), rng.choice([UP, DOWN]))
            for _ in range(rng.randint(1, 10))
        ]
        c = calibration_stats(samples)
        ys = [1.0 if w is UP else 0.0 for _, w in samples]
        assert c.n_scored == len(samples)
        assert c.brier_model == pytest.approx(
            sum((s.p_up - y) ** 2 for (s, _), y in zip(samples, ys, strict=True)) / len(ys)
        )
        assert c.brier_market == pytest.approx(
            sum((s.up_mid - y) ** 2 for (s, _), y in zip(samples, ys, strict=True)) / len(ys)  # type: ignore[operator]
        )
        assert 0.0 <= (c.brier_model or 0.0) <= 1.0


# --------------------------------------------------------------------------- serialisation


def test_to_dict_is_json_safe_stable_and_complete() -> None:
    result, _ = run_s1(UP, source_label="synthetic:test")
    d = result.to_dict()
    text = json.dumps(d, allow_nan=False)
    assert json.loads(text) == d
    assert list(d)[:6] == [
        "source_label", "invariants_checked", "equity_sample_every", "n_events", "n_snapshots",
        "n_resolutions",
    ]  # fmt: skip
    assert "fills" not in d and d["fills_sha256"] == result.fills_sha256
    assert set(d) == {f.name for f in dataclasses.fields(BacktestResult)} - {"fills"}
    assert d["markets"][0]["winner"] == "UP"  # enums are plain strings
    assert isinstance(d["equity_curve"][0], list) and len(d["equity_curve"][0]) == 4
    # a second identical run gives a byte-identical document
    again, _ = run_s1(UP, source_label="synthetic:test")
    assert json.dumps(again.to_dict()) == json.dumps(d)


def test_to_dict_rejects_non_finite_numbers() -> None:
    result, _ = run_s1(UP)
    result.runner_stats["bad"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        result.to_dict()


def test_to_dict_rejects_unserialisable_values() -> None:
    result, _ = run_s1(UP)
    result.runner_stats["bad"] = object()
    with pytest.raises(TypeError, match="cannot serialise"):
        result.to_dict()
