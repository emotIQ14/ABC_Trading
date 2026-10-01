"""Whole-system tests added by the integration pass: real feed -> engine -> paper exchange.

Each module has its own tests; these check properties that only exist once the modules are
connected (DESIGN section 7, items 5-9), on short synthetic runs (BTC, 300 s windows, ~0.3 s each).
They check MECHANICS only: nothing here says anything about profitability, and no number is tuned.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from abc_trading.backtest.runner import BacktestResult, BacktestRunner
from abc_trading.config import BotConfig, FeeConfig, PaperExchangeConfig
from abc_trading.exchange.paper import PaperExchange
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.types import (
    Action,
    BookSnapshot,
    FeedEvent,
    Level,
    MarketResolved,
    MarketSnapshot,
    MarketSpec,
    OpenOrder,
    OrderRequest,
    Outcome,
    Phase,
    PlaceOrder,
    Side,
    SubmitResult,
    TimeInForce,
)

EPS = 1e-9
UP, DOWN = Outcome.UP, Outcome.DOWN


def small_cfg(*, latency: int = 1, **sim: object) -> BotConfig:
    """Two 300 s BTC windows (602 events); ``sim`` overrides SimConfig fields."""
    cfg = BotConfig()
    fields: dict[str, object] = {"n_windows": 2, "assets": ("BTC",), "window_seconds": 300}
    fields.update(sim)
    return replace(
        cfg,
        sim=replace(cfg.sim, **fields),  # type: ignore[arg-type]
        exchange=replace(cfg.exchange, latency_ticks=latency),
    )


# --------------------------------------------------------------------------- engine decisions


class _Audit:
    """Counters and violations collected while auditing every ``PlaceOrder`` of an engine."""

    def __init__(self) -> None:
        self.n: Counter[str] = Counter()
        self.violations: list[str] = []

    def bad(self, what: str, snap: MarketSnapshot, extra: str = "") -> None:
        t = snap.ts - snap.market.start_ts
        self.violations.append(f"{what}: {snap.market.market_id} t={t:.0f} {extra}")


def _audit_snapshot(
    engine: MarketMakerEngine,
    cfg: BotConfig,
    snap: MarketSnapshot,
    actions: Sequence[Action],
    audit: _Audit,
) -> None:
    mid = snap.market.market_id
    phase = engine.phase_of(mid)
    inv = engine.portfolio.inventory(mid)  # merges are applied later, so this is pre-action
    tick, margin = snap.market.tick_size, cfg.pair.target_margin
    gated = engine.risk.kill_switch or not engine.risk.spot_ok(snap)
    placed = [a.request for a in actions if isinstance(a, PlaceOrder)]
    if phase in (Phase.WARMUP, Phase.DONE) and placed:
        audit.bad("DESIGN 7.9: order in WARMUP/DONE", snap)
    for req in placed:
        book = snap.book(req.token)
        opposite = req.token.opposite
        completing = inv.qty[opposite] > inv.qty[req.token] + EPS
        if req.tif is TimeInForce.POST_ONLY:
            audit.n["post_only"] += 1
            ask = book.best_ask
            if req.side is not Side.BUY:
                audit.bad("post-only SELL", snap)
            if ask is None or req.price > ask - tick + EPS:  # DESIGN 7.7
                audit.bad("DESIGN 7.7: post-only crosses at emission", snap, f"{req.price} {ask}")
            if phase not in (Phase.ACCUMULATE, Phase.WIND_DOWN) or gated:
                audit.bad("DESIGN 7.9: quote in a no-quote state", snap, str(phase))
            avg = inv.avg_cost(opposite)
            if completing and avg is not None:
                audit.n["completing_bids"] += 1
                if req.price + avg > 1.0 - margin + EPS:  # DESIGN 7.8
                    audit.bad("DESIGN 7.8: pair cap violated", snap, f"{req.price}+{avg}")
        elif req.side is Side.BUY:
            audit.n["ioc_buys"] += 1
            if phase not in (Phase.ACCUMULATE, Phase.WIND_DOWN) or gated:
                audit.bad("DESIGN 7.9: taker buy in a no-trade state", snap, str(phase))
            if not completing or req.size > inv.qty[opposite] - inv.qty[req.token] + EPS:
                audit.bad("IOC buy that does not complete pairs", snap)
            if req.price != book.best_ask:
                audit.bad("IOC buy not at the best ask", snap)
        else:
            audit.n["ioc_sells"] += 1
            heavy, _ = inv.unpaired()
            if phase is not Phase.FLATTEN or heavy is not req.token:
                audit.bad("sell outside FLATTEN or of the light token", snap, str(phase))
            if req.size > inv.qty[req.token] + EPS or req.price != book.best_bid:
                audit.bad("sell larger than the holding or not at the best bid", snap)


def _stale_spot_stretches(events: Sequence[FeedEvent]) -> list[FeedEvent]:
    """Make the spot 30 s old between 60 s and 140 s into every window (a feed outage)."""
    out: list[FeedEvent] = []
    for e in events:
        if isinstance(e, MarketSnapshot) and 60.0 <= e.seconds_since_start < 140.0:
            e = dataclasses.replace(e, spot_ts=e.ts - 30.0)
        out.append(e)
    return out


def _run_audited(
    cfg: BotConfig, transform: Callable[[list[FeedEvent]], list[FeedEvent]] | None = None
) -> tuple[BacktestResult, _Audit]:
    runner = BacktestRunner(cfg, check_invariants=True)
    engine, audit = runner.engine, _Audit()
    original = engine.on_snapshot

    def audited(snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]:
        actions = original(snap, open_orders)
        _audit_snapshot(engine, cfg, snap, actions, audit)
        return actions

    engine.on_snapshot = audited  # type: ignore[method-assign]
    events = list(SyntheticFeed(cfg))
    for event in events if transform is None else transform(events):
        runner.feed(event)
    return runner.finish(), audit


@pytest.mark.parametrize(
    ("name", "cfg"),
    [
        ("default", small_cfg()),
        ("latency0", small_cfg(latency=0)),
        ("latency3", small_cfg(latency=3)),
        ("no_adverse_selection", small_cfg(informed_flow=0.0)),
        ("no_market_lag", small_cfg(market_lag_seconds=0.0)),
        (
            "directional_off",
            replace(small_cfg(), directional=replace(BotConfig().directional, enabled=False)),
        ),
    ],
)
def test_engine_decisions_respect_design_invariants_7_8_9_in_a_real_run(
    name: str, cfg: BotConfig
) -> None:
    """DESIGN 7.7 (no crossing post-only), 7.8 (pair cap) and 7.9 (nothing outside the quoting
    phases, nothing while gated) hold for every order of a run through the real exchange."""
    result, audit = _run_audited(cfg)
    assert audit.violations == [], name
    assert audit.n["post_only"] > 100  # the run really quoted
    assert result.runner_stats["order_rejections"] == {}  # and the exchange never refused one
    assert result.runner_stats["merge_failures"] == 0 and result.runner_stats["cancel_misses"] == 0


def test_a_latched_kill_switch_and_a_stale_spot_stop_all_quoting_in_a_real_run() -> None:
    """DESIGN 7.9: no quotes or taker buys while the kill switch is latched or the spot is stale
    (the audit's ``gated`` flag), in runs where both gates actually trip."""
    windows = 3
    base = small_cfg(latency=0, seed=2, n_windows=windows)
    tight = replace(base, risk=replace(base.risk, max_daily_loss_usd=15.0))
    result, audit = _run_audited(tight)
    assert audit.violations == [] and result.kill_switch is True
    assert result.engine_stats["risk_blocked_snapshots"] > 200  # blocked for the rest of the run
    result, audit = _run_audited(base, _stale_spot_stretches)
    assert audit.violations == [] and result.kill_switch is False
    # 80 stale snapshots per window, and in each of them the gate cancelled instead of quoting
    assert result.engine_stats["risk_trips_spot_stale"] == windows
    assert result.engine_stats["risk_blocked_snapshots"] >= 80 * windows


def test_the_audit_runs_exercise_completing_bids_taker_buys_and_flatten_sells() -> None:
    _, audit = _run_audited(small_cfg(latency=0, seed=2))
    assert audit.violations == []
    assert audit.n["completing_bids"] > 0 and audit.n["ioc_buys"] > 0 and audit.n["ioc_sells"] > 0


# --------------------------------------------------------------------------- fills vs the feed


@pytest.mark.parametrize("latency", [0, 1, 2])
def test_every_fill_is_explained_by_the_feed(latency: int) -> None:
    """Each maker fill needs a print of its token at or through the fill price in the same
    snapshot and an order that was live for a snapshot first; each taker fill needs a displayed
    level of the activation snapshot with enough size (DESIGN 5 and 7.6, checked end to end)."""
    cfg = small_cfg(latency=latency, n_windows=3)
    runner = BacktestRunner(cfg, check_invariants=True)
    created: dict[str, tuple[float, TimeInForce, float, float]] = {}
    original = runner.exchange.submit

    def submit(req: OrderRequest, ts: float) -> SubmitResult:
        result = original(req, ts)
        if result.ok and result.order_id is not None:
            created[result.order_id] = (ts, req.tif, req.price, req.size)
        return result

    runner.exchange.submit = submit  # type: ignore[method-assign]
    tick_s = cfg.sim.tick_seconds
    seen = Counter[str]()
    for event in SyntheticFeed(cfg):
        before = len(runner._fills)
        runner.feed(event)
        if not isinstance(event, MarketSnapshot):
            continue
        for fill in runner._fills[before:]:
            ts0, tif, limit, size = created[fill.order_id]
            assert fill.size <= size + EPS and fill.ts == event.ts
            if fill.is_maker:
                seen["maker"] += 1
                assert tif is TimeInForce.POST_ONLY and fill.price == limit
                assert fill.ts - ts0 >= (latency + 1) * tick_s - EPS  # live for one snapshot first
                aggressor = Side.SELL if fill.side is Side.BUY else Side.BUY
                sign = 1.0 if fill.side is Side.BUY else -1.0  # BUY: print <= price; SELL: >=
                assert any(
                    t.token is fill.token
                    and t.aggressor is aggressor
                    and sign * (t.price - fill.price) <= EPS
                    for t in event.trades
                ), fill
            else:
                seen["taker"] += 1
                assert tif is TimeInForce.IOC and fill.ts - ts0 >= latency * tick_s - EPS
                book = event.book(fill.token)
                ladder = book.asks if fill.side is Side.BUY else book.bids
                shown = sum(lv.size for lv in ladder if abs(lv.price - fill.price) <= EPS)
                assert shown > 0.0 and fill.size <= shown + 1e-6, fill
                if fill.side is Side.BUY:
                    assert fill.price <= limit + EPS
                else:
                    assert fill.price >= limit - EPS
    runner.finish()
    assert seen["maker"] >= 5 and seen["taker"] >= 1


# --------------------------------------------------------------------------- no look-ahead


def _fills(result: BacktestResult, until: float) -> list[tuple[object, ...]]:
    return [
        (f.fill_id, f.order_id, f.market_id, f.token, f.side, f.price, f.size, f.fee, f.ts)
        for f in result.fills
        if f.ts <= until
    ]


def _scrambled_future(events: Sequence[FeedEvent], cut: float) -> list[FeedEvent]:
    """Same events up to ``cut``; afterwards spots are doubled, prints dropped, winners flipped."""
    out: list[FeedEvent] = []
    for e in events:
        if e.ts <= cut:
            out.append(e)
        elif isinstance(e, MarketSnapshot):
            spot = None if e.spot is None else 2.0 * e.spot
            out.append(dataclasses.replace(e, spot=spot, trades=()))
        else:
            out.append(MarketResolved(e.ts, e.market_id, e.winner.opposite))
    return out


class _PeekingEngine(MarketMakerEngine):
    """Cheats: reads the LAST event of its stream (the final winner) and, if UP won, never bids
    DOWN (else never UP). A negative control for the causality check below."""

    def __init__(self, cfg: BotConfig, events: Sequence[FeedEvent]) -> None:
        super().__init__(cfg)
        last = events[-1]
        assert isinstance(last, MarketResolved)
        self._skip = Outcome.DOWN if last.winner is Outcome.UP else Outcome.UP

    def on_snapshot(self, snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]:
        return [
            a
            for a in super().on_snapshot(snap, open_orders)
            if not (isinstance(a, PlaceOrder) and a.request.token is self._skip)
        ]


def _run(
    cfg: BotConfig,
    events: Sequence[FeedEvent],
    make_engine: Callable[[BotConfig, Sequence[FeedEvent]], MarketMakerEngine] | None = None,
) -> BacktestResult:
    runner = BacktestRunner(cfg, check_invariants=True, allow_unresolved=True)
    if make_engine is not None:
        runner.engine = make_engine(cfg, events)
    for event in events:
        runner.feed(event)
    return runner.finish()


@pytest.mark.parametrize("latency", [0, 1])
def test_changing_the_future_never_changes_the_past(latency: int) -> None:
    """DESIGN 7.6 end to end: decisions and fills up to ``cut`` depend only on events <= cut."""
    cfg = small_cfg(latency=latency)
    events = list(SyntheticFeed(cfg))
    base = _run(cfg, events)
    cut = events[0].ts + 0.55 * (events[-1].ts - events[0].ts)
    assert _fills(_run(cfg, _scrambled_future(events, cut)), cut) == _fills(base, cut)
    prefix = [e for e in events if e.ts <= cut]  # a truncated recording gives the same past
    assert _fills(_run(cfg, prefix), cut) == _fills(base, cut)
    assert len(_fills(base, cut)) > 10  # there is a past worth comparing


def test_the_causality_check_detects_an_engine_that_peeks_at_the_future() -> None:
    cfg = small_cfg(latency=0)
    events = list(SyntheticFeed(cfg))
    cut = events[0].ts + 0.75 * (events[-1].ts - events[0].ts)
    scrambled = _scrambled_future(events, cut)

    def peeking(c: BotConfig, evs: Sequence[FeedEvent]) -> MarketMakerEngine:
        return _PeekingEngine(c, evs)

    honest = [_fills(_run(cfg, e), cut) for e in (events, scrambled)]
    cheat = [_fills(_run(cfg, e, peeking), cut) for e in (events, scrambled)]
    assert honest[0] == honest[1] and len(honest[0]) > 10
    assert cheat[0] != cheat[1]


# --------------------------------------------------------------------------- exchange guards


def _book_snap(ts: float = 1.0) -> MarketSnapshot:
    bids, asks = ((0.46, 100.0),), ((0.52, 100.0),)
    return MarketSnapshot(
        ts=ts,
        market=MarketSpec("m1", "BTC", 0.0, 900.0, tick_size=0.01, min_order_size=5.0),
        up_book=BookSnapshot(
            UP, tuple(Level(p, s) for p, s in bids), tuple(Level(p, s) for p, s in asks)
        ),
        down_book=BookSnapshot(
            DOWN,
            tuple(Level(round(1.0 - p, 6), s) for p, s in asks),
            tuple(Level(round(1.0 - p, 6), s) for p, s in bids),
        ),
    )


def _exchange_with_orders() -> PaperExchange:
    """Latency-0 exchange holding 50 UP (bought by IOC) with a resting UP bid and UP ask."""
    cfg = BotConfig(
        fees=FeeConfig(taker_fee_rate=0.0),
        exchange=PaperExchangeConfig(initial_cash=1_000.0, latency_ticks=0),
    )
    ex = PaperExchange(cfg)
    ex.process(_book_snap())

    def order(side: Side, price: float, size: float, tif: TimeInForce) -> None:
        assert ex.submit(OrderRequest(f"c{price}{side}", "m1", UP, side, price, size, tif), 1.0).ok

    order(Side.BUY, 0.52, 50.0, TimeInForce.IOC)  # takes the ask: position 50 UP, cash 974
    order(Side.BUY, 0.45, 100.0, TimeInForce.POST_ONLY)  # resting bid reserves 45
    order(Side.SELL, 0.60, 20.0, TimeInForce.POST_ONLY)  # resting ask reserves 20 shares
    ex.drain_fills()
    ex._check_invariants()  # a clean state passes
    return ex


def _corrupt_cash(ex: PaperExchange) -> None:
    ex._cash = -1.0


def _corrupt_commitments(ex: PaperExchange) -> None:
    ex._cash = 10.0  # the resting bid needs 45


def _corrupt_empty_order(ex: PaperExchange) -> None:
    next(iter(ex._orders.values())).order.remaining = 0.0


def _corrupt_negative_position(ex: PaperExchange) -> None:
    ex._positions[("m1", DOWN)] = -2.0


def _corrupt_oversold(ex: PaperExchange) -> None:
    ex._positions[("m1", UP)] = 5.0  # 20 shares are offered for sale


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_corrupt_cash, "negative cash"),
        (_corrupt_commitments, "commitments"),
        (_corrupt_empty_order, "no remaining size"),
        (_corrupt_negative_position, "negative position"),
        (_corrupt_oversold, "open sells"),
    ],
)
def test_the_exchange_asserts_its_own_invariants(
    corrupt: Callable[[PaperExchange], None], message: str
) -> None:
    ex = _exchange_with_orders()
    corrupt(ex)
    with pytest.raises(AssertionError, match=message):
        ex._check_invariants()


def test_a_clean_exchange_state_matches_the_hand_computed_reservations() -> None:
    ex = _exchange_with_orders()
    # IOC buy 50 @ 0.52 = 26 (no fees); the resting bid reserves 100 * 0.45 = 45
    assert ex.balance() == pytest.approx(1_000.0 - 26.0)
    assert ex.reserved_cash == pytest.approx(45.0)
    assert ex.position("m1", UP) == pytest.approx(50.0)


# --------------------------------------------------------------------------- determinism


def test_a_full_backtest_does_not_depend_on_hash_randomisation(tmp_path: Path) -> None:
    """The engine, exchange and runner use dicts and sets of strings: two interpreters with
    different PYTHONHASHSEEDs must still produce byte-identical results."""
    src = str(Path(__file__).resolve().parents[1] / "src")
    digests = []
    for hash_seed in ("0", "12345"):
        out = tmp_path / f"run_{hash_seed}"
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": src}
        cmd = [sys.executable, "-m", "abc_trading", "backtest", "--seed", "3", "--windows", "1"]
        cmd += ["--assets", "BTC", "--set", "sim.window_seconds=300", "--out", str(out)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        result = json.loads((out / "result.json").read_text())
        assert result["activity"]["n_fills"] > 0
        digests.append((result["fills_sha256"], (out / "result.json").read_bytes()))
    assert digests[0] == digests[1]
