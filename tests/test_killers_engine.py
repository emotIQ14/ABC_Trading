"""Engine behaviour that mutation testing showed nothing else pinned.

A mutation run over the strategy (one-line changes: ``and`` -> ``or``, ``flatten=True`` ->
``False``, ``-`` -> ``+``, a dropped phase check) found changes of trading behaviour that the whole
suite let through, because every other engine test either starts with no order of ours open or
never asks for a merge, a completion or a sale outside ACCUMULATE. This file is the decision table
the per-module tests lacked: what the engine does in each phase for each kind of inventory, plus
the real-run floors a degraded rebalancing leg would fall under.

Each row states the spec clause it pins. The numbers are mechanics only (they follow from the
default config and the scripted books), nothing here is tuned for or says anything about profit.

Market "m1": BTC, window 1000..1900 (tick 0.01, min order size 5). The default timing gives
WARMUP < 1015 <= ACCUMULATE < 1810 <= WIND_DOWN < 1875 <= FLATTEN < 1900 <= DONE. Both books are
0.49 / 0.51, the scripted fair value is 0.5 and all inventory is bought at 0.47, so completing a
pair by taker order costs 0.51 + fee 0.00796 and locks 1 - 0.47 - 0.51796 = 0.012 > 0.005.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

import pytest

from abc_trading.backtest.runner import BacktestRunner
from abc_trading.config import BotConfig, ModelConfig, RiskConfig
from abc_trading.model.fair_value import FairValue
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.types import (
    Action,
    BookSnapshot,
    CancelOrder,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    MergePairs,
    OpenOrder,
    Outcome,
    PlaceOrder,
    Side,
    TimeInForce,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
BUY, SELL = Side.BUY, Side.SELL
MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)
WARMUP_TS, ACC_TS, WD_TS, FL_TS, DONE_TS = 1010.0, 1030.0, 1820.0, 1880.0, 1900.0


class _Model:
    """Scripted fair value (valid, p_up 0.5) in place of the real model."""

    def observe(self, asset: str, ts: float, spot: float) -> None:
        pass

    def estimate(self, snap: MarketSnapshot) -> FairValue:
        return FairValue(0.5, 0.5, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, True)


def _snap(ts: float, market: MarketSpec = MKT) -> MarketSnapshot:
    def book(token: Outcome) -> BookSnapshot:
        return BookSnapshot(token, (Level(0.49, 400.0),), (Level(0.51, 400.0),))

    return MarketSnapshot(
        ts, market, book(UP), book(DOWN), spot=100.0, spot_ts=ts, ref_price=100.0
    )  # fmt: skip


def _engine(up: float = 0.0, down: float = 0.0) -> MarketMakerEngine:
    engine = MarketMakerEngine(BotConfig(model=ModelConfig(shrink_to_market=0.0)))
    engine.model = _Model()  # type: ignore[assignment]
    for token, qty in ((UP, up), (DOWN, down)):
        if qty > 0:
            engine.on_fill(
                Fill(f"f{token.value}", "o", "m1", token, BUY, 0.47, qty, 0.0, True, 1.0)
            )
    return engine


def _summary(
    actions: Sequence[Action],
) -> tuple[list[float], list[tuple[str, str, float]], set[str]]:
    """(merge sizes, IOC orders as (token, side, size), tokens that got a passive quote)."""
    merges = [a.size for a in actions if isinstance(a, MergePairs)]
    orders = [a.request for a in actions if isinstance(a, PlaceOrder)]
    iocs = [(r.token.value, r.side.value, r.size) for r in orders if r.tif is TimeInForce.IOC]
    quoted = {r.token.value for r in orders if r.tif is TimeInForce.POST_ONLY}
    return merges, iocs, quoted


NONE: tuple[list[float], list[tuple[str, str, float]], set[str]] = ([], [], set())

# (phase / ts, UP held, DOWN held) -> (merges, taker orders, tokens quoted passively)
DECISIONS = [
    # WARMUP and DONE: nothing is sent whatever is held (DESIGN 2.6, 7.9).
    pytest.param(WARMUP_TS, 150, 0, NONE, id="warmup-unpaired-150"),
    pytest.param(DONE_TS, 150, 0, NONE, id="done-unpaired-150"),
    # ACCUMULATE: merge at merge_min_pairs (2.3), complete by taker order at the trigger (2.5).
    pytest.param(ACC_TS, 30, 30, ([], [], {"UP", "DOWN"}), id="accumulate-30-pairs-below-min"),
    pytest.param(ACC_TS, 60, 60, ([60.0], [], {"UP", "DOWN"}), id="accumulate-60-pairs"),
    pytest.param(ACC_TS, 80, 0, ([], [], {"UP", "DOWN"}), id="accumulate-80-below-trigger"),
    pytest.param(ACC_TS, 150, 0, ([], [("DOWN", "BUY", 150.0)], {"UP"}), id="accumulate-150-up"),
    # WIND_DOWN: only the token that completes pairs is quoted, but merging and the taker
    # completion are not restricted by phase (2.3 "whenever paired_qty >= merge_min_pairs", 2.5).
    pytest.param(WD_TS, 30, 30, NONE, id="wind-down-30-pairs-below-min"),
    pytest.param(WD_TS, 60, 60, ([60.0], [], set()), id="wind-down-60-pairs-merged"),
    pytest.param(WD_TS, 150, 0, ([], [("DOWN", "BUY", 150.0)], set()), id="wind-down-150-up"),
    pytest.param(WD_TS, 0, 150, ([], [("UP", "BUY", 150.0)], set()), id="wind-down-150-down"),
    pytest.param(WD_TS, 80, 0, ([], [], {"DOWN"}), id="wind-down-80-below-trigger"),
    # FLATTEN: merge "any pairs" (2.3), then sell what the model does not back (2.6); no quotes.
    pytest.param(FL_TS, 30, 30, ([30.0], [], set()), id="flatten-30-pairs-below-min"),
    pytest.param(FL_TS, 60, 60, ([60.0], [], set()), id="flatten-60-pairs"),
    pytest.param(
        FL_TS, 100, 60, ([60.0], [("UP", "SELL", 40.0)], set()), id="flatten-60-pairs-40-up"
    ),
    pytest.param(FL_TS, 30, 0, ([], [("UP", "SELL", 30.0)], set()), id="flatten-30-up"),
]


@pytest.mark.parametrize(("ts", "up", "down", "expected"), DECISIONS)
def test_phase_by_inventory_decision_table(
    ts: float,
    up: float,
    down: float,
    expected: tuple[list[float], list[tuple[str, str, float]], set[str]],
) -> None:
    assert _summary(_engine(up, down).on_snapshot(_snap(ts), [])) == expected


def test_completion_is_planned_while_own_passive_quotes_rest_in_every_trading_phase() -> None:
    # A resting POST_ONLY bid of ours never blocks the taker completion; only an IOC BUY still in
    # flight does. (The quotes of a live engine are always resting, so a completion that waited for
    # them to clear would almost never be sent.)
    resting = OpenOrder("ox", "cx", "m1", UP, BUY, 0.47, 100.0, 100.0, TimeInForce.POST_ONLY, 1.0)
    for ts in (ACC_TS, WD_TS):
        _, iocs, _ = _summary(_engine(150, 0).on_snapshot(_snap(ts), [resting]))
        assert iocs == [("DOWN", "BUY", 150.0)], ts


def test_wind_down_cancels_what_it_no_longer_wants_while_it_completes() -> None:
    # The completion is planned in WIND_DOWN together with the cancel of the quotes that stop being
    # allowed there (UP, the heavy token), so the one snapshot both exits and completes.
    rest = [
        OpenOrder(
            f"o{i}",
            f"c{i}",
            "m1",
            UP,
            BUY,
            0.49 - 0.01 * i,
            100.0,
            100.0,
            TimeInForce.POST_ONLY,
            1.0,
        )
        for i in range(3)
    ]
    actions = _engine(150, 0).on_snapshot(_snap(WD_TS), rest)
    assert {a.order_id for a in actions if isinstance(a, CancelOrder)} == {"o0", "o1", "o2"}
    assert _summary(actions)[1] == [("DOWN", "BUY", 150.0)]


# --------------------------------------------------------------------------- real runs


def _run(seed: int, *, windows: int = 4, cfg: BotConfig | None = None) -> BacktestRunner:
    base = cfg or BotConfig()
    cfg = replace(
        base,
        sim=replace(base.sim, seed=seed, n_windows=windows, assets=("BTC",), window_seconds=300),
    )
    runner = BacktestRunner(cfg, check_invariants=True)
    for event in SyntheticFeed(cfg):
        runner.feed(event)
    runner.finish()
    return runner


def test_real_runs_rebalance_by_taker_completion_at_the_observed_rate() -> None:
    """The rebalancing leg of the strategy, end to end, over eight short seeds.

    The default config sends 145 taker completions in these runs (9 to 28 a seed). A guard that
    let any resting bid of ours block the completion cut that to 44. The floor of 70 is below half
    of today's count, so it leaves room for changes of model or feed, yet well above the broken one.
    """
    completions = [_run(seed).engine.stats["taker_completions"] for seed in range(1, 9)]
    assert sum(completions) >= 70, completions


def test_real_runs_never_send_more_orders_in_a_second_than_the_rate_limit_allows() -> None:
    """DESIGN 2.7: placements (quotes, completions and FLATTEN sells alike) obey
    ``max_orders_per_second`` across all markets; cancels never wait but use up slots.

    Two assets share the limiter and the snapshots are 1 s apart, so every placement of a
    timestamp falls in the same one-second window: their total may not exceed the limit.
    """
    limit = 2
    cfg = BotConfig(risk=RiskConfig(max_orders_per_second=float(limit)))
    cfg = replace(cfg, sim=replace(cfg.sim, seed=3, n_windows=3, assets=("BTC", "ETH")))
    cfg = replace(cfg, sim=replace(cfg.sim, window_seconds=300))
    runner = BacktestRunner(cfg, check_invariants=True)
    engine = runner.engine
    original = engine.on_snapshot
    placed: Counter[float] = Counter()
    kinds: Counter[str] = Counter()

    def audited(snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]:
        actions = original(snap, open_orders)
        for a in actions:
            if isinstance(a, PlaceOrder):
                placed[snap.ts] += 1
                kinds[f"{a.request.tif.value}-{a.request.side.value}"] += 1
        return actions

    engine.on_snapshot = audited  # type: ignore[method-assign]
    for event in SyntheticFeed(cfg):
        runner.feed(event)
    runner.finish()
    assert max(placed.values()) <= limit, max(placed.items(), key=lambda kv: kv[1])
    assert engine.stats["orders_rate_limited"] > 0  # the limiter really bound
    assert kinds["IOC-BUY"] > 0 and kinds["POST_ONLY-BUY"] > 100, kinds
