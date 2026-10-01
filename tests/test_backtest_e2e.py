"""End-to-end tests: SyntheticFeed -> real engine -> real PaperExchange -> BacktestResult.

Short sims (BTC only, 300 s windows) keep every run around 0.2 s. These tests check MECHANICS
(invariants, determinism, accounting identities recomputed independently from the raw fills); they
say nothing about profitability, and no number is tuned.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from abc_trading.backtest.runner import BacktestResult, InvariantError, run_backtest
from abc_trading.config import BotConfig
from abc_trading.data.events import read_jsonl, write_jsonl
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.types import FeedEvent, MarketResolved, Outcome, Side

LABEL = "synthetic:e2e"


def small_cfg(**sim: object) -> BotConfig:
    """Two 300 s BTC windows (602 events); ``sim`` overrides SimConfig fields."""
    cfg = BotConfig()
    fields: dict[str, object] = {"n_windows": 2, "assets": ("BTC",), "window_seconds": 300}
    fields.update(sim)
    return replace(cfg, sim=replace(cfg.sim, **fields))  # type: ignore[arg-type]


def run(cfg: BotConfig, **kwargs: object) -> BacktestResult:
    return run_backtest(cfg, SyntheticFeed(cfg), source_label=LABEL, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def base() -> tuple[BotConfig, BacktestResult]:
    cfg = small_cfg()
    return cfg, run(cfg)


def oracle_final_cash(result: BacktestResult) -> float:
    """Final cash recomputed from the raw fills only (independent of the Portfolio code):
    buys pay price*size + fee, sells receive price*size - fee, each merged pair returns $1 and
    every market pays $1 per winning share still held at resolution."""
    cash = result.initial_cash
    held: dict[tuple[str, Outcome], float] = defaultdict(float)
    for f in result.fills:
        if f.side is Side.BUY:
            cash -= f.price * f.size + f.fee
            held[(f.market_id, f.token)] += f.size
        else:
            cash += f.price * f.size - f.fee
            held[(f.market_id, f.token)] -= f.size
    for m in result.markets:
        cash += m.merged_pairs
        held[(m.market_id, Outcome.UP)] -= m.merged_pairs
        held[(m.market_id, Outcome.DOWN)] -= m.merged_pairs
    for m in result.markets:
        assert m.winner is not None
        cash += max(held[(m.market_id, Outcome(m.winner))], 0.0)
    return cash


def assert_accounting_consistent(result: BacktestResult) -> None:
    assert result.unresolved_markets == []
    # the headline number equals the independently recomputed cash, and every PnL view agrees
    assert result.final_equity == pytest.approx(oracle_final_cash(result), abs=1e-6)
    assert result.total_pnl == pytest.approx(result.final_equity - result.initial_cash, abs=1e-9)
    assert sum(m.pnl for m in result.markets) == pytest.approx(result.total_pnl, abs=1e-6)
    p = result.pnl
    parts = p.pnl_sell + p.pnl_merge + p.pnl_settle_pair + p.pnl_settle_directional
    assert parts == pytest.approx(result.total_pnl, abs=1e-6)
    for m in result.markets:
        assert m.sell_pnl + m.merge_pnl + m.settle_pair_pnl + m.settle_directional_pnl == (
            pytest.approx(m.pnl, abs=1e-9)
        )


# --------------------------------------------------------------------------- invariants


def test_invariants_hold_on_a_synthetic_run(base: tuple[BotConfig, BacktestResult]) -> None:
    _, r = base
    assert r.invariants_checked is True
    assert (r.n_events, r.n_snapshots, r.n_resolutions) == (602, 600, 2)
    assert r.engine_stats["snapshots"] == 600 and r.engine_stats["snapshots_ignored"] == 0
    assert r.exchange_stats["settled"] == 2
    assert [m.resolved for m in r.markets] == [True, True]
    assert_accounting_consistent(r)


def test_result_is_consistent_with_the_raw_fills_and_counters(
    base: tuple[BotConfig, BacktestResult],
) -> None:
    _, r = base
    fills = r.fills
    assert fills and len(fills) == r.activity.n_fills
    assert [f.ts for f in fills] == sorted(f.ts for f in fills)
    assert all(f.size > 0 and 0.0 <= f.price <= 1.0 for f in fills)
    assert r.activity.n_buy_fills + r.activity.n_sell_fills == len(fills)
    assert r.activity.n_maker_fills == sum(f.is_maker for f in fills)
    assert r.activity.avg_fill_notional == pytest.approx(
        sum(f.notional for f in fills) / len(fills)
    )
    assert r.activity.volume_maker == pytest.approx(r.exchange_stats["volume_maker"])
    assert r.activity.volume_taker == pytest.approx(r.exchange_stats["volume_taker"])
    assert r.exchange_stats["filled_maker"] + r.exchange_stats["filled_taker"] == len(fills)
    assert r.engine_stats["fills"] == len(fills)
    assert sum(m.n_fills for m in r.markets) == len(fills)
    assert sum(m.volume_maker + m.volume_taker for m in r.markets) == pytest.approx(
        sum(f.notional for f in fills)
    )
    assert sum(m.fees_paid for m in r.markets) == pytest.approx(sum(f.fee for f in fills))
    # every order the runner submitted was counted by the exchange; rejections agree by reason
    st, ex = r.runner_stats, r.exchange_stats
    assert st["orders_submitted"] == ex["submitted"]
    rejected = {k.removeprefix("rejected_"): v for k, v in ex.items() if k.startswith("rejected_")}
    assert st["order_rejections"] == rejected
    assert ex["accepted"] == ex["submitted"] - sum(rejected.values())
    assert st["merges_executed"] == ex["merges"]
    assert st["cancel_misses"] == 0  # the engine only cancels orders the exchange reported


def test_per_market_risk_limits_respected(base: tuple[BotConfig, BacktestResult]) -> None:
    cfg, r = base
    for m in r.markets:
        assert m.max_capital_at_risk <= cfg.risk.max_capital_per_market_usd + 1e-6
        assert m.max_inventory_shares <= 2 * cfg.pair.max_inventory_per_side_shares
        assert 0.0 <= m.max_net_shares <= m.max_inventory_shares + 1e-9
        assert m.paired_at_close >= 0.0 and m.unpaired_at_close >= 0.0


def test_calibration_matches_an_independent_recomputation(
    base: tuple[BotConfig, BacktestResult],
) -> None:
    _, r = base
    c = r.calibration
    assert c.n_resolved == 2
    assert c.n_scored + c.n_invalid + c.n_no_accumulate == c.n_resolved
    assert c.n_scored == 2  # both markets reach ACCUMULATE with a valid model on this seed
    outcomes = [1.0 if m.winner == "UP" else 0.0 for m in r.markets]
    forecasts = [m.p_up_last_accumulate for m in r.markets]
    assert all(p is not None and 0.0 <= p <= 1.0 for p in forecasts)
    squared = [(p - y) ** 2 for p, y in zip(forecasts, outcomes, strict=True) if p is not None]
    expected = sum(squared) / len(squared)
    assert c.brier_model == pytest.approx(expected)
    assert c.brier_model_raw is not None and 0.0 <= c.brier_model_raw <= 1.0
    assert c.brier_market is not None and 0.0 <= c.brier_market <= 1.0


def test_equity_curve_schedule_and_endpoints(base: tuple[BotConfig, BacktestResult]) -> None:
    cfg, r = base
    n = r.n_events
    assert [p.event_index for p in r.equity_curve] == [1, *range(60, n, 60), n]
    first, last = r.equity_curve[0], r.equity_curve[-1]
    assert first.equity == cfg.exchange.initial_cash == first.cash  # WARMUP: nothing traded yet
    assert last.equity == pytest.approx(r.final_equity) and last.cash == pytest.approx(
        r.final_equity
    )
    assert [p.ts for p in r.equity_curve] == sorted(p.ts for p in r.equity_curve)
    assert r.pnl.peak_equity == max(p.equity for p in r.equity_curve)
    assert r.pnl.max_drawdown_usd >= 0.0
    # sampling every event gives the same endpoints and a point per event
    full = run(base[0], equity_sample_every=1)
    assert [p.event_index for p in full.equity_curve] == list(range(1, n + 1))
    assert full.final_equity == r.final_equity  # sampling never changes the trading result
    assert full.fills_sha256 == r.fills_sha256
    assert full.pnl.max_drawdown_usd >= r.pnl.max_drawdown_usd - 1e-9  # finer curve sees >= dips


# --------------------------------------------------------------------------- determinism


def test_same_seed_gives_identical_results(base: tuple[BotConfig, BacktestResult]) -> None:
    cfg, r = base
    again = run(cfg)
    assert again.to_dict() == r.to_dict()
    assert json.dumps(again.to_dict()) == json.dumps(r.to_dict())
    assert again.fills_sha256 == r.fills_sha256


def test_sim_to_jsonl_to_replay_gives_identical_results(
    base: tuple[BotConfig, BacktestResult], tmp_path: Path
) -> None:
    cfg, r = base
    path = tmp_path / "events.jsonl"
    count = write_jsonl(path, SyntheticFeed(cfg))
    assert count == r.n_events
    replayed = run_backtest(cfg, read_jsonl(path), source_label=LABEL)
    assert replayed.to_dict() == r.to_dict()
    assert replayed.fills == r.fills


def test_different_seeds_give_different_results(base: tuple[BotConfig, BacktestResult]) -> None:
    _, r = base
    other = run(small_cfg(seed=2))
    assert other.fills_sha256 != r.fills_sha256
    assert other.final_equity != r.final_equity
    assert_accounting_consistent(other)


# --------------------------------------------------------------------------- sanity / variants


def test_the_engine_actually_trades_without_adverse_selection() -> None:
    # no informed sweeps and no market lag: the quotes are not picked off, so the bot must trade
    cfg = small_cfg(informed_flow=0.0, market_lag_seconds=0.0)
    r = run(cfg)
    assert r.activity.n_fills > 0
    assert r.activity.n_maker_fills > 0
    assert r.pairs.shares_bought > 0
    assert r.activity.avg_fill_notional is not None and r.activity.avg_fill_notional > 0
    assert_accounting_consistent(r)
    assert r.engine_stats["quotes_placed"] > 0


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("latency0", lambda c: replace(c, exchange=replace(c.exchange, latency_ticks=0))),
        ("latency2", lambda c: replace(c, exchange=replace(c.exchange, latency_ticks=2))),
        (
            "half-queue-slippage",
            lambda c: replace(
                c,
                exchange=replace(
                    c.exchange, queue_ahead_fraction=0.5, trade_fill_fraction=0.5,
                    taker_slippage_ticks=1,
                ),
            ),
        ),
        ("maker-fee", lambda c: replace(c, fees=replace(c.fees, maker_fee_rate=0.01))),
        ("maker-rebate", lambda c: replace(c, fees=replace(c.fees, maker_rebate_rate=0.005))),
        (
            "no-directional",
            lambda c: replace(c, directional=replace(c.directional, enabled=False)),
        ),
        (
            "equity-sizing",
            lambda c: replace(c, sizing=replace(c.sizing, clip_equity_fraction=0.02)),
        ),
    ],
)  # fmt: skip
def test_variants_hold_invariants_and_the_accounting_identity(name: str, mutate: object) -> None:
    cfg = mutate(small_cfg(seed=3, informed_flow=0.3, n_windows=1))  # type: ignore[operator]
    r = run(cfg)
    assert r.n_resolutions == 1, name
    assert_accounting_consistent(r)


def test_two_assets_run_in_global_time_order() -> None:
    cfg = small_cfg(assets=("BTC", "ETH"), n_windows=1)
    r = run(cfg)
    assert sorted(m.asset for m in r.markets) == ["BTC", "ETH"]
    assert r.n_events == 2 * 300 + 2
    assert_accounting_consistent(r)


def test_randomised_configs_never_violate_an_invariant() -> None:
    rng = random.Random(20240607)
    for case in range(8):
        base_cfg = small_cfg(
            seed=rng.randint(1, 10_000),
            n_windows=1,
            informed_flow=rng.choice([0.0, 0.5, 1.0]),
            market_lag_seconds=rng.choice([0.0, 2.0, 5.0]),
            uninformed_trades_per_sec=rng.choice([0.1, 0.3, 1.0]),
        )
        cfg = replace(
            base_cfg,
            exchange=replace(
                base_cfg.exchange,
                latency_ticks=rng.randint(0, 2),
                queue_ahead_fraction=rng.choice([0.0, 0.5, 1.0, 2.0]),
                trade_fill_fraction=rng.choice([0.25, 0.5, 1.0]),
                taker_slippage_ticks=rng.randint(0, 2),
            ),
            fees=replace(
                base_cfg.fees,
                maker_fee_rate=rng.choice([0.0, 0.002]),
                maker_rebate_rate=rng.choice([0.0, 0.001]),
                taker_fee_rate=rng.choice([0.05, 0.25]),
            ),
            sizing=replace(
                base_cfg.sizing,
                clip_shares=rng.choice([20.0, 100.0, 300.0]),
                ladder_levels=rng.choice([1, 3, 5]),
                requote_tolerance_ticks=rng.randint(0, 2),
            ),
            pair=replace(base_cfg.pair, merge_min_pairs=rng.choice([5.0, 50.0])),
            directional=replace(base_cfg.directional, enabled=rng.random() < 0.7),
        )
        r = run(cfg)
        assert r.n_resolutions == 1, case
        assert_accounting_consistent(r)


def test_kill_switch_is_reported_and_the_run_still_ends_clean(
    base: tuple[BotConfig, BacktestResult],
) -> None:
    assert base[1].kill_switch is False and base[1].engine_stats["risk_trips_kill_switch"] == 0
    cfg = small_cfg()
    cfg = replace(cfg, risk=replace(cfg.risk, max_daily_loss_usd=0.01))
    r = run(cfg)
    assert r.kill_switch is True
    assert r.engine_stats["risk_trips_kill_switch"] == 1
    assert r.engine_stats["risk_blocked_snapshots"] > 0
    assert_accounting_consistent(r)  # merges/settlement still reconcile after the latch


# --------------------------------------------------------------------------- truncated stream


def test_truncated_recording_is_an_invariant_4_violation_unless_allowed() -> None:
    cfg = small_cfg(n_windows=1)
    events: list[FeedEvent] = [e for e in SyntheticFeed(cfg) if not isinstance(e, MarketResolved)]
    with pytest.raises(InvariantError, match=r"invariant 4.*1 market\(s\) never resolved"):
        run_backtest(cfg, events, source_label=LABEL)
    r = run_backtest(cfg, events, source_label=LABEL, allow_unresolved=True)
    assert r.n_resolutions == 0 and len(r.unresolved_markets) == 1
    assert [m.resolved for m in r.markets] == [False]
    assert all(m.winner is None for m in r.markets)
    assert r.pnl.n_markets_resolved == 0 and r.calibration.n_resolved == 0
    # nothing is settled, so the equity is cash plus positions marked at the last snapshot mids
    assert r.final_equity != r.initial_cash or r.activity.n_fills == 0
    json.dumps(r.to_dict(), allow_nan=False)
    # the same truncation without invariant checks is accepted silently
    quiet = run_backtest(cfg, events, source_label=LABEL, check_invariants=False)
    assert quiet.unresolved_markets == r.unresolved_markets and not quiet.invariants_checked


def test_partially_resolved_stream_reports_the_open_market() -> None:
    cfg = small_cfg()
    events = list(SyntheticFeed(cfg))
    assert isinstance(events[-1], MarketResolved)
    r = run_backtest(cfg, events[:-1], source_label=LABEL, allow_unresolved=True)
    assert r.unresolved_markets == [events[-1].market_id]
    assert [m.resolved for m in r.markets] == [True, False]
    assert r.pnl.n_markets_resolved == 1
