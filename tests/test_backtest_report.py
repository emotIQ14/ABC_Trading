"""Tests for abc_trading.backtest.report: format_report and write_run_dir.

The text tests use a hand-built ``BacktestResult`` with round numbers, so every printed value is
known in advance; the run-directory tests also round-trip a real (short) synthetic run.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from abc_trading.backtest.report import (
    GENERIC_NOTE,
    MAX_MARKET_ROWS,
    PAPER_BANNER,
    PAPER_UNVERIFIED_NOTE,
    RUN_FILES,
    SYNTHETIC_BANNER,
    format_report,
    headline_banner,
    is_synthetic,
    write_run_dir,
)
from abc_trading.backtest.runner import (
    ActivityStats,
    BacktestResult,
    CalibrationStats,
    EquityPoint,
    MarketRecord,
    PairStats,
    PnLStats,
    fills_digest,
    run_backtest,
)
from abc_trading.config import BotConfig, config_from_dict, config_to_dict
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.types import FeedEvent, Fill, MarketResolved, Outcome, Side

UP, DOWN = Outcome.UP, Outcome.DOWN


def record(**overrides: object) -> MarketRecord:
    base = MarketRecord(
        market_id="BTC-1700000100-900s", asset="BTC", resolved=True, winner="UP", sell_pnl=0.0,
        merge_pnl=2.0, settle_pair_pnl=0.0, settle_directional_pnl=11.0, fees_paid=0.0, pnl=13.0,
        n_fills=3, buy_shares=220.0, sell_shares=0.0, volume_maker=107.0, volume_taker=0.0,
        max_inventory_shares=200.0, max_net_shares=100.0, max_capital_at_risk=98.0,
        merged_pairs=100.0, paired_at_close=0.0, avg_pair_cost_at_close=None,
        avg_merge_pair_cost=0.98, unpaired_token="UP", unpaired_at_close=20.0,
        ended_unhedged=True, pair_completion_rate=200.0 / 220.0, p_up_last_accumulate=0.42,
    )  # fmt: skip
    return replace(base, **overrides)  # type: ignore[arg-type]


def make_fills() -> list[Fill]:
    return [
        Fill("f1", "o1", "m1", UP, Side.BUY, 0.49, 100.0, 0.0, True, 1102.0),
        Fill("f2", "o2", "m1", DOWN, Side.BUY, 0.51, 50.0, 0.0125, False, 1102.0),
        Fill("f3", "o3", "m1", UP, Side.SELL, 0.3, 10.0, -0.001, True, 1105.5),
    ]


def make_result(label: str = "synthetic:unit", **overrides: object) -> BacktestResult:
    result = BacktestResult(
        source_label=label,
        invariants_checked=True,
        equity_sample_every=60,
        n_events=1802,
        n_snapshots=1800,
        n_resolutions=2,
        first_ts=1_700_000_100.0,
        last_ts=1_700_001_900.0,
        initial_cash=10_000.0,
        final_equity=10_013.0,
        total_pnl=13.0,
        pnl_pct_of_initial_cash=0.13,
        kill_switch=False,
        activity=ActivityStats(3, 3, 0, 3, 0, 107.0, 0.0, 107.0 / 3.0, 49.0),
        pairs=PairStats(1, 1, 220.0, 100.0, 0.0, 100.0, 200.0 / 220.0, 0.98, None, 1, 1.0, 1.0),
        pnl=PnLStats(
            n_markets_resolved=1,
            n_markets_resolved_traded=1,
            mean_market_pnl=13.0,
            std_market_pnl=0.0,
            sharpe_like=0.0,
            best_market_pnl=13.0,
            worst_market_pnl=13.0,
            win_rate_traded=1.0,
            pnl_sell=0.0,
            pnl_merge=2.0,
            pnl_settle_pair=0.0,
            pnl_settle_directional=11.0,
            pnl_unrealised=0.0,
            fees_paid=0.0,
            peak_equity=10_013.0,
            max_drawdown_usd=0.0,
            max_drawdown_frac=0.0,
        ),  # fmt: skip
        calibration=CalibrationStats(1, 1, 0, 0, 0.1, 0.12, 0.2, 0.5),
        unresolved_markets=[],
        markets=[record()],
        engine_stats={"fills": 3, "quotes_placed": 1234},
        exchange_stats={"submitted": 4, "volume_maker": 107.5},
        runner_stats={"orders_submitted": 4, "order_rejections": {"bad_tick": 2, "min_size": 1}},
        equity_curve=[EquityPoint(1, 1_700_000_100.0, 10_000.0, 10_000.0)],
        fills_sha256=fills_digest(make_fills()),
        fills=make_fills(),
    )
    return replace(result, **overrides)  # type: ignore[arg-type]


def section(text: str, title: str) -> dict[str, str]:
    """Rows ``label -> value`` of one titled section, asserting the column alignment."""
    lines = text.splitlines()
    start = lines.index(title)
    assert set(lines[start + 1]) == {"-"} and len(lines[start + 1]) == len(title)
    rows: dict[str, str] = {}
    starts: set[int] = set()
    for line in lines[start + 2 :]:
        if not line.strip():
            break
        m = re.match(r"  (\S.*?) {2,}(\S.*)$", line)
        assert m, f"unparsable row: {line!r}"
        rows[m.group(1)] = m.group(2)
        starts.add(m.start(2))
    assert len(starts) == 1, f"values of section {title!r} are not aligned: {sorted(starts)}"
    return rows


# --------------------------------------------------------------------------- headline


def test_synthetic_label_gets_the_exact_honesty_banner() -> None:
    text = format_report(make_result("synthetic:unit"))
    lines = text.splitlines()
    assert lines[0] == "ABC_Trading backtest report"
    assert SYNTHETIC_BANNER == (
        "SYNTHETIC DATA - mechanics validation only, not evidence of profitability"
    )
    assert lines[2] == SYNTHETIC_BANNER  # headline block, before any number
    assert text.count("SYNTHETIC DATA") == 1


@pytest.mark.parametrize(
    "label",
    ["synthetic", "Synthetic seed=1", "synthetic-replay: events.jsonl", "my SYNTHETIC run"],
)
def test_is_synthetic_is_a_case_insensitive_substring_test(label: str) -> None:
    assert is_synthetic(label)
    assert headline_banner(label) == SYNTHETIC_BANNER


def test_non_synthetic_labels_never_claim_to_be_synthetic_but_still_carry_a_caveat() -> None:
    for label in ("replay: events.jsonl", "unknown", ""):
        assert not is_synthetic(label)
        text = format_report(make_result(label))
        assert "SYNTHETIC DATA" not in text
        assert text.splitlines()[2] == GENERIC_NOTE
    assert "not evidence of profitability" in GENERIC_NOTE.lower()
    paper = format_report(make_result("paper-live"))
    assert paper.splitlines()[2] == PAPER_BANNER
    assert "no order was or can be placed" in PAPER_BANNER


# --------------------------------------------------------------------------- content


def test_report_rows_for_a_hand_built_result() -> None:
    text = format_report(make_result())
    run = section(text, "Run")
    assert run["Source"] == "synthetic:unit"
    assert run["Events"] == "1,802 (1,800 snapshots, 2 resolutions)"
    assert run["Time span (unix s)"] == "1,700,000,100 .. 1,700,001,900"
    assert run["Invariants"] == "checked (DESIGN section 7, 1-4)"
    assert run["Unresolved markets"] == "none"

    res = section(text, "Result")
    assert res["Initial cash"] == "10,000.00"
    assert res["Final equity"] == "10,013.00"
    assert res["Total PnL"] == "+13.00  (+0.13%)"
    assert res["Max drawdown"] == "0.00  (0.00% of peak; sampled every 60 events)"
    assert res["Kill switch"] == "not tripped"

    pnl = section(text, "Per-market PnL")
    assert pnl["Markets resolved"] == "1 of 1 (1 traded)"
    assert pnl["Mean / std per market"] == "+13.00 / 0.00"
    assert pnl["Win rate (traded markets)"] == "100.0%"
    assert pnl["PnL from merges"] == "+2.00"
    assert pnl["PnL at settlement, directional"] == "+11.00"
    assert pnl["Unrealised, open inventory at mid"] == "+0.00"
    assert "Statistics cover" not in pnl  # every market is resolved: nothing to qualify
    assert pnl["Fees paid (inside the above)"] == "0.00"

    act = section(text, "Trading activity")
    assert act["Trades (fills)"] == "3 (3 buys, 0 sells)"
    assert act["Maker / taker fills"] == "3 / 0"
    assert act["Volume maker / taker (USDC)"] == "107.00 / 0.00"
    assert act["Average trade notional (USDC)"].startswith("35.67")
    assert "~$53" in act["Average trade notional (USDC)"]
    assert "unverified" in act["Average trade notional (USDC)"]
    assert act["Median trade notional (USDC)"] == "49.00"

    pairs = section(text, "Pairing and inventory")
    assert pairs["Shares bought"] == "220.0"
    assert pairs["Pairs formed"] == "100.0 (merged 100, held at close 0.0)"
    assert pairs["Pair completion rate"] == "90.9%"  # 200 / 220
    assert pairs["Avg all-in cost per merged pair"] == "0.9800  (margin +2.00c per pair)"
    assert pairs["Avg cost per pair held at close"] == "n/a"
    assert pairs["Markets ending unhedged (>=1 share)"] == "1 of 1 (100.0%); of traded: 100.0%"
    assert pairs["Max inventory / max net (shares)"] == "200 / 100"

    cal = section(text, "Model calibration (Brier score, lower is better)")
    assert cal["Markets scored"].startswith("1 of 1")
    assert cal["Engine p_up (model shrunk to market)"] == "0.1000"
    assert cal["Market UP mid (same snapshot)"] == "0.2000"
    assert cal["Skill vs market (positive = better)"] == "+0.500"

    assert section(text, "Engine counters") == {"fills": "3", "quotes_placed": "1,234"}
    assert section(text, "Exchange counters") == {"submitted": "4", "volume_maker": "107.50"}
    assert section(text, "Runner counters") == {
        "orders_submitted": "4",
        "order_rejections": "bad_tick=2, min_size=1",
    }


def test_market_table_and_footnotes() -> None:
    text = format_report(make_result())
    table = text.split("\nMarkets\n-------\n")[1].split("\n\n")[0].splitlines()
    assert table[0].split() == [
        "market", "winner", "pnl", "fills", "max", "inv", "max", "net", "merged", "held",
        "pairs", "pair", "cost", "unpaired",
    ]  # fmt: skip
    assert set(table[1].replace(" ", "")) == {"-"}
    row = table[2].split()
    assert row == [
        "BTC-1700000100-900s", "UP", "+13.00", "3", "200", "100", "100", "0.0", "n/a", "20.0", "*",
    ]  # fmt: skip
    assert "* unpaired inventory of at least 1 share remained at resolution" in text
    # an unresolved market is labelled as such, and a hedged market gets no star
    unresolved = record(resolved=False, winner=None, ended_unhedged=False, unpaired_at_close=0.0)
    text2 = format_report(make_result(markets=[unresolved], unresolved_markets=["m"]))
    assert "unresolved" in text2 and "* unpaired inventory" not in text2
    assert section(text2, "Run")["Unresolved markets"] == "1"


def test_long_market_lists_are_truncated_with_a_pointer_to_the_csv() -> None:
    markets = [record(market_id=f"BTC-{i}") for i in range(MAX_MARKET_ROWS + 5)]
    text = format_report(make_result(markets=markets))
    assert f"BTC-{MAX_MARKET_ROWS - 1} " in text and f"BTC-{MAX_MARKET_ROWS} " not in text
    assert "... 5 more (see markets.csv)" in text
    exactly = format_report(make_result(markets=markets[:MAX_MARKET_ROWS]))
    assert "more (see markets.csv)" not in exactly


def test_report_shape_and_determinism() -> None:
    text = format_report(make_result())
    assert text == format_report(make_result())
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert text.isascii()
    assert all(line == line.rstrip() for line in text.splitlines())
    assert "No live trading exists in this project" in text  # the closing caveats
    assert "uncalibrated" in text


def test_kill_switch_and_disabled_invariants_are_stated_loudly() -> None:
    text = format_report(make_result(kill_switch=True, invariants_checked=False))
    assert section(text, "Result")["Kill switch"] == "TRIPPED"
    assert section(text, "Run")["Invariants"] == "NOT checked"


def test_negative_numbers_and_negative_zero() -> None:
    result = make_result(total_pnl=-137.6, pnl_pct_of_initial_cash=-1.376)
    assert section(format_report(result), "Result")["Total PnL"] == "-137.60  (-1.38%)"
    tiny = make_result(total_pnl=-1e-9, pnl_pct_of_initial_cash=-1e-11)
    assert section(format_report(tiny), "Result")["Total PnL"] == "+0.00  (+0.00%)"  # never -0.00


def test_report_of_an_empty_run_prints_n_a_instead_of_crashing() -> None:
    result = run_backtest(BotConfig(), [], source_label="synthetic:empty")
    text = format_report(result)
    assert section(text, "Trading activity")["Average trade notional (USDC)"] == "n/a"
    assert section(text, "Pairing and inventory")["Pair completion rate"] == "n/a"
    assert section(text, "Run")["Time span (unix s)"] == "n/a .. n/a"
    assert "Markets\n-------" in text


# --------------------------------------------------------------------------- unresolved markets
# A paper run (and `replay --allow-unresolved`) ends with markets still open: those markets count
# in Total PnL (marked at the last mids), in the pairing section and in the markets table, so the
# PnL block must not silently leave them out.


def num(text: str) -> float:
    return float(text.replace(",", ""))


def short_sim_cfg(n_windows: int) -> BotConfig:
    base = BotConfig()
    return replace(base, sim=replace(base.sim, seed=1, n_windows=n_windows))


@pytest.fixture(scope="module")
def truncated() -> BacktestResult:
    """What `paper` sees on Ctrl-C inside the first window: nothing has resolved yet."""
    cfg = short_sim_cfg(1)
    events = [e for e in SyntheticFeed(cfg) if not isinstance(e, MarketResolved)]
    return run_backtest(cfg, events[:1400], source_label="paper-live", allow_unresolved=True)


@pytest.fixture(scope="module")
def partial() -> BacktestResult:
    """Window 1 resolved (BTC and ETH), window 2 cut after 300 events (both still open)."""
    cfg = short_sim_cfg(2)
    events: list[FeedEvent] = list(SyntheticFeed(cfg))
    second_resolution = [i for i, e in enumerate(events) if isinstance(e, MarketResolved)][1]
    return run_backtest(
        cfg, events[: second_resolution + 1 + 300], source_label="paper-live",
        allow_unresolved=True,
    )  # fmt: skip


def decomposition(r: BacktestResult) -> dict[str, float]:
    """The PnL rows of the report as numbers."""
    pnl = section(format_report(r), "Per-market PnL")
    keys = {
        "merge": "PnL from merges",
        "sell": "PnL from sells (flatten)",
        "settle_pair": "PnL at settlement, pairs",
        "settle_dir": "PnL at settlement, directional",
        "unrealised": "Unrealised, open inventory at mid",
        "fees": "Fees paid (inside the above)",
    }
    return {name: num(pnl[label]) for name, label in keys.items()}


def test_pnl_rows_cover_every_market_and_add_up_to_total_pnl_in_a_partial_run(
    partial: BacktestResult,
) -> None:
    r = partial
    assert (len(r.unresolved_markets), r.pairs.n_markets) == (2, 4)
    resolved, still_open = (
        [m for m in r.markets if m.resolved],
        [m for m in r.markets if not m.resolved],
    )
    # the open markets really did merge, sell or pay fees: leaving them out would change the rows
    assert sum(m.merge_pnl for m in still_open) != pytest.approx(0.0, abs=0.01)
    assert sum(m.fees_paid for m in still_open) > 0.0
    rows = decomposition(r)
    assert rows["merge"] == pytest.approx(sum(m.merge_pnl for m in r.markets), abs=0.006)
    assert rows["sell"] == pytest.approx(sum(m.sell_pnl for m in r.markets), abs=0.006)
    assert rows["settle_pair"] == pytest.approx(sum(m.settle_pair_pnl for m in resolved), abs=0.006)
    assert rows["fees"] == pytest.approx(sum(f.fee for f in r.fills), abs=0.006)
    # ... and, with the unrealised row, the five rows reconcile with Total PnL
    parts = rows["merge"] + rows["sell"] + rows["settle_pair"] + rows["settle_dir"]
    assert parts + rows["unrealised"] == pytest.approx(r.total_pnl, abs=0.03)
    assert rows["unrealised"] != pytest.approx(0.0, abs=0.01)
    p = r.pnl
    exact = p.pnl_sell + p.pnl_merge + p.pnl_settle_pair + p.pnl_settle_directional
    assert exact + p.pnl_unrealised == pytest.approx(r.total_pnl, abs=1e-9)


def test_pnl_rows_of_a_run_that_has_not_resolved_anything_are_not_all_zero(
    truncated: BacktestResult,
) -> None:
    r = truncated
    assert r.pnl.n_markets_resolved == 0 and len(r.unresolved_markets) == r.pairs.n_markets == 2
    merged = sum(m.merge_pnl for m in r.markets)
    assert merged > 1.0 and r.pairs.merged_pairs > 0  # the pairing section reports merges ...
    rows = decomposition(r)
    assert rows["merge"] == pytest.approx(merged, abs=0.006)  # ... so the PnL block must too
    assert rows["fees"] == pytest.approx(sum(f.fee for f in r.fills), abs=0.006)
    assert rows["fees"] > 0.0
    assert (rows["settle_pair"], rows["settle_dir"]) == (0.0, 0.0)  # nothing settled
    parts = rows["merge"] + rows["sell"] + rows["unrealised"]
    assert parts == pytest.approx(r.total_pnl, abs=0.03)


def test_markets_resolved_row_counts_the_traded_markets_among_the_resolved_ones(
    partial: BacktestResult, truncated: BacktestResult
) -> None:
    resolved_traded = sum(1 for m in partial.markets if m.resolved and m.traded)
    # the all-markets traded count (what the row used to print) is a different number
    assert partial.pairs.n_markets_traded == 4 != resolved_traded
    pnl = section(format_report(partial), "Per-market PnL")
    assert pnl["Markets resolved"] == f"2 of 4 ({resolved_traded} traded)"
    assert (
        pnl["Statistics cover"] == "resolved markets only (2 of 4); the PnL rows below cover all 4"
    )
    assert partial.pnl.n_markets_resolved_traded == resolved_traded
    none = section(format_report(truncated), "Per-market PnL")
    assert none["Markets resolved"] == "0 of 2 (0 traded)"
    assert truncated.pairs.n_markets_traded == 2  # traded, but none of them resolved


def test_a_fully_resolved_run_has_no_unrealised_part_and_no_scope_note() -> None:
    cfg = replace(
        BotConfig(), sim=replace(BotConfig().sim, n_windows=1, assets=("BTC",), window_seconds=300)
    )
    r = run_backtest(cfg, SyntheticFeed(cfg), source_label="synthetic:resolved")
    assert r.unresolved_markets == []
    assert r.pnl.pnl_unrealised == pytest.approx(0.0, abs=1e-6)
    text = format_report(r)
    pnl = section(text, "Per-market PnL")
    assert pnl["Markets resolved"] == f"1 of 1 ({r.pairs.n_markets_traded} traded)"
    assert "Statistics cover" not in pnl and "Of which unrealised" not in section(text, "Result")
    rows = decomposition(r)
    assert rows["unrealised"] == 0.0
    assert rows["merge"] + rows["sell"] + rows["settle_pair"] + rows["settle_dir"] == (
        pytest.approx(r.total_pnl, abs=0.03)
    )


def test_result_block_names_the_unrealised_part_of_total_pnl_for_open_markets() -> None:
    pnl = replace(make_result().pnl, pnl_unrealised=-5.0)
    open_run = make_result(unresolved_markets=["a", "b"], pnl=pnl)
    row = section(format_report(open_run), "Result")["Of which unrealised"]
    assert row == "-5.00  (open inventory of 2 unresolved markets, marked at the last mids)"
    one = make_result(unresolved_markets=["a"], pnl=pnl)
    assert "1 unresolved market," in section(format_report(one), "Result")["Of which unrealised"]
    assert "Of which unrealised" not in section(format_report(make_result()), "Result")


def test_invariants_row_says_which_invariants_ran_when_markets_are_unresolved(
    truncated: BacktestResult,
) -> None:
    def label(**kw: object) -> str:
        return section(format_report(make_result(**kw)), "Run")["Invariants"]

    # DESIGN section 7: invariant 4 (everything settled, no open orders or reserved cash, the
    # PnL breakdown sums to final equity) cannot hold or be checked while markets are open
    two = "checked 1-3; 4 (final accounting) skipped: 2 unresolved markets"
    assert label(unresolved_markets=["a", "b"]) == two
    assert label(unresolved_markets=["a"]) == (
        "checked 1-3; 4 (final accounting) skipped: 1 unresolved market"
    )
    assert label(unresolved_markets=["a"], invariants_checked=False) == "NOT checked"
    assert label() == "checked (DESIGN section 7, 1-4)"
    assert section(format_report(truncated), "Run")["Invariants"] == two


def test_unresolved_markets_are_not_described_as_unhedged_at_resolution() -> None:
    resolved = make_result()
    pairs = section(format_report(resolved), "Pairing and inventory")
    assert "Markets ending unhedged (>=1 share)" in pairs
    assert "* unpaired inventory of at least 1 share remained at resolution\n" in (
        format_report(resolved)
    )
    open_market = record(resolved=False, winner=None)
    text = format_report(make_result(markets=[open_market], unresolved_markets=["m"]))
    pairs = section(text, "Pairing and inventory")
    assert "Markets ending unhedged (>=1 share)" not in pairs
    assert pairs["Markets unhedged at close or stream end (>=1 share)"].startswith("1 of 1")
    assert pairs["Pairs formed"] == "100.0 (merged 100, held at close or stream end 0.0)"
    assert "Avg cost per pair held at close or stream end" in pairs
    assert "(unresolved markets: at the end of the stream)" in text


# --------------------------------------------------------------------------- caveats


def test_paper_report_says_the_data_path_is_unverified_and_winners_are_inferred() -> None:
    lines = format_report(make_result("paper-live")).splitlines()
    assert lines[2] == PAPER_BANNER and lines[3] == PAPER_UNVERIFIED_NOTE
    assert headline_banner("paper-live") == PAPER_BANNER + "\n" + PAPER_UNVERIFIED_NOTE
    assert "UNVERIFIED" in PAPER_UNVERIFIED_NOTE and "INFERRED" in PAPER_UNVERIFIED_NOTE
    text = format_report(make_result("paper-live"))
    caveats = " ".join(text.split("\nCaveats\n-------\n")[1].split())
    assert "winners are inferred from Binance/Coinbase spot" in caveats
    assert "endpoints" in caveats and "never been checked" in caveats
    # the data-path caveat belongs to paper sources only
    for label in ("synthetic:unit", "replay: events.jsonl"):
        other = format_report(make_result(label))
        assert "UNVERIFIED" not in other and "inferred" not in other


@pytest.mark.parametrize("label", ["synthetic:unit", "paper-live", "replay: events.jsonl"])
def test_caveats_call_the_fee_parameters_placeholders_for_every_source(label: str) -> None:
    text = format_report(make_result(label))
    caveats = " ".join(text.split("\nCaveats\n-------\n")[1].split())
    assert "Fee parameters are placeholders, not Polymarket's verified fee schedule" in caveats
    assert "uncalibrated defaults" not in caveats
    assert "model weights and strategy thresholds are uncalibrated" in caveats


# --------------------------------------------------------------------------- run directory


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_write_run_dir_writes_exactly_the_five_files(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "run"
    paths = write_run_dir(make_result(), BotConfig(), out)
    assert tuple(paths) == RUN_FILES == (
        "result.json", "config.json", "fills.csv", "markets.csv", "equity.csv",
    )  # fmt: skip
    assert sorted(p.name for p in out.iterdir()) == sorted(RUN_FILES)
    assert all(p.parent == out and p.is_file() for p in paths.values())


def test_json_files_round_trip(tmp_path: Path) -> None:
    result, cfg = make_result(), BotConfig()
    paths = write_run_dir(result, cfg, tmp_path)
    text = paths["result.json"].read_text(encoding="utf-8")
    assert json.loads(text) == result.to_dict()
    assert text.endswith("}\n") and "\r" not in text
    loaded = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    assert loaded == config_to_dict(cfg)
    assert config_from_dict(loaded) == cfg  # the written config reproduces the run's settings


def test_fills_csv_has_one_exact_row_per_fill(tmp_path: Path) -> None:
    result = make_result()
    paths = write_run_dir(result, BotConfig(), tmp_path)
    raw = paths["fills.csv"].read_text(encoding="utf-8").splitlines()
    assert raw[0] == "fill_id,ts,market_id,order_id,token,side,price,size,notional,fee,is_maker"
    assert raw[1] == "f1,1102.0,m1,o1,UP,BUY,0.49,100.0,49.0,0.0,True"
    rows = read_csv(paths["fills.csv"])
    assert len(rows) == len(result.fills) == 3
    for row, fill in zip(rows, result.fills, strict=True):
        assert row["fill_id"] == fill.fill_id and row["market_id"] == fill.market_id
        assert (row["token"], row["side"]) == (fill.token.value, fill.side.value)
        assert float(row["price"]) == fill.price and float(row["size"]) == fill.size
        assert float(row["fee"]) == fill.fee and float(row["ts"]) == fill.ts
        assert float(row["notional"]) == fill.notional
        assert row["is_maker"] == str(fill.is_maker)
    assert float(rows[2]["fee"]) == -0.001  # a maker rebate stays negative


def test_markets_csv_has_every_record_field_and_blank_for_none(tmp_path: Path) -> None:
    result = make_result(markets=[record(), record(market_id="m2", avg_merge_pair_cost=None)])
    paths = write_run_dir(result, BotConfig(), tmp_path)
    rows = read_csv(paths["markets.csv"])
    assert list(rows[0]) == [f.name for f in dataclasses.fields(MarketRecord)]
    assert rows[0]["market_id"] == "BTC-1700000100-900s" and rows[1]["market_id"] == "m2"
    assert rows[0]["avg_pair_cost_at_close"] == ""  # None
    assert rows[0]["avg_merge_pair_cost"] == "0.98" and rows[1]["avg_merge_pair_cost"] == ""
    assert (rows[0]["resolved"], rows[0]["ended_unhedged"]) == ("True", "True")
    assert float(rows[0]["pnl"]) == 13.0 and rows[0]["unpaired_token"] == "UP"


def test_equity_csv(tmp_path: Path) -> None:
    curve = [EquityPoint(1, 100.0, 10_000.0, 10_000.0), EquityPoint(60, 159.0, 9_987.1, 9_902.1)]
    paths = write_run_dir(make_result(equity_curve=curve), BotConfig(), tmp_path)
    assert paths["equity.csv"].read_text(encoding="utf-8").splitlines() == [
        "event_index,ts,equity,cash",
        "1,100.0,10000.0,10000.0",
        "60,159.0,9987.1,9902.1",
    ]
    empty = write_run_dir(make_result(equity_curve=[], fills=[], markets=[]), BotConfig(), tmp_path)
    assert empty["equity.csv"].read_text(encoding="utf-8") == "event_index,ts,equity,cash\n"
    assert len(empty["fills.csv"].read_text(encoding="utf-8").splitlines()) == 1  # header only


def test_existing_run_dir_is_overwritten_and_other_files_are_left_alone(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text("stale", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
    write_run_dir(make_result(), BotConfig(), tmp_path)
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["n_events"] == 1802
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_run_dir_is_deterministic(tmp_path: Path) -> None:
    write_run_dir(make_result(), BotConfig(), tmp_path / "a")
    write_run_dir(make_result(), BotConfig(), tmp_path / "b")
    for name in RUN_FILES:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_run_dir_of_a_real_run_matches_the_result(tmp_path: Path) -> None:
    cfg = replace(
        BotConfig(), sim=replace(BotConfig().sim, n_windows=1, assets=("BTC",), window_seconds=300)
    )
    result = run_backtest(cfg, SyntheticFeed(cfg), source_label="synthetic:real")
    paths = write_run_dir(result, cfg, tmp_path)
    fills = read_csv(paths["fills.csv"])
    assert len(fills) == result.activity.n_fills
    assert sum(float(r["notional"]) for r in fills) == pytest.approx(
        result.activity.volume_maker + result.activity.volume_taker
    )
    markets = read_csv(paths["markets.csv"])
    assert [r["market_id"] for r in markets] == [m.market_id for m in result.markets]
    assert sum(float(r["pnl"]) for r in markets) == pytest.approx(result.total_pnl)
    equity = read_csv(paths["equity.csv"])
    assert [int(r["event_index"]) for r in equity] == [p.event_index for p in result.equity_curve]
    assert float(equity[-1]["equity"]) == result.final_equity
    assert json.loads(paths["result.json"].read_text(encoding="utf-8")) == result.to_dict()
    assert "SYNTHETIC DATA" in format_report(result)
