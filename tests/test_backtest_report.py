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
from abc_trading.types import Fill, Outcome, Side

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
            1, 13.0, 0.0, 0.0, 13.0, 13.0, 1.0, 0.0, 2.0, 0.0, 11.0, 0.0, 10_013.0, 0.0, 0.0
        ),
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
    assert pnl["Markets resolved"] == "1 (1 traded)"
    assert pnl["Mean / std per market"] == "+13.00 / 0.00"
    assert pnl["Win rate (traded markets)"] == "100.0%"
    assert pnl["PnL from merges"] == "+2.00"
    assert pnl["PnL at settlement, directional"] == "+11.00"
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
