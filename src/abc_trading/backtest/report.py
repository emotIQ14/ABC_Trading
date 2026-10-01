"""Plain-text report and run-directory output for a ``BacktestResult``.

``format_report`` returns aligned plain text. Its headline block always says what the numbers
are: a result whose ``source_label`` contains "synthetic" carries the banner ``SYNTHETIC DATA -
mechanics validation only, not evidence of profitability``. Every other source still gets a
caveat, because fees, model weights and the paper fill model are uncalibrated placeholders.

``write_run_dir`` writes ``result.json``, ``config.json``, ``fills.csv``, ``markets.csv`` and
``equity.csv`` (UTF-8, ``\\n`` line endings, full float precision, empty cell for None). The
files are a pure function of the result and the config.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from abc_trading.backtest.runner import BacktestResult, MarketRecord
from abc_trading.config import BotConfig, config_to_dict

SYNTHETIC_BANNER = "SYNTHETIC DATA - mechanics validation only, not evidence of profitability"
PAPER_BANNER = (
    "PAPER MODE - live public data with SIMULATED fills; no order was or can be placed on any "
    "exchange. Not evidence of profitability."
)
GENERIC_NOTE = (
    "Source not marked synthetic. Results depend entirely on the data and on the paper "
    "exchange's fill model; fees and model weights are uncalibrated placeholders. "
    "Not evidence of profitability."
)
SOURCE_CLAIM_AVG_NOTIONAL = 53.0  # unverified figure from the source claim, for comparison only
MAX_MARKET_ROWS = 40

RUN_FILES = ("result.json", "config.json", "fills.csv", "markets.csv", "equity.csv")


def is_synthetic(source_label: str) -> bool:
    """True iff the label marks the data as synthetic (case-insensitive 'synthetic')."""
    return "synthetic" in source_label.lower()


def headline_banner(source_label: str) -> str:
    """The honesty statement that opens every report for this kind of source."""
    if is_synthetic(source_label):
        return SYNTHETIC_BANNER
    if source_label.lower().startswith("paper"):
        return PAPER_BANNER
    return GENERIC_NOTE


# --------------------------------------------------------------------------- formatting helpers


def _num(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{digits}f}"


def _signed(x: float | None, digits: int = 2) -> str:
    """Signed number; a value that rounds to zero prints as +0.00, never -0.00."""
    if x is None:
        return "n/a"
    shown = x if round(x, digits) != 0 else 0.0
    return f"{shown:+,.{digits}f}"


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def _section(title: str, rows: Sequence[tuple[str, str]]) -> list[str]:
    """A titled block with the values aligned in one column."""
    width = max((len(k) for k, _ in rows), default=0)
    lines = ["", title, "-" * len(title)]
    lines += [f"  {k.ljust(width)}  {v}" for k, v in rows]
    return lines


def _counters(title: str, counters: dict[str, Any]) -> list[str]:
    rows = [(k, _count(v)) for k, v in counters.items() if not isinstance(v, dict)]
    for k, v in counters.items():
        if isinstance(v, dict):
            rows.append((k, ", ".join(f"{a}={_count(b)}" for a, b in v.items()) or "none"))
    return _section(title, rows)


def _count(v: Any) -> str:
    if isinstance(v, float) and not v.is_integer():
        return f"{v:,.2f}"
    return f"{int(v):,}" if isinstance(v, int | float) and not isinstance(v, bool) else str(v)


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    body = [list(r) for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in body)) if body else len(h) for i, h in enumerate(header)
    ]

    def line(cells: Sequence[str]) -> str:
        # first two columns are text (left), the rest numbers (right)
        out = [
            c.ljust(w) if i < 2 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  " + "  ".join(out).rstrip()

    return [line(header), line(["-" * w for w in widths]), *(line(r) for r in body)]


def _market_row(r: MarketRecord) -> list[str]:
    pair_cost = "n/a" if r.avg_pair_cost_at_close is None else f"{r.avg_pair_cost_at_close:.4f}"
    return [
        r.market_id,
        r.winner or "unresolved",
        _signed(r.pnl),
        str(r.n_fills),
        _num(r.max_inventory_shares, 0),
        _num(r.max_net_shares, 0),
        _num(r.merged_pairs, 0),
        _num(r.paired_at_close, 1),
        pair_cost,
        _num(r.unpaired_at_close, 1) + (" *" if r.ended_unhedged else ""),
    ]


# --------------------------------------------------------------------------- the report


def format_report(result: BacktestResult) -> str:
    """Aligned plain-text report; deterministic for a given result."""
    lines = [
        "ABC_Trading backtest report",
        "===========================",
        headline_banner(result.source_label),
    ]
    lines += _section("Run", _run_rows(result))
    lines += _section("Result", _result_rows(result))
    lines += _section("Per-market PnL", _pnl_rows(result))
    lines += _section("Trading activity", _activity_rows(result))
    lines += _section("Pairing and inventory", _pair_rows(result))
    lines += _section("Model calibration (Brier score, lower is better)", _calibration_rows(result))
    lines += _counters("Engine counters", result.engine_stats)
    lines += _counters("Exchange counters", result.exchange_stats)
    lines += _counters("Runner counters", result.runner_stats)
    lines += _markets_block(result)
    lines += _caveats()
    return "\n".join(lines) + "\n"


def _run_rows(r: BacktestResult) -> list[tuple[str, str]]:
    return [
        ("Source", r.source_label),
        (
            "Events",
            f"{r.n_events:,} ({r.n_snapshots:,} snapshots, {r.n_resolutions:,} resolutions)",
        ),
        ("Time span (unix s)", f"{_num(r.first_ts, 0)} .. {_num(r.last_ts, 0)}"),
        (
            "Invariants",
            "checked (DESIGN section 7, 1-4)" if r.invariants_checked else "NOT checked",
        ),
        ("Unresolved markets", str(len(r.unresolved_markets)) if r.unresolved_markets else "none"),
    ]


def _result_rows(r: BacktestResult) -> list[tuple[str, str]]:
    p = r.pnl
    return [
        ("Initial cash", _num(r.initial_cash)),
        ("Final equity", _num(r.final_equity)),
        ("Total PnL", f"{_signed(r.total_pnl)}  ({_signed(r.pnl_pct_of_initial_cash)}%)"),
        (
            "Max drawdown",
            f"{_num(p.max_drawdown_usd)}  ({_pct(p.max_drawdown_frac, 2)} of peak; "
            f"sampled every {r.equity_sample_every} events)",
        ),
        ("Kill switch", "TRIPPED" if r.kill_switch else "not tripped"),
    ]


def _pnl_rows(r: BacktestResult) -> list[tuple[str, str]]:
    p = r.pnl
    return [
        ("Markets resolved", f"{p.n_markets_resolved} ({r.pairs.n_markets_traded} traded)"),
        ("Mean / std per market", f"{_signed(p.mean_market_pnl)} / {_num(p.std_market_pnl)}"),
        ("Sharpe-like (mean/std*sqrt(n))", _signed(p.sharpe_like)),
        ("Best / worst market", f"{_signed(p.best_market_pnl)} / {_signed(p.worst_market_pnl)}"),
        ("Win rate (traded markets)", _pct(p.win_rate_traded)),
        ("PnL from merges", _signed(p.pnl_merge)),
        ("PnL from sells (flatten)", _signed(p.pnl_sell)),
        ("PnL at settlement, pairs", _signed(p.pnl_settle_pair)),
        ("PnL at settlement, directional", _signed(p.pnl_settle_directional)),
        ("Fees paid (inside the above)", _num(p.fees_paid)),
    ]


def _activity_rows(r: BacktestResult) -> list[tuple[str, str]]:
    a = r.activity
    avg = a.avg_fill_notional
    claim = (
        f"   (source claim: ~${SOURCE_CLAIM_AVG_NOTIONAL:.0f}, unverified; not tuned to it)"
        if avg is not None
        else ""
    )
    return [
        ("Trades (fills)", f"{a.n_fills:,} ({a.n_buy_fills:,} buys, {a.n_sell_fills:,} sells)"),
        ("Maker / taker fills", f"{a.n_maker_fills:,} / {a.n_taker_fills:,}"),
        ("Volume maker / taker (USDC)", f"{_num(a.volume_maker)} / {_num(a.volume_taker)}"),
        ("Average trade notional (USDC)", _num(avg) + claim),
        ("Median trade notional (USDC)", _num(a.median_fill_notional)),
    ]


def _pair_rows(r: BacktestResult) -> list[tuple[str, str]]:
    s = r.pairs
    cost = s.avg_merge_pair_cost
    merged_cost = (
        "n/a" if cost is None else f"{cost:.4f}  (margin {_signed(100.0 * (1.0 - cost))}c per pair)"
    )
    max_inv = max((m.max_inventory_shares for m in r.markets), default=0.0)
    max_net = max((m.max_net_shares for m in r.markets), default=0.0)
    return [
        ("Shares bought", _num(s.shares_bought, 1)),
        (
            "Pairs formed",
            f"{_num(s.pairs_formed, 1)} (merged {_num(s.merged_pairs, 0)}, held at close "
            f"{_num(s.paired_at_close, 1)})",
        ),
        ("Pair completion rate", _pct(s.pair_completion_rate)),
        ("Avg all-in cost per merged pair", merged_cost),
        ("Avg cost per pair held at close", _num(s.avg_close_pair_cost, 4)),
        (
            "Markets ending unhedged (>=1 share)",
            f"{s.n_markets_unhedged} of {s.n_markets} ({_pct(s.frac_markets_unhedged)}); "
            f"of traded: {_pct(s.frac_traded_markets_unhedged)}",
        ),
        ("Max inventory / max net (shares)", f"{_num(max_inv, 0)} / {_num(max_net, 0)}"),
    ]


def _calibration_rows(r: BacktestResult) -> list[tuple[str, str]]:
    c = r.calibration
    return [
        (
            "Markets scored",
            f"{c.n_scored} of {c.n_resolved} (no ACCUMULATE snapshot: {c.n_no_accumulate}, "
            f"invalid fair value: {c.n_invalid})",
        ),
        ("Engine p_up (model shrunk to market)", _num(c.brier_model, 4)),
        ("Raw model probability", _num(c.brier_model_raw, 4)),
        ("Market UP mid (same snapshot)", _num(c.brier_market, 4)),
        ("Skill vs market (positive = better)", _signed(c.skill_vs_market, 3)),
    ]


def _markets_block(r: BacktestResult) -> list[str]:
    header = [
        "market", "winner", "pnl", "fills", "max inv", "max net", "merged", "held pairs",
        "pair cost", "unpaired",
    ]  # fmt: skip
    shown = r.markets[:MAX_MARKET_ROWS]
    lines = ["", "Markets", "-------"]
    lines += _table(header, (_market_row(m) for m in shown))
    if len(r.markets) > len(shown):
        lines.append(f"  ... {len(r.markets) - len(shown)} more (see markets.csv)")
    if any(m.ended_unhedged for m in shown):
        lines.append("  * unpaired inventory of at least 1 share remained at resolution")
    return lines


def _caveats() -> list[str]:
    return [
        "",
        "Caveats",
        "-------",
        "  * Fee parameters, model weights and strategy thresholds are uncalibrated defaults.",
        "  * The paper exchange fill model is conservative by design but not validated against",
        "    real order books or queues.",
        "  * No live trading exists in this project: nothing here places or signs real orders.",
    ]


# --------------------------------------------------------------------------- run directory


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def _write_json(path: Path, data: object) -> None:
    text = json.dumps(data, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def write_run_dir(result: BacktestResult, cfg: BotConfig, out_dir: str | Path) -> dict[str, Path]:
    """Write the five run files into ``out_dir`` (created if needed; existing files replaced).

    Returns ``{file name: path}``. ``fills.csv`` has one row per fill, ``markets.csv`` one row
    per ``MarketRecord`` field, ``equity.csv`` the sampled mark-to-mid equity curve.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / name for name in RUN_FILES}
    _write_json(paths["result.json"], result.to_dict())
    _write_json(paths["config.json"], config_to_dict(cfg))
    _write_csv(
        paths["fills.csv"],
        ["fill_id", "ts", "market_id", "order_id", "token", "side", "price", "size", "notional",
         "fee", "is_maker"],
        (
            [f.fill_id, f.ts, f.market_id, f.order_id, f.token.value, f.side.value, f.price,
             f.size, f.notional, f.fee, f.is_maker]
            for f in result.fills
        ),
    )  # fmt: skip
    fields = [f.name for f in dataclasses.fields(MarketRecord)]
    _write_csv(
        paths["markets.csv"],
        fields,
        ([getattr(m, name) for name in fields] for m in result.markets),
    )
    _write_csv(
        paths["equity.csv"],
        ["event_index", "ts", "equity", "cash"],
        ([p.event_index, p.ts, p.equity, p.cash] for p in result.equity_curve),
    )
    return paths
