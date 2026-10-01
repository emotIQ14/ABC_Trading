"""Backtest runner: drives ``MarketMakerEngine`` and ``PaperExchange`` over a feed of events.

There is no live trading here. The runner only replays ``FeedEvent`` s (synthetic, recorded or
live-polled public data) into the simulated exchange. Results validate MECHANICS, never
profitability: fee parameters, model weights and the fill model are uncalibrated placeholders.

Event loop (DESIGN section 7 ordering)
--------------------------------------
``MarketSnapshot``:
    1. ``fills = exchange.process(snap)``; each fill goes to ``engine.on_fill``.
    2. ``open = exchange.open_orders(market_id)``; ``actions = engine.on_snapshot(snap, open)``.
    3. Actions run in order: ``CancelOrder`` -> ``exchange.cancel``; ``MergePairs`` ->
       ``exchange.merge`` then ``engine.on_merge`` on success; ``PlaceOrder`` ->
       ``exchange.submit`` (rejections are counted by reason; the engine is NOT told).
    4. ``exchange.drain_fills()`` goes to ``engine.on_fill``: with ``latency_ticks == 0`` an IOC
       fills inside ``submit``, and the engine's portfolio must see it before the next check.
``MarketResolved``: ``exchange.settle`` then ``engine.on_settlement``.

Events must have finite, non-decreasing timestamps (``ValueError`` otherwise), snapshots of a
resolved market and resolutions of unknown or already resolved markets are errors too.

Invariants (``check_invariants=True``; ``InvariantError`` with a precise message)
    1. Reconciliation: engine ``Portfolio.cash`` equals ``PaperExchange.balance()`` and the
       per-market positions agree (<= ``TOL``), after every event for the market it touched, and
       for every market every ``FULL_SWEEP_EVERY`` events and at the end.
    2. ``cash + capital_at_risk - initial_cash == realised pnl`` after every event.
    3. No negative cash or quantity, ``reserved_cash <= cash``.
    4. At the end every market is resolved, no open orders remain, positions are zero and
       ``sum(PnLBreakdown.total) == final equity - initial cash``. A stream that ends with
       unresolved markets violates this unless ``allow_unresolved=True`` (paper mode); those markets
       are then reported with ``resolved=False`` and marked at the last snapshot mids.
A ``ValueError`` raised by the engine's portfolio when it is handed an exchange fill / merge /
settlement is reported as an invariant-1 violation (the two disagree), and an ``AssertionError``
from a component's own internal check is wrapped into ``InvariantError`` as well.

Metrics
-------
* ``equity_curve``: cash plus open positions marked at the SNAPSHOT MIDS (each token at its own
  last mid, else ``1 - the other token's mid``, else cost); a point is stored for the first event,
  every ``equity_sample_every``-th event and the last event. Max drawdown is measured on that
  sampled curve (use ``equity_sample_every=1`` for event-granular drawdown).
* Per-market PnL is ``PnLBreakdown.total``; ``sharpe_like = mean / std * sqrt(n)`` over resolved
  markets (sample std; a t-statistic, not an annualised Sharpe ratio).
* Pairing: ``pairs_formed = merged pairs + pairs held at close``; ``pair_completion_rate =
  2 * pairs_formed / shares bought`` (share of bought shares that ended in a pair); a market
  ``ended_unhedged`` iff at least ``DUST_SHARES`` (1.0) unpaired shares remained at resolution
  (this includes deliberate holds; merges are whole pairs, so a fractional remainder below one
  share is dust: still reported in ``unpaired_at_close`` but not counted as unhedged).
* Calibration: Brier score of the engine's ``p_up`` at the LAST ACCUMULATE snapshot of each market
  versus the winner, next to the raw model probability and the market's UP mid at that same
  snapshot (so the model can be compared with the price it traded against). Markets with no
  ACCUMULATE snapshot or an invalid fair value there are not scored (counted).
* ``avg_fill_notional`` is the mean ``price * size`` over all fills (reported for comparison with
  the unverified "~$53 average trade" in the source claim; nothing is tuned to it).

``BacktestResult.to_dict()`` is JSON-serialisable (no NaN/inf), has a stable key order and is a
pure function of the inputs; the fills are represented by their count and a SHA-256 digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from abc_trading.backtest import metrics
from abc_trading.config import BotConfig
from abc_trading.exchange.paper import PaperExchange
from abc_trading.inventory import MarketInventory, PnLBreakdown
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.types import (
    Action,
    CancelOrder,
    FeedEvent,
    Fill,
    MarketResolved,
    MarketSnapshot,
    MarketSpec,
    MergePairs,
    Outcome,
    Phase,
    PlaceOrder,
    Side,
)

TOL = 1e-6  # USDC / shares tolerance of the reconciliation and accounting invariants
FULL_SWEEP_EVERY = 500  # events between full all-market invariant sweeps
DUST_SHARES = 1.0  # unpaired inventory below this cannot be merged (whole pairs) and is dust


class InvariantError(Exception):
    """A DESIGN section 7 invariant was violated (an accounting or reconciliation bug)."""


class EquityPoint(NamedTuple):
    """One sampled point of the mark-to-mid equity curve (USDC)."""

    event_index: int  # 1-based count of events processed so far
    ts: float
    equity: float
    cash: float


@dataclass(frozen=True, slots=True)
class MarketRecord:
    """Outcome of one market. Money in USDC, quantities in shares.

    ``resolved=False`` (stream ended before the resolution) leaves ``winner`` None and the
    settlement pnl at 0; ``pnl`` is then the realised sell + merge pnl so far.
    """

    market_id: str
    asset: str
    resolved: bool
    winner: str | None
    sell_pnl: float
    merge_pnl: float
    settle_pair_pnl: float
    settle_directional_pnl: float
    fees_paid: float
    pnl: float
    n_fills: int
    buy_shares: float
    sell_shares: float
    volume_maker: float  # USDC notional of maker fills
    volume_taker: float  # USDC notional of taker fills
    max_inventory_shares: float  # max of qty[UP] + qty[DOWN] held at once
    max_net_shares: float  # max of |qty[UP] - qty[DOWN]|
    max_capital_at_risk: float  # max all-in cost basis of open positions
    merged_pairs: float
    paired_at_close: float
    avg_pair_cost_at_close: float | None  # avg_cost[UP] + avg_cost[DOWN]; None if one side empty
    avg_merge_pair_cost: float | None  # 1 - merge_pnl / merged_pairs; None if nothing merged
    unpaired_token: str | None
    unpaired_at_close: float
    ended_unhedged: bool
    pair_completion_rate: float | None  # 2 * (merged + paired at close) / buy_shares
    p_up_last_accumulate: float | None  # engine p_up at the last ACCUMULATE snapshot

    @property
    def traded(self) -> bool:
        return self.n_fills > 0


@dataclass(frozen=True, slots=True)
class ActivityStats:
    n_fills: int
    n_buy_fills: int
    n_sell_fills: int
    n_maker_fills: int
    n_taker_fills: int
    volume_maker: float  # USDC notional
    volume_taker: float
    avg_fill_notional: float | None  # mean price * size; None without fills
    median_fill_notional: float | None


@dataclass(frozen=True, slots=True)
class PairStats:
    n_markets: int  # markets in the result (resolved or not)
    n_markets_traded: int
    shares_bought: float
    merged_pairs: float
    paired_at_close: float
    pairs_formed: float
    pair_completion_rate: float | None
    avg_merge_pair_cost: float | None  # pooled over all merges
    avg_close_pair_cost: float | None  # pooled over pairs held at close
    n_markets_unhedged: int
    frac_markets_unhedged: float | None  # over all markets; None without markets
    frac_traded_markets_unhedged: float | None  # over markets with at least one fill


@dataclass(frozen=True, slots=True)
class PnLStats:
    n_markets_resolved: int
    mean_market_pnl: float
    std_market_pnl: float
    sharpe_like: float
    best_market_pnl: float
    worst_market_pnl: float
    win_rate_traded: float | None  # share of resolved traded markets with pnl > 0
    pnl_sell: float
    pnl_merge: float
    pnl_settle_pair: float
    pnl_settle_directional: float
    fees_paid: float
    peak_equity: float
    max_drawdown_usd: float
    max_drawdown_frac: float


@dataclass(frozen=True, slots=True)
class CalibrationStats:
    n_resolved: int
    n_scored: int
    n_no_accumulate: int  # resolved markets that never had an ACCUMULATE snapshot
    n_invalid: int  # last ACCUMULATE snapshot had an invalid fair value or no UP mid
    brier_model: float | None  # engine p_up (model shrunk toward the market)
    brier_model_raw: float | None  # p_up_model before shrinking
    brier_market: float | None  # UP mid of the market at the same snapshot
    skill_vs_market: float | None  # 1 - brier_model / brier_market


@dataclass(slots=True)
class BacktestResult:
    """Everything a backtest produced. ``to_dict()`` is the JSON-safe, deterministic view."""

    source_label: str
    invariants_checked: bool
    equity_sample_every: int
    n_events: int
    n_snapshots: int
    n_resolutions: int
    first_ts: float | None
    last_ts: float | None
    initial_cash: float
    final_equity: float
    total_pnl: float
    pnl_pct_of_initial_cash: float
    kill_switch: bool
    activity: ActivityStats
    pairs: PairStats
    pnl: PnLStats
    calibration: CalibrationStats
    unresolved_markets: list[str]
    markets: list[MarketRecord]
    engine_stats: dict[str, int]
    exchange_stats: dict[str, float]
    runner_stats: dict[str, Any]
    equity_curve: list[EquityPoint]
    fills_sha256: str
    fills: list[Fill] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict, JSON-serialisable and deterministic (fixed key order). The raw
        ``fills`` are summarised by ``activity`` and ``fills_sha256``."""
        return {
            f.name: _plain(getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.name != "fills"
        }


def _plain(obj: Any) -> Any:
    """Recursively convert to JSON types; raises ValueError on a non-finite float."""
    if isinstance(obj, Enum):
        return obj.value
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float in result: {obj!r}")
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    raise TypeError(f"cannot serialise {type(obj).__name__}")


# --------------------------------------------------------------------------- pure aggregations


def fills_digest(fills: Sequence[Fill]) -> str:
    """SHA-256 hex digest of the fills (every field, full float precision, in order)."""
    h = hashlib.sha256()
    for f in fills:
        row = (
            f"{f.fill_id}|{f.order_id}|{f.market_id}|{f.token.value}|{f.side.value}|"
            f"{f.price!r}|{f.size!r}|{f.fee!r}|{int(f.is_maker)}|{f.ts!r}\n"
        )
        h.update(row.encode("utf-8"))
    return h.hexdigest()


def activity_stats(fills: Sequence[Fill]) -> ActivityStats:
    """Counts, volumes and notional statistics of a list of fills."""
    notionals = [f.notional for f in fills]
    return ActivityStats(
        n_fills=len(fills),
        n_buy_fills=sum(1 for f in fills if f.side is Side.BUY),
        n_sell_fills=sum(1 for f in fills if f.side is Side.SELL),
        n_maker_fills=sum(1 for f in fills if f.is_maker),
        n_taker_fills=sum(1 for f in fills if not f.is_maker),
        volume_maker=math.fsum(f.notional for f in fills if f.is_maker),
        volume_taker=math.fsum(f.notional for f in fills if not f.is_maker),
        avg_fill_notional=metrics.mean(notionals) if notionals else None,
        median_fill_notional=metrics.median(notionals) if notionals else None,
    )


def pair_stats(records: Sequence[MarketRecord]) -> PairStats:
    """Pooled pairing statistics over market records (definitions in the module docstring)."""
    bought = math.fsum(r.buy_shares for r in records)
    merged = math.fsum(r.merged_pairs for r in records)
    closed = math.fsum(r.paired_at_close for r in records)
    merge_pnl = math.fsum(r.merge_pnl for r in records)
    close_cost = math.fsum(
        r.avg_pair_cost_at_close * r.paired_at_close
        for r in records
        if r.avg_pair_cost_at_close is not None
    )
    traded = [r for r in records if r.traded]
    unhedged = sum(1 for r in records if r.ended_unhedged)
    unhedged_traded = sum(1 for r in traded if r.ended_unhedged)
    return PairStats(
        n_markets=len(records),
        n_markets_traded=len(traded),
        shares_bought=bought,
        merged_pairs=merged,
        paired_at_close=closed,
        pairs_formed=merged + closed,
        pair_completion_rate=metrics.ratio(2.0 * (merged + closed), bought),
        avg_merge_pair_cost=None if merged <= 0.0 else 1.0 - merge_pnl / merged,
        avg_close_pair_cost=None if closed <= 0.0 else close_cost / closed,
        n_markets_unhedged=unhedged,
        frac_markets_unhedged=metrics.ratio(unhedged, len(records)),
        frac_traded_markets_unhedged=metrics.ratio(unhedged_traded, len(traded)),
    )


def pnl_stats(records: Sequence[MarketRecord], equity: Sequence[float], initial: float) -> PnLStats:
    """Per-market PnL statistics (resolved markets) and drawdown of the sampled equity curve."""
    resolved = [r for r in records if r.resolved]
    pnls = [r.pnl for r in resolved]
    traded = [r for r in resolved if r.traded]
    dd = metrics.max_drawdown(equity) if equity else metrics.Drawdown(0.0, 0.0, 0, 0)
    return PnLStats(
        n_markets_resolved=len(resolved),
        mean_market_pnl=metrics.mean(pnls) if pnls else 0.0,
        std_market_pnl=metrics.std(pnls) if len(pnls) >= 2 else 0.0,
        sharpe_like=metrics.sharpe_like(pnls),
        best_market_pnl=max(pnls, default=0.0),
        worst_market_pnl=min(pnls, default=0.0),
        win_rate_traded=metrics.ratio(sum(1 for r in traded if r.pnl > 0.0), len(traded)),
        pnl_sell=math.fsum(r.sell_pnl for r in resolved),
        pnl_merge=math.fsum(r.merge_pnl for r in resolved),
        pnl_settle_pair=math.fsum(r.settle_pair_pnl for r in resolved),
        pnl_settle_directional=math.fsum(r.settle_directional_pnl for r in resolved),
        fees_paid=math.fsum(r.fees_paid for r in resolved),
        peak_equity=max(equity, default=initial),
        max_drawdown_usd=dd.depth,
        max_drawdown_frac=dd.fraction,
    )


@dataclass(frozen=True, slots=True)
class CalibSample:
    """Forecasts recorded at a market's last ACCUMULATE snapshot."""

    p_up: float  # engine probability (model shrunk toward the market)
    p_model: float  # model probability before shrinking
    up_mid: float | None  # market UP mid at that snapshot
    valid: bool  # the fair value was valid there


def calibration_stats(samples: Sequence[tuple[CalibSample | None, Outcome]]) -> CalibrationStats:
    """Brier scores over resolved markets: ``(sample or None, winner)`` per market."""
    scored = [(s, w) for s, w in samples if s is not None and s.valid and s.up_mid is not None]
    n_none = sum(1 for s, _ in samples if s is None)
    if not scored:
        return CalibrationStats(
            len(samples), 0, n_none, len(samples) - n_none, None, None, None, None
        )
    winners = [w for _, w in scored]
    model = metrics.brier_up([s.p_up for s, _ in scored], winners)
    raw = metrics.brier_up([s.p_model for s, _ in scored], winners)
    market = metrics.brier_up([s.up_mid for s, _ in scored if s.up_mid is not None], winners)
    return CalibrationStats(
        n_resolved=len(samples),
        n_scored=len(scored),
        n_no_accumulate=n_none,
        n_invalid=len(samples) - n_none - len(scored),
        brier_model=model,
        brier_model_raw=raw,
        brier_market=market,
        skill_vs_market=metrics.brier_skill(model, market),
    )


def mark_to_mid(inv: MarketInventory, up_mid: float | None, down_mid: float | None) -> float:
    """Value of an open inventory at snapshot mids: each token at its own mid, else at one minus
    the other token's mid, else at cost."""
    total = 0.0
    for token, own, other in (
        (Outcome.UP, up_mid, down_mid),
        (Outcome.DOWN, down_mid, up_mid),
    ):
        qty = inv.qty[token]
        if qty <= 0.0:
            continue
        if own is not None:
            total += qty * own
        elif other is not None:
            total += qty * (1.0 - other)
        else:
            total += inv.cost[token]
    return total


# --------------------------------------------------------------------------- runner state


@dataclass(slots=True)
class _Close:
    """Inventory shape at the moment of resolution (or at the end of an unresolved stream)."""

    paired: float
    avg_up: float | None
    avg_down: float | None
    unpaired_token: Outcome | None
    unpaired_qty: float


@dataclass(slots=True)
class _Track:
    spec: MarketSpec
    max_inventory: float = 0.0
    max_net: float = 0.0
    max_capital: float = 0.0
    merged_pairs: float = 0.0
    winner: Outcome | None = None
    breakdown: PnLBreakdown | None = None
    close: _Close | None = None
    calib: CalibSample | None = None


def _close_of(inv: MarketInventory) -> _Close:
    token, qty = inv.unpaired()
    return _Close(
        paired=inv.paired_qty,
        avg_up=inv.avg_cost(Outcome.UP),
        avg_down=inv.avg_cost(Outcome.DOWN),
        unpaired_token=token,
        unpaired_qty=qty,
    )


def _market_record(track: _Track, inv: MarketInventory, fills: Sequence[Fill]) -> MarketRecord:
    """Build the record of one market from its tracker, final inventory and fills."""
    bd, close = track.breakdown, track.close
    assert close is not None  # set at resolution, or when the unresolved market is finalised
    buy = math.fsum(f.size for f in fills if f.side is Side.BUY)
    sell = math.fsum(f.size for f in fills if f.side is Side.SELL)
    pair_cost = (
        None if close.avg_up is None or close.avg_down is None else close.avg_up + close.avg_down
    )
    sell_pnl, merge_pnl = (bd.sell_pnl, bd.merge_pnl) if bd else (inv.sell_pnl, inv.merge_pnl)
    pair_pnl, dir_pnl = (bd.settle_pair_pnl, bd.settle_directional_pnl) if bd else (0.0, 0.0)
    merged = track.merged_pairs
    return MarketRecord(
        market_id=track.spec.market_id,
        asset=track.spec.asset,
        resolved=bd is not None,
        winner=None if track.winner is None else track.winner.value,
        sell_pnl=sell_pnl,
        merge_pnl=merge_pnl,
        settle_pair_pnl=pair_pnl,
        settle_directional_pnl=dir_pnl,
        fees_paid=inv.fees_paid,
        pnl=sell_pnl + merge_pnl + pair_pnl + dir_pnl,
        n_fills=len(fills),
        buy_shares=buy,
        sell_shares=sell,
        volume_maker=math.fsum(f.notional for f in fills if f.is_maker),
        volume_taker=math.fsum(f.notional for f in fills if not f.is_maker),
        max_inventory_shares=track.max_inventory,
        max_net_shares=track.max_net,
        max_capital_at_risk=track.max_capital,
        merged_pairs=merged,
        paired_at_close=close.paired,
        avg_pair_cost_at_close=pair_cost,
        avg_merge_pair_cost=None if merged <= 0.0 else 1.0 - merge_pnl / merged,
        unpaired_token=None if close.unpaired_token is None else close.unpaired_token.value,
        unpaired_at_close=close.unpaired_qty,
        ended_unhedged=close.unpaired_qty >= DUST_SHARES,
        pair_completion_rate=metrics.ratio(2.0 * (merged + close.paired), buy),
        p_up_last_accumulate=None if track.calib is None else track.calib.p_up,
    )


class BacktestRunner:
    """Feeds events one at a time into an engine and a paper exchange (see module docstring).

    ``feed`` processes one event, ``finish`` runs the end-of-stream checks and builds the result.
    ``engine`` and ``exchange`` are public so tests and reports can inspect them.
    """

    def __init__(
        self,
        cfg: BotConfig,
        *,
        check_invariants: bool = True,
        equity_sample_every: int = 60,
        source_label: str = "unknown",
        allow_unresolved: bool = False,
    ) -> None:
        if (
            isinstance(equity_sample_every, bool)
            or not isinstance(equity_sample_every, int)
            or equity_sample_every < 1
        ):
            raise ValueError(
                f"equity_sample_every must be an int >= 1, got {equity_sample_every!r}"
            )
        cfg.validate()
        self.cfg = cfg
        self.exchange = PaperExchange(cfg)
        self.engine = MarketMakerEngine(cfg)
        self.check_invariants = check_invariants
        self.source_label = source_label
        self.allow_unresolved = allow_unresolved
        self._sample_every = equity_sample_every
        self._initial = cfg.exchange.initial_cash
        self._tracks: dict[str, _Track] = {}
        self._open: dict[str, _Track] = {}
        self._mids: dict[str, tuple[float | None, float | None]] = {}
        self._fills: list[Fill] = []
        self._curve: list[EquityPoint] = []
        self._settled_pnl = 0.0
        self._n = 0
        self._n_snap = 0
        self._n_res = 0
        self._first_ts: float | None = None
        self._last_ts: float | None = None
        self._rejections: Counter[str] = Counter()
        self._orders_submitted = 0
        self._cancel_misses = 0
        self._merges_done = 0
        self._merge_failures = 0
        self._finished = False

    # ------------------------------------------------------------------ public API

    def feed(self, event: FeedEvent) -> None:
        """Process one event (snapshot or resolution)."""
        if self._finished:
            raise ValueError("runner already finished")
        if not isinstance(event, MarketSnapshot | MarketResolved):
            raise TypeError(f"event #{self._n + 1}: not a FeedEvent: {type(event).__name__}")
        ts = event.ts
        if not math.isfinite(ts):
            raise ValueError(f"event #{self._n + 1}: ts must be finite, got {ts!r}")
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(
                f"event #{self._n + 1}: ts {ts!r} goes backwards (previous event at "
                f"{self._last_ts!r}); events must be in non-decreasing time order"
            )
        self._n += 1
        if self._first_ts is None:
            self._first_ts = ts
        self._last_ts = ts
        try:
            if isinstance(event, MarketSnapshot):
                self._on_snapshot(event)
            else:
                self._on_resolved(event)
        except AssertionError as exc:
            raise InvariantError(
                f"event #{self._n} (ts={ts!r}): component check failed: {exc}"
            ) from exc
        if self._n == 1 or self._n % self._sample_every == 0:
            self._sample()

    def finish(self) -> BacktestResult:
        """End-of-stream checks (invariant 4) and the result. The runner cannot be reused."""
        if self._finished:
            raise ValueError("runner already finished")
        self._finished = True
        unresolved = list(self._open)
        try:
            if self.check_invariants:
                self._check_end(unresolved)
        except AssertionError as exc:
            raise InvariantError(f"end of stream: component check failed: {exc}") from exc
        if self._n > 0 and (not self._curve or self._curve[-1].event_index != self._n):
            self._sample()
        return self._build_result(unresolved)

    # ------------------------------------------------------------------ events

    def _where(self, market_id: str | None = None) -> str:
        loc = f"event #{self._n} (ts={self._last_ts!r}"
        return loc + (f", market={market_id!r})" if market_id else ")")

    def _on_snapshot(self, snap: MarketSnapshot) -> None:
        mid = snap.market.market_id
        track = self._tracks.get(mid)
        if track is not None and mid not in self._open:
            raise ValueError(f"{self._where(mid)}: snapshot for an already resolved market")
        if track is None:
            track = self._tracks[mid] = self._open[mid] = _Track(snap.market)
        self._n_snap += 1
        self._mids[mid] = (snap.up_book.mid, snap.down_book.mid)
        self._apply_fills(self.exchange.process(snap), track)
        actions = self.engine.on_snapshot(snap, self.exchange.open_orders(mid))
        self._execute(actions, snap.ts, track)
        self._apply_fills(self.exchange.drain_fills(), track)
        self._record_calibration(snap, track)
        self._after_event(mid)

    def _on_resolved(self, ev: MarketResolved) -> None:
        mid = ev.market_id
        track = self._tracks.get(mid)
        if track is None:
            raise ValueError(f"{self._where(mid)}: MarketResolved for a market never seen")
        if mid not in self._open:
            raise ValueError(f"{self._where(mid)}: market resolved twice")
        self._n_res += 1
        self._apply_fills(self.exchange.drain_fills(), track)
        track.close = _close_of(self.engine.portfolio.inventory(mid))
        settlement = self.exchange.settle(mid, ev.winner, ev.ts)
        try:
            breakdown = self.engine.on_settlement(settlement)
        except ValueError as exc:
            raise InvariantError(
                f"invariant 1 (reconciliation) violated at {self._where(mid)}: engine rejected "
                f"the exchange settlement: {exc}"
            ) from exc
        track.winner, track.breakdown = ev.winner, breakdown
        self._settled_pnl += breakdown.total
        del self._open[mid]
        self._after_event(mid)

    def _after_event(self, market_id: str) -> None:
        if not self.check_invariants:
            return
        self._check_event(market_id)
        if self._n % FULL_SWEEP_EVERY == 0:
            self._check_full()

    def _apply_fills(self, fills: Sequence[Fill], track: _Track) -> None:
        for fill in fills:
            try:
                self.engine.on_fill(fill)
            except ValueError as exc:
                raise InvariantError(
                    f"invariant 1 (reconciliation) violated at {self._where(fill.market_id)}: "
                    f"engine rejected exchange fill {fill.fill_id}: {exc}"
                ) from exc
            self._fills.append(fill)
            self._touch(track)

    def _execute(self, actions: Sequence[Action], ts: float, track: _Track) -> None:
        for action in actions:
            if isinstance(action, CancelOrder):
                if not self.exchange.cancel(action.order_id, ts):
                    self._cancel_misses += 1
            elif isinstance(action, MergePairs):
                self._merge(action, ts, track)
            elif isinstance(action, PlaceOrder):
                self._orders_submitted += 1
                result = self.exchange.submit(action.request, ts)
                if not result.ok:
                    self._rejections[result.reason or "unknown"] += 1
            else:
                raise TypeError(f"{self._where()}: unknown action {type(action).__name__}")

    def _merge(self, action: MergePairs, ts: float, track: _Track) -> None:
        result = self.exchange.merge(action.market_id, action.size, ts)
        if result is None:
            self._merge_failures += 1
            return
        try:
            self.engine.on_merge(result)
        except ValueError as exc:
            raise InvariantError(
                f"invariant 1 (reconciliation) violated at {self._where(action.market_id)}: "
                f"engine rejected the exchange merge of {result.size:g} pairs: {exc}"
            ) from exc
        self._merges_done += 1
        track.merged_pairs += result.size
        self._touch(track)

    def _touch(self, track: _Track) -> None:
        """Update the running maxima of a market after its inventory changed."""
        inv = self.engine.portfolio.inventory(track.spec.market_id)
        up, down = inv.qty[Outcome.UP], inv.qty[Outcome.DOWN]
        track.max_inventory = max(track.max_inventory, up + down)
        track.max_net = max(track.max_net, abs(up - down))
        track.max_capital = max(track.max_capital, inv.capital_at_risk())

    def _record_calibration(self, snap: MarketSnapshot, track: _Track) -> None:
        mid = snap.market.market_id
        if self.engine.phase_of(mid) is not Phase.ACCUMULATE:
            return
        fair = self.engine.last_fair(mid)
        if fair is not None:
            track.calib = CalibSample(fair.p_up, fair.p_up_model, snap.up_book.mid, fair.valid)

    # ------------------------------------------------------------------ equity

    def equity(self) -> float:
        """Cash plus open positions marked at the last snapshot mids of their markets."""
        total = self.engine.portfolio.cash
        for mid in self._open:
            inv = self.engine.portfolio.inventories.get(mid)
            if inv is not None:
                up_mid, down_mid = self._mids.get(mid, (None, None))
                total += mark_to_mid(inv, up_mid, down_mid)
        return total

    def _sample(self) -> None:
        assert self._last_ts is not None
        self._curve.append(
            EquityPoint(self._n, self._last_ts, self.equity(), self.engine.portfolio.cash)
        )

    # ------------------------------------------------------------------ invariants

    def _violation(self, number: int, name: str, where: str, message: str) -> InvariantError:
        return InvariantError(f"invariant {number} ({name}) violated at {where}: {message}")

    def _check_event(self, market_id: str) -> None:
        """Invariants 1-3 after one event, restricted to the market it touched (O(open))."""
        where = self._where(market_id)
        pf, ex = self.engine.portfolio, self.exchange
        cash_e, cash_x = pf.cash, ex.balance()
        if abs(cash_e - cash_x) > TOL:
            raise self._violation(
                1,
                "reconciliation",
                where,
                f"engine cash {cash_e:.9g} != exchange cash {cash_x:.9g} "
                f"(diff {cash_e - cash_x:.3g})",
            )
        self._check_positions(market_id, where)
        open_invs = [pf.inventories[m] for m in self._open if m in pf.inventories]
        actual = cash_e + math.fsum(i.capital_at_risk() for i in open_invs) - self._initial
        expected = self._settled_pnl + math.fsum(i.realised_pnl for i in open_invs)
        if abs(actual - expected) > TOL:
            raise self._violation(
                2,
                "realised-pnl identity",
                where,
                f"cash + capital_at_risk - initial_cash = {actual:.9g} but realised pnl = "
                f"{expected:.9g} (diff {actual - expected:.3g})",
            )
        self._check_nonnegative(cash_e, cash_x, where)

    def _check_positions(self, market_id: str, where: str) -> None:
        inv = self.engine.portfolio.inventories.get(market_id)
        for token in Outcome:
            q_e = 0.0 if inv is None else inv.qty[token]
            q_x = self.exchange.position(market_id, token)
            if abs(q_e - q_x) > TOL:
                raise self._violation(
                    1,
                    "reconciliation",
                    where,
                    f"engine {token.value} position {q_e:.9g} != exchange {q_x:.9g}",
                )
            if q_e < -TOL or q_x < -TOL:
                raise self._violation(
                    3,
                    "non-negative",
                    where,
                    f"negative {token.value} quantity (engine {q_e:.9g}, exchange {q_x:.9g})",
                )

    def _check_nonnegative(self, cash_e: float, cash_x: float, where: str) -> None:
        if cash_e < -TOL or cash_x < -TOL:
            raise self._violation(
                3,
                "non-negative",
                where,
                f"negative cash (engine {cash_e:.9g}, exchange {cash_x:.9g})",
            )
        reserved = self.exchange.reserved_cash
        if reserved > cash_x + TOL:
            raise self._violation(
                3,
                "reserved cash",
                where,
                f"reserved_cash {reserved:.9g} exceeds cash {cash_x:.9g}",
            )

    def _check_full(self) -> None:
        """Invariants 1-3 over every market ever seen, plus order sanity."""
        where = f"full sweep after event #{self._n}"
        pf = self.engine.portfolio
        for market_id in pf.inventories:
            self._check_positions(market_id, where)
        total = math.fsum(i.realised_pnl for i in pf.inventories.values())
        if abs(pf.realised_pnl() - total) > TOL:
            raise self._violation(
                2,
                "realised-pnl identity",
                where,
                f"portfolio realised pnl {pf.realised_pnl():.9g} != sum over markets {total:.9g}",
            )
        for order in self.exchange.open_orders():
            if order.market_id not in self._open:
                raise self._violation(
                    4,
                    "no stray orders",
                    where,
                    f"open order {order.order_id} in resolved/unknown market {order.market_id!r}",
                )
            if order.remaining <= 0.0:
                raise self._violation(
                    3,
                    "non-negative",
                    where,
                    f"open order {order.order_id} has remaining {order.remaining!r}",
                )

    def _check_end(self, unresolved: Sequence[str]) -> None:
        """Invariant 4 (and a last full sweep)."""
        where = "end of stream"
        if unresolved and not self.allow_unresolved:
            raise self._violation(
                4,
                "all markets resolved",
                where,
                f"{len(unresolved)} market(s) never resolved: {list(unresolved)[:5]} "
                "(truncated recording? pass allow_unresolved=True to run anyway)",
            )
        self._check_full()
        if unresolved:
            return
        pf, ex = self.engine.portfolio, self.exchange
        orders = ex.open_orders()
        if orders:
            raise self._violation(
                4, "no open orders", where, f"{len(orders)} order(s) remain, e.g. {orders[0]}"
            )
        if abs(ex.reserved_cash) > TOL:
            raise self._violation(4, "no reserved cash", where, f"reserved {ex.reserved_cash:.9g}")
        for market_id in self._tracks:
            for token in Outcome:
                if abs(ex.position(market_id, token)) > TOL:
                    raise self._violation(
                        4,
                        "positions closed",
                        where,
                        f"exchange still holds {token.value} in {market_id!r}",
                    )
        pnl_sum = math.fsum(t.breakdown.total for t in self._tracks.values() if t.breakdown)
        gain = pf.cash - self._initial
        if abs(pnl_sum - gain) > TOL:
            raise self._violation(
                4,
                "final accounting",
                where,
                f"sum of PnLBreakdown.total {pnl_sum:.9g} != final equity - initial cash "
                f"{gain:.9g} (diff {pnl_sum - gain:.3g})",
            )

    # ------------------------------------------------------------------ result

    def _build_result(self, unresolved: Sequence[str]) -> BacktestResult:
        by_market: dict[str, list[Fill]] = {m: [] for m in self._tracks}
        for fill in self._fills:
            by_market[fill.market_id].append(fill)
        records: list[MarketRecord] = []
        for mid, track in self._tracks.items():
            inv = self.engine.portfolio.inventory(mid)
            if track.close is None:
                track.close = _close_of(inv)
            records.append(_market_record(track, inv, by_market[mid]))
        final_equity = self.equity()
        curve = list(self._curve)
        samples = [(t.calib, t.winner) for t in self._tracks.values() if t.winner is not None]
        ex_stats = dict(sorted(self.exchange.stats.items()))
        return BacktestResult(
            source_label=self.source_label,
            invariants_checked=self.check_invariants,
            equity_sample_every=self._sample_every,
            n_events=self._n,
            n_snapshots=self._n_snap,
            n_resolutions=self._n_res,
            first_ts=self._first_ts,
            last_ts=self._last_ts,
            initial_cash=self._initial,
            final_equity=final_equity,
            total_pnl=final_equity - self._initial,
            pnl_pct_of_initial_cash=100.0 * (final_equity - self._initial) / self._initial,
            kill_switch=self.engine.risk.kill_switch,
            activity=activity_stats(self._fills),
            pairs=pair_stats(records),
            pnl=pnl_stats(records, [p.equity for p in curve], self._initial),
            calibration=calibration_stats(samples),
            unresolved_markets=list(unresolved),
            markets=records,
            engine_stats=dict(sorted(self.engine.stats.items())),
            exchange_stats=ex_stats,
            runner_stats={
                "orders_submitted": self._orders_submitted,
                "order_rejections": dict(sorted(self._rejections.items())),
                "cancel_misses": self._cancel_misses,
                "merges_executed": self._merges_done,
                "merge_failures": self._merge_failures,
            },
            equity_curve=curve,
            fills_sha256=fills_digest(self._fills),
            fills=list(self._fills),
        )


def run_backtest(
    cfg: BotConfig,
    events: Iterable[FeedEvent],
    *,
    check_invariants: bool = True,
    equity_sample_every: int = 60,
    source_label: str = "unknown",
    allow_unresolved: bool = False,
) -> BacktestResult:
    """Run the engine against the paper exchange over ``events`` and return the result.

    Raises ``InvariantError`` when an invariant is violated (only with ``check_invariants``;
    the exchange's own internal assertions are always active) and ``ValueError`` for a malformed
    event stream. ``allow_unresolved`` accepts a stream that ends before some markets resolve
    (live paper mode). Deterministic: identical inputs give identical ``to_dict()`` results.
    """
    runner = BacktestRunner(
        cfg,
        check_invariants=check_invariants,
        equity_sample_every=equity_sample_every,
        source_label=source_label,
        allow_unresolved=allow_unresolved,
    )
    for event in events:
        runner.feed(event)
    return runner.finish()


__all__ = [
    "DUST_SHARES",
    "FULL_SWEEP_EVERY",
    "TOL",
    "ActivityStats",
    "BacktestResult",
    "BacktestRunner",
    "CalibSample",
    "CalibrationStats",
    "EquityPoint",
    "InvariantError",
    "MarketRecord",
    "PairStats",
    "PnLStats",
    "activity_stats",
    "calibration_stats",
    "fills_digest",
    "mark_to_mid",
    "pair_stats",
    "pnl_stats",
    "run_backtest",
]
