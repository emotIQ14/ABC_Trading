"""Paper exchange: deterministic, conservative matching of our orders against public data.

There is NO live trading here: no signing, keys, wallets or network. The exchange only replays
market data (books and public prints) handed to ``process`` and decides which of *our* orders
would have filled. Results validate mechanics, not profitability.

Units: prices are USDC per share in (0, 1) for orders; sizes are shares; cash is USDC; ``ts`` is
unix seconds. Comparisons use ``EPS`` (1e-9) tolerance.

Order lifecycle (DESIGN section 5)
----------------------------------
``submit`` validates, then the order is *in flight* (``OpenOrder.live=False``) for
``latency_ticks`` snapshots **of its market** and then *live*. With ``latency_ticks == 0`` it is
live at submit time. ``process(snap)`` does, in this order: (1) match already-live resting
orders against ``snap.trades``; (2) activate in-flight orders whose delay elapsed (POST_ONLY is
re-checked against ``snap``'s book, crossing means cancel + ``post_only_cancels``; the queue
position is set; IOC executes against ``snap``); (3) remember ``snap``'s books as the last known
books. Hence an order never fills from a print that predates the moment it became live.

Fill model (all conservative)
-----------------------------
* A resting order's fill price is ALWAYS its own limit price (maker). A resting BUY at ``p`` vs a
  SELL-aggressor print on the same token at ``tp``: ``tp < p`` is a through-fill (fills
  ``min(remaining, avail)``); ``tp == p`` first drains ``queue_ahead`` and the remainder fills us;
  ``tp > p`` never fills. Symmetric for resting SELLs vs BUY-aggressor prints. Through-prints do
  not reset ``queue_ahead`` (the conservative reading). A print's availability is
  ``size * trade_fill_fraction``, shared across our orders in priority order: better price
  first, then lower numeric order id. Within one order a print first drains that order's
  ``queue_ahead``; the drained part is not available to later orders (also conservative).
* ``queue_ahead = queue_ahead_fraction * displayed size at our price`` on the book the order went
  live on (0 when we improve the touch or the level is absent; 0 if no book is known yet).
* A print is invisible to an order if ``trade.ts`` is strictly before the moment the order went
  live (submit ts at latency 0, otherwise the activating snapshot's ts).
* IOC fills walk the book levels within the limit, best first. The fill price is the level price
  moved ``taker_slippage_ticks`` ticks against us but never beyond the limit. Depth consumed by
  one IOC is not available to the next one against the same snapshot (shared per-snapshot
  consumption). Taker fees come from ``FeeModel``; maker fees/rebates from the same model.
* Fills are timestamped with the snapshot ``ts`` (or the submit ``ts`` for submit-time IOC).

Cash and positions
------------------
* A BUY needs free cash: ``cash - commitments >= price*size`` (+ the worst-case fee), where a
  resting or in-flight BUY *reserves* ``price*remaining`` (``reserved_cash``). A resting BUY with
  a positive net maker fee additionally commits that fee internally, so that cash never drops
  below ``reserved_cash`` when it fills; with zero maker fee the two coincide. An IOC BUY needs
  ``price*size + fee(price, size, taker)`` at submit and is capped by affordable cash again at
  execution (a partial fill, never an overdraw).
* A SELL reserves position quantity; open SELLs can never exceed the shares held, ``merge``
  only merges shares not reserved by open SELLs.
* ``cash >= 0`` and ``cash >= reserved_cash`` and ``position >= reserved sells`` are asserted
  after every mutating call (``AssertionError`` = bug in this module).

Additions beyond DESIGN 4.4
---------------------------
* ``drain_fills()``: with ``latency_ticks == 0`` an IOC executes inside ``submit`` and the
  protocol has no way to return its fills, so they are queued and returned by the next
  ``process`` (first in the list). ``drain_fills()`` hands them over earlier. ``settle`` raises
  ``ValueError`` if the market still has undelivered fills.
* Market parameters (tick, min size) come from the market's last snapshot; before the first
  snapshot they fall back to ``cfg.sim.tick_size`` / ``cfg.sim.min_order_size``.
* ``stats`` is ``dict[str, float]`` (volumes are USDC notionals, counters stay ints) with extra
  keys ``accepted`` and ``cancelled``; ``submitted`` counts every ``submit`` call.
* ``process`` raises ``ValueError`` for malformed data, an out-of-order snapshot timestamp (equal
  is allowed) or a snapshot of a settled market, before changing any state.

Rejection reasons (``SubmitResult.reason`` and the ``rejected_<reason>`` stats keys, checked in
this order): ``market_settled``, ``invalid_size``, ``invalid_price`` (not in the open interval
(0, 1)), ``bad_tick``, ``min_size``, ``no_book`` (IOC at latency 0 before any snapshot),
``post_only_crosses``, ``insufficient_cash``, ``insufficient_position``.

Not detected: duplicate prints or duplicate snapshots (``Trade`` has no id, and equal snapshot
timestamps are legal), and a repeated ``client_id``. Feeding the same event twice would fill
twice; the data source is responsible for not doing that.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

from abc_trading.config import BotConfig
from abc_trading.fees import FeeModel
from abc_trading.types import (
    EPS,
    BookSnapshot,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    MergeResult,
    OpenOrder,
    OrderRequest,
    Outcome,
    Settlement,
    Side,
    SubmitResult,
    TimeInForce,
    Trade,
    snap_price,
)

CASH_TOL = 1e-9  # float-noise slack when comparing cash amounts
_ASSERT_TOL = 1e-7  # an invariant is broken (a bug) only beyond this

REJECT_SETTLED = "market_settled"
REJECT_SIZE = "invalid_size"
REJECT_PRICE = "invalid_price"
REJECT_TICK = "bad_tick"
REJECT_MIN_SIZE = "min_size"
REJECT_NO_BOOK = "no_book"
REJECT_POST_ONLY = "post_only_crosses"
REJECT_CASH = "insufficient_cash"
REJECT_POSITION = "insufficient_position"

_Key = tuple[str, Outcome]  # (market_id, token)


@dataclass(slots=True)
class _Order:
    """Exchange-side state of one open order (the public view is ``order``)."""

    order: OpenOrder
    seq: int  # numeric part of the order id; the priority tiebreak
    activate_at: int  # market snapshot count at which the order goes live
    live_ts: float  # prints dated before this are invisible to the order
    queue_ahead: float = 0.0  # shares ahead of us at our price (resting orders)


@dataclass(frozen=True, slots=True)
class _MarketView:
    """Last known market parameters and books."""

    spec: MarketSpec
    up: BookSnapshot
    down: BookSnapshot

    def book(self, token: Outcome) -> BookSnapshot:
        return self.up if token is Outcome.UP else self.down


# --------------------------------------------------------------------------- pure helpers


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _invariant(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"PaperExchange invariant broken: {msg}")


def _ladder(levels: Iterable[Level], *, ascending: bool) -> list[tuple[float, float]]:
    """(price, size) per distinct price with positive size; best level first."""
    agg: dict[float, float] = {}
    for lv in levels:
        if lv.size > EPS:
            agg[lv.price] = agg.get(lv.price, 0.0) + lv.size
    return sorted(agg.items(), reverse=not ascending)


def _best(levels: Iterable[Level], *, lowest: bool) -> float | None:
    prices = [lv.price for lv in levels if lv.size > EPS]
    if not prices:
        return None
    return min(prices) if lowest else max(prices)


def _crosses(side: Side, price: float, book: BookSnapshot) -> bool:
    """Would a limit order at ``price`` be marketable against ``book``?"""
    if side is Side.BUY:
        ask = _best(book.asks, lowest=True)
        return ask is not None and price >= ask - EPS
    bid = _best(book.bids, lowest=False)
    return bid is not None and price <= bid + EPS


# The validators below run on every level of every snapshot, so they test first and only build
# the (repr-heavy) message when something is wrong.


def _validate_book(book: BookSnapshot, token: Outcome) -> None:
    if book.token is not token:
        raise ValueError(f"{token.value} book slot holds a {book.token.value} book")
    for lv in (*book.bids, *book.asks):
        if not (_finite(lv.price) and 0.0 <= lv.price <= 1.0):
            raise ValueError(f"book level price must be in [0, 1], got {lv.price!r}")
        if not (_finite(lv.size) and lv.size >= 0.0):
            raise ValueError(f"book level size invalid: {lv.size!r}")


def _validate_trade(trade: Trade) -> None:
    if not _finite(trade.ts):
        raise ValueError(f"trade ts must be finite, got {trade.ts!r}")
    if not (_finite(trade.price) and 0.0 <= trade.price <= 1.0):
        raise ValueError(f"trade price must be in [0, 1], got {trade.price!r}")
    if not (_finite(trade.size) and trade.size >= 0.0):
        raise ValueError(f"trade size invalid: {trade.size!r}")


def _validate_config(cfg: BotConfig) -> None:
    x, f, sim = cfg.exchange, cfg.fees, cfg.sim
    _require(_finite(x.initial_cash) and x.initial_cash >= 0.0, "exchange.initial_cash invalid")
    _require(
        isinstance(x.latency_ticks, int) and x.latency_ticks >= 0,
        "exchange.latency_ticks must be an int >= 0",
    )
    _require(
        _finite(x.queue_ahead_fraction) and x.queue_ahead_fraction >= 0.0,
        "exchange.queue_ahead_fraction must be >= 0",
    )
    _require(
        _finite(x.trade_fill_fraction) and 0.0 <= x.trade_fill_fraction <= 1.0,
        "exchange.trade_fill_fraction must be in [0, 1]",
    )
    _require(
        isinstance(x.taker_slippage_ticks, int) and x.taker_slippage_ticks >= 0,
        "exchange.taker_slippage_ticks must be an int >= 0",
    )
    # A fee above 100% of notional could make a SELL reduce cash; we assume that never happens.
    _require(f.maker_fee_rate <= 1.0, "fees.maker_fee_rate must be <= 1 for the paper exchange")
    _require(f.taker_fee_rate <= 1.0, "fees.taker_fee_rate must be <= 1 for the paper exchange")
    _require(_finite(sim.tick_size) and sim.tick_size > 0.0, "sim.tick_size must be > 0")
    _require(
        _finite(sim.min_order_size) and sim.min_order_size >= 0.0,
        "sim.min_order_size must be >= 0",
    )


# --------------------------------------------------------------------------- the exchange


class PaperExchange:
    """Implements ``Exchange`` for backtests and paper trading (see module docstring)."""

    def __init__(self, cfg: BotConfig) -> None:
        _validate_config(cfg)
        self._xcfg = cfg.exchange
        self._fees = FeeModel(cfg.fees)
        self._fallback_tick = cfg.sim.tick_size
        self._fallback_min_size = cfg.sim.min_order_size
        self._cash = cfg.exchange.initial_cash
        self._positions: dict[_Key, float] = {}
        self._orders: dict[str, _Order] = {}
        self._snap_count: dict[str, int] = {}  # snapshots processed per market (latency clock)
        self._last_ts: dict[str, float] = {}
        self._views: dict[str, _MarketView] = {}
        # depth already taken from the last known books: market -> (token, taker side, level price)
        self._consumed: dict[str, dict[tuple[Outcome, Side, float], float]] = {}
        self._settled: set[str] = set()
        self._pending: list[Fill] = []  # fills made at submit time, not yet handed out
        self._next_order = 1
        self._next_fill = 1
        self.stats: dict[str, float] = {
            "submitted": 0,
            "accepted": 0,
            "cancelled": 0,
            "filled_maker": 0,
            "filled_taker": 0,
            "post_only_cancels": 0,
            "merges": 0,
            "merge_rejects": 0,
            "settled": 0,
            "volume_maker": 0.0,
            "volume_taker": 0.0,
        }

    # ------------------------------------------------------------------ queries

    def balance(self) -> float:
        """Free cash in USDC (reservations for resting bids are not deducted)."""
        return self._cash

    def position(self, market_id: str, token: Outcome) -> float:
        """Shares of ``token`` held in ``market_id`` (0 if none)."""
        return self._positions.get((market_id, token), 0.0)

    @property
    def reserved_cash(self) -> float:
        """Cash reserved by open BUY orders: ``sum(price * remaining)``, in flight or live."""
        return sum(
            (
                st.order.price * st.order.remaining
                for st in self._orders.values()
                if st.order.side is Side.BUY
            ),
            0.0,
        )

    def open_orders(self, market_id: str | None = None) -> list[OpenOrder]:
        """Copies of the open orders sorted by numeric order id."""
        sts = [
            st
            for st in self._orders.values()
            if market_id is None or st.order.market_id == market_id
        ]
        sts.sort(key=lambda st: st.seq)
        return [replace(st.order) for st in sts]

    def drain_fills(self) -> list[Fill]:
        """Hand out (and forget) fills made inside ``submit`` (latency-0 IOC), oldest first."""
        out, self._pending = self._pending, []
        return out

    # ------------------------------------------------------------------ order entry

    def submit(self, req: OrderRequest, ts: float) -> SubmitResult:
        """Validate and accept or reject ``req``. Never raises for a bad order."""
        _require(_finite(ts), f"ts must be finite, got {ts!r}")
        self.stats["submitted"] += 1
        reason = self._reject_reason(req)
        if reason:
            key = f"rejected_{reason}"
            self.stats[key] = self.stats.get(key, 0) + 1
            return SubmitResult(ok=False, order_id=None, reason=reason)
        return self._accept(req, ts)

    def cancel(self, order_id: str, ts: float) -> bool:
        """Cancel an open (live or in-flight) order; ``False`` if unknown or already done."""
        _require(_finite(ts), f"ts must be finite, got {ts!r}")
        if self._orders.pop(order_id, None) is None:
            return False
        self.stats["cancelled"] += 1
        self._check_invariants()
        return True

    def merge(self, market_id: str, size: float, ts: float) -> MergeResult | None:
        """Merge ``size`` pairs for ``$size``; ``None`` (and ``merge_rejects``) if impossible.

        Needs ``size`` free (not reserved by open SELLs) shares of both tokens, within ``EPS``.
        """
        _require(_finite(ts), f"ts must be finite, got {ts!r}")
        ok = (
            _finite(size)
            and size > 0.0
            and market_id not in self._settled
            and size <= min(self._free_position(market_id, t) for t in Outcome) + EPS
        )
        if not ok:
            self.stats["merge_rejects"] += 1
            return None
        for token in Outcome:
            self._add_position(market_id, token, -size)
        self._add_cash(size)
        self.stats["merges"] += 1
        self._check_invariants()
        return MergeResult(market_id=market_id, size=size, cash=size, ts=ts)

    def settle(self, market_id: str, winner: Outcome, ts: float) -> Settlement:
        """Resolve a market: cancel its open orders, pay ``$1`` per winning share, zero positions.

        Raises ``ValueError`` if already settled or if fills made at submit time for this
        market have not been delivered (``drain_fills`` / ``process``).
        """
        _require(_finite(ts), f"ts must be finite, got {ts!r}")
        _require(market_id not in self._settled, f"market {market_id!r} is already settled")
        _require(
            all(f.market_id != market_id for f in self._pending),
            f"market {market_id!r} has undelivered fills; call drain_fills() or process() first",
        )
        for oid in [oid for oid, st in self._orders.items() if st.order.market_id == market_id]:
            del self._orders[oid]
        payout = self.position(market_id, winner)  # $1 per winning share
        for token in Outcome:
            self._positions.pop((market_id, token), None)
        self._add_cash(payout)
        self._settled.add(market_id)
        self._views.pop(market_id, None)
        self._consumed.pop(market_id, None)
        self.stats["settled"] += 1
        self._check_invariants()
        return Settlement(market_id=market_id, winner=winner, ts=ts, payout=payout)

    # ------------------------------------------------------------------ market data

    def process(self, snap: MarketSnapshot) -> list[Fill]:
        """Advance to ``snap``; return fills (submit-time fills first, then prints, then IOC)."""
        self._validate_snapshot(snap)
        mid = snap.market.market_id
        fills = self.drain_fills()
        self._snap_count[mid] = self._snap_count.get(mid, 0) + 1
        self._last_ts[mid] = snap.ts
        self._consumed[mid] = {}  # fresh books, fresh depth
        fills.extend(self._match_prints(snap))
        fills.extend(self._activate(snap))
        self._views[mid] = _MarketView(spec=snap.market, up=snap.up_book, down=snap.down_book)
        self._check_invariants()
        return fills

    def _validate_snapshot(self, snap: MarketSnapshot) -> None:
        mid = snap.market.market_id
        _require(_finite(snap.ts), f"snapshot ts must be finite, got {snap.ts!r}")
        _require(mid not in self._settled, f"snapshot for settled market {mid!r}")
        last = self._last_ts.get(mid)
        _require(
            last is None or snap.ts >= last,
            f"snapshot ts {snap.ts!r} is before the previous one {last!r} for {mid!r}",
        )
        spec = snap.market
        _require(_finite(spec.tick_size) and spec.tick_size > 0.0, "market tick_size must be > 0")
        _require(
            _finite(spec.min_order_size) and spec.min_order_size >= 0.0,
            "market min_order_size must be >= 0",
        )
        _validate_book(snap.up_book, Outcome.UP)
        _validate_book(snap.down_book, Outcome.DOWN)
        for trade in snap.trades:
            _validate_trade(trade)

    def _match_prints(self, snap: MarketSnapshot) -> list[Fill]:
        """Step 1: fill already-live resting orders from this snapshot's prints, in list order."""
        mid = snap.market.market_id
        frac = self._xcfg.trade_fill_fraction
        fills: list[Fill] = []
        for trade in snap.trades:
            avail = trade.size * frac
            if avail <= EPS:
                continue
            side = trade.aggressor.opposite  # the side of ours this print can hit
            sts = [
                st
                for st in self._orders.values()
                if st.order.market_id == mid
                and st.order.live
                and st.order.tif is TimeInForce.POST_ONLY
                and st.order.token is trade.token
                and st.order.side is side
                and trade.ts >= st.live_ts
            ]
            sts.sort(
                key=lambda st: (-st.order.price if side is Side.BUY else st.order.price, st.seq)
            )
            for st in sts:
                if avail <= EPS:
                    break
                avail, fill = self._fill_from_print(st, trade, avail, snap.ts)
                if fill is not None:
                    fills.append(fill)
        return fills

    def _fill_from_print(
        self, st: _Order, trade: Trade, avail: float, ts: float
    ) -> tuple[float, Fill | None]:
        """Apply one print to one resting order; returns (availability left, fill or None)."""
        o = st.order
        if o.side is Side.BUY:
            if trade.price > o.price + EPS:  # print above our bid: not for us
                return avail, None
            through = trade.price < o.price - EPS
        else:
            if trade.price < o.price - EPS:
                return avail, None
            through = trade.price > o.price + EPS
        if not through:  # at our price: the queue ahead of us trades first
            drained = min(st.queue_ahead, avail)
            st.queue_ahead -= drained
            avail -= drained
        qty = min(o.remaining, avail)
        if qty <= EPS:
            return avail, None
        return avail - qty, self._book_fill(o, o.price, qty, is_maker=True, ts=ts)

    def _activate(self, snap: MarketSnapshot) -> list[Fill]:
        """Step 2: in-flight orders whose delay elapsed go live (POST_ONLY) or execute (IOC)."""
        mid = snap.market.market_id
        count = self._snap_count[mid]
        due = [
            st
            for st in self._orders.values()
            if st.order.market_id == mid and not st.order.live and st.activate_at <= count
        ]
        due.sort(key=lambda st: st.seq)
        fills: list[Fill] = []
        for st in due:
            o = st.order
            book = snap.book(o.token)
            if o.tif is TimeInForce.IOC:
                fills.extend(self._execute_ioc(st, book, snap.market.tick_size, snap.ts))
                self._orders.pop(o.order_id, None)  # a full fill already removed it
            elif _crosses(o.side, o.price, book):
                del self._orders[o.order_id]
                self.stats["post_only_cancels"] += 1
            else:
                o.live = True
                st.live_ts = snap.ts
                st.queue_ahead = self._queue_ahead(o, book)
        return fills

    # ------------------------------------------------------------------ submit internals

    def _market_params(self, market_id: str) -> tuple[float, float]:
        view = self._views.get(market_id)
        if view is None:
            return self._fallback_tick, self._fallback_min_size
        return view.spec.tick_size, view.spec.min_order_size

    def _reject_reason(self, req: OrderRequest) -> str:
        """Why ``req`` must be rejected, or ``""`` if it is acceptable."""
        mid, size, price = req.market_id, req.size, req.price
        if mid in self._settled:
            return REJECT_SETTLED
        if not (_finite(size) and size > 0.0):
            return REJECT_SIZE
        if not (_finite(price) and 0.0 < price < 1.0):
            return REJECT_PRICE
        tick, min_size = self._market_params(mid)
        if abs(price - snap_price(price, tick)) > EPS:
            return REJECT_TICK
        if self._xcfg.enforce_min_order_size and size < min_size - EPS:
            return REJECT_MIN_SIZE
        view = self._views.get(mid)
        if req.tif is TimeInForce.IOC and self._xcfg.latency_ticks == 0 and view is None:
            return REJECT_NO_BOOK
        if (
            req.tif is TimeInForce.POST_ONLY
            and view is not None
            and _crosses(req.side, price, view.book(req.token))
        ):
            return REJECT_POST_ONLY
        return self._funds_reason(req)

    def _funds_reason(self, req: OrderRequest) -> str:
        if req.side is Side.SELL:
            free = self._free_position(req.market_id, req.token)
            return REJECT_POSITION if req.size > free + EPS else ""
        if req.tif is TimeInForce.IOC:
            fee = self._fees.fee(req.price, req.size, False)
        else:
            fee = max(0.0, self._fees.fee(req.price, req.size, True))
        need = req.price * req.size + fee
        return REJECT_CASH if need > self._cash - self._committed() + CASH_TOL else ""

    def _accept(self, req: OrderRequest, ts: float) -> SubmitResult:
        tick, _ = self._market_params(req.market_id)
        seq = self._next_order
        self._next_order += 1
        order = OpenOrder(
            order_id=f"o{seq}",
            client_id=req.client_id,
            market_id=req.market_id,
            token=req.token,
            side=req.side,
            price=snap_price(req.price, tick),
            size=req.size,
            remaining=req.size,
            tif=req.tif,
            created_ts=ts,
            live=False,
        )
        latency = self._xcfg.latency_ticks
        st = _Order(
            order=order,
            seq=seq,
            activate_at=self._snap_count.get(req.market_id, 0) + latency,
            live_ts=ts,
        )
        self.stats["accepted"] += 1
        view = self._views.get(req.market_id)
        if latency == 0 and order.tif is TimeInForce.IOC:
            assert view is not None  # guaranteed by _reject_reason
            order.live = True
            self._pending.extend(self._execute_ioc(st, view.book(order.token), tick, ts))
        else:
            self._orders[order.order_id] = st
            if latency == 0:
                order.live = True
                if view is not None:
                    st.queue_ahead = self._queue_ahead(order, view.book(order.token))
        self._check_invariants()
        return SubmitResult(ok=True, order_id=order.order_id)

    def _queue_ahead(self, order: OpenOrder, book: BookSnapshot) -> float:
        return self._xcfg.queue_ahead_fraction * book.size_at(order.side, order.price)

    # ------------------------------------------------------------------ IOC execution

    def _execute_ioc(self, st: _Order, book: BookSnapshot, tick: float, ts: float) -> list[Fill]:
        """Walk ``book`` for an IOC order; the unfilled rest is simply dropped by the caller.

        BUY is limited by affordable cash (cash minus what the other open BUYs commit), SELL by
        the shares not reserved by other open SELLs. Depth taken here stays taken for the rest
        of this snapshot.
        """
        o = st.order
        is_buy = o.side is Side.BUY
        levels = _ladder(book.asks if is_buy else book.bids, ascending=is_buy)
        consumed = self._consumed.setdefault(o.market_id, {})
        slip = self._xcfg.taker_slippage_ticks * tick
        budget = (
            self._cash - self._committed(skip=o.order_id)
            if is_buy
            else self._free_position(o.market_id, o.token, skip=o.order_id)
        )
        fills: list[Fill] = []
        for level_price, level_size in levels:
            if o.remaining <= EPS:
                break
            if is_buy:
                if level_price > o.price + EPS:
                    break
                price = min(level_price + slip, o.price)
            else:
                if level_price < o.price - EPS:
                    break
                price = max(level_price - slip, o.price)
            price = round(price, 9)
            key = (o.token, o.side, level_price)
            qty = min(o.remaining, level_size - consumed.get(key, 0.0))
            if qty <= EPS:  # level used up by earlier IOCs on this snapshot
                continue
            qty = min(qty, self._affordable(is_buy, price, budget))
            if qty <= EPS:  # out of cash / shares
                break
            fill = self._book_fill(o, price, qty, is_maker=False, ts=ts)
            consumed[key] = consumed.get(key, 0.0) + qty
            budget -= (fill.price * fill.size + fill.fee) if is_buy else fill.size
            fills.append(fill)
        return fills

    def _affordable(self, is_buy: bool, price: float, budget: float) -> float:
        """Largest size we can still trade at ``price`` within ``budget`` (cash or shares)."""
        if not is_buy:
            return budget
        unit = price + self._fees.taker_fee_per_share(price)
        return budget / unit if unit > 0.0 else math.inf

    # ------------------------------------------------------------------ accounting

    def _book_fill(
        self, o: OpenOrder, price: float, qty: float, *, is_maker: bool, ts: float
    ) -> Fill:
        """Book a fill of ``qty`` at ``price`` against ``o``: cash, position, order, stats."""
        fee = self._fees.fee(price, qty, is_maker)
        if o.side is Side.BUY:
            self._add_cash(-(price * qty + fee))
            self._add_position(o.market_id, o.token, qty)
        else:
            self._add_cash(price * qty - fee)
            self._add_position(o.market_id, o.token, -qty)
        o.remaining -= qty
        if o.remaining <= EPS:
            o.remaining = 0.0
            self._orders.pop(o.order_id, None)
        self.stats["filled_maker" if is_maker else "filled_taker"] += 1
        self.stats["volume_maker" if is_maker else "volume_taker"] += price * qty
        fill = Fill(
            fill_id=f"f{self._next_fill}",
            order_id=o.order_id,
            market_id=o.market_id,
            token=o.token,
            side=o.side,
            price=price,
            size=qty,
            fee=fee,
            is_maker=is_maker,
            ts=ts,
        )
        self._next_fill += 1
        return fill

    def _add_cash(self, delta: float) -> None:
        self._cash += delta
        if -CASH_TOL <= self._cash < 0.0:  # float noise, not an overdraft
            self._cash = 0.0

    def _add_position(self, market_id: str, token: Outcome, delta: float) -> None:
        new = self.position(market_id, token) + delta
        _invariant(new >= -_ASSERT_TOL, f"negative {token.value} position in {market_id!r}")
        if abs(new) < EPS:
            self._positions.pop((market_id, token), None)
        else:
            self._positions[(market_id, token)] = new

    def _commitment(self, o: OpenOrder) -> float:
        """Cash an open order claims: ``price*remaining`` (+ a positive maker fee if resting)."""
        if o.side is not Side.BUY:
            return 0.0
        total = o.price * o.remaining
        if o.tif is TimeInForce.POST_ONLY:
            total += max(0.0, self._fees.fee(o.price, o.remaining, True))
        return total

    def _committed(self, skip: str | None = None) -> float:
        return sum(
            (self._commitment(st.order) for oid, st in self._orders.items() if oid != skip), 0.0
        )

    def _free_position(self, market_id: str, token: Outcome, skip: str | None = None) -> float:
        """Shares held minus the remaining size of open SELLs (other than ``skip``)."""
        reserved = sum(
            (
                st.order.remaining
                for oid, st in self._orders.items()
                if oid != skip
                and st.order.side is Side.SELL
                and st.order.market_id == market_id
                and st.order.token is token
            ),
            0.0,
        )
        return self.position(market_id, token) - reserved

    def _check_invariants(self) -> None:
        """Assert the module's own invariants (``AssertionError`` means a bug in this module).

        One pass over the open orders: this runs after every mutating call, so it is the hot
        spot of a backtest, and building the failure messages only on failure matters too.
        """
        cash = self._cash
        if cash < -_ASSERT_TOL:
            raise AssertionError(f"PaperExchange invariant broken: negative cash {cash!r}")
        committed = reserved = 0.0
        sells: dict[_Key, float] = {}
        for st in self._orders.values():
            o = st.order
            if o.remaining <= 0.0:
                raise AssertionError(
                    f"PaperExchange invariant broken: open order {o.order_id} has no remaining size"
                )
            if o.side is Side.BUY:
                reserved += o.price * o.remaining
                committed += self._commitment(o)
            else:
                key = (o.market_id, o.token)
                sells[key] = sells.get(key, 0.0) + o.remaining
        if committed > cash + _ASSERT_TOL:
            raise AssertionError(
                f"PaperExchange invariant broken: commitments {committed!r} exceed cash {cash!r}"
            )
        if reserved > cash + _ASSERT_TOL:
            raise AssertionError(
                f"PaperExchange invariant broken: reserved cash {reserved!r} exceeds cash {cash!r}"
            )
        for key, qty in self._positions.items():
            if qty < 0.0:
                raise AssertionError(f"PaperExchange invariant broken: negative position {key!r}")
        for key, held in sells.items():
            if held > self._positions.get(key, 0.0) + _ASSERT_TOL:
                raise AssertionError(
                    f"PaperExchange invariant broken: open sells {held!r} exceed the position "
                    f"in {key!r}"
                )
