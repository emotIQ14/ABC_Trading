"""Risk gate (DESIGN 2.7): kill switch, spot staleness, book sanity, rate limiter, capital caps.

Units: money is USDC, timestamps are unix seconds, spreads are in price units (tick multiples).

``RiskManager`` holds the only mutable risk state: the latched kill switch and the order-rate
window. Everything else (``spot_ok``, ``book_ok``, ``capital_ok``) is a pure check of its
arguments against ``RiskConfig``; the engine decides what a failed check means (cancel quotes,
place nothing except merges and flattening).

Kill switch
    ``update_equity(equity)`` latches the switch once ``equity - starting_equity <=
    -max_daily_loss_usd`` (with ``EPS`` slack, so it trips marginally early, never late). There is
    deliberately no way to reset it: once tripped it stays tripped for the life of the object.

Spot staleness
    ``spot_ok`` is False when the spot is missing / non-positive / non-finite, when its timestamp
    is missing (age cannot be established, so fail closed), when it lies in the future of the
    snapshot (look-ahead), or when ``snap.ts - spot_ts > max_spot_staleness_seconds``.

Book sanity
    ``book_ok`` needs, on BOTH tokens' books, at least one bid and one ask, prices inside
    [0, 1], no crossed or locked book, and ``spread <= max_ticks * tick_size``.

Rate limiter
    A sliding window of one second over order *messages*: ``allow_orders(ts, n)`` grants and
    records up to ``floor(max_orders_per_second)`` placements per window; ``record_cancels``
    records cancels, which are never refused but occupy window slots (so a burst of cancels can
    starve placements for up to a second, never the other way round). An event at ``t`` is inside
    the window of ``ts`` iff ``t > ts - 1`` (events exactly one second old have expired).
    Merges are not orders and are neither limited nor counted. Timestamps must be non-decreasing
    (``ValueError`` otherwise). ``max_orders_per_second`` below 1 is rejected, since a one-second
    window could then never admit an order.

Capital
    ``capital_ok(market_capital, total_capital, extra)`` where capital is cost basis plus cash
    reserved by open bids; ``capital_headroom`` gives the largest ``extra`` that still passes.
"""

from __future__ import annotations

import math
from collections import deque

from abc_trading.config import RiskConfig
from abc_trading.types import EPS, BookSnapshot, MarketSnapshot

_WINDOW_SECONDS = 1.0


def _book_has_sane_quotes(book: BookSnapshot) -> bool:
    if not book.bids or not book.asks:
        return False
    for price in (book.bids[0].price, book.asks[0].price):
        if not (math.isfinite(price) and -EPS <= price <= 1.0 + EPS):
            return False
    return not book.is_crossed


def books_quotable(snap: MarketSnapshot, max_spread_ticks: int) -> bool:
    """True iff both books are two-sided, sane, uncrossed and no wider than the tick limit.

    Pure function shared by ``RiskManager.book_ok`` and the quoter (which re-checks it so that
    ``compute_quotes`` is safe to call on its own).
    """
    tick = snap.market.tick_size
    for book in (snap.up_book, snap.down_book):
        if not _book_has_sane_quotes(book):
            return False
        spread = book.spread
        if spread is None or spread > max_spread_ticks * tick + EPS:
            return False
    return True


class RiskManager:
    """Latching kill switch, order-rate limiter and stateless gate checks."""

    def __init__(self, cfg: RiskConfig, starting_equity: float) -> None:
        if not math.isfinite(starting_equity):
            raise ValueError(f"starting_equity must be finite, got {starting_equity!r}")
        if not (math.isfinite(cfg.max_orders_per_second) and cfg.max_orders_per_second >= 1.0):
            raise ValueError(
                f"max_orders_per_second must be >= 1 (one-second window), "
                f"got {cfg.max_orders_per_second!r}"
            )
        self._cfg = cfg
        self._starting_equity = starting_equity
        self._kill = False
        self._kill_equity: float | None = None
        self._order_cap = math.floor(cfg.max_orders_per_second + EPS)
        self._events: deque[tuple[float, int]] = deque()  # (ts, count), oldest first
        self._in_window = 0
        self._last_ts: float | None = None

    # ------------------------------------------------------------------ kill switch

    @property
    def kill_switch(self) -> bool:
        return self._kill

    @property
    def starting_equity(self) -> float:
        return self._starting_equity

    @property
    def kill_equity(self) -> float | None:
        """Equity at the moment the kill switch latched (None while untripped)."""
        return self._kill_equity

    def update_equity(self, equity: float) -> None:
        """Feed the current equity (cash + marked positions); latches the kill switch."""
        if not math.isfinite(equity):
            raise ValueError(f"equity must be finite, got {equity!r}")
        if self._kill:
            return
        if equity - self._starting_equity <= -self._cfg.max_daily_loss_usd + EPS:
            self._kill = True
            self._kill_equity = equity

    # ------------------------------------------------------------------ stateless checks

    def spot_ok(self, snap: MarketSnapshot) -> bool:
        """Spot present, finite, positive, timestamped, not from the future, not stale."""
        spot, spot_ts = snap.spot, snap.spot_ts
        if spot is None or spot_ts is None:
            return False
        if not (math.isfinite(spot) and spot > 0.0 and math.isfinite(spot_ts)):
            return False
        age = snap.ts - spot_ts
        if age < -EPS:
            return False
        return age <= self._cfg.max_spot_staleness_seconds + EPS

    def book_ok(self, snap: MarketSnapshot, cfg_tick_limit: int) -> bool:
        """Both books two-sided, uncrossed and spread <= ``cfg_tick_limit`` ticks."""
        return books_quotable(snap, cfg_tick_limit)

    def capital_headroom(self, market_capital: float, total_capital: float) -> float:
        """Largest extra capital that keeps both caps satisfied (>= 0)."""
        _check_capital(market_capital, "market_capital")
        _check_capital(total_capital, "total_capital")
        per_market = self._cfg.max_capital_per_market_usd - market_capital
        total = self._cfg.max_total_capital_usd - total_capital
        return max(0.0, min(per_market, total))

    def capital_ok(self, market_capital: float, total_capital: float, extra: float) -> bool:
        """True iff adding ``extra`` keeps market and total capital within their caps."""
        _check_capital(market_capital, "market_capital")
        _check_capital(total_capital, "total_capital")
        _check_capital(extra, "extra")
        cfg = self._cfg
        return (
            market_capital + extra <= cfg.max_capital_per_market_usd + EPS
            and total_capital + extra <= cfg.max_total_capital_usd + EPS
        )

    # ------------------------------------------------------------------ rate limiter

    def allow_orders(self, ts: float, n: int) -> int:
        """How many of ``n`` new orders may be sent at ``ts``; the granted count is recorded."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n!r}")
        self._advance(ts)
        granted = min(n, max(0, self._order_cap - self._in_window))
        self._record(ts, granted)
        return granted

    def record_cancels(self, ts: float, n: int) -> None:
        """Record ``n`` cancels at ``ts``. Cancels are never refused, but they use window slots."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n!r}")
        self._advance(ts)
        self._record(ts, n)

    def orders_in_window(self, ts: float) -> int:
        """Messages (orders + cancels) currently inside the one-second window ending at ``ts``."""
        self._advance(ts)
        return self._in_window

    def _advance(self, ts: float) -> None:
        if not math.isfinite(ts):
            raise ValueError(f"ts must be finite, got {ts!r}")
        if self._last_ts is not None and ts < self._last_ts - EPS:
            raise ValueError(f"ts went backwards ({ts!r} < {self._last_ts!r})")
        self._last_ts = ts if self._last_ts is None else max(ts, self._last_ts)
        cutoff = self._last_ts - _WINDOW_SECONDS + EPS
        while self._events and self._events[0][0] <= cutoff:
            self._in_window -= self._events.popleft()[1]

    def _record(self, ts: float, n: int) -> None:
        if n == 0:
            return
        assert self._last_ts is not None
        self._events.append((self._last_ts, n))
        self._in_window += n


def _check_capital(value: float, what: str) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{what} must be a finite number >= 0, got {value!r}")
