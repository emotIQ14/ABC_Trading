"""The exchange protocol: the only surface the engine and the backtest runner talk to.

``PaperExchange`` (``exchange/paper.py``) is the only implementation in this project. There is
deliberately NO live adapter, no order signing, no keys and no wallet: the protocol exists so the
engine is written against an interface, not so that real money can be moved.

Units: prices are USDC per share in [0, 1]; sizes are shares; cash is USDC; ``ts`` is unix seconds.

Contract (every implementation must honour it):

* ``submit`` never raises for a bad *order*; it returns ``SubmitResult(ok=False, reason=...)``.
  An accepted order gets a unique ``order_id``.
* ``cancel`` is a message to the venue like an order, so it takes the same time to land: it returns
  ``True`` iff the request was accepted (the order is open, live or in flight, and no cancel for it
  is pending yet). An implementation with a non-zero order latency must apply the same latency to
  the cancel: until it lands the order can still fill, so a quote is not safe from prints just
  because the strategy has already decided to drop it. A cancel never overtakes the order it
  targets. ``PaperExchange`` documents the details (funds, stats, ``pending_cancels``).
* ``merge`` returns ``None`` when it cannot merge ``size`` pairs, otherwise a ``MergeResult``.
* ``open_orders`` returns copies, so callers can never mutate exchange state. Orders whose cancel
  was accepted but has not landed are not reported: the caller already asked for them to go.
* ``process`` feeds one market snapshot to the exchange and returns the fills it caused. Fills
  never come from market data older than the moment the order became live (no look-ahead).
* ``balance`` is free cash in USDC (reservations for resting bids are *not* deducted from it);
  ``position`` is shares held of one token.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from abc_trading.types import (
    Fill,
    MarketSnapshot,
    MergeResult,
    OpenOrder,
    OrderRequest,
    Outcome,
    SubmitResult,
)


@runtime_checkable
class Exchange(Protocol):
    """Order entry, matching and custody of cash and positions."""

    def submit(self, req: OrderRequest, ts: float) -> SubmitResult:
        """Validate and accept (or reject) ``req`` at time ``ts``."""
        ...

    def cancel(self, order_id: str, ts: float) -> bool:
        """Ask for an open order to be cancelled; it lands after the order latency.

        ``False`` if the order is unknown, already filled/cancelled or already being cancelled.
        """
        ...

    def merge(self, market_id: str, size: float, ts: float) -> MergeResult | None:
        """Merge ``size`` UP+DOWN pairs into ``$size``; ``None`` if not possible."""
        ...

    def open_orders(self, market_id: str | None = None) -> list[OpenOrder]:
        """Copies of the open orders, without those being cancelled (all markets when ``None``)."""
        ...

    def process(self, snap: MarketSnapshot) -> list[Fill]:
        """Advance the exchange to ``snap`` and return the fills it produced, in order."""
        ...

    def balance(self) -> float:
        """Free cash in USDC."""
        ...

    def position(self, market_id: str, token: Outcome) -> float:
        """Shares of ``token`` held in ``market_id``."""
        ...
