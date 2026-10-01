"""Core domain types shared by every module.

FROZEN CONTRACT: other modules are written against these definitions. Change them only
deliberately, and update docs/DESIGN.md in the same commit.

Conventions
-----------
* Prices are floats in [0, 1] (USDC per share); sizes are shares (floats); money is USDC.
* Every binary market has two tokens, UP and DOWN. Exactly one pays $1 at resolution, the
  other pays $0, so one UP share + one DOWN share ("a pair") is always worth exactly $1 at
  resolution and can be merged into $1 at any time.
* Compare prices with ``EPS`` tolerance; snap prices with ``snap_price``.
* All timestamps are unix seconds (float).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

EPS = 1e-9


class Outcome(StrEnum):
    UP = "UP"
    DOWN = "DOWN"

    @property
    def opposite(self) -> Outcome:
        return Outcome.DOWN if self is Outcome.UP else Outcome.UP


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class TimeInForce(StrEnum):
    POST_ONLY = "POST_ONLY"  # maker: rejected (never filled as taker) if it would cross
    IOC = "IOC"  # taker: fill what is available within the limit, cancel the rest


class Phase(StrEnum):
    WARMUP = "WARMUP"  # window just started: no quoting yet
    ACCUMULATE = "ACCUMULATE"  # quote both sides, build pairs
    WIND_DOWN = "WIND_DOWN"  # only add the side that completes pairs / is model-favoured
    FLATTEN = "FLATTEN"  # cancel quotes, merge pairs, resolve the unpaired remainder
    DONE = "DONE"  # window over, waiting for resolution


def snap_price(price: float, tick: float) -> float:
    """Round ``price`` to the nearest multiple of ``tick`` (no clamping)."""
    return round(round(price / tick) * tick, 6)


def floor_price(price: float, tick: float) -> float:
    """Largest multiple of ``tick`` that is <= price (+EPS slack)."""
    return round(math.floor(price / tick + EPS) * tick, 6)


def ceil_price(price: float, tick: float) -> float:
    """Smallest multiple of ``tick`` that is >= price (-EPS slack)."""
    return round(math.ceil(price / tick - EPS) * tick, 6)


# --------------------------------------------------------------------------- order books


@dataclass(frozen=True, slots=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """One token's book. ``bids`` best (highest) first, ``asks`` best (lowest) first."""

    token: Outcome
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price

    @property
    def is_crossed(self) -> bool:
        return bool(self.bids and self.asks and self.bids[0].price >= self.asks[0].price - EPS)

    def bid_depth(self, levels: int = 5) -> float:
        return sum(lv.size for lv in self.bids[:levels])

    def ask_depth(self, levels: int = 5) -> float:
        return sum(lv.size for lv in self.asks[:levels])

    def imbalance(self, levels: int = 5) -> float:
        """(bid depth - ask depth) / (bid depth + ask depth) in [-1, 1]; 0 if empty."""
        b, a = self.bid_depth(levels), self.ask_depth(levels)
        tot = b + a
        return 0.0 if tot <= 0 else (b - a) / tot

    def size_at(self, side: Side, price: float) -> float:
        """Displayed size at exactly ``price`` on the bid (BUY) or ask (SELL) ladder."""
        ladder = self.bids if side is Side.BUY else self.asks
        for lv in ladder:
            if abs(lv.price - price) <= EPS:
                return lv.size
        return 0.0


# --------------------------------------------------------------------------- market data


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """Static description of one Up/Down window market."""

    market_id: str
    asset: str  # "BTC" | "ETH" | ...
    start_ts: float
    end_ts: float
    tick_size: float = 0.01
    min_order_size: float = 5.0
    up_token_id: str = ""  # exchange token ids (live only; empty in simulation)
    down_token_id: str = ""
    slug: str = ""

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


@dataclass(frozen=True, slots=True)
class Trade:
    """A public trade print on one token. ``aggressor`` is the side of the taker.

    A SELL print at price x on UP is economically the same event as a BUY print at 1-x on
    DOWN. Data sources MUST emit both mirrored prints (same size) so that consumers can treat
    each token's trade stream independently.
    """

    ts: float
    token: Outcome
    price: float
    size: float
    aggressor: Side


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Everything the strategy sees about one market at one instant."""

    ts: float
    market: MarketSpec
    up_book: BookSnapshot
    down_book: BookSnapshot
    spot: float | None = None  # underlying spot price now
    spot_ts: float | None = None  # when ``spot`` was last updated (staleness check)
    ref_price: float | None = None  # underlying price at window start ("price to beat")
    trades: tuple[Trade, ...] = ()  # public prints since the previous snapshot of this market

    @property
    def seconds_to_end(self) -> float:
        return self.market.end_ts - self.ts

    @property
    def seconds_since_start(self) -> float:
        return self.ts - self.market.start_ts

    def book(self, token: Outcome) -> BookSnapshot:
        return self.up_book if token is Outcome.UP else self.down_book


@dataclass(frozen=True, slots=True)
class MarketResolved:
    """Feed event: the market has resolved and ``winner`` pays $1 per share."""

    ts: float
    market_id: str
    winner: Outcome


FeedEvent = MarketSnapshot | MarketResolved


# --------------------------------------------------------------------------- orders & fills


@dataclass(frozen=True, slots=True)
class OrderRequest:
    client_id: str  # unique per request; engine-generated
    market_id: str
    token: Outcome
    side: Side
    price: float
    size: float
    tif: TimeInForce


@dataclass(slots=True)
class OpenOrder:
    """A resting order as reported by the exchange."""

    order_id: str
    client_id: str
    market_id: str
    token: Outcome
    side: Side
    price: float
    size: float  # original size
    remaining: float
    tif: TimeInForce
    created_ts: float
    live: bool = True  # False while still "in flight" (latency) — cannot fill yet


@dataclass(frozen=True, slots=True)
class SubmitResult:
    ok: bool
    order_id: str | None = None
    reason: str = ""  # populated when ok is False


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    market_id: str
    token: Outcome
    side: Side
    price: float
    size: float
    fee: float  # USDC charged on top of price*size (negative = maker rebate)
    is_maker: bool
    ts: float

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass(frozen=True, slots=True)
class MergeResult:
    market_id: str
    size: float  # pairs merged
    cash: float  # USDC released (== size)
    ts: float


@dataclass(frozen=True, slots=True)
class Settlement:
    market_id: str
    winner: Outcome
    ts: float
    payout: float  # USDC credited for the winning shares held


# --------------------------------------------------------------------------- engine actions


@dataclass(frozen=True, slots=True)
class PlaceOrder:
    request: OrderRequest


@dataclass(frozen=True, slots=True)
class CancelOrder:
    order_id: str


@dataclass(frozen=True, slots=True)
class MergePairs:
    market_id: str
    size: float


Action = PlaceOrder | CancelOrder | MergePairs


def fmt_actions(actions: Sequence[Action]) -> str:
    """Compact human-readable rendering, handy for logs/tests."""
    out: list[str] = []
    for a in actions:
        if isinstance(a, PlaceOrder):
            r = a.request
            out.append(f"{r.side.value} {r.token.value} {r.size:g}@{r.price:.3f} {r.tif.value}")
        elif isinstance(a, CancelOrder):
            out.append(f"CANCEL {a.order_id}")
        else:
            out.append(f"MERGE {a.market_id} {a.size:g}")
    return "; ".join(out)
