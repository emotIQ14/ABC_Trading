"""Taker rebalancing and end-of-window flattening (DESIGN 2.3, 2.5, 2.6).

Units: prices USDC per share in [0, 1], sizes shares (floored to 0.01), ``cash_budget`` USDC.
The planners are pure and return small frozen dataclasses, not ``Action`` objects; the engine
converts them into IOC ``PlaceOrder`` / ``MergePairs`` and enforces rate limits.

Merge (``pairs_to_merge``)
    Outside FLATTEN merge when ``paired_qty >= merge_min_pairs``; in FLATTEN merge any whole pair.
    The size is ``floor(paired)`` (``EPS`` slack), never more than ``paired`` itself, so a float
    dust shortfall such as 99.9999999995 merges exactly what is held. 0 when merging is disabled.

Completion (``plan_completion``, 2.5)
    The heavy token ``T`` holds ``U = |net|`` unpaired shares. Only the part that is not deliberate
    directional exposure is acted on: ``excess = U - max(0, sign_T * target)`` with ``target`` the
    clamped target (0 when the overlay is off or the model invalid). Nothing happens unless
    ``excess >= rebalance_trigger_shares``; "not in a directional hold" is therefore read as "the
    imbalance beyond the deliberate allowance is large". The planner buys the opposite token ``O``
    at its best ask (IOC) iff the net pair profit per share
    ``1 - avg_cost[T] - (ask_O + taker_fee_per_share(ask_O))`` is at least
    ``taker_lock_margin - rebalance_max_loss_per_pair``. Size is the floor (0.01) of
    ``min(excess, displayed ask size, cash_budget / all-in unit cost)``; below
    ``market.min_order_size`` there is no order. With the default config this only ever completes
    pairs at a net profit.

Flatten (``plan_flatten``, 2.6)
    Merge all pairs, then resolve the unpaired remainder ``U`` of the heavy token. The remainder
    is HELD to resolution (up to ``max_directional_shares``) iff the overlay is enabled, the model
    valid and ``p_T - (bid_T - taker_fee_per_share(bid_T)) >= hold_margin`` (held value beats the
    net sale proceeds); everything else is sold by IOC at the best bid. With no usable bid (empty,
    <= 0 or crossed book) nothing is sold and everything is held (``reason == "no_bid"``). A sale
    smaller than ``market.min_order_size`` is not placed and the dust is held. ``pending_sells`` are
    shares already offered by in-flight IOC sells; they are subtracted so a slow exchange is never
    asked to sell the same shares twice. The sell size is not capped by the displayed bid size (the
    IOC takes what the limit allows; the next snapshot re-plans the rest).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from abc_trading.config import BotConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.directional import effective_target
from abc_trading.strategy.quoter import floor_size
from abc_trading.types import EPS, MarketSnapshot, Outcome

# FlattenPlan.reason values
REASON_FLAT = "flat"  # nothing unpaired
REASON_SELL = "sell"  # the remainder is sold
REASON_EDGE = "edge"  # part or all of the remainder is held because the model says so
REASON_NO_BID = "no_bid"  # no usable bid: nothing can be sold, everything is held
REASON_BELOW_MIN = "below_min_size"  # the sale would be under min_order_size: dust is held


@dataclass(frozen=True, slots=True)
class CompletionOrder:
    """IOC BUY of ``size`` shares of ``token`` with limit ``price`` (the best ask)."""

    token: Outcome
    price: float
    size: float
    net_profit_per_share: float  # 1 - avg_cost[heavy] - (price + taker fee per share)


@dataclass(frozen=True, slots=True)
class SellOrder:
    """IOC SELL of ``size`` shares of ``token`` with limit ``price`` (the best bid)."""

    token: Outcome
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class FlattenPlan:
    """What to do at FLATTEN: merge ``merge_size`` pairs, send ``sells``, hold ``hold_size``.

    ``hold_size`` is the unpaired quantity expected to remain after the planned (and pending)
    sells; ``hold_token`` is the heavy token (None when nothing is unpaired).
    """

    merge_size: float
    sells: tuple[SellOrder, ...]
    hold_token: Outcome | None
    hold_size: float
    reason: str


def pairs_to_merge(inv: MarketInventory, cfg: BotConfig, *, flatten: bool) -> float:
    """Pairs to merge now (0 if none); see the module docstring."""
    if not cfg.pair.merge_enabled:
        return 0.0
    paired = inv.paired_qty
    if not flatten and paired < cfg.pair.merge_min_pairs - EPS:
        return 0.0
    size = min(float(math.floor(paired + EPS)), paired)
    return size if size >= 1.0 - EPS else 0.0


def _check_budget(cash_budget: float) -> None:
    if not (math.isfinite(cash_budget) and cash_budget >= 0.0):
        raise ValueError(f"cash_budget must be a finite number >= 0, got {cash_budget!r}")


def plan_completion(
    *,
    snap: MarketSnapshot,
    inv: MarketInventory,
    fair: FairValue,
    cfg: BotConfig,
    fees: FeeModel,
    cash_budget: float,
    target_net: float,
) -> list[CompletionOrder]:
    """At most one IOC order that completes pairs by crossing the spread (empty = do nothing)."""
    _check_budget(cash_budget)
    target = effective_target(fair=fair, cfg=cfg, target_net=target_net)
    heavy, unpaired = inv.unpaired()
    if heavy is None:
        return []
    sign = 1.0 if heavy is Outcome.UP else -1.0
    excess = unpaired - max(0.0, sign * target)
    if excess <= EPS or excess < cfg.pair.rebalance_trigger_shares - EPS:
        return []

    buy = heavy.opposite
    asks = snap.book(buy).asks
    avg_heavy = inv.avg_cost(heavy)
    if not asks or avg_heavy is None:
        return []
    ask, displayed = asks[0].price, asks[0].size
    if not (math.isfinite(ask) and 0.0 < ask < 1.0):
        return []
    unit_cost = ask + fees.taker_fee_per_share(ask)
    profit = 1.0 - avg_heavy - unit_cost
    if profit < cfg.pair.taker_lock_margin - cfg.pair.rebalance_max_loss_per_pair - EPS:
        return []

    size = floor_size(min(excess, displayed, cash_budget / unit_cost))
    while size > 0.0 and size * unit_cost > cash_budget:  # float round-off guard
        size = floor_size(size - 0.01)
    if size < snap.market.min_order_size - EPS or size <= EPS:
        return []
    return [CompletionOrder(buy, ask, size, profit)]


def plan_flatten(
    *,
    snap: MarketSnapshot,
    inv: MarketInventory,
    fair: FairValue,
    cfg: BotConfig,
    fees: FeeModel,
    pending_sells: Mapping[Outcome, float] | None = None,
) -> FlattenPlan:
    """Merge / sell / hold decision for the FLATTEN phase; see the module docstring."""
    merge = pairs_to_merge(inv, cfg, flatten=True)
    heavy, unpaired = inv.unpaired()
    if heavy is None:
        return FlattenPlan(merge, (), None, 0.0, REASON_FLAT)

    book = snap.book(heavy)
    bid = book.best_bid
    if bid is None or not (math.isfinite(bid) and bid > EPS) or book.is_crossed:
        return FlattenPlan(merge, (), heavy, unpaired, REASON_NO_BID)

    d = cfg.directional
    hold_ok = (
        d.enabled
        and fair.valid
        and fair.p(heavy) - (bid - fees.taker_fee_per_share(min(bid, 1.0))) >= d.hold_margin - EPS
    )
    keep = min(unpaired, d.max_directional_shares) if hold_ok else 0.0
    already = 0.0 if pending_sells is None else max(0.0, pending_sells.get(heavy, 0.0))
    to_sell = floor_size(max(0.0, unpaired - keep - already))
    remaining = unpaired - already

    if to_sell <= EPS:
        reason = REASON_EDGE if keep > EPS else REASON_SELL
        return FlattenPlan(merge, (), heavy, max(0.0, remaining), reason)
    if to_sell < snap.market.min_order_size - EPS:
        return FlattenPlan(merge, (), heavy, unpaired, REASON_BELOW_MIN)
    reason = REASON_EDGE if keep > EPS else REASON_SELL
    return FlattenPlan(
        merge, (SellOrder(heavy, bid, to_sell),), heavy, max(0.0, remaining - to_sell), reason
    )
