"""Directional overlay (DESIGN 2.4): how much deliberate net exposure the model justifies.

Units: probabilities and prices are in [0, 1] (USDC per share), sizes are shares, ``equity`` is
USDC. The signed ``target_net`` follows ``MarketInventory.net_shares``: positive means long UP.

``directional_target``
    For each token T the edge is ``fair_T - c_T`` where ``c_T = ask_T + taker_fee_per_share(ask_T)``
    is the all-in price of buying one share *now* (the executable, taker-side price; the target is
    a statement about value, not about how the shares will be acquired). A token qualifies iff
    ``edge >= min_edge`` (``EPS`` slack) and ``edge > 0``; with both qualifying the larger edge
    wins (ties go to UP). The target is::

        shares = min(max_directional_shares, kelly_fraction * f* * equity / c_T),
        f*     = binary_kelly(p_T, c_T) = max(0, (p_T - c_T) / (1 - c_T))

    additionally limited so that holding it cannot breach ``max_inventory_per_side_shares`` given
    the opposite token already held. It is 0 when the overlay is disabled, the model is invalid,
    ``equity <= 0``, no token qualifies, or the book has no usable ask.

``effective_target`` is the helper the quoter and the rebalancer use to turn whatever
``target_net`` they were handed into the *deliberate exposure they may act on*: 0 when the overlay
is off or the model invalid, otherwise clamped to ``[-max_directional_shares, +max_...]``.
"""

from __future__ import annotations

import math

from abc_trading.config import BotConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.types import EPS, MarketSnapshot, Outcome


def binary_kelly(p: float, price: float) -> float:
    """Kelly fraction ``max(0, (p - price) / (1 - price))`` for a binary share.

    ``p`` is the win probability and ``price`` the all-in cost of a share that pays 1 on a win;
    the result is the fraction of bankroll to stake. Both must be in [0, 1] (``ValueError``
    otherwise). A price of 1 offers no upside, so the fraction is 0.
    """
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"p must be a finite number in [0, 1], got {p!r}")
    if not (math.isfinite(price) and 0.0 <= price <= 1.0):
        raise ValueError(f"price must be a finite number in [0, 1], got {price!r}")
    if price >= 1.0:
        return 0.0
    return max(0.0, (p - price) / (1.0 - price))


def effective_target(*, fair: FairValue, cfg: BotConfig, target_net: float) -> float:
    """Deliberate net exposure (signed shares) that callers may act on.

    0 when the overlay is disabled or ``fair`` is invalid; otherwise ``target_net`` clamped to
    ``+-max_directional_shares``. Raises ValueError on a non-finite ``target_net``.
    """
    if not math.isfinite(target_net):
        raise ValueError(f"target_net must be finite, got {target_net!r}")
    if not cfg.directional.enabled or not fair.valid:
        return 0.0
    limit = cfg.directional.max_directional_shares
    return max(-limit, min(limit, target_net))


def _executable_price(snap: MarketSnapshot, token: Outcome, fees: FeeModel) -> float | None:
    """Ask plus taker fee per share for ``token``, or None if there is no usable ask."""
    ask = snap.book(token).best_ask
    if ask is None or not (math.isfinite(ask) and 0.0 < ask < 1.0):
        return None
    price = ask + fees.taker_fee_per_share(ask)
    return price if price < 1.0 else None


def _side_room(inv: MarketInventory, token: Outcome, cfg: BotConfig) -> float:
    """Net shares in ``token``'s direction that fit under ``max_inventory_per_side_shares``."""
    return cfg.pair.max_inventory_per_side_shares - inv.qty[token.opposite]


def directional_target(
    *,
    snap: MarketSnapshot,
    fair: FairValue,
    inv: MarketInventory,
    cfg: BotConfig,
    fees: FeeModel,
    equity: float,
) -> float:
    """Signed target for deliberate net shares (UP positive); see the module docstring."""
    if not math.isfinite(equity):
        raise ValueError(f"equity must be finite, got {equity!r}")
    d = cfg.directional
    if not d.enabled or not fair.valid or equity <= 0.0:
        return 0.0
    if d.max_directional_shares <= 0.0 or d.kelly_fraction <= 0.0:
        return 0.0

    best: tuple[float, Outcome, float] | None = None  # (edge, token, all-in price)
    for token in (Outcome.UP, Outcome.DOWN):
        price = _executable_price(snap, token, fees)
        if price is None:
            continue
        edge = fair.p(token) - price
        if edge <= EPS or edge + EPS < d.min_edge:
            continue
        if best is None or edge > best[0] + EPS:
            best = (edge, token, price)
    if best is None:
        return 0.0

    _, token, price = best
    p = fair.p(token)
    shares = d.kelly_fraction * binary_kelly(p, price) * equity / price
    shares = min(shares, d.max_directional_shares, _side_room(inv, token, cfg))
    if shares <= EPS:
        return 0.0
    return shares if token is Outcome.UP else -shares
