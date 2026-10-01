"""Quote construction (DESIGN 2.2): a pure function from market state to a bid ladder.

Units: prices are USDC per share in (0, 1), sizes are shares (floored to 0.01), ``cash_budget`` is
USDC, ``target_net`` is signed shares (positive = long UP, see ``directional.py``). Only post-only
BUY orders are produced; exits and rebalances are the taker logic in ``rebalance.py``.

For each token ``T`` (opposite ``O``) in the order UP, DOWN:

1. Cap price (``max_bid_for_token``). If unpaired ``O`` is held (``qty[O] > qty[T] + EPS``) the
   cap is ``1 - target_margin - avg_cost[O]`` (``avg_cost`` is already all-in), so every fill locks
   at least ``target_margin`` per pair. Otherwise it is ``fair_T - target_margin / 2`` (the two caps
   then sum to ``1 - target_margin``) and needs a valid model. A positive maker fee rate divides
   the cap by ``1 + rate`` so the all-in price still respects it; a maker rebate never raises it.
2. Directional tilt. The favoured token is the one the (clamped) target asks for more of: UP iff
   ``target > 0`` and ``net < target``, DOWN iff ``target < 0`` and ``net > target``. When it is NOT
   completing pairs its cap becomes ``max(cap, fair_T - min_edge)`` (never worse than the balanced
   cap; with the default ``min_edge > target_margin / 2`` the raise is a no-op). The unfavoured
   token's size is scaled by ``1 - progress`` where ``progress = clamp(net_toward_target /
   |target|, 0, 1)``, so it is removed once the target is reached. The tilt applies only when
   ``directional.enabled`` and the model is valid; the target is clamped to
   ``+-max_directional_shares``. A completing token always keeps the strict pair cap (invariant 8).
3. Inventory skew and hard stops, measured against the deliberate target (identical to ``|net|``
   when the target is 0): the heavy side (``excess = sign_T * (net - target) > 0``) is shifted down
   by ``skew_ticks_per_100_shares * excess / 100`` ticks. Total resting size per token is limited to
   the room left under ``max_inventory_per_side_shares`` and under ``max_net_imbalance_shares`` plus
   the directional allowance in that direction (``max(0, sign_T * target)``); no room means no bid
   (the hard stop).
4. Ladder. Level ``k`` price is ``min(cap, best_bid - k * spacing * tick, best_ask - tick,
   1 - tick) - skew`` snapped DOWN to a tick (so a bid never crosses and never reaches 1). Levels
   below one tick end the ladder; a level whose price repeats an earlier level (cap binding below
   the touch) is skipped, not merged. Size is ``clip * decay**k * scale``, floored to 0.01, limited
   by the remaining room and the remaining cash, and dropped if below ``market.min_order_size``.
   Cash is allocated sequentially (UP level 0, 1, ..., then DOWN level 0, 1, ...), charging
   ``price * size`` plus any positive maker fee, so the emitted bids never cost more than
   ``cash_budget``.
5. No quotes in WARMUP / FLATTEN / DONE, with an empty / crossed / too wide book, or for a token
   with no cap (invalid model and nothing to complete). In WIND_DOWN a token is quoted only if it
   completes pairs (room limited to the unpaired amount) or is the favoured token (room limited to
   the remaining directional exposure).

Helper signatures chosen beyond DESIGN 4.3: ``plan_tokens`` / ``TokenPlan`` expose the per-token
price ceiling, room and size scale that ``compute_quotes`` uses (the engine uses the ceiling to
spot resting orders that are no longer allowed at all); ``compute_quotes`` takes an extra optional
``skip_tokens`` so the engine can exclude a token it is completing by taker order without that
token consuming the cash budget; ``floor_size`` is the shared 0.01 lot rounding.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass

from abc_trading.config import BotConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.directional import effective_target
from abc_trading.strategy.risk import books_quotable
from abc_trading.types import EPS, MarketSnapshot, Outcome, Phase, floor_price

_TOKENS: tuple[Outcome, Outcome] = (Outcome.UP, Outcome.DOWN)
_QUOTING_PHASES = frozenset({Phase.ACCUMULATE, Phase.WIND_DOWN})
_LOT = 100.0  # sizes are floored to 1 / _LOT shares


@dataclass(frozen=True, slots=True)
class QuoteLevel:
    """One desired post-only bid: ``size`` shares of ``token`` at ``price``."""

    token: Outcome
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class TokenPlan:
    """Constraints (2.2 (1)-(3)) on the bids for one token.

    ``cap`` is the uncapped-by-book cap price (None = do not bid); ``ceiling`` is the highest bid
    price allowed right now (cap minus skew, snapped down, below the best ask), None when no bid is
    allowed at all; ``room`` is the most shares that may rest on this token; ``size_scale`` in
    [0, 1] scales ladder sizes (the unfavoured token under a directional tilt); ``skew_ticks`` is
    the downward price shift in ticks.
    """

    token: Outcome
    cap: float | None
    ceiling: float | None
    room: float
    size_scale: float
    skew_ticks: float


def floor_size(size: float) -> float:
    """Floor ``size`` shares to the 0.01 lot (``EPS`` slack for float noise)."""
    return math.floor(size * _LOT + EPS) / _LOT


def _sign(token: Outcome) -> float:
    return 1.0 if token is Outcome.UP else -1.0


def _is_completing(inv: MarketInventory, token: Outcome) -> bool:
    """True iff buying ``token`` completes pairs: unpaired opposite inventory is held."""
    return inv.qty[token.opposite] > inv.qty[token] + EPS


def _is_favoured(inv: MarketInventory, token: Outcome, target: float) -> bool:
    """True iff the (effective) ``target`` asks for more ``token`` exposure than is held."""
    net = inv.net_shares
    if token is Outcome.UP:
        return target > EPS and net < target - EPS
    return target < -EPS and net > target + EPS


def _maker_fee_ratio(cfg: BotConfig) -> float:
    """Maker fee (negative = rebate) as a fraction of notional."""
    return cfg.fees.maker_fee_rate - cfg.fees.maker_rebate_rate


def max_bid_for_token(
    *,
    token: Outcome,
    fair: FairValue,
    inv: MarketInventory,
    cfg: BotConfig,
    target_net: float,
) -> float | None:
    """Cap price of 2.2 (1)+(2): never bid above it. None means do not bid this token.

    Not snapped to a tick and not limited by the book; positive and finite when not None. Raises
    ValueError on a non-finite ``target_net``.
    """
    target = effective_target(fair=fair, cfg=cfg, target_net=target_net)
    margin = cfg.pair.target_margin
    if _is_completing(inv, token):
        avg_other = inv.avg_cost(token.opposite)
        if avg_other is None:
            return None
        cap = 1.0 - margin - avg_other
    elif not fair.valid:
        return None
    else:
        p = fair.p(token)
        cap = p - margin / 2.0
        if _is_favoured(inv, token, target):
            cap = max(cap, p - cfg.directional.min_edge)
    ratio = _maker_fee_ratio(cfg)
    if ratio > 0.0:
        cap /= 1.0 + ratio
    return cap if cap > EPS else None


def _unfavoured_scale(inv: MarketInventory, token: Outcome, target: float) -> float:
    """``1 - progress`` for the token opposite to the target direction, else 1."""
    sign = _sign(token)
    if abs(target) <= EPS or sign * target > 0.0:
        return 1.0
    progress = (-sign) * inv.net_shares / abs(target)
    return 1.0 - max(0.0, min(1.0, progress))


def _room(
    inv: MarketInventory,
    token: Outcome,
    cfg: BotConfig,
    phase: Phase,
    target: float,
    *,
    completing: bool,
    favoured: bool,
) -> float:
    """Most shares that may rest on ``token`` (>= 0)."""
    sign = _sign(token)
    held, other = inv.qty[token], inv.qty[token.opposite]
    net_in_dir = held - other
    allowance = max(0.0, sign * target)
    side_room = cfg.pair.max_inventory_per_side_shares - held
    net_room = cfg.pair.max_net_imbalance_shares + allowance - net_in_dir
    room = min(side_room, net_room)
    if phase is Phase.WIND_DOWN:
        wind = 0.0
        if completing:
            wind = other - held
        if favoured:
            wind = max(wind, sign * target - net_in_dir)
        room = min(room, wind)
    return max(0.0, room)


def _plan_token(
    token: Outcome,
    snap: MarketSnapshot,
    fair: FairValue,
    inv: MarketInventory,
    phase: Phase,
    cfg: BotConfig,
    target_net: float,
) -> TokenPlan:
    blocked = TokenPlan(token, None, None, 0.0, 1.0, 0.0)
    if phase not in _QUOTING_PHASES:
        return blocked
    best_ask = snap.book(token).best_ask
    cap = max_bid_for_token(token=token, fair=fair, inv=inv, cfg=cfg, target_net=target_net)
    if best_ask is None or cap is None:
        return blocked
    target = effective_target(fair=fair, cfg=cfg, target_net=target_net)
    completing = _is_completing(inv, token)
    favoured = _is_favoured(inv, token, target)
    if phase is Phase.WIND_DOWN and not (completing or favoured):
        return blocked

    tick = snap.market.tick_size
    excess = _sign(token) * (inv.net_shares - target)
    skew_ticks = cfg.pair.skew_ticks_per_100_shares * max(0.0, excess) / 100.0
    ceiling = floor_price(min(cap, best_ask - tick, 1.0 - tick) - skew_ticks * tick, tick)
    scale = _unfavoured_scale(inv, token, target)
    room = _room(inv, token, cfg, phase, target, completing=completing, favoured=favoured)
    if ceiling < tick - EPS or room <= EPS or scale <= EPS:
        return TokenPlan(token, cap, None, 0.0, scale, skew_ticks)
    return TokenPlan(token, cap, ceiling, room, scale, skew_ticks)


def plan_tokens(
    *,
    snap: MarketSnapshot,
    fair: FairValue,
    inv: MarketInventory,
    phase: Phase,
    cfg: BotConfig,
    target_net: float,
) -> dict[Outcome, TokenPlan]:
    """Per-token constraints used by ``compute_quotes`` (UP and DOWN keys, always both)."""
    return {token: _plan_token(token, snap, fair, inv, phase, cfg, target_net) for token in _TOKENS}


def _bid_cost(fees: FeeModel, price: float, size: float) -> float:
    """Cash needed to rest and fill a bid: notional plus any positive maker fee."""
    return price * size + max(0.0, fees.fee(price, size, True))


def _fit_size(want: float, price: float, room: float, budget: float, fees: FeeModel) -> float:
    """Largest 0.01-lot size <= ``want`` and ``room`` whose cost fits ``budget`` (0 if none)."""
    size = floor_size(min(want, room))
    if size <= 0.0 or budget <= 0.0:
        return 0.0
    per_share = price + max(0.0, fees.fee(price, 1.0, True))
    size = min(size, floor_size(budget / per_share))
    for _ in range(4):  # float round-off can leave the cost a hair over the budget
        if size <= 0.0 or _bid_cost(fees, price, size) <= budget:
            return max(size, 0.0)
        size = floor_size(size - 1.0 / _LOT)
    return 0.0


def compute_quotes(
    *,
    snap: MarketSnapshot,
    fair: FairValue,
    inv: MarketInventory,
    phase: Phase,
    cfg: BotConfig,
    fees: FeeModel,
    clip_shares: float,
    cash_budget: float,
    target_net: float,
    skip_tokens: Collection[Outcome] = (),
) -> list[QuoteLevel]:
    """Desired post-only bid ladder (see the module docstring), UP then DOWN, level 0 first.

    ``clip_shares`` is the size of level 0 (before decay and scaling); ``cash_budget`` bounds the
    total cost of all returned levels. ``skip_tokens`` are not quoted and use no budget. Raises
    ValueError on non-finite or non-positive ``clip_shares``, negative or non-finite
    ``cash_budget``, or non-finite ``target_net``. Returns [] when nothing may be quoted.
    """
    if not (math.isfinite(clip_shares) and clip_shares > 0.0):
        raise ValueError(f"clip_shares must be a finite number > 0, got {clip_shares!r}")
    if not (math.isfinite(cash_budget) and cash_budget >= 0.0):
        raise ValueError(f"cash_budget must be a finite number >= 0, got {cash_budget!r}")
    if not math.isfinite(target_net):
        raise ValueError(f"target_net must be finite, got {target_net!r}")
    if phase not in _QUOTING_PHASES or not books_quotable(snap, cfg.risk.max_spread_ticks_to_quote):
        return []

    plans = plan_tokens(snap=snap, fair=fair, inv=inv, phase=phase, cfg=cfg, target_net=target_net)
    sizing = cfg.sizing
    tick = snap.market.tick_size
    min_size = snap.market.min_order_size
    budget = cash_budget
    out: list[QuoteLevel] = []
    for token in _TOKENS:
        plan = plans[token]
        book = snap.book(token)
        best_bid, best_ask = book.best_bid, book.best_ask
        if token in skip_tokens or plan.cap is None or plan.ceiling is None:
            continue
        if best_bid is None or best_ask is None:
            continue
        room = plan.room
        seen: set[float] = set()
        for k in range(sizing.ladder_levels):
            raw = min(
                plan.cap,
                best_bid - k * sizing.ladder_spacing_ticks * tick,
                best_ask - tick,
                1.0 - tick,
            )
            price = floor_price(raw - plan.skew_ticks * tick, tick)
            if price < tick - EPS:
                break  # deeper levels only get cheaper
            if price in seen:
                continue
            want = clip_shares * sizing.ladder_size_decay**k * plan.size_scale
            size = _fit_size(want, price, room, budget, fees)
            if size < min_size - EPS or size <= EPS:
                continue
            seen.add(price)
            out.append(QuoteLevel(token, price, size))
            room -= size
            budget -= _bid_cost(fees, price, size)
    return out
