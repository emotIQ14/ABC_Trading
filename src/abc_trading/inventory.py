"""Per-market inventory and portfolio cash accounting (all-in average-cost basis).

Units: prices are USDC per share in [0, 1]; quantities are shares; money is USDC.

Accounting rules (DESIGN 2.1):

* Cost basis is all-in. BUY: ``qty += size``, ``cost += price * size + fee``. Average cost per
  share is ``cost / qty``.
* SELL removes cost pro-rata (average-cost method): ``removed = cost * size / qty`` and realises
  ``price * size - fee - removed``.
* A pair (1 UP + 1 DOWN) is worth exactly $1. ``apply_merge(s)`` removes ``s`` shares of each
  token at average cost and realises ``s - (removed_up + removed_down)``.
* ``apply_settlement(winner)`` pays ``qty[winner]`` and splits the result into a pair part
  ``paired * (1 - avg_up - avg_down)`` and a directional remainder defined so that the two parts
  sum to ``payout - remaining_total_cost``.

Invariants (checked by tests):

* ``cash + capital_at_risk - initial_cash == realised_pnl`` where ``realised_pnl`` is the sum over
  markets of ``sell_pnl + merge_pnl`` (plus the settlement pnl once settled).
* Quantities and costs are never negative; a position whose quantity drops below ``EPS`` is
  snapped to exactly 0 (its leftover cost is realised, so the identity above keeps holding).
* Every mutating method validates *before* it mutates: a ``ValueError`` leaves all state
  unchanged. Nonsense (negative/zero sizes, overselling, over-merging, filling or settling a
  settled market, a fill for another market, cash overdraft) raises instead of being clamped.

Average-cost accounting treats paired and unpaired shares as fungible. That is an acknowledged
approximation, exact once the market is flat or fully paired. Duplicate fills are not detected
(``Fill.fill_id`` is not tracked); idempotency is the caller's job.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from abc_trading.types import EPS, Fill, MergeResult, OpenOrder, Outcome, Settlement, Side

CASH_TOL = 1e-9  # slack allowed when comparing cash amounts and settlement payouts


def _check_unit(value: float, what: str, *, tol: float = 0.0) -> None:
    if not (math.isfinite(value) and -tol <= value <= 1.0 + tol):
        raise ValueError(f"{what} must be a finite number in [0, 1], got {value!r}")


def _check_fill(fill: Fill) -> None:
    _check_unit(fill.price, "fill price")
    if not (math.isfinite(fill.size) and fill.size > 0.0):
        raise ValueError(f"fill size must be a finite number > 0, got {fill.size!r}")
    if not math.isfinite(fill.fee):
        raise ValueError(f"fill fee must be finite, got {fill.fee!r}")


@dataclass(frozen=True, slots=True)
class PnLBreakdown:
    market_id: str
    sell_pnl: float  # realised on sells (proceeds - fee - removed cost)
    merge_pnl: float  # realised on merges (size - removed cost)
    settle_pair_pnl: float  # paired remainder at settlement: paired*(1 - cu - cd)
    settle_directional_pnl: float  # unpaired remainder: payout - cost
    fees_paid: float  # informational; already inside the pnl numbers above
    total: float  # sell + merge + settle_pair + settle_directional


class MarketInventory:
    """Quantities and all-in cost basis for the UP and DOWN tokens of one market."""

    def __init__(self, market_id: str) -> None:
        self.market_id = market_id
        self.qty: dict[Outcome, float] = {Outcome.UP: 0.0, Outcome.DOWN: 0.0}  # shares
        self.cost: dict[Outcome, float] = {Outcome.UP: 0.0, Outcome.DOWN: 0.0}  # USDC, all-in
        self.fees_paid = 0.0  # sum of fill fees (negative = net rebate)
        self.sell_pnl = 0.0
        self.merge_pnl = 0.0
        self.settle_pnl = 0.0  # settle_pair + settle_directional; 0 until settled
        self.settled = False

    def __repr__(self) -> str:
        return (
            f"MarketInventory({self.market_id!r}, up={self.qty[Outcome.UP]:g}"
            f"@{self.cost[Outcome.UP]:g}, down={self.qty[Outcome.DOWN]:g}"
            f"@{self.cost[Outcome.DOWN]:g}, settled={self.settled})"
        )

    # ------------------------------------------------------------------ queries

    @property
    def paired_qty(self) -> float:
        return min(self.qty[Outcome.UP], self.qty[Outcome.DOWN])

    @property
    def net_shares(self) -> float:
        """``qty[UP] - qty[DOWN]`` (positive = long UP)."""
        return self.qty[Outcome.UP] - self.qty[Outcome.DOWN]

    @property
    def realised_pnl(self) -> float:
        """Realised so far: sells + merges (+ settlement once settled)."""
        return self.sell_pnl + self.merge_pnl + self.settle_pnl

    def unpaired(self) -> tuple[Outcome | None, float]:
        """(heavy token, |net shares|), or ``(None, 0.0)`` when balanced within ``EPS``."""
        net = self.net_shares
        if abs(net) <= EPS:
            return None, 0.0
        return (Outcome.UP if net > 0.0 else Outcome.DOWN), abs(net)

    def avg_cost(self, token: Outcome) -> float | None:
        """All-in cost per share of the open position, or ``None`` if there is none."""
        q = self.qty[token]
        return self.cost[token] / q if q > 0.0 else None

    def locked_profit(self) -> float:
        """``paired_qty * (1 - avg_up - avg_down)``; 0 if either side is empty."""
        avg_up, avg_down = self.avg_cost(Outcome.UP), self.avg_cost(Outcome.DOWN)
        if avg_up is None or avg_down is None:
            return 0.0
        return self.paired_qty * (1.0 - avg_up - avg_down)

    def capital_at_risk(self) -> float:
        """All-in cost basis of the open positions (USDC)."""
        return self.cost[Outcome.UP] + self.cost[Outcome.DOWN]

    def value_at(self, p_up: float) -> float:
        """Mark-to-model value: ``qty[UP] * p_up + qty[DOWN] * (1 - p_up)``."""
        _check_unit(p_up, "p_up", tol=EPS)
        return self.qty[Outcome.UP] * p_up + self.qty[Outcome.DOWN] * (1.0 - p_up)

    # ------------------------------------------------------------------ mutations

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill. BUY adds qty/cost; SELL removes cost pro-rata and books sell pnl."""
        self._require_open()
        if fill.market_id != self.market_id:
            raise ValueError(f"fill for market {fill.market_id!r} applied to {self.market_id!r}")
        _check_fill(fill)
        if fill.side is Side.BUY:
            self.qty[fill.token] += fill.size
            self.cost[fill.token] += fill.price * fill.size + fill.fee
        elif fill.side is Side.SELL:
            removed = self._remove(fill.token, fill.size)
            self.sell_pnl += fill.price * fill.size - fill.fee - removed
        else:
            raise ValueError(f"unknown side {fill.side!r}")
        self.fees_paid += fill.fee

    def apply_merge(self, size: float) -> float:
        """Merge ``size`` pairs into ``$size``; returns the realised merge pnl."""
        self._require_open()
        if not (math.isfinite(size) and size > 0.0):
            raise ValueError(f"merge size must be a finite number > 0, got {size!r}")
        paired = self.paired_qty
        if paired <= 0.0 or size > paired + EPS:
            raise ValueError(f"cannot merge {size:g} pairs: only {paired:g} paired")
        removed = self._remove(Outcome.UP, size) + self._remove(Outcome.DOWN, size)
        pnl = size - removed
        self.merge_pnl += pnl
        return pnl

    def apply_settlement(self, winner: Outcome) -> PnLBreakdown:
        """Settle: pay ``$1`` per winning share, zero both positions, mark settled."""
        self._require_open()
        payout = self.qty[winner]
        total_cost = self.capital_at_risk()
        avg_up, avg_down = self.avg_cost(Outcome.UP), self.avg_cost(Outcome.DOWN)
        if avg_up is None or avg_down is None:
            pair_pnl = 0.0
        else:
            pair_pnl = self.paired_qty * (1.0 - avg_up - avg_down)
        directional_pnl = (payout - total_cost) - pair_pnl
        for token in Outcome:
            self.qty[token] = 0.0
            self.cost[token] = 0.0
        self.settle_pnl = pair_pnl + directional_pnl
        self.settled = True
        return PnLBreakdown(
            market_id=self.market_id,
            sell_pnl=self.sell_pnl,
            merge_pnl=self.merge_pnl,
            settle_pair_pnl=pair_pnl,
            settle_directional_pnl=directional_pnl,
            fees_paid=self.fees_paid,
            total=self.sell_pnl + self.merge_pnl + pair_pnl + directional_pnl,
        )

    # ------------------------------------------------------------------ internals

    def _require_open(self) -> None:
        if self.settled:
            raise ValueError(f"market {self.market_id!r} is already settled")

    def _remove(self, token: Outcome, size: float) -> float:
        """Remove ``size`` shares at average cost; return the USDC cost removed.

        Raises before mutating if the position is too small. A residual below ``EPS`` is snapped
        to exactly 0 and its leftover cost is counted as removed (so no float dust survives).
        """
        q, c = self.qty[token], self.cost[token]
        if q <= 0.0:
            raise ValueError(f"no {token.value} inventory to remove in {self.market_id!r}")
        if size > q + EPS:
            raise ValueError(f"cannot remove {size:g} {token.value}: only {q:g} held")
        remaining = q - size
        if remaining < EPS:
            self.qty[token] = 0.0
            self.cost[token] = 0.0
            return c
        new_cost = c * remaining / q
        self.qty[token] = remaining
        self.cost[token] = new_cost
        return c - new_cost


class Portfolio:
    """Cash plus one ``MarketInventory`` per market; the single source of cash truth.

    Cash moves with every fill (price * size, plus/minus the fee), merge and settlement, so
    ``realised_pnl() = cash + capital_at_risk - initial_cash`` always reconciles with the
    per-market realised pnl.
    """

    def __init__(self, initial_cash: float) -> None:
        if not (math.isfinite(initial_cash) and initial_cash >= 0.0):
            raise ValueError(f"initial_cash must be a finite number >= 0, got {initial_cash!r}")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.inventories: dict[str, MarketInventory] = {}

    def inventory(self, market_id: str) -> MarketInventory:
        """Get-or-create the inventory for ``market_id``."""
        inv = self.inventories.get(market_id)
        if inv is None:
            inv = self.inventories[market_id] = MarketInventory(market_id)
        return inv

    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill and move cash. A BUY that would overdraw cash raises (the exchange must
        have rejected it); a failed fill leaves the portfolio untouched."""
        _check_fill(fill)
        inv = self._get_or_new(fill.market_id)
        if fill.side is Side.BUY:
            delta = -(fill.price * fill.size + fill.fee)
            if not (self.cash + delta >= -CASH_TOL):
                raise ValueError(
                    f"BUY of {fill.size:g}@{fill.price:g} costs {-delta:g}, cash is {self.cash:g}"
                )
        else:
            delta = fill.price * fill.size - fill.fee
        inv.apply_fill(fill)
        self.inventories.setdefault(fill.market_id, inv)
        self.cash += delta

    def apply_merge(self, result: MergeResult) -> float:
        """Apply a merge: positions shrink, ``cash += result.cash``. Returns the merge pnl."""
        inv = self.inventories.get(result.market_id)
        if inv is None:
            raise ValueError(f"cannot merge in unknown market {result.market_id!r}")
        if not abs(result.cash - result.size) <= CASH_TOL:
            raise ValueError(f"merge cash {result.cash!r} must equal merged size {result.size!r}")
        pnl = inv.apply_merge(result.size)
        self.cash += result.cash
        return pnl

    def apply_settlement(self, s: Settlement) -> PnLBreakdown:
        """Settle a market: ``cash += s.payout``, which must equal the winning quantity held."""
        inv = self._get_or_new(s.market_id)
        if inv.settled:
            raise ValueError(f"market {s.market_id!r} is already settled")
        expected = inv.qty[s.winner]
        if not abs(s.payout - expected) <= CASH_TOL:
            raise ValueError(
                f"settlement payout {s.payout!r} != {expected!r} winning shares held "
                f"in {s.market_id!r}"
            )
        breakdown = inv.apply_settlement(s.winner)
        self.inventories.setdefault(s.market_id, inv)
        self.cash += s.payout
        return breakdown

    def _get_or_new(self, market_id: str) -> MarketInventory:
        """Existing inventory, or a fresh one that is registered only once an operation succeeds
        (so a rejected event does not leave a stray empty inventory behind)."""
        inv = self.inventories.get(market_id)
        return MarketInventory(market_id) if inv is None else inv

    def capital_at_risk(self) -> float:
        """Total all-in cost basis of open positions across markets (USDC)."""
        return sum((inv.capital_at_risk() for inv in self.inventories.values()), 0.0)

    def realised_pnl(self) -> float:
        return self.cash + self.capital_at_risk() - self.initial_cash

    def equity(self, marks: Mapping[str, float]) -> float:
        """Cash plus open positions marked at ``marks[market_id]`` (a p_up); a market without a
        mark is valued at cost. Settled markets are already in cash."""
        total = self.cash
        for market_id, inv in self.inventories.items():
            if inv.settled:
                continue
            mark = marks.get(market_id)
            total += inv.capital_at_risk() if mark is None else inv.value_at(mark)
        return total

    def reserved_cash(self, open_orders: Iterable[OpenOrder]) -> float:
        """Cash tied up by resting BUY orders: ``sum(price * remaining)`` (SELLs reserve none)."""
        total = 0.0
        for order in open_orders:
            if order.side is not Side.BUY:
                continue
            _check_unit(order.price, f"order {order.order_id!r} price")
            if not order.remaining >= -EPS:
                raise ValueError(f"order {order.order_id!r} has negative remaining")
            total += order.price * max(order.remaining, 0.0)
        return total

    def available_cash(self, open_orders: Iterable[OpenOrder]) -> float:
        """``cash - reserved_cash``; not clamped (a negative value signals over-reservation)."""
        return self.cash - self.reserved_cash(open_orders)
