"""Strategy engine (DESIGN 2.2-2.8, 4.3): one object driving every market through its phases.

The engine is a pure function of the events it is fed: ``on_snapshot`` (returns ``Action``s),
``on_fill``, ``on_merge`` and ``on_settlement``. It never talks to an exchange, has no clock and
no randomness. Client ids are the deterministic sequence ``c1``, ``c2``, ...

Per snapshot (``on_snapshot``)
    1. Ignore the snapshot if the market is settled, if its ts is not greater than the last
       processed ts of that market, or if it is older than any snapshot already processed (the
       model and the rate limiter need non-decreasing time; feeds emit in global time order).
    2. Observe the spot into the model (at ``spot_ts`` when present, else ``snap.ts``; once per
       (asset, ts); a spot stamped in the future is never observed).
    3. Phase (``phase_for``) and fair value. The fair value is computed before the risk gate so the
       kill switch marks equity with this snapshot's mark. When the spot is stale the fair value is
       treated as invalid for every decision (``last_fair`` still returns the raw model output).
    4. Risk gate: kill switch, spot staleness, book sanity, per-market / total capital. Any trip
       cancels the market's resting quotes and places nothing except merges and FLATTEN sells.
    5. WARMUP / DONE: cancel any resting orders, nothing else. FLATTEN: cancel everything, merge
       all pairs, then ``plan_flatten`` (hold or IOC sell).
    6. Merge when ``paired >= merge_min_pairs``.
    7. ``directional_target`` (needs equity), then ``plan_completion``; a token bought by taker
       order this tick is not quoted this tick.
    8. ``compute_quotes`` against the cash budget below, then reconcile (2.8).

Action order within one snapshot: cancels, merge, taker orders (completion / flatten sells), then
new quotes. Merging does not change the engine's own books until ``on_merge`` is called with the
exchange's result, so everything else in the same snapshot sees the pre-merge inventory.

Cash and capital accounting
    ``open_orders`` passed to ``on_snapshot`` belong to the market being processed; orders of other
    markets are ignored. Capital is cost basis plus the cash reserved by open BUY orders
    (``price * remaining``, plus the worst-case taker fee for IOC orders, which is charged on top
    when they execute). The engine keeps, per market, its *belief* of that reservation: refreshed
    from ``open_orders`` at each of the market's snapshots, lowered by BUY fills (and by the fee of
    taker fills), and set after each snapshot to (orders not cancelled) + (orders placed). For
    market M, ``R_other`` is the sum of the beliefs of all other markets. A belief can only be too
    high between a market's snapshots (a fill converts reservation into cost basis), which is the
    safe direction; it is dropped at settlement.

    The quote budget is NOT ``portfolio.available_cash(open_orders)``: M's own resting quotes are
    re-planned as a whole each snapshot (kept orders hold at most the cash of the desired order they
    match), so counting their reservation again would shrink the budget every tick and make a
    cash-limited ladder flip-flop between "all" and "nothing". Only IOC orders still in flight
    (``pinned``; never re-planned) are subtracted::

        quote_budget = min(cash - R_other - pinned,
                           max_capital_per_market - cost_M - pinned,
                           max_total_capital - cost_total - R_other - pinned)
                       - all-in cost of the taker completion planned this tick - 1e-6

    which bounds the total reservation of M's desired ladder. The taker completion is sized
    against the stricter ``cash - R_other - R_own`` (it runs before stale quotes are cancelled).
    Kept orders have price <= and remaining <= those of the desired order they match, so the
    budget carries over to the reconciled book. With a real exchange this gave zero rejections and
    never exceeded either capital cap at any latency in a randomised end-to-end sweep.

Reconcile (2.8)
    An existing resting order is kept iff it matches a desired level: same token, BUY, price within
    ``requote_tolerance_ticks`` BELOW (never above) the desired price, and remaining size between
    80% and 100% of the desired size. Never keeping an order above the desired price or larger than
    desired is a deliberate tightening of "within tolerance / within 20%": it keeps every resting
    order inside the cap, room and cash limits computed this tick. Unmatched orders are cancelled
    first, then missing levels are placed. A requote throttle of ``min_requote_interval_seconds``
    per market suppresses discretionary changes; cancels of orders that are no longer allowed at
    all (price above the current ceiling, or the token not quotable) bypass it. Placements are
    limited by the rate limiter, level-major (every token's level 0 before any level 1); cancels
    are never limited but count toward the limit.

Stats (``stats``) are monotone integer counters: ``snapshots``, ``snapshots_ignored``,
``quotes_placed`` / ``quotes_cancelled`` (orders emitted, not exchange acknowledgements),
``merges`` (MergePairs emitted), ``taker_completions``, ``flatten_sells``, ``holds`` (markets
whose FLATTEN decision held shares to resolution, counted once per market), ``fills``,
``settlements``, ``requotes_throttled``, ``orders_rate_limited`` (placements dropped by the rate
limiter), ``risk_blocked_snapshots`` and ``risk_trips_<kind>`` for kind in ``kill_switch``
(counted once, when it latches), ``spot_stale``, ``book``, ``capital`` (counted when the condition
newly appears for a market).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from abc_trading.config import BotConfig
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory, PnLBreakdown, Portfolio
from abc_trading.model.fair_value import FairValue, FairValueModel
from abc_trading.strategy.directional import directional_target
from abc_trading.strategy.phases import phase_for
from abc_trading.strategy.quoter import QuoteLevel, TokenPlan, compute_quotes, plan_tokens
from abc_trading.strategy.rebalance import pairs_to_merge, plan_completion, plan_flatten
from abc_trading.strategy.risk import RiskManager
from abc_trading.types import (
    EPS,
    Action,
    CancelOrder,
    Fill,
    MarketSnapshot,
    MergePairs,
    MergeResult,
    OpenOrder,
    OrderRequest,
    Outcome,
    Phase,
    PlaceOrder,
    Settlement,
    Side,
    TimeInForce,
)

_BUDGET_SLACK = 1e-6  # USDC kept back so float round-off never trips an exchange cash check
_SIZE_KEEP_FRACTION = 0.8  # a resting order is kept if remaining >= 80% of the desired size

_STAT_KEYS = (
    "snapshots",
    "snapshots_ignored",
    "quotes_placed",
    "quotes_cancelled",
    "merges",
    "taker_completions",
    "flatten_sells",
    "holds",
    "fills",
    "settlements",
    "requotes_throttled",
    "orders_rate_limited",
    "risk_blocked_snapshots",
    "risk_trips_kill_switch",
    "risk_trips_spot_stale",
    "risk_trips_book",
    "risk_trips_capital",
)


@dataclass(slots=True)
class _MarketState:
    last_ts: float | None = None
    last_requote_ts: float | None = None
    reserved: float = 0.0  # believed cash reserved by this market's open bids
    tripped: set[str] = field(default_factory=set)
    hold_counted: bool = False


@dataclass(slots=True)
class _Reconciled:
    cancels: list[OpenOrder]
    places: list[QuoteLevel]
    suppressed: bool  # True iff the requote throttle held back a discretionary change


@dataclass(frozen=True, slots=True)
class _Ledger:
    """Cash and capital position seen by one market at the start of its snapshot (USDC).

    ``reserved_own`` is the capital tied up by this market's open BUY orders (as passed in, with
    the worst-case taker fee on IOC orders), ``pinned`` the part of it held by IOC orders (they are
    never re-planned, so their reservation cannot be reused by the new ladder), and
    ``reserved_other`` the engine's belief of the same quantity in all other markets. See the
    module docstring for why the two budgets below differ.
    """

    cash: float
    cost_market: float
    cost_total: float
    reserved_own: float
    pinned: float
    reserved_other: float

    def taker_budget(self, risk: RiskManager) -> float:
        """Cash a taker order may use *now*: own resting bids stay reserved until cancelled."""
        free = self.cash - self.reserved_other - self.reserved_own
        headroom = risk.capital_headroom(
            self.cost_market + self.reserved_own,
            self.cost_total + self.reserved_other + self.reserved_own,
        )
        return max(0.0, min(free, headroom) - _BUDGET_SLACK)

    def quote_budget(self, risk: RiskManager, committed: float) -> float:
        """Total reservation allowed for the market's whole desired ladder, after ``committed``
        USDC already set aside for a taker order planned this snapshot."""
        headroom = risk.capital_headroom(
            self.cost_market + self.pinned,
            self.cost_total + self.reserved_other + self.pinned,
        )
        free = self.cash - self.reserved_other - self.pinned
        return max(0.0, min(free, headroom) - committed - _BUDGET_SLACK)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class MarketMakerEngine:
    """Two-sided market maker with pair accumulation, merging and rebalancing (see module doc)."""

    def __init__(self, cfg: BotConfig, *, initial_cash: float | None = None) -> None:
        cfg.validate()
        cash = cfg.exchange.initial_cash if initial_cash is None else initial_cash
        self.cfg = cfg
        self.portfolio = Portfolio(cash)
        self.fees = FeeModel(cfg.fees)
        self.model = FairValueModel(cfg.model)
        self.risk = RiskManager(cfg.risk, cash)
        self.stats: dict[str, int] = dict.fromkeys(_STAT_KEYS, 0)
        self._states: dict[str, _MarketState] = {}
        self._settled: set[str] = set()
        self._marks: dict[str, float] = {}
        self._fair: dict[str, FairValue] = {}
        self._phase: dict[str, Phase] = {}
        self._last_obs: dict[str, float] = {}  # asset -> ts of the newest observed spot
        self._clock: float | None = None
        self._next_id = 1
        self._kill_counted = False

    # ------------------------------------------------------------------ read-only views

    def marks(self) -> dict[str, float]:
        """Last p_up mark per unsettled market (the model p_up, else the market mid)."""
        return dict(self._marks)

    def equity(self) -> float:
        return self.portfolio.equity(self._marks)

    def last_fair(self, market_id: str) -> FairValue | None:
        """Raw model output at the market's last processed snapshot (kept after settlement)."""
        return self._fair.get(market_id)

    def phase_of(self, market_id: str) -> Phase | None:
        """Phase at the market's last processed snapshot (kept after settlement)."""
        return self._phase.get(market_id)

    # ------------------------------------------------------------------ events

    def on_fill(self, fill: Fill) -> None:
        """Apply a fill to the portfolio. Raises ValueError for a settled market or bad fill."""
        self.portfolio.apply_fill(fill)
        self._bump("fills")
        st = self._states.get(fill.market_id)
        if st is not None and fill.side is Side.BUY:
            # A taker fill also releases the worst-case fee that was set aside for its IOC order.
            released = fill.price * fill.size + (0.0 if fill.is_maker else max(fill.fee, 0.0))
            st.reserved = max(0.0, st.reserved - released)

    def on_merge(self, result: MergeResult) -> None:
        self.portfolio.apply_merge(result)

    def on_settlement(self, s: Settlement) -> PnLBreakdown:
        """Settle the market in the portfolio and clear its engine state (marks, throttles)."""
        breakdown = self.portfolio.apply_settlement(s)
        self._settled.add(s.market_id)
        self._states.pop(s.market_id, None)
        self._marks.pop(s.market_id, None)
        self._bump("settlements")
        return breakdown

    def on_snapshot(self, snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]:
        """Process one snapshot; returns the actions to execute, in order."""
        market_id, ts = snap.market.market_id, snap.ts
        if not math.isfinite(ts):
            raise ValueError(f"snapshot ts must be finite, got {ts!r}")
        st = self._states.get(market_id)
        if market_id in self._settled or self._is_stale(st, ts):
            self._bump("snapshots_ignored")
            return []
        if st is None:
            st = self._states[market_id] = _MarketState()
        st.last_ts = ts
        self._clock = ts
        self._bump("snapshots")

        self._observe_spot(snap)
        raw_fair = self.model.estimate(snap)
        self._fair[market_id] = raw_fair
        self._marks[market_id] = self._mark_for(snap, raw_fair)
        phase = phase_for(ts, snap.market, self.cfg.timing)
        self._phase[market_id] = phase

        orders = [o for o in open_orders if o.market_id == market_id]
        resting = [o for o in orders if o.tif is not TimeInForce.IOC]
        inv = self.portfolio.inventory(market_id)
        r_own = self._reserved_capital(orders)
        r_other = self._reserved_elsewhere(market_id)

        self.risk.update_equity(self.equity())
        tripped = self._evaluate_gate(snap, st, inv, r_own, r_other)
        fair = (
            raw_fair if "spot_stale" not in tripped else dataclasses.replace(raw_fair, valid=False)
        )

        cancel_actions: list[CancelOrder] = []
        merges: list[Action] = []
        takers: list[PlaceOrder] = []
        places: list[PlaceOrder] = []

        # Cancels are recorded in the rate limiter before any placement is granted, so they use
        # window slots first (they themselves are never refused).
        if phase in (Phase.WARMUP, Phase.DONE):
            cancel_actions = self._emit_cancels(resting, ts)
        elif phase is Phase.FLATTEN:
            cancel_actions = self._emit_cancels(resting, ts)
            self._plan_flatten(snap, st, inv, fair, orders, merges, takers)
        else:
            self._plan_merge(inv, merges)
            if tripped:
                cancel_actions = self._emit_cancels(resting, ts)
            else:
                ledger = _Ledger(
                    self.portfolio.cash,
                    inv.capital_at_risk(),
                    self.portfolio.capital_at_risk(),
                    r_own,
                    self._reserved_capital([o for o in orders if o.tif is TimeInForce.IOC]),
                    r_other,
                )
                cancel_actions, takers, places = self._plan_trading(
                    snap, st, inv, fair, phase, orders, resting, ledger
                )

        actions: list[Action] = [*cancel_actions, *merges, *takers, *places]
        cancelled_ids = {c.order_id for c in cancel_actions}
        survivors = [o for o in orders if o.order_id not in cancelled_ids]
        st.reserved = self._reserved_capital(survivors) + self._reserved_capital(
            [self._as_open_order(p.request) for p in (*takers, *places)]
        )
        return actions

    # ------------------------------------------------------------------ gate

    def _is_stale(self, st: _MarketState | None, ts: float) -> bool:
        if st is not None and st.last_ts is not None and ts <= st.last_ts:
            return True
        return self._clock is not None and ts < self._clock

    def _evaluate_gate(
        self,
        snap: MarketSnapshot,
        st: _MarketState,
        inv: MarketInventory,
        r_own: float,
        r_other: float,
    ) -> set[str]:
        """Risk gate kinds currently tripped for this market; counts newly appearing trips."""
        kinds: set[str] = set()
        if self.risk.kill_switch:
            kinds.add("kill_switch")
        if not self.risk.spot_ok(snap):
            kinds.add("spot_stale")
        if not self.risk.book_ok(snap, self.cfg.risk.max_spread_ticks_to_quote):
            kinds.add("book")
        market_capital = inv.capital_at_risk() + r_own
        total_capital = self.portfolio.capital_at_risk() + r_other + r_own
        if not self.risk.capital_ok(market_capital, total_capital, 0.0):
            kinds.add("capital")

        if "kill_switch" in kinds and not self._kill_counted:
            self._kill_counted = True
            self._bump("risk_trips_kill_switch")
        for kind in sorted(kinds - st.tripped - {"kill_switch"}):
            self._bump(f"risk_trips_{kind}")
        st.tripped = kinds
        if kinds:
            self._bump("risk_blocked_snapshots")
        return kinds

    # ------------------------------------------------------------------ planning

    def _plan_merge(self, inv: MarketInventory, merges: list[Action]) -> None:
        size = pairs_to_merge(inv, self.cfg, flatten=False)
        if size > 0.0:
            merges.append(MergePairs(inv.market_id, size))
            self._bump("merges")

    def _plan_flatten(
        self,
        snap: MarketSnapshot,
        st: _MarketState,
        inv: MarketInventory,
        fair: FairValue,
        orders: Sequence[OpenOrder],
        merges: list[Action],
        takers: list[PlaceOrder],
    ) -> None:
        pending: dict[Outcome, float] = {}
        for o in orders:
            if o.tif is TimeInForce.IOC and o.side is Side.SELL:
                pending[o.token] = pending.get(o.token, 0.0) + o.remaining
        plan = plan_flatten(
            snap=snap, inv=inv, fair=fair, cfg=self.cfg, fees=self.fees, pending_sells=pending
        )
        if plan.merge_size > 0.0:
            merges.append(MergePairs(inv.market_id, plan.merge_size))
            self._bump("merges")
        for sell in plan.sells:
            if self.risk.allow_orders(snap.ts, 1) == 0:
                self._bump("orders_rate_limited")
                continue
            takers.append(
                self._order(inv.market_id, sell.token, Side.SELL, sell.price, sell.size, True)
            )
            self._bump("flatten_sells")
        if plan.hold_size > EPS and not st.hold_counted:
            st.hold_counted = True
            self._bump("holds")

    def _plan_trading(
        self,
        snap: MarketSnapshot,
        st: _MarketState,
        inv: MarketInventory,
        fair: FairValue,
        phase: Phase,
        orders: Sequence[OpenOrder],
        resting: Sequence[OpenOrder],
        ledger: _Ledger,
    ) -> tuple[list[CancelOrder], list[PlaceOrder], list[PlaceOrder]]:
        """ACCUMULATE / WIND_DOWN with a clean gate: returns (cancels, takers, places)."""
        cfg, ts = self.cfg, snap.ts
        equity = self.equity()
        target = directional_target(
            snap=snap, fair=fair, inv=inv, cfg=cfg, fees=self.fees, equity=equity
        )
        takers, completed, committed = self._plan_completion(
            snap, inv, fair, orders, ledger, target
        )
        skip = () if completed is None else (completed,)
        desired = compute_quotes(
            snap=snap,
            fair=fair,
            inv=inv,
            phase=phase,
            cfg=cfg,
            fees=self.fees,
            clip_shares=self._clip_shares(snap, equity),
            cash_budget=ledger.quote_budget(self.risk, committed),
            target_net=target,
            skip_tokens=skip,
        )
        plans = plan_tokens(snap=snap, fair=fair, inv=inv, phase=phase, cfg=cfg, target_net=target)
        rec = self._reconcile(st, snap, resting, desired, plans, skip)
        if rec.suppressed:
            self._bump("requotes_throttled")
        cancel_actions = self._emit_cancels(rec.cancels, ts)
        places = self._place_quotes(inv.market_id, rec, ts)
        if cancel_actions or places:
            st.last_requote_ts = ts
        return cancel_actions, takers, places

    def _plan_completion(
        self,
        snap: MarketSnapshot,
        inv: MarketInventory,
        fair: FairValue,
        orders: Sequence[OpenOrder],
        ledger: _Ledger,
        target: float,
    ) -> tuple[list[PlaceOrder], Outcome | None, float]:
        """Taker completion orders granted by the rate limiter: (orders, token bought, USDC cost
        incl. fee). Nothing is planned while an IOC buy of this market is still in flight."""
        if any(o.tif is TimeInForce.IOC and o.side is Side.BUY for o in orders):
            return [], None, 0.0
        takers: list[PlaceOrder] = []
        completed: Outcome | None = None
        cost = 0.0
        for plan in plan_completion(
            snap=snap,
            inv=inv,
            fair=fair,
            cfg=self.cfg,
            fees=self.fees,
            cash_budget=ledger.taker_budget(self.risk),
            target_net=target,
        ):
            if self.risk.allow_orders(snap.ts, 1) == 0:
                self._bump("orders_rate_limited")
                continue
            takers.append(
                self._order(inv.market_id, plan.token, Side.BUY, plan.price, plan.size, True)
            )
            completed = plan.token
            cost += plan.size * (plan.price + self.fees.taker_fee_per_share(plan.price))
            self._bump("taker_completions")
        return takers, completed, cost

    # ------------------------------------------------------------------ reconcile

    def _reconcile(
        self,
        st: _MarketState,
        snap: MarketSnapshot,
        resting: Sequence[OpenOrder],
        desired: Sequence[QuoteLevel],
        plans: dict[Outcome, TokenPlan],
        skip: Sequence[Outcome],
    ) -> _Reconciled:
        tol = self.cfg.sizing.requote_tolerance_ticks * snap.market.tick_size
        unmatched = list(resting)
        missing: list[QuoteLevel] = []
        for q in desired:
            best: OpenOrder | None = None
            best_key: tuple[float, float, str] | None = None
            for o in unmatched:
                if o.token is not q.token or o.side is not Side.BUY:
                    continue
                if not (q.price - tol - EPS <= o.price <= q.price + EPS):
                    continue
                if not (_SIZE_KEEP_FRACTION * q.size - EPS <= o.remaining <= q.size + EPS):
                    continue
                key = (q.price - o.price, abs(q.size - o.remaining), o.order_id)
                if best_key is None or key < best_key:
                    best, best_key = o, key
            if best is None:
                missing.append(q)
            else:
                unmatched.remove(best)

        ts = snap.ts
        interval = self.cfg.timing.min_requote_interval_seconds
        throttled = st.last_requote_ts is not None and ts - st.last_requote_ts + EPS < interval
        if not throttled:
            return _Reconciled(unmatched, missing, False)
        urgent = [o for o in unmatched if self._is_disallowed(o, plans, skip)]
        suppressed = bool(missing) or len(urgent) < len(unmatched)
        return _Reconciled(urgent, [], suppressed)

    @staticmethod
    def _is_disallowed(
        o: OpenOrder, plans: dict[Outcome, TokenPlan], skip: Sequence[Outcome]
    ) -> bool:
        """True iff the order may not rest at all under the current plan (above the ceiling)."""
        if o.side is not Side.BUY or o.token in skip:
            return True
        ceiling = plans[o.token].ceiling
        return ceiling is None or o.price > ceiling + EPS

    def _place_quotes(self, market_id: str, rec: _Reconciled, ts: float) -> list[PlaceOrder]:
        """Rate-limit the missing levels (level-major priority) and build their orders."""
        if not rec.places:
            return []
        rank: dict[Outcome, int] = {}
        keyed: list[tuple[int, int, int]] = []  # (level rank, token order, index)
        for i, q in enumerate(rec.places):
            level = rank.get(q.token, 0)
            rank[q.token] = level + 1
            keyed.append((level, 0 if q.token is Outcome.UP else 1, i))
        granted = self.risk.allow_orders(ts, len(rec.places))
        allowed = {i for _, _, i in sorted(keyed)[:granted]}
        dropped = len(rec.places) - granted
        if dropped:
            self._bump("orders_rate_limited", dropped)
        out: list[PlaceOrder] = []
        for i, q in enumerate(rec.places):
            if i in allowed:
                out.append(self._order(market_id, q.token, Side.BUY, q.price, q.size, False))
                self._bump("quotes_placed")
        return out

    # ------------------------------------------------------------------ helpers

    def _emit_cancels(self, cancels: Sequence[OpenOrder], ts: float) -> list[CancelOrder]:
        if not cancels:
            return []
        self.risk.record_cancels(ts, len(cancels))
        self._bump("quotes_cancelled", len(cancels))
        return [CancelOrder(o.order_id) for o in cancels]

    def _order(
        self, market_id: str, token: Outcome, side: Side, price: float, size: float, ioc: bool
    ) -> PlaceOrder:
        client_id = f"c{self._next_id}"
        self._next_id += 1
        tif = TimeInForce.IOC if ioc else TimeInForce.POST_ONLY
        return PlaceOrder(OrderRequest(client_id, market_id, token, side, price, size, tif))

    def _bump(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def _reserved_capital(self, orders: Sequence[OpenOrder]) -> float:
        """Capital tied up by BUY orders: ``price * remaining`` (what the exchange reserves) plus
        the worst-case taker fee of IOC orders, which is charged on top when they execute."""
        total = self.portfolio.reserved_cash(orders)  # also validates prices and sizes
        for o in orders:
            if o.side is Side.BUY and o.tif is TimeInForce.IOC:
                total += max(0.0, self.fees.fee(o.price, max(o.remaining, 0.0), False))
        return total

    @staticmethod
    def _as_open_order(req: OrderRequest) -> OpenOrder:
        """A just-emitted request viewed as the open order it will become (for reservations)."""
        return OpenOrder(
            "", req.client_id, req.market_id, req.token, req.side, req.price, req.size, req.size,
            req.tif, 0.0,
        )  # fmt: skip

    def _reserved_elsewhere(self, market_id: str) -> float:
        return sum(s.reserved for k, s in self._states.items() if k != market_id)

    def _observe_spot(self, snap: MarketSnapshot) -> None:
        spot = snap.spot
        if spot is None or not (math.isfinite(spot) and spot > 0.0):
            return
        t = snap.ts if snap.spot_ts is None else snap.spot_ts
        if not math.isfinite(t) or t > snap.ts:
            return
        asset = snap.market.asset
        last = self._last_obs.get(asset)
        if last is not None and t <= last:
            return
        self.model.observe(asset, t, spot)
        self._last_obs[asset] = t

    def _mark_for(self, snap: MarketSnapshot, fair: FairValue) -> float:
        """p_up mark: the model's when valid, else the UP mid, else 1 - DOWN mid, else the last."""
        if fair.valid:
            return _clamp01(fair.p_up)
        up_mid = snap.up_book.mid
        if up_mid is not None:
            return _clamp01(up_mid)
        down_mid = snap.down_book.mid
        if down_mid is not None:
            return _clamp01(1.0 - down_mid)
        return self._marks.get(snap.market.market_id, _clamp01(fair.p_up))

    def _clip_shares(self, snap: MarketSnapshot, equity: float) -> float:
        """Level-0 size: ``clip_shares``, or compounding ``equity * fraction / price`` clamped.

        With ``clip_equity_fraction`` the price is the DEARER token's mid, so no level-0 order
        costs more than ``equity * fraction`` and both tokens share one clip (pairs need equal
        share counts on the two sides).
        """
        s = self.cfg.sizing
        if s.clip_equity_fraction is None:
            return s.clip_shares
        mids = [m for m in (snap.up_book.mid, snap.down_book.mid) if m is not None]
        price = max(max(mids, default=0.5), snap.market.tick_size)
        clip = max(equity, 0.0) * s.clip_equity_fraction / price
        return max(s.min_clip_shares, min(s.max_clip_shares, clip))
