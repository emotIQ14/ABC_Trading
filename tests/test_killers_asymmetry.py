"""UP / DOWN asymmetry: the blind spot that mirrored books and UP-only tests left open.

Most fixtures of this suite are built from one UP book and its exact mirror (0.49 / 0.51 on both
tokens), a zero-imbalance DOWN book, a UP-heavy inventory or a UP-favoured target. In such a state
"the UP book", "the DOWN book" and "one minus the other" are the same numbers, so a one-line change
that swaps a token, ignores a token's book or treats only the UP direction specially is invisible.
A mutation run found eleven of them that the whole suite let through. Where each one is pinned now:

* spread limit applied to one book only (risk.py), order-book feature that ignores the DOWN book
  (fair_value.py): ``test_strategy_risk.py``, ``test_model.py``;
* Brier "market" scored with the DOWN mid, equity marked crosswise, pair cost and completion rate of
  the pairs held at close (runner.py): ``test_backtest_runner.py``;
* completion allowance honoured for UP only (rebalance.py), in-flight FLATTEN sells of DOWN not
  recognised and compounding clip priced at the UP mid whichever token is dearer (engine.py),
  WIND_DOWN room of a DOWN-favoured token (quoter.py), latency-0 queue position read from the UP
  book (paper.py): here.

This file has two layers.

1. A directed test per rule, parametrised over BOTH tokens, on books that do not mirror each other,
   with the arithmetic in a comment.
2. A systemic check: the strategy is label-symmetric. Swap the UP and DOWN labels of a whole state
   (books, prints, inventory, orders, fair value, target) and every planner must return the swapped
   answer (DESIGN 2.2(4) says so for the quoter: "With the UP and DOWN labels swapped the quotes are
   therefore swapped too"). Randomised states cover every phase, both signs of the directional
   target, budgets that bind and books that are wide on one token only. A rule applied to one token
   only, or a constant written for UP, breaks the equivalence for some state.

Also here: the tick helpers of the frozen contract (``types.ceil_price`` was neither used nor
tested). Everything is mechanics: no number here is tuned for, or says anything about, profit.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from abc_trading.config import (
    BotConfig,
    DirectionalConfig,
    FeeConfig,
    ModelConfig,
    PairConfig,
    PaperExchangeConfig,
    RiskConfig,
    SizingConfig,
)
from abc_trading.exchange.paper import PaperExchange
from abc_trading.fees import FeeModel
from abc_trading.inventory import MarketInventory
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.strategy.quoter import QuoteLevel, compute_quotes
from abc_trading.strategy.rebalance import FlattenPlan, plan_completion, plan_flatten
from abc_trading.strategy.risk import RiskManager, books_quotable
from abc_trading.types import (
    Action,
    BookSnapshot,
    CancelOrder,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    MergePairs,
    OpenOrder,
    OrderRequest,
    Outcome,
    Phase,
    PlaceOrder,
    Side,
    TimeInForce,
    Trade,
    ceil_price,
    floor_price,
    snap_price,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
BUY, SELL = Side.BUY, Side.SELL
POST, IOC = TimeInForce.POST_ONLY, TimeInForce.IOC
MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)  # tick 0.01, min order size 5
ACC_TS, WD_TS, FL_TS = 1030.0, 1820.0, 1880.0  # default timing: ACCUMULATE, WIND_DOWN, FLATTEN
TOKENS = pytest.mark.parametrize("token", [UP, DOWN], ids=["UP", "DOWN"])


# --------------------------------------------------------------------------- builders


def book(token: Outcome, bid: float, ask: float, size: float = 400.0) -> BookSnapshot:
    return BookSnapshot(token, (Level(bid, size),), (Level(ask, size),))


def snap(
    ts: float = 1100.0,
    up: tuple[float, float] = (0.49, 0.51),
    down: tuple[float, float] = (0.49, 0.51),
) -> MarketSnapshot:
    """One 400-share level per side; ``up`` and ``down`` are (bid, ask) and need not mirror."""
    return MarketSnapshot(
        ts, MKT, book(UP, *up), book(DOWN, *down), spot=100.0, spot_ts=ts, ref_price=100.0
    )


def fv(p_up: float, *, valid: bool = True) -> FairValue:
    return FairValue(p_up, p_up, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, valid)


def held(
    up: float = 0.0, down: float = 0.0, up_px: float = 0.47, down_px: float = 0.52
) -> MarketInventory:
    inv = MarketInventory("m1")
    for token, qty, px in ((UP, up, up_px), (DOWN, down, down_px)):
        if qty > 0.0:
            inv.apply_fill(Fill("f", "o", "m1", token, BUY, px, qty, 0.0, True, 1.0))
    return inv


def held_of(token: Outcome, qty: float, px: float = 0.47) -> MarketInventory:
    return held(up=qty, up_px=px) if token is UP else held(down=qty, down_px=px)


def p_up_favouring(token: Outcome) -> float:
    return 0.7 if token is UP else 0.3


class StubModel:
    """A fixed fair value in place of the real model (valid, ``p_up`` as given)."""

    def __init__(self, p_up: float = 0.5) -> None:
        self.p_up = p_up

    def observe(self, asset: str, ts: float, spot: float) -> None:
        pass

    def estimate(self, snap: MarketSnapshot) -> FairValue:
        return fv(self.p_up)


def engine_with(
    p_up: float = 0.5,
    cfg: BotConfig | None = None,
    *,
    cash: float | None = None,
    up: float = 0.0,
    down: float = 0.0,
) -> MarketMakerEngine:
    """An engine whose model says ``p_up`` and that holds ``up`` / ``down`` shares (bought at 0.47,
    out of ``cash``, which must cover them: it is the account's starting cash)."""
    cfg = cfg or BotConfig(model=ModelConfig(shrink_to_market=0.0))
    engine = MarketMakerEngine(cfg, initial_cash=cash)
    engine.model = StubModel(p_up)  # type: ignore[assignment]
    for token, qty in ((UP, up), (DOWN, down)):
        if qty > 0.0:
            fill = Fill(f"f{token.value}", "o", "m1", token, BUY, 0.47, qty, 0.0, True, 1.0)
            engine.on_fill(fill)
    return engine


def places(actions: Sequence[Action]) -> list[OrderRequest]:
    return [a.request for a in actions if isinstance(a, PlaceOrder)]


# --------------------------------------------------------------------------- label swap


def swap_book(b: BookSnapshot) -> BookSnapshot:
    return BookSnapshot(b.token.opposite, b.bids, b.asks)


def swap_snap(s: MarketSnapshot) -> MarketSnapshot:
    """The same market with the UP and DOWN labels exchanged: each token keeps its own book and
    its own prints (prices are the token's own), only the names change."""
    return replace(
        s,
        up_book=swap_book(s.down_book),
        down_book=swap_book(s.up_book),
        trades=tuple(replace(t, token=t.token.opposite) for t in s.trades),
    )


def swap_fair(f: FairValue) -> FairValue:
    return FairValue(
        1.0 - f.p_up, 1.0 - f.p_up_model, -f.z, -f.z_adj, f.sigma, f.tau,
        -f.momentum, -f.accel, -f.book_imbalance, f.valid,
    )  # fmt: skip


def swap_inv(inv: MarketInventory) -> MarketInventory:
    out = MarketInventory(inv.market_id)
    for token in (UP, DOWN):
        out.qty[token.opposite], out.cost[token.opposite] = inv.qty[token], inv.cost[token]
    return out


def swap_order(o: OpenOrder) -> OpenOrder:
    return replace(o, token=o.token.opposite)


# =========================================================================== directed tests


# ---- rebalance.py: the taker completion leaves the deliberate directional allowance alone


@TOKENS
def test_completion_leaves_a_deliberate_position_in_either_direction_alone(token: Outcome) -> None:
    # 150 unpaired shares of ``token`` (bought at 0.47, so completing at the 0.51 ask would lock
    # 1 - 0.47 - 0.518 = 1.2c) and a deliberate target of 100 in that token's direction (UP is +,
    # DOWN is -): excess = 150 - 100 = 50 < rebalance_trigger_shares 100 -> nothing to do.
    cfg = BotConfig()
    sign = 1.0 if token is UP else -1.0

    def plan(target: float) -> list[tuple[Outcome, float]]:
        out = plan_completion(
            snap=snap(), inv=held_of(token, 150.0), fair=fv(p_up_favouring(token)), cfg=cfg,
            fees=FeeModel(cfg.fees), cash_budget=1e4, target_net=target,
        )  # fmt: skip
        return [(o.token, o.size) for o in out]

    assert plan(sign * 100.0) == []  # deliberate: kept
    assert plan(sign * 49.0) == [(token.opposite, 101.0)]  # excess 101 >= trigger: completed
    # a target in the OTHER direction is no allowance for these shares (max(0, sign * target))
    assert plan(-sign * 100.0) == [(token.opposite, 150.0)]
    assert plan(0.0) == [(token.opposite, 150.0)]


@TOKENS
def test_completion_buys_only_the_excess_over_the_directional_allowance(token: Outcome) -> None:
    # 250 held, deliberate 100 in its direction -> complete 250 - 100 = 150 with the other token
    cfg = BotConfig()
    sign = 1.0 if token is UP else -1.0
    (order,) = plan_completion(
        snap=snap(), inv=held_of(token, 250.0), fair=fv(p_up_favouring(token)), cfg=cfg,
        fees=FeeModel(cfg.fees), cash_budget=1e4, target_net=sign * 100.0,
    )  # fmt: skip
    assert (order.token, order.price, order.size) == (token.opposite, 0.51, 150.0)


# ---- engine.py: FLATTEN does not sell the same shares twice


def flatten_sells(token: Outcome, pending: Sequence[OpenOrder]) -> list[tuple[Outcome, float]]:
    engine = engine_with(up=50.0 if token is UP else 0.0, down=50.0 if token is DOWN else 0.0)
    acts = engine.on_snapshot(snap(FL_TS), pending)
    return [(r.token, r.size) for r in places(acts) if r.side is SELL]


def in_flight_sell(token: Outcome, size: float) -> OpenOrder:
    return OpenOrder("ox", "cx", "m1", token, SELL, 0.49, size, size, IOC, 1879.0, live=False)


@TOKENS
def test_pending_flatten_sell_of_either_token_is_not_resent(token: Outcome) -> None:
    # 50 shares held. With an IOC sell of all 50 still in flight (latency) there is nothing more to
    # sell; with 25 in flight the other 25 are sold now (DESIGN 2.6, plan_flatten.pending_sells).
    assert flatten_sells(token, []) == [(token, 50.0)]
    assert flatten_sells(token, [in_flight_sell(token, 50.0)]) == []
    assert flatten_sells(token, [in_flight_sell(token, 25.0)]) == [(token, 25.0)]


@TOKENS
def test_a_pending_sell_of_the_other_token_does_not_hide_the_remainder(token: Outcome) -> None:
    # 50 ``token`` shares held; the in-flight sell is of the OTHER token: still sell all 50
    assert flatten_sells(token, [in_flight_sell(token.opposite, 50.0)]) == [(token, 50.0)]


# ---- engine.py: the compounding clip is priced at the dearer token


@pytest.mark.parametrize("dear", [UP, DOWN], ids=["UP-dear", "DOWN-dear"])
def test_compounding_clip_is_priced_at_the_dearer_token_whichever_it_is(dear: Outcome) -> None:
    # equity 10000, clip_equity_fraction 0.005: the clip is 0.005 * 10000 / price with price the
    # DEARER token's mid (DESIGN 2.2(4): no level-0 order costs more than equity * fraction):
    # dear mid 0.80 -> 62.5 shares for both tokens. Pricing it at the cheaper token's 0.20 would
    # give 250, and "the UP mid" gives 250 when DOWN is the dear one.
    cfg = BotConfig(
        model=ModelConfig(shrink_to_market=0.0),
        directional=DirectionalConfig(enabled=False),
        sizing=SizingConfig(clip_equity_fraction=0.005),
    )
    cheap, rich = (0.19, 0.21), (0.79, 0.81)
    s = snap(up=rich, down=cheap) if dear is UP else snap(up=cheap, down=rich)
    first: dict[Outcome, float] = {}
    for r in places(engine_with(0.8 if dear is UP else 0.2, cfg).on_snapshot(s, [])):
        first.setdefault(r.token, r.size)
    assert first == {UP: 62.5, DOWN: 62.5}


# ---- quoter.py: WIND_DOWN room of a token favoured by the directional target


def wind_down_quotes(token: Outcome, inv: MarketInventory, target: float) -> list[QuoteLevel]:
    cfg = BotConfig()
    return compute_quotes(
        snap=snap(1850.0), fair=fv(p_up_favouring(token)), inv=inv, phase=Phase.WIND_DOWN,
        cfg=cfg, fees=FeeModel(cfg.fees), clip_shares=100.0, cash_budget=1e6, target_net=target,
    )  # fmt: skip


@TOKENS
def test_wind_down_grants_a_flat_book_room_for_the_favoured_token_only(token: Outcome) -> None:
    # Flat, deliberate target of 100 in ``token``'s direction: WIND_DOWN quotes only that token
    # (it is model-favoured and nothing completes pairs), no more than the 100 shares of target.
    sign = 1.0 if token is UP else -1.0
    qs = wind_down_quotes(token, held(), sign * 100.0)
    assert {q.token for q in qs} == {token}
    assert sum(q.size for q in qs) == 100.0


@TOKENS
def test_wind_down_room_of_the_favoured_token_is_the_remaining_directional_budget(
    token: Outcome,
) -> None:
    # 60 of a 100-share target already held -> room for 40 more (target - net in that direction)
    sign = 1.0 if token is UP else -1.0
    qs = wind_down_quotes(token, held_of(token, 60.0), sign * 100.0)
    assert sum(q.size for q in qs if q.token is token) == 40.0


# ---- exchange/paper.py: the queue ahead of an order is its own book's


@TOKENS
@pytest.mark.parametrize("latency", [0, 1], ids=["latency0", "latency1"])
def test_queue_position_comes_from_the_orders_own_book(token: Outcome, latency: int) -> None:
    # Our token's book shows 300 shares at our price 0.49, the other token's only 10. A print of 100
    # at 0.49 is absorbed by the 300 ahead of us (no fill); a print of 350 drains them and fills 50.
    # With the other book's 10 shares as the queue, the first print would fill us for 90.
    own, other = book(token, 0.49, 0.51, 300.0), book(token.opposite, 0.49, 0.51, 10.0)

    def market(ts: float, trades: tuple[Trade, ...] = ()) -> MarketSnapshot:
        up, down = (own, other) if token is UP else (other, own)
        return MarketSnapshot(
            ts, MKT, up, down, spot=100.0, spot_ts=ts, ref_price=100.0, trades=trades
        )

    def filled(print_size: float) -> float:
        ex = PaperExchange(BotConfig(exchange=PaperExchangeConfig(latency_ticks=latency)))
        ex.process(market(1030.0))
        assert ex.submit(OrderRequest("c1", "m1", token, BUY, 0.49, 50.0, POST), 1030.0).ok
        for ts in range(1031, 1031 + latency):  # in flight until the order has become live
            ex.process(market(float(ts)))
        t = Trade(1032.0 + latency, token, 0.49, print_size, SELL)
        fills = ex.process(market(1032.0 + latency, (t,)))
        assert all(f.token is token for f in fills)
        return sum(f.size for f in fills)

    assert filled(100.0) == 0.0
    assert filled(350.0) == 50.0


# ---- types.py: the tick helpers of the frozen contract


@pytest.mark.parametrize(
    ("price", "tick", "floor", "ceil", "snapped"),
    [
        (0.4949, 0.01, 0.49, 0.50, 0.49),
        (0.4951, 0.01, 0.49, 0.50, 0.50),
        (0.495, 0.01, 0.49, 0.50, 0.50),
        (0.49, 0.01, 0.49, 0.49, 0.49),  # on the grid: all three keep it
        (0.57, 0.01, 0.57, 0.57, 0.57),  # 0.57 / 0.01 = 56.99999999999999: the EPS slack matters
        (0.0, 0.01, 0.0, 0.0, 0.0),
        (0.1234, 0.001, 0.123, 0.124, 0.123),
        (0.5, 0.02, 0.5, 0.5, 0.5),
        (0.51, 0.02, 0.5, 0.52, 0.52),
    ],
)
def test_price_helpers_floor_ceil_and_snap_to_the_tick_grid(
    price: float, tick: float, floor: float, ceil: float, snapped: float
) -> None:
    assert floor_price(price, tick) == floor
    assert ceil_price(price, tick) == ceil
    assert snap_price(price, tick) == snapped


def test_ceil_and_floor_price_bracket_every_price_on_a_random_grid() -> None:
    rng = random.Random(5)
    for _ in range(2000):
        tick = rng.choice([0.01, 0.001, 0.02, 0.005])
        price = rng.uniform(0.0, 1.0)
        lo, hi = floor_price(price, tick), ceil_price(price, tick)
        assert lo <= price + 1e-9 and hi >= price - 1e-9
        assert 0.0 <= hi - lo <= tick + 1e-9
        for p in (lo, hi):  # both are on the grid
            assert abs(p / tick - round(p / tick)) < 1e-6


# =========================================================================== label symmetry


def random_cfg(rng: random.Random, *, many_orders: bool = False) -> BotConfig:
    max_net = rng.choice([0.0, 100.0, 300.0, 300.0, 600.0])
    return BotConfig(
        fees=FeeConfig(
            maker_fee_rate=rng.choice([0.0, 0.0, 0.01]),
            taker_fee_rate=rng.choice([0.0, 0.02, 0.05]),
        ),
        sizing=SizingConfig(
            ladder_levels=rng.randint(1, 4),
            ladder_spacing_ticks=rng.randint(1, 2),
            ladder_size_decay=rng.choice([0.5, 0.7, 1.0]),
            clip_equity_fraction=rng.choice([None, None, 0.005, 0.02]),
        ),
        pair=PairConfig(
            target_margin=rng.choice([0.0, 0.005, 0.01, 0.03]),
            max_net_imbalance_shares=max_net,
            max_inventory_per_side_shares=max_net + rng.choice([0.0, 200.0, 1200.0]),
            skew_ticks_per_100_shares=rng.choice([0.0, 0.5, 1.0, 3.0]),
            rebalance_trigger_shares=rng.choice([30.0, 100.0]),
        ),
        directional=DirectionalConfig(
            enabled=rng.random() < 0.8,
            min_edge=rng.choice([0.0, 0.01, 0.03]),
            max_directional_shares=rng.choice([0.0, 100.0, 200.0]),
        ),
        model=ModelConfig(shrink_to_market=0.0),
        risk=RiskConfig(
            max_spread_ticks_to_quote=rng.choice([2, 6, 6]),
            max_orders_per_second=1000.0 if many_orders else 20.0,
        ),
    )


def random_book(rng: random.Random, token: Outcome) -> BookSnapshot:
    """A random book of its own: it does NOT depend on the other token's book."""
    bid = rng.randint(3, 96)
    spread = rng.choice([1, 1, 2, 3, 7])  # 7 ticks is wider than every quoting limit
    ask = min(99, bid + spread)
    bid = min(bid, ask - 1)
    n = rng.randint(1, 3)
    return BookSnapshot(
        token,
        tuple(Level((bid - j) / 100, float(rng.randint(20, 400))) for j in range(n) if bid - j > 0),
        tuple(
            Level((ask + j) / 100, float(rng.randint(20, 400))) for j in range(n) if ask + j < 100
        ),
    )


def random_snap(rng: random.Random, ts: float) -> MarketSnapshot:
    return MarketSnapshot(
        ts, MKT, random_book(rng, UP), random_book(rng, DOWN),
        spot=100.0, spot_ts=ts, ref_price=100.0,
    )  # fmt: skip


def random_inventory(rng: random.Random, max_side: float) -> MarketInventory:
    top = int(max(11.0, max_side * 1.05))
    kind = rng.random()
    if kind < 0.25:
        up = down = 0.0
    elif kind < 0.4:
        up = down = float(rng.randint(10, top))
    else:
        up = rng.choice([0.0, float(rng.randint(10, top))])
        down = rng.choice([0.0, float(rng.randint(10, top))])
    return held(up, down, up_px=rng.choice([0.4, 0.47, 0.55]), down_px=rng.choice([0.4, 0.52, 0.6]))


def dyadic_p(rng: random.Random) -> float:
    """A probability k/64, so that ``1 - p`` is exact and a label swap adds no rounding noise."""
    return rng.randint(3, 61) / 64.0


# ---- the quoter


@dataclass(frozen=True)
class QuoteState:
    cfg: BotConfig
    snap: MarketSnapshot
    inv: MarketInventory
    fair: FairValue
    phase: Phase
    clip: float
    budget: float
    target: float

    def quotes(self, *, budget: float | None = None) -> list[QuoteLevel]:
        return compute_quotes(
            snap=self.snap, fair=self.fair, inv=self.inv, phase=self.phase, cfg=self.cfg,
            fees=FeeModel(self.cfg.fees), clip_shares=self.clip,
            cash_budget=self.budget if budget is None else budget, target_net=self.target,
        )  # fmt: skip

    def swapped(self) -> QuoteState:
        return replace(
            self,
            snap=swap_snap(self.snap),
            inv=swap_inv(self.inv),
            fair=swap_fair(self.fair),
            target=-self.target,
        )


def random_quote_state(rng: random.Random) -> QuoteState:
    cfg = random_cfg(rng)
    return QuoteState(
        cfg=cfg,
        snap=random_snap(rng, 1100.0),
        inv=random_inventory(rng, cfg.pair.max_inventory_per_side_shares),
        fair=fv(dyadic_p(rng), valid=rng.random() < 0.9),
        phase=rng.choice([Phase.ACCUMULATE] * 4 + [Phase.WIND_DOWN] * 4 + [Phase.FLATTEN]),
        clip=rng.choice([5.0, 20.0, 100.0, 400.0]),
        budget=rng.choice([rng.uniform(0.0, 120.0), 1e5, 1e5]),
        target=rng.choice([0.0, rng.uniform(-400.0, 400.0), rng.uniform(-400.0, 400.0)]),
    )


def quote_tuples(qs: Sequence[QuoteLevel], *, swap: bool) -> list[tuple[str, float, float]]:
    """The quotes as sorted comparable tuples, with the token names swapped back if ``swap``."""
    return sorted(((q.token.opposite if swap else q.token).value, q.price, q.size) for q in qs)


def test_quotes_are_label_symmetric_in_every_phase_on_unmirrored_books() -> None:
    rng = random.Random(20260101)
    seen: Counter[str] = Counter()
    for _ in range(4000):
        st = random_quote_state(rng)
        qs = st.quotes()
        assert quote_tuples(qs, swap=False) == quote_tuples(st.swapped().quotes(), swap=True)
        tokens = {q.token for q in qs}
        seen["quoted"] += bool(qs)
        seen["up_only"] += tokens == {UP}
        seen["down_only"] += tokens == {DOWN}
        seen["wind_down"] += bool(qs) and st.phase is Phase.WIND_DOWN
        seen["wind_down_down_target"] += (
            bool(qs)
            and st.phase is Phase.WIND_DOWN
            and st.target < 0
            and st.cfg.directional.enabled
        )
        seen["budget_bound"] += bool(qs) and qs != st.quotes(budget=1e9)
    # not vacuous: many states quote, on either token alone and on both, in WIND_DOWN too
    assert seen["quoted"] > 1000
    assert seen["up_only"] > 100 and seen["down_only"] > 100
    assert seen["wind_down"] > 250 and seen["wind_down_down_target"] > 30
    assert seen["budget_bound"] > 100


# ---- the rebalancer


def test_taker_completion_is_label_symmetric_on_unmirrored_books() -> None:
    rng = random.Random(20260102)
    seen: Counter[str] = Counter()
    for _ in range(4000):
        cfg = random_cfg(rng)
        s, inv = random_snap(rng, 1100.0), random_inventory(rng, 600.0)
        fair = fv(dyadic_p(rng), valid=rng.random() < 0.9)
        target = rng.choice([0.0, rng.uniform(-300.0, 300.0), rng.uniform(-300.0, 300.0)])
        budget = rng.choice([rng.uniform(0.0, 200.0), 1e5])
        fees = FeeModel(cfg.fees)
        out = plan_completion(
            snap=s, inv=inv, fair=fair, cfg=cfg, fees=fees, cash_budget=budget, target_net=target
        )
        mirror = plan_completion(
            snap=swap_snap(s), inv=swap_inv(inv), fair=swap_fair(fair), cfg=cfg, fees=fees,
            cash_budget=budget, target_net=-target,
        )  # fmt: skip
        assert [(o.token, o.price, o.size, o.net_profit_per_share) for o in out] == [
            (o.token.opposite, o.price, o.size, o.net_profit_per_share) for o in mirror
        ]
        seen["planned"] += bool(out)
        seen["buying_up"] += bool(out) and out[0].token is UP
        seen["buying_down"] += bool(out) and out[0].token is DOWN
        seen["allowance"] += bool(out) and abs(target) > 1.0 and cfg.directional.enabled
    assert seen["planned"] > 300
    assert seen["buying_up"] > 100 and seen["buying_down"] > 100
    assert seen["allowance"] > 50


def flatten_view(plan: FlattenPlan, *, swap: bool) -> tuple[object, ...]:
    def name(t: Outcome) -> str:
        return (t.opposite if swap else t).value

    return (
        plan.merge_size,
        [(name(x.token), x.price, x.size) for x in plan.sells],
        None if plan.hold_token is None else name(plan.hold_token),
        plan.hold_size,
        plan.reason,
    )


def test_flatten_plan_is_label_symmetric_on_unmirrored_books() -> None:
    rng = random.Random(20260103)
    seen: Counter[str] = Counter()
    for _ in range(4000):
        cfg = random_cfg(rng)
        s, inv = random_snap(rng, 1880.0), random_inventory(rng, 600.0)
        fair = fv(dyadic_p(rng), valid=rng.random() < 0.9)
        pending = {tok: float(rng.choice([0, 0, 10, 40])) for tok in (UP, DOWN)}
        fees = FeeModel(cfg.fees)
        plan = plan_flatten(snap=s, inv=inv, fair=fair, cfg=cfg, fees=fees, pending_sells=pending)
        mirror = plan_flatten(
            snap=swap_snap(s), inv=swap_inv(inv), fair=swap_fair(fair), cfg=cfg, fees=fees,
            pending_sells={tok.opposite: n for tok, n in pending.items()},
        )  # fmt: skip
        assert flatten_view(plan, swap=False) == flatten_view(mirror, swap=True)
        seen["sells_up"] += bool(plan.sells) and plan.sells[0].token is UP
        seen["sells_down"] += bool(plan.sells) and plan.sells[0].token is DOWN
        seen["held_on_edge"] += plan.reason == "edge"
    assert seen["sells_up"] > 100 and seen["sells_down"] > 100 and seen["held_on_edge"] > 30


# ---- the risk gate


def test_book_gate_is_label_symmetric_on_unmirrored_books() -> None:
    rng = random.Random(20260104)
    n_true = 0
    for _ in range(3000):
        s = random_snap(rng, 1100.0)
        limit = rng.choice([1, 2, 6])
        risk = RiskManager(RiskConfig(max_spread_ticks_to_quote=limit), 10_000.0)
        verdict = risk.book_ok(s, limit)
        assert verdict == risk.book_ok(swap_snap(s), limit)
        assert books_quotable(s, limit) == books_quotable(swap_snap(s), limit) == verdict
        n_true += verdict
    assert 500 < n_true < 2500  # both verdicts are well represented


# ---- the engine, end to end


@dataclass(frozen=True)
class EngineState:
    cfg: BotConfig
    snap: MarketSnapshot
    up: float
    down: float
    p_up: float
    orders: tuple[OpenOrder, ...]
    cash: float

    def run(self, *, swap: bool) -> tuple[Counter[tuple[object, ...]], dict[str, int]]:
        """What a fresh engine does with this state, as comparable tuples (client ids dropped)
        and its counters. With ``swap`` the whole state is label-swapped and the answer is
        translated back, so that both calls can be compared directly."""
        if swap:
            engine = engine_with(
                1.0 - self.p_up, self.cfg, cash=self.cash, up=self.down, down=self.up
            )
            acts = engine.on_snapshot(swap_snap(self.snap), [swap_order(o) for o in self.orders])
        else:
            engine = engine_with(self.p_up, self.cfg, cash=self.cash, up=self.up, down=self.down)
            acts = engine.on_snapshot(self.snap, list(self.orders))
        return normalise(acts, swap=swap), engine.stats


def normalise(actions: Sequence[Action], *, swap: bool) -> Counter[tuple[object, ...]]:
    out: Counter[tuple[object, ...]] = Counter()
    for a in actions:
        if isinstance(a, PlaceOrder):
            r = a.request
            token = (r.token.opposite if swap else r.token).value
            out[("place", token, r.side.value, r.tif.value, r.price, r.size)] += 1
        elif isinstance(a, CancelOrder):
            out[("cancel", a.order_id)] += 1
        elif isinstance(a, MergePairs):
            out[("merge", a.size)] += 1
        else:  # pragma: no cover - the engine emits no other action
            raise AssertionError(a)
    return out


def random_orders(rng: random.Random) -> tuple[OpenOrder, ...]:
    orders = []
    for i in range(rng.choice([0, 0, 1, 2, 4])):
        size = float(rng.choice([20, 50, 100]))
        price = rng.randint(30, 60) / 100
        token = rng.choice([UP, DOWN])
        orders.append(
            OpenOrder(f"o{i}", f"c{i}", "m1", token, BUY, price, size, size, POST, 1029.0)
        )  # fmt: skip
    if rng.random() < 0.3:  # an IOC buy or sell still in flight
        token, side = rng.choice([UP, DOWN]), rng.choice([BUY, SELL])
        orders.append(OpenOrder("ox", "cx", "m1", token, side, 0.5, 40.0, 40.0, IOC, 1029.0, False))
    return tuple(orders)


def random_engine_state(rng: random.Random) -> EngineState:
    cfg = random_cfg(rng, many_orders=True)
    inv = random_inventory(rng, cfg.pair.max_inventory_per_side_shares)
    ts = rng.choice([ACC_TS, ACC_TS, WD_TS, WD_TS, FL_TS])
    return EngineState(
        cfg=cfg,
        snap=random_snap(rng, ts),
        up=inv.qty[UP],
        down=inv.qty[DOWN],
        p_up=dyadic_p(rng),
        orders=random_orders(rng),
        # the starting cash pays for the inventory (bought at 0.47) and leaves some free cash,
        # often little, so that the cash budget of the ladders binds
        cash=0.47 * (inv.qty[UP] + inv.qty[DOWN])
        + rng.choice([rng.uniform(50.0, 400.0), 10_000.0]),
    )


def test_engine_actions_are_label_symmetric_in_every_phase() -> None:
    rng = random.Random(20260105)
    seen: Counter[str] = Counter()
    for _ in range(1500):
        st = random_engine_state(rng)
        acts, stats = st.run(swap=False)
        mirror, mirror_stats = st.run(swap=True)
        assert acts == mirror
        assert stats == mirror_stats
        kinds = {k[:4] for k in acts if k[0] == "place"}  # ("place", token, side, tif)
        seen["acted"] += bool(acts)
        for token in ("UP", "DOWN"):
            tag = token.lower()
            seen[f"sell_{tag}"] += ("place", token, "SELL", "IOC") in kinds
            seen[f"complete_{tag}"] += ("place", token, "BUY", "IOC") in kinds
            seen[f"passive_{tag}"] += ("place", token, "BUY", "POST_ONLY") in kinds
        seen["merge"] += any(k[0] == "merge" for k in acts)
        seen["cancel"] += any(k[0] == "cancel" for k in acts)
    assert seen["acted"] > 1000
    for key in ("sell", "complete", "passive"):
        assert seen[f"{key}_up"] > 30 and seen[f"{key}_down"] > 30, key
    assert seen["merge"] > 30 and seen["cancel"] > 30


# ---- the paper exchange


Ladder = tuple[tuple[float, float], ...]  # (price, size) per level


@dataclass(frozen=True)
class Step:
    """One snapshot of a random exchange scenario, in plain data (labels not yet applied)."""

    up: tuple[Ladder, Ladder]  # bids, asks
    down: tuple[Ladder, Ladder]
    prints: tuple[tuple[Outcome, float, float, Side], ...]  # token, price, size, aggressor
    orders: tuple[tuple[Outcome, Side, float, float, TimeInForce], ...]
    cancel: int | None  # index into the open orders (sorted by id) to cancel after the orders


def ladders(b: BookSnapshot) -> tuple[Ladder, Ladder]:
    return (
        tuple((lv.price, lv.size) for lv in b.bids),
        tuple((lv.price, lv.size) for lv in b.asks),
    )


def random_order(
    rng: random.Random, token: Outcome, bid: float, ask: float, *, early: bool
) -> tuple[Outcome, Side, float, float, TimeInForce]:
    """Early in a scenario mostly buys (to have shares), later mostly sells of them."""
    size = float(rng.randint(5, 150))
    kinds = (
        ["bid", "ioc_buy", "ioc_buy", "ioc_buy"]
        if early
        else ["ask", "ioc_sell", "ioc_sell", "bid"]
    )
    kind = rng.choice(kinds)
    if kind == "bid":
        return token, BUY, round(max(0.01, ask - 0.01 * rng.randint(1, 3)), 2), size, POST
    if kind == "ask":
        return token, SELL, round(ask + 0.01 * rng.randint(0, 2), 2), size, POST
    if kind == "ioc_buy":
        return token, BUY, round(ask + 0.01 * rng.randint(0, 1), 2), size, IOC
    return token, SELL, round(max(0.01, bid - 0.01 * rng.randint(0, 1)), 2), size, IOC


def random_scenario(rng: random.Random, n_steps: int) -> list[Step]:
    steps = []
    for i in range(n_steps):
        books = {UP: random_book(rng, UP), DOWN: random_book(rng, DOWN)}
        touch = {t: (b.bids[0].price, b.asks[0].price) for t, b in books.items()}
        prints = []
        for _ in range(rng.randint(0, 4)):
            token = rng.choice([UP, DOWN])
            bid, ask = touch[token]
            price = rng.choice([bid, bid, round(bid - 0.01, 2), ask, ask, round(ask + 0.01, 2)])
            prints.append((token, price, float(rng.randint(10, 900)), rng.choice([BUY, SELL])))
        orders = []
        for _ in range(rng.randint(0, 3)):
            token = rng.choice([UP, DOWN])
            orders.append(random_order(rng, token, *touch[token], early=i < n_steps // 3))
        cancel = rng.randint(0, 3) if rng.random() < 0.3 else None
        steps.append(
            Step(ladders(books[UP]), ladders(books[DOWN]), tuple(prints), tuple(orders), cancel)
        )
    return steps


def play(cfg: BotConfig, steps: Sequence[Step], *, swap: bool) -> tuple[list[object], Counter[str]]:
    """Run the scenario on a fresh exchange (with the labels exchanged if ``swap``) and return a
    trace in canonical (un-swapped) labels, so that the two worlds can be compared directly."""

    def lab(t: Outcome) -> Outcome:  # the label this world gives to the scenario's token t
        return t.opposite if swap else t

    def canon(t: Outcome) -> str:  # the scenario's name of the token this world calls t
        return (t.opposite if swap else t).value

    def book_of(token: Outcome, lad: tuple[Ladder, Ladder]) -> BookSnapshot:
        return BookSnapshot(
            token, tuple(Level(p, z) for p, z in lad[0]), tuple(Level(p, z) for p, z in lad[1])
        )

    ex = PaperExchange(cfg)
    trace: list[object] = []
    seen: Counter[str] = Counter()
    for i, st in enumerate(steps):
        ts = 1000.0 + i
        up_lad, down_lad = (st.down, st.up) if swap else (st.up, st.down)
        s = MarketSnapshot(
            ts, MKT, book_of(UP, up_lad), book_of(DOWN, down_lad),
            spot=100.0, spot_ts=ts, ref_price=100.0,
            trades=tuple(Trade(ts, lab(tok), p, z, side) for tok, p, z, side in st.prints),
        )  # fmt: skip
        fills = ex.process(s)
        outcomes: list[object] = []
        for tok, side, price, size, tif in st.orders:
            r = ex.submit(OrderRequest(f"c{i}", "m1", lab(tok), side, price, size, tif), ts)
            outcomes.append((r.ok, r.reason))
        if st.cancel is not None:
            open_ = sorted(ex.open_orders("m1"), key=lambda o: o.order_id)
            if open_:
                outcomes.append(("cancel", ex.cancel(open_[st.cancel % len(open_)].order_id, ts)))
        fills = fills + ex.drain_fills()  # latency-0 IOC fills are queued, not returned by submit
        for f in fills:
            seen[canon(f.token)] += 1
            seen["maker" if f.is_maker else "taker"] += 1
            seen[f.side.value] += 1
        filled = sorted(
            (canon(f.token), f.side.value, f.price, f.size, f.fee, f.is_maker) for f in fills
        )
        resting = sorted(
            (canon(o.token), o.side.value, o.price, o.remaining, o.live)
            for o in ex.open_orders("m1")
        )
        positions = (ex.position("m1", lab(UP)), ex.position("m1", lab(DOWN)))
        trace.append((outcomes, filled, ex.balance(), positions, resting))
    return trace, seen


@pytest.mark.parametrize("latency", [0, 1, 2])
def test_paper_exchange_is_label_symmetric_on_unmirrored_books(latency: int) -> None:
    rng = random.Random(20260106 + latency)
    totals: Counter[str] = Counter()
    for _ in range(120):
        cfg = BotConfig(
            fees=FeeConfig(
                maker_fee_rate=rng.choice([0.0, 0.01]), taker_fee_rate=rng.choice([0.0, 0.03])
            ),
            exchange=PaperExchangeConfig(
                latency_ticks=latency,
                queue_ahead_fraction=rng.choice([0.0, 0.5, 1.0]),
                trade_fill_fraction=rng.choice([0.5, 1.0]),
                taker_slippage_ticks=rng.choice([0, 1]),
            ),
        )
        steps = random_scenario(rng, 10)
        plain, seen = play(cfg, steps, swap=False)
        mirror, _ = play(cfg, steps, swap=True)
        for k, (x, y) in enumerate(zip(plain, mirror, strict=True)):
            assert x == y, f"step {k}"  # the first difference is the most useful message
        totals.update(seen)
    # the scenarios exercise the exchange on both tokens, both sides and both liquidity roles
    for key in ("UP", "DOWN", "maker", "taker", "BUY", "SELL"):
        assert totals[key] > 40, key


# ---- the harness itself


def test_the_swap_helpers_are_involutions() -> None:
    s = replace(
        random_snap(random.Random(1), 1100.0),
        trades=(Trade(1100.0, UP, 0.49, 10.0, SELL), Trade(1100.0, DOWN, 0.51, 10.0, BUY)),
    )
    assert swap_snap(swap_snap(s)) == s
    assert swap_snap(s).up_book.bids == s.down_book.bids
    assert swap_snap(s).trades[0].token is DOWN
    f = fv(0.25)
    assert swap_fair(swap_fair(f)) == f
    assert math.isclose(swap_fair(f).p(UP), f.p(DOWN))
    inv = held(up=30.0, down=10.0)
    twice, once = swap_inv(swap_inv(inv)), swap_inv(inv)
    assert (twice.qty, twice.cost) == (inv.qty, inv.cost)
    assert (once.qty[UP], once.qty[DOWN]) == (10.0, 30.0)
