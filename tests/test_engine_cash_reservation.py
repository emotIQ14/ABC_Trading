"""The engine's reservation belief must equal what the paper exchange commits for a resting bid.

``PaperExchange`` commits ``price * remaining + max(0, maker fee)`` for every resting or in-flight
POST_ONLY BUY. The engine's quote budget is ``cash - R_other - ...`` where ``R_other`` is its belief
of the other markets' reservations, so a belief that omits the maker fee lets the next market
size its ladder up to the full cash and the exchange then refuses the last levels with
``insufficient_cash`` (every snapshot, for as long as the account is cash-limited). With a zero
maker fee the two coincide, which is why only a positive maker fee plus binding cash shows it.

The tests drive the real engine against the real ``PaperExchange`` (two markets, a scripted fair
value so the ladder is deterministic) and also check the bookkeeping directly: the belief after
each snapshot, the release on a fill, and the capital-cap accounting. The full synthetic runs
that go through ``run_backtest`` live in ``test_backtest_e2e.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from abc_trading.config import BotConfig, FeeConfig, ModelConfig, PaperExchangeConfig, RiskConfig
from abc_trading.exchange.paper import PaperExchange
from abc_trading.fees import FeeModel
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.types import (
    BookSnapshot,
    CancelOrder,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
    OpenOrder,
    Outcome,
    PlaceOrder,
    Side,
    TimeInForce,
)

UP, DOWN = Outcome.UP, Outcome.DOWN
BUY = Side.BUY
POST, IOC = TimeInForce.POST_ONLY, TimeInForce.IOC
MARKETS = (
    MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0),
    MarketSpec("m2", "ETH", start_ts=1000.0, end_ts=1900.0),
)
FIRST_TS = 1030  # ACCUMULATE
N_TICKS = 10


class StubModel:
    """Scripted fair value (p_up 0.5, valid) so the ladder is the same on every snapshot."""

    def observe(self, asset: str, ts: float, spot: float) -> None:
        pass

    def estimate(self, snap: MarketSnapshot) -> FairValue:
        return FairValue(0.5, 0.5, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, True)


def make_cfg(cash: float, maker_fee: float, rebate: float = 0.0, latency: int = 1) -> BotConfig:
    return BotConfig(
        fees=FeeConfig(maker_fee_rate=maker_fee, maker_rebate_rate=rebate),
        model=ModelConfig(shrink_to_market=0.0),
        exchange=PaperExchangeConfig(initial_cash=cash, latency_ticks=latency),
    )


def snapshot(market: MarketSpec, ts: float) -> MarketSnapshot:
    def book(token: Outcome) -> BookSnapshot:
        return BookSnapshot(token, (Level(0.49, 400.0),), (Level(0.51, 400.0),))

    return MarketSnapshot(
        ts=ts, market=market, up_book=book(UP), down_book=book(DOWN),
        spot=100.0, spot_ts=ts, ref_price=100.0,
    )  # fmt: skip


def exchange_commitment(cfg: BotConfig, orders: list[OpenOrder]) -> float:
    """What ``PaperExchange`` claims for these open orders, spelled out independently."""
    fees = FeeModel(cfg.fees)
    return sum(
        o.price * o.remaining + max(0.0, fees.fee(o.price, o.remaining, True))
        for o in orders
        if o.side is BUY and o.tif is POST
    )


def drive(
    cfg: BotConfig, *, check_belief: bool = False
) -> tuple[MarketMakerEngine, PaperExchange, int, dict[str, int]]:
    """Run the engine against the real exchange; returns (engine, exchange, orders sent, rejects).

    Mirrors the backtest runner's loop: per snapshot the exchange processes the market, fills
    reach the engine, then the engine's cancels and placements are executed. Rejections are
    counted by reason (the engine is never told about them, as in the runner).
    """
    engine, ex = MarketMakerEngine(cfg), PaperExchange(cfg)
    engine.model = StubModel()  # type: ignore[assignment]
    sent = 0
    rejects: dict[str, int] = {}
    for tick in range(N_TICKS):
        for market in MARKETS:
            ts = float(FIRST_TS + tick)
            snap = snapshot(market, ts)
            for fill in ex.process(snap):
                engine.on_fill(fill)
            for action in engine.on_snapshot(snap, ex.open_orders(market.market_id)):
                if isinstance(action, CancelOrder):
                    ex.cancel(action.order_id, ts)
                elif isinstance(action, PlaceOrder):
                    sent += 1
                    result = ex.submit(action.request, ts)
                    if not result.ok:
                        rejects[result.reason] = rejects.get(result.reason, 0) + 1
            if check_belief:
                belief = engine._states[market.market_id].reserved
                held = exchange_commitment(cfg, ex.open_orders(market.market_id))
                assert belief == pytest.approx(held, abs=1e-9), (tick, market.market_id)
    return engine, ex, sent, rejects


# --------------------------------------------------------------------------- the belief


@pytest.mark.parametrize(
    ("maker_fee", "rebate"),
    [(0.0, 0.0), (0.001, 0.0), (0.02, 0.0), (0.02, 0.005), (0.0, 0.01)],
    ids=["no-fee", "fee-0.1%", "fee-2%", "fee-2%-rebate-0.5%", "rebate-only"],
)
def test_the_reservation_belief_equals_the_exchange_commitment_after_every_snapshot(
    maker_fee: float, rebate: float
) -> None:
    """Ample cash (nothing is refused), so the belief must be exactly what the exchange holds:
    price * remaining plus any positive net maker fee, per market, after every snapshot."""
    cfg = make_cfg(10_000.0, maker_fee, rebate)
    _, ex, sent, rejects = drive(cfg, check_belief=True)
    assert rejects == {} and sent > 0
    assert exchange_commitment(cfg, ex.open_orders()) > 0.0  # the ladders really are resting


@pytest.mark.parametrize("cash", [100.0, 150.0, 200.0, 250.0])
@pytest.mark.parametrize("maker_fee", [0.0, 0.001, 0.02])
@pytest.mark.parametrize("latency", [0, 1])
def test_a_cash_limited_two_market_ladder_is_never_refused_for_cash(
    cash: float, maker_fee: float, latency: int
) -> None:
    cfg = make_cfg(cash, maker_fee, latency=latency)
    _, ex, _, rejects = drive(cfg)
    assert rejects == {}
    # cash really is the binding constraint here: the resting bids use nearly all of it ...
    committed = exchange_commitment(cfg, ex.open_orders())
    assert cash - 5.0 <= committed <= cash + 1e-9
    # ... and a roomy account would have quoted a larger ladder
    roomy = make_cfg(10_000.0, maker_fee, latency=latency)
    _, roomy_ex, _, _ = drive(roomy)
    assert exchange_commitment(roomy, roomy_ex.open_orders()) > cash


def test_the_engines_orders_never_commit_more_than_the_cash_it_holds() -> None:
    """The engine alone (no exchange refusing anything): whatever it keeps and places has a
    total commitment, maker fees included, within the account's cash after every snapshot."""
    cfg = make_cfg(200.0, 0.02)
    engine = MarketMakerEngine(cfg)
    engine.model = StubModel()  # type: ignore[assignment]
    book: dict[str, OpenOrder] = {}
    seq = 0
    for tick in range(N_TICKS):
        for market in MARKETS:
            ts = float(FIRST_TS + tick)
            mine = [o for o in book.values() if o.market_id == market.market_id]
            for action in engine.on_snapshot(snapshot(market, ts), mine):
                if isinstance(action, CancelOrder):
                    book.pop(action.order_id, None)
                elif isinstance(action, PlaceOrder):
                    r, seq = action.request, seq + 1
                    book[f"o{seq}"] = OpenOrder(
                        f"o{seq}", r.client_id, r.market_id, r.token, r.side, r.price, r.size,
                        r.size, r.tif, ts,
                    )  # fmt: skip
            assert exchange_commitment(cfg, list(book.values())) <= engine.portfolio.cash + 1e-9
    assert book  # something was quoted


# --------------------------------------------------------------------------- the release


def place_ladder(cfg: BotConfig) -> tuple[MarketMakerEngine, list[OpenOrder]]:
    """Engine after one snapshot of m1 with no resting orders, and the orders it emitted."""
    engine = MarketMakerEngine(cfg)
    engine.model = StubModel()  # type: ignore[assignment]
    actions = engine.on_snapshot(snapshot(MARKETS[0], float(FIRST_TS)), [])
    orders = [
        OpenOrder(
            f"o{i}",
            a.request.client_id,
            "m1",
            a.request.token,
            a.request.side,
            a.request.price,
            a.request.size,
            a.request.size,
            a.request.tif,
            float(FIRST_TS),
        )  # fmt: skip
        for i, a in enumerate(actions)
        if isinstance(a, PlaceOrder)
    ]
    assert len(orders) >= 4 and all(o.tif is POST for o in orders)
    return engine, orders


@pytest.mark.parametrize(
    ("maker_fee", "rebate"),
    [(0.0, 0.0), (0.02, 0.0), (0.02, 0.005), (0.0, 0.01)],
    ids=["no-fee", "fee-2%", "fee-2%-rebate-0.5%", "rebate-only"],
)
def test_a_maker_fill_releases_its_notional_and_its_positive_fee(
    maker_fee: float, rebate: float
) -> None:
    cfg = make_cfg(10_000.0, maker_fee, rebate)
    fees = FeeModel(cfg.fees)
    engine, orders = place_ladder(cfg)
    state = engine._states["m1"]
    assert state.reserved == pytest.approx(exchange_commitment(cfg, orders), abs=1e-9)

    def maker_fill(o: OpenOrder, qty: float) -> Fill:
        fee = fees.fee(o.price, qty, True)
        return Fill(f"f-{o.order_id}", o.order_id, "m1", o.token, BUY, o.price, qty, fee, True, 1.0)

    # a partial fill of the first order: the exchange's commitment shrinks proportionally
    first = orders[0]
    qty = first.remaining * 0.4
    engine.on_fill(maker_fill(first, qty))
    left = [replace(first, remaining=first.remaining - qty), *orders[1:]]
    assert state.reserved == pytest.approx(exchange_commitment(cfg, left), abs=1e-9)

    # filling every other order completely leaves exactly the first order's remainder
    for o in orders[1:]:
        engine.on_fill(maker_fill(o, o.remaining))
    assert state.reserved == pytest.approx(exchange_commitment(cfg, left[:1]), abs=1e-9)


# --------------------------------------------------------------------------- capital caps


def test_the_capital_caps_count_the_maker_fee_like_a_cost_basis_does() -> None:
    """A filled bid enters the cost basis as price * size + fee (DESIGN 2.1), so the same bid
    resting counts that much against ``max_capital_per_market_usd`` (DESIGN 2.7). 100 @ 0.5 is
    50 USDC, 51 USDC with a 2% maker fee: a 50.5 cap trips only in the second case."""
    resting = OpenOrder("o1", "c1", "m1", UP, BUY, 0.5, 100.0, 100.0, POST, 1.0)

    def capital_trips(maker_fee: float) -> int:
        cfg = replace(
            make_cfg(10_000.0, maker_fee),
            risk=replace(RiskConfig(), max_capital_per_market_usd=50.5),
        )
        engine = MarketMakerEngine(cfg)
        engine.model = StubModel()  # type: ignore[assignment]
        engine.on_snapshot(snapshot(MARKETS[0], float(FIRST_TS)), [resting])
        return engine.stats["risk_trips_capital"]

    assert capital_trips(0.0) == 0
    assert capital_trips(0.02) == 1
