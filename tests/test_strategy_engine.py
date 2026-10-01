"""Tests for abc_trading.strategy.engine.MarketMakerEngine.

Scripted harness (no real exchange): ``StubModel`` replaces ``engine.model`` so each test controls
the fair value exactly (p_up, valid); ``FakeExchange`` applies the engine's actions to a tiny
order book (cash / post-only checks like the real exchange, immediate IOC fills, maker fills on
demand) and records every rejection, so a test can also prove that the engine never asks for
more cash than it has. The real ``FairValueModel`` is used in the integration tests at the end.

Market "m1": BTC, window 1000..1900, tick 0.01, min order size 5. Default timing gives
WARMUP < 1015 <= ACCUMULATE < 1810 <= WIND_DOWN < 1875 <= FLATTEN < 1900 <= DONE. Default books are
the mirrored 0.49 / 0.51 on both tokens; with fair 0.5 the balanced caps are 0.495 so the ladder
sits at 0.49, 0.48, 0.47 with sizes 100, 70, 49 per token (clip 100, decay 0.7).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace

import pytest

from abc_trading.config import (
    BotConfig,
    DirectionalConfig,
    ModelConfig,
    PairConfig,
    RiskConfig,
    SizingConfig,
    TimingConfig,
)
from abc_trading.model.fair_value import FairValue
from abc_trading.strategy.engine import MarketMakerEngine
from abc_trading.types import (
    EPS,
    Action,
    BookSnapshot,
    CancelOrder,
    Fill,
    Level,
    MarketSnapshot,
    MarketSpec,
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

UP, DOWN = Outcome.UP, Outcome.DOWN
BUY, SELL = Side.BUY, Side.SELL
POST, IOC = TimeInForce.POST_ONLY, TimeInForce.IOC
MKT = MarketSpec("m1", "BTC", start_ts=1000.0, end_ts=1900.0)
MKT2 = MarketSpec("m2", "ETH", start_ts=1000.0, end_ts=1900.0)
CFG = BotConfig(model=ModelConfig(shrink_to_market=0.0))
NO_DIRECTIONAL = replace(CFG, directional=DirectionalConfig(enabled=False))
ACC_TS = 1030.0  # a normal ACCUMULATE timestamp
WD_TS = 1820.0  # WIND_DOWN (left = 80)
FL_TS = 1880.0  # FLATTEN (left = 20)


# --------------------------------------------------------------------------- harness


class StubModel:
    """Stands in for FairValueModel: scripted fair value, records observations."""

    def __init__(self, p_up: float = 0.5, valid: bool = True) -> None:
        self.p_up = p_up
        self.valid = valid
        self.observed: list[tuple[str, float, float]] = []

    def observe(self, asset: str, ts: float, spot: float) -> None:
        self.observed.append((asset, ts, spot))

    def estimate(self, snap: MarketSnapshot) -> FairValue:
        return FairValue(self.p_up, self.p_up, 0.0, 0.0, 1e-4, 800.0, 0.0, 0.0, 0.0, self.valid)


def mkbook(
    token: Outcome, bid: float | None, ask: float | None, size: float = 400.0
) -> BookSnapshot:
    return BookSnapshot(
        token,
        () if bid is None else (Level(bid, size),),
        () if ask is None else (Level(ask, size),),
    )


def mksnap(
    ts: float,
    up: tuple[float | None, float | None] = (0.49, 0.51),
    down: tuple[float | None, float | None] = (0.49, 0.51),
    *,
    market: MarketSpec = MKT,
    spot: float | None = 100.0,
    spot_age: float | None = 0.0,
) -> MarketSnapshot:
    return MarketSnapshot(
        ts=ts,
        market=market,
        up_book=mkbook(UP, *up),
        down_book=mkbook(DOWN, *down),
        spot=spot,
        spot_ts=None if spot_age is None else ts - spot_age,
        ref_price=100.0,
    )


class FakeExchange:
    """Minimal exchange: validates like the real one and applies fills to the engine."""

    def __init__(self, engine: MarketMakerEngine, *, delay_ioc: bool = False) -> None:
        self.engine = engine
        self.orders: dict[str, OpenOrder] = {}
        self.rejected: list[tuple[str, str]] = []
        self.fill_count = 0
        self.delay_ioc = delay_ioc  # IOC orders stay in flight until the next step()

    def open_orders(self, market_id: str | None = None) -> list[OpenOrder]:
        return [o for o in self.orders.values() if market_id in (None, o.market_id)]

    def reserved(self, market_id: str | None = None) -> float:
        return sum(o.price * o.remaining for o in self.open_orders(market_id) if o.side is BUY)

    def execute_in_flight_iocs(self, ts: float) -> None:
        """Execute (and remove) every in-flight IOC order at its limit price."""
        for oid in [oid for oid, o in self.orders.items() if o.tif is IOC]:
            o = self.orders.pop(oid)
            self._fill(o.market_id, o.token, o.side, o.price, o.remaining, False, ts)

    def step(self, snap: MarketSnapshot, extra: Sequence[OpenOrder] = ()) -> list[Action]:
        self.execute_in_flight_iocs(snap.ts)
        actions = self.engine.on_snapshot(
            snap, self.open_orders(snap.market.market_id) + list(extra)
        )
        self.apply(actions, snap)
        return actions

    def apply(self, actions: Sequence[Action], snap: MarketSnapshot) -> None:
        for a in actions:
            if isinstance(a, PlaceOrder):
                self.submit(a.request, snap)
            elif isinstance(a, CancelOrder):
                self.orders.pop(a.order_id, None)
            else:
                inv = self.engine.portfolio.inventory(a.market_id)
                if a.size <= 0 or a.size > inv.paired_qty + 1e-9:
                    self.rejected.append(("merge", f"{a.size} > {inv.paired_qty}"))
                    continue
                self.engine.on_merge(MergeResult(a.market_id, a.size, a.size, snap.ts))

    def submit(self, req: OrderRequest, snap: MarketSnapshot) -> None:
        tick, min_size = snap.market.tick_size, snap.market.min_order_size
        book = snap.book(req.token)

        def reject(why: str) -> None:
            self.rejected.append((req.client_id, why))

        if not (0.0 < req.price < 1.0) or abs(req.price / tick - round(req.price / tick)) > 1e-6:
            return reject("price")
        if req.size < min_size - 1e-9:
            return reject("size")
        fees = self.engine.fees
        if req.side is BUY:
            need = req.price * req.size
            if req.tif is IOC:
                need += fees.fee(req.price, req.size, False)
            if need > self.engine.portfolio.cash - self.reserved() + 1e-9:
                return reject("cash")
            if req.tif is POST:
                if book.best_ask is None or req.price >= book.best_ask - EPS:
                    return reject("crosses")
                self.orders["o" + req.client_id] = OpenOrder(
                    "o" + req.client_id, req.client_id, req.market_id, req.token, BUY,
                    req.price, req.size, req.size, POST, snap.ts,
                )  # fmt: skip
                return
            if self.delay_ioc:
                self.orders["o" + req.client_id] = OpenOrder(
                    "o" + req.client_id, req.client_id, req.market_id, req.token, BUY,
                    req.price, req.size, req.size, IOC, snap.ts, live=False,
                )  # fmt: skip
                return
            self._fill(req.market_id, req.token, BUY, req.price, req.size, False, snap.ts)
        else:
            held = self.engine.portfolio.inventory(req.market_id).qty[req.token]
            if req.size > held + 1e-9:
                return reject("oversell")
            self._fill(req.market_id, req.token, SELL, req.price, req.size, False, snap.ts)

    def _fill(
        self,
        mid: str,
        token: Outcome,
        side: Side,
        price: float,
        size: float,
        maker: bool,
        ts: float,
    ) -> Fill:
        self.fill_count += 1
        fee = self.engine.fees.fee(price, size, maker)
        fill = Fill(f"f{self.fill_count}", "o", mid, token, side, price, size, fee, maker, ts)
        self.engine.on_fill(fill)
        return fill

    def maker_fill(self, order_id: str, size: float | None = None, ts: float = 0.0) -> Fill:
        o = self.orders[order_id]
        qty = o.remaining if size is None else size
        fill = self._fill(o.market_id, o.token, BUY, o.price, qty, True, ts)
        o.remaining -= qty
        if o.remaining <= EPS:
            del self.orders[order_id]
        return fill


def make_env(
    cfg: BotConfig = CFG,
    *,
    p_up: float = 0.5,
    valid: bool = True,
    initial_cash: float | None = None,
) -> tuple[MarketMakerEngine, StubModel, FakeExchange]:
    engine = MarketMakerEngine(cfg, initial_cash=initial_cash)
    stub = StubModel(p_up, valid)
    engine.model = stub  # type: ignore[assignment]
    return engine, stub, FakeExchange(engine)


def give(engine: MarketMakerEngine, up: float = 0.0, down: float = 0.0, *, market: str = "m1",
         up_px: float = 0.47, down_px: float = 0.52) -> None:  # fmt: skip
    """Put inventory into the portfolio by direct maker fills (no resting order involved)."""
    for token, qty, px in ((UP, up, up_px), (DOWN, down, down_px)):
        if qty > 0:
            engine.on_fill(Fill("g", "o", market, token, BUY, px, qty, 0.0, True, 1.0))


def places(actions: Sequence[Action]) -> list[OrderRequest]:
    return [a.request for a in actions if isinstance(a, PlaceOrder)]


def quote_view(actions: Sequence[Action]) -> list[tuple[str, float, float]]:
    return [(r.token.value, r.price, r.size) for r in places(actions) if r.tif is POST]


def cancels(actions: Sequence[Action]) -> list[str]:
    return [a.order_id for a in actions if isinstance(a, CancelOrder)]


FULL_LADDER = [
    ("UP", 0.49, 100.0),
    ("UP", 0.48, 70.0),
    ("UP", 0.47, 49.0),
    ("DOWN", 0.49, 100.0),
    ("DOWN", 0.48, 70.0),
    ("DOWN", 0.47, 49.0),
]


# --------------------------------------------------------------------------- quoting basics


def test_accumulate_emits_a_two_sided_ladder_with_deterministic_ids() -> None:
    engine, _, ex = make_env()
    actions = ex.step(mksnap(ACC_TS))
    assert quote_view(actions) == FULL_LADDER
    reqs = places(actions)
    assert [r.client_id for r in reqs] == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert all(r.side is BUY and r.tif is POST and r.market_id == "m1" for r in reqs)
    # Top caps sum to <= 1 - target_margin: 0.49 + 0.49 = 0.98 <= 0.99.
    top = {t: max(r.price for r in reqs if r.token is t) for t in (UP, DOWN)}
    assert top[UP] + top[DOWN] <= 1 - CFG.pair.target_margin + EPS
    assert engine.stats["quotes_placed"] == 6
    assert ex.rejected == []
    # Ids keep counting across snapshots. The UP touch drops to 0.48 (UP orders at 0.49/0.48/0.47
    # are stale, desired 0.48 x 100, 0.47 x 70, 0.46 x 49) and the DOWN touch rises to 0.50:
    # DOWN caps at 0.495 -> 0.49 (L1 duplicates it, dropped), then 0.48 x 49. The existing
    # DOWN 0.49 x 100 (oc4) still matches and is kept; oc5 (0.48 x 70) and oc6 are stale.
    later = ex.step(mksnap(ACC_TS + 1, up=(0.48, 0.50), down=(0.50, 0.52)))
    assert cancels(later) == ["oc1", "oc2", "oc3", "oc5", "oc6"]
    assert quote_view(later) == [
        ("UP", 0.48, 100.0),
        ("UP", 0.47, 70.0),
        ("UP", 0.46, 49.0),
        ("DOWN", 0.48, 49.0),
    ]
    assert [r.client_id for r in places(later)] == ["c7", "c8", "c9", "c10"]


def test_steady_state_keeps_matching_orders() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    again = ex.step(mksnap(ACC_TS + 1))
    assert again == []
    assert len(ex.orders) == 6
    assert engine.stats["quotes_cancelled"] == 0
    assert engine.stats["quotes_placed"] == 6


def test_a_fill_reprices_the_heavy_side_and_keeps_the_light_side() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    ex.maker_fill("oc2")  # UP 0.48 x 70 fully filled -> inventory UP 70 @ 0.48 (net +70)
    actions = ex.step(mksnap(ACC_TS + 1))
    assert engine.portfolio.inventory("m1").qty[UP] == 70.0
    # UP is heavy by 70: skew 0.7 ticks, net room 300 - 70 = 230. Levels:
    #   0.49 - 0.007 = 0.483 -> 0.48 (100), 0.48 - 0.007 = 0.473 -> 0.47 (70), -> 0.46 (49).
    # Existing UP orders: oc1 0.49 x 100 (above 0.48: stale), oc3 0.47 x 49 (desired 70: stale).
    # DOWN completes at cap 0.99 - 0.48 = 0.51 (above the 0.49 touch): ladder unchanged, so the
    # three resting DOWN orders oc4..oc6 match exactly and stay.
    assert cancels(actions) == ["oc1", "oc3"]
    assert quote_view(actions) == [("UP", 0.48, 100.0), ("UP", 0.47, 70.0), ("UP", 0.46, 49.0)]
    assert ex.rejected == []


def test_up_fill_turns_the_down_cap_into_the_pair_cap() -> None:
    # Overlay off: with fair 0.5 and an UP ask of 0.45 the overlay would (rightly) see a 4c edge.
    engine, _, ex = make_env(NO_DIRECTIONAL)
    up_book, down_book = (0.43, 0.45), (0.55, 0.57)
    first = ex.step(mksnap(ACC_TS, up=up_book, down=down_book))
    # fair 0.5 -> cap 0.495 on both tokens. UP hangs from its touch 0.43; the DOWN touch (0.55) is
    # above the cap, so every DOWN level clamps to 0.49 and only one level survives.
    assert quote_view(first) == [
        ("UP", 0.43, 100.0),
        ("UP", 0.42, 70.0),
        ("UP", 0.41, 49.0),
        ("DOWN", 0.49, 100.0),
    ]
    give(engine, up=100)  # UP 100 @ 0.47 filled
    second = ex.step(mksnap(ACC_TS + 1, up=up_book, down=down_book))
    # DOWN cap = 1 - 0.01 - 0.47 = 0.52 (above its old 0.495): one level, 0.52 x 100.
    # UP is heavy by 100: 1 tick of skew (0.43 -> 0.42) and a net room of 300 - 100 = 200:
    # sizes 100, 70, 30 (the third level is trimmed to the room left).
    assert cancels(second) == ["oc1", "oc2", "oc3", "oc4"]  # every old order is stale
    assert quote_view(second) == [
        ("UP", 0.42, 100.0),
        ("UP", 0.41, 70.0),
        ("UP", 0.40, 30.0),
        ("DOWN", 0.52, 100.0),
    ]
    assert 1 - CFG.pair.target_margin + EPS >= 0.52 + 0.47  # DESIGN invariant 8
    assert ex.rejected == []


def test_directional_overlay_sees_a_cheap_ask_and_skews_the_other_side() -> None:
    # Same books but with the overlay on: UP ask 0.45 vs fair 0.5 -> edge 0.5 - (0.45 + fee 0.00689)
    # = 0.0431 >= 0.03, Kelly shares far above the cap -> target +200. UP is favoured; DOWN is
    # 200 shares "heavy relative to the target": 2 ticks of skew, and only one DOWN level remains.
    _, _, ex = make_env()
    actions = ex.step(mksnap(ACC_TS, up=(0.43, 0.45), down=(0.55, 0.57)))
    assert quote_view(actions) == [
        ("UP", 0.43, 100.0),
        ("UP", 0.42, 70.0),
        ("UP", 0.41, 49.0),
        ("DOWN", 0.47, 100.0),  # 0.495 -> 0.49 minus 2 ticks of skew
    ]


def test_wind_down_quotes_only_the_completing_token_and_cancels_the_rest() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))  # 6 resting orders c1..c6
    give(engine, up=80)  # unpaired UP 80 @ 0.47
    actions = ex.step(mksnap(WD_TS))
    assert engine.phase_of("m1") is Phase.WIND_DOWN
    # UP is not allowed at all (cancel); the DOWN orders no longer match the single wanted level
    # (DOWN 0.49 x 80: room = the 80 unpaired shares), so all six are cancelled and one is placed.
    assert cancels(actions) == [f"oc{i}" for i in range(1, 7)]
    assert quote_view(actions) == [("DOWN", 0.49, 80.0)]
    # Balanced inventory in wind-down: nothing to complete, nothing quoted, resting orders pulled.
    engine2, _, ex2 = make_env()
    ex2.step(mksnap(ACC_TS))
    give(engine2, up=80, down=80)
    actions = ex2.step(mksnap(WD_TS))
    assert places(actions) == []
    assert len(cancels(actions)) == 6


def test_wind_down_with_a_directional_favoured_token() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(min_edge=0.0))
    engine, _, ex = make_env(cfg, p_up=0.98)  # strong UP view: target +200
    actions = ex.step(mksnap(WD_TS))
    # Flat inventory: UP is favoured (room = the 200 wanted shares), DOWN (fair 0.02) is not.
    ups = [r for r in places(actions) if r.token is UP]
    downs = [r for r in places(actions) if r.token is DOWN]
    assert sum(r.size for r in ups) == 200.0
    assert downs == []
    assert engine.stats["taker_completions"] == 0


# --------------------------------------------------------------------------- warm-up / model


def test_warmup_emits_nothing_even_with_a_valid_model() -> None:
    cfg = replace(CFG, timing=TimingConfig(warmup_seconds=60.0))
    engine, _, ex = make_env(cfg)
    for ts in range(1000, 1060):
        assert ex.step(mksnap(float(ts))) == []
        assert engine.phase_of("m1") is Phase.WARMUP
    assert quote_view(ex.step(mksnap(1060.0))) == FULL_LADDER  # first ACCUMULATE snapshot


def test_invalid_model_and_flat_inventory_emit_nothing_but_pairs_are_still_completed() -> None:
    engine, stub, ex = make_env(valid=False)
    assert ex.step(mksnap(ACC_TS)) == []
    # With unpaired UP the DOWN completion bid is still quoted (it needs no model).
    give(engine, up=80)
    actions = ex.step(mksnap(ACC_TS + 1))
    assert {r.token for r in places(actions)} == {DOWN}
    assert stub.valid is False


# --------------------------------------------------------------------------- merge


def test_merge_is_emitted_at_the_threshold_and_before_new_quotes() -> None:
    engine, _, ex = make_env()
    give(engine, up=49, down=49)
    below = ex.step(mksnap(ACC_TS))
    assert not any(isinstance(a, MergePairs) for a in below)
    give(engine, up=1, down=1)  # 50 pairs (UP now 50 @ avg 0.47)
    actions = ex.step(mksnap(ACC_TS + 1))
    merges = [a for a in actions if isinstance(a, MergePairs)]
    assert merges == [MergePairs("m1", 50.0)]
    assert engine.stats["merges"] == 1
    # The merge is applied by the harness through on_merge: inventory empties, cash is freed.
    inv = engine.portfolio.inventory("m1")
    assert inv.qty == {UP: 0.0, DOWN: 0.0}
    assert engine.portfolio.cash == pytest.approx(10_000.0 - 50 * 0.47 - 50 * 0.52 + 50.0)
    # Nothing is merged twice.
    assert not any(isinstance(a, MergePairs) for a in ex.step(mksnap(ACC_TS + 2)))


def test_merge_floors_fractional_pairs_and_comes_first_in_the_action_list() -> None:
    engine, _, ex = make_env()
    give(engine, up=50.6, down=52.1)
    actions = ex.step(mksnap(ACC_TS))
    assert actions[0] == MergePairs("m1", 50.0)
    assert ex.rejected == []


def test_merge_disabled_never_merges() -> None:
    cfg = replace(CFG, pair=PairConfig(merge_enabled=False))
    engine, _, ex = make_env(cfg)
    give(engine, up=80, down=80)
    assert not any(isinstance(a, MergePairs) for a in ex.step(mksnap(ACC_TS)))
    assert not any(isinstance(a, MergePairs) for a in ex.step(mksnap(FL_TS)))


# --------------------------------------------------------------------------- flatten / done


def resting_ladder_env(
    cfg: BotConfig = CFG, p_up: float = 0.5
) -> tuple[MarketMakerEngine, StubModel, FakeExchange]:
    engine, stub, ex = make_env(cfg, p_up=p_up)
    ex.step(mksnap(ACC_TS))
    assert len(ex.orders) == 6
    return engine, stub, ex


def test_flatten_cancels_everything_merges_and_sells_without_edge() -> None:
    engine, stub, ex = resting_ladder_env()
    give(engine, up=100, down=60)  # 60 pairs + 40 unpaired UP
    stub.p_up = 0.50  # exit net 0.49 - 0.00765 = 0.48235; 0.50 - 0.48235 < hold_margin 0.02
    actions = ex.step(mksnap(FL_TS))
    assert engine.phase_of("m1") is Phase.FLATTEN
    assert cancels(actions) == [f"oc{i}" for i in range(1, 7)]
    assert actions[6] == MergePairs("m1", 60.0)
    sells = [r for r in places(actions) if r.side is SELL]
    assert [(r.token, r.price, r.size, r.tif) for r in sells] == [(UP, 0.49, 40.0, IOC)]
    assert [r for r in places(actions) if r.side is BUY] == []  # no quotes in FLATTEN
    assert engine.stats["flatten_sells"] == 1
    assert engine.stats["holds"] == 0
    assert ex.rejected == []
    inv = engine.portfolio.inventory("m1")
    assert inv.qty == {UP: 0.0, DOWN: 0.0}  # merged and sold in the harness


def test_flatten_holds_the_remainder_when_the_model_has_edge() -> None:
    engine, stub, ex = resting_ladder_env()
    give(engine, up=100, down=60)
    stub.p_up = 0.55  # 0.55 - 0.48235 = 0.0676 >= 0.02 -> hold the 40 unpaired UP
    actions = ex.step(mksnap(FL_TS))
    assert len(cancels(actions)) == 6
    assert actions[6] == MergePairs("m1", 60.0)
    assert [r for r in places(actions)] == []
    assert engine.stats["holds"] == 1
    assert engine.stats["flatten_sells"] == 0
    assert engine.portfolio.inventory("m1").qty == {UP: 40.0, DOWN: 0.0}
    # The hold is counted once per market even though FLATTEN lasts many snapshots.
    ex.step(mksnap(FL_TS + 1))
    assert engine.stats["holds"] == 1


def test_flatten_with_overlay_disabled_sells_the_remainder() -> None:
    cfg = replace(CFG, directional=DirectionalConfig(enabled=False))
    engine, _, ex = make_env(cfg, p_up=0.95)
    give(engine, up=40)
    actions = ex.step(mksnap(FL_TS))
    assert [(r.token, r.size) for r in places(actions)] == [(UP, 40.0)]


def test_flatten_with_a_stale_spot_treats_the_model_as_invalid() -> None:
    engine, _, ex = make_env(p_up=0.95)
    give(engine, up=40)
    actions = ex.step(mksnap(FL_TS, spot_age=30.0))  # spot 30 s old: the p_up = 0.95 is not trusted
    assert [(r.token, r.size, r.side) for r in places(actions)] == [(UP, 40.0, SELL)]
    assert engine.stats["risk_trips_spot_stale"] == 1


def test_flatten_with_no_bid_holds() -> None:
    engine, _, ex = make_env(p_up=0.1)
    give(engine, up=40)
    actions = ex.step(mksnap(FL_TS, up=(None, 0.51)))
    assert places(actions) == []
    assert engine.stats["holds"] == 1


def test_flatten_does_not_resend_a_pending_ioc_sell() -> None:
    engine, _, ex = make_env(p_up=0.5)
    give(engine, up=40)
    pending = OpenOrder("ox", "cx", "m1", UP, SELL, 0.49, 40.0, 40.0, IOC, FL_TS, live=False)
    actions = engine.on_snapshot(mksnap(FL_TS), [pending])
    assert actions == []  # the 40 shares are already being sold


def test_done_phase_only_cancels_stragglers() -> None:
    engine, _, ex = make_env()
    give(engine, up=100, down=100)
    straggler = OpenOrder("ox", "cx", "m1", UP, BUY, 0.49, 100.0, 100.0, POST, 1.0)
    actions = engine.on_snapshot(mksnap(1900.0), [straggler])
    assert actions == [CancelOrder("ox")]
    assert engine.phase_of("m1") is Phase.DONE
    assert engine.on_snapshot(mksnap(1901.0), []) == []


# --------------------------------------------------------------------------- risk gate


def test_kill_switch_latches_cancels_everything_and_stops_quoting() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_daily_loss_usd=400.0))
    engine, stub, ex = make_env(cfg)
    ex.step(mksnap(ACC_TS))  # 6 resting orders
    give(engine, up=1000, up_px=0.5)  # cost 500, marked at p_up 0.5 -> equity unchanged
    ex.step(mksnap(ACC_TS + 1))
    assert not engine.risk.kill_switch
    # p_up -> 0.02: position worth 20, equity = 10000 - 500 + 20 = 9520, loss 480 >= 400.
    stub.p_up = 0.02
    actions = ex.step(mksnap(ACC_TS + 2))
    assert engine.risk.kill_switch
    assert places(actions) == []
    # Every remaining resting order is pulled. (After the 1000-share UP position the UP side was
    # over the hard stop, so oc1..oc3 were already cancelled; the DOWN ladder oc4..oc6 remained.)
    assert cancels(actions) == ["oc4", "oc5", "oc6"]
    assert ex.open_orders() == []
    assert engine.stats["risk_trips_kill_switch"] == 1
    # The loss recovers, but the switch stays tripped: no new orders ever.
    stub.p_up = 0.5
    for dt in range(3, 8):
        assert places(ex.step(mksnap(ACC_TS + dt))) == []
    assert engine.risk.kill_switch
    assert engine.stats["risk_trips_kill_switch"] == 1  # counted once, when it latched


def test_kill_switch_still_allows_merges() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_daily_loss_usd=400.0))
    engine, stub, ex = make_env(cfg)
    give(engine, up=1000, up_px=0.5)
    stub.p_up = 0.02
    ex.step(mksnap(ACC_TS))
    assert engine.risk.kill_switch
    give(engine, down=60, down_px=0.02)  # 60 pairs now exist
    actions = ex.step(mksnap(ACC_TS + 1))
    assert actions == [MergePairs("m1", 60.0)]


def test_stale_spot_cancels_quotes_and_recovers() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    actions = ex.step(mksnap(ACC_TS + 1, spot_age=10.0))  # 10 s > max_spot_staleness 5 s
    assert places(actions) == []
    assert cancels(actions) == [f"oc{i}" for i in range(1, 7)]
    assert ex.open_orders() == []
    assert engine.stats["risk_trips_spot_stale"] == 1
    still = ex.step(mksnap(ACC_TS + 2, spot_age=10.0))
    assert still == []
    assert engine.stats["risk_trips_spot_stale"] == 1  # one continuous trip
    back = ex.step(mksnap(ACC_TS + 3))
    assert quote_view(back) == FULL_LADDER
    assert engine.stats["risk_blocked_snapshots"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"spot": None},
        {"spot_age": None},  # no timestamp
        {"spot_age": 5.01},  # just past the limit
    ],
)
def test_missing_or_old_spot_blocks_quoting(bad: dict[str, float | None]) -> None:
    engine, _, ex = make_env()
    assert ex.step(mksnap(ACC_TS, **bad)) == []  # type: ignore[arg-type]
    assert engine.stats["risk_trips_spot_stale"] == 1


def test_spot_age_exactly_at_the_limit_is_fine() -> None:
    engine, _, ex = make_env()
    assert quote_view(ex.step(mksnap(ACC_TS, spot_age=5.0))) == FULL_LADDER


@pytest.mark.parametrize(
    ("up", "down"),
    [
        ((0.51, 0.49), (0.49, 0.51)),  # crossed
        ((None, 0.51), (0.49, 0.51)),  # empty side
        ((0.46, 0.53), (0.47, 0.54)),  # 7 ticks wide
    ],
)
def test_bad_books_cancel_quotes(
    up: tuple[float | None, float | None], down: tuple[float | None, float | None]
) -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    actions = ex.step(mksnap(ACC_TS + 1, up=up, down=down))
    assert places(actions) == []
    assert len(cancels(actions)) == 6
    assert engine.stats["risk_trips_book"] == 1


def test_cost_basis_over_the_market_cap_trips_the_capital_gate() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_capital_per_market_usd=40.0))
    engine, _, ex = make_env(cfg)
    give(engine, up=100, up_px=0.5)  # cost basis 50 > 40
    actions = ex.step(mksnap(ACC_TS))
    assert places(actions) == []
    assert engine.stats["risk_trips_capital"] == 1


# --------------------------------------------------------------------------- reconcile


def test_reconcile_keeps_orders_within_the_size_window_and_below_the_price() -> None:
    engine, _, _ = make_env()
    mk = lambda oid, token, px, rem: OpenOrder(  # noqa: E731
        oid, "c" + oid, "m1", token, BUY, px, 100.0, rem, POST, 1.0
    )
    existing = [
        mk("a", UP, 0.49, 100.0),  # exact match of UP L0 (100)
        mk("b", UP, 0.48, 60.0),  # desired 70: 60 is within [56, 70] -> kept
        mk("c", UP, 0.47, 55.0),  # desired 49: 55 > 49 -> stale (never keep an oversized order)
        mk("d", DOWN, 0.49, 55.0),  # desired 100: 55 < 80 -> stale
        mk("e", DOWN, 0.50, 100.0),  # above every DOWN price wanted -> stale
        mk("f", DOWN, 0.48, 70.0),  # exact match of DOWN L1 (70)
    ]
    actions = engine.on_snapshot(mksnap(ACC_TS), existing)
    assert sorted(cancels(actions)) == ["c", "d", "e"]
    # Missing: UP .47 (49), DOWN .49 (100) and DOWN .47 (49).
    assert quote_view(actions) == [("UP", 0.47, 49.0), ("DOWN", 0.49, 100.0), ("DOWN", 0.47, 49.0)]
    assert actions.index(CancelOrder("e")) < actions.index(PlaceOrder(places(actions)[0]))


def test_reconcile_tolerance_keeps_orders_priced_below_the_desired_level() -> None:
    cfg = replace(CFG, sizing=SizingConfig(requote_tolerance_ticks=1))
    engine, _, _ = make_env(cfg)
    mk = lambda oid, token, px, rem: OpenOrder(  # noqa: E731
        oid, "c" + oid, "m1", token, BUY, px, 100.0, rem, POST, 1.0
    )
    existing = [
        mk("a", UP, 0.48, 100.0),  # one tick below desired 0.49: within tolerance -> kept
        mk("b", UP, 0.46, 70.0),  # two ticks below desired 0.48: out of tolerance
        mk("c", DOWN, 0.50, 100.0),  # one tick ABOVE desired 0.49: never kept
    ]
    actions = engine.on_snapshot(mksnap(ACC_TS), existing)
    assert sorted(cancels(actions)) == ["b", "c"]
    assert ("UP", 0.49, 100.0) not in quote_view(actions)  # L0 satisfied by order a
    assert ("DOWN", 0.49, 100.0) in quote_view(actions)


def test_requote_throttle_defers_discretionary_changes() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    ex.maker_fill("oc1")  # UP 0.49 x 100 filled: a gap in the ladder
    deferred = ex.step(mksnap(ACC_TS + 0.5))  # 0.5 s < min_requote_interval 1.0
    assert deferred == []
    assert engine.stats["requotes_throttled"] == 1
    later = ex.step(mksnap(ACC_TS + 1.0))  # the interval has elapsed
    assert quote_view(later)  # the ladder is repaired
    assert engine.stats["requotes_throttled"] == 1


def test_requote_throttle_never_delays_cancelling_an_order_above_the_cap() -> None:
    engine, _, ex = make_env()
    up_book, down_book = (0.43, 0.45), (0.55, 0.57)
    ex.step(mksnap(ACC_TS, up=up_book, down=down_book))
    give(engine, up=100)  # DOWN cap drops to the pair cap 0.52
    rogue = OpenOrder("rogue", "cr", "m1", DOWN, BUY, 0.53, 100.0, 100.0, POST, 1.0)  # > 0.52
    actions = ex.step(mksnap(ACC_TS + 0.5, up=up_book, down=down_book), extra=[rogue])
    # Inside the throttle window only the order priced above the new cap is cancelled.
    assert actions == [CancelOrder("rogue")]
    assert engine.stats["requotes_throttled"] == 1


def test_reconcile_ignores_orders_of_other_markets() -> None:
    engine, _, _ = make_env()
    foreign = OpenOrder("zz", "cz", "m2", UP, BUY, 0.49, 100.0, 100.0, POST, 1.0)
    actions = engine.on_snapshot(mksnap(ACC_TS), [foreign])
    assert "zz" not in cancels(actions)
    assert quote_view(actions) == FULL_LADDER  # and it does not satisfy m1's ladder


# --------------------------------------------------------------------------- rate limiter


def test_rate_limiter_places_level_major_and_never_blocks_cancels() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_orders_per_second=4.0))
    engine, _, ex = make_env(cfg)
    first = ex.step(mksnap(ACC_TS))
    # Priority is level-major: UP L0, DOWN L0, UP L1, DOWN L1 (emitted UP before DOWN).
    assert quote_view(first) == [
        ("UP", 0.49, 100.0),
        ("UP", 0.48, 70.0),
        ("DOWN", 0.49, 100.0),
        ("DOWN", 0.48, 70.0),
    ]
    assert engine.stats["orders_rate_limited"] == 2
    # One second later the window is empty: the two deferred level-2 orders go out.
    second = ex.step(mksnap(ACC_TS + 1))
    assert quote_view(second) == [("UP", 0.47, 49.0), ("DOWN", 0.47, 49.0)]
    assert ex.step(mksnap(ACC_TS + 2)) == []
    assert engine.stats["quotes_placed"] == 6

    # Cancels are never refused even when the window is full: 6 resting orders become stale,
    # all 6 are cancelled and (the window now being over capacity) nothing is placed.
    engine2, _, ex2 = make_env(cfg)
    ex2.step(mksnap(ACC_TS))
    ex2.step(mksnap(ACC_TS + 1))
    give(engine2, up=80)
    actions = ex2.step(mksnap(WD_TS))
    assert len(cancels(actions)) == 6
    assert engine2.stats["quotes_cancelled"] == 6
    assert places(actions) == []  # 6 cancels exceed the cap of 4 within the window
    assert engine2.stats["orders_rate_limited"] == 2 + 1  # 2 earlier + the DOWN order dropped


def test_taker_orders_count_against_the_rate_limit_and_come_first() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_orders_per_second=2.0))
    engine, _, ex = make_env(cfg)
    give(engine, up=150)  # unpaired 150 UP -> taker completion of DOWN
    actions = ex.step(mksnap(ACC_TS))
    reqs = places(actions)
    assert (reqs[0].token, reqs[0].tif, reqs[0].side) == (DOWN, IOC, BUY)  # taker first
    assert len(reqs) == 2  # 2 per second in total: the completion + one quote
    assert engine.stats["taker_completions"] == 1
    assert engine.stats["orders_rate_limited"] >= 1


# --------------------------------------------------------------------------- capital & cash


def test_per_market_capital_cap_limits_the_ladder() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_capital_per_market_usd=60.0))
    engine, _, ex = make_env(cfg)
    actions = ex.step(mksnap(ACC_TS))
    # Budget 60 - 1e-6: UP 100 @ 0.49 = 49.0, then (59.999999 - 49) / 0.48 = 22.91 shares.
    assert quote_view(actions) == [("UP", 0.49, 100.0), ("UP", 0.48, 22.91)]
    capital = ex.reserved("m1") + engine.portfolio.inventory("m1").capital_at_risk()
    assert capital == pytest.approx(49.0 + 0.48 * 22.91, abs=1e-9)
    assert capital <= 60.0
    assert ex.rejected == []


def test_total_capital_cap_is_shared_across_markets() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_total_capital_usd=100.0))
    engine, _, ex = make_env(cfg)
    a = ex.step(mksnap(ACC_TS))
    # Market m1 consumes the whole 100 USDC budget with UP levels: 49 + 33.6 + 0.47 * 37.02.
    assert quote_view(a) == [("UP", 0.49, 100.0), ("UP", 0.48, 70.0), ("UP", 0.47, 37.02)]
    reserved_a = 49.0 + 33.6 + 0.47 * 37.02
    assert ex.reserved("m1") == pytest.approx(reserved_a)
    # Market m2 sees the same budget minus m1's believed reservation: 0.0006 USDC -> nothing.
    b = ex.step(mksnap(ACC_TS, market=MKT2))
    assert places(b) == []
    assert ex.reserved() <= 100.0
    assert ex.rejected == []


def test_a_market_budget_frees_up_when_the_other_market_cancels() -> None:
    cfg = replace(CFG, risk=RiskConfig(max_total_capital_usd=100.0))
    engine, _, ex = make_env(cfg)
    late = MarketSpec("m3", "ETH", start_ts=1000.0, end_ts=2800.0)  # still ACCUMULATE at 1880
    ex.step(mksnap(ACC_TS))  # m1 reserves ~100 of the 100 USDC total cap
    assert places(ex.step(mksnap(ACC_TS, market=late))) == []
    # At 1880 m1 is in FLATTEN: its ladder is cancelled and its reservation belief drops to 0.
    assert len(cancels(ex.step(mksnap(FL_TS)))) == 3
    assert ex.reserved("m1") == 0.0
    # Now m3 may use the whole budget: the same ladder m1 had (49 + 33.6 + 0.47 * 37.02).
    freed = ex.step(mksnap(FL_TS, market=late))
    assert [(r.token.value, r.price, r.size) for r in places(freed)] == [
        ("UP", 0.49, 100.0),
        ("UP", 0.48, 70.0),
        ("UP", 0.47, 37.02),
    ]
    assert ex.rejected == []


def test_cash_limited_ladder_does_not_flip_flop() -> None:
    # Only 120 USDC of cash: the desired ladder is cash-bound. Counting the market's own resting
    # orders against the budget would cancel them all on the next tick; they must be kept instead.
    engine, _, ex = make_env(initial_cash=120.0)
    first = ex.step(mksnap(ACC_TS))
    reserved = ex.reserved()
    assert 100.0 < reserved <= 120.0
    for dt in range(1, 6):
        assert ex.step(mksnap(ACC_TS + dt)) == []
    assert ex.reserved() == reserved
    assert ex.rejected == []
    assert quote_view(first)


def test_equal_budget_after_a_partial_fill_never_overdraws() -> None:
    engine, _, ex = make_env(initial_cash=150.0)
    ex.step(mksnap(ACC_TS))
    ex.maker_fill("oc1", size=40.0)  # partial fill of UP 0.49 x 100
    ex.step(mksnap(ACC_TS + 1))
    assert ex.rejected == []
    assert engine.portfolio.cash - ex.reserved() >= -1e-6


# --------------------------------------------------------------------------- taker completion


def test_taker_completion_is_emitted_and_the_token_is_not_also_quoted() -> None:
    engine, _, ex = make_env()
    give(engine, up=150)  # unpaired UP 150 @ 0.47
    actions = ex.step(mksnap(ACC_TS))
    reqs = places(actions)
    # DOWN ask 0.51 (all-in 0.517962): net profit 0.012038 >= 0.005 -> IOC BUY DOWN 150 @ 0.51.
    assert (reqs[0].token, reqs[0].side, reqs[0].tif, reqs[0].price, reqs[0].size) == (
        DOWN, BUY, IOC, 0.51, 150.0,
    )  # fmt: skip
    assert reqs[0].client_id == "c1"
    assert engine.stats["taker_completions"] == 1
    # UP quotes continue (skew 1.5 ticks, net room 150): 0.47 x 100, 0.46 x 50; no DOWN quotes.
    assert quote_view(actions) == [("UP", 0.47, 100.0), ("UP", 0.46, 50.0)]
    assert ex.rejected == []
    # The harness filled the IOC immediately: the pair is complete and merged next tick.
    inv = engine.portfolio.inventory("m1")
    assert inv.qty[DOWN] == 150.0


def test_completion_cost_is_set_aside_before_quoting() -> None:
    # Cash 200; UP 150 @ 0.47 costs 70.5 -> cash 129.5. The IOC completion (150 @ 0.51 + fee
    # 0.00796 = 0.517962 each = 77.694 USDC) is committed first, leaving 51.80564 for quotes:
    # UP 0.47 x 100 = 47.0, then (51.80564 - 47) / 0.46 = 10.447 -> 10.44 shares at 0.46.
    # Without the reservation the ladder (70 USDC) would overdraw the account.
    engine, _, ex = make_env(initial_cash=200.0)
    give(engine, up=150)
    actions = ex.step(mksnap(ACC_TS))
    assert [(r.token, r.tif, r.size) for r in places(actions)][0] == (DOWN, IOC, 150.0)
    assert quote_view(actions) == [("UP", 0.47, 100.0), ("UP", 0.46, 10.44)]
    assert ex.rejected == []
    assert engine.portfolio.cash - ex.reserved() >= -1e-9


def test_other_markets_see_the_taker_fee_of_an_in_flight_completion() -> None:
    # Total capital cap 260. m1 holds UP 150 @ 0.47 (cost 70.5) and sends an IOC DOWN 150 @ 0.51
    # that stays in flight: it ties up 76.5 + worst-case fee 150 * 0.007962376 = 77.69436. m1's
    # ladder (UP 0.47 x 100 + 0.46 x 50 = 70) completes the belief: 147.69436.
    # m2's budget = 260 - 70.5 - 147.69436 - 1e-6 = 41.80564 -> UP 0.49 x floor(41.80564/0.49)
    # = 85.31. (Without the fee the belief would be 146.5 and m2 would size 87.75.)
    cfg = replace(CFG, risk=RiskConfig(max_total_capital_usd=260.0))
    engine, _, _ = make_env(cfg)
    ex = FakeExchange(engine, delay_ioc=True)
    give(engine, up=150)
    first = ex.step(mksnap(ACC_TS))
    assert [(r.token, r.tif, r.size) for r in places(first)][0] == (DOWN, IOC, 150.0)
    assert quote_view(first) == [("UP", 0.47, 100.0), ("UP", 0.46, 50.0)]
    second = ex.step(mksnap(ACC_TS, market=MKT2))
    assert quote_view(second)[0] == ("UP", 0.49, 85.31)
    assert ex.rejected == []


def test_in_flight_ioc_reservation_is_not_reused_by_the_new_ladder() -> None:
    # Cash 200, UP 150 @ 0.47 (cash 129.5). A DOWN IOC for 150 @ 0.51 from an earlier snapshot
    # is still in flight and holds 76.5 + fee 1.19436 = 77.69436 (it is never re-planned). The new
    # ladder may use 129.5 - 77.69436 - 1e-6 = 51.80564: UP 0.47 x 100 = 47, then 10.44 @ 0.46.
    engine, _, _ = make_env(initial_cash=200.0)
    give(engine, up=150)
    pending = OpenOrder("ox", "cx", "m1", DOWN, BUY, 0.51, 150.0, 150.0, IOC, 1.0, live=False)
    actions = engine.on_snapshot(mksnap(ACC_TS), [pending])
    assert quote_view(actions) == [("UP", 0.47, 100.0), ("UP", 0.46, 10.44)]
    assert [r for r in places(actions) if r.tif is IOC] == []  # and no second completion


def test_a_taker_fill_releases_its_reserved_fee() -> None:
    engine, _, _ = make_env()
    ex = FakeExchange(engine, delay_ioc=True)
    give(engine, up=150)
    ex.step(mksnap(ACC_TS))
    state = engine._states["m1"]
    before = state.reserved  # UP quotes 70 + IOC 76.5 + fee 1.19436
    assert before == pytest.approx(70.0 + 76.5 + 150 * 0.007962376275, abs=1e-9)
    ex.execute_in_flight_iocs(ACC_TS + 1)  # the IOC fills: price * size + fee leave the belief
    assert state.reserved == pytest.approx(70.0, abs=1e-9)


def test_no_completion_while_one_is_in_flight_or_unprofitable() -> None:
    engine, _, ex = make_env()
    give(engine, up=150)
    pending = OpenOrder("ox", "cx", "m1", DOWN, BUY, 0.51, 150.0, 150.0, IOC, 1.0, live=False)
    actions = engine.on_snapshot(mksnap(ACC_TS), [pending])
    assert [r for r in places(actions) if r.tif is IOC] == []
    # Unprofitable: DOWN ask 0.53 -> 1 - 0.47 - 0.538 < 0.005.
    engine2, _, _ = make_env()
    give(engine2, up=150)
    actions = engine2.on_snapshot(mksnap(ACC_TS, down=(0.51, 0.53)), [])
    assert [r for r in places(actions) if r.tif is IOC] == []
    assert engine2.stats["taker_completions"] == 0


def test_completion_respects_the_directional_allowance() -> None:
    engine, _, ex = make_env(p_up=0.98)  # target +200 (clipped)
    give(engine, up=250)  # 250 unpaired UP, 200 of them deliberate -> excess 50 < trigger 100
    actions = ex.step(mksnap(ACC_TS))
    assert [r for r in places(actions) if r.tif is IOC] == []


# --------------------------------------------------------------------------- directional / sizing


def test_directional_view_quotes_only_the_favoured_side() -> None:
    engine, _, ex = make_env(p_up=0.98)
    actions = ex.step(mksnap(ACC_TS))
    # fair_DOWN = 0.02 -> cap 0.015, below the 2-tick skew: no DOWN bids. UP: favoured, cap 0.975,
    # the ladder hangs from the touch.
    assert quote_view(actions) == FULL_LADDER[:3]


def test_clip_equity_fraction_compounds_with_equity() -> None:
    cfg = replace(CFG, sizing=SizingConfig(clip_equity_fraction=0.01))
    # equity 10000: clip = 10000 * 0.01 / 0.5 (the dearer mid) = 200 -> wanted 200, 140, 98 per
    # token, but the net-imbalance room (300) trims each token's ladder to 200 + 100.
    _, _, ex = make_env(cfg)
    assert quote_view(ex.step(mksnap(ACC_TS))) == [
        ("UP", 0.49, 200.0),
        ("UP", 0.48, 100.0),
        ("DOWN", 0.49, 200.0),
        ("DOWN", 0.48, 100.0),
    ]
    # equity 2000: clip 40 -> 40, 28, 19.6.
    _, _, ex2 = make_env(cfg, initial_cash=2000.0)
    assert [s for _, _, s in quote_view(ex2.step(mksnap(ACC_TS)))[:3]] == [40.0, 28.0, 19.6]
    # The clip is clamped to [min_clip 5, max_clip 1000].
    # (10^6 of equity would ask for 20000 shares; wide limits let the 1000 cap show through.)
    wide = replace(
        cfg,
        pair=PairConfig(max_net_imbalance_shares=5000.0, max_inventory_per_side_shares=6000.0),
        risk=RiskConfig(max_capital_per_market_usd=1e6, max_total_capital_usd=1e6),
    )
    _, _, ex3 = make_env(wide, initial_cash=1_000_000.0)
    assert quote_view(ex3.step(mksnap(ACC_TS)))[0][2] == 1000.0
    _, _, ex4 = make_env(cfg, initial_cash=100.0)  # clip would be 2 -> clamped up to 5
    assert [s for _, _, s in quote_view(ex4.step(mksnap(ACC_TS)))] == [5.0, 5.0]


def test_dearer_token_mid_prices_the_compounding_clip() -> None:
    cfg = replace(CFG, sizing=SizingConfig(clip_equity_fraction=0.01))
    _, _, ex = make_env(cfg)
    # UP mid 0.80 / DOWN mid 0.20: clip = 100 / 0.80 = 125 for both tokens.
    s = mksnap(ACC_TS, up=(0.79, 0.81), down=(0.19, 0.21))
    sizes = {t: sz for t, _, sz in quote_view(ex.step(s))}
    assert sizes["UP"] == 125.0


# ------------------------------------------------------------------- state, snapshots, settlement


def test_stale_and_out_of_order_snapshots_are_ignored() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    before = dict(engine.stats)
    assert engine.on_snapshot(mksnap(ACC_TS), []) == []  # same ts
    assert engine.on_snapshot(mksnap(ACC_TS - 5), []) == []  # older
    assert engine.stats["snapshots_ignored"] == 2
    assert engine.stats["snapshots"] == before["snapshots"]
    # Cross-market order: m2 at ts 1040 then m1 at ts 1035 (older than the latest seen, 1040).
    engine2, _, _ = make_env()
    engine2.on_snapshot(mksnap(1040.0, market=MKT2), [])
    assert engine2.on_snapshot(mksnap(1035.0), []) == []
    assert engine2.stats["snapshots_ignored"] == 1
    assert engine2.on_snapshot(mksnap(1040.0), []) != []  # equal ts across markets is fine


def test_spot_is_observed_once_per_asset_and_time_without_lookahead() -> None:
    engine, stub, _ = make_env()
    engine.on_snapshot(mksnap(1100.0, spot=100.0), [])
    engine.on_snapshot(mksnap(1100.0, market=MKT2, spot=3000.0), [])  # other asset, same ts
    other_btc = MarketSpec("m3", "BTC", 1000.0, 1900.0)
    engine.on_snapshot(mksnap(1100.0, market=other_btc, spot=100.0), [])  # same asset+ts: skipped
    assert stub.observed == [("BTC", 1100.0, 100.0), ("ETH", 1100.0, 3000.0)]
    # A stale spot (older stamp than already observed) and a spot from the future are not observed.
    engine.on_snapshot(mksnap(1101.0, spot=101.0, spot_age=5.0), [])  # stamp 1096 < 1100
    engine.on_snapshot(mksnap(1102.0, spot=102.0, spot_age=-3.0), [])  # stamp 1105 > 1102
    engine.on_snapshot(mksnap(1103.0, spot=None), [])
    assert stub.observed == [("BTC", 1100.0, 100.0), ("ETH", 1100.0, 3000.0)]
    engine.on_snapshot(mksnap(1104.0, spot=104.0), [])
    assert stub.observed[-1] == ("BTC", 1104.0, 104.0)
    # With no spot_ts the snapshot ts is the observation time.
    engine.on_snapshot(mksnap(1105.0, spot=105.0, spot_age=None), [])
    assert stub.observed[-1] == ("BTC", 1105.0, 105.0)


def test_nothing_is_emitted_for_a_settled_market() -> None:
    engine, _, ex = make_env()
    ex.step(mksnap(ACC_TS))
    give(engine, up=100, down=100)  # unmerged pairs
    # UP 100 @ 0.47 + DOWN 100 @ 0.52, winner UP pays 100: pair pnl = 100 * (1 - 0.47 - 0.52) = 1.
    breakdown = engine.on_settlement(Settlement("m1", UP, 1900.0, payout=100.0))
    assert breakdown.market_id == "m1"
    assert breakdown.settle_pair_pnl == pytest.approx(1.0)
    assert breakdown.settle_directional_pnl == pytest.approx(0.0)
    assert breakdown.total == pytest.approx(1.0)
    assert engine.portfolio.cash == pytest.approx(10_000.0 - 47.0 - 52.0 + 100.0)
    assert engine.marks() == {}  # market state cleared
    assert engine.equity() == pytest.approx(10_001.0)
    assert engine.on_snapshot(mksnap(1901.0), []) == []
    assert engine.on_snapshot(mksnap(ACC_TS + 5), []) == []
    assert engine.stats["settlements"] == 1
    assert engine.stats["snapshots_ignored"] == 2
    # Fills for a settled market are an error; a second settlement too.
    with pytest.raises(ValueError):
        engine.on_fill(Fill("f", "o", "m1", UP, BUY, 0.5, 10.0, 0.0, True, 1.0))
    with pytest.raises(ValueError):
        engine.on_settlement(Settlement("m1", UP, 1900.0, payout=0.0))


def test_settlement_with_a_directional_remainder() -> None:
    engine, _, _ = make_env()
    give(engine, up=100, down=60)  # 40 unpaired UP; winner DOWN -> UP shares expire worthless
    breakdown = engine.on_settlement(Settlement("m1", DOWN, 1900.0, payout=60.0))
    # cost = 47 + 31.2 = 78.2; payout 60 -> total = 60 - 78.2 = -18.2, of which the pair part is
    # 60 * (1 - 0.47 - 0.52) = 0.6 and the directional remainder is -18.8.
    assert breakdown.settle_pair_pnl == pytest.approx(0.6)
    assert breakdown.settle_directional_pnl == pytest.approx(-18.8)
    assert breakdown.total == pytest.approx(-18.2)
    assert engine.equity() == pytest.approx(10_000.0 - 78.2 + 60.0)


def test_accessors_marks_and_equity() -> None:
    engine, stub, ex = make_env(p_up=0.5)
    assert engine.last_fair("m1") is None and engine.phase_of("m1") is None
    ex.step(mksnap(ACC_TS))
    fair = engine.last_fair("m1")
    assert fair is not None and fair.p_up == 0.5 and fair.valid
    assert engine.phase_of("m1") is Phase.ACCUMULATE
    assert engine.marks() == {"m1": 0.5}
    give(engine, up=100)  # UP 100 @ 0.47 -> cash 9953, value 100 * 0.5 = 50
    assert engine.equity() == pytest.approx(9953.0 + 50.0)
    stub.p_up = 0.6
    ex.step(mksnap(ACC_TS + 1))
    assert engine.marks() == {"m1": 0.6}
    assert engine.equity() == pytest.approx(9953.0 + 60.0)


def test_invalid_model_marks_at_the_market_mid() -> None:
    engine, _, ex = make_env(valid=False, p_up=0.5)
    ex.step(mksnap(ACC_TS, up=(0.59, 0.61), down=(0.39, 0.41)))
    assert engine.marks()["m1"] == pytest.approx(0.60)
    ex.step(mksnap(ACC_TS + 1, up=(None, 0.61), down=(0.39, 0.41)))  # no UP mid: 1 - DOWN mid
    assert engine.marks()["m1"] == pytest.approx(0.60)
    ex.step(mksnap(ACC_TS + 2, up=(None, None), down=(None, None)))  # nothing: keep the last
    assert engine.marks()["m1"] == pytest.approx(0.60)


def test_constructor_validates_config_and_cash() -> None:
    bad = replace(CFG, pair=PairConfig(target_margin=0.7))
    with pytest.raises(ValueError):
        MarketMakerEngine(bad)
    engine = MarketMakerEngine(CFG, initial_cash=1234.5)
    assert engine.portfolio.cash == 1234.5
    assert engine.risk.starting_equity == 1234.5
    assert MarketMakerEngine(CFG).portfolio.cash == CFG.exchange.initial_cash
    with pytest.raises(ValueError):
        MarketMakerEngine(CFG, initial_cash=-1.0)
    with pytest.raises(ValueError, match="finite"):
        engine.on_snapshot(mksnap(float("nan")), [])


# --------------------------------------------------------------------------- real model integration


def test_real_model_warms_up_then_quotes_and_is_deterministic() -> None:
    def run() -> list[list[Action]]:
        engine = MarketMakerEngine(CFG)
        ex = FakeExchange(engine)
        out: list[list[Action]] = []
        for i in range(0, 60):
            ts = 1000.0 + i
            spot = 100.0 + 0.001 * (i % 5)
            out.append(ex.step(mksnap(ts, spot=spot)))
        assert ex.rejected == []
        return out

    first = run()
    assert first == run()  # same inputs, same actions (ids included)
    # WARMUP (< 1015) and the model warm-up (needs 30 s of spot history) emit nothing.
    assert all(a == [] for a in first[:30])
    assert any(places(a) for a in first[30:])


# --------------------------------------------------------------------------- randomised soundness


def _run_random_episode(
    seed: int,
) -> tuple[MarketMakerEngine, FakeExchange, list[list[Action]], int]:
    """Two markets (BTC, ETH) over a 300 s window with random walks, fills and glitches."""
    rng = random.Random(seed)
    cfg = replace(
        CFG,
        model=ModelConfig(),
        risk=RiskConfig(max_capital_per_market_usd=900.0, max_total_capital_usd=1400.0),
        exchange=CFG.exchange,
    )
    engine = MarketMakerEngine(cfg, initial_cash=3000.0)
    ex = FakeExchange(engine)
    end = 1300.0
    markets = [MarketSpec("m1", "BTC", 1000.0, end), MarketSpec("m2", "ETH", 1000.0, end)]
    mid = {"m1": 0.5, "m2": 0.5}
    spot = {"BTC": 100.0, "ETH": 50.0}
    ref = dict(spot)
    log: list[list[Action]] = []
    pair_cap_checks = 0
    for step in range(0, 305):
        ts = 1000.0 + step
        for asset in spot:
            spot[asset] *= 1.0 + rng.gauss(0.0, 6e-4)
        for m in markets:
            if ts < m.end_ts:
                mid[m.market_id] = min(0.95, max(0.05, mid[m.market_id] + rng.gauss(0.0, 0.01)))
        for m in markets:
            mid_i = round(mid[m.market_id], 2)
            half = rng.choice([1, 1, 1, 2])
            up = (round(mid_i - 0.01 * half, 2), round(mid_i + 0.01 * half, 2))
            down = (round(1 - up[1], 2), round(1 - up[0], 2))
            if rng.random() < 0.01:
                up = (up[1], up[0])  # glitch: crossed book
            snap_spot: float | None = spot[m.asset]
            age: float | None = 0.0
            glitch = rng.random()
            if glitch < 0.04:
                age = 8.0  # stale spot
            elif glitch < 0.05:
                snap_spot = None
            s = MarketSnapshot(
                ts, m, mkbook(UP, *up), mkbook(DOWN, *down), snap_spot,
                None if age is None else ts - age, ref[m.asset],
            )  # fmt: skip
            if ts >= m.end_ts and engine.phase_of(m.market_id) is Phase.DONE:
                continue
            inv = engine.portfolio.inventory(m.market_id)
            actions = engine.on_snapshot(s, ex.open_orders(m.market_id))
            phase = engine.phase_of(m.market_id)
            gated = engine.risk.kill_switch or not engine.risk.spot_ok(s)
            for a in actions:
                if isinstance(a, PlaceOrder):
                    r = a.request
                    book = s.book(r.token)
                    if r.tif is POST:
                        assert r.side is BUY
                        assert phase in (Phase.ACCUMULATE, Phase.WIND_DOWN)
                        assert not gated, "no quotes with a tripped kill switch / stale spot"
                        assert book.best_ask is not None and r.price < book.best_ask  # inv. 7
                        opp = r.token.opposite
                        if inv.qty[opp] > inv.qty[r.token] + EPS:  # inv. 8
                            avg = inv.avg_cost(opp)
                            assert avg is not None
                            assert r.price + avg <= 1 - cfg.pair.target_margin + EPS
                            pair_cap_checks += 1
                    else:
                        assert phase in (Phase.ACCUMULATE, Phase.WIND_DOWN, Phase.FLATTEN)
                        if phase is not Phase.FLATTEN:
                            assert not gated
                    if phase in (Phase.WARMUP, Phase.DONE):
                        raise AssertionError("order placed in WARMUP/DONE")
                if isinstance(a, MergePairs):
                    assert a.size <= inv.paired_qty + 1e-9
            ex.apply(actions, s)
            # random maker fills of resting orders (any phase before FLATTEN)
            for o in list(ex.open_orders(m.market_id)):
                if o.order_id in ex.orders and rng.random() < 0.12:
                    frac = rng.choice([0.3, 1.0, 1.0])
                    qty = round(o.remaining * frac, 2)
                    if qty >= 0.01:
                        ex.maker_fill(o.order_id, size=min(qty, o.remaining), ts=ts)
            log.append(actions)
            assert engine.portfolio.cash - ex.reserved() >= -1e-6
            for mk in markets:
                cap_m = engine.portfolio.inventory(mk.market_id).capital_at_risk() + ex.reserved(
                    mk.market_id
                )
                assert cap_m <= cfg.risk.max_capital_per_market_usd + 1e-6
            total = engine.portfolio.capital_at_risk() + ex.reserved()
            assert total <= cfg.risk.max_total_capital_usd + 1e-6
    # settle every market: the books must balance (DESIGN invariant 4)
    totals = 0.0
    for m in markets:
        inv = engine.portfolio.inventory(m.market_id)
        winner = UP if rng.random() < 0.5 else DOWN
        for o in list(ex.open_orders(m.market_id)):
            del ex.orders[o.order_id]
        totals += engine.on_settlement(
            Settlement(m.market_id, winner, end, payout=inv.qty[winner])
        ).total
    assert engine.portfolio.cash == pytest.approx(3000.0 + totals, abs=1e-6)
    assert engine.equity() == pytest.approx(engine.portfolio.cash, abs=1e-9)
    return engine, ex, log, pair_cap_checks


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_random_episodes_never_overdraw_never_cross_and_balance(seed: int) -> None:
    engine, ex, log, _ = _run_random_episode(seed)
    assert ex.rejected == [], ex.rejected[:5]  # the engine never asked for what it could not pay
    ids = [r.client_id for actions in log for r in places(actions)]
    assert len(ids) == len(set(ids))  # client ids are unique
    assert engine.stats["quotes_placed"] > 20  # the episode actually traded
    assert engine.stats["snapshots"] >= 500


def test_random_episodes_exercise_every_engine_path() -> None:
    totals: dict[str, int] = {}
    pair_cap_checks = 0
    for seed in range(1, 7):
        engine, _, _, checks = _run_random_episode(seed)
        pair_cap_checks += checks
        for key, value in engine.stats.items():
            totals[key] = totals.get(key, 0) + value
    for key in (
        "quotes_placed",
        "quotes_cancelled",
        "merges",
        "taker_completions",
        "flatten_sells",
        "holds",
        "fills",
        "settlements",
        "risk_trips_spot_stale",
        "risk_trips_book",
        "orders_rate_limited",
    ):
        assert totals[key] > 0, key
    assert pair_cap_checks > 100  # invariant 8 was checked on many emitted bids


def test_random_episodes_are_reproducible() -> None:
    _, _, a, _ = _run_random_episode(11)
    _, _, b, _ = _run_random_episode(11)
    assert a == b
    _, _, c, _ = _run_random_episode(12)
    assert a != c
