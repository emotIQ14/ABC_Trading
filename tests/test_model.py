"""Tests for the fair-probability model (DESIGN sections 3, 4.2 and the model-related invariants).

Hand-computed fixtures
----------------------
HAND config: halflife 1 s (so dt = 1 gives alpha = 0.5 exactly), L_mom = 2 s, L_accel = 1 s,
all feature weights 0, no shrinkage, tiny p_floor. Log-price path ``x`` at ts 0..4 (spot =
100 * exp(x)):  x = 0, 0.02, 0.02, 0.04, 0.05.  EWMA of per-second squared log-return:

    ts1: r2 = 0.02**2 = 4e-4          -> var = 4e-4            (initialised)
    ts2: r2 = 0                       -> var = .5*4e-4 + 0     = 2e-4
    ts3: r2 = 0.02**2 = 4e-4          -> var = .5*2e-4 + .5*4e-4 = 3e-4
    ts4: r2 = 0.01**2 = 1e-4          -> var = .5*3e-4 + .5*1e-4 = 2e-4
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable, Sequence

import pytest

from abc_trading import model as model_pkg
from abc_trading.config import ModelConfig
from abc_trading.model import fair_value as fv_mod
from abc_trading.model.fair_value import (
    FairValue,
    FairValueModel,
    brier_score,
    fit_logistic_weights,
    log_loss,
    norm_cdf,
)
from abc_trading.types import BookSnapshot, Level, MarketSnapshot, MarketSpec, Outcome

UP, DOWN = Outcome.UP, Outcome.DOWN

HAND = ModelConfig(
    vol_halflife_seconds=1.0,
    vol_floor=2e-5,
    momentum_lookback_seconds=2.0,
    accel_lookback_seconds=1.0,
    w_momentum=0.0,
    w_accel=0.0,
    w_book_imbalance=0.0,
    shrink_to_market=0.0,
    p_floor=1e-6,
    min_tau_seconds=1.0,
)
HAND_X = [0.0, 0.02, 0.02, 0.04, 0.05]
SIGMA_TS4 = math.sqrt(2e-4)  # 0.014142135623730952, the hand EWMA sigma after ts4


def spot_of(x: float) -> float:
    return 100.0 * math.exp(x)


def cfg_with(base: ModelConfig = HAND, **kw: float) -> ModelConfig:
    return dataclasses.replace(base, **kw)


def book(
    token: Outcome, bids: list[tuple[float, float]], asks: list[tuple[float, float]]
) -> BookSnapshot:
    return BookSnapshot(
        token,
        tuple(Level(p, s) for p, s in bids),
        tuple(Level(p, s) for p, s in asks),
    )


def books(
    up_bid_sz: float = 20.0,
    up_ask_sz: float = 20.0,
    dn_bid_sz: float = 20.0,
    dn_ask_sz: float = 20.0,
) -> tuple[BookSnapshot, BookSnapshot]:
    """UP 0.54/0.56 (mid 0.55), DOWN mirror 0.44/0.46; one level each side."""
    up = book(UP, [(0.54, up_bid_sz)], [(0.56, up_ask_sz)])
    dn = book(DOWN, [(0.44, dn_bid_sz)], [(0.46, dn_ask_sz)])
    return up, dn


def make_snap(
    ts: float,
    spot: float | None,
    ref: float | None,
    *,
    tau: float = 16.0,
    asset: str = "BTC",
    bk: tuple[BookSnapshot, BookSnapshot] | None = None,
) -> MarketSnapshot:
    up, dn = bk if bk is not None else books()
    market = MarketSpec(market_id="m1", asset=asset, start_ts=ts - 100.0, end_ts=ts + tau)
    return MarketSnapshot(ts=ts, market=market, up_book=up, down_book=dn, spot=spot, ref_price=ref)


def hand_model(cfg: ModelConfig = HAND, asset: str = "BTC") -> FairValueModel:
    m = FairValueModel(cfg)
    for i, x in enumerate(HAND_X):
        m.observe(asset, float(i), spot_of(x))
    return m


def hand_snap(bk: tuple[BookSnapshot, BookSnapshot] | None = None) -> MarketSnapshot:
    """ts=4, spot x=0.05, ref x=0.01 (ln(spot/ref) = 0.04), tau = 16."""
    return make_snap(4.0, spot_of(0.05), spot_of(0.01), bk=bk)


def random_walk(
    rng: random.Random, n: int, per_sqrt_sec: float = 5e-4
) -> list[tuple[float, float]]:
    ts, spot = 0.0, 100.0
    out = [(ts, spot)]
    for _ in range(n - 1):
        dt = rng.uniform(0.3, 2.5)
        ts += dt
        spot *= math.exp(rng.gauss(0.0, per_sqrt_sec * math.sqrt(dt)))
        out.append((ts, spot))
    return out


def feed(m: FairValueModel, obs: list[tuple[float, float]], asset: str = "BTC") -> FairValueModel:
    for ts, spot in obs:
        m.observe(asset, ts, spot)
    return m


# --------------------------------------------------------------------------- norm_cdf


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.5),
        (0.5, 0.6914624612740131),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (1.96, 0.9750021048517796),
        (2.0, 0.9772498680518208),
    ],
)
def test_norm_cdf_known_values(x: float, expected: float) -> None:
    assert norm_cdf(x) == pytest.approx(expected, abs=1e-12)


def test_norm_cdf_symmetry_monotone_and_limits() -> None:
    xs = [i / 10.0 for i in range(-60, 61)]
    vals = [norm_cdf(x) for x in xs]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))
    for x in xs:
        assert norm_cdf(x) + norm_cdf(-x) == pytest.approx(1.0, abs=1e-15)
    assert norm_cdf(math.inf) == 1.0
    assert norm_cdf(-math.inf) == 0.0
    assert norm_cdf(40.0) == 1.0
    assert norm_cdf(-40.0) == pytest.approx(0.0, abs=1e-300)


def test_norm_cdf_rejects_nan() -> None:
    with pytest.raises(ValueError):
        norm_cdf(math.nan)


def test_package_reexports_public_names() -> None:
    for name in (
        "FairValue",
        "FairValueModel",
        "norm_cdf",
        "fit_logistic_weights",
        "brier_score",
        "log_loss",
    ):
        assert getattr(model_pkg, name) is getattr(fv_mod, name)
        assert name in model_pkg.__all__


# --------------------------------------------------------------------------- FairValue


def test_fair_value_p_token_is_complementary() -> None:
    fv = FairValue(0.3, 0.3, 0.0, 0.0, 1e-4, 10.0, 0.0, 0.0, 0.0, True)
    assert fv.p(UP) == 0.3
    assert fv.p(DOWN) == 0.7
    rng = random.Random(1)
    for _ in range(200):
        p = rng.random()
        f = dataclasses.replace(fv, p_up=p)
        assert f.p(UP) + f.p(DOWN) == pytest.approx(1.0, abs=1e-15)


# --------------------------------------------------------------------------- observe / EWMA vol


def test_vol_none_until_two_observations() -> None:
    m = FairValueModel(HAND)
    assert m.vol("BTC") is None
    m.observe("BTC", 0.0, 100.0)
    assert m.vol("BTC") is None
    m.observe("BTC", 1.0, spot_of(0.02))
    assert m.vol("BTC") is not None
    assert m.vol("ETH") is None


def test_ewma_vol_hand_computed_series() -> None:
    m = FairValueModel(HAND)
    m.observe("BTC", 0.0, spot_of(HAND_X[0]))
    # expected variance after each observation, from the table in the module docstring
    expected_var = [4e-4, 2e-4, 3e-4, 2e-4]
    for i, var in enumerate(expected_var, start=1):
        m.observe("BTC", float(i), spot_of(HAND_X[i]))
        assert m.vol("BTC") == pytest.approx(math.sqrt(var), rel=1e-12)
    assert m.vol("BTC") == pytest.approx(SIGMA_TS4, rel=1e-12)


def test_ewma_vol_irregular_dt_hand_computed() -> None:
    # halflife 1. x: ts0=0, ts2=0.06, ts3=0.03, ts3.5=0.04
    #   ts2:   dt=2, r2 = 0.06**2/2 = 0.0018                              -> var = 0.0018
    #   ts3:   dt=1, r2 = 0.03**2/1 = 9e-4, alpha = 1 - 0.5**1 = 0.5      -> var = 0.00135
    #   ts3.5: dt=.5, r2 = 0.01**2/.5 = 2e-4, alpha = 1 - 0.5**0.5 = 0.29289321881345254
    #          var = 0.7071067811865475*0.00135 + 0.29289321881345254*2e-4 = 0.0010131727983645
    m = FairValueModel(HAND)
    m.observe("BTC", 0.0, 100.0)
    m.observe("BTC", 2.0, spot_of(0.06))
    assert m.vol("BTC") == pytest.approx(math.sqrt(0.0018), rel=1e-12)
    m.observe("BTC", 3.0, spot_of(0.03))
    assert m.vol("BTC") == pytest.approx(math.sqrt(0.00135), rel=1e-12)
    m.observe("BTC", 3.5, spot_of(0.04))
    assert m.vol("BTC") == pytest.approx(math.sqrt(0.0010131727983645298), rel=1e-12)


def test_ewma_uses_halflife_in_seconds() -> None:
    # halflife 10, dt 10 -> alpha = 0.5. Same arithmetic as the HAND table but time-stretched:
    # r2 = r**2 / 10. ts10: var = 0.02**2/10 = 4e-5. ts20: r=0 -> var = .5*4e-5 = 2e-5.
    m = FairValueModel(cfg_with(vol_halflife_seconds=10.0, vol_floor=1e-9))
    m.observe("BTC", 0.0, 100.0)
    m.observe("BTC", 10.0, spot_of(0.02))
    assert m.vol("BTC") == pytest.approx(math.sqrt(4e-5), rel=1e-12)
    m.observe("BTC", 20.0, spot_of(0.02))
    assert m.vol("BTC") == pytest.approx(math.sqrt(2e-5), rel=1e-12)


def test_vol_floor_applies_to_vol_and_estimate() -> None:
    m = FairValueModel(HAND)
    for i in range(5):
        m.observe("BTC", float(i), 100.0)  # zero returns -> var 0 -> sigma = floor
    assert m.vol("BTC") == 2e-5
    fv = m.estimate(make_snap(4.0, 100.0, 100.0))
    assert fv.valid
    assert fv.sigma == 2e-5


def test_vol_floor_binds_when_realised_vol_is_tiny_but_nonzero() -> None:
    # r = ln(1 + 1e-7) ~ 1e-7 per 1 s -> var = 1e-14 -> sqrt = 1e-7 < floor 2e-5 -> floor wins.
    m = FairValueModel(HAND)
    m.observe("BTC", 0.0, 100.0)
    m.observe("BTC", 1.0, 100.0 * (1.0 + 1e-7))
    assert m.vol("BTC") == 2e-5
    # ... and a realised vol above the floor is returned untouched (r = 0.02 -> sigma = 0.02)
    m2 = FairValueModel(HAND)
    m2.observe("BTC", 0.0, 100.0)
    m2.observe("BTC", 1.0, spot_of(0.02))
    assert m2.vol("BTC") == pytest.approx(0.02, rel=1e-12)


def test_observe_is_idempotent_per_ts_and_first_value_wins() -> None:
    a, b = hand_model(), hand_model()
    b.observe("BTC", 4.0, spot_of(0.05))  # exact repeat
    b.observe("BTC", 4.0, spot_of(0.9))  # same ts, different spot: ignored
    assert b.vol("BTC") == a.vol("BTC")
    snap = hand_snap()
    assert a.estimate(snap) == b.estimate(snap)
    # and the model keeps working normally afterwards
    a.observe("BTC", 5.0, spot_of(0.06))
    b.observe("BTC", 5.0, spot_of(0.06))
    assert a.vol("BTC") == b.vol("BTC")


def test_observe_rejects_backwards_ts_without_mutating() -> None:
    m = hand_model()
    before = m.vol("BTC")
    with pytest.raises(ValueError, match="backwards"):
        m.observe("BTC", 3.999, 100.0)
    assert m.vol("BTC") == before
    assert m.estimate(hand_snap()) == hand_model().estimate(hand_snap())


@pytest.mark.parametrize("spot", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_observe_rejects_bad_spot(spot: float) -> None:
    m = FairValueModel(HAND)
    with pytest.raises(ValueError, match="spot"):
        m.observe("BTC", 0.0, spot)
    assert m.vol("BTC") is None
    m.observe("BTC", 0.0, 100.0)
    with pytest.raises(ValueError, match="spot"):
        m.observe("BTC", 1.0, spot)
    assert m.vol("BTC") is None  # the failed observe did not create a return


@pytest.mark.parametrize("ts", [math.nan, math.inf])
def test_observe_rejects_non_finite_ts(ts: float) -> None:
    with pytest.raises(ValueError, match="ts"):
        FairValueModel(HAND).observe("BTC", ts, 100.0)


def test_assets_are_independent() -> None:
    m = FairValueModel(HAND)
    m.observe("BTC", 0.0, 100.0)
    m.observe("BTC", 1.0, spot_of(0.02))
    m.observe("ETH", 0.0, 50.0)
    m.observe("ETH", 5.0, 50.0)
    assert m.vol("BTC") == pytest.approx(0.02, rel=1e-12)  # r2 = 0.02**2 / 1
    assert m.vol("ETH") == 2e-5  # floored zero
    m.observe("ETH", 3.0 + 2.5, 51.0)  # ETH time may differ from BTC time
    assert m.vol("BTC") == pytest.approx(0.02, rel=1e-12)


def test_history_is_pruned_but_keeps_an_anchor() -> None:
    cfg = cfg_with(momentum_lookback_seconds=30.0, accel_lookback_seconds=10.0)
    m = FairValueModel(cfg)
    for t in range(5000):
        m.observe("BTC", float(t), 100.0 + (t % 7) * 0.01)
    hist = m._history["BTC"]
    # retention = max lookback (30) + margin (60) = 90 s before the newest ts (4999):
    # cutoff = 4909. One anchor at 4909 is kept plus every later observation: 4909..4999.
    assert hist[0].ts == 4909.0
    assert len(hist) == 91
    assert m.estimate(make_snap(4999.0, 100.0, 100.0)).valid


def test_invalid_model_config_rejected() -> None:
    bad = [
        cfg_with(vol_halflife_seconds=0.0),
        cfg_with(vol_floor=0.0),
        cfg_with(momentum_lookback_seconds=0.0),
        cfg_with(accel_lookback_seconds=-1.0),
        cfg_with(min_tau_seconds=0.0),
        cfg_with(shrink_to_market=1.5),
        cfg_with(shrink_to_market=-0.1),
        cfg_with(p_floor=0.0),
        cfg_with(p_floor=0.5),
        cfg_with(w_momentum=math.nan),
        cfg_with(vol_halflife_seconds=math.inf),
    ]
    for cfg in bad:
        with pytest.raises(ValueError, match="ModelConfig"):
            FairValueModel(cfg)


# ------------------------------------------------------------------ estimate: invalid paths


def assert_invalid(fv: FairValue, fallback: float) -> None:
    assert fv.valid is False
    assert fv.p_up == fallback
    assert fv.p_up_model == fallback
    assert (fv.z, fv.z_adj, fv.momentum, fv.accel, fv.book_imbalance) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert fv.sigma > 0.0
    assert fv.tau >= 1.0


@pytest.mark.parametrize(
    ("spot", "ref"),
    [
        (None, spot_of(0.0)),
        (spot_of(0.0), None),
        (None, None),
        (0.0, 100.0),
        (-5.0, 100.0),
        (100.0, 0.0),
        (100.0, -5.0),
        (math.nan, 100.0),
        (100.0, math.nan),
        (math.inf, 100.0),
        (100.0, math.inf),
    ],
)
def test_invalid_when_spot_or_ref_missing_or_bad(spot: float | None, ref: float | None) -> None:
    m = hand_model()  # plenty of history: only the spot/ref condition can fail
    fv = m.estimate(make_snap(4.0, spot, ref))
    assert_invalid(fv, fallback=(0.54 + 0.56) / 2.0)
    assert fv.sigma == pytest.approx(SIGMA_TS4, rel=1e-12)  # sigma is still reported


def test_invalid_fallback_is_half_when_up_book_has_no_mid() -> None:
    m = hand_model()
    one_sided = book(UP, [(0.54, 10.0)], [])
    _, dn = books()
    fv = m.estimate(make_snap(4.0, None, None, bk=(one_sided, dn)))
    assert_invalid(fv, 0.5)
    empty = BookSnapshot(UP)
    fv2 = m.estimate(make_snap(4.0, None, 100.0, bk=(empty, BookSnapshot(DOWN))))
    assert_invalid(fv2, 0.5)


def test_invalid_without_observations() -> None:
    fv = FairValueModel(HAND).estimate(make_snap(4.0, 100.0, 100.0))
    assert_invalid(fv, 0.55)
    assert fv.sigma == HAND.vol_floor  # no vol estimate -> floor


def test_invalid_with_single_observation() -> None:
    m = FairValueModel(HAND)
    m.observe("BTC", 0.0, 100.0)
    assert_invalid(m.estimate(make_snap(50.0, 100.0, 100.0)), 0.55)


def test_invalid_when_only_other_asset_has_history() -> None:
    m = hand_model(asset="ETH")
    assert_invalid(m.estimate(make_snap(4.0, 100.0, 100.0, asset="BTC")), 0.55)
    assert m.estimate(make_snap(4.0, 100.0, 100.0, asset="ETH")).valid


def test_invalid_when_snapshot_precedes_all_but_one_observation() -> None:
    m = hand_model()
    # at ts=0 only one observation is visible (the others are the future)
    assert_invalid(m.estimate(make_snap(0.0, 100.0, 100.0)), 0.55)
    assert_invalid(m.estimate(make_snap(-10.0, 100.0, 100.0)), 0.55)


def test_warmup_rule_boundary_is_oldest_obs_at_or_before_ts_minus_horizon() -> None:
    m = hand_model()  # observations at ts 0..4; horizon = max(2, 1) = 2
    # ts=1: obs {0,1}; need oldest(0) <= 1-2 = -1 -> False
    fv = m.estimate(make_snap(1.0, 100.0, 100.0))
    assert_invalid(fv, 0.55)
    assert fv.sigma == pytest.approx(math.sqrt(4e-4), rel=1e-12)  # sigma as of ts=1 is reported
    # ts=1.99: need 0 <= -0.01 -> False;  ts=2: need 0 <= 0 -> True (inclusive)
    assert not m.estimate(make_snap(1.99, 100.0, 100.0)).valid
    assert m.estimate(make_snap(2.0, 100.0, 100.0)).valid
    assert m.estimate(make_snap(3.0, 100.0, 100.0)).valid


def test_warmup_requires_the_longer_of_the_two_lookbacks() -> None:
    # accel lookback (3) longer than momentum lookback (1): horizon = 3.
    cfg = cfg_with(momentum_lookback_seconds=1.0, accel_lookback_seconds=3.0)
    m = hand_model(cfg)
    assert not m.estimate(make_snap(2.0, 100.0, 100.0)).valid  # 0 <= 2-3 False
    assert m.estimate(make_snap(3.0, 100.0, 100.0)).valid  # 0 <= 0 True
    # hand numbers at ts=3 (obs 0..3, var 3e-4, sigma = sqrt(3e-4) = 0.017320508075688773):
    #   spot_now = x 0.04; spot_at(3-1=2) -> x 0.02; spot_at(3-3=0) -> x 0
    #   mom = 0.02 / (sigma*sqrt(1))   = 1.1547005383792515
    #   short = 0.04 / (sigma*sqrt(3)) = 0.04/0.03 = 1.3333333333333333
    #   accel = short - mom = 0.17863279495408202
    fv = m.estimate(make_snap(3.0, spot_of(0.04), spot_of(0.04)))
    assert fv.sigma == pytest.approx(math.sqrt(3e-4), rel=1e-12)
    assert fv.momentum == pytest.approx(1.1547005383792515, abs=1e-12)
    assert fv.accel == pytest.approx(0.17863279495408202, abs=1e-12)


def test_history_older_than_retention_reports_invalid_not_stale_data() -> None:
    m = FairValueModel(HAND)  # horizon 2 -> retention 62 s
    for t in range(1000):
        m.observe("BTC", float(t), 100.0)
    assert m.estimate(make_snap(999.0, 100.0, 100.0)).valid
    assert m.estimate(make_snap(990.0, 100.0, 100.0)).valid  # inside the retained window
    assert_invalid(m.estimate(make_snap(500.0, 100.0, 100.0)), 0.55)  # long since pruned


# --------------------------------------------------------------------------- estimate: hand numbers


def test_estimate_pure_z_hand_computed() -> None:
    # sigma = sqrt(2e-4), tau = 16 -> sigma*sqrt(tau) = 0.05656854249492381
    # z = ln(spot/ref) / that = 0.04 / 0.05656854249492381 = 0.7071067811865475 (= 1/sqrt 2)
    # Phi(1/sqrt 2) = 0.5 * (1 + erf(0.5)) = 0.7602499389065233
    fv = hand_model().estimate(hand_snap())
    assert fv.valid
    assert fv.sigma == pytest.approx(SIGMA_TS4, rel=1e-12)
    assert fv.tau == 16.0
    assert fv.z == pytest.approx(0.7071067811865475, abs=1e-12)
    assert fv.z_adj == pytest.approx(fv.z, abs=1e-15)  # weights are zero
    assert fv.p_up_model == pytest.approx(0.7602499389065233, abs=1e-12)
    assert fv.p_up == pytest.approx(0.7602499389065233, abs=1e-12)  # shrink = 0
    assert fv.p(DOWN) == pytest.approx(1.0 - 0.7602499389065233, abs=1e-12)


def test_estimate_features_and_weights_hand_computed() -> None:
    # spot x=0.05; spot_at(4-2=2) -> x=0.02; spot_at(4-1=3) -> x=0.04
    #   mom      = 0.03 / (sigma * sqrt(2)) = 0.03 / 0.02     = 1.5
    #   mom_short= 0.01 / (sigma * sqrt(1)) = 0.01 / 0.014142 = 0.7071067811865475
    #   accel    = 0.7071067811865475 - 1.5                   = -0.7928932188134525
    #   obi      = imbalance(UP: 30 vs 10) - imbalance(DOWN: 20 vs 20) = 0.5 - 0 = 0.5
    # z_adj = 0.7071067811865475 + 0.2*1.5 + 0.1*(-0.7928932188134525) + 0.5*0.5
    #       = 0.7071067811865475 + 0.3 - 0.07928932188134525 + 0.25 = 1.177817459305202
    # Phi(1.177817459305202) = 0.8805653066632398
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5)
    fv = hand_model(cfg).estimate(hand_snap(bk=books(30.0, 10.0, 20.0, 20.0)))
    assert fv.momentum == pytest.approx(1.5, abs=1e-12)
    assert fv.accel == pytest.approx(-0.7928932188134525, abs=1e-12)
    assert fv.book_imbalance == pytest.approx(0.5, abs=1e-15)
    assert fv.z == pytest.approx(0.7071067811865475, abs=1e-12)
    assert fv.z_adj == pytest.approx(1.177817459305202, abs=1e-12)
    assert fv.p_up_model == pytest.approx(0.8805653066632398, abs=1e-12)


def test_shrink_to_market_hand_computed() -> None:
    # same state as above; up mid = (0.54 + 0.56)/2 = 0.55
    # shrink 0.3: p_up = 0.7 * 0.8805653066632398 + 0.3 * 0.55 = 0.6163957146642679 + 0.165
    #                  = 0.7813957146642679
    base = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5)
    snap = hand_snap(bk=books(30.0, 10.0, 20.0, 20.0))
    fv = hand_model(cfg_with(base, shrink_to_market=0.3)).estimate(snap)
    assert fv.p_up_model == pytest.approx(0.8805653066632398, abs=1e-12)
    assert fv.p_up == pytest.approx(0.7813957146642679, abs=1e-12)


def test_shrink_extremes_return_mid_or_model() -> None:
    snap = hand_snap(bk=books(30.0, 10.0, 20.0, 20.0))
    only_market = hand_model(cfg_with(shrink_to_market=1.0)).estimate(snap)
    only_model = hand_model(cfg_with(shrink_to_market=0.0)).estimate(snap)
    assert only_market.p_up == pytest.approx(0.55, abs=1e-15)
    assert only_market.p_up_model == pytest.approx(0.7602499389065233, abs=1e-12)
    assert only_model.p_up == only_model.p_up_model
    assert only_model.p_up == pytest.approx(0.7602499389065233, abs=1e-12)


def test_no_shrinkage_without_up_mid() -> None:
    # one-sided UP book (no ask) has no mid, so even shrink = 1 returns the pure model value;
    # all weights are 0 so that value is the pure-z hand number Phi(1/sqrt 2)
    up = book(UP, [(0.54, 10.0)], [])
    _, dn = books()
    fv = hand_model(cfg_with(shrink_to_market=1.0)).estimate(hand_snap(bk=(up, dn)))
    assert fv.valid
    assert fv.p_up == fv.p_up_model
    assert fv.p_up == pytest.approx(0.7602499389065233, abs=1e-12)


def test_book_imbalance_is_clipped_to_unit_interval() -> None:
    # UP bids only -> imbalance +1; DOWN asks only -> imbalance -1; difference 2 -> clipped to 1
    up = book(UP, [(0.54, 10.0)], [])
    dn = book(DOWN, [], [(0.46, 10.0)])
    fv = hand_model().estimate(hand_snap(bk=(up, dn)))
    assert fv.book_imbalance == 1.0
    up2 = book(UP, [], [(0.56, 10.0)])
    dn2 = book(DOWN, [(0.44, 10.0)], [])
    assert hand_model().estimate(hand_snap(bk=(up2, dn2))).book_imbalance == -1.0


def test_book_imbalance_uses_top_five_levels_and_empty_books_are_neutral() -> None:
    bids = [(0.54 - 0.01 * i, 10.0) for i in range(5)] + [(0.40, 10_000.0)]  # 6th level ignored
    asks = [(0.56 + 0.01 * i, 10.0) for i in range(5)]
    up = book(UP, bids, asks)
    _, dn = books()
    fv = hand_model().estimate(hand_snap(bk=(up, dn)))
    assert fv.book_imbalance == pytest.approx(0.0, abs=1e-15)  # 50 vs 50, DOWN 20 vs 20
    empty = hand_model().estimate(hand_snap(bk=(BookSnapshot(UP), BookSnapshot(DOWN))))
    assert empty.valid
    assert empty.book_imbalance == 0.0
    assert empty.p_up == empty.p_up_model


def test_book_imbalance_weight_moves_z_by_exactly_w_times_obi() -> None:
    bk = books(10.0, 30.0, 20.0, 20.0)  # UP imbalance -0.5, DOWN 0 -> obi = -0.5
    base = hand_model().estimate(hand_snap(bk=bk))
    weighted = hand_model(cfg_with(w_book_imbalance=0.4)).estimate(hand_snap(bk=bk))
    assert weighted.book_imbalance == pytest.approx(-0.5, abs=1e-15)
    assert weighted.z_adj - base.z_adj == pytest.approx(0.4 * -0.5, abs=1e-12)


# --------------------------------------------------------------------------- estimate: behaviour


@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0, 7.0, 300.0, 900.0, 1e6])
def test_spot_equals_ref_with_zero_momentum_gives_one_half(tau: float) -> None:
    cfg = ModelConfig()  # production defaults: non-zero weights, shrink 0.5, lookbacks 30/10
    m = FairValueModel(cfg)
    for t in range(41):
        m.observe("BTC", float(t), 60_000.0)  # flat tape: momentum = accel = 0
    fv = m.estimate(make_snap(40.0, 60_000.0, 60_000.0, tau=tau))
    assert fv.valid
    assert (fv.z, fv.momentum, fv.accel, fv.book_imbalance) == (0.0, 0.0, 0.0, 0.0)
    assert fv.p_up_model == pytest.approx(0.5, abs=1e-12)
    assert fv.p_up == pytest.approx(0.5 * 0.5 + 0.5 * 0.55, abs=1e-12)  # shrink to mid 0.55


def test_p_increases_with_spot_above_ref_and_is_above_half() -> None:
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5)
    m = hand_model(cfg)
    ref = spot_of(0.01)
    ps = []
    for d in [-0.1, -0.05, -0.02, -0.005, 0.0, 0.005, 0.02, 0.05, 0.1]:
        ps.append(m.estimate(make_snap(4.0, spot_of(0.01 + d), ref)).p_up_model)
    assert all(b > a for a, b in zip(ps, ps[1:], strict=False))  # strictly increasing, no clipping
    pure = hand_model()
    above = pure.estimate(make_snap(4.0, spot_of(0.02), ref)).p_up_model
    below = pure.estimate(make_snap(4.0, spot_of(0.00), ref)).p_up_model
    assert above > 0.5 > below


def test_p_moves_away_from_half_as_tau_shrinks() -> None:
    # z = ln(spot/ref) / (sigma * sqrt(tau)) with ln(spot/ref) = +/-0.01, sigma = sqrt(2e-4).
    # tau=100 -> z = 0.01/(0.014142135623730952*10) = 0.07071067811865475
    # tau=16 -> 0.1767766952966369, tau=4 -> 0.3535533905932738, tau=1 -> 0.7071067811865475
    m = hand_model()
    ref = spot_of(0.01)
    taus = [900.0, 300.0, 100.0, 16.0, 4.0, 1.0]
    up = [m.estimate(make_snap(4.0, spot_of(0.02), ref, tau=t)) for t in taus]
    dn = [m.estimate(make_snap(4.0, spot_of(0.00), ref, tau=t)) for t in taus]
    assert [f.tau for f in up] == taus
    assert all(b.p_up_model > a.p_up_model for a, b in zip(up, up[1:], strict=False))
    assert all(b.p_up_model < a.p_up_model for a, b in zip(dn, dn[1:], strict=False))
    for t, f in zip(taus, up, strict=True):
        assert f.z == pytest.approx(0.01 / (SIGMA_TS4 * math.sqrt(t)), abs=1e-12)
    assert up[3].z == pytest.approx(0.1767766952966369, abs=1e-12)
    assert up[-1].p_up_model == pytest.approx(0.7602499389065233, abs=1e-12)


def test_tau_is_floored_at_min_tau_and_gives_a_near_step() -> None:
    cfg = cfg_with(min_tau_seconds=2.0, p_floor=0.02)
    m = FairValueModel(cfg)
    for t in range(5):
        m.observe("BTC", float(t), 100.0)  # flat tape: sigma = floor 2e-5
    # seconds_to_end below / at min_tau all give tau = 2 and identical output
    outs = [m.estimate(make_snap(4.0, 100.1, 100.0, tau=s)) for s in (0.5, 0.0, -30.0, 2.0)]
    assert all(o.tau == 2.0 for o in outs)
    assert all(o == outs[0] for o in outs)
    assert m.estimate(make_snap(4.0, 100.1, 100.0, tau=3.0)).tau == 3.0
    # step: z = ln(1.001)/(2e-5*sqrt(2)) ~ 35 -> clipped to the floors; spot == ref -> exactly 1/2
    hi = m.estimate(make_snap(4.0, 100.1, 100.0, tau=0.0))
    lo = m.estimate(make_snap(4.0, 99.9, 100.0, tau=0.0))
    mid = m.estimate(make_snap(4.0, 100.0, 100.0, tau=0.0))
    assert hi.p_up_model == 1.0 - 0.02
    assert lo.p_up_model == 0.02
    assert mid.p_up_model == 0.5
    # with 900 s to go the same 10 bp displacement is nowhere near a step:
    # z = ln(1.001)/(2e-5*sqrt(900)) = 1.6658338884723718 -> Phi = 0.9521267478270308
    far = m.estimate(make_snap(4.0, 100.1, 100.0, tau=900.0))
    assert far.p_up_model == pytest.approx(0.9521267478270308, abs=1e-9)


def test_p_model_is_clipped_to_floor_and_ceiling() -> None:
    cfg = cfg_with(p_floor=0.02)
    m = hand_model(cfg)
    ref = spot_of(0.01)
    up = m.estimate(make_snap(4.0, spot_of(0.5), ref))  # z ~ 8.7
    dn = m.estimate(make_snap(4.0, spot_of(-0.5), ref))
    assert up.z > 5 and dn.z < -5
    assert up.p_up_model == 1.0 - 0.02
    assert dn.p_up_model == 0.02
    assert up.p_up == 1.0 - 0.02  # shrink 0


def test_clipped_model_then_shrunk_hand_computed() -> None:
    # p_model clipped to 0.98; shrink 0.5 with mid 0.55 -> 0.5*0.98 + 0.5*0.55 = 0.765
    cfg = cfg_with(p_floor=0.02, shrink_to_market=0.5)
    fv = hand_model(cfg).estimate(make_snap(4.0, spot_of(0.5), spot_of(0.01)))
    assert fv.p_up_model == 0.98
    assert fv.p_up == pytest.approx(0.765, abs=1e-12)


def test_down_probability_is_complement() -> None:
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5, shrink_to_market=0.3)
    m = hand_model(cfg)
    for x in (-0.2, -0.01, 0.0, 0.03, 0.4):
        fv = m.estimate(make_snap(4.0, spot_of(x), spot_of(0.01), bk=books(30.0, 10.0, 20.0, 20.0)))
        assert fv.p(UP) + fv.p(DOWN) == pytest.approx(1.0, abs=1e-15)
        assert fv.p(DOWN) == 1.0 - fv.p_up


def test_antisymmetry_negating_displacement_and_features_gives_complement() -> None:
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5, p_floor=0.02)
    a = hand_model(cfg)
    b = FairValueModel(cfg)  # mirror-image tape: x -> -x, so every log-return flips sign
    for i, x in enumerate(HAND_X):
        b.observe("BTC", float(i), spot_of(-x))
    for xs, xr in [(0.05, 0.01), (-0.03, 0.02), (0.4, 0.0), (0.0, 0.0), (0.011, -0.007)]:
        sa = make_snap(4.0, spot_of(xs), spot_of(xr), bk=books(30.0, 10.0, 20.0, 20.0))
        # mirrored books: UP imbalance 10 vs 30 = -0.5, DOWN unchanged -> obi = -0.5
        sb = make_snap(4.0, spot_of(-xs), spot_of(-xr), bk=books(10.0, 30.0, 20.0, 20.0))
        fa, fb = a.estimate(sa), b.estimate(sb)
        assert fa.sigma == pytest.approx(fb.sigma, rel=1e-12)
        assert fb.z == pytest.approx(-fa.z, abs=1e-12)
        assert fb.momentum == pytest.approx(-fa.momentum, abs=1e-12)
        assert fb.accel == pytest.approx(-fa.accel, abs=1e-12)
        assert fb.book_imbalance == pytest.approx(-fa.book_imbalance, abs=1e-15)
        assert fb.z_adj == pytest.approx(-fa.z_adj, abs=1e-12)
        assert fb.p_up_model == pytest.approx(1.0 - fa.p_up_model, abs=1e-12)


# --------------------------------------------------------------------------- state and look-ahead


def test_estimate_does_not_mutate_state() -> None:
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5, shrink_to_market=0.3)
    probed, control = hand_model(cfg), hand_model(cfg)
    snap = hand_snap(bk=books(30.0, 10.0, 20.0, 20.0))
    first = probed.estimate(snap)
    for ts in (0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 1e6):  # including far-future and warm-up snapshots
        probed.estimate(make_snap(ts, 100.0, 100.0))
    assert probed.estimate(snap) == first
    assert probed.vol("BTC") == control.vol("BTC")
    assert probed._history["BTC"] == control._history["BTC"]
    for m in (probed, control):  # later observations behave identically with / without probing
        m.observe("BTC", 5.0, spot_of(0.07))
    assert probed.vol("BTC") == control.vol("BTC")
    later = make_snap(5.0, spot_of(0.07), spot_of(0.01), bk=books(30.0, 10.0, 20.0, 20.0))
    assert probed.estimate(later) == control.estimate(later)


def test_future_observations_do_not_change_earlier_estimates() -> None:
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5, shrink_to_market=0.3)
    plain, fed = hand_model(cfg), hand_model(cfg)
    snaps = [
        make_snap(ts, spot_of(0.05), spot_of(0.01), bk=books(30.0, 10.0, 20.0, 20.0))
        for ts in (2.0, 3.0, 4.0)
    ]
    before = [fed.estimate(s) for s in snaps]
    # feed wild FUTURE observations (and a vol spike) to one model only
    for i, x in enumerate([0.9, -0.8, 1.2, -1.0, 0.0], start=5):
        fed.observe("BTC", float(i), spot_of(x))
    assert fed.vol("BTC") != plain.vol("BTC")  # the model really did change ...
    for s, b in zip(snaps, before, strict=True):
        assert fed.estimate(s) == b  # ... but the earlier-ts view did not
        assert fed.estimate(s) == plain.estimate(s)
    # the TS=4 estimate still equals the hand-computed value computed from obs <= 4 only
    assert fed.estimate(snaps[-1]).z == pytest.approx(0.7071067811865475, abs=1e-12)


def test_estimate_at_earlier_ts_uses_only_observations_up_to_that_ts() -> None:
    cfg = cfg_with(w_momentum=0.2, w_accel=0.1, w_book_imbalance=0.5)
    full = hand_model(cfg)
    for i, x in enumerate([0.08, -0.02, 0.5], start=5):
        full.observe("BTC", float(i), spot_of(x))
    # a fresh model that only ever saw ts <= 3 must agree exactly with `full` queried at ts=3
    only_past = FairValueModel(cfg)
    for i, x in enumerate(HAND_X[:4]):
        only_past.observe("BTC", float(i), spot_of(x))
    snap = make_snap(3.0, spot_of(0.04), spot_of(0.01), bk=books(30.0, 10.0, 20.0, 20.0))
    assert full.estimate(snap) == only_past.estimate(snap)
    assert full.estimate(snap).sigma == pytest.approx(math.sqrt(3e-4), rel=1e-12)  # var after ts3


def test_observation_exactly_at_snapshot_ts_is_visible() -> None:
    m = hand_model()
    fv = m.estimate(hand_snap())
    assert fv.sigma == pytest.approx(math.sqrt(2e-4), rel=1e-12)  # includes the ts=4 return
    ahead = hand_model()
    ahead.observe("BTC", 4.5, spot_of(0.9))
    assert ahead.estimate(hand_snap()) == fv


# --------------------------------------------------------------------------- randomised properties


def oracle(
    cfg: ModelConfig,
    obs: list[tuple[float, float]],
    snap_ts: float,
    spot: float,
    ref: float,
    seconds_to_end: float,
    obi: float,
) -> tuple[bool, float, float, float, float, float]:
    """Independent full-history reimplementation: (valid, sigma, z, mom, accel, p_model)."""
    vis = [(t, s) for t, s in obs if t <= snap_ts]
    horizon = max(cfg.momentum_lookback_seconds, cfg.accel_lookback_seconds)
    if len(vis) < 2 or vis[0][0] > snap_ts - horizon:
        return False, 0.0, 0.0, 0.0, 0.0, 0.5
    var: float | None = None
    for (t0, s0), (t1, s1) in zip(vis, vis[1:], strict=False):
        dt = t1 - t0
        r2 = math.log(s1 / s0) ** 2 / dt
        if var is None:
            var = r2
        else:
            alpha = 1.0 - 0.5 ** (dt / cfg.vol_halflife_seconds)
            var = (1.0 - alpha) * var + alpha * r2
    assert var is not None
    sigma = max(math.sqrt(var), cfg.vol_floor)

    def at(target: float) -> float:
        return [s for t, s in vis if t <= target][-1]

    tau = max(seconds_to_end, cfg.min_tau_seconds)
    z = math.log(spot / ref) / (sigma * math.sqrt(tau))
    lm, la = cfg.momentum_lookback_seconds, cfg.accel_lookback_seconds
    mom = math.log(spot / at(snap_ts - lm)) / (sigma * math.sqrt(lm))
    accel = math.log(spot / at(snap_ts - la)) / (sigma * math.sqrt(la)) - mom
    z_adj = z + cfg.w_momentum * mom + cfg.w_accel * accel + cfg.w_book_imbalance * obi
    p = min(max(0.5 * (1.0 + math.erf(z_adj / math.sqrt(2.0))), cfg.p_floor), 1.0 - cfg.p_floor)
    return True, sigma, z, mom, accel, p


def test_random_series_match_independent_full_history_oracle() -> None:
    cfg = ModelConfig(shrink_to_market=0.0)  # production weights, lookbacks 30 / 10
    for seed in range(12):
        rng = random.Random(seed)
        obs = random_walk(rng, 400)  # ~ 500 s, so pruning (90 s retention) definitely happens
        m = feed(FairValueModel(cfg), obs)
        end = obs[-1][0]
        assert len(m._history["BTC"]) < 100
        full_vol = oracle(cfg, obs, end, 100.0, 100.0, 50.0, 0.0)[1]
        assert m.vol("BTC") == pytest.approx(full_vol, rel=1e-12)
        for _ in range(10):
            snap_ts = end - rng.uniform(0.0, 50.0)  # inside retained window
            spot = obs[-1][1] * math.exp(rng.gauss(0.0, 1e-3))
            ref = obs[-1][1] * math.exp(rng.gauss(0.0, 1e-3))
            tau = rng.uniform(0.0, 900.0)
            bk = books(
                rng.uniform(1, 100), rng.uniform(1, 100), rng.uniform(1, 100), rng.uniform(1, 100)
            )
            snap = make_snap(snap_ts, spot, ref, tau=tau, bk=bk)
            fv = m.estimate(snap)
            valid, sigma, z, mom, accel, p = oracle(
                cfg, obs, snap_ts, spot, ref, tau, fv.book_imbalance
            )
            assert fv.valid is valid is True
            assert fv.sigma == pytest.approx(sigma, rel=1e-12)
            assert fv.z == pytest.approx(z, rel=1e-9, abs=1e-12)
            assert fv.momentum == pytest.approx(mom, rel=1e-9, abs=1e-12)
            assert fv.accel == pytest.approx(accel, rel=1e-9, abs=1e-12)
            assert fv.p_up_model == pytest.approx(p, abs=1e-12)


def test_random_no_lookahead_future_data_never_changes_past_estimates() -> None:
    cfg = ModelConfig()
    for seed in range(15):
        rng = random.Random(100 + seed)
        obs = random_walk(rng, 220)
        full = feed(FairValueModel(cfg), obs)
        end = obs[-1][0]
        for _ in range(8):
            snap_ts = end - rng.uniform(0.0, 50.0)
            spot = 100.0 * math.exp(rng.gauss(0.0, 2e-3))
            snap = make_snap(snap_ts, spot, 100.0, tau=rng.uniform(1.0, 600.0))
            past_only = feed(FairValueModel(cfg), [o for o in obs if o[0] <= snap_ts])
            assert full.estimate(snap) == past_only.estimate(snap)


def test_random_estimates_satisfy_range_invariants() -> None:
    for seed in range(15):
        rng = random.Random(500 + seed)
        cfg = ModelConfig(
            w_momentum=rng.uniform(-0.5, 0.5),
            w_accel=rng.uniform(-0.5, 0.5),
            w_book_imbalance=rng.uniform(-0.5, 0.5),
            shrink_to_market=rng.choice([0.0, 0.25, 0.5, 1.0]),
            p_floor=rng.choice([0.01, 0.02, 0.1]),
        )
        obs = random_walk(rng, 150)
        m = feed(FairValueModel(cfg), obs)
        end, last = obs[-1]
        for k in range(-8, 9):  # increasing spots against a fixed ref
            snap = make_snap(end, last * math.exp(k * 4e-4), last, tau=rng.uniform(1.0, 900.0))
            fv = m.estimate(snap)
            assert fv.valid
            assert cfg.p_floor <= fv.p_up_model <= 1.0 - cfg.p_floor
            assert 0.0 <= fv.p_up <= 1.0
            assert fv.p(UP) + fv.p(DOWN) == pytest.approx(1.0, abs=1e-15)
            assert -1.0 <= fv.book_imbalance <= 1.0
            assert fv.sigma >= cfg.vol_floor and fv.tau >= cfg.min_tau_seconds


def test_random_p_is_monotone_in_spot_for_non_negative_weights() -> None:
    for seed in range(15):
        rng = random.Random(900 + seed)
        cfg = ModelConfig(
            w_momentum=rng.uniform(0.0, 0.5),
            w_accel=0.0,  # accel's spot-derivative is sign-indefinite when L_a > L_m
            w_book_imbalance=rng.uniform(0.0, 0.5),
        )
        obs = random_walk(rng, 150)
        m = feed(FairValueModel(cfg), obs)
        end, last = obs[-1]
        ps = []
        for k in range(-10, 11):
            snap = make_snap(end, last * math.exp(k * 3e-4), last, tau=300.0)
            ps.append(m.estimate(snap))
        assert all(f.valid for f in ps)
        assert all(b.p_up_model >= a.p_up_model for a, b in zip(ps, ps[1:], strict=False))
        assert all(b.p_up >= a.p_up - 1e-15 for a, b in zip(ps, ps[1:], strict=False))
        assert all(b.z > a.z for a, b in zip(ps, ps[1:], strict=False))


# --------------------------------------------------------------------------- brier / log loss


def test_brier_score_hand_computed() -> None:
    # preds .9,.2,.5 vs outcomes 1,0,1:
    # ((.9-1)^2 + (.2-0)^2 + (.5-1)^2)/3 = (0.01 + 0.04 + 0.25)/3 = 0.1
    assert brier_score([0.9, 0.2, 0.5], [1, 0, 1]) == pytest.approx(0.1, abs=1e-12)
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert brier_score([0.5] * 4, [1, 0, 0, 1]) == 0.25
    assert brier_score([0.3], [1]) == pytest.approx(0.49, abs=1e-15)


def test_log_loss_hand_computed() -> None:
    # -(ln .9 + ln .8 + ln .5)/3 = (0.10536051565782628 + 0.2231435513142097 + 0.6931471805599453)/3
    #                            = 1.0216512475319814/3 = 0.34055041584399376
    assert log_loss([0.9, 0.2, 0.5], [1, 0, 1]) == pytest.approx(0.34055041584399376, abs=1e-12)
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2.0), abs=1e-15)
    assert log_loss([0.25], [0]) == pytest.approx(-math.log(0.75), abs=1e-15)


def test_log_loss_clips_by_eps() -> None:
    # confident miss: p=0 vs y=1 -> clipped to eps -> -ln(1e-9) = 20.72326583694641
    assert log_loss([0.0], [1]) == pytest.approx(20.72326583694641, abs=1e-9)
    assert log_loss([1.0], [0]) == pytest.approx(20.72326583694641, abs=1e-9)
    assert log_loss([0.0], [1], eps=1e-3) == pytest.approx(-math.log(1e-3), abs=1e-12)
    # confident hit costs ~ eps: -ln(1 - 1e-9) ~ 1e-9
    assert log_loss([1.0], [1]) == pytest.approx(1e-9, abs=1e-12)
    assert log_loss([0.0], [0]) == pytest.approx(1e-9, abs=1e-12)
    # values already inside [eps, 1-eps] are untouched
    assert log_loss([0.4], [1], eps=0.1) == pytest.approx(-math.log(0.4), abs=1e-15)
    # eps = 0.1 clips 0.01 up to 0.1 -> -ln(0.1)
    assert log_loss([0.01], [1], eps=0.1) == pytest.approx(-math.log(0.1), abs=1e-15)


@pytest.mark.parametrize("fn", [brier_score, log_loss])
def test_scores_validate_inputs(fn: Callable[[Sequence[float], Sequence[int]], float]) -> None:
    f = fn
    with pytest.raises(ValueError, match="length"):
        f([0.5, 0.5], [1])
    with pytest.raises(ValueError, match="at least one"):
        f([], [])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        f([1.01], [1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        f([-0.01], [1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        f([math.nan], [1])
    with pytest.raises(ValueError, match="0 or 1"):
        f([0.5], [2])
    with pytest.raises(ValueError, match="0 or 1"):
        f([0.5], [-1])
    with pytest.raises(ValueError, match="0 or 1"):
        f([0.5], [0.5])  # type: ignore[list-item]


@pytest.mark.parametrize("eps", [0.0, -1e-9, 0.5, 0.7, math.nan])
def test_log_loss_rejects_bad_eps(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        log_loss([0.5], [1], eps=eps)


def test_scores_random_properties() -> None:
    rng = random.Random(7)
    for _ in range(50):
        n = rng.randint(1, 40)
        preds = [rng.random() for _ in range(n)]
        outs = [rng.randint(0, 1) for _ in range(n)]
        b, ll = brier_score(preds, outs), log_loss(preds, outs)
        assert 0.0 <= b <= 1.0
        assert ll >= 0.0
        # label-flip symmetry: p -> 1-p with y -> 1-y leaves both scores unchanged
        flipped_p, flipped_y = [1.0 - p for p in preds], [1 - y for y in outs]
        assert brier_score(flipped_p, flipped_y) == pytest.approx(b, abs=1e-12)
        assert log_loss(flipped_p, flipped_y) == pytest.approx(ll, abs=1e-9)
        # a perfectly informed predictor beats any other on both scores
        perfect = [float(y) for y in outs]
        assert brier_score(perfect, outs) == 0.0 <= b
        assert log_loss(perfect, outs) <= ll + 1e-12


# --------------------------------------------------------------------------- fit_logistic_weights


def synthetic(rng: random.Random, w_true: list[float], n: int) -> list[tuple[list[float], int]]:
    out = []
    for _ in range(n):
        x = [rng.gauss(0.0, 1.0) for _ in w_true]
        p = 1.0 / (1.0 + math.exp(-sum(a * b for a, b in zip(w_true, x, strict=True))))
        out.append((x, 1 if rng.random() < p else 0))
    return out


def test_fit_one_step_hand_computed() -> None:
    # samples (x=+1, y=1), (x=-1, y=0); w0 = 0 -> p = 0.5 everywhere.
    # grad = ((0.5-1)*1 + (0.5-0)*(-1)) / 2 = -0.5 ; l2*w = 0 ; w1 = 0 - 0.1*(-0.5) = 0.05
    samples = [([1.0], 1), ([-1.0], 0)]
    assert fit_logistic_weights(samples, iters=1, lr=0.1) == [pytest.approx(0.05, abs=1e-15)]
    assert fit_logistic_weights(samples, iters=0) == [0.0]


def test_fit_two_steps_with_l2_hand_computed() -> None:
    # step 2 (l2 = 0.02): s = sigmoid(0.05) = 0.5124973964842103
    # grad = ((s-1)*1 + (sigmoid(-0.05)-0)*(-1))/2 = (s-1) = -0.48750260351578967
    # w2 = 0.05 - 0.1*(-0.48750260351578967 + 0.02*0.05) = 0.09865026035157898
    samples = [([1.0], 1), ([-1.0], 0)]
    w = fit_logistic_weights(samples, l2=0.02, iters=2, lr=0.1)
    assert w == [pytest.approx(0.09865026035157898, abs=1e-12)]


def test_fit_recovers_known_weights_from_seeded_data() -> None:
    w_true = [0.9, -0.6, 0.3]
    samples = synthetic(random.Random(7), w_true, 5000)
    w = fit_logistic_weights(samples)  # defaults: l2 1e-3, 500 iters, lr 0.1
    assert len(w) == 3
    for fitted, truth in zip(w, w_true, strict=True):
        assert fitted == pytest.approx(truth, abs=0.1)  # sampling s.e. is about 0.035
    # sign and ordering of effect sizes are recovered
    assert w[0] > 0 > w[1] and w[2] > 0 and abs(w[0]) > abs(w[1]) > abs(w[2])


def test_fit_converges_to_a_stationary_point_of_the_regularised_loss() -> None:
    samples = synthetic(random.Random(11), [1.2, -0.7], 1500)
    l2 = 0.01
    w = fit_logistic_weights(samples, l2=l2, iters=400, lr=1.0)
    grad = [l2 * wj for wj in w]
    for x, y in samples:
        err = 1.0 / (1.0 + math.exp(-(w[0] * x[0] + w[1] * x[1]))) - y
        grad[0] += err * x[0] / len(samples)
        grad[1] += err * x[1] / len(samples)
    assert max(abs(g) for g in grad) < 1e-6
    # and a different (lr, iters) schedule lands on the same optimum
    other = fit_logistic_weights(samples, l2=l2, iters=1200, lr=0.5)
    assert other == [pytest.approx(w[0], abs=1e-4), pytest.approx(w[1], abs=1e-4)]


def test_fit_improves_calibration_over_uninformative_model() -> None:
    samples = synthetic(random.Random(3), [1.5, -1.0], 2000)
    w = fit_logistic_weights(samples, iters=200, lr=0.5)
    labels = [y for _, y in samples]
    fitted = [1.0 / (1.0 + math.exp(-(w[0] * x[0] + w[1] * x[1]))) for x, _ in samples]
    assert brier_score(fitted, labels) < 0.25 - 0.03
    assert log_loss(fitted, labels) < math.log(2.0) - 0.05


def test_fit_l2_shrinks_weights() -> None:
    samples = synthetic(random.Random(5), [1.0, 1.0], 800)
    light = fit_logistic_weights(samples, l2=1e-4, iters=300, lr=0.5)
    heavy = fit_logistic_weights(samples, l2=1.0, iters=300, lr=0.5)
    assert math.hypot(*heavy) < 0.5 * math.hypot(*light)


def test_fit_is_deterministic_and_antisymmetric_in_features() -> None:
    samples = synthetic(random.Random(9), [0.8, -0.4], 300)
    a = fit_logistic_weights(samples, iters=100, lr=0.5)
    assert a == fit_logistic_weights(samples, iters=100, lr=0.5)
    flipped = [([-v for v in x], y) for x, y in samples]
    b = fit_logistic_weights(flipped, iters=100, lr=0.5)
    assert b == [pytest.approx(-a[0], abs=1e-9), pytest.approx(-a[1], abs=1e-9)]


def test_fit_handles_separable_data_and_huge_features_without_overflow() -> None:
    samples = [([1000.0], 1), ([-1000.0], 0)] * 5
    w = fit_logistic_weights(samples, l2=1e-3, iters=50, lr=0.1)
    assert math.isfinite(w[0]) and w[0] > 0.0


def test_fit_accepts_tuples_and_does_not_mutate_input() -> None:
    samples: list[tuple[tuple[float, ...], int]] = [((1.0, 2.0), 1), ((-1.0, 0.5), 0)]
    snapshot = list(samples)
    fit_logistic_weights(samples, iters=5)
    assert samples == snapshot


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [([1.0], 1), ([1.0, 2.0], 0)],  # ragged
        [([1.0], 2)],  # bad label
        [([1.0], -1)],
        [([math.nan], 1)],
        [([math.inf], 0)],
        [([], 1)],  # no features
    ],
)
def test_fit_rejects_bad_samples(samples: list[tuple[list[float], int]]) -> None:
    with pytest.raises(ValueError, match="fit_logistic_weights"):
        fit_logistic_weights(samples)


@pytest.mark.parametrize("kw", [{"l2": -0.1}, {"iters": -1}, {"lr": 0.0}, {"lr": -0.1}])
def test_fit_rejects_bad_hyperparameters(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="fit_logistic_weights"):
        fit_logistic_weights([([1.0], 1)], **kw)  # type: ignore[arg-type]
