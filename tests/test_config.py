"""Tests for abc_trading.config (defaults, TOML loading, round trips, validation) and for the
``--set section.key=value`` override machinery of abc_trading.cli."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import tomllib
import typing
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from abc_trading.backtest.runner import run_backtest
from abc_trading.cli import (
    ConfigError,
    apply_overrides,
    coerce_value,
    main,
    parse_override,
)
from abc_trading.config import (
    BotConfig,
    SimConfig,
    config_from_dict,
    config_to_dict,
    load_config,
)
from abc_trading.exchange.paper import PaperExchange
from abc_trading.fees import FeeModel
from abc_trading.model.fair_value import FairValueModel
from abc_trading.sim import feed as feed_mod
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.strategy.risk import RiskManager

DEFAULT_TOML = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def mod(section: str, **kw: object) -> BotConfig:
    """BotConfig() with the fields of one section replaced."""
    cfg = BotConfig()
    return replace(cfg, **{section: replace(getattr(cfg, section), **kw)})


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and prefix.count(".") < 1:  # sections only, dicts stay values
            out.update(flatten(value, path + "."))
        else:
            out[path] = value
    return out


# --------------------------------------------------------------------------- defaults and TOML


def test_defaults_validate() -> None:
    BotConfig().validate()
    assert load_config(None) == BotConfig()


def test_default_toml_equals_botconfig_defaults() -> None:
    assert load_config(DEFAULT_TOML) == BotConfig()
    assert load_config(str(DEFAULT_TOML)) == BotConfig()


def test_default_toml_lists_every_key_except_the_none_valued_one() -> None:
    toml = flatten(tomllib.loads(DEFAULT_TOML.read_text(encoding="utf-8")))
    defaults = flatten(config_to_dict(BotConfig()))
    assert defaults["sizing.clip_equity_fraction"] is None  # TOML cannot represent None
    assert set(toml) == set(defaults) - {"sizing.clip_equity_fraction"}
    for key, value in toml.items():
        assert value == defaults[key], key  # same value (lists compare equal to the dumped tuples)
        assert type(value) is type(defaults[key]), key  # 100 vs 100.0 would be a silent drift


def test_default_toml_documents_every_key_and_the_omitted_one() -> None:
    text = DEFAULT_TOML.read_text(encoding="utf-8")
    assignments = [line for line in text.splitlines() if re.match(r"^[a-z_0-9]+ *= ", line)]
    assert len(assignments) == len(flatten(config_to_dict(BotConfig()))) - 1
    assert all("#" in line for line in assignments), [a for a in assignments if "#" not in a]
    assert "clip_equity_fraction omitted" in text
    assert "not calibrated" in text.lower() or "NOT calibrated" in text


def test_partial_toml_fills_in_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.toml"
    path.write_text(
        '[pair]\ntarget_margin = 0.02\n[sim]\nassets = ["BTC"]\nn_windows = 3\n'
        "[sizing]\nclip_shares = 50\nclip_equity_fraction = 0.05\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.pair.target_margin == 0.02 and cfg.pair.merge_min_pairs == 50.0
    assert cfg.sim.assets == ("BTC",) and cfg.sim.n_windows == 3  # list -> tuple
    assert cfg.sizing.clip_shares == 50.0 and isinstance(cfg.sizing.clip_shares, float)
    assert cfg.sizing.clip_equity_fraction == 0.05
    assert cfg.fees == BotConfig().fees


def test_toml_errors_are_value_errors(tmp_path: Path) -> None:
    bad_key = tmp_path / "unknown.toml"
    bad_key.write_text("[pair]\ntarget_margins = 0.02\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"pair: unknown key\(s\) \['target_margins'\]"):
        load_config(bad_key)
    bad_syntax = tmp_path / "syntax.toml"
    bad_syntax.write_text("[pair\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_syntax)  # tomllib.TOMLDecodeError is a ValueError
    bad_value = tmp_path / "value.toml"
    bad_value.write_text("[pair]\ntarget_margin = 0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target_margin"):
        load_config(bad_value)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")


# --------------------------------------------------------------------------- round trips


def custom_config() -> BotConfig:
    cfg = BotConfig()
    return replace(
        cfg,
        sizing=replace(cfg.sizing, clip_equity_fraction=0.05, ladder_levels=4),
        sim=replace(
            cfg.sim, assets=("BTC",), spot0={"BTC": 61_000.0, "ETH": 3_100.0}, n_windows=3, seed=7
        ),
        fees=replace(cfg.fees, maker_rebate_rate=0.001),
        directional=replace(cfg.directional, enabled=False),
    )


@pytest.mark.parametrize("cfg", [BotConfig(), custom_config()], ids=["default", "custom"])
def test_config_round_trips_through_dict_and_json(cfg: BotConfig) -> None:
    data = config_to_dict(cfg)
    assert config_from_dict(data) == cfg
    text = json.dumps(data, allow_nan=False)
    assert config_from_dict(json.loads(text)) == cfg
    assert isinstance(data["sim"]["assets"], list)  # tuples are dumped as lists


def test_config_to_dict_is_independent_of_the_config() -> None:
    cfg = BotConfig()
    data = config_to_dict(cfg)
    data["sim"]["spot0"]["BTC"] = 1.0
    assert cfg.sim.spot0["BTC"] == 60_000.0


def test_config_from_dict_rejects_unknown_keys_and_non_tables() -> None:
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['bogus'\]"):
        config_from_dict({"bogus": {}})
    with pytest.raises(ValueError, match=r"pair: unknown key\(s\) \['bogus'\]"):
        config_from_dict({"pair": {"bogus": 1}})
    with pytest.raises(ValueError, match="expected a table"):
        config_from_dict({"pair": 3})
    assert config_from_dict({}) == BotConfig()


# --------------------------------------------------------------------------- validate() rules

INVALID: list[tuple[str, BotConfig, str]] = [
    ("target_margin<0", mod("pair", target_margin=-0.01), "pair.target_margin"),
    ("target_margin>=0.5", mod("pair", target_margin=0.5), "pair.target_margin"),
    ("taker_lock_margin<0", mod("pair", taker_lock_margin=-0.001), "pair.taker_lock_margin"),
    ("max_loss<0", mod("pair", rebalance_max_loss_per_pair=-0.1), "rebalance_max_loss_per_pair"),
    ("net_imbalance<0", mod("pair", max_net_imbalance_shares=-1.0), "max_net_imbalance_shares"),
    (
        "side_cap<net_cap",
        mod("pair", max_inventory_per_side_shares=100.0, max_net_imbalance_shares=300.0),
        "max_inventory_per_side_shares must be >=",
    ),
    ("merge_min_pairs=0", mod("pair", merge_min_pairs=0.0), "pair.merge_min_pairs"),
    ("clip=0", mod("sizing", clip_shares=0.0), "sizing.clip_shares"),
    ("min_clip=0", mod("sizing", min_clip_shares=0.0), "sizing.min_clip_shares"),
    (
        "min_clip>max_clip",
        mod("sizing", min_clip_shares=2000.0, max_clip_shares=1000.0),
        "sizing.min_clip_shares",
    ),
    ("ladder_levels=0", mod("sizing", ladder_levels=0), "sizing.ladder_levels"),
    ("spacing=0", mod("sizing", ladder_spacing_ticks=0), "sizing.ladder_spacing_ticks"),
    ("decay=0", mod("sizing", ladder_size_decay=0.0), "sizing.ladder_size_decay"),
    ("decay>1", mod("sizing", ladder_size_decay=1.1), "sizing.ladder_size_decay"),
    ("equity_fraction=0", mod("sizing", clip_equity_fraction=0.0), "clip_equity_fraction"),
    ("equity_fraction>1", mod("sizing", clip_equity_fraction=1.5), "clip_equity_fraction"),
    ("min_edge<0", mod("directional", min_edge=-0.01), "directional.min_edge"),
    ("min_edge>=0.5", mod("directional", min_edge=0.5), "directional.min_edge"),
    ("max_dir<0", mod("directional", max_directional_shares=-1.0), "max_directional_shares"),
    ("kelly<0", mod("directional", kelly_fraction=-0.1), "directional.kelly_fraction"),
    ("kelly>1", mod("directional", kelly_fraction=1.5), "directional.kelly_fraction"),
    ("flatten>wind_down", mod("timing", flatten_seconds=100.0, wind_down_seconds=90.0), "timing:"),
    ("flatten<0", mod("timing", flatten_seconds=-1.0), "timing:"),
    ("warmup<0", mod("timing", warmup_seconds=-1.0), "timing.warmup_seconds"),
    ("shrink<0", mod("model", shrink_to_market=-0.1), "model.shrink_to_market"),
    ("shrink>1", mod("model", shrink_to_market=1.5), "model.shrink_to_market"),
    ("p_floor=0", mod("model", p_floor=0.0), "model.p_floor"),
    ("p_floor=0.5", mod("model", p_floor=0.5), "model.p_floor"),
    ("halflife=0", mod("model", vol_halflife_seconds=0.0), "model vol params"),
    ("vol_floor=0", mod("model", vol_floor=0.0), "model vol params"),
    ("min_tau=0", mod("model", min_tau_seconds=0.0), "model.min_tau_seconds"),
    ("maker_fee<0", mod("fees", maker_fee_rate=-0.1), "fee rates"),
    ("maker_rebate<0", mod("fees", maker_rebate_rate=-0.1), "fee rates"),
    ("taker_fee<0", mod("fees", taker_fee_rate=-0.1), "taker fee params"),
    ("taker_exponent<0", mod("fees", taker_fee_exponent=-1.0), "taker fee params"),
    ("market_cap=0", mod("risk", max_capital_per_market_usd=0.0), "risk caps"),
    ("total_cap=0", mod("risk", max_total_capital_usd=0.0), "risk caps"),
    ("daily_loss=0", mod("risk", max_daily_loss_usd=0.0), "risk.max_daily_loss_usd"),
    ("orders_per_s=0", mod("risk", max_orders_per_second=0.0), "risk.max_orders_per_second"),
    ("cash=0", mod("exchange", initial_cash=0.0), "exchange.initial_cash"),
    ("latency<0", mod("exchange", latency_ticks=-1), "exchange.latency_ticks"),
    ("queue<0", mod("exchange", queue_ahead_fraction=-0.1), "queue params"),
    ("trade_fill>1", mod("exchange", trade_fill_fraction=1.1), "queue params"),
    ("trade_fill<0", mod("exchange", trade_fill_fraction=-0.1), "queue params"),
    ("window=0", mod("sim", window_seconds=0), "sim timing"),
    ("tick_seconds=0", mod("sim", tick_seconds=0.0), "sim timing"),
    ("n_windows=0", mod("sim", n_windows=0), "sim.n_windows"),
    ("tick_size=0", mod("sim", tick_size=0.0), "sim.tick_size"),
    ("tick_size=0.1", mod("sim", tick_size=0.1), "sim.tick_size"),
    ("asset w/o spot0", mod("sim", assets=("SOL",)), "missing spot0/annual_vol for SOL"),
    (
        "asset w/o vol",
        mod("sim", assets=("SOL",), spot0={"SOL": 150.0}),
        "missing spot0/annual_vol for SOL",
    ),
    # --- ranges that used to pass validate() and then silently neutered or inverted a control,
    # or failed later (mid-run or only at feed construction) without naming the config key
    ("requote_tol<0", mod("sizing", requote_tolerance_ticks=-1), "sizing.requote_tolerance_ticks"),
    ("max_spread=0", mod("risk", max_spread_ticks_to_quote=0), "risk.max_spread_ticks_to_quote"),
    ("max_spread<0", mod("risk", max_spread_ticks_to_quote=-1), "risk.max_spread_ticks_to_quote"),
    ("staleness<0", mod("risk", max_spot_staleness_seconds=-1.0), "risk.max_spot_staleness"),
    ("orders_per_s<1", mod("risk", max_orders_per_second=0.5), "risk.max_orders_per_second"),
    (
        "requote_interval<0",
        mod("timing", min_requote_interval_seconds=-1.0),
        "min_requote_interval",
    ),
    ("skew<0", mod("pair", skew_ticks_per_100_shares=-1.0), "pair.skew_ticks_per_100_shares"),
    ("rebalance_trigger<0", mod("pair", rebalance_trigger_shares=-1.0), "rebalance_trigger_shares"),
    ("maker_rebate>1", mod("fees", maker_rebate_rate=5.0), "fees.maker_rebate_rate"),
    ("maker_fee>1", mod("fees", maker_fee_rate=1.5), "fees.maker_fee_rate"),
    ("taker_fee>1", mod("fees", taker_fee_rate=1.5), "fees.taker_fee_rate"),
    ("slippage<0", mod("exchange", taker_slippage_ticks=-1), "exchange.taker_slippage_ticks"),
    ("momentum_lookback=0", mod("model", momentum_lookback_seconds=0.0), "momentum_lookback"),
    ("accel_lookback<0", mod("model", accel_lookback_seconds=-1.0), "accel_lookback"),
    ("no assets", mod("sim", assets=()), "sim.assets must not be empty"),
    ("duplicate assets", mod("sim", assets=("BTC", "BTC")), "sim.assets must be unique"),
    ("tick off grid", mod("sim", tick_size=0.07), "sim.tick_size"),
    ("tick off grid 2", mod("sim", tick_size=0.099), "sim.tick_size"),
    ("tick too fine", mod("sim", tick_size=0.00001), "sim.tick_size"),
    ("window w/o accumulate", mod("sim", window_seconds=60), "sim.window_seconds"),
    ("window = warmup+wind_down", mod("sim", window_seconds=105), "sim.window_seconds"),
    (
        "window not a tick multiple",
        mod("sim", tick_seconds=7.0),
        "sim.window_seconds must be a multiple of sim.tick_seconds",
    ),
    ("fractional window", mod("sim", window_seconds=900.5), "sim.window_seconds"),
    ("min_order_size=0", mod("sim", min_order_size=0.0), "sim.min_order_size"),
    ("spot0<=0", mod("sim", spot0={"BTC": 0.0, "ETH": 3000.0}), "sim.spot0[BTC]"),
    ("annual_vol<=0", mod("sim", annual_vol={"BTC": 0.55, "ETH": -0.1}), "sim.annual_vol[ETH]"),
    ("vol_multiplier=0", mod("sim", market_vol_multiplier=0.0), "sim.market_vol_multiplier"),
    ("lag<0", mod("sim", market_lag_seconds=-1.0), "sim.market_lag_seconds"),
    ("noise<0", mod("sim", pricing_noise_ticks=-0.1), "sim.pricing_noise_ticks"),
    ("spread<1", mod("sim", mean_spread_ticks=0.5), "sim.mean_spread_ticks"),
    ("depth=0", mod("sim", depth_mean_shares=0.0), "sim.depth_mean_shares"),
    ("n_levels=0", mod("sim", n_levels=0), "sim.n_levels"),
    ("uninformed<0", mod("sim", uninformed_trades_per_sec=-1.0), "uninformed_trades_per_sec"),
    ("informed>1", mod("sim", informed_flow=1.5), "sim.informed_flow"),
    ("informed<0", mod("sim", informed_flow=-0.5), "sim.informed_flow"),
    # non-finite numbers anywhere (also in the fields that have no range check of their own)
    ("hold_margin=nan", mod("directional", hold_margin=float("nan")), "directional.hold_margin"),
    ("w_momentum=inf", mod("model", w_momentum=float("inf")), "model.w_momentum"),
    ("max_clip=inf", mod("sizing", max_clip_shares=float("inf")), "sizing.max_clip_shares"),
    ("tolerance float nan", mod("sim", pricing_noise_ticks=float("nan")), "pricing_noise_ticks"),
    ("spot0 inf", mod("sim", spot0={"BTC": float("inf"), "ETH": 3000.0}), "sim.spot0[BTC]"),
]


@pytest.mark.parametrize(("name", "cfg", "message"), INVALID, ids=[i[0] for i in INVALID])
def test_each_validate_rule_has_a_failing_example(name: str, cfg: BotConfig, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        cfg.validate()


def test_invalid_configs_are_rejected_when_loaded_from_dicts() -> None:
    for _, cfg, message in INVALID[:6]:
        with pytest.raises(ValueError, match=re.escape(message)):
            config_from_dict(config_to_dict(cfg))


@pytest.mark.parametrize("field", ["target_margin", "taker_lock_margin"])
def test_nan_is_rejected_by_the_validated_ranges(field: str) -> None:
    with pytest.raises(ValueError):
        mod("pair", **{field: float("nan")}).validate()
    with pytest.raises(ValueError):
        mod("risk", max_daily_loss_usd=float("nan")).validate()
    with pytest.raises(ValueError):
        mod("fees", maker_fee_rate=float("nan")).validate()


@pytest.mark.parametrize(
    "cfg",
    [
        mod("pair", target_margin=0.0),
        mod("pair", target_margin=0.4999),
        mod("pair", taker_lock_margin=0.0, rebalance_max_loss_per_pair=0.0),
        mod("pair", max_inventory_per_side_shares=300.0),  # == max_net_imbalance_shares
        mod("sizing", clip_equity_fraction=1.0, ladder_size_decay=1.0),
        mod("sizing", min_clip_shares=1000.0),  # == max_clip_shares
        mod("directional", min_edge=0.0, kelly_fraction=0.0, max_directional_shares=0.0),
        mod("directional", kelly_fraction=1.0),
        mod("timing", flatten_seconds=90.0, wind_down_seconds=90.0, warmup_seconds=0.0),
        mod("timing", flatten_seconds=0.0, wind_down_seconds=0.0),
        mod("model", shrink_to_market=0.0),
        mod("model", shrink_to_market=1.0, p_floor=1e-6),
        mod("exchange", latency_ticks=0, queue_ahead_fraction=0.0, trade_fill_fraction=0.0),
        mod("exchange", trade_fill_fraction=1.0, queue_ahead_fraction=5.0),
        mod("sim", n_windows=1, tick_size=0.05),
        mod("sim", tick_size=0.001),
        mod("sim", tick_size=0.0001),  # the finest grid the feed supports (10 000 ticks)
        mod("sim", window_seconds=106),  # one second of ACCUMULATE after 15 s warm-up + 90 s
        mod("sim", window_seconds=900, tick_seconds=0.5),
        mod("sim", informed_flow=0.0, uninformed_trades_per_sec=0.0, market_lag_seconds=0.0),
        mod("sim", informed_flow=1.0, mean_spread_ticks=1.0, pricing_noise_ticks=0.0),
        mod("risk", max_spread_ticks_to_quote=1, max_spot_staleness_seconds=0.0),
        mod("risk", max_orders_per_second=1.0),
        mod("sizing", requote_tolerance_ticks=0),
        mod("pair", skew_ticks_per_100_shares=0.0, rebalance_trigger_shares=0.0),  # 0 = any excess
        mod("timing", min_requote_interval_seconds=0.0),
        mod("fees", maker_fee_rate=1.0, maker_rebate_rate=1.0, taker_fee_rate=1.0),
        mod("exchange", initial_cash=1001.0, taker_slippage_ticks=0),  # > max_daily_loss_usd
    ],
)
def test_boundary_values_are_accepted(cfg: BotConfig) -> None:
    cfg.validate()


# --------------------------------------------------------------------------- strict value types

# (name, TOML text, expected error). Every row used to load without a complaint: the quoted
# booleans stayed truthy ("false" switched merging/directional trading ON), and the floats and
# strings crashed later with a bare TypeError (exit 1) or were echoed back by `config` (exit 0).
WRONG_TYPE_TOML: list[tuple[str, str, str]] = [
    (
        "quoted merge switch",
        '[pair]\nmerge_enabled = "false"\n',
        "pair.merge_enabled: expected a boolean, got str",
    ),
    (
        "quoted directional switch",
        '[directional]\nenabled = "false"\n',
        "directional.enabled: expected a boolean, got str",
    ),
    (
        "int for bool",
        "[pair]\nmerge_enabled = 1\n",
        "pair.merge_enabled: expected a boolean, got int",
    ),
    (
        "float for int",
        "[sizing]\nladder_levels = 3.0\n",
        "sizing.ladder_levels: expected an integer, got float",
    ),
    (
        "float for sim int",
        "[sim]\nn_levels = 5.0\n",
        "sim.n_levels: expected an integer, got float",
    ),
    (
        "fractional n_windows",
        "[sim]\nn_windows = 2.5\n",
        "sim.n_windows: expected an integer, got float",
    ),
    (
        "bool for int",
        "[sizing]\nladder_levels = true\n",
        "sizing.ladder_levels: expected an integer, got bool",
    ),
    (
        "str for int",
        '[exchange]\nlatency_ticks = "1"\n',
        "exchange.latency_ticks: expected an integer, got str",
    ),
    (
        "str for float",
        '[sizing]\nclip_shares = "100"\n',
        "sizing.clip_shares: expected a number, got str",
    ),
    (
        "bool for float",
        "[pair]\ntarget_margin = true\n",
        "pair.target_margin: expected a number, got bool",
    ),
    (
        "str for optional float",
        '[sizing]\nclip_equity_fraction = "0.1"\n',
        "sizing.clip_equity_fraction: expected a number, got str",
    ),
    ("nan", "[sizing]\nclip_shares = nan\n", "sizing.clip_shares: expected a finite number"),
    (
        "inf",
        "[sizing]\nmax_clip_shares = inf\n",
        "sizing.max_clip_shares: expected a finite number",
    ),
    (
        "huge int for float",
        "[sizing]\nclip_shares = 1" + "0" * 400 + "\n",
        "sizing.clip_shares: expected a finite number",
    ),
    ("str for tuple", '[sim]\nassets = "BTC"\n', "sim.assets: expected a list, got str"),
    (
        "number in str tuple",
        '[sim]\nassets = ["BTC", 1]\n',
        "sim.assets[1]: expected a string, got int",
    ),
    ("number for table", "[sim]\nspot0 = 5\n", "sim.spot0: expected a table, got int"),
    (
        "str in float table",
        '[sim.spot0]\nBTC = "60000"\n',
        "sim.spot0.BTC: expected a number, got str",
    ),
    (
        "bool in float table",
        "[sim.annual_vol]\nBTC = true\n",
        "sim.annual_vol.BTC: expected a number, got bool",
    ),
    ("list for section", "pair = [1]\n", "pair: expected a table, got list"),
]


@pytest.mark.parametrize(
    ("toml_text", "message"),
    [(t, m) for _, t, m in WRONG_TYPE_TOML],
    ids=[n for n, _, _ in WRONG_TYPE_TOML],
)
def test_toml_values_of_the_wrong_type_are_rejected_naming_the_key(
    tmp_path: Path, toml_text: str, message: str
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(toml_text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        load_config(path)


def wrong_values(tp: Any) -> list[Any]:
    """Values a TOML file can hold that are NOT acceptable for a field declared as ``tp``."""
    origin = typing.get_origin(tp)
    if tp is bool:
        return ["false", "true", 1, 0, 1.0, None]
    if tp is int:
        return [1.5, 3.0, "1", True, None, [1]]
    if tp is float:
        return ["1", "1.0", True, None, [1.0], float("nan"), float("inf")]
    if tp == float | None:
        return ["1", True, [1.0], float("nan")]
    if origin is tuple:
        return ["BTC", 1, [1], {"BTC": 1}, [["BTC"]]]
    if origin is dict:
        return [5, "BTC", [1.0], {"BTC": "x"}, {"BTC": True}, {"BTC": None}]
    raise AssertionError(f"no wrong-type examples for {tp!r}: extend wrong_values()")


def every_field() -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    for section, section_type in typing.get_type_hints(BotConfig).items():
        for key, tp in typing.get_type_hints(section_type).items():
            out.append((section, key, tp))
    return out


@pytest.mark.parametrize(("section", "key", "tp"), every_field(), ids=lambda v: str(v)[:24])
def test_every_field_rejects_every_wrong_type(section: str, key: str, tp: Any) -> None:
    for bad in wrong_values(tp):
        with pytest.raises(ValueError, match=re.escape(f"{section}.{key}")):
            config_from_dict({section: {key: bad}})


def test_the_declared_types_are_accepted_and_normalised() -> None:
    cfg = config_from_dict(
        {
            "sizing": {"clip_shares": 50, "clip_equity_fraction": 1, "ladder_levels": 4},
            "pair": {"merge_enabled": False, "target_margin": 0},
            "sim": {
                "assets": ["BTC"],  # a list becomes a tuple
                "spot0": {"BTC": 61_000, "ETH": 3_000},  # ints become floats
                "annual_vol": {"BTC": 1, "ETH": 1},
                "n_windows": 3,
            },
        }
    )
    assert cfg.sizing.clip_shares == 50.0 and type(cfg.sizing.clip_shares) is float
    assert cfg.sizing.clip_equity_fraction == 1.0 and type(cfg.sizing.clip_equity_fraction) is float
    assert cfg.sizing.ladder_levels == 4 and type(cfg.sizing.ladder_levels) is int
    assert cfg.pair.merge_enabled is False and cfg.pair.target_margin == 0.0
    assert cfg.sim.assets == ("BTC",) and isinstance(cfg.sim.assets, tuple)
    assert cfg.sim.spot0 == {"BTC": 61_000.0, "ETH": 3_000.0}
    assert all(type(v) is float for v in [*cfg.sim.spot0.values(), *cfg.sim.annual_vol.values()])
    assert config_from_dict({"sim": {"assets": ("BTC", "ETH")}}).sim.assets == ("BTC", "ETH")
    assert config_from_dict({"sizing": {"clip_equity_fraction": None}}).sizing == (
        BotConfig().sizing
    )


CLI_CASES = [
    (name, text, message.split(":")[0])
    for name, text, message in WRONG_TYPE_TOML
    if name
    in {
        "quoted merge switch",
        "quoted directional switch",
        "float for int",
        "float for sim int",
        "fractional n_windows",
        "str for float",
        "str for tuple",
        "str in float table",
    }
]


@pytest.mark.parametrize(
    ("toml_text", "key"), [(t, k) for _, t, k in CLI_CASES], ids=[n for n, _, _ in CLI_CASES]
)
def test_cli_reports_wrong_toml_types_as_configuration_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], toml_text: str, key: str
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(toml_text, encoding="utf-8")
    for argv in (
        ["config", "--config", str(path)],
        ["backtest", "--assets", "BTC", "--windows", "1", "--config", str(path)],
    ):
        assert main(argv) == 2, argv  # not 0 (silently accepted) and not an uncaught TypeError
        captured = capsys.readouterr()
        assert key in captured.err and "expected" in captured.err, captured.err
        assert captured.out == ""  # nothing was printed or run


def test_a_quoted_false_no_longer_leaves_merging_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["backtest", "--assets", "BTC", "--windows", "1", "--set", "sim.window_seconds=300"]
    no_merge_pnl = re.compile(r"PnL from merges\s+\+0\.00\b")
    assert main(base) == 0
    assert not no_merge_pnl.search(capsys.readouterr().out)  # control: merging is on by default
    real = tmp_path / "real.toml"
    real.write_text("[pair]\nmerge_enabled = false\n", encoding="utf-8")
    assert main([*base, "--config", str(real)]) == 0
    assert no_merge_pnl.search(capsys.readouterr().out)
    quoted = tmp_path / "quoted.toml"  # used to load as the truthy string "false": merging ON
    quoted.write_text('[pair]\nmerge_enabled = "false"\n', encoding="utf-8")
    assert main([*base, "--config", str(quoted)]) == 2
    captured = capsys.readouterr()
    assert "pair.merge_enabled: expected a boolean" in captured.err and captured.out == ""


def test_overrides_still_round_trip_every_field_type() -> None:
    cfg = apply_overrides(
        BotConfig(),
        ["sim.assets=ETH", "sim.spot0=ETH=3100", "sizing.clip_equity_fraction=0.1",
         "pair.merge_enabled=no", "sim.n_windows=2", "fees.taker_fee_rate=1"],
    )  # fmt: skip
    assert cfg.sim.assets == ("ETH",) and cfg.sim.spot0["ETH"] == 3100.0
    assert cfg.pair.merge_enabled is False and cfg.fees.taker_fee_rate == 1.0


# --------------------------------------------------------------------------- validate() vs the rest


def downstream_rejections(cfg: BotConfig) -> list[str]:
    """Names of the components that refuse to be built from ``cfg`` (any exception counts)."""
    builders = {
        "SyntheticFeed": lambda: SyntheticFeed(cfg),
        "FairValueModel": lambda: FairValueModel(cfg.model),
        "PaperExchange": lambda: PaperExchange(cfg),
        "RiskManager": lambda: RiskManager(cfg.risk, cfg.exchange.initial_cash),
        "FeeModel": lambda: FeeModel(cfg.fees),
    }
    rejected = []
    for name, build in builders.items():
        try:
            build()
        except Exception:
            rejected.append(name)
    return rejected


def candidate_values(tp: Any) -> list[Any]:
    if tp is int:
        return [-1, 0, 1, 2, 7]
    if tp is float or tp == float | None:
        nan, inf = float("nan"), float("inf")
        return [nan, inf, -inf, -1.0, 0.0, 0.5, 1.0, 1.5, 3.0]
    if typing.get_origin(tp) is tuple:
        return [(), ("BTC", "BTC"), ("SOL",), ("BTC",)]
    if typing.get_origin(tp) is dict:
        return [
            {"BTC": 0.0, "ETH": 1.0}, {"BTC": -1.0, "ETH": 1.0}, {"BTC": 1.0, "ETH": float("nan")},
            {"BTC": 1.0}, {"ETH": 1.0},
        ]  # fmt: skip
    return []  # bool: both values are fine


def test_validate_is_at_least_as_strict_as_every_component_constructor() -> None:
    """Whatever SyntheticFeed, FairValueModel, PaperExchange, RiskManager or FeeModel refuse to be
    built from (a late failure that names no config key and that `config` never sees) is refused
    by validate() first. Single-field mutations of a small valid config, every field of every
    section."""
    base = replace(BotConfig(), sim=replace(BotConfig().sim, n_windows=1, assets=("BTC",)))
    base = replace(base, sim=replace(base.sim, window_seconds=300))
    base.validate()
    assert downstream_rejections(base) == []
    checked = rejected = 0
    for section, key, tp in every_field():
        for value in candidate_values(tp):
            cfg = replace(base, **{section: replace(getattr(base, section), **{key: value})})
            checked += 1
            late = downstream_rejections(cfg)
            if not late:
                continue
            rejected += 1
            with pytest.raises(ValueError, match="."):
                cfg.validate()  # AssertionError here names the case through the parametrization
    assert checked > 300 and rejected > 100  # the mutation set really exercised the checks


@pytest.mark.parametrize(("name", "cfg", "message"), INVALID, ids=[i[0] for i in INVALID])
def test_new_rules_agree_with_the_constructors(name: str, cfg: BotConfig, message: str) -> None:
    """Where a component also refuses the config, validate() must too (the converse is not
    required: validate() is allowed to be stricter, e.g. for a silently inert setting)."""
    if downstream_rejections(cfg):
        with pytest.raises(ValueError, match=re.escape(message)):
            cfg.validate()


def test_validate_checks_every_cross_field_dependency_the_repro_listed() -> None:
    # sim.window_seconds=60 ran to completion with zero fills (no ACCUMULATE phase at all)
    with pytest.raises(ValueError, match=r"timing.*sim\.window_seconds"):
        apply_overrides(BotConfig(), ["sim.window_seconds=60"])
    # the maker rebate may not exceed the notional: the cost of a bid would turn negative
    with pytest.raises(ValueError, match="fees.maker_rebate_rate"):
        apply_overrides(BotConfig(), ["fees.maker_rebate_rate=5"])
    # sim.assets= and sim.tick_size=0.07 used to pass `config` and fail in `backtest`/`record`
    for override in ("sim.assets=BTC,BTC", "sim.tick_size=0.07", "sim.tick_seconds=7"):
        with pytest.raises(ValueError, match="sim"):
            apply_overrides(BotConfig(), [override])


def test_a_kill_switch_that_can_never_fire_is_an_advisory_not_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Not an error: cash-limited runs (small initial_cash with the default loss limit) are a
    # legitimate setup and several suites use them. It is surfaced instead.
    assert BotConfig().advisories() == []
    small = mod("exchange", initial_cash=500.0)
    small.validate()
    (note,) = small.advisories()
    assert "risk.max_daily_loss_usd (1000)" in note and "exchange.initial_cash (500)" in note
    assert "kill switch can never fire" in note
    assert mod("exchange", initial_cash=1000.0).advisories() == [note.replace("500", "1000")]
    assert replace(small, risk=replace(small.risk, max_daily_loss_usd=499.0)).advisories() == []
    assert mod("exchange", initial_cash=1001.0).advisories() == []
    # the CLI prints it on stderr and carries on
    assert main(["config", "--set", "exchange.initial_cash=500"]) == 0
    captured = capsys.readouterr()
    assert captured.err.startswith("warning: risk.max_daily_loss_usd (1000)")
    assert json.loads(captured.out)["exchange"]["initial_cash"] == 500.0
    assert main(["config"]) == 0
    assert capsys.readouterr().err == ""  # the defaults are silent


@pytest.mark.parametrize("rebate", [0.5, 1.0])
def test_the_largest_accepted_maker_rebate_runs_to_completion(rebate: float) -> None:
    cfg = apply_overrides(
        BotConfig(),
        ["sim.n_windows=1", "sim.assets=BTC", "sim.window_seconds=300",
         f"fees.maker_rebate_rate={rebate}"],
    )  # fmt: skip
    result = run_backtest(cfg, SyntheticFeed(cfg))  # invariants 1-4 are checked inside
    assert result.activity.n_fills > 0 and math.isfinite(result.total_pnl)


def test_the_tick_grid_constants_match_the_feed() -> None:
    from abc_trading import config as config_mod

    assert config_mod.MAX_PRICE_TICKS == feed_mod.MAX_PRICE_TICKS
    assert config_mod.GRID_TOL == feed_mod._GRID_TOL


def test_every_sim_field_is_covered_by_the_mutation_test() -> None:
    # guards the generic test above: a new SimConfig field with an unsupported type would
    # silently get no candidate values
    for f in dataclasses.fields(SimConfig):
        tp = typing.get_type_hints(SimConfig)[f.name]
        assert tp is bool or candidate_values(tp), f.name


# --------------------------------------------------------------------------- --set: parsing


def test_parse_override_splits_on_the_first_equals_sign() -> None:
    assert parse_override("pair.target_margin=0.02") == ("pair", "target_margin", "0.02")
    assert parse_override("  pair.x = 1 ") == ("pair", "x", "1")  # whitespace is trimmed
    assert parse_override("sim.spot0=BTC=61000") == ("sim", "spot0", "BTC=61000")
    assert parse_override("sim.assets=") == ("sim", "assets", "")


@pytest.mark.parametrize(
    "text", ["", "novalue", "pair=1", "a.b.c=1", ".key=1", "section.=1", "=1", "pair.key"]
)
def test_parse_override_rejects_malformed_text(text: str) -> None:
    with pytest.raises(ConfigError, match="expected section.key=value"):
        parse_override(text)


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


# --------------------------------------------------------------------------- --set: coercion


def test_overrides_are_coerced_through_the_field_types() -> None:
    cfg = apply_overrides(
        BotConfig(),
        [
            "pair.target_margin=0.02",  # float
            "sizing.ladder_levels=5",  # int
            "sizing.clip_shares=50",  # float field given an integer literal
            "directional.enabled=false",  # bool
            "sizing.clip_equity_fraction=0.05",  # float | None
            "sim.assets=BTC",  # tuple[str, ...]
            "exchange.latency_ticks=0",
        ],
    )
    assert cfg.pair.target_margin == 0.02 and isinstance(cfg.pair.target_margin, float)
    assert cfg.sizing.ladder_levels == 5 and type(cfg.sizing.ladder_levels) is int
    assert cfg.sizing.clip_shares == 50.0 and type(cfg.sizing.clip_shares) is float
    assert cfg.directional.enabled is False
    assert cfg.sizing.clip_equity_fraction == 0.05
    assert cfg.sim.assets == ("BTC",)
    assert cfg.exchange.latency_ticks == 0
    assert cfg.fees == BotConfig().fees  # untouched sections are unchanged


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("True", True), ("YES", True), ("on", True), ("1", True),
     ("false", False), ("No", False), ("OFF", False), ("0", False)],
)  # fmt: skip
def test_bool_spellings(raw: str, expected: bool) -> None:
    assert coerce_value(bool, raw, "p") is expected


def test_bad_values_are_config_errors() -> None:
    with pytest.raises(ConfigError, match="not a boolean"):
        coerce_value(bool, "maybe", "directional.enabled")
    with pytest.raises(ConfigError, match="not an integer"):
        coerce_value(int, "1.5", "sizing.ladder_levels")
    with pytest.raises(ConfigError, match="not an integer"):
        coerce_value(int, "abc", "sizing.ladder_levels")
    with pytest.raises(ConfigError, match="not a number"):
        coerce_value(float, "abc", "pair.target_margin")
    for bad in ("nan", "inf", "-inf"):
        with pytest.raises(ConfigError, match="not a finite number"):
            coerce_value(float, bad, "pair.target_margin")
    with pytest.raises(ConfigError, match="unsupported config type"):
        coerce_value(list[int], "1", "p")


def test_optional_float_accepts_none_spellings() -> None:
    tp = float | None
    assert coerce_value(tp, "none", "p") is None
    assert coerce_value(tp, "NULL", "p") is None
    assert coerce_value(tp, "0.25", "p") == 0.25
    with pytest.raises(ConfigError, match="not a number"):
        coerce_value(tp, "abc", "p")
    with pytest.raises(ConfigError, match="not a number"):
        coerce_value(float, "none", "p")  # only Optional fields accept none
    cfg = apply_overrides(
        mod("sizing", clip_equity_fraction=0.1), ["sizing.clip_equity_fraction=none"]
    )
    assert cfg.sizing.clip_equity_fraction is None


def test_tuple_and_dict_values() -> None:
    assert coerce_value(tuple[str, ...], "BTC, ETH,", "p") == ("BTC", "ETH")
    assert coerce_value(tuple[float, ...], "1,2.5", "p") == (1.0, 2.5)
    assert coerce_value(dict[str, float], "A=1, B=2.5", "p") == {"A": 1.0, "B": 2.5}
    assert coerce_value(str, "text", "p") == "text"
    for bad in ("A", "=1", "A=x"):
        with pytest.raises(ConfigError):
            coerce_value(dict[str, float], bad, "p")


def test_dict_overrides_are_merged_into_the_existing_mapping() -> None:
    cfg = apply_overrides(BotConfig(), ["sim.spot0=BTC=61000"])
    assert cfg.sim.spot0 == {"BTC": 61_000.0, "ETH": 3_000.0}  # ETH kept
    cfg = apply_overrides(
        BotConfig(), ["sim.spot0=SOL=150", "sim.annual_vol=SOL=0.9", "sim.assets=SOL"]
    )
    assert cfg.sim.spot0 == {"BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0}
    assert cfg.sim.assets == ("SOL",)


# --------------------------------------------------------------------------- --set: errors


def test_unknown_section_and_key_are_config_errors() -> None:
    with pytest.raises(ConfigError, match=r"unknown config section 'bogus'.*'pair'"):
        apply_overrides(BotConfig(), ["bogus.x=1"])
    with pytest.raises(ConfigError, match=r"unknown key pair\.bogus.*target_margin"):
        apply_overrides(BotConfig(), ["pair.bogus=1"])
    with pytest.raises(ConfigError, match="expected section.key=value"):
        apply_overrides(BotConfig(), ["pair"])


def test_overrides_are_validated_after_applying() -> None:
    with pytest.raises(ValueError, match="pair.target_margin must be in"):
        apply_overrides(BotConfig(), ["pair.target_margin=0.9"])
    with pytest.raises(ValueError, match="missing spot0/annual_vol for SOL"):
        apply_overrides(BotConfig(), ["sim.assets=SOL"])
    # individually valid overrides can be jointly invalid
    with pytest.raises(ValueError, match="max_inventory_per_side_shares"):
        apply_overrides(BotConfig(), ["pair.max_inventory_per_side_shares=100"])


def test_overrides_apply_in_order_and_leave_the_input_untouched() -> None:
    base = BotConfig()
    cfg = apply_overrides(base, ["pair.target_margin=0.02", "pair.target_margin=0.03"])
    assert cfg.pair.target_margin == 0.03  # the later override wins
    assert base == BotConfig()
    assert apply_overrides(base, []) == base
    assert isinstance(cfg, BotConfig)


def test_overrides_compose_with_a_loaded_file(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[pair]\ntarget_margin = 0.02\nmerge_min_pairs = 10\n", encoding="utf-8")
    cfg = apply_overrides(load_config(path), ["pair.merge_min_pairs=20"])
    assert (cfg.pair.target_margin, cfg.pair.merge_min_pairs) == (0.02, 20.0)
