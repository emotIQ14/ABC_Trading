"""Tests for abc_trading.config (defaults, TOML loading, round trips, validation) and for the
``--set section.key=value`` override machinery of abc_trading.cli."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from abc_trading.cli import (
    ConfigError,
    apply_overrides,
    coerce_value,
    parse_override,
)
from abc_trading.config import (
    BotConfig,
    config_from_dict,
    config_to_dict,
    load_config,
)

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
        mod("sim", n_windows=1, tick_size=0.099),
    ],
)
def test_boundary_values_are_accepted(cfg: BotConfig) -> None:
    cfg.validate()


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
