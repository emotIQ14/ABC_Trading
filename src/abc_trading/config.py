"""Configuration for every module. FROZEN CONTRACT (see ``types.py``).

All defaults are *illustrative starting points*, not tuned or validated values. Nothing in
here has been calibrated against real Polymarket data. Fee parameters in particular MUST be
checked against Polymarket's current published fee schedule before any PnL number is trusted.
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeeConfig:
    """fee_usdc = size * price * rate * (price * (1 - price)) ** exponent   (taker)

    The taker curve is a placeholder shaped like the fee curve Polymarket described for
    short-dated crypto markets (largest near 50c, tiny near 0/100c). VERIFY before trusting.
    Makers pay ``maker_fee_rate * notional`` and receive ``maker_rebate_rate * notional``.
    """

    maker_fee_rate: float = 0.0
    maker_rebate_rate: float = 0.0
    taker_fee_rate: float = 0.25
    taker_fee_exponent: float = 2.0


@dataclass(frozen=True)
class SizingConfig:
    clip_shares: float = 100.0  # base size per ladder level (~$50 at 50c)
    clip_equity_fraction: float | None = None  # if set: clip = equity * fraction / price
    min_clip_shares: float = 5.0
    max_clip_shares: float = 1000.0
    ladder_levels: int = 3
    ladder_spacing_ticks: int = 1
    ladder_size_decay: float = 0.7  # size multiplier per deeper ladder level
    requote_tolerance_ticks: int = 0  # keep a resting order if price within this many ticks


@dataclass(frozen=True)
class PairConfig:
    target_margin: float = 0.01  # want avg_up_cost + avg_down_cost <= 1 - target_margin
    taker_lock_margin: float = 0.005  # min NET margin to complete a pair by crossing the spread
    rebalance_max_loss_per_pair: float = 0.0  # extra loss tolerated when completing a pair
    max_net_imbalance_shares: float = 300.0  # |UP - DOWN| cap in hedged-neutral mode
    max_inventory_per_side_shares: float = 1500.0
    skew_ticks_per_100_shares: float = 1.0  # bid skew against the heavy side
    rebalance_trigger_shares: float = 100.0  # unpaired qty that triggers taker completion
    merge_enabled: bool = True
    merge_min_pairs: float = 50.0  # batch merges (on-chain in live trading)


@dataclass(frozen=True)
class DirectionalConfig:
    enabled: bool = True
    min_edge: float = 0.03  # model prob minus price paid needed to add/hold directional shares
    max_directional_shares: float = 200.0  # max deliberate |UP - DOWN| per market
    kelly_fraction: float = 0.25  # fraction of full Kelly used to size the directional target
    hold_margin: float = 0.02  # at FLATTEN: hold unpaired shares iff p_model - exit_price >= this


@dataclass(frozen=True)
class TimingConfig:
    warmup_seconds: float = 15.0  # after window start before quoting
    wind_down_seconds: float = 90.0  # before end: stop adding to the heavy side
    flatten_seconds: float = 25.0  # before end: cancel quotes, merge, resolve remainder
    min_requote_interval_seconds: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    vol_halflife_seconds: float = 300.0  # EWMA half-life for realised vol
    vol_floor: float = 2e-5  # floor on per-sqrt(second) log-return vol
    momentum_lookback_seconds: float = 30.0
    accel_lookback_seconds: float = 10.0
    w_momentum: float = 0.15  # weights on the normalised features added to z (uncalibrated)
    w_accel: float = 0.05
    w_book_imbalance: float = 0.10
    shrink_to_market: float = 0.5  # 0 = pure model, 1 = pure market mid
    p_floor: float = 0.02
    min_tau_seconds: float = 1.0


@dataclass(frozen=True)
class RiskConfig:
    max_capital_per_market_usd: float = 1500.0
    max_total_capital_usd: float = 5000.0
    max_daily_loss_usd: float = 1000.0  # kill switch (realised + mark-to-model)
    max_spot_staleness_seconds: float = 5.0
    max_orders_per_second: float = 20.0
    max_spread_ticks_to_quote: int = 6  # do not quote dislocated/wide books


@dataclass(frozen=True)
class PaperExchangeConfig:
    initial_cash: float = 10_000.0
    latency_ticks: int = 1  # snapshots before a new order can fill
    queue_ahead_fraction: float = 1.0  # share of displayed size at our price assumed ahead of us
    trade_fill_fraction: float = 1.0  # share of a print assumed available to us after the queue
    taker_slippage_ticks: int = 0  # extra adverse ticks charged on IOC fills
    enforce_min_order_size: bool = True


def _default_spots() -> dict[str, float]:
    return {"BTC": 60_000.0, "ETH": 3_000.0}


def _default_vols() -> dict[str, float]:
    return {"BTC": 0.55, "ETH": 0.75}


@dataclass(frozen=True)
class SimConfig:
    """Synthetic market generator. Results on it validate MECHANICS, never profitability."""

    seed: int = 1
    assets: tuple[str, ...] = ("BTC", "ETH")
    window_seconds: int = 900
    n_windows: int = 8
    tick_seconds: float = 1.0
    spot0: dict[str, float] = field(default_factory=_default_spots)
    annual_vol: dict[str, float] = field(default_factory=_default_vols)
    market_vol_multiplier: float = 1.0  # market's implied vol / true vol
    market_lag_seconds: float = 2.0  # market prices off spot this many seconds stale
    pricing_noise_ticks: float = 0.7  # AR(1) noise on the market's UP mid, in ticks
    tick_size: float = 0.01
    min_order_size: float = 5.0
    mean_spread_ticks: float = 1.5
    depth_mean_shares: float = 400.0
    n_levels: int = 5
    uninformed_trades_per_sec: float = 0.3  # random touch trades per sec per token side
    informed_flow: float = 1.0  # 0 = no sweeps on price moves, i.e. no adverse selection


@dataclass(frozen=True)
class BotConfig:
    fees: FeeConfig = field(default_factory=FeeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    pair: PairConfig = field(default_factory=PairConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exchange: PaperExchangeConfig = field(default_factory=PaperExchangeConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def validate(self) -> None:
        """Raise ValueError on nonsensical settings."""
        p, s, t, m, f, r, d, e, sim = (
            self.pair, self.sizing, self.timing, self.model, self.fees, self.risk,
            self.directional, self.exchange, self.sim,
        )  # fmt: skip

        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        need(0.0 <= p.target_margin < 0.5, "pair.target_margin must be in [0, 0.5)")
        need(p.taker_lock_margin >= 0.0, "pair.taker_lock_margin must be >= 0")
        need(p.rebalance_max_loss_per_pair >= 0.0, "pair.rebalance_max_loss_per_pair must be >= 0")
        need(p.max_net_imbalance_shares >= 0, "pair.max_net_imbalance_shares must be >= 0")
        need(
            p.max_inventory_per_side_shares >= p.max_net_imbalance_shares,
            "pair.max_inventory_per_side_shares must be >= max_net_imbalance_shares",
        )
        need(p.merge_min_pairs > 0, "pair.merge_min_pairs must be > 0")
        need(s.clip_shares > 0, "sizing.clip_shares must be > 0")
        need(
            0 < s.min_clip_shares <= s.max_clip_shares,
            "sizing.min_clip_shares must be in (0, max_clip_shares]",
        )
        need(s.ladder_levels >= 1, "sizing.ladder_levels must be >= 1")
        need(s.ladder_spacing_ticks >= 1, "sizing.ladder_spacing_ticks must be >= 1")
        need(0 < s.ladder_size_decay <= 1, "sizing.ladder_size_decay must be in (0, 1]")
        need(
            s.clip_equity_fraction is None or 0 < s.clip_equity_fraction <= 1,
            "sizing.clip_equity_fraction must be None or in (0, 1]",
        )
        need(0 <= d.min_edge < 0.5, "directional.min_edge must be in [0, 0.5)")
        need(d.max_directional_shares >= 0, "directional.max_directional_shares must be >= 0")
        need(0 <= d.kelly_fraction <= 1, "directional.kelly_fraction must be in [0, 1]")
        need(
            0 <= t.flatten_seconds <= t.wind_down_seconds,
            "timing: need 0 <= flatten_seconds <= wind_down_seconds",
        )
        need(t.warmup_seconds >= 0, "timing.warmup_seconds must be >= 0")
        need(0.0 <= m.shrink_to_market <= 1.0, "model.shrink_to_market must be in [0, 1]")
        need(0.0 < m.p_floor < 0.5, "model.p_floor must be in (0, 0.5)")
        need(m.vol_halflife_seconds > 0 and m.vol_floor > 0, "model vol params must be > 0")
        need(m.min_tau_seconds > 0, "model.min_tau_seconds must be > 0")
        need(f.maker_fee_rate >= 0 and f.maker_rebate_rate >= 0, "fee rates must be >= 0")
        need(f.taker_fee_rate >= 0 and f.taker_fee_exponent >= 0, "taker fee params must be >= 0")
        need(r.max_capital_per_market_usd > 0 and r.max_total_capital_usd > 0, "risk caps > 0")
        need(r.max_daily_loss_usd > 0, "risk.max_daily_loss_usd must be > 0")
        need(r.max_orders_per_second > 0, "risk.max_orders_per_second must be > 0")
        need(e.initial_cash > 0, "exchange.initial_cash must be > 0")
        need(e.latency_ticks >= 0, "exchange.latency_ticks must be >= 0")
        need(e.queue_ahead_fraction >= 0 and 0 <= e.trade_fill_fraction <= 1, "bad queue params")
        need(sim.window_seconds > 0 and sim.tick_seconds > 0, "sim timing must be > 0")
        need(sim.n_windows >= 1, "sim.n_windows must be >= 1")
        need(0 < sim.tick_size < 0.1, "sim.tick_size must be in (0, 0.1)")
        for a in sim.assets:
            need(a in sim.spot0 and a in sim.annual_vol, f"sim: missing spot0/annual_vol for {a}")


# --------------------------------------------------------------------------- (de)serialisation


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a table, got {type(value).__name__}")
        return _build(tp, value, path)
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            return _coerce(arg, value, path)
    if origin is tuple:
        return tuple(value)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path or 'config'}: unknown key(s) {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{path}.{k}" if path else k) for k, v in data.items()}
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> BotConfig:
    """Build and validate a BotConfig from a nested dict (unknown keys are errors)."""
    cfg: BotConfig = _build(BotConfig, data, "")
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None) -> BotConfig:
    """Load a TOML config file (or defaults when ``path`` is None) and validate it."""
    if path is None:
        cfg = BotConfig()
        cfg.validate()
        return cfg
    with open(path, "rb") as fh:
        return config_from_dict(tomllib.load(fh))


def config_to_dict(cfg: BotConfig) -> dict[str, Any]:
    """Plain nested dict (JSON/TOML friendly; tuples become lists)."""

    def conv(x: Any) -> Any:
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return {f.name: conv(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, tuple | list):
            return [conv(v) for v in x]
        return x

    result: dict[str, Any] = conv(cfg)
    return result
