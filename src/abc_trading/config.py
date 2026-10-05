"""Configuration for every module. FROZEN CONTRACT (see ``types.py``).

All defaults are *illustrative starting points*, not tuned or validated values. Nothing in
here has been calibrated against real Polymarket data. Fee parameters in particular MUST be
checked against Polymarket's current published fee schedule before any PnL number is trusted.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The synthetic feed (sim/feed.py) puts prices on a grid of ``1 / tick_size`` ticks and needs the
# window to be a whole number of ``tick_seconds``; ``BotConfig.validate`` enforces the same limits
# so a bad value is refused up front, by every command (tests pin the equality with the feed).
MAX_PRICE_TICKS = 10_000
GRID_TOL = 1e-9


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
    requote_tolerance_ticks: int = 0  # keep a resting bid at most this many ticks BELOW desired


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
    latency_ticks: int = 1  # snapshots before a new order can fill or a cancel takes effect
    queue_ahead_fraction: float = 1.0  # share of displayed size at our price assumed ahead of us
    trade_fill_fraction: float = 1.0  # share of a print assumed available to us after the queue
    taker_slippage_ticks: int = 0  # ticks an IOC fill moves against us, capped at its limit
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
    uninformed_trades_per_sec: float = 0.3  # random touch trades per sec per market (on both books)
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
        """Raise ``ValueError`` naming the offending key on a nonsensical setting.

        This is the one gate every configuration passes (``load_config``, ``config_from_dict``,
        the CLI overrides, and the engine and the runner, which call it again). It repeats the
        checks that ``SyntheticFeed``, ``FairValueModel``, ``PaperExchange``, ``RiskManager`` and
        ``FeeModel`` make when they are built, so a bad value is reported here, with its key
        (exit status 2 from the CLI), by every command, not mid-run or only by some of them. It
        is also stricter than those components where they would accept a setting that silently
        switches a control off or inverts it (a negative tolerance or skew, a spread or staleness
        limit that never quotes, a rebate above the notional, a window with no ACCUMULATE phase).
        Field *types* are checked when a config is built from a dict or TOML
        (``config_from_dict``); here every number must also be finite.
        """
        p, s, t, m, f, r, d, e, sim = (
            self.pair, self.sizing, self.timing, self.model, self.fees, self.risk,
            self.directional, self.exchange, self.sim,
        )  # fmt: skip

        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        _check_finite(self)

        # --- fees
        need(f.maker_fee_rate >= 0 and f.maker_rebate_rate >= 0, "fee rates must be >= 0")
        need(f.taker_fee_rate >= 0 and f.taker_fee_exponent >= 0, "taker fee params must be >= 0")
        # A fee above 100% of the notional could make a SELL reduce cash; a rebate above 100%
        # would make a bid cost less than nothing (the risk book then fails mid-run).
        need(f.maker_fee_rate <= 1.0, "fees.maker_fee_rate must be <= 1 (a fraction of notional)")
        need(
            f.maker_rebate_rate <= 1.0,
            "fees.maker_rebate_rate must be <= 1 (a fraction of notional: a larger rebate "
            "would make a bid cost less than nothing)",
        )
        need(f.taker_fee_rate <= 1.0, "fees.taker_fee_rate must be <= 1 (a fraction of notional)")

        # --- sizing
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
        need(
            s.requote_tolerance_ticks >= 0,
            "sizing.requote_tolerance_ticks must be >= 0 (a negative tolerance re-prices every "
            "resting order on every snapshot, so none ever fills)",
        )

        # --- pair
        need(0.0 <= p.target_margin < 0.5, "pair.target_margin must be in [0, 0.5)")
        need(p.taker_lock_margin >= 0.0, "pair.taker_lock_margin must be >= 0")
        need(p.rebalance_max_loss_per_pair >= 0.0, "pair.rebalance_max_loss_per_pair must be >= 0")
        need(p.max_net_imbalance_shares >= 0, "pair.max_net_imbalance_shares must be >= 0")
        need(
            p.max_inventory_per_side_shares >= p.max_net_imbalance_shares,
            "pair.max_inventory_per_side_shares must be >= max_net_imbalance_shares",
        )
        need(
            p.skew_ticks_per_100_shares >= 0,
            "pair.skew_ticks_per_100_shares must be >= 0 (a negative skew would bid higher on "
            "the heavy side)",
        )
        need(
            p.rebalance_trigger_shares >= 0,
            "pair.rebalance_trigger_shares must be >= 0 (0 completes any unpaired excess)",
        )
        need(p.merge_min_pairs > 0, "pair.merge_min_pairs must be > 0")

        # --- directional
        need(0 <= d.min_edge < 0.5, "directional.min_edge must be in [0, 0.5)")
        need(d.max_directional_shares >= 0, "directional.max_directional_shares must be >= 0")
        need(0 <= d.kelly_fraction <= 1, "directional.kelly_fraction must be in [0, 1]")

        # --- timing
        need(
            0 <= t.flatten_seconds <= t.wind_down_seconds,
            "timing: need 0 <= flatten_seconds <= wind_down_seconds",
        )
        need(t.warmup_seconds >= 0, "timing.warmup_seconds must be >= 0")
        need(
            t.min_requote_interval_seconds >= 0,
            "timing.min_requote_interval_seconds must be >= 0",
        )

        # --- model
        need(0.0 <= m.shrink_to_market <= 1.0, "model.shrink_to_market must be in [0, 1]")
        need(0.0 < m.p_floor < 0.5, "model.p_floor must be in (0, 0.5)")
        need(m.vol_halflife_seconds > 0 and m.vol_floor > 0, "model vol params must be > 0")
        need(m.min_tau_seconds > 0, "model.min_tau_seconds must be > 0")
        need(m.momentum_lookback_seconds > 0, "model.momentum_lookback_seconds must be > 0")
        need(m.accel_lookback_seconds > 0, "model.accel_lookback_seconds must be > 0")

        # --- risk
        need(r.max_capital_per_market_usd > 0 and r.max_total_capital_usd > 0, "risk caps > 0")
        need(r.max_daily_loss_usd > 0, "risk.max_daily_loss_usd must be > 0")
        need(
            r.max_orders_per_second >= 1.0,
            "risk.max_orders_per_second must be >= 1 (orders are counted per one-second window)",
        )
        need(
            r.max_spread_ticks_to_quote >= 1,
            "risk.max_spread_ticks_to_quote must be >= 1 (an uncrossed book is never narrower "
            "than one tick, so a smaller limit would never quote)",
        )
        need(
            r.max_spot_staleness_seconds >= 0,
            "risk.max_spot_staleness_seconds must be >= 0 (a negative limit would never quote)",
        )

        # --- paper exchange
        need(e.initial_cash > 0, "exchange.initial_cash must be > 0")
        need(e.latency_ticks >= 0, "exchange.latency_ticks must be >= 0")
        need(e.queue_ahead_fraction >= 0 and 0 <= e.trade_fill_fraction <= 1, "bad queue params")
        need(e.taker_slippage_ticks >= 0, "exchange.taker_slippage_ticks must be >= 0")

        # --- synthetic feed
        need(sim.window_seconds > 0 and sim.tick_seconds > 0, "sim timing must be > 0")
        need(_is_whole(sim.window_seconds), "sim.window_seconds must be a whole number of seconds")
        need(
            _grid_steps(sim.window_seconds, sim.tick_seconds) is not None,
            "sim.window_seconds must be a multiple of sim.tick_seconds",
        )
        need(sim.n_windows >= 1, "sim.n_windows must be >= 1")
        need(
            t.warmup_seconds + t.wind_down_seconds < sim.window_seconds,
            f"timing.warmup_seconds + timing.wind_down_seconds "
            f"({t.warmup_seconds + t.wind_down_seconds:g}) must be < sim.window_seconds "
            f"({sim.window_seconds:g}): otherwise the window has no ACCUMULATE phase",
        )
        need(0 < sim.tick_size < 0.1, "sim.tick_size must be in (0, 0.1)")
        ticks = _grid_steps(1.0, sim.tick_size)
        need(
            ticks is not None and ticks <= MAX_PRICE_TICKS,
            f"sim.tick_size must divide 1 into a whole number of ticks (at most "
            f"{MAX_PRICE_TICKS}) so the mirrored DOWN book stays on-grid, got {sim.tick_size!r}",
        )
        need(sim.min_order_size > 0, "sim.min_order_size must be > 0")
        need(len(sim.assets) >= 1, "sim.assets must not be empty")
        need(len(set(sim.assets)) == len(sim.assets), "sim.assets must be unique")
        for a in sim.assets:
            need(a in sim.spot0 and a in sim.annual_vol, f"sim: missing spot0/annual_vol for {a}")
            need(sim.spot0[a] > 0, f"sim.spot0[{a}] must be > 0")
            need(sim.annual_vol[a] > 0, f"sim.annual_vol[{a}] must be > 0")
        need(sim.market_vol_multiplier > 0, "sim.market_vol_multiplier must be > 0")
        need(sim.market_lag_seconds >= 0, "sim.market_lag_seconds must be >= 0")
        need(sim.pricing_noise_ticks >= 0, "sim.pricing_noise_ticks must be >= 0")
        need(sim.mean_spread_ticks >= 1, "sim.mean_spread_ticks must be >= 1")
        need(sim.depth_mean_shares > 0, "sim.depth_mean_shares must be > 0")
        need(sim.n_levels >= 1, "sim.n_levels must be >= 1")
        need(sim.uninformed_trades_per_sec >= 0, "sim.uninformed_trades_per_sec must be >= 0")
        need(0 <= sim.informed_flow <= 1, "sim.informed_flow must be in [0, 1]")

    def advisories(self) -> list[str]:
        """Notes about settings that are legal but probably not what was meant (not errors).

        ``validate`` cannot reject these: runs that deliberately use a small account (the
        cash-limited tests, for one) rely on the default loss limit staying out of the way. The CLI
        prints each note as a ``warning:`` line on stderr; the list is empty for the defaults.
        """
        notes: list[str] = []
        r, e = self.risk, self.exchange
        if r.max_daily_loss_usd >= e.initial_cash:
            notes.append(
                f"risk.max_daily_loss_usd ({r.max_daily_loss_usd:g}) is not below "
                f"exchange.initial_cash ({e.initial_cash:g}), so the kill switch can never fire "
                "(equity cannot lose more than the starting cash)"
            )
        return notes


def _check_finite(cfg: BotConfig) -> None:
    """ValueError for any NaN or infinite number (plain floats and the values of float tables).

    Most ranges in ``validate`` already reject NaN (a comparison with it is False) but not
    infinity, and some fields have no range at all; JSON/TOML round trips cannot carry them.
    """
    for section in dataclasses.fields(cfg):
        obj = getattr(cfg, section.name)
        for fld in dataclasses.fields(obj):
            value = getattr(obj, fld.name)
            entries = value.items() if isinstance(value, dict) else [(None, value)]
            for key, number in entries:
                if isinstance(number, float) and not math.isfinite(number):
                    label = f"{section.name}.{fld.name}" + ("" if key is None else f"[{key}]")
                    raise ValueError(f"{label} must be a finite number, got {number!r}")


def _is_whole(x: float) -> bool:
    return isinstance(x, int) or x.is_integer()


def _grid_steps(span: float, step: float) -> int | None:
    """``round(span / step)`` if ``span`` is a whole (>= 1) multiple of ``step``, else None.

    The same rule as ``sim/feed.py`` (relative tolerance ``GRID_TOL``). Never raises.
    """
    if not step > 0:
        return None
    try:
        n = round(span / step)
    except (OverflowError, ValueError):  # the quotient overflowed to inf, or is NaN
        return None
    return n if n >= 1 and abs(n * step - span) <= GRID_TOL * span else None


# --------------------------------------------------------------------------- (de)serialisation


def _describe(value: Any) -> str:
    """Type name of ``value`` plus, for a scalar, the value itself (``str 'false'``)."""
    name = type(value).__name__
    if not isinstance(value, bool | int | float | str):
        return name
    text = repr(value)
    return f"{name} {text if len(text) <= 40 else text[:37] + '...'}"


def _coerce(tp: Any, value: Any, path: str) -> Any:
    """Check ``value`` against the declared type ``tp`` and normalise it (or ValueError(path)).

    Strict on purpose: a TOML value must have the declared type. A quoted ``"false"`` is not a
    bool (it would be truthy), a float is not an int, a string is not a number or a list. The only
    conversions are the lossless ones: int -> float, list -> tuple (also element-wise), and the
    values of a ``dict[str, float]`` table.
    """
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a table, got {type(value).__name__}")
        return _build(tp, value, path)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"{path}: unsupported config field type {tp!r}")
        return _coerce(inner[0], value, path)
    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise TypeError(f"{path}: unsupported config field type {tp!r}")
        if not isinstance(value, list | tuple):
            raise ValueError(f"{path}: expected a list, got {_describe(value)}")
        return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        key_tp, value_tp = typing.get_args(tp)
        if key_tp is not str:
            raise TypeError(f"{path}: unsupported config field type {tp!r}")
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected a table, got {_describe(value)}")
        table: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: table keys must be strings, got {_describe(key)}")
            table[key] = _coerce(value_tp, item, f"{path}.{key}")
        return table
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected a boolean, got {_describe(value)}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected an integer, got {_describe(value)}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: expected a number, got {_describe(value)}")
        try:
            number = float(value)
        except OverflowError:  # an integer too large for a float
            number = math.inf
        if not math.isfinite(number):
            raise ValueError(f"{path}: expected a finite number, got {_describe(value)}")
        return number
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected a string, got {_describe(value)}")
        return value
    raise TypeError(f"{path}: unsupported config field type {tp!r}")


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
