"""Synthetic Up/Down market feed (DESIGN section 6).

This feed exists to exercise the *mechanics* of the strategy, the paper exchange and the
accounting. It contains a deliberate market lag, so a model that reads spot directly has
something to find: results on it are circular by construction and say nothing about
profitability in real markets.

Units: timestamps are unix seconds, prices are USDC per share on the tick grid, sizes are
shares, spot prices are positive underlying prices, volatilities are per sqrt(year) unless
stated otherwise.

Time grid
---------
``ts0`` is ``1_700_000_000`` rounded to the *nearest* multiple of ``window_seconds`` (half up).
Window ``w`` of every asset spans ``[ts0 + w * W, ts0 + (w + 1) * W)`` and the market id is
``"<ASSET>-<start_ts as int>-<W>s"``. A snapshot is emitted for each ``ts = start + k * tick``,
``k = 0 .. W/tick - 1`` (the first one is AT ``start``); the ``MarketResolved`` has
``ts == end`` and comes after that market's last snapshot. Ties on ``ts`` are ordered: every
``MarketResolved`` first (sorted by ``market_id``), then snapshots (sorted by ``market_id``),
so a window's resolutions always precede the next window's first snapshots. ``spot``,
``spot_ts`` (== ``ts``) and ``ref_price`` are set on every snapshot.

Spot
----
Correlated GBM with zero drift and an exact lognormal step (never negative):
``S' = S * exp(-sigma**2 / 2 + sigma * z)``, ``sigma = annual_vol * sqrt(tick / 31_557_600)``,
``z = sqrt(0.8) * f + sqrt(0.2) * e_asset`` with ``f`` a normal shared by all assets and ``e``
independent, so every pair of assets has correlation 0.8. The path is continuous across
windows: ``ref_price`` of window ``w + 1`` equals the spot at the end of window ``w``. The
winner is UP iff ``spot_end >= ref`` (ties resolve UP).

Market UP mid
-------------
``p = Phi(ln(spot_lagged / ref) / (sigma_mkt * sqrt(tau)))`` with ``sigma_mkt = annual_vol /
sqrt(31_557_600) * market_vol_multiplier`` (per sqrt(second)) and ``tau = max(end - ts,
tick_seconds)``. ``spot_lagged`` is the latest spot sample at or before ``ts -
market_lag_seconds``, but never earlier than the window start (so at the window start, and for
the first ``lag`` seconds, the market sees ``ref``). AR(1) noise (coefficient 0.9, stationary
std ``pricing_noise_ticks * tick_size``) is added to ``p``; the sum is snapped to the tick grid
and clipped into ``[tick, 1 - tick]``. Note that because ``sigma_mkt`` scales with the true
vol, ``annual_vol`` cancels out of ``p``: only ``market_vol_multiplier`` changes how fast the
market price moves.

Book
----
``spread_ticks = 1 + Geometric`` (non-negative integer, mean ``mean_spread_ticks - 1``).
``up_bid = mid - spread/2`` snapped DOWN to the grid, ``up_ask = up_bid + spread_ticks``; both
are shifted inward when needed so that ``0 < bid < ask < 1`` (this keeps the spread, so an odd
spread puts the book's centre half a tick below ``mid``). Each side has up to ``n_levels`` levels
one tick apart (levels at price <= 0 or >= 1 are dropped) with lognormal sizes of mean
``depth_mean_shares``. The DOWN book is the exact mirror: ``down_bid = 1 - up_ask``,
``down_ask = 1 - up_bid``, UP ask level at ``a`` <-> DOWN bid level at ``1 - a`` with equal size.
Because the mirror must stay on the grid, ``1 / tick_size`` must be an integer.

Trades (every print is emitted together with its mirror, same size, UP print first)
------------------------------------------------------------------------------------
* Informed sweeps: if this tick's UP best bid is ``k > 0`` ticks below the previous tick's, a
  SELL print on UP is emitted at each OLD bid level from the old best bid down to
  ``new best bid + 1 tick`` (only levels that were displayed, so at most ``n_levels``), sized
  ``informed_flow * old displayed size * U(0.3, 1.0)``; symmetrically BUY prints at old ask
  levels when the best ask rises. ``informed_flow`` must be in ``[0, 1]`` (a print never exceeds
  the displayed size) and 0 disables sweeps. Sweeps fire on ANY move of the touch, including
  moves caused by the random spread or the pricing noise, exactly as specified.
* Uninformed flow: ``Poisson(uninformed_trades_per_sec * tick_seconds)`` arrivals per tick; each
  picks a token and a side uniformly and prints at the touch of the CURRENT book, size
  lognormal with mean 30 shares and a floor of 1. Because both mirrored prints are emitted,
  each of the four (token, side) streams gets ``rate / 2`` prints per second.
* Prints of a snapshot are those since the previous snapshot of the market and carry
  ``ts == snapshot.ts``; sweeps come before uninformed prints; the first snapshot has none.

Randomness and determinism
--------------------------
Everything is drawn from private ``random.Random`` streams seeded with strings derived from
``cfg.sim.seed`` (spot factor, per-asset spot noise, and per market: price noise, spread,
depth, sweeps, uninformed flow), so changing one knob, the number of windows or the asset list
leaves unrelated randomness untouched where possible. Each ``iter(feed)`` replays from scratch
and yields identical events; there is no global RNG state and no wall clock.

Requirements checked at construction (``ValueError``): ``window_seconds`` is a positive
integer multiple of ``tick_seconds``; ``1 / tick_size`` is an integer in ``[3, 10_000]``; every
asset has positive finite ``spot0`` / ``annual_vol``; ``informed_flow`` in ``[0, 1]``; and the
other ``SimConfig`` fields are in their obvious ranges.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from abc_trading.config import BotConfig, SimConfig
from abc_trading.types import (
    BookSnapshot,
    FeedEvent,
    Level,
    MarketResolved,
    MarketSnapshot,
    MarketSpec,
    Outcome,
    Side,
    Trade,
)

SECONDS_PER_YEAR = 31_557_600.0  # 365.25 days
EPOCH_TS = 1_700_000_000.0
SPOT_CORRELATION = 0.8  # between any two assets, via a shared normal factor
NOISE_AR_COEF = 0.9
DEPTH_LOG_SIGMA = 0.5  # log-sd of displayed level sizes (mean = depth_mean_shares)
UNINFORMED_SIZE_MEAN = 30.0  # shares
UNINFORMED_SIZE_LOG_SIGMA = 0.6
UNINFORMED_SIZE_MIN = 1.0
SWEEP_FRACTION_LOW = 0.3  # a sweep consumes U(low, high) of the old displayed size
SWEEP_FRACTION_HIGH = 1.0
MAX_PRICE_TICKS = 10_000
_GRID_TOL = 1e-9


# --------------------------------------------------------------------------- parameters


@dataclass(frozen=True, slots=True)
class _Params:
    """Validated, immutable copy of the ``SimConfig`` fields the feed uses (plus derived)."""

    seed: int
    assets: tuple[str, ...]
    window_seconds: int
    n_windows: int
    tick_seconds: float
    ticks_per_window: int  # window_seconds / tick_seconds
    tick_size: float
    n_ticks: int  # 1 / tick_size: prices live on the integer grid 0 .. n_ticks
    prices: tuple[float, ...]  # prices[i] = price of i ticks, rounded to 6 decimals
    min_order_size: float
    ts0: float
    spot0: dict[str, float]
    tick_vol: dict[str, float]  # per-tick log-return sd of the true spot
    market_sigma: dict[str, float]  # market's implied vol per sqrt(second)
    lag_ticks: int
    noise_std: float  # stationary std of the UP-mid AR(1) noise, in price units
    mean_spread_extra: float  # mean_spread_ticks - 1
    depth_mean: float
    n_levels: int
    uninformed_per_tick: float
    informed_flow: float

    def px(self, ticks: int) -> float:
        """Price of an integer tick count in ``[0, n_ticks]`` (a table lookup)."""
        return self.prices[ticks]


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"sim config: {msg}")


def _finite_positive(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _derive_params(sim: SimConfig) -> _Params:
    window = sim.window_seconds
    _need(float(window).is_integer() and window > 0, "window_seconds must be a positive integer")
    _need(_finite_positive(sim.tick_seconds), "tick_seconds must be > 0")
    ticks_per_window = round(window / sim.tick_seconds)
    _need(
        ticks_per_window >= 1
        and abs(ticks_per_window * sim.tick_seconds - window) <= _GRID_TOL * window,
        "window_seconds must be a multiple of tick_seconds",
    )
    _need(sim.n_windows >= 1, "n_windows must be >= 1")
    _need(_finite_positive(sim.tick_size), "tick_size must be > 0")
    n_ticks = round(1.0 / sim.tick_size)
    _need(
        3 <= n_ticks <= MAX_PRICE_TICKS and abs(n_ticks * sim.tick_size - 1.0) <= _GRID_TOL,
        "1 / tick_size must be an integer in [3, 10000] so the mirrored DOWN book stays on-grid",
    )
    _need(_finite_positive(sim.min_order_size), "min_order_size must be > 0")
    _need(len(sim.assets) >= 1, "assets must not be empty")
    _need(len(set(sim.assets)) == len(sim.assets), "assets must be unique")
    for asset in sim.assets:
        _need(
            asset in sim.spot0 and asset in sim.annual_vol,
            f"missing spot0/annual_vol for {asset}",
        )
        _need(_finite_positive(sim.spot0[asset]), f"spot0[{asset}] must be > 0")
        _need(_finite_positive(sim.annual_vol[asset]), f"annual_vol[{asset}] must be > 0")
    _need(_finite_positive(sim.market_vol_multiplier), "market_vol_multiplier must be > 0")
    _need(math.isfinite(sim.market_lag_seconds) and sim.market_lag_seconds >= 0, "lag must be >= 0")
    _need(
        math.isfinite(sim.pricing_noise_ticks) and sim.pricing_noise_ticks >= 0,
        "pricing_noise_ticks must be >= 0",
    )
    _need(
        math.isfinite(sim.mean_spread_ticks) and sim.mean_spread_ticks >= 1.0,
        "mean_spread_ticks must be >= 1",
    )
    _need(_finite_positive(sim.depth_mean_shares), "depth_mean_shares must be > 0")
    _need(sim.n_levels >= 1, "n_levels must be >= 1")
    _need(
        math.isfinite(sim.uninformed_trades_per_sec) and sim.uninformed_trades_per_sec >= 0,
        "uninformed_trades_per_sec must be >= 0",
    )
    _need(0.0 <= sim.informed_flow <= 1.0, "informed_flow must be in [0, 1]")

    year_root = math.sqrt(SECONDS_PER_YEAR)
    base = int(EPOCH_TS)
    ts0 = float(((base + window // 2) // window) * window)  # nearest multiple, half up
    return _Params(
        seed=sim.seed,
        assets=tuple(sim.assets),
        window_seconds=int(window),
        n_windows=sim.n_windows,
        tick_seconds=sim.tick_seconds,
        ticks_per_window=ticks_per_window,
        tick_size=sim.tick_size,
        n_ticks=n_ticks,
        prices=tuple(round(i * sim.tick_size, 6) for i in range(n_ticks + 1)),
        min_order_size=sim.min_order_size,
        ts0=ts0,
        spot0={a: sim.spot0[a] for a in sim.assets},
        tick_vol={
            a: sim.annual_vol[a] * math.sqrt(sim.tick_seconds / SECONDS_PER_YEAR)
            for a in sim.assets
        },
        market_sigma={
            a: sim.annual_vol[a] / year_root * sim.market_vol_multiplier for a in sim.assets
        },
        lag_ticks=math.ceil(sim.market_lag_seconds / sim.tick_seconds - _GRID_TOL),
        noise_std=sim.pricing_noise_ticks * sim.tick_size,
        mean_spread_extra=sim.mean_spread_ticks - 1.0,
        depth_mean=sim.depth_mean_shares,
        n_levels=sim.n_levels,
        uninformed_per_tick=sim.uninformed_trades_per_sec * sim.tick_seconds,
        informed_flow=sim.informed_flow,
    )


# --------------------------------------------------------------------------- pure helpers


def _by_market_id(spec: MarketSpec) -> str:
    return spec.market_id


def _market_id(asset: str, start_ts: float, window_seconds: int) -> str:
    return f"{asset}-{int(start_ts)}-{window_seconds}s"


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _up_probability(spot_lagged: float, ref: float, sigma: float, tau: float) -> float:
    """Driftless-GBM P(spot_end >= ref): ``Phi(ln(spot_lagged / ref) / (sigma * sqrt(tau)))``.

    ``sigma`` is the vol per sqrt(second), ``tau`` the seconds left (> 0); both prices > 0.
    """
    return _norm_cdf(math.log(spot_lagged / ref) / (sigma * math.sqrt(tau)))


def _lognormal(rng: random.Random, mean: float, log_sigma: float) -> float:
    """Lognormal draw whose arithmetic mean is ``mean``."""
    return mean * math.exp(log_sigma * rng.gauss(0.0, 1.0) - 0.5 * log_sigma * log_sigma)


def _geometric(rng: random.Random, mean: float) -> int:
    """Non-negative integer with ``P(X >= k) = q**k``, ``q = mean / (1 + mean)``; E[X] = mean.

    Always consumes exactly one uniform so the stream stays aligned when ``mean`` changes.
    """
    u = rng.random()
    if mean <= 0.0:
        return 0
    q = mean / (1.0 + mean)
    return int(math.log1p(-u) / math.log(q))  # 1 - u is in (0, 1], so the log is finite


def _poisson(rng: random.Random, expected: float) -> int:
    """Poisson count: unit-rate exponential arrivals inside ``[0, expected)`` (no underflow)."""
    if expected <= 0.0:
        return 0
    count = 0
    t = rng.expovariate(1.0)
    while t < expected:
        count += 1
        t += rng.expovariate(1.0)
    return count


def _quote_ticks(mid: int, spread_ticks: int, n_ticks: int) -> tuple[int, int]:
    """UP (bid, ask) in integer ticks for a ``mid`` and spread, with ``1 <= bid < ask <= n-1``.

    ``bid = mid - spread/2`` snapped DOWN (i.e. ``mid - ceil(spread / 2)``), ``ask = bid +
    spread``; the pair is shifted inward when it leaves the grid, and ``spread`` is capped at
    ``n_ticks - 2`` (the widest book that fits).
    """
    if not 1 <= mid <= n_ticks - 1:
        raise ValueError(f"mid {mid} outside [1, {n_ticks - 1}] ticks")
    if spread_ticks < 1:
        raise ValueError(f"spread_ticks must be >= 1, got {spread_ticks}")
    spread = min(spread_ticks, n_ticks - 2)
    bid = mid - (spread + 1) // 2
    ask = bid + spread
    if bid < 1:
        ask += 1 - bid
        bid = 1
    elif ask > n_ticks - 1:
        bid -= ask - (n_ticks - 1)
        ask = n_ticks - 1
    return bid, ask


@dataclass(frozen=True, slots=True)
class _UpBook:
    """UP book in integer ticks; ``(price_ticks, size)`` pairs, best first on each side."""

    bids: tuple[tuple[int, float], ...]
    asks: tuple[tuple[int, float], ...]


def _swept_levels(
    old_levels: Sequence[tuple[int, float]], new_best: int, side: Side
) -> list[tuple[int, float]]:
    """Old displayed levels consumed by a move of the touch to ``new_best``.

    ``side`` is the aggressor: SELL sweeps old bid levels strictly above ``new_best`` (the
    old best bid down to ``new_best + 1``); BUY sweeps old ask levels strictly below it.
    Levels deeper than what was displayed cannot be swept, so the result has at most
    ``len(old_levels)`` entries. Best level first.
    """
    if side is Side.SELL:
        return [(n, size) for n, size in old_levels if n > new_best]
    return [(n, size) for n, size in old_levels if n < new_best]


def _to_books(up: _UpBook, p: _Params) -> tuple[BookSnapshot, BookSnapshot]:
    """UP book and its exact mirror (DOWN bid at ``1 - ask``, DOWN ask at ``1 - bid``)."""
    n = p.n_ticks
    up_book = BookSnapshot(
        Outcome.UP,
        bids=tuple(Level(p.px(t), s) for t, s in up.bids),
        asks=tuple(Level(p.px(t), s) for t, s in up.asks),
    )
    down_book = BookSnapshot(
        Outcome.DOWN,
        bids=tuple(Level(p.px(n - t), s) for t, s in up.asks),
        asks=tuple(Level(p.px(n - t), s) for t, s in up.bids),
    )
    return up_book, down_book


def _print_pair(ts: float, up_side: Side, up_ticks: int, size: float, p: _Params) -> list[Trade]:
    """UP print and its mirror: ``up_side`` at ``x`` on UP == opposite side at ``1 - x`` on DOWN."""
    return [
        Trade(ts, Outcome.UP, p.px(up_ticks), size, up_side),
        Trade(ts, Outcome.DOWN, p.px(p.n_ticks - up_ticks), size, up_side.opposite),
    ]


# --------------------------------------------------------------------------- spot path


def _winner_of(path: Sequence[float]) -> Outcome:
    """UP iff the last sample is at or above the first (``spot_end >= ref``)."""
    return Outcome.UP if path[-1] >= path[0] else Outcome.DOWN


def _spot_windows(p: _Params) -> Iterator[dict[str, list[float]]]:
    """Yield, per window, each asset's ``ticks_per_window + 1`` spot samples.

    Sample ``k`` is the spot at ``start + k * tick`` and the last sample is the spot at the
    window end, which is also the first sample of the next window. The path depends only on
    the seed, the assets' own parameters and the tick, never on ``n_windows``.
    """
    factor_rng = random.Random(f"{p.seed}:spot:factor")
    idio_rngs = {a: random.Random(f"{p.seed}:spot:idio:{a}") for a in p.assets}
    shared, own = math.sqrt(SPOT_CORRELATION), math.sqrt(1.0 - SPOT_CORRELATION)
    level = dict(p.spot0)
    for _ in range(p.n_windows):
        window = {a: [level[a]] for a in p.assets}
        for _ in range(p.ticks_per_window):
            factor = factor_rng.gauss(0.0, 1.0)
            for a in p.assets:
                sigma = p.tick_vol[a]
                z = shared * factor + own * idio_rngs[a].gauss(0.0, 1.0)
                try:
                    level[a] *= math.exp(-0.5 * sigma * sigma + sigma * z)
                except OverflowError:
                    level[a] = math.inf
                if not (math.isfinite(level[a]) and level[a] > 0.0):
                    raise ValueError(f"sim config: spot path of {a} left (0, inf); vol too high")
                window[a].append(level[a])
        yield window


# --------------------------------------------------------------------------- one market


class _MarketSim:
    """Generates the snapshots of one market, in order ``k = 0, 1, ...``; holds all its state."""

    def __init__(self, spec: MarketSpec, spots: Sequence[float], p: _Params) -> None:
        self._spec = spec
        self._spots = spots
        self._p = p
        self._ref = spots[0]
        self._sigma = p.market_sigma[spec.asset]
        mid = spec.market_id
        self._noise_rng = random.Random(f"{p.seed}:noise:{mid}")
        self._spread_rng = random.Random(f"{p.seed}:spread:{mid}")
        self._depth_rng = random.Random(f"{p.seed}:depth:{mid}")
        self._sweep_rng = random.Random(f"{p.seed}:sweep:{mid}")
        self._flow_rng = random.Random(f"{p.seed}:flow:{mid}")
        self._noise: float | None = None
        self._prev: _UpBook | None = None

    def snapshot(self, k: int) -> MarketSnapshot:
        """Snapshot at ``ts = start + k * tick`` (call with k = 0, 1, 2, ... in order)."""
        p = self._p
        ts = self._spec.start_ts + k * p.tick_seconds
        up = self._draw_book(k, ts)
        trades = () if self._prev is None else tuple(self._draw_prints(ts, self._prev, up))
        self._prev = up
        up_book, down_book = _to_books(up, p)
        return MarketSnapshot(
            ts=ts,
            market=self._spec,
            up_book=up_book,
            down_book=down_book,
            spot=self._spots[k],
            spot_ts=ts,
            ref_price=self._ref,
            trades=trades,
        )

    def _next_noise(self) -> float:
        p = self._p
        draw = p.noise_std * self._noise_rng.gauss(0.0, 1.0)
        if self._noise is None:
            self._noise = draw  # start in the stationary distribution
        else:
            innovation = math.sqrt(1.0 - NOISE_AR_COEF * NOISE_AR_COEF)
            self._noise = NOISE_AR_COEF * self._noise + innovation * draw
        return self._noise

    def _mid_ticks(self, k: int, ts: float) -> int:
        p = self._p
        lagged = self._spots[max(0, k - p.lag_ticks)]
        tau = max(self._spec.end_ts - ts, p.tick_seconds)
        price = _up_probability(lagged, self._ref, self._sigma, tau) + self._next_noise()
        mid = math.floor(price / p.tick_size + 0.5)
        return min(max(mid, 1), p.n_ticks - 1)

    def _draw_book(self, k: int, ts: float) -> _UpBook:
        p = self._p
        mid = self._mid_ticks(k, ts)
        spread = 1 + _geometric(self._spread_rng, p.mean_spread_extra)
        bid, ask = _quote_ticks(mid, spread, p.n_ticks)
        bid_sizes = [
            _lognormal(self._depth_rng, p.depth_mean, DEPTH_LOG_SIGMA) for _ in range(p.n_levels)
        ]
        ask_sizes = [
            _lognormal(self._depth_rng, p.depth_mean, DEPTH_LOG_SIGMA) for _ in range(p.n_levels)
        ]
        return _UpBook(
            bids=tuple((bid - i, bid_sizes[i]) for i in range(p.n_levels) if bid - i >= 1),
            asks=tuple(
                (ask + i, ask_sizes[i]) for i in range(p.n_levels) if ask + i <= p.n_ticks - 1
            ),
        )

    def _draw_prints(self, ts: float, old: _UpBook, new: _UpBook) -> list[Trade]:
        return self._informed_prints(ts, old, new) + self._uninformed_prints(ts, new)

    def _informed_prints(self, ts: float, old: _UpBook, new: _UpBook) -> list[Trade]:
        p = self._p
        if p.informed_flow <= 0.0:
            return []
        prints: list[Trade] = []
        sweeps = (
            (Side.SELL, _swept_levels(old.bids, new.bids[0][0], Side.SELL)),
            (Side.BUY, _swept_levels(old.asks, new.asks[0][0], Side.BUY)),
        )
        for side, levels in sweeps:
            for ticks, shown in levels:
                fraction = self._sweep_rng.uniform(SWEEP_FRACTION_LOW, SWEEP_FRACTION_HIGH)
                prints += _print_pair(ts, side, ticks, p.informed_flow * shown * fraction, p)
        return prints

    def _uninformed_prints(self, ts: float, book: _UpBook) -> list[Trade]:
        p = self._p
        rng = self._flow_rng
        prints: list[Trade] = []
        for _ in range(_poisson(rng, p.uninformed_per_tick)):
            token = Outcome.UP if rng.random() < 0.5 else Outcome.DOWN
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            size = max(
                UNINFORMED_SIZE_MIN,
                _lognormal(rng, UNINFORMED_SIZE_MEAN, UNINFORMED_SIZE_LOG_SIGMA),
            )
            up_side = side if token is Outcome.UP else side.opposite
            up_ticks = book.asks[0][0] if up_side is Side.BUY else book.bids[0][0]
            prints += _print_pair(ts, up_side, up_ticks, size, p)
        return prints


# --------------------------------------------------------------------------- public feed


class SyntheticFeed:
    """Deterministic, re-iterable synthetic feed of ``FeedEvent`` (see the module docstring).

    ``markets`` lists every window of every asset (window-major, then ``cfg.sim.assets``
    order). ``winner(market_id)`` is available from construction on: the spot path is
    generated up front (O(n_windows) memory), which is stronger than the guarantee that it is
    valid once the window has been generated, and it always equals the ``MarketResolved``
    winner that iteration emits for that market.
    """

    def __init__(self, cfg: BotConfig) -> None:
        p = _derive_params(cfg.sim)
        self._p = p
        specs: list[MarketSpec] = []
        winners: dict[str, Outcome] = {}
        for w, window in enumerate(_spot_windows(p)):
            start = p.ts0 + w * p.window_seconds
            for asset in p.assets:
                spec = MarketSpec(
                    market_id=_market_id(asset, start, p.window_seconds),
                    asset=asset,
                    start_ts=start,
                    end_ts=start + p.window_seconds,
                    tick_size=p.tick_size,
                    min_order_size=p.min_order_size,
                )
                specs.append(spec)
                winners[spec.market_id] = _winner_of(window[asset])
        self._specs = tuple(specs)
        self._winners = winners

    @property
    def markets(self) -> list[MarketSpec]:
        """Every market (all windows, all assets); a fresh list, safe to mutate."""
        return list(self._specs)

    def winner(self, market_id: str) -> Outcome:
        """Winner of ``market_id`` (UP iff ``spot_end >= ref``); ``ValueError`` if unknown."""
        try:
            return self._winners[market_id]
        except KeyError:
            raise ValueError(f"unknown market_id {market_id!r}") from None

    def __iter__(self) -> Iterator[FeedEvent]:
        """A fresh pass over the feed; every pass yields identical events."""
        p = self._p
        n_assets = len(p.assets)
        for w, window in enumerate(_spot_windows(p)):
            specs = sorted(self._specs[w * n_assets : (w + 1) * n_assets], key=_by_market_id)
            sims = [_MarketSim(spec, window[spec.asset], p) for spec in specs]
            for k in range(p.ticks_per_window):
                for sim in sims:
                    yield sim.snapshot(k)
            for spec in specs:
                yield MarketResolved(spec.end_ts, spec.market_id, _winner_of(window[spec.asset]))
