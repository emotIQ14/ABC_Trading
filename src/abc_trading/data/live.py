"""``LiveFeed``: poll the public clients and yield ``FeedEvent`` s. PAPER MODE ONLY.

READ-ONLY: this module only reads public data through ``PolymarketPublicClient`` / ``SpotClient``
(whose endpoints are UNVERIFIED, see ``public_api``). It places no orders and holds no keys.

*** ASSUMPTIONS THAT THE FEED MAKES UP (none verified against the real venue) ***

1. **Window grid.** Windows are ``window_seconds`` long and start at unix times that are exact
   multiples of ``window_seconds``.
2. **Market identity.** The market of an asset's window is found by ``slug_template`` (default
   ``'{asset_lower}-updown-{minutes}m-{start_ts}'``, e.g. ``btc-updown-5m-1700000100``).
   Placeholders: ``asset``, ``asset_lower``, ``asset_upper``, ``minutes``, ``window_seconds``,
   ``start_ts`` and ``end_ts`` (integer unix seconds). The slug scheme is a GUESS.
3. **"Price to beat" (``ref_price``).** The first spot reading (Binance, else Coinbase) taken at
   or after the window start. This is an approximation: the real market has its own reference
   price source. A window whose first successful reading arrives more than
   ``max_ref_lag_seconds`` after its start (e.g. the feed was started mid-window) is SKIPPED
   (no events at all), because its reference would be wrong.
4. **Winner.** There is no resolution API call: the feed itself decides. At the first tick at or
   after the window end it reads spot; winner is UP iff ``spot >= ref_price`` (ties UP, as in
   ``docs/DESIGN.md``). The ``MarketResolved`` events are therefore this feed's INFERENCE, and
   paper results inherit its error. The same reading becomes the next window's reference.
5. **Trade prints** come from the best-effort trades endpoint and carry the server's
   timestamps, which may be skewed against the local clock.

Behaviour
---------
* Iterating polls once per ``poll_seconds`` (``clock`` / ``sleep`` are injected, so tests need no
  real time). Per tick and asset: read spot, resolve an ended window, open the new one, look the
  market up (once per window), fetch both books, fetch new trades, emit a ``MarketSnapshot``
  with ``ts`` = tick time, ``spot_ts`` = tick time and ``ref_price``.
* ``DataError`` from the market lookup, spot or books skips that asset for the tick and is
  counted in ``stats``; the window state survives, so the next tick retries. A pending
  resolution simply waits for the next successful spot reading. A trades failure does NOT skip
  the tick (trades are best effort; the snapshot is emitted without new prints and the missed
  ones are retried next tick). Anything that is not a ``DataError`` propagates.
* Events are emitted in non-decreasing ``ts``: resolutions (``ts`` = max(window end, last
  event ts)) before the tick's snapshots; a backwards-moving ``clock`` is clamped to the last
  event ts. ``MarketResolved`` is emitted exactly once per window and only for windows that
  produced at least one snapshot.
* ``max_events`` stops the iteration after that many events (``0`` yields nothing).

Caveats: ``assets`` are used verbatim as ``MarketSpec.asset`` (pass "BTC" / "ETH"); a window
that is still open when iteration stops is never resolved; the feed has no persistence, so a
restart waits for the next window start.

``record(events, path)`` tees any event stream into a JSONL file (see ``events``).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from abc_trading.config import BotConfig
from abc_trading.data.events import event_to_json
from abc_trading.data.public_api import DataError, PolymarketPublicClient, SpotClient
from abc_trading.types import (
    FeedEvent,
    MarketResolved,
    MarketSnapshot,
    MarketSpec,
    Outcome,
    Trade,
)

DEFAULT_SLUG_TEMPLATE = "{asset_lower}-updown-{minutes}m-{start_ts}"
MIN_REF_LAG_SECONDS = 5.0

STAT_KEYS = (
    "ticks",
    "snapshots",
    "resolutions",
    "windows_opened",
    "windows_skipped_late",
    "errors",  # total DataErrors, including trades
    "errors_market",
    "errors_spot",
    "errors_book",
    "errors_trades",
)


@dataclass(slots=True)
class _Window:
    """One asset's current window. ``ref_price`` None means 'skipped (late)'."""

    start_ts: int
    end_ts: int
    ref_price: float | None
    trade_cursor: float
    spec: MarketSpec | None = None
    snapshots: int = 0


class LiveFeed:
    """Iterator of ``MarketSnapshot`` / ``MarketResolved`` built by polling (paper mode only).

    ``cfg`` supplies the default tick size / minimum order size (``cfg.sim``) for markets whose
    lookup does not report them. ``stats`` (see ``STAT_KEYS``) counts ticks, events and errors;
    ``last_error`` holds the latest ``DataError`` message.
    """

    def __init__(
        self,
        cfg: BotConfig,
        client: PolymarketPublicClient,
        spot: SpotClient,
        *,
        assets: Sequence[str],
        window_seconds: int,
        slug_template: str = DEFAULT_SLUG_TEMPLATE,
        poll_seconds: float = 1.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        max_events: int | None = None,
        max_ref_lag_seconds: float | None = None,
    ) -> None:
        names = tuple(assets)
        if not names or any(not isinstance(a, str) or not a.strip() for a in names):
            raise ValueError("assets must be a non-empty sequence of non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"assets must be unique, got {names!r}")
        if isinstance(window_seconds, bool) or not isinstance(window_seconds, int):
            raise ValueError(f"window_seconds must be an int, got {window_seconds!r}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds!r}")
        if not (math.isfinite(poll_seconds) and poll_seconds > 0):
            raise ValueError(f"poll_seconds must be a finite number > 0, got {poll_seconds!r}")
        if max_events is not None and max_events < 0:
            raise ValueError(f"max_events must be None or >= 0, got {max_events!r}")
        if max_ref_lag_seconds is None:
            max_ref_lag_seconds = max(MIN_REF_LAG_SECONDS, 2.0 * poll_seconds)
        if not (math.isfinite(max_ref_lag_seconds) and max_ref_lag_seconds >= 0):
            raise ValueError(
                f"max_ref_lag_seconds must be a finite number >= 0, got {max_ref_lag_seconds!r}"
            )
        self._cfg = cfg
        self._client = client
        self._spot = spot
        self._assets = names
        self._window = window_seconds
        self._template = slug_template
        self._poll = poll_seconds
        self._clock = clock
        self._sleep = sleep
        self._max_events = max_events
        self._max_ref_lag = max_ref_lag_seconds
        self._windows: dict[str, _Window] = {}
        self._last_ts = -math.inf
        self._gen: Iterator[FeedEvent] | None = None
        self.stats: dict[str, int] = dict.fromkeys(STAT_KEYS, 0)
        self.last_error: str | None = None
        self._slug(names[0], 0)  # fail at construction on a bad template / window length

    # ------------------------------------------------------------------ iterator protocol

    def __iter__(self) -> LiveFeed:
        return self

    def __next__(self) -> FeedEvent:
        if self._gen is None:
            self._gen = self._run()
        return next(self._gen)

    # ------------------------------------------------------------------ main loop

    def _run(self) -> Iterator[FeedEvent]:
        emitted = 0
        if self._max_events is not None and self._max_events <= 0:
            return
        while True:
            raw_now = self._clock()
            if not math.isfinite(raw_now):
                raise ValueError(f"clock returned a non-finite time: {raw_now!r}")
            now = max(raw_now, self._last_ts)
            for event in self._poll_once(now):
                self._last_ts = max(self._last_ts, event.ts)
                self.stats["resolutions" if isinstance(event, MarketResolved) else "snapshots"] += 1
                yield event
                emitted += 1
                if self._max_events is not None and emitted >= self._max_events:
                    return
            delay = self._poll - (self._clock() - raw_now)
            if delay > 0:
                self._sleep(delay)

    def _poll_once(self, now: float) -> list[FeedEvent]:
        """One tick: all resolutions first, then all snapshots (non-decreasing ``ts``)."""
        self.stats["ticks"] += 1
        resolutions: list[FeedEvent] = []
        snapshots: list[FeedEvent] = []
        for asset in self._assets:
            self._poll_asset(asset, now, resolutions, snapshots)
        resolutions.sort(key=lambda e: e.ts)  # stable; delayed ones may carry later timestamps
        return resolutions + snapshots

    def _poll_asset(
        self, asset: str, now: float, resolutions: list[FeedEvent], snapshots: list[FeedEvent]
    ) -> None:
        window = self._windows.get(asset)
        if window is not None and window.ref_price is None and now < window.end_ts:
            return  # skipped (late) window: nothing to do until the next one starts
        price = self._read_spot(asset)
        if price is None:
            return
        if window is None or now >= window.end_ts:
            if window is not None:
                self._finish(window, price, resolutions)
            window = self._open_window(now, price)
            self._windows[asset] = window
        if window.ref_price is None:
            return
        if window.spec is None:
            window.spec = self._lookup(asset, window)
            if window.spec is None:
                return
        snapshot = self._snapshot(window, window.spec, now, price)
        if snapshot is not None:
            window.snapshots += 1
            snapshots.append(snapshot)

    # ------------------------------------------------------------------ window lifecycle

    def _open_window(self, now: float, price: float) -> _Window:
        start = math.floor(now / self._window) * self._window
        late = now - start > self._max_ref_lag
        self.stats["windows_skipped_late" if late else "windows_opened"] += 1
        return _Window(
            start_ts=start,
            end_ts=start + self._window,
            ref_price=None if late else price,
            trade_cursor=float(start),
        )

    def _finish(self, window: _Window, price: float, out: list[FeedEvent]) -> None:
        """Emit the (inferred) resolution of an ended window, once."""
        if window.spec is None or window.ref_price is None or window.snapshots == 0:
            return
        winner = Outcome.UP if price >= window.ref_price else Outcome.DOWN
        ts = max(float(window.end_ts), self._last_ts)
        out.append(MarketResolved(ts=ts, market_id=window.spec.market_id, winner=winner))

    def _slug(self, asset: str, start_ts: int) -> str:
        if "{minutes}" in self._template and self._window % 60 != 0:
            raise ValueError(
                "slug_template uses {minutes}: window_seconds must be a multiple of 60"
            )
        try:
            return self._template.format(
                asset=asset,
                asset_lower=asset.lower(),
                asset_upper=asset.upper(),
                minutes=self._window // 60,
                window_seconds=self._window,
                start_ts=start_ts,
                end_ts=start_ts + self._window,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"bad slug_template {self._template!r}: {exc!r}") from exc

    def _lookup(self, asset: str, window: _Window) -> MarketSpec | None:
        slug = self._slug(asset, window.start_ts)
        try:
            info = self._client.market_by_slug(slug)
        except DataError as exc:
            self._note_error("market", exc)
            return None
        return MarketSpec(
            market_id=info["condition_id"],
            asset=asset,
            start_ts=float(window.start_ts),
            end_ts=float(window.end_ts),
            tick_size=info.get("tick_size", self._cfg.sim.tick_size),
            min_order_size=info.get("min_order_size", self._cfg.sim.min_order_size),
            up_token_id=info["up_token_id"],
            down_token_id=info["down_token_id"],
            slug=slug,
        )

    # ------------------------------------------------------------------ data access

    def _snapshot(
        self, window: _Window, spec: MarketSpec, now: float, price: float
    ) -> MarketSnapshot | None:
        try:
            up_book = self._client.book(spec.up_token_id, Outcome.UP)
            down_book = self._client.book(spec.down_token_id, Outcome.DOWN)
        except DataError as exc:
            self._note_error("book", exc)
            return None
        trades = self._new_trades(window, spec, now)
        return MarketSnapshot(
            ts=now,
            market=spec,
            up_book=up_book,
            down_book=down_book,
            spot=price,
            spot_ts=now,
            ref_price=window.ref_price,
            trades=trades,
        )

    def _new_trades(self, window: _Window, spec: MarketSpec, now: float) -> tuple[Trade, ...]:
        try:
            trades = self._client.recent_trades(
                spec.market_id,
                window.trade_cursor,
                up_token_id=spec.up_token_id,
                down_token_id=spec.down_token_id,
            )
        except DataError as exc:
            self._note_error("trades", exc)
            return ()
        window.trade_cursor = now
        return tuple(trades)

    def _read_spot(self, asset: str) -> float | None:
        try:
            return self._spot.spot(asset)
        except DataError as exc:
            self._note_error("spot", exc)
            return None

    def _note_error(self, kind: str, exc: DataError) -> None:
        self.stats["errors"] += 1
        self.stats[f"errors_{kind}"] += 1
        self.last_error = f"{kind}: {exc}"


def record(
    events: Iterable[FeedEvent], path: str | os.PathLike[str]
) -> Generator[FeedEvent, None, None]:
    """Pass ``events`` through unchanged while appending each to a JSONL file.

    Each event is written and flushed BEFORE it is yielded, so a crash loses nothing that the
    consumer has seen. The file (parent directories created, existing file truncated) is opened
    on the first ``next()`` and closed when the stream ends or the generator is closed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for event in events:
            fh.write(event_to_json(event) + "\n")
            fh.flush()
            yield event
