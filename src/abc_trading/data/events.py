"""Lossless JSON (de)serialisation of ``FeedEvent`` and JSONL event files.

OPTIONAL EDGE LAYER: nothing in the core modules imports this package.

Wire format (schema version ``SCHEMA_VERSION`` = 1), one JSON object per event::

    {"type":"snapshot","v":1,"ts":T,"market":{...MarketSpec fields...},
     "up_book":{"token":"UP","bids":[[price,size],...],"asks":[[price,size],...]},
     "down_book":{...},"spot":S|null,"spot_ts":S|null,"ref_price":S|null,
     "trades":[{"ts":T,"token":"UP","price":P,"size":Z,"aggressor":"BUY"},...]}
    {"type":"resolved","v":1,"ts":T,"market_id":"...","winner":"UP"}

Units follow ``types.py``: prices are USDC per share in [0, 1], sizes are shares, timestamps
are unix seconds. Book ladders are stored in the order given (best level first); the decoder
checks ranges (with ``EPS`` tolerance) and types but does not re-sort or clamp.

Invariants
----------
* ``event_from_dict(event_to_dict(e)) == e`` for every valid event, and floats round-trip
  bit-exactly (``json`` writes ``repr`` of a float, which is the shortest exact decimal).
* Every key is mandatory on decode and unknown keys are rejected: a typo or a schema drift
  raises ``ValueError`` instead of being silently defaulted. ``None`` is written as ``null``.
* Non-finite floats (NaN / Infinity) are rejected in both directions.
* All data errors are ``ValueError``; nothing is clamped or repaired.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from abc_trading.types import (
    EPS,
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

SCHEMA_VERSION = 1
TYPE_SNAPSHOT = "snapshot"
TYPE_RESOLVED = "resolved"

_SNAPSHOT_KEYS = (
    "type", "v", "ts", "market", "up_book", "down_book", "spot", "spot_ts", "ref_price", "trades",
)  # fmt: skip
_RESOLVED_KEYS = ("type", "v", "ts", "market_id", "winner")
_MARKET_KEYS = (
    "market_id", "asset", "start_ts", "end_ts", "tick_size", "min_order_size",
    "up_token_id", "down_token_id", "slug",
)  # fmt: skip
_BOOK_KEYS = ("token", "bids", "asks")
_TRADE_KEYS = ("ts", "token", "price", "size", "aggressor")


# --------------------------------------------------------------------------- encoding


def _book_to_dict(book: BookSnapshot) -> dict[str, Any]:
    return {
        "token": book.token.value,
        "bids": [[lv.price, lv.size] for lv in book.bids],
        "asks": [[lv.price, lv.size] for lv in book.asks],
    }


def _market_to_dict(m: MarketSpec) -> dict[str, Any]:
    return {
        "market_id": m.market_id,
        "asset": m.asset,
        "start_ts": m.start_ts,
        "end_ts": m.end_ts,
        "tick_size": m.tick_size,
        "min_order_size": m.min_order_size,
        "up_token_id": m.up_token_id,
        "down_token_id": m.down_token_id,
        "slug": m.slug,
    }


def _trade_to_dict(t: Trade) -> dict[str, Any]:
    return {
        "ts": t.ts,
        "token": t.token.value,
        "price": t.price,
        "size": t.size,
        "aggressor": t.aggressor.value,
    }


def event_to_dict(event: FeedEvent) -> dict[str, Any]:
    """Plain JSON-native dict (lists, not tuples) for one event; see the module docstring."""
    if isinstance(event, MarketSnapshot):
        return {
            "type": TYPE_SNAPSHOT,
            "v": SCHEMA_VERSION,
            "ts": event.ts,
            "market": _market_to_dict(event.market),
            "up_book": _book_to_dict(event.up_book),
            "down_book": _book_to_dict(event.down_book),
            "spot": event.spot,
            "spot_ts": event.spot_ts,
            "ref_price": event.ref_price,
            "trades": [_trade_to_dict(t) for t in event.trades],
        }
    if isinstance(event, MarketResolved):
        return {
            "type": TYPE_RESOLVED,
            "v": SCHEMA_VERSION,
            "ts": event.ts,
            "market_id": event.market_id,
            "winner": event.winner.value,
        }
    raise TypeError(f"not a FeedEvent: {type(event).__name__}")


def event_to_json(event: FeedEvent) -> str:
    """One compact JSON line (no trailing newline). ValueError on non-finite floats."""
    return json.dumps(event_to_dict(event), separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------- decoding


def _check_keys(obj: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    """``obj`` must be a mapping with exactly the keys ``keys``."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"{where}: expected an object, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{where}: missing key(s) {missing}")
    unknown = sorted((k for k in obj if k not in keys), key=str)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")
    return obj


def _num(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: expected a number, got {value!r}")
    try:
        x = float(value)
    except OverflowError:
        raise ValueError(f"{where}: number out of range: {value!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{where}: number must be finite, got {value!r}")
    return x


def _opt_num(value: Any, where: str) -> float | None:
    return None if value is None else _num(value, where)


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where}: expected a string, got {value!r}")
    return value


def _outcome(value: Any, where: str) -> Outcome:
    try:
        return Outcome(_str(value, where))
    except ValueError:
        raise ValueError(f"{where}: expected 'UP' or 'DOWN', got {value!r}") from None


def _side(value: Any, where: str) -> Side:
    try:
        return Side(_str(value, where))
    except ValueError:
        raise ValueError(f"{where}: expected 'BUY' or 'SELL', got {value!r}") from None


def _price(value: Any, where: str) -> float:
    x = _num(value, where)
    if not -EPS <= x <= 1.0 + EPS:  # float noise tolerated (as in types.py); value kept as is
        raise ValueError(f"{where}: price must be in [0, 1], got {x!r}")
    return x


def _size(value: Any, where: str) -> float:
    x = _num(value, where)
    if x < -EPS:
        raise ValueError(f"{where}: size must be >= 0, got {x!r}")
    return x


def _level(item: Any, where: str) -> Level:
    if not isinstance(item, list | tuple) or len(item) != 2:
        raise ValueError(f"{where}: expected [price, size], got {item!r}")
    return Level(_price(item[0], f"{where}[0]"), _size(item[1], f"{where}[1]"))


def _levels(raw: Any, where: str) -> tuple[Level, ...]:
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{where}: expected a list, got {raw!r}")
    return tuple(_level(item, f"{where}[{i}]") for i, item in enumerate(raw))


def _book_from_dict(raw: Any, expected: Outcome, where: str) -> BookSnapshot:
    d = _check_keys(raw, _BOOK_KEYS, where)
    token = _outcome(d["token"], f"{where}.token")
    if token is not expected:
        raise ValueError(f"{where}.token: expected {expected.value!r}, got {token.value!r}")
    return BookSnapshot(
        token=token,
        bids=_levels(d["bids"], f"{where}.bids"),
        asks=_levels(d["asks"], f"{where}.asks"),
    )


def _market_from_dict(raw: Any, where: str) -> MarketSpec:
    d = _check_keys(raw, _MARKET_KEYS, where)
    return MarketSpec(
        market_id=_str(d["market_id"], f"{where}.market_id"),
        asset=_str(d["asset"], f"{where}.asset"),
        start_ts=_num(d["start_ts"], f"{where}.start_ts"),
        end_ts=_num(d["end_ts"], f"{where}.end_ts"),
        tick_size=_num(d["tick_size"], f"{where}.tick_size"),
        min_order_size=_num(d["min_order_size"], f"{where}.min_order_size"),
        up_token_id=_str(d["up_token_id"], f"{where}.up_token_id"),
        down_token_id=_str(d["down_token_id"], f"{where}.down_token_id"),
        slug=_str(d["slug"], f"{where}.slug"),
    )


def _trade_from_dict(raw: Any, where: str) -> Trade:
    d = _check_keys(raw, _TRADE_KEYS, where)
    return Trade(
        ts=_num(d["ts"], f"{where}.ts"),
        token=_outcome(d["token"], f"{where}.token"),
        price=_price(d["price"], f"{where}.price"),
        size=_size(d["size"], f"{where}.size"),
        aggressor=_side(d["aggressor"], f"{where}.aggressor"),
    )


def _snapshot_from_dict(data: Mapping[str, Any]) -> MarketSnapshot:
    d = _check_keys(data, _SNAPSHOT_KEYS, TYPE_SNAPSHOT)
    raw_trades = d["trades"]
    if not isinstance(raw_trades, list | tuple):
        raise ValueError(f"trades: expected a list, got {raw_trades!r}")
    return MarketSnapshot(
        ts=_num(d["ts"], "ts"),
        market=_market_from_dict(d["market"], "market"),
        up_book=_book_from_dict(d["up_book"], Outcome.UP, "up_book"),
        down_book=_book_from_dict(d["down_book"], Outcome.DOWN, "down_book"),
        spot=_opt_num(d["spot"], "spot"),
        spot_ts=_opt_num(d["spot_ts"], "spot_ts"),
        ref_price=_opt_num(d["ref_price"], "ref_price"),
        trades=tuple(_trade_from_dict(t, f"trades[{i}]") for i, t in enumerate(raw_trades)),
    )


def _resolved_from_dict(data: Mapping[str, Any]) -> MarketResolved:
    d = _check_keys(data, _RESOLVED_KEYS, TYPE_RESOLVED)
    return MarketResolved(
        ts=_num(d["ts"], "ts"),
        market_id=_str(d["market_id"], "market_id"),
        winner=_outcome(d["winner"], "winner"),
    )


def event_from_dict(data: Mapping[str, Any]) -> FeedEvent:
    """Inverse of ``event_to_dict``. ValueError on unknown type/version or any malformed field."""
    if not isinstance(data, Mapping):
        raise ValueError(f"event: expected an object, got {type(data).__name__}")
    if "v" not in data:
        raise ValueError("event: missing schema version key 'v'")
    version = data["v"]
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {version!r} (reader supports {SCHEMA_VERSION})"
        )
    if "type" not in data:
        raise ValueError("event: missing discriminator key 'type'")
    etype = data["type"]
    if etype == TYPE_SNAPSHOT:
        return _snapshot_from_dict(data)
    if etype == TYPE_RESOLVED:
        return _resolved_from_dict(data)
    raise ValueError(f"unknown event type {etype!r}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def event_from_json(text: str) -> FeedEvent:
    """Parse one JSON line produced by ``event_to_json``. ValueError on any problem."""
    try:
        data = json.loads(
            text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} (column {exc.colno})") from None
    return event_from_dict(data)


# --------------------------------------------------------------------------- JSONL files


def write_jsonl(path: str | os.PathLike[str], events: Iterable[FeedEvent]) -> int:
    """Write one compact JSON object per line (UTF-8, ``\\n``); returns the event count.

    Parent directories are created. The file is written to a temporary sibling and moved into
    place only after every event encoded successfully, so ``path`` never holds a truncated
    recording after an error (ValueError names the 1-based event number). An existing file is
    replaced.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    count = 0
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for count, event in enumerate(events, start=1):
                try:
                    line = event_to_json(event)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{out}: event {count}: {exc}") from exc
                fh.write(line + "\n")
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[FeedEvent]:
    """Lazily yield the events of a JSONL file (blank lines skipped).

    This is a generator: the file is opened on the first ``next()``. Any malformed line raises
    ``ValueError("<path>: line N: ...")`` with the 1-based physical line number.
    """
    src = Path(path)
    with src.open("rb") as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                event = event_from_json(raw.decode("utf-8"))
            except ValueError as exc:  # includes UnicodeDecodeError
                raise ValueError(f"{src}: line {lineno}: {exc}") from exc
            yield event
