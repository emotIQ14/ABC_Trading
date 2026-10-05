"""Tests for abc_trading.data.events (lossless JSON / JSONL round trip of FeedEvent)."""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import random
import re
import struct
from pathlib import Path
from typing import Any

import pytest

import abc_trading.data as data_pkg
from abc_trading.data.events import (
    SCHEMA_VERSION,
    event_from_dict,
    event_from_json,
    event_to_dict,
    event_to_json,
    read_jsonl,
    write_jsonl,
)
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

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "events_v1_golden.jsonl"


def spec(**kw: Any) -> MarketSpec:
    base: dict[str, Any] = {
        "market_id": "0xabc123",
        "asset": "BTC",
        "start_ts": 1704067200.0,
        "end_ts": 1704067500.0,
        "tick_size": 0.01,
        "min_order_size": 5.0,
        "up_token_id": "u1",
        "down_token_id": "d1",
        "slug": "btc-updown-5m-1704067200",
    }
    base.update(kw)
    return MarketSpec(**base)


def golden_events() -> list[FeedEvent]:
    """The three events of tests/fixtures/events_v1_golden.jsonl, built by hand."""
    full = MarketSnapshot(
        ts=1704067201.5,
        market=spec(),
        up_book=BookSnapshot(
            Outcome.UP, (Level(0.48, 80.0), Level(0.47, 130.0)), (Level(0.5, 40.25),)
        ),
        down_book=BookSnapshot(Outcome.DOWN, (Level(0.5, 40.25),), (Level(0.52, 80.0),)),
        spot=43250.12,
        spot_ts=1704067201.0,
        ref_price=43200.0,
        trades=(
            Trade(1704067201.2, Outcome.UP, 0.48, 10.0, Side.SELL),
            Trade(1704067201.2, Outcome.DOWN, 0.52, 10.0, Side.BUY),
        ),
    )
    bare = MarketSnapshot(
        ts=1704067202.0,
        market=spec(),
        up_book=BookSnapshot(Outcome.UP),
        down_book=BookSnapshot(Outcome.DOWN),
    )
    resolved = MarketResolved(ts=1704067500.0, market_id="0xabc123", winner=Outcome.DOWN)
    return [full, bare, resolved]


# --------------------------------------------------------------------------- golden / exact


def test_read_golden_file_equals_hand_built_events() -> None:
    assert list(read_jsonl(GOLDEN)) == golden_events()


def test_write_golden_events_is_byte_identical_to_golden_file(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    assert write_jsonl(out, golden_events()) == 3
    assert out.read_bytes() == GOLDEN.read_bytes()


def test_resolved_exact_dict_and_json() -> None:
    ev = MarketResolved(ts=1704067500.0, market_id="0xabc123", winner=Outcome.DOWN)
    assert event_to_dict(ev) == {
        "type": "resolved",
        "v": 1,
        "ts": 1704067500.0,
        "market_id": "0xabc123",
        "winner": "DOWN",
    }
    assert event_to_json(ev) == (
        '{"type":"resolved","v":1,"ts":1704067500.0,"market_id":"0xabc123","winner":"DOWN"}'
    )


def test_snapshot_dict_is_json_native_with_explicit_nulls() -> None:
    bare = golden_events()[1]
    d = event_to_dict(bare)
    assert d["spot"] is None and d["spot_ts"] is None and d["ref_price"] is None
    assert d["trades"] == []
    assert d["up_book"] == {"token": "UP", "bids": [], "asks": []}
    assert list(d)[:3] == ["type", "v", "ts"]  # discriminator and version lead every line
    assert json.loads(json.dumps(d)) == d  # nothing but JSON-native types


def test_schema_version_constant() -> None:
    assert SCHEMA_VERSION == 1


# --------------------------------------------------------------------------- round trips


def roundtrip(ev: FeedEvent) -> FeedEvent:
    return event_from_json(event_to_json(ev))


@pytest.mark.parametrize("ev", golden_events(), ids=["full", "bare-empty-books", "resolved"])
def test_roundtrip_golden_shapes(ev: FeedEvent) -> None:
    assert event_from_dict(event_to_dict(ev)) == ev
    assert roundtrip(ev) == ev


def test_roundtrip_both_winners() -> None:
    for w in Outcome:
        ev = MarketResolved(ts=5.0, market_id="m", winner=w)
        assert roundtrip(ev) == ev


def test_roundtrip_optional_fields_individually() -> None:
    base = golden_events()[1]
    assert isinstance(base, MarketSnapshot)
    changes: list[dict[str, Any]] = [{"spot": 1.5}, {"spot_ts": 2.5}, {"ref_price": 3.5}]
    for kw in changes:
        ev = dataclasses.replace(base, **kw)
        assert roundtrip(ev) == ev


def test_roundtrip_many_trades_preserves_order() -> None:
    trades = tuple(
        Trade(100.0 + i, Outcome.UP if i % 2 else Outcome.DOWN, 0.01 * (i + 1), 1.0 + i, Side.BUY)
        for i in range(25)
    )
    ev = MarketSnapshot(
        ts=200.0,
        market=spec(),
        up_book=BookSnapshot(Outcome.UP),
        down_book=BookSnapshot(Outcome.DOWN),
        trades=trades,
    )
    back = roundtrip(ev)
    assert isinstance(back, MarketSnapshot)
    assert back.trades == trades


def test_roundtrip_empty_market_strings_and_boundary_prices() -> None:
    ev = MarketSnapshot(
        ts=0.0,
        market=spec(up_token_id="", down_token_id="", slug=""),
        up_book=BookSnapshot(Outcome.UP, (Level(1.0, 0.0),), (Level(1.0, 1e-9),)),
        down_book=BookSnapshot(Outcome.DOWN, (Level(0.0, 3.0),), ()),
    )
    assert roundtrip(ev) == ev


def test_range_checks_tolerate_float_noise_but_never_alter_values() -> None:
    eps_noise = 1.0 + 2e-12  # e.g. an accumulated rounding error right at the boundary
    ev = MarketSnapshot(
        ts=1.0,
        market=spec(),
        up_book=BookSnapshot(Outcome.UP, (Level(eps_noise, -1e-12),), (Level(-1e-12, 3.0),)),
        down_book=BookSnapshot(Outcome.DOWN),
    )
    back = roundtrip(ev)
    assert back == ev  # accepted and preserved bit for bit, not clamped to 1.0 / 0.0
    assert isinstance(back, MarketSnapshot) and back.up_book.bids[0].price == eps_noise
    d = event_to_dict(ev)
    d["up_book"]["bids"][0][0] = 1.0 + 1e-6  # a real violation, well beyond EPS
    with pytest.raises(ValueError, match=r"price must be in \[0, 1\]"):
        event_from_dict(d)


def test_floats_roundtrip_bit_exactly() -> None:
    nasty = [0.1 + 0.2, 1e-7, 5e-324, 1.7976931348623157e308, 1e22, 123456789.123456789, 2.0**-52]
    ev = MarketSnapshot(
        ts=nasty[4],
        market=spec(start_ts=nasty[0], end_ts=nasty[3]),
        up_book=BookSnapshot(Outcome.UP, (Level(0.5, nasty[1]),), (Level(0.6, nasty[2]),)),
        down_book=BookSnapshot(Outcome.DOWN),
        spot=nasty[5],
        spot_ts=nasty[6],
        ref_price=-0.0,
    )
    back = roundtrip(ev)
    assert isinstance(back, MarketSnapshot)

    def bits(x: float) -> bytes:
        return struct.pack("<d", x)

    assert bits(back.market.start_ts) == bits(0.1 + 0.2)  # 0.30000000000000004, not 0.3
    assert bits(back.up_book.bids[0].size) == bits(1e-7)
    assert bits(back.up_book.asks[0].size) == bits(5e-324)
    assert bits(back.market.end_ts) == bits(1.7976931348623157e308)
    assert back.ref_price is not None and math.copysign(1.0, back.ref_price) == -1.0
    assert back == ev


def test_integers_in_numeric_fields_decode_as_floats() -> None:
    d = event_to_dict(MarketResolved(ts=5.0, market_id="m", winner=Outcome.UP))
    d["ts"] = 5
    ev = event_from_dict(d)
    assert isinstance(ev, MarketResolved)
    assert ev.ts == 5.0 and isinstance(ev.ts, float)


def random_snapshot(rng: random.Random, i: int) -> MarketSnapshot:
    def ladder(descending: bool) -> tuple[Level, ...]:
        prices = sorted({round(rng.uniform(0.01, 0.99), 6) for _ in range(rng.randint(0, 6))})
        if descending:
            prices.reverse()
        return tuple(Level(p, rng.uniform(0.0, 5000.0)) for p in prices)

    def maybe(x: float) -> float | None:
        return x if rng.random() < 0.6 else None

    start = float(rng.randint(1_600_000_000, 1_800_000_000))
    trades = tuple(
        Trade(
            ts=start + rng.uniform(0, 900),
            token=rng.choice(list(Outcome)),
            price=rng.uniform(0.0, 1.0),
            size=rng.uniform(0.0, 1000.0),
            aggressor=rng.choice(list(Side)),
        )
        for _ in range(rng.randint(0, 5))
    )
    return MarketSnapshot(
        ts=start + rng.uniform(0, 900),
        market=MarketSpec(
            market_id=f"0x{rng.getrandbits(64):016x}",
            asset=rng.choice(["BTC", "ETH", "SOL"]),
            start_ts=start,
            end_ts=start + rng.choice([300.0, 900.0]),
            tick_size=rng.choice([0.01, 0.001]),
            min_order_size=rng.choice([1.0, 5.0, 0.5]),
            up_token_id=str(rng.getrandbits(200)),
            down_token_id=str(rng.getrandbits(200)),
            slug=f"slug-{i}",
        ),
        up_book=BookSnapshot(Outcome.UP, ladder(True), ladder(False)),
        down_book=BookSnapshot(Outcome.DOWN, ladder(True), ladder(False)),
        spot=maybe(rng.uniform(1.0, 100_000.0)),
        spot_ts=maybe(start + rng.uniform(0, 900)),
        ref_price=maybe(rng.uniform(1.0, 100_000.0)),
        trades=trades,
    )


def test_seeded_random_roundtrip_500_snapshots_in_memory_and_via_file(tmp_path: Path) -> None:
    rng = random.Random(20240601)
    events = [random_snapshot(rng, i) for i in range(500)]
    for ev in events:
        assert roundtrip(ev) == ev
    path = tmp_path / "rand.jsonl"
    assert write_jsonl(path, events) == 500
    assert list(read_jsonl(path)) == events
    assert len(path.read_text().splitlines()) == 500


# --------------------------------------------------------------------------- decode errors


def good_snapshot_dict() -> dict[str, Any]:
    d = event_to_dict(golden_events()[0])
    return json.loads(json.dumps(d))  # deep copy


def test_unknown_type_raises() -> None:
    d = event_to_dict(golden_events()[2])
    d["type"] = "heartbeat"
    with pytest.raises(ValueError, match="unknown event type 'heartbeat'"):
        event_from_dict(d)


@pytest.mark.parametrize("bad", [2, 0, 1.5, "1", True, None])
def test_unsupported_version_raises(bad: Any) -> None:
    d = event_to_dict(golden_events()[2])
    d["v"] = bad
    with pytest.raises(ValueError, match="unsupported schema version"):
        event_from_dict(d)


def test_missing_type_or_version_raises() -> None:
    d = event_to_dict(golden_events()[2])
    no_type = {k: v for k, v in d.items() if k != "type"}
    no_v = {k: v for k, v in d.items() if k != "v"}
    with pytest.raises(ValueError, match="missing discriminator key 'type'"):
        event_from_dict(no_type)
    with pytest.raises(ValueError, match="missing schema version"):
        event_from_dict(no_v)


def test_non_object_event_raises() -> None:
    with pytest.raises(ValueError, match="expected an object"):
        event_from_dict([1, 2])  # type: ignore[arg-type]


def test_missing_and_unknown_keys_raise() -> None:
    d = good_snapshot_dict()
    del d["spot"]
    with pytest.raises(ValueError, match=r"missing key\(s\) \['spot'\]"):
        event_from_dict(d)
    d = good_snapshot_dict()
    d["spott"] = 1.0
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['spott'\]"):
        event_from_dict(d)
    d = good_snapshot_dict()
    d["market"]["extra"] = 1
    with pytest.raises(ValueError, match=r"market: unknown key"):
        event_from_dict(d)


def mutate(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    d = good_snapshot_dict()
    node: Any = d
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return d


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("ts",), "1704067201.5", "ts: expected a number"),
        (("ts",), True, "ts: expected a number"),
        (("ts",), None, "ts: expected a number"),
        (("spot",), "43250", "spot: expected a number"),
        (("market", "start_ts"), [], "market.start_ts: expected a number"),
        (("market", "slug"), 5, "market.slug: expected a string"),
        (("up_book", "token"), "DOWN", r"up_book.token: expected 'UP'"),
        (("down_book", "token"), "UP", r"down_book.token: expected 'DOWN'"),
        (("up_book", "token"), "SIDEWAYS", "up_book.token: expected 'UP' or 'DOWN'"),
        (("up_book", "bids"), "x", "up_book.bids: expected a list"),
        (("up_book", "bids", 0), [0.5], r"up_book.bids\[0\]: expected \[price, size\]"),
        (("up_book", "bids", 0, 0), 1.01, r"up_book.bids\[0\]\[0\]: price must be in \[0, 1\]"),
        (("up_book", "bids", 0, 0), -0.01, r"price must be in \[0, 1\]"),
        (("up_book", "asks", 0, 1), -1.0, r"up_book.asks\[0\]\[1\]: size must be >= 0"),
        (("trades",), {}, "trades: expected a list"),
        (("trades", 0, "aggressor"), "HOLD", r"trades\[0\].aggressor: expected 'BUY' or 'SELL'"),
        (("trades", 1, "token"), "FLAT", r"trades\[1\].token: expected 'UP' or 'DOWN'"),
        (("trades", 0, "price"), 2, r"trades\[0\].price: price must be in \[0, 1\]"),
        (("trades", 0, "size"), -3, r"trades\[0\].size: size must be >= 0"),
        (("trades", 0), {"ts": 1.0}, r"trades\[0\]: missing key"),
    ],
)
def test_malformed_snapshot_fields_raise_with_path(
    path: tuple[Any, ...], value: Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        event_from_dict(mutate(path, value))


def test_bad_resolved_winner_raises() -> None:
    d = event_to_dict(golden_events()[2])
    d["winner"] = "up"  # labels are exact: lower case is not accepted on the wire
    with pytest.raises(ValueError, match="winner: expected 'UP' or 'DOWN'"):
        event_from_dict(d)


def test_event_to_dict_rejects_non_events() -> None:
    with pytest.raises(TypeError, match="not a FeedEvent"):
        event_to_dict("snapshot")  # type: ignore[arg-type]


def test_json_level_errors() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        event_from_json("{not json")
    with pytest.raises(ValueError, match="expected an object"):
        event_from_json("[1,2,3]")
    with pytest.raises(ValueError, match="non-finite number NaN"):
        event_from_json('{"type":"resolved","v":1,"ts":NaN,"market_id":"m","winner":"UP"}')
    with pytest.raises(ValueError, match="non-finite number Infinity"):
        event_from_json('{"type":"resolved","v":1,"ts":Infinity,"market_id":"m","winner":"UP"}')
    with pytest.raises(ValueError, match="duplicate key 'ts'"):
        event_from_json('{"type":"resolved","v":1,"ts":1,"ts":2,"market_id":"m","winner":"UP"}')


def test_huge_integer_is_rejected_not_overflowed() -> None:
    line = '{"type":"resolved","v":1,"ts":1' + "0" * 400 + ',"market_id":"m","winner":"UP"}'
    with pytest.raises(ValueError, match="ts"):
        event_from_json(line)


def test_non_finite_floats_are_rejected_on_encode() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            event_to_json(MarketResolved(ts=bad, market_id="m", winner=Outcome.UP))


# --------------------------------------------------------------------------- JSONL files


def test_read_jsonl_skips_blank_lines_and_handles_crlf(tmp_path: Path) -> None:
    lines = [event_to_json(e) for e in golden_events()]
    path = tmp_path / "blanks.jsonl"
    path.write_bytes(("\r\n".join([lines[0], "", "   ", lines[1], lines[2]]) + "\r\n\r\n").encode())
    assert list(read_jsonl(path)) == golden_events()


def test_malformed_line_error_names_the_physical_line_number(tmp_path: Path) -> None:
    good = event_to_json(golden_events()[2])
    path = tmp_path / "bad.jsonl"
    path.write_text(good + "\n\n" + "{broken\n" + good + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl: line 3: invalid JSON"):
        list(read_jsonl(path))


def test_schema_error_line_number_and_laziness(tmp_path: Path) -> None:
    good = event_to_json(golden_events()[2])
    bad = good.replace('"winner":"DOWN"', '"winner":"SIDEWAYS"')
    path = tmp_path / "lazy.jsonl"
    path.write_text(good + "\n" + bad + "\n" + good + "\n")
    it = read_jsonl(path)
    assert next(it) == golden_events()[2]  # first event is delivered before line 2 is parsed
    with pytest.raises(ValueError, match=r"line 2: winner: expected 'UP' or 'DOWN'"):
        next(it)


def test_non_utf8_line_error_carries_line_number(tmp_path: Path) -> None:
    good = event_to_json(golden_events()[2]).encode()
    path = tmp_path / "latin.jsonl"
    path.write_bytes(good + b"\n" + b'{"type":"resolved","market_id":"\xe9"}\n')
    with pytest.raises(ValueError, match="line 2"):
        list(read_jsonl(path))


def test_read_missing_file_raises_on_first_next(tmp_path: Path) -> None:
    it = read_jsonl(tmp_path / "nope.jsonl")  # generator: nothing happens yet
    with pytest.raises(FileNotFoundError):
        next(it)


def test_write_jsonl_creates_parents_returns_count_and_accepts_generators(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c.jsonl"
    gen = (e for e in golden_events())
    assert write_jsonl(path, gen) == 3
    assert list(read_jsonl(path)) == golden_events()
    assert path.read_text().endswith("}\n")


def test_write_jsonl_empty_creates_empty_file_and_replaces_existing(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    path.write_text("old content\n")
    assert write_jsonl(path, []) == 0
    assert path.read_text() == ""
    assert list(read_jsonl(path)) == []


def test_write_failure_leaves_no_partial_or_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    bad = MarketResolved(ts=math.nan, market_id="m", winner=Outcome.UP)
    events = [golden_events()[2], bad, golden_events()[2]]
    with pytest.raises(ValueError, match=r"x\.jsonl: event 2:"):
        write_jsonl(path, events)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []

    path.write_text("previous good recording\n")
    with pytest.raises(ValueError, match="event 2"):
        write_jsonl(path, events)
    assert path.read_text() == "previous good recording\n"  # untouched
    assert [p.name for p in tmp_path.iterdir()] == ["x.jsonl"]


def test_write_failure_from_iterable_is_propagated_and_cleaned(tmp_path: Path) -> None:
    def boom() -> Any:
        yield golden_events()[2]
        raise RuntimeError("source died")

    path = tmp_path / "y.jsonl"
    with pytest.raises(RuntimeError, match="source died"):
        write_jsonl(path, boom())
    assert list(tmp_path.iterdir()) == []


def test_write_rejects_non_event_with_event_number(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="event 1: not a FeedEvent"):
        write_jsonl(tmp_path / "z.jsonl", ["nope"])  # type: ignore[list-item]


def test_each_line_is_compact_single_line_json(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    write_jsonl(path, golden_events())
    for line in path.read_text().splitlines():
        assert " " not in line.replace('"slug":"btc-updown-5m-1704067200"', "")
        assert json.loads(line)["v"] == 1


# --------------------------------------------------------------------------- package hygiene


def test_package_reexports_resolve() -> None:
    for name in data_pkg.__all__:
        assert hasattr(data_pkg, name), name
    assert data_pkg.read_jsonl is read_jsonl


CORE_MODULES = (
    "types.py", "config.py", "fees.py", "inventory.py",
    "model", "strategy", "exchange", "backtest", "sim",
)  # fmt: skip


def test_core_modules_do_not_import_the_data_package() -> None:
    root = Path(data_pkg.__file__).resolve().parents[1]
    offenders: list[str] = []
    for name in CORE_MODULES:
        target = root / name
        files = [target] if target.suffix == ".py" else sorted(target.rglob("*.py"))
        for file in files:
            if not file.exists():
                continue
            for node in ast.walk(ast.parse(file.read_text())):
                mods: list[str] = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [("." * node.level) + (node.module or "")]
                    mods += [f"{mods[0]}.{a.name}" for a in node.names]
                if any(re.search(r"(^|\.)data(\.|$)", m) for m in mods):
                    offenders.append(f"{file.relative_to(root)}: {mods}")
    assert offenders == []
