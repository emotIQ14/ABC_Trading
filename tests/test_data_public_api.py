"""Tests for abc_trading.data.public_api.

Everything here is offline. The parsers are exercised against SYNTHETIC fixtures
(tests/fixtures/synthetic_*.json) that this project invented in the shapes it expects: they prove
the parsing logic, NOT that the real Polymarket / Binance / Coinbase endpoints look like this.
``UrllibTransport`` is tested with injected openers and with a loopback server on 127.0.0.1.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import abc_trading.data.public_api as api
from abc_trading.data.public_api import (
    DEFAULT_USER_AGENT,
    DataError,
    PolymarketPublicClient,
    SpotClient,
    Transport,
    UrllibTransport,
)
from abc_trading.types import Level, Outcome, Side, Trade

FIXTURES = Path(__file__).parent / "fixtures"

UP_ID = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
DOWN_ID = "52114319501245915516055106046884209969926127482827954674443846427813813222426"

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_URL = "https://clob.polymarket.com/book"
DATA_URL = "https://data-api.polymarket.com/trades"
BINANCE_URL = "https://api.binance.com/api/v3/ticker/price"
COINBASE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FakeTransport:
    """Serves canned JSON by exact URL; records every call. Unknown URLs fail the test."""

    def __init__(self, routes: Mapping[str, Any]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def get_json(
        self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
    ) -> Any:
        self.calls.append((url, dict(params or {}), timeout))
        if url not in self.routes:
            raise AssertionError(f"unexpected URL {url}")
        route = self.routes[url]
        if isinstance(route, Exception):
            raise route
        return route


def poly(routes: Mapping[str, Any], **kw: Any) -> tuple[PolymarketPublicClient, FakeTransport]:
    transport = FakeTransport(routes)
    t: Transport = transport  # structural conformance with the Protocol is checked by mypy
    return PolymarketPublicClient(t, **kw), transport


# --------------------------------------------------------------------------- market_by_slug


def test_market_by_slug_string_encoded_fields() -> None:
    client, fake = poly({GAMMA_URL: fixture("synthetic_gamma_market_strings.json")})
    info = client.market_by_slug("btc-updown-5m-1704067200")
    # eventStartTime 2024-01-01T00:00:00Z = 19723 days * 86400 s = 1_704_067_200 (preferred
    # over startDate 23:50, i.e. market creation); endDate = +300 s.
    assert info == {
        "slug": "btc-updown-5m-1704067200",
        "condition_id": "0xabc123",
        "up_token_id": UP_ID,
        "down_token_id": DOWN_ID,
        "start_ts": 1704067200.0,
        "end_ts": 1704067500.0,
        "tick_size": 0.01,
        "min_order_size": 5.0,
    }
    assert fake.calls == [(GAMMA_URL, {"slug": "btc-updown-5m-1704067200"}, 10.0)]


def test_market_by_slug_list_encoded_fields_reversed_labels_and_int_token_id() -> None:
    client, _ = poly({GAMMA_URL: fixture("synthetic_gamma_market_lists.json")})
    info = client.market_by_slug("eth-updown-15m-1704067200")
    # outcomes ["down", "UP"] pair with ids [DOWN, UP]; the UP id arrives as a JSON integer.
    # startDate 01:00+01:00 = 00:00Z = 1_704_067_200; null endDate and "" orderMinSize omitted.
    assert info == {
        "slug": "eth-updown-15m-1704067200",
        "condition_id": "0xdef456",
        "up_token_id": UP_ID,
        "down_token_id": DOWN_ID,
        "start_ts": 1704067200.0,
        "tick_size": 0.001,
    }


def test_market_by_slug_picks_the_matching_slug_from_several() -> None:
    other = dict(
        fixture("synthetic_gamma_market_strings.json")[0],
        slug="btc-updown-5m-1",
        conditionId="0xother",
    )
    right = fixture("synthetic_gamma_market_strings.json")[0]
    client, _ = poly({GAMMA_URL: [other, right]})
    assert client.market_by_slug("btc-updown-5m-1704067200")["condition_id"] == "0xabc123"


def test_market_not_found() -> None:
    client, _ = poly({GAMMA_URL: []})
    with pytest.raises(DataError, match="market not found"):
        client.market_by_slug("btc-updown-5m-1")
    other = [dict(fixture("synthetic_gamma_market_strings.json")[0], slug="something-else")]
    client, _ = poly({GAMMA_URL: other})
    with pytest.raises(DataError, match="market not found"):
        client.market_by_slug("btc-updown-5m-1704067200")


def with_market(**changes: Any) -> Any:
    m = dict(fixture("synthetic_gamma_market_strings.json")[0])
    for k, v in changes.items():
        if v is None:
            m.pop(k, None)
        else:
            m[k] = v
    return [m]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not a market", "expected a list"),
        (with_market(conditionId=None), "no conditionId"),
        (with_market(conditionId="  "), "no conditionId"),
        (with_market(outcomes=None), "outcomes: expected a list"),
        (with_market(outcomes='["Yes", "No"]'), "do not contain both Up and Down"),
        (with_market(outcomes='["Up", "Up"]'), "duplicate label"),
        (with_market(outcomes='["Up"]'), "differ"),
        (with_market(outcomes="[1, 2]"), "expected a string"),
        (with_market(outcomes="not json"), "string is not valid JSON"),
        (with_market(outcomes='{"a": 1}'), "expected a list"),
        (with_market(clobTokenIds=None), "clobTokenIds: expected a list"),
        (with_market(clobTokenIds='["5", "5"]'), "same token id"),
        (with_market(clobTokenIds='["5", ""]'), "empty token id"),
        (with_market(clobTokenIds='["5", 1.5]'), "expected a token id"),
        (with_market(clobTokenIds='["5", true]'), "expected a token id"),
        (with_market(orderPriceMinTickSize=0), "must be > 0"),
        (with_market(orderMinSize="abc"), "not a number"),
        (with_market(endDate="next tuesday"), "unparseable time"),
        (with_market(eventStartTime=True), "expected a number"),
    ],
)
def test_malformed_market_payloads_raise_data_error(payload: Any, match: str) -> None:
    client, _ = poly({GAMMA_URL: payload})
    with pytest.raises(DataError, match=match):
        client.market_by_slug("btc-updown-5m-1704067200")


def test_market_times_accept_millis_numeric_strings_and_naive_iso() -> None:
    payload = with_market(eventStartTime="1704067200000", endDate="2024-01-01T00:05:00")
    client, _ = poly({GAMMA_URL: payload})
    info = client.market_by_slug("btc-updown-5m-1704067200")
    assert info["start_ts"] == 1704067200.0  # milliseconds detected and converted
    assert info["end_ts"] == 1704067500.0  # naive ISO time is taken as UTC


def test_transport_data_error_propagates_unchanged() -> None:
    boom = DataError("HTTP 503 from https://gamma-api.polymarket.com/markets")
    client, _ = poly({GAMMA_URL: boom})
    with pytest.raises(DataError, match="HTTP 503"):
        client.market_by_slug("x")


def test_client_argument_validation_and_base_normalisation() -> None:
    client, fake = poly(
        {"http://gamma.test/markets": fixture("synthetic_gamma_market_strings.json")},
        gamma_base="http://gamma.test/",
        timeout=2.5,
    )
    client.market_by_slug("btc-updown-5m-1704067200")
    assert fake.calls == [("http://gamma.test/markets", {"slug": "btc-updown-5m-1704067200"}, 2.5)]
    with pytest.raises(ValueError, match="slug"):
        client.market_by_slug("")
    with pytest.raises(ValueError, match="base URL"):
        PolymarketPublicClient(FakeTransport({}), gamma_base="ftp://x")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout"):
            PolymarketPublicClient(FakeTransport({}), timeout=bad)


# --------------------------------------------------------------------------- book


def test_book_from_fixture_sorts_merges_and_drops_non_positive_sizes() -> None:
    client, fake = poly({CLOB_URL: fixture("synthetic_clob_book.json")})
    book = client.book(UP_ID, Outcome.UP)
    assert book.token is Outcome.UP
    # bids high->low; two 0.47 entries merge: 120.5 + 9.5 = 130; size 0 (0.46) and -3 (0.45) dropped
    assert book.bids == (Level(0.48, 80.0), Level(0.47, 130.0), Level(0.01, 500.0))
    assert book.asks == (
        Level(0.5, 40.25),
        Level(0.51, 10.0),
        Level(0.52, 60.0),
        Level(0.99, 1000.0),
    )
    assert book.best_bid == 0.48 and book.best_ask == 0.5
    assert book.mid == pytest.approx(0.49) and book.spread == pytest.approx(0.02)
    assert not book.is_crossed
    assert fake.calls == [(CLOB_URL, {"token_id": UP_ID}, 10.0)]


def test_book_token_label_is_stamped_from_argument() -> None:
    client, _ = poly({CLOB_URL: fixture("synthetic_clob_book.json")})
    assert client.book(DOWN_ID, Outcome.DOWN).token is Outcome.DOWN


def test_empty_and_one_sided_books() -> None:
    client, fake = poly({CLOB_URL: {"bids": [], "asks": []}})
    book = client.book("1", Outcome.UP)
    assert book.bids == () and book.asks == () and book.mid is None
    fake.routes[CLOB_URL] = {"bids": [{"price": 0.4, "size": 5}]}  # numbers, asks key absent
    book = client.book("1", Outcome.UP)
    assert book.bids == (Level(0.4, 5.0),) and book.asks == ()
    fake.routes[CLOB_URL] = {"bids": None, "asks": [{"price": "0.6", "size": "0"}]}
    book = client.book("1", Outcome.UP)
    assert book.bids == () and book.asks == ()  # only a zero-size level: dropped


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "expected an object"),
        ({"market": "x"}, "neither bids nor asks"),
        ({"bids": "x"}, "bids: expected a list"),
        ({"bids": ["0.5"]}, r"bids\[0\]: expected an object"),
        ({"bids": [{"price": "0.5"}]}, r"bids\[0\]: expected an object"),
        ({"bids": [{"price": "abc", "size": "1"}]}, "not a number"),
        ({"bids": [{"price": "1.2", "size": "1"}]}, r"price must be in \[0, 1\]"),
        ({"asks": [{"price": "-0.1", "size": "1"}]}, r"price must be in \[0, 1\]"),
        ({"asks": [{"price": "nan", "size": "1"}]}, "must be finite"),
        ({"asks": [{"price": "0.5", "size": "inf"}]}, "must be finite"),
        ({"asks": [{"price": "0.5", "size": None}]}, "expected a number"),
    ],
)
def test_malformed_book_payloads_raise_data_error(payload: Any, match: str) -> None:
    client, _ = poly({CLOB_URL: payload})
    with pytest.raises(DataError, match=match):
        client.book("1", Outcome.UP)


def test_book_rejects_empty_token_id() -> None:
    client, _ = poly({})
    with pytest.raises(ValueError, match="token_id"):
        client.book("", Outcome.UP)


# --------------------------------------------------------------------------- recent_trades


def trades_client(
    payload: Any = None, *, lookup: bool = True
) -> tuple[PolymarketPublicClient, FakeTransport]:
    routes: dict[str, Any] = {
        DATA_URL: payload if payload is not None else fixture("synthetic_data_trades.json")
    }
    if lookup:
        routes[GAMMA_URL] = fixture("synthetic_gamma_market_strings.json")
    client, fake = poly(routes)
    if lookup:
        client.market_by_slug("btc-updown-5m-1704067200")  # remembers the token ids
        fake.calls.clear()
    return client, fake


EXPECTED_TRADES = [
    # real BUY Up 10 @ 0.45 (ts ...210) and its mirror: SELL Down @ 1 - 0.45 = 0.55
    Trade(1704067210.0, Outcome.UP, 0.45, 10.0, Side.BUY),
    Trade(1704067210.0, Outcome.DOWN, 0.55, 10.0, Side.SELL),
    # real SELL Down 25.5 @ 0.30 (ts ...212) and its mirror: BUY Up @ 1 - 0.30 = 0.70
    Trade(1704067212.0, Outcome.DOWN, 0.3, 25.5, Side.SELL),
    Trade(1704067212.0, Outcome.UP, 0.7, 25.5, Side.BUY),
]


def test_recent_trades_mirrors_dedupes_sorts_and_filters() -> None:
    client, fake = trades_client()
    got = client.recent_trades("0xabc123", 1704067200.7)
    assert got == EXPECTED_TRADES
    # skipped: missing price, price 1.5, foreign conditionId; one in-response duplicate;
    # the ts ...190 record is before floor(since_ts) = ...200 and is simply filtered
    assert client.stats == {"trades_skipped": 3, "trades_duplicate": 1}
    assert fake.calls == [(DATA_URL, {"market": "0xabc123", "limit": "500"}, 10.0)]


def test_recent_trades_are_not_returned_twice_across_calls() -> None:
    client, _ = trades_client()
    assert client.recent_trades("0xabc123", 1704067200.0) == EXPECTED_TRADES
    assert client.recent_trades("0xabc123", 1704067200.0) == []  # same payload again
    # call 1: the in-response copy of 0xt2... (1). call 2: 0xt2, 0xt1 and that copy again (3).
    assert client.stats["trades_duplicate"] == 1 + 3


def test_recent_trades_new_trade_appears_between_calls() -> None:
    client, fake = trades_client()
    client.recent_trades("0xabc123", 1704067200.0)
    new = {
        "side": "buy",  # lower case accepted
        "asset": UP_ID,
        "conditionId": "0xabc123",
        "size": "4",
        "price": "0.07",
        "timestamp": 1704067213,
        "transactionHash": "0xt3",
    }
    fake.routes[DATA_URL] = [new, *fixture("synthetic_data_trades.json")]
    got = client.recent_trades("0xabc123", 1704067210.9)
    # 1 - 0.07 = 0.9299999999999999 in binary floating point; the mirror price is rounded to 6 dp
    assert got == [
        Trade(1704067213.0, Outcome.UP, 0.07, 4.0, Side.BUY),
        Trade(1704067213.0, Outcome.DOWN, 0.93, 4.0, Side.SELL),
    ]


def test_since_ts_lower_bound_is_floored_and_inclusive() -> None:
    client, _ = trades_client()
    # floor(1704067210.9) = 1704067210: the ts ...210 trade is kept, ts ...190 dropped
    got = client.recent_trades("0xabc123", 1704067210.9)
    assert [t.ts for t in got] == [1704067210.0, 1704067210.0, 1704067212.0, 1704067212.0]
    client2, _ = trades_client()
    assert client2.recent_trades("0xabc123", 1704067211.0) == EXPECTED_TRADES[2:]
    client3, _ = trades_client()
    assert client3.recent_trades("0xabc123", 1704067213.0) == []


def test_label_fallback_when_token_ids_unknown_and_explicit_ids() -> None:
    client, _ = trades_client(lookup=False)
    assert client.recent_trades("0xabc123", 1704067200.0) == EXPECTED_TRADES  # via "outcome"
    client2, _ = trades_client(lookup=False)
    swapped = client2.recent_trades(
        "0xabc123", 1704067200.0, up_token_id=DOWN_ID, down_token_id=UP_ID
    )
    # explicit ids win over the "Up" label: the BUY on asset UP_ID (ts ...210) is now a DOWN print
    assert swapped[:2] == [
        Trade(1704067210.0, Outcome.DOWN, 0.45, 10.0, Side.BUY),
        Trade(1704067210.0, Outcome.UP, 0.55, 10.0, Side.SELL),
    ]
    with pytest.raises(ValueError, match="both"):
        client2.recent_trades("0xabc123", 0.0, up_token_id=UP_ID)


def test_unmappable_records_are_skipped_and_counted() -> None:
    payload = [
        {"side": "BUY", "asset": "unknown-asset", "size": 1, "price": 0.5, "timestamp": 10},
        {"side": "HOLD", "asset": UP_ID, "size": 1, "price": 0.5, "timestamp": 10},
        {"side": "BUY", "asset": UP_ID, "size": 0, "price": 0.5, "timestamp": 10},
        {"side": "BUY", "asset": UP_ID, "size": 1, "price": 0.5},
        {"side": "BUY", "asset": UP_ID, "size": 1, "price": 0.5, "timestamp": "soon"},
        "not an object",
    ]
    client, _ = trades_client(payload)
    assert client.recent_trades("0xabc123", 0.0) == []
    assert client.stats["trades_skipped"] == 6


def test_millisecond_trade_timestamps_are_converted() -> None:
    rec = {"side": "BUY", "asset": UP_ID, "size": 2, "price": 0.5, "timestamp": 1704067210123}
    client, _ = trades_client([rec])
    got = client.recent_trades("0xabc123", 1704067210.0)
    assert [t.ts for t in got] == [1704067210.123, 1704067210.123]


def test_trades_sorted_by_time_even_if_endpoint_is_newest_first() -> None:
    recs = [
        {
            "side": "BUY",
            "asset": UP_ID,
            "size": 1,
            "price": 0.5,
            "timestamp": 30,
            "transactionHash": "c",
        },
        {
            "side": "BUY",
            "asset": UP_ID,
            "size": 2,
            "price": 0.5,
            "timestamp": 20,
            "transactionHash": "b",
        },
        {
            "side": "BUY",
            "asset": UP_ID,
            "size": 3,
            "price": 0.5,
            "timestamp": 10,
            "transactionHash": "a",
        },
    ]
    client, _ = trades_client(recs)
    got = client.recent_trades("0xabc123", 0.0)
    assert [(t.ts, t.size, t.token) for t in got] == [
        (10.0, 3.0, Outcome.UP), (10.0, 3.0, Outcome.DOWN),
        (20.0, 2.0, Outcome.UP), (20.0, 2.0, Outcome.DOWN),
        (30.0, 1.0, Outcome.UP), (30.0, 1.0, Outcome.DOWN),
    ]  # fmt: skip


def test_dedupe_memory_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "MAX_SEEN_TRADES", 3)

    def rec(i: int) -> dict[str, Any]:
        return {
            "side": "BUY", "asset": UP_ID, "size": 1, "price": 0.5,
            "timestamp": 10 + i, "transactionHash": f"t{i}",
        }  # fmt: skip

    client, fake = trades_client([rec(0), rec(1), rec(2)])
    assert len(client.recent_trades("0xabc123", 0.0)) == 6  # 3 real trades, both prints
    fake.routes[DATA_URL] = [rec(3)]
    assert len(client.recent_trades("0xabc123", 0.0)) == 2  # memory now holds t1, t2, t3
    # t1 is still remembered; t0 was evicted (oldest first) and is reported again.
    fake.routes[DATA_URL] = [rec(1), rec(0)]
    again = client.recent_trades("0xabc123", 0.0)
    assert [t.ts for t in again if t.token is Outcome.UP] == [10.0]


def test_recent_trades_limit_param_and_validation() -> None:
    client, fake = trades_client([])
    assert client.recent_trades("0xabc123", 0.0, limit=50) == []
    assert fake.calls[-1][1] == {"market": "0xabc123", "limit": "50"}
    with pytest.raises(ValueError, match="limit"):
        client.recent_trades("0xabc123", 0.0, limit=0)
    with pytest.raises(ValueError, match="since_ts"):
        client.recent_trades("0xabc123", float("nan"))
    with pytest.raises(ValueError, match="condition_id"):
        client.recent_trades("", 0.0)


@pytest.mark.parametrize("payload", [{"data": []}, "oops", None, 5])
def test_recent_trades_non_list_payload_is_data_error(payload: Any) -> None:
    client, fake = trades_client(lookup=False)
    fake.routes[DATA_URL] = payload
    with pytest.raises(DataError, match="expected a list"):
        client.recent_trades("0xabc123", 0.0)


# --------------------------------------------------------------------------- spot


def spot_client(routes: Mapping[str, Any]) -> tuple[SpotClient, FakeTransport]:
    fake = FakeTransport(routes)
    return SpotClient(fake), fake


def test_binance_and_coinbase_prices() -> None:
    client, fake = spot_client(
        {
            BINANCE_URL: fixture("synthetic_binance_ticker.json"),
            COINBASE_URL: fixture("synthetic_coinbase_spot.json"),
        }
    )
    assert client.binance_price("BTCUSDT") == 43250.12
    assert client.coinbase_price("BTC-USD") == 43249.5
    assert fake.calls == [
        (BINANCE_URL, {"symbol": "BTCUSDT"}, 10.0),
        (COINBASE_URL, {}, 10.0),
    ]


def test_spot_prefers_binance_and_records_source() -> None:
    client, fake = spot_client({BINANCE_URL: fixture("synthetic_binance_ticker.json")})
    assert client.spot("btc") == 43250.12  # case-insensitive asset
    assert client.last_source == "binance"
    assert [c[0] for c in fake.calls] == [BINANCE_URL]  # Coinbase never contacted


def test_spot_falls_back_to_coinbase_when_binance_fails() -> None:
    client, fake = spot_client(
        {
            BINANCE_URL: DataError("HTTP 451 from https://api.binance.com/api/v3/ticker/price"),
            COINBASE_URL: fixture("synthetic_coinbase_spot.json"),
        }
    )
    assert client.spot("BTC") == 43249.5
    assert client.last_source == "coinbase"
    assert [c[0] for c in fake.calls] == [BINANCE_URL, COINBASE_URL]
    assert fake.calls[0][1] == {"symbol": "BTCUSDT"}


def test_spot_falls_back_when_binance_payload_is_malformed() -> None:
    client, _ = spot_client(
        {BINANCE_URL: {"price": "0"}, COINBASE_URL: fixture("synthetic_coinbase_spot.json")}
    )
    assert client.spot("BTC") == 43249.5


def test_spot_both_fail_reports_both_reasons() -> None:
    client, _ = spot_client(
        {BINANCE_URL: DataError("HTTP 451 from b"), COINBASE_URL: DataError("timeout from c")}
    )
    client.last_source = "binance"
    with pytest.raises(DataError) as info:
        client.spot("BTC")
    msg = str(info.value)
    assert (
        "spot unavailable for BTC" in msg
        and "binance: HTTP 451" in msg
        and "coinbase: timeout" in msg
    )
    assert client.last_source is None


BINANCE_BAD = [[], {}, {"price": None}, {"price": "0"}, {"price": "-5"}, {"price": "nan"},
               {"price": "abc"}, {"price": True}]  # fmt: skip


@pytest.mark.parametrize("payload", BINANCE_BAD)
def test_binance_malformed_payloads(payload: Any) -> None:
    client, _ = spot_client({BINANCE_URL: payload})
    with pytest.raises(DataError):
        client.binance_price("BTCUSDT")


@pytest.mark.parametrize("payload", [[], {}, {"data": []}, {"data": {}}, {"data": {"amount": "0"}},
                                     {"data": {"amount": "x"}}])  # fmt: skip
def test_coinbase_malformed_payloads(payload: Any) -> None:
    client, _ = spot_client({COINBASE_URL: payload})
    with pytest.raises(DataError):
        client.coinbase_price("BTC-USD")


@pytest.mark.parametrize("asset", ["", "BTC/USD", "../x", "BTC USD", "B" * 16, "BTC-USD"])
def test_invalid_asset_names_are_rejected_before_any_request(asset: str) -> None:
    client, fake = spot_client({})
    with pytest.raises(ValueError, match="invalid asset"):
        client.spot(asset)
    assert fake.calls == []


def test_invalid_symbol_and_product_rejected() -> None:
    client, fake = spot_client({})
    with pytest.raises(ValueError, match="Binance symbol"):
        client.binance_price("BTC/USDT")
    with pytest.raises(ValueError, match="Coinbase product"):
        client.coinbase_price("BTC")
    with pytest.raises(ValueError, match="Coinbase product"):
        client.coinbase_price("../../v2/time")
    assert fake.calls == []


# ------------------------------------------------------------------ UrllibTransport (fakes)


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        return self._body if n < 0 else self._body[:n]


class FakeOpener:
    def __init__(self, result: bytes | BaseException) -> None:
        self.result = result
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []

    def open(self, fullurl: urllib.request.Request, /, *, timeout: float) -> FakeResponse:
        self.requests.append(fullurl)
        self.timeouts.append(timeout)
        if isinstance(self.result, BaseException):
            raise self.result
        return FakeResponse(self.result)


def test_urllib_transport_builds_request_and_parses_json() -> None:
    opener = FakeOpener(b'{"ok": [1, 2.5, "x"]}')
    transport = UrllibTransport(opener=opener)
    out = transport.get_json("https://example.test/path", {"q": "a b&c", "n": "1"}, timeout=3.5)
    assert out == {"ok": [1, 2.5, "x"]}
    (request,) = opener.requests
    assert request.full_url == "https://example.test/path?q=a+b%26c&n=1"
    assert request.get_method() == "GET" and request.data is None
    assert request.get_header("User-agent") == DEFAULT_USER_AGENT
    assert request.get_header("Accept") == "application/json"
    assert opener.timeouts == [3.5]


def test_urllib_transport_appends_to_existing_query_and_custom_user_agent() -> None:
    opener = FakeOpener(b"[]")
    transport = UrllibTransport("my-agent/9", opener=opener)
    assert transport.get_json("https://example.test/p?a=1", {"b": "2"}) == []
    assert opener.requests[0].full_url == "https://example.test/p?a=1&b=2"
    assert opener.requests[0].get_header("User-agent") == "my-agent/9"
    assert opener.timeouts == [10.0]  # default timeout
    assert transport.get_json("https://example.test/p") == []
    assert opener.requests[1].full_url == "https://example.test/p"


def http_error(code: int) -> urllib.error.HTTPError:
    url = "https://example.test/p?secret=1"
    return urllib.error.HTTPError(url, code, "Reason", http.client.HTTPMessage(), io.BytesIO(b""))


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (http_error(503), r"HTTP 503 from https://example\.test/p$"),
        (http_error(404), "HTTP 404"),
        (urllib.error.URLError("name resolution failed"), "network error from .*name resolution"),
        (urllib.error.URLError(TimeoutError("timed out")), "network error"),
        (TimeoutError("read timed out"), "timeout after 2.5s"),
        (ConnectionResetError("reset by peer"), "connection error .*reset by peer"),
        (http.client.IncompleteRead(b"abc"), "connection error"),
        (http.client.RemoteDisconnected("closed"), "connection error"),
        (http.client.InvalidURL("bad url"), "connection error .*bad url"),
        (UnicodeEncodeError("ascii", "\u00e9", 0, 1, "not ascii"), "connection error"),
        (OSError("ssl exploded " * 40), r"connection error .*\.\.\.$"),
    ],
)
def test_urllib_transport_maps_failures_to_short_data_errors(
    failure: BaseException, match: str
) -> None:
    transport = UrllibTransport(opener=FakeOpener(failure))
    with pytest.raises(DataError, match=match) as info:
        transport.get_json("https://example.test/p", {"secret": "1"}, timeout=2.5)
    assert len(str(info.value)) < 250
    assert "secret" not in str(info.value)  # query strings are never echoed


@pytest.mark.parametrize("body", [b"", b"<html>nope</html>", b"{truncated", b"\xff\xfe\x00bad"])
def test_urllib_transport_invalid_json_body(body: bytes) -> None:
    transport = UrllibTransport(opener=FakeOpener(body))
    with pytest.raises(DataError, match="invalid JSON from https://example.test/p"):
        transport.get_json("https://example.test/p")


def test_urllib_transport_caps_response_size() -> None:
    transport = UrllibTransport(opener=FakeOpener(b'["' + b"x" * 20 + b'"]'), max_bytes=10)
    with pytest.raises(DataError, match="exceeds 10 bytes"):
        transport.get_json("https://example.test/p")
    exact = UrllibTransport(opener=FakeOpener(b'["abcd"]'), max_bytes=8)  # exactly 8 bytes: ok
    assert exact.get_json("https://example.test/p") == ["abcd"]


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.test/x", "example.test/x", ""]
)
def test_urllib_transport_refuses_non_http_urls(url: str) -> None:
    opener = FakeOpener(b"{}")
    with pytest.raises(DataError, match="refusing non-http"):
        UrllibTransport(opener=opener).get_json(url)
    assert opener.requests == []


def test_urllib_transport_argument_validation() -> None:
    transport = UrllibTransport(opener=FakeOpener(b"{}"))
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="timeout"):
            transport.get_json("https://example.test/p", timeout=bad)
    with pytest.raises(ValueError, match="max_bytes"):
        UrllibTransport(max_bytes=0)
    assert isinstance(UrllibTransport(), UrllibTransport)  # default opener builds offline


# ---------------------------------------------------------------- UrllibTransport (loopback)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/echo"):
            body = json.dumps(
                {
                    "path": self.path,
                    "ua": self.headers.get("User-Agent"),
                    "accept": self.headers.get("Accept"),
                }
            ).encode()
            self._send(200, body)
        elif self.path.startswith("/badjson"):
            self._send(200, b"<html>not json</html>")
        else:
            self._send(404, b'{"error": "not found"}')

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def loopback() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def direct_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # ignore proxy env


def test_loopback_roundtrip_headers_and_query(loopback: str) -> None:
    transport = UrllibTransport(opener=direct_opener())
    out = transport.get_json(loopback + "/echo", {"slug": "btc up", "n": "1"}, timeout=5.0)
    assert out == {
        "path": "/echo?slug=btc+up&n=1",
        "ua": DEFAULT_USER_AGENT,
        "accept": "application/json",
    }


def test_loopback_http_error_and_bad_body(loopback: str) -> None:
    transport = UrllibTransport(opener=direct_opener())
    with pytest.raises(DataError, match=r"HTTP 404 from http://127\.0\.0\.1:\d+/missing$"):
        transport.get_json(loopback + "/missing", {"x": "1"}, timeout=5.0)
    with pytest.raises(DataError, match="invalid JSON"):
        transport.get_json(loopback + "/badjson", timeout=5.0)


def test_loopback_connection_refused_is_data_error(loopback: str) -> None:
    dead = "http://127.0.0.1:1"  # port 1: nothing listens
    with pytest.raises(DataError, match="network error|connection error"):
        UrllibTransport(opener=direct_opener()).get_json(dead + "/x", timeout=2.0)


def clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            monkeypatch.delenv(key)


def test_default_opener_honours_proxy_environment(
    loopback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_proxy_env(monkeypatch)
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")  # a proxy nobody listens on
    with pytest.raises(DataError, match="network error|connection error"):
        UrllibTransport().get_json(loopback + "/echo", timeout=2.0)  # went to the proxy
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    assert UrllibTransport().get_json(loopback + "/echo", timeout=5.0)["path"] == "/echo"


def test_module_constants_hold_every_endpoint_string() -> None:
    # The UNVERIFIED endpoint strings must live in named constants (see the module docstring).
    assert api.GAMMA_MARKETS_PATH == "/markets" and api.CLOB_BOOK_PATH == "/book"
    assert api.DATA_TRADES_PATH == "/trades"
    assert api.BINANCE_TICKER_PATH.startswith("/") and "{product}" in api.COINBASE_SPOT_PATH
    assert "UNVERIFIED" in (api.__doc__ or "")
    _: Callable[..., Any] = api.PolymarketPublicClient  # public names exist
