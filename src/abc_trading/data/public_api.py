"""READ-ONLY public-data clients (stdlib ``urllib`` only): Polymarket and spot prices.

*** UNVERIFIED: EVERY ENDPOINT PATH, QUERY PARAMETER AND RESPONSE FIELD NAME IN THIS MODULE
*** WAS WRITTEN FROM MEMORY OF THE PUBLIC DOCUMENTATION. The build sandbox cannot reach any of
*** these hosts, so nothing here has ever been run against a live endpoint. The parsers are
*** unit-tested only against SYNTHETIC fixtures that this project made up in the shapes it
*** expects (``tests/fixtures/synthetic_*.json``). Expect to adjust the constants below, and
*** treat every number this module returns as unvalidated until compared with the website.

Scope: HTTP GET of public data only. There is no authentication, no API key, no order signing
and no wallet anywhere in this package. The network is optional and never used by the tests;
they inject a fake ``Transport``.

Design
------
* ``Transport`` is the single seam to the network. ``UrllibTransport`` implements it with
  ``urllib`` (honouring the standard ``HTTP(S)_PROXY`` / ``NO_PROXY`` environment variables,
  because ``urllib.request.build_opener`` installs a ``ProxyHandler``).
* Every endpoint string lives in a module constant (``*_BASE_URL``, ``*_PATH``, ``*_PARAM``)
  and every response field name in a ``_*_KEYS`` tuple of accepted spellings, so the guesses
  are easy to find and fix.
* Parsing is defensive: any unexpected shape raises ``DataError`` with a short message;
  nothing is silently defaulted. Prices/sizes are floats (USDC per share / shares, as in
  ``types.py``); timestamps are unix seconds.
* Clients hold only small caches / counters; they never touch a clock or the disk.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, NotRequired, Protocol, TypedDict

from abc_trading.types import BookSnapshot, Level, Outcome, Side, Trade

# --------------------------------------------------------------------------- UNVERIFIED constants

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DATA_BASE_URL = "https://data-api.polymarket.com"

GAMMA_MARKETS_PATH = "/markets"  # GET ?slug=<slug>  -> list of market objects
GAMMA_SLUG_PARAM = "slug"
CLOB_BOOK_PATH = "/book"  # GET ?token_id=<id> -> {"bids": [{"price","size"}], "asks": [...]}
CLOB_TOKEN_PARAM = "token_id"
DATA_TRADES_PATH = "/trades"  # GET ?market=<conditionId>&limit=<n> -> list of trade objects
DATA_MARKET_PARAM = "market"
DATA_LIMIT_PARAM = "limit"
DEFAULT_TRADE_LIMIT = 500

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TICKER_PATH = "/api/v3/ticker/price"  # GET ?symbol=BTCUSDT -> {"price": "..."}
BINANCE_SYMBOL_PARAM = "symbol"
COINBASE_BASE_URL = "https://api.coinbase.com"
COINBASE_SPOT_PATH = "/v2/prices/{product}/spot"  # -> {"data": {"amount": "..."}}

DEFAULT_USER_AGENT = "abc-trading/0.1 (read-only research client)"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SEEN_TRADES = 20_000  # bound on the cross-call trade de-duplication memory

# Accepted spellings of response fields (first present, non-null key wins).
_CONDITION_ID_KEYS = ("conditionId", "condition_id")
_TOKEN_IDS_KEYS = ("clobTokenIds", "clob_token_ids")
_OUTCOMES_KEYS = ("outcomes",)
_MARKET_START_KEYS = ("eventStartTime", "startDate")  # window start, else market creation
_MARKET_END_KEYS = ("endDate",)
_TICK_SIZE_KEYS = ("orderPriceMinTickSize",)
_MIN_ORDER_SIZE_KEYS = ("orderMinSize",)
_MARKET_SLUG_KEYS = ("slug",)
_BOOK_BIDS_KEYS = ("bids",)
_BOOK_ASKS_KEYS = ("asks",)
_TRADE_TS_KEYS = ("timestamp",)
_TRADE_SIDE_KEYS = ("side",)
_TRADE_ASSET_KEYS = ("asset", "asset_id")
_TRADE_PRICE_KEYS = ("price",)
_TRADE_SIZE_KEYS = ("size",)
_TRADE_OUTCOME_KEYS = ("outcome",)
_TRADE_TX_KEYS = ("transactionHash", "transaction_hash")
_TRADE_CONDITION_KEYS = ("conditionId", "condition_id")

_SYMBOL_RE = re.compile(r"[A-Za-z0-9]{1,15}")
_PRODUCT_RE = re.compile(r"[A-Za-z0-9]{1,15}-[A-Za-z0-9]{1,15}")
_MS_THRESHOLD = 1e11  # a "seconds" timestamp this large would be the year 5138: it is millis


class DataError(Exception):
    """A network, HTTP, JSON or response-shape problem. The message is short and safe to log."""


# --------------------------------------------------------------------------- transport


class Transport(Protocol):
    """GET a URL and return the decoded JSON body; raise ``DataError`` on any failure."""

    def get_json(
        self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
    ) -> Any: ...


class _Opener(Protocol):
    def open(self, fullurl: urllib.request.Request, /, *, timeout: float) -> Any: ...


def _short(text: object, limit: int = 120) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _where(url: str) -> str:
    """``url`` without query string / fragment (what error messages show)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _with_params(url: str, params: Mapping[str, str] | None) -> str:
    if not params:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + urllib.parse.urlencode(dict(params))


class UrllibTransport:
    """``Transport`` over ``urllib``. Only http(s) URLs are fetched (never ``file:`` etc.).

    ``opener`` is injectable for tests; the default ``build_opener()`` reads the proxy
    environment variables when the transport is constructed. Redirects are followed, the body
    is capped at ``max_bytes``, and every failure becomes a short ``DataError``.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        *,
        opener: _Opener | None = None,
        max_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes!r}")
        self._user_agent = user_agent
        self._opener: _Opener = opener if opener is not None else urllib.request.build_opener()
        self._max_bytes = max_bytes

    def get_json(
        self,
        url: str,
        params: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Any:
        if not timeout > 0:
            raise ValueError(f"timeout must be > 0, got {timeout!r}")
        full = _with_params(url, params)
        where = _where(full)
        if urllib.parse.urlsplit(full).scheme not in ("http", "https"):
            raise DataError(f"refusing non-http(s) URL: {where}")
        request = urllib.request.Request(
            full,
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
            method="GET",
        )
        body = self._fetch(request, where, timeout)
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError:  # JSONDecodeError and UnicodeDecodeError
            raise DataError(f"invalid JSON from {where}") from None

    def _fetch(self, request: urllib.request.Request, where: str, timeout: float) -> bytes:
        try:
            with self._opener.open(request, timeout=timeout) as resp:
                body: bytes = resp.read(self._max_bytes + 1)
        except urllib.error.HTTPError as exc:
            exc.close()
            raise DataError(f"HTTP {exc.code} from {where}") from None
        except urllib.error.URLError as exc:
            raise DataError(f"network error from {where}: {_short(exc.reason)}") from None
        except TimeoutError:
            raise DataError(f"timeout after {timeout:g}s from {where}") from None
        except (OSError, http.client.HTTPException, ValueError) as exc:  # incl. InvalidURL
            raise DataError(f"connection error from {where}: {_short(exc)}") from None
        if len(body) > self._max_bytes:
            raise DataError(f"response from {where} exceeds {self._max_bytes} bytes")
        return body


# --------------------------------------------------------------------------- parsing helpers


def _first(obj: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return None


def _to_float(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise DataError(f"{what}: expected a number, got {_short(repr(value))}")
    try:
        x = float(value)
    except (ValueError, OverflowError):
        raise DataError(f"{what}: not a number: {_short(repr(value))}") from None
    if not math.isfinite(x):
        raise DataError(f"{what}: number must be finite, got {_short(repr(value))}")
    return x


def _to_price(value: Any, what: str) -> float:
    x = _to_float(value, what)
    if not 0.0 <= x <= 1.0:
        raise DataError(f"{what}: price must be in [0, 1], got {x!r}")
    return x


def _as_list(value: Any, what: str) -> list[Any]:
    """A list that may arrive JSON-encoded inside a string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise DataError(f"{what}: string is not valid JSON") from None
    if not isinstance(value, list):
        raise DataError(f"{what}: expected a list, got {type(value).__name__}")
    return value


def _token_id(value: Any, what: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise DataError(f"{what}: expected a token id string, got {_short(repr(value))}")
    text = str(value).strip()
    if not text:
        raise DataError(f"{what}: empty token id")
    return text


def _to_seconds(value: Any, what: str) -> float:
    """A unix timestamp in seconds or milliseconds, or an ISO-8601 string (naive = UTC)."""
    if isinstance(value, str) and not _looks_numeric(value):
        try:
            return _iso_to_seconds(value)
        except ValueError:
            raise DataError(f"{what}: unparseable time {_short(repr(value))}") from None
    x = _to_float(value, what)
    return x / 1000.0 if x > _MS_THRESHOLD else x


def _looks_numeric(text: str) -> bool:
    return re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", text) is not None


def _iso_to_seconds(text: str) -> float:
    dt = datetime.fromisoformat(text.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _check_url_base(base: str) -> str:
    if not base.startswith(("http://", "https://")):
        raise ValueError(f"base URL must start with http:// or https://, got {base!r}")
    return base.rstrip("/")


def _check_timeout(timeout: float) -> float:
    if not (math.isfinite(timeout) and timeout > 0):
        raise ValueError(f"timeout must be a finite number > 0, got {timeout!r}")
    return timeout


# --------------------------------------------------------------------------- Polymarket


class MarketInfo(TypedDict):
    """Parsed Gamma market. Times are unix seconds; optional keys appear only if present."""

    slug: str
    condition_id: str
    up_token_id: str
    down_token_id: str
    start_ts: NotRequired[float]
    end_ts: NotRequired[float]
    tick_size: NotRequired[float]
    min_order_size: NotRequired[float]


def _select_market(payload: Any, slug: str) -> Mapping[str, Any]:
    """The market object for ``slug`` from a list (or a single object) response."""
    items = [payload] if isinstance(payload, Mapping) else payload
    if not isinstance(items, list):
        raise DataError(f"markets response: expected a list, got {type(payload).__name__}")
    candidates = [
        m for m in items if isinstance(m, Mapping) and _first(m, _MARKET_SLUG_KEYS) in (None, slug)
    ]
    if not candidates:
        raise DataError(f"market not found for slug {slug!r}")
    return candidates[0]


def _parse_token_ids(market: Mapping[str, Any]) -> tuple[str, str]:
    """(up_token_id, down_token_id), mapping ids to outcomes by label (case-insensitive)."""
    outcomes = _as_list(_first(market, _OUTCOMES_KEYS), "outcomes")
    ids = _as_list(_first(market, _TOKEN_IDS_KEYS), "clobTokenIds")
    if len(outcomes) != len(ids):
        raise DataError(f"outcomes ({len(outcomes)}) and clobTokenIds ({len(ids)}) differ")
    by_label: dict[str, str] = {}
    for i, (label, raw_id) in enumerate(zip(outcomes, ids, strict=True)):
        if not isinstance(label, str):
            raise DataError(f"outcomes[{i}]: expected a string, got {type(label).__name__}")
        key = label.strip().casefold()
        if key in by_label:
            raise DataError(f"outcomes: duplicate label {label!r}")
        by_label[key] = _token_id(raw_id, f"clobTokenIds[{i}]")
    up, down = by_label.get("up"), by_label.get("down")
    if up is None or down is None:
        raise DataError(f"outcomes {outcomes!r} do not contain both Up and Down")
    if up == down:
        raise DataError("Up and Down map to the same token id")
    return up, down


def _parse_market(market: Mapping[str, Any], slug: str) -> MarketInfo:
    condition_id = _first(market, _CONDITION_ID_KEYS)
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise DataError("market has no conditionId")
    up, down = _parse_token_ids(market)
    info: MarketInfo = {
        "slug": slug,
        "condition_id": condition_id.strip(),
        "up_token_id": up,
        "down_token_id": down,
    }
    start = _optional(market, _MARKET_START_KEYS, _to_seconds)
    end = _optional(market, _MARKET_END_KEYS, _to_seconds)
    tick = _optional(market, _TICK_SIZE_KEYS, _to_float, positive=True)
    min_size = _optional(market, _MIN_ORDER_SIZE_KEYS, _to_float, positive=True)
    if start is not None:
        info["start_ts"] = start
    if end is not None:
        info["end_ts"] = end
    if tick is not None:
        info["tick_size"] = tick
    if min_size is not None:
        info["min_order_size"] = min_size
    return info


def _optional(
    market: Mapping[str, Any],
    keys: tuple[str, ...],
    parse: Callable[[Any, str], float],
    *,
    positive: bool = False,
) -> float | None:
    """An optional numeric field: None if absent/null/empty, DataError if present but bad."""
    raw = _first(market, keys)
    if raw is None or raw == "":
        return None
    value = parse(raw, keys[0])
    if positive and value <= 0:
        raise DataError(f"{keys[0]}: must be > 0, got {value!r}")
    return value


def _parse_levels(raw: Any, what: str) -> list[Level]:
    """Levels with positive size, equal prices merged; unsorted."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DataError(f"{what}: expected a list, got {type(raw).__name__}")
    merged: dict[float, float] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping) or "price" not in item or "size" not in item:
            raise DataError(f"{what}[{i}]: expected an object with price and size")
        price = _to_price(item["price"], f"{what}[{i}].price")
        size = _to_float(item["size"], f"{what}[{i}].size")
        if size > 0:
            merged[price] = merged.get(price, 0.0) + size
    return [Level(p, s) for p, s in merged.items()]


def _parse_book(payload: Any, token: Outcome) -> BookSnapshot:
    if not isinstance(payload, Mapping):
        raise DataError(f"book response: expected an object, got {type(payload).__name__}")
    raw_bids, raw_asks = _first(payload, _BOOK_BIDS_KEYS), _first(payload, _BOOK_ASKS_KEYS)
    if not any(k in payload for k in _BOOK_BIDS_KEYS + _BOOK_ASKS_KEYS):
        raise DataError("book response has neither bids nor asks")
    bids = sorted(_parse_levels(raw_bids, "bids"), key=lambda lv: -lv.price)
    asks = sorted(_parse_levels(raw_asks, "asks"), key=lambda lv: lv.price)
    return BookSnapshot(token=token, bids=tuple(bids), asks=tuple(asks))


def _outcome_from_label(label: Any) -> Outcome | None:
    if not isinstance(label, str):
        return None
    return {"up": Outcome.UP, "down": Outcome.DOWN}.get(label.strip().casefold())


def _mirror(trade: Trade) -> Trade:
    """The economically identical print on the other token (see ``types.Trade``)."""
    return Trade(
        ts=trade.ts,
        token=trade.token.opposite,
        price=round(1.0 - trade.price, 6),
        size=trade.size,
        aggressor=trade.aggressor.opposite,
    )


class PolymarketPublicClient:
    """Read-only Polymarket public data (UNVERIFIED endpoints; see the module docstring).

    ``stats`` counts trade records dropped as malformed / mismatching and duplicates, so a
    shape drift in the (best-effort) trades endpoint is visible rather than silent.
    """

    def __init__(
        self,
        transport: Transport,
        gamma_base: str = GAMMA_BASE_URL,
        clob_base: str = CLOB_BASE_URL,
        data_base: str = DATA_BASE_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._transport = transport
        self._gamma = _check_url_base(gamma_base)
        self._clob = _check_url_base(clob_base)
        self._data = _check_url_base(data_base)
        self._timeout = _check_timeout(timeout)
        self._tokens: dict[str, tuple[str, str]] = {}
        self._seen: OrderedDict[tuple[Any, ...], None] = OrderedDict()
        self.stats: dict[str, int] = {"trades_skipped": 0, "trades_duplicate": 0}

    def market_by_slug(self, slug: str) -> MarketInfo:
        """Look a market up by slug; remembers its token ids for ``recent_trades``."""
        if not slug:
            raise ValueError("slug must be non-empty")
        payload = self._transport.get_json(
            self._gamma + GAMMA_MARKETS_PATH, {GAMMA_SLUG_PARAM: slug}, self._timeout
        )
        info = _parse_market(_select_market(payload, slug), slug)
        self._tokens[info["condition_id"]] = (info["up_token_id"], info["down_token_id"])
        return info

    def book(self, token_id: str, token: Outcome) -> BookSnapshot:
        """Order book of one token: bids high->low, asks low->high, non-positive sizes dropped."""
        if not token_id:
            raise ValueError("token_id must be non-empty")
        payload = self._transport.get_json(
            self._clob + CLOB_BOOK_PATH, {CLOB_TOKEN_PARAM: token_id}, self._timeout
        )
        return _parse_book(payload, token)

    def recent_trades(
        self,
        condition_id: str,
        since_ts: float,
        *,
        up_token_id: str | None = None,
        down_token_id: str | None = None,
        limit: int = DEFAULT_TRADE_LIMIT,
    ) -> list[Trade]:
        """Public trades of a market with timestamp >= floor(``since_ts``), oldest first.

        BEST EFFORT. Each real trade is returned once per client instance (identity: tx hash,
        token, side, price, size, timestamp) and is emitted as BOTH mirrored prints
        (``types.Trade``). The lower bound is floored because the endpoint is assumed to
        report whole seconds; re-reported trades are removed by the de-duplication memory.
        The reported ``side`` is assumed to be the taker's side (UNVERIFIED). Token ids default
        to those remembered from ``market_by_slug``; failing that the record's ``outcome``
        label (Up/Down) is used. Malformed or foreign records are skipped and counted.
        """
        if not condition_id:
            raise ValueError("condition_id must be non-empty")
        if not math.isfinite(since_ts):
            raise ValueError(f"since_ts must be finite, got {since_ts!r}")
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit!r}")
        if bool(up_token_id) != bool(down_token_id):
            raise ValueError("pass both up_token_id and down_token_id, or neither")
        tokens = self._tokens.get(condition_id)
        if up_token_id and down_token_id:
            tokens = (up_token_id, down_token_id)
        payload = self._transport.get_json(
            self._data + DATA_TRADES_PATH,
            {DATA_MARKET_PARAM: condition_id, DATA_LIMIT_PARAM: str(limit)},
            self._timeout,
        )
        if not isinstance(payload, list):
            raise DataError(f"trades response: expected a list, got {type(payload).__name__}")
        lower = math.floor(since_ts)
        out: list[Trade] = []
        for record in payload:
            try:
                key, trade = self._parse_trade(record, condition_id, tokens)
            except DataError:
                self.stats["trades_skipped"] += 1
                continue
            if trade.ts < lower:
                continue
            if self._remember(key):
                self.stats["trades_duplicate"] += 1
                continue
            out.append(trade)
        out.sort(key=lambda t: t.ts)  # stable: keeps each real trade next to its mirror
        return [p for t in out for p in (t, _mirror(t))]

    def _remember(self, key: tuple[Any, ...]) -> bool:
        """Record ``key``; True if it was already known."""
        if key in self._seen:
            return True
        self._seen[key] = None
        if len(self._seen) > MAX_SEEN_TRADES:
            self._seen.popitem(last=False)
        return False

    @staticmethod
    def _parse_trade(
        record: Any, condition_id: str, tokens: tuple[str, str] | None
    ) -> tuple[tuple[Any, ...], Trade]:
        """One real trade (not its mirror) plus its de-duplication key; DataError if unusable."""
        if not isinstance(record, Mapping):
            raise DataError("trade record is not an object")
        cond = _first(record, _TRADE_CONDITION_KEYS)
        if cond is not None and cond != condition_id:
            raise DataError("trade belongs to another market")
        side_raw = _first(record, _TRADE_SIDE_KEYS)
        try:
            side = Side(str(side_raw).strip().upper())
        except ValueError:
            raise DataError(f"unknown trade side {_short(repr(side_raw))}") from None
        asset = _first(record, _TRADE_ASSET_KEYS)
        token: Outcome | None = None
        if tokens is not None and asset is not None:
            token = {tokens[0]: Outcome.UP, tokens[1]: Outcome.DOWN}.get(str(asset))
        if token is None:
            token = _outcome_from_label(_first(record, _TRADE_OUTCOME_KEYS))
        if token is None:
            raise DataError("cannot map trade to Up/Down")
        price = _to_price(_first(record, _TRADE_PRICE_KEYS), "trade price")
        size = _to_float(_first(record, _TRADE_SIZE_KEYS), "trade size")
        if size <= 0:
            raise DataError("trade size must be > 0")
        ts = _to_seconds(_first(record, _TRADE_TS_KEYS), "trade timestamp")
        tx = _first(record, _TRADE_TX_KEYS)
        key = (condition_id, tx, token, side, price, size, ts)
        return key, Trade(ts=ts, token=token, price=price, size=size, aggressor=side)


# --------------------------------------------------------------------------- spot prices


def _price_from(value: Any, what: str) -> float:
    x = _to_float(value, what)
    if x <= 0:
        raise DataError(f"{what}: price must be > 0, got {x!r}")
    return x


class SpotClient:
    """Read-only spot prices from Binance and Coinbase (UNVERIFIED endpoints).

    Note: the real markets resolve against their own price source, which may differ from these
    exchanges; spot here is an approximation used for the model and for paper-mode resolution.
    ``last_source`` names the venue that served the latest successful ``spot()`` call.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        binance_base: str = BINANCE_BASE_URL,
        coinbase_base: str = COINBASE_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._transport = transport
        self._binance = _check_url_base(binance_base)
        self._coinbase = _check_url_base(coinbase_base)
        self._timeout = _check_timeout(timeout)
        self.last_source: str | None = None

    def binance_price(self, symbol: str) -> float:
        """Last price for a Binance symbol such as ``BTCUSDT``."""
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError(f"invalid Binance symbol {symbol!r}")
        payload = self._transport.get_json(
            self._binance + BINANCE_TICKER_PATH, {BINANCE_SYMBOL_PARAM: symbol}, self._timeout
        )
        if not isinstance(payload, Mapping) or "price" not in payload:
            raise DataError("binance response: expected an object with 'price'")
        return _price_from(payload["price"], "binance price")

    def coinbase_price(self, product: str) -> float:
        """Spot price for a Coinbase product such as ``BTC-USD``."""
        if not _PRODUCT_RE.fullmatch(product):
            raise ValueError(f"invalid Coinbase product {product!r}")
        payload = self._transport.get_json(
            self._coinbase + COINBASE_SPOT_PATH.format(product=product), None, self._timeout
        )
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, Mapping) or "amount" not in data:
            raise DataError("coinbase response: expected {'data': {'amount': ...}}")
        return _price_from(data["amount"], "coinbase price")

    def spot(self, asset: str) -> float:
        """Spot for ``asset`` ("BTC", "ETH", ...): Binance (``<ASSET>USDT``) then Coinbase
        (``<ASSET>-USD``). DataError if both fail; ValueError for a malformed asset name."""
        name = asset.strip().upper()
        if not _SYMBOL_RE.fullmatch(name):
            raise ValueError(f"invalid asset {asset!r}")
        try:
            price = self.binance_price(name + "USDT")
            self.last_source = "binance"
            return price
        except DataError as first:
            try:
                price = self.coinbase_price(f"{name}-USD")
            except DataError as second:
                self.last_source = None
                raise DataError(
                    f"spot unavailable for {name}: binance: {first}; coinbase: {second}"
                ) from None
            self.last_source = "coinbase"
            return price
