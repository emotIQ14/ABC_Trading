"""Optional edge layer: event (de)serialisation and READ-ONLY public data clients.

Nothing in the core modules (types, config, fees, inventory, model, strategy, exchange,
backtest) may import this package; it is used only by the CLI. There is no live trading code
here: only JSONL files, HTTP GETs of public data, and a paper-mode polling feed. The network
endpoints in ``public_api`` are unverified (see its docstring).
"""

from abc_trading.data.events import (
    SCHEMA_VERSION,
    event_from_dict,
    event_from_json,
    event_to_dict,
    event_to_json,
    read_jsonl,
    write_jsonl,
)
from abc_trading.data.live import LiveFeed, record
from abc_trading.data.public_api import (
    DataError,
    MarketInfo,
    PolymarketPublicClient,
    SpotClient,
    Transport,
    UrllibTransport,
)

__all__ = [
    "SCHEMA_VERSION",
    "DataError",
    "LiveFeed",
    "MarketInfo",
    "PolymarketPublicClient",
    "SpotClient",
    "Transport",
    "UrllibTransport",
    "event_from_dict",
    "event_from_json",
    "event_to_dict",
    "event_to_json",
    "read_jsonl",
    "record",
    "write_jsonl",
]
