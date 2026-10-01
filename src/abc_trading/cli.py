"""Command line interface: ``python -m abc_trading <command>`` (or ``abc-trading <command>``).

There is NO live trading anywhere in this project: no order signing, keys, wallets or real
exchange adapter. Every command either runs the simulated paper exchange over synthetic or
recorded events, or (``paper``) reads public market data over HTTP and feeds it to the same
simulated exchange.

Commands
--------
``backtest``  run the strategy on a synthetic feed (``--seed``, ``--windows``, ``--assets``),
              check the DESIGN section 7 invariants, print the report, optionally write a run
              directory (``--out DIR``).
``replay FILE.jsonl``  run the strategy on a recorded event file (see ``record``).
``record --out FILE``  write the synthetic feed to JSONL (plus a ``FILE.meta.json`` sidecar that
              marks the file as synthetic, so ``replay`` labels it correctly).
``paper``     LIVE READ-ONLY public data + paper exchange. Needs network access and the flag
              ``--i-understand-this-is-paper-only``. Endpoints are UNVERIFIED (see
              ``abc_trading.data.public_api``). It can never place a real order.
``config``    print the effective configuration as JSON.

Configuration: ``--config PATH`` (TOML, default: built-in defaults), then the dedicated flags
(``--seed``, ``--windows``, ``--assets``), then every ``--set section.key=value`` in order.
Values are coerced through the config dataclass field types: ``bool`` (true/false/yes/no/on/off/
1/0), ``int``, ``float``, ``float | None`` (``none``), tuples (``a,b``) and dicts (``A=1,B=2``,
merged into the existing mapping). An unknown section or key is an error.

Exit codes: 0 ok, 2 usage / configuration / input-data error, 3 invariant violation, 4 (paper
only) public data unreachable, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
import typing
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from abc_trading.backtest.report import format_report, write_run_dir
from abc_trading.backtest.runner import BacktestResult, InvariantError, run_backtest
from abc_trading.config import BotConfig, config_from_dict, config_to_dict, load_config
from abc_trading.data.events import read_jsonl, write_jsonl
from abc_trading.data.live import DEFAULT_SLUG_TEMPLATE, LiveFeed, record
from abc_trading.data.public_api import (
    DataError,
    PolymarketPublicClient,
    SpotClient,
    Transport,
    UrllibTransport,
)
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.types import FeedEvent

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_INVARIANT = 3
EXIT_UNREACHABLE = 4
EXIT_INTERRUPTED = 130

PAPER_FLAG = "--i-understand-this-is-paper-only"
META_SUFFIX = ".meta.json"


class ConfigError(ValueError):
    """A bad ``--set`` override or configuration (reported with exit code 2)."""


# --------------------------------------------------------------------------- --set overrides

_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


def parse_override(text: str) -> tuple[str, str, str]:
    """Split ``section.key=value`` into ``(section, key, raw value)``; the value may contain '='."""
    path, sep, raw = text.partition("=")
    parts = path.strip().split(".")
    if not sep or len(parts) != 2 or not all(parts):
        raise ConfigError(f"bad --set {text!r}: expected section.key=value")
    return parts[0], parts[1], raw.strip()


def _coerce_float(raw: str, path: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{path}: {raw!r} is not a number") from None
    if not math.isfinite(value):
        raise ConfigError(f"{path}: {raw!r} is not a finite number")
    return value


def _coerce_scalar(tp: Any, raw: str, path: str) -> Any:
    if tp is bool:
        low = raw.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ConfigError(f"{path}: {raw!r} is not a boolean (use true/false)")
    if tp is int:
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(f"{path}: {raw!r} is not an integer") from None
    if tp is float:
        return _coerce_float(raw, path)
    if tp is str:
        return raw
    raise ConfigError(f"{path}: unsupported config type {tp!r}")


def coerce_value(tp: Any, raw: str, path: str) -> Any:
    """Convert the command-line text ``raw`` to the (typing) type ``tp`` of a config field."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        args = typing.get_args(tp)
        inner = [a for a in args if a is not type(None)]
        if len(inner) < len(args) and raw.lower() in ("none", "null"):
            return None
        return coerce_value(inner[0], raw, path)
    if origin is tuple:
        elem = typing.get_args(tp)[0]
        return tuple(coerce_value(elem, x.strip(), path) for x in raw.split(",") if x.strip())
    if origin is dict:
        value_tp = typing.get_args(tp)[1]
        out: dict[str, Any] = {}
        for item in (x for x in raw.split(",") if x.strip()):
            key, sep, val = item.partition("=")
            if not sep or not key.strip():
                raise ConfigError(f"{path}: expected KEY=VALUE pairs separated by commas")
            out[key.strip()] = coerce_value(value_tp, val.strip(), f"{path}.{key.strip()}")
        return out
    return _coerce_scalar(tp, raw, path)


def apply_overrides(cfg: BotConfig, overrides: Iterable[str]) -> BotConfig:
    """Apply ``section.key=value`` overrides in order and re-validate (ValueError if invalid)."""
    data = config_to_dict(cfg)
    sections = typing.get_type_hints(BotConfig)
    for text in overrides:
        section, key, raw = parse_override(text)
        if section not in sections:
            raise ConfigError(f"unknown config section {section!r} (known: {sorted(sections)})")
        fields = typing.get_type_hints(sections[section])
        if key not in fields:
            raise ConfigError(f"unknown key {section}.{key} (known: {sorted(fields)})")
        value = coerce_value(fields[key], raw, f"{section}.{key}")
        if isinstance(value, dict):  # mappings are merged, so one asset can be overridden alone
            value = {**data[section][key], **value}
        data[section][key] = value
    return config_from_dict(data)


def build_config(args: argparse.Namespace) -> BotConfig:
    """Effective config: file (or defaults), then --seed/--windows/--assets, then --set."""
    cfg = load_config(args.config)
    overrides: list[str] = []
    if getattr(args, "seed", None) is not None:
        overrides.append(f"sim.seed={args.seed}")
    if getattr(args, "windows", None) is not None:
        overrides.append(f"sim.n_windows={args.windows}")
    if getattr(args, "sim_assets", None) is not None:
        overrides.append("sim.assets=" + ",".join(args.sim_assets))
    overrides.extend(args.set)
    return apply_overrides(cfg, overrides) if overrides else cfg


# --------------------------------------------------------------------------- parser


def _asset_list(text: str) -> tuple[str, ...]:
    assets = tuple(a.strip().upper() for a in text.split(",") if a.strip())
    if not assets:
        raise argparse.ArgumentTypeError("expected a comma-separated list such as BTC,ETH")
    if len(set(assets)) != len(assets):
        raise argparse.ArgumentTypeError(f"duplicate asset in {text!r}")
    return assets


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {text!r}") from None
    if not (math.isfinite(value) and value > 0.0):
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {text!r}")
    return value


def _add_config_args(p: argparse.ArgumentParser, *, sim: bool) -> None:
    p.add_argument("--config", metavar="PATH", help="TOML config file (default: built-in defaults)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value, repeatable, applied last "
        "(e.g. --set pair.target_margin=0.02)",
    )
    if sim:
        p.add_argument("--seed", type=int, help="synthetic feed seed (sim.seed)")
        p.add_argument("--windows", type=int, help="number of windows per asset (sim.n_windows)")
        p.add_argument(
            "--assets",
            dest="sim_assets",
            type=_asset_list,
            help="assets, e.g. BTC,ETH (sim.assets)",
        )


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--out", metavar="DIR", help="write result.json, config.json, fills.csv, ... here"
    )
    p.add_argument("--no-invariants", action="store_true", help="skip the invariant checks")
    p.add_argument(
        "--sample-every",
        type=int,
        default=60,
        metavar="N",
        help="equity curve sampling interval in events (default 60)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abc-trading",
        description="Backtest / paper-trading framework for a Polymarket Up/Down market maker. "
        "No live trading exists; synthetic results validate mechanics, not profitability.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("backtest", help="run the strategy on a synthetic feed")
    _add_config_args(p, sim=True)
    _add_run_args(p)
    p.set_defaults(handler=_cmd_backtest)

    p = sub.add_parser("replay", help="run the strategy on a recorded JSONL event file")
    p.add_argument("file", metavar="FILE.jsonl")
    _add_config_args(p, sim=False)
    _add_run_args(p)
    p.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="accept a file that ends before every market resolved (truncated recording)",
    )
    p.set_defaults(handler=_cmd_replay)

    p = sub.add_parser("record", help="write the synthetic feed to a JSONL file")
    _add_config_args(p, sim=True)
    p.add_argument("--out", metavar="FILE", required=True, help="output JSONL path")
    p.set_defaults(handler=_cmd_record)

    p = sub.add_parser(
        "paper", help="LIVE READ-ONLY public data + paper exchange (never places real orders)"
    )
    _add_config_args(p, sim=False)
    _add_run_args(p)
    p.add_argument(
        PAPER_FLAG, dest="paper_ack", action="store_true", help="required acknowledgement"
    )
    p.add_argument("--assets", type=_asset_list, help="assets to follow (default: sim.assets)")
    p.add_argument(
        "--window-seconds",
        type=_positive_int,
        help="market window length (default: sim.window_seconds)",
    )
    p.add_argument(
        "--poll-seconds", type=_positive_float, default=1.0, help="polling interval (default 1.0)"
    )
    p.add_argument(
        "--max-events",
        type=_positive_int,
        help="stop after this many events (default: run until Ctrl-C)",
    )
    p.add_argument(
        "--slug-template", default=DEFAULT_SLUG_TEMPLATE, help="market slug template (a guess)"
    )
    p.add_argument("--record", metavar="FILE", help="also record the events to this JSONL file")
    p.set_defaults(handler=_cmd_paper)

    p = sub.add_parser("config", help="print the effective configuration as JSON")
    _add_config_args(p, sim=True)
    p.set_defaults(handler=_cmd_config)
    return parser


# --------------------------------------------------------------------------- commands


def _emit(result: BacktestResult, cfg: BotConfig, out: str | None) -> int:
    print(format_report(result), end="")
    if out:
        write_run_dir(result, cfg, out)
        print(f"run directory written: {out}")
    return EXIT_OK


def _synthetic_label(cfg: BotConfig) -> str:
    s = cfg.sim
    return (
        f"synthetic (seed={s.seed}, assets={','.join(s.assets)}, "
        f"{s.n_windows} x {s.window_seconds}s windows)"
    )


def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    result = run_backtest(
        cfg,
        SyntheticFeed(cfg),
        check_invariants=not args.no_invariants,
        equity_sample_every=args.sample_every,
        source_label=_synthetic_label(cfg),
    )
    return _emit(result, cfg, args.out)


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + META_SUFFIX)


def _is_marked_synthetic(path: Path) -> bool:
    """True iff the ``record`` sidecar next to ``path`` marks it as synthetic."""
    try:
        meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and meta.get("source") == "synthetic"


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    path = Path(args.file)
    label = ("synthetic-replay: " if _is_marked_synthetic(path) else "replay: ") + path.name
    result = run_backtest(
        cfg,
        read_jsonl(path),
        check_invariants=not args.no_invariants,
        equity_sample_every=args.sample_every,
        source_label=label,
        allow_unresolved=args.allow_unresolved,
    )
    return _emit(result, cfg, args.out)


def _cmd_record(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    path = Path(args.out)
    count = write_jsonl(path, SyntheticFeed(cfg))
    meta = {
        "source": "synthetic",
        "generator": "abc_trading.sim.SyntheticFeed",
        "events": count,
        "seed": cfg.sim.seed,
        "assets": list(cfg.sim.assets),
        "n_windows": cfg.sim.n_windows,
        "window_seconds": cfg.sim.window_seconds,
    }
    _meta_path(path).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {count} synthetic events to {path} (sidecar {_meta_path(path).name})")
    return EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    print(json.dumps(config_to_dict(build_config(args)), indent=2, allow_nan=False))
    return EXIT_OK


def _make_transport() -> Transport:
    """The HTTP transport used by ``paper`` (a seam for tests; never used by other commands)."""
    return UrllibTransport()


def _make_live_feed(
    cfg: BotConfig,
    client: PolymarketPublicClient,
    spot: SpotClient,
    *,
    assets: Sequence[str],
    window_seconds: int,
    poll_seconds: float,
    max_events: int | None,
    slug_template: str,
) -> Iterable[FeedEvent]:
    """Build the polling feed for ``paper`` (a seam for tests)."""
    return LiveFeed(
        cfg,
        client,
        spot,
        assets=assets,
        window_seconds=window_seconds,
        slug_template=slug_template,
        poll_seconds=poll_seconds,
        max_events=max_events,
    )


def _until_interrupt(events: Iterable[FeedEvent]) -> Iterator[FeedEvent]:
    """Pass events through; Ctrl-C while waiting for the next one ends the stream cleanly."""
    it = iter(events)
    while True:
        try:
            yield next(it)
        except (StopIteration, KeyboardInterrupt):
            return


def _cmd_paper(args: argparse.Namespace) -> int:
    if not args.paper_ack:
        print(
            f"error: paper mode reads live public data and simulates fills. It never places "
            f"real orders, but you must pass {PAPER_FLAG} to run it.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    cfg = build_config(args)
    assets = args.assets or cfg.sim.assets
    window = args.window_seconds or cfg.sim.window_seconds
    transport = _make_transport()
    client, spot = PolymarketPublicClient(transport), SpotClient(transport)
    # Building the feed only validates the options (no I/O), so a bad option fails before any
    # network access happens.
    feed = _make_live_feed(
        cfg, client, spot, assets=assets, window_seconds=window, poll_seconds=args.poll_seconds,
        max_events=args.max_events, slug_template=args.slug_template,
    )  # fmt: skip
    try:
        for asset in assets:
            spot.spot(asset)
    except DataError as exc:
        print(
            "error: cannot reach the public spot-price endpoints (paper mode needs network "
            f"access to Binance/Coinbase and Polymarket): {exc}",
            file=sys.stderr,
        )
        return EXIT_UNREACHABLE
    events: Iterable[FeedEvent] = record(feed, args.record) if args.record else feed
    print(
        f"paper mode: following {','.join(assets)} ({window}s windows), polling every "
        f"{args.poll_seconds:g}s. Public data only, simulated fills. Ctrl-C to stop.",
        file=sys.stderr,
    )
    result = run_backtest(
        cfg,
        _until_interrupt(events),
        check_invariants=not args.no_invariants,
        equity_sample_every=args.sample_every,
        source_label="paper-live",
        allow_unresolved=True,
    )
    code = _emit(result, cfg, args.out)
    stats = getattr(feed, "stats", None)
    if isinstance(stats, dict):
        print("live feed stats: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    if result.n_events == 0:
        print(
            f"error: no events were produced; last feed error: {getattr(feed, 'last_error', None)}",
            file=sys.stderr,
        )
        return EXIT_UNREACHABLE
    return code


# --------------------------------------------------------------------------- entry point


def _exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return EXIT_USAGE


Handler = Callable[[argparse.Namespace], int]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (see the module docstring)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse: --help (0) or a usage error (2)
        return _exit_code(exc)
    handler: Handler = args.handler
    try:
        return handler(args)
    except InvariantError as exc:
        print(f"INVARIANT VIOLATION: {exc}", file=sys.stderr)
        return EXIT_INVARIANT
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
