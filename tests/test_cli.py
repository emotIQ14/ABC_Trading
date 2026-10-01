"""Tests for abc_trading.cli: subcommands, exit codes, output files, paper-mode guard rails.

Every test is offline: an autouse fixture makes the paper transport factory fail loudly, so a
test that forgets to inject a fake can never touch the network. Runs are short (BTC, 300 s
windows, 1-2 windows).
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from abc_trading import cli
from abc_trading.backtest import runner as runner_mod
from abc_trading.backtest.runner import InvariantError
from abc_trading.cli import main
from abc_trading.config import BotConfig, config_to_dict
from abc_trading.data.events import read_jsonl
from abc_trading.data.public_api import DataError
from abc_trading.exchange.paper import PaperExchange
from abc_trading.sim.feed import SyntheticFeed
from abc_trading.types import FeedEvent

REPO = Path(__file__).resolve().parents[1]
SMALL = ["--assets", "BTC", "--set", "sim.window_seconds=300"]  # 300 snapshots per window
BANNER = "SYNTHETIC DATA - mechanics validation only, not evidence of profitability"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> object:
        raise AssertionError("a test tried to build the real HTTP transport")

    monkeypatch.setattr(cli, "_make_transport", forbidden)


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- backtest


def test_backtest_runs_prints_the_report_and_writes_the_run_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code, stdout, stderr = run_cli(capsys, "backtest", "--windows", "1", *SMALL, "--out", str(out))
    assert (code, stderr) == (0, "")
    assert stdout.startswith("ABC_Trading backtest report\n")
    assert BANNER in stdout.splitlines()[2]  # honesty banner is part of the headline block
    assert f"run directory written: {out}" in stdout
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json", "equity.csv", "fills.csv", "markets.csv", "result.json",
    ]  # fmt: skip
    result = read_json(out / "result.json")
    assert (result["n_events"], result["n_snapshots"], result["n_resolutions"]) == (301, 300, 1)
    assert result["invariants_checked"] is True
    assert result["source_label"] == "synthetic (seed=1, assets=BTC, 1 x 300s windows)"
    cfg = read_json(out / "config.json")
    assert cfg["sim"]["n_windows"] == 1 and cfg["sim"]["assets"] == ["BTC"]
    assert cfg["sim"]["window_seconds"] == 300


def test_backtest_without_out_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    code, stdout, _ = run_cli(capsys, "backtest", "--windows", "1", *SMALL)
    assert code == 0 and "run directory written" not in stdout
    assert list(tmp_path.iterdir()) == []


def test_backtest_is_deterministic_and_seed_sensitive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, seed in (("a", "1"), ("b", "1"), ("c", "2")):
        args = ["backtest", "--windows", "1", *SMALL, "--seed", seed, "--out", str(tmp_path / name)]
        assert run_cli(capsys, *args)[0] == 0
    assert (tmp_path / "a" / "result.json").read_bytes() == (
        tmp_path / "b" / "result.json"
    ).read_bytes()
    assert (tmp_path / "a" / "fills.csv").read_bytes() == (
        tmp_path / "b" / "fills.csv"
    ).read_bytes()
    a, c = read_json(tmp_path / "a" / "result.json"), read_json(tmp_path / "c" / "result.json")
    assert a["fills_sha256"] != c["fills_sha256"]
    assert "seed=2" in c["source_label"]


def test_set_overrides_are_applied_and_win_over_the_dedicated_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code, _, _ = run_cli(
        capsys, "backtest", *SMALL, "--windows", "3", "--seed", "9",
        "--set", "sim.n_windows=1", "--set", "pair.target_margin=0.02",
        "--set", "sizing.clip_equity_fraction=0.05", "--out", str(out),
    )  # fmt: skip
    assert code == 0
    cfg = read_json(out / "config.json")
    assert cfg["sim"]["n_windows"] == 1 and cfg["sim"]["seed"] == 9  # --set beat --windows
    assert cfg["pair"]["target_margin"] == 0.02
    assert cfg["sizing"]["clip_equity_fraction"] == 0.05
    assert read_json(out / "result.json")["n_resolutions"] == 1


def test_config_file_plus_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "c.toml"
    path.write_text(
        '[sim]\nassets = ["BTC"]\nwindow_seconds = 300\nn_windows = 1\n'
        "[pair]\ntarget_margin = 0.02\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"
    code, _, _ = run_cli(
        capsys,
        "backtest",
        "--config",
        str(path),
        "--set",
        "pair.merge_min_pairs=20",
        "--out",
        str(out),
    )
    assert code == 0
    cfg = read_json(out / "config.json")
    assert cfg["pair"]["target_margin"] == 0.02 and cfg["pair"]["merge_min_pairs"] == 20.0
    assert read_json(out / "result.json")["n_events"] == 301


def test_no_invariants_flag_and_sample_every(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code, stdout, _ = run_cli(
        capsys, "backtest", "--windows", "1", *SMALL, "--no-invariants", "--sample-every", "10",
        "--out", str(out),
    )  # fmt: skip
    assert code == 0 and "NOT checked" in stdout
    result = read_json(out / "result.json")
    assert result["invariants_checked"] is False and result["equity_sample_every"] == 10
    # first event, every 10th of 301 events (10 .. 300) and the last event
    assert [p[0] for p in result["equity_curve"]] == [1, *range(10, 301, 10), 301]


@pytest.mark.parametrize("assets", ["BTC,ETH", "btc, eth"])
def test_assets_flag_accepts_lists_and_normalises_case(
    assets: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code, _, _ = run_cli(
        capsys, "backtest", "--windows", "1", "--assets", assets,
        "--set", "sim.window_seconds=300", "--out", str(out),
    )  # fmt: skip
    assert code == 0
    assert read_json(out / "config.json")["sim"]["assets"] == ["BTC", "ETH"]
    assert read_json(out / "result.json")["n_resolutions"] == 2


# --------------------------------------------------------------------------- exit code 2


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["backtest", "--set", "pair.bogus=1"], "unknown key pair.bogus"),
        (["backtest", "--set", "bogus.x=1"], "unknown config section 'bogus'"),
        (["backtest", "--set", "pair.target_margin=abc"], "not a number"),
        (["backtest", "--set", "pair.target_margin=nan"], "not a finite number"),
        (["backtest", "--set", "pair.target_margin=0.9"], "pair.target_margin must be in"),
        (["backtest", "--set", "sizing.ladder_levels=1.5"], "not an integer"),
        (["backtest", "--set", "directional.enabled=maybe"], "not a boolean"),
        (["backtest", "--set", "nokey"], "expected section.key=value"),
        (["backtest", "--config", "/nonexistent/c.toml"], "No such file"),
        (["backtest", "--windows", "0"], "sim.n_windows must be >= 1"),
        (["backtest", "--assets", "SOL"], "missing spot0/annual_vol for SOL"),
        (["backtest", "--sample-every", "0"], "equity_sample_every"),
        (["backtest", "--windows", "x"], "invalid int value"),
        (["backtest", "--assets", ""], "comma-separated list"),
        (["backtest", "--assets", "BTC,BTC"], "duplicate asset"),
        (["backtest", "--seed", "1.5"], "invalid int value"),
        (["config", "--set", "pair.bogus=1"], "unknown key pair.bogus"),
        (["record"], "--out"),
        (["replay"], "FILE.jsonl"),
        (["frobnicate"], "invalid choice"),
        ([], "required"),
        (["backtest", "--no-such-flag"], "unrecognized arguments"),
    ],
)
def test_usage_and_config_errors_exit_2_with_a_message_on_stderr(
    argv: list[str], fragment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, stdout, stderr = run_cli(capsys, *argv)
    assert code == 2
    assert stdout == ""
    assert fragment in stderr


def test_bad_toml_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[pair\n", encoding="utf-8")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text("[pair]\nbogus = 1\n", encoding="utf-8")
    assert run_cli(capsys, "backtest", "--config", str(broken))[0] == 2
    code, _, stderr = run_cli(capsys, "config", "--config", str(unknown))
    assert code == 2 and "unknown key(s) ['bogus']" in stderr


def test_unwritable_output_directory_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    code, _, stderr = run_cli(
        capsys, "backtest", "--windows", "1", *SMALL, "--out", str(blocker / "run")
    )
    assert code == 2 and stderr.startswith("error:")


@pytest.mark.parametrize("command", [None, "backtest", "replay", "record", "paper", "config"])
def test_help_exits_0(command: str | None, capsys: pytest.CaptureFixture[str]) -> None:
    argv = [command, "--help"] if command else ["--help"]
    code, stdout, _ = run_cli(capsys, *[a for a in argv if a])
    assert code == 0 and stdout.startswith("usage:")


# --------------------------------------------------------------------------- exit code 3


def test_invariant_error_exits_3(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise InvariantError("invariant 1 (reconciliation) violated at event #7: boom")

    monkeypatch.setattr(cli, "run_backtest", explode)
    code, stdout, stderr = run_cli(capsys, "backtest", "--windows", "1", *SMALL)
    assert code == 3 and stdout == ""
    assert (
        stderr == "INVARIANT VIOLATION: invariant 1 (reconciliation) violated at event #7: boom\n"
    )


def test_a_real_accounting_bug_is_caught_and_exits_3(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class LeakyExchange(PaperExchange):
        def balance(self) -> float:  # reports 1 cent more cash than it holds
            return super().balance() + 0.01

    monkeypatch.setattr(runner_mod, "PaperExchange", LeakyExchange)
    code, _, stderr = run_cli(capsys, "backtest", "--windows", "1", *SMALL)
    assert code == 3
    assert "INVARIANT VIOLATION: invariant 1 (reconciliation)" in stderr
    assert "engine cash 10000 != exchange cash 10000.01" in stderr
    # the same bug goes unnoticed (by design) when the checks are switched off
    assert run_cli(capsys, "backtest", "--windows", "1", *SMALL, "--no-invariants")[0] == 0


def test_keyboard_interrupt_exits_130(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_backtest", interrupted)
    code, _, stderr = run_cli(capsys, "backtest", "--windows", "1", *SMALL)
    assert (code, stderr) == (130, "interrupted\n")


# --------------------------------------------------------------------------- record + replay


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One recorded synthetic file (1 window of 300 s = 301 events) shared by the replay tests."""
    path = tmp_path_factory.mktemp("recorded") / "events.jsonl"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["record", "--windows", "1", *SMALL, "--out", str(path)])
    assert code == 0
    assert f"wrote 301 synthetic events to {path}" in buffer.getvalue()
    return path


def head_of(path: Path, tmp_path: Path, n: int, name: str = "head.jsonl") -> Path:
    """The first ``n`` events of a recording as a new file (no sidecar)."""
    lines = path.read_text(encoding="utf-8").splitlines()[:n]
    out = tmp_path / name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def test_record_writes_jsonl_and_a_synthetic_sidecar(recorded: Path) -> None:
    lines = recorded.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 301  # 300 snapshots + 1 resolution
    assert json.loads(lines[0])["type"] == "snapshot"
    assert json.loads(lines[-1])["type"] == "resolved"
    meta = read_json(Path(str(recorded) + ".meta.json"))
    assert meta["source"] == "synthetic" and meta["events"] == 301 and meta["seed"] == 1
    assert meta["assets"] == ["BTC"] and meta["n_windows"] == 1 and meta["window_seconds"] == 300


def test_record_output_matches_the_synthetic_feed(recorded: Path) -> None:
    cfg = cli.build_config(cli.build_parser().parse_args(["backtest", "--windows", "1", *SMALL]))
    assert list(read_jsonl(recorded)) == list(SyntheticFeed(cfg))


def test_replay_of_a_recorded_file_reproduces_the_backtest_exactly(
    recorded: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_cli(capsys, "backtest", "--windows", "1", *SMALL, "--out", str(tmp_path / "direct"))
    code, stdout, _ = run_cli(capsys, "replay", str(recorded), "--out", str(tmp_path / "replayed"))
    assert code == 0
    assert BANNER in stdout.splitlines()[2]  # the sidecar marks the file as synthetic
    direct = read_json(tmp_path / "direct" / "result.json")
    replayed = read_json(tmp_path / "replayed" / "result.json")
    assert replayed["source_label"] == "synthetic-replay: events.jsonl"
    assert direct["activity"]["n_fills"] > 0  # not a vacuous comparison
    direct.pop("source_label")
    replayed.pop("source_label")
    assert replayed == direct  # DESIGN invariant 5 through the CLI
    assert (tmp_path / "direct" / "fills.csv").read_bytes() == (
        tmp_path / "replayed" / "fills.csv"
    ).read_bytes()


def test_replay_without_a_sidecar_is_not_labelled_synthetic(
    recorded: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = tmp_path / "plain.jsonl"
    plain.write_bytes(recorded.read_bytes())  # same events, no .meta.json next to it
    code, stdout, _ = run_cli(capsys, "replay", str(plain))
    assert code == 0
    assert "SYNTHETIC DATA" not in stdout
    assert "replay: plain.jsonl" in stdout and "Source not marked synthetic" in stdout


@pytest.mark.parametrize("sidecar", ["{not json", "[]", '{"source": "real"}', ""])
def test_unreadable_or_foreign_sidecars_are_ignored(
    sidecar: str, recorded: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = head_of(recorded, tmp_path, 30)
    Path(str(path) + ".meta.json").write_text(sidecar, encoding="utf-8")
    code, stdout, _ = run_cli(capsys, "replay", str(path), "--allow-unresolved")
    assert code == 0 and "SYNTHETIC DATA" not in stdout
    # while a valid sidecar next to the very same file does switch the banner on
    Path(str(path) + ".meta.json").write_text('{"source": "synthetic"}', encoding="utf-8")
    code, stdout, _ = run_cli(capsys, "replay", str(path), "--allow-unresolved")
    assert code == 0 and BANNER in stdout


def test_replay_applies_config_overrides(
    recorded: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = head_of(recorded, tmp_path, 30)
    out = tmp_path / "run"
    code, _, _ = run_cli(
        capsys, "replay", str(path), "--allow-unresolved",
        "--set", "exchange.latency_ticks=0", "--out", str(out),
    )  # fmt: skip
    assert code == 0
    assert read_json(out / "config.json")["exchange"]["latency_ticks"] == 0


def test_replay_of_a_truncated_recording_is_invariant_4_unless_allowed(
    recorded: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cut = head_of(recorded, tmp_path, 300)  # all snapshots, but the resolution is missing
    code, _, stderr = run_cli(capsys, "replay", str(cut))
    assert code == 3 and "invariant 4" in stderr and "never resolved" in stderr
    code, stdout, _ = run_cli(capsys, "replay", str(cut), "--allow-unresolved")
    assert code == 0 and re.search(r"Unresolved markets +1\n", stdout)


def test_replay_input_errors_exit_2(
    recorded: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, stderr = run_cli(capsys, "replay", str(tmp_path / "missing.jsonl"))
    assert code == 2 and "No such file" in stderr
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type": "snapshot"}\n', encoding="utf-8")
    code, _, stderr = run_cli(capsys, "replay", str(bad))
    assert code == 2 and "bad.jsonl: line 1" in stderr
    # events out of time order are rejected, not silently skipped
    lines = recorded.read_text(encoding="utf-8").splitlines()
    swapped = tmp_path / "swapped.jsonl"
    swapped.write_text("\n".join([lines[5], lines[0], *lines[1:5]]) + "\n", encoding="utf-8")
    code, _, stderr = run_cli(capsys, "replay", str(swapped))
    assert code == 2 and "goes backwards" in stderr


def test_replay_of_an_empty_file_is_a_clean_empty_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    code, stdout, _ = run_cli(capsys, "replay", str(empty))
    assert code == 0 and "Events" in stdout and "0 snapshots" in stdout


# --------------------------------------------------------------------------- config command


def test_config_command_prints_the_effective_config_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, stdout, stderr = run_cli(capsys, "config")
    assert (code, stderr) == (0, "")
    assert json.loads(stdout) == config_to_dict(BotConfig())


def test_config_command_applies_flags_and_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, _ = run_cli(
        capsys, "config", "--seed", "5", "--windows", "2", "--assets", "BTC",
        "--set", "pair.target_margin=0.02", "--set", "sim.spot0=BTC=61000",
    )  # fmt: skip
    assert code == 0
    cfg = json.loads(stdout)
    assert (
        cfg["sim"]["seed"] == 5 and cfg["sim"]["n_windows"] == 2 and cfg["sim"]["assets"] == ["BTC"]
    )
    assert cfg["pair"]["target_margin"] == 0.02
    assert cfg["sim"]["spot0"] == {"BTC": 61_000.0, "ETH": 3_000.0}


# --------------------------------------------------------------------------- paper mode


class FakeSpotTransport:
    """Answers every GET like a Binance ticker; never touches the network."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def get_json(
        self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
    ) -> Any:
        self.urls.append(url)
        return {"price": "60000.5"}


class DownTransport:
    def get_json(
        self, url: str, params: Mapping[str, str] | None = None, timeout: float = 10.0
    ) -> Any:
        raise DataError("network down")


class StubFeed:
    """Stands in for LiveFeed: replays canned events and mimics its stats/last_error."""

    def __init__(self, events: Sequence[FeedEvent], *, interrupt_after: int | None = None) -> None:
        self._events, self._interrupt_after = list(events), interrupt_after
        self.stats = {"ticks": len(self._events), "errors": 0}
        self.last_error: str | None = None

    def __iter__(self) -> Iterator[FeedEvent]:
        for i, event in enumerate(self._events):
            if self._interrupt_after is not None and i == self._interrupt_after:
                raise KeyboardInterrupt
            yield event


def synthetic_events(n: int) -> list[FeedEvent]:
    cfg = cli.build_config(cli.build_parser().parse_args(["backtest", "--windows", "2", *SMALL]))
    return list(SyntheticFeed(cfg))[:n]


def patch_paper(monkeypatch: pytest.MonkeyPatch, feed: StubFeed) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def make_feed(cfg: BotConfig, client: object, spot: object, **kwargs: Any) -> StubFeed:
        seen.update(kwargs)
        return feed

    transport = FakeSpotTransport()
    seen["transport"] = transport
    monkeypatch.setattr(cli, "_make_transport", lambda: transport)
    monkeypatch.setattr(cli, "_make_live_feed", make_feed)
    return seen


def test_paper_requires_the_acknowledgement_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, stderr = run_cli(
        capsys, "paper"
    )  # the autouse fixture would blow up on a network try
    assert code == 2 and stdout == ""
    assert "--i-understand-this-is-paper-only" in stderr
    assert "never places real orders" in stderr


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (["--poll-seconds", "0"], "must be a finite number > 0"),
        (["--poll-seconds", "abc"], "invalid float value"),
        (["--window-seconds", "0"], "must be >= 1"),
        (["--max-events", "0"], "must be >= 1"),
        (["--max-events", "x"], "invalid int value"),
        (["--slug-template", "{nope}"], "bad slug_template"),
        (["--assets", "BTC,BTC"], "duplicate asset"),
    ],
)
def test_paper_option_errors_exit_2_before_any_network_access(
    args: list[str],
    fragment: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeSpotTransport()
    monkeypatch.setattr(cli, "_make_transport", lambda: transport)
    code, stdout, stderr = run_cli(capsys, "paper", "--i-understand-this-is-paper-only", *args)
    assert (code, stdout) == (2, "")
    assert fragment in stderr
    assert transport.urls == []  # nothing was requested


def test_paper_reports_unreachable_data_clearly(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_make_transport", lambda: DownTransport())
    code, stdout, stderr = run_cli(capsys, "paper", "--i-understand-this-is-paper-only")
    assert code == 4 and stdout == ""
    assert "cannot reach the public spot-price endpoints" in stderr
    assert "network access" in stderr and "network down" in stderr


def test_paper_runs_the_feed_through_the_paper_exchange(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    events = synthetic_events(120)  # 120 snapshots of a window that never resolves
    seen = patch_paper(monkeypatch, StubFeed(events))
    out, rec = tmp_path / "run", tmp_path / "tee.jsonl"
    code, stdout, stderr = run_cli(
        capsys, "paper", "--i-understand-this-is-paper-only", "--assets", "BTC",
        "--window-seconds", "300", "--poll-seconds", "2", "--max-events", "120",
        "--out", str(out), "--record", str(rec),
    )  # fmt: skip
    assert code == 0
    assert stdout.splitlines()[2].startswith("PAPER MODE - live public data with SIMULATED fills")
    assert "SYNTHETIC DATA" not in stdout
    assert re.search(r"Source +paper-live\n", stdout)
    assert "live feed stats: ticks=120, errors=0" in stdout
    assert "Public data only, simulated fills" in stderr
    # the CLI wired its options through to the feed factory
    assert seen["assets"] == ("BTC",) and seen["window_seconds"] == 300
    assert seen["poll_seconds"] == 2.0 and seen["max_events"] == 120
    assert seen["slug_template"] == "{asset_lower}-updown-{minutes}m-{start_ts}"
    assert any("binance" in url for url in seen["transport"].urls)  # the connectivity pre-check
    result = read_json(out / "result.json")
    assert result["source_label"] == "paper-live"
    assert result["n_events"] == 120 and len(result["unresolved_markets"]) == 1
    assert len(rec.read_text(encoding="utf-8").splitlines()) == 120  # --record tees the events


def test_paper_defaults_come_from_the_config(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = patch_paper(monkeypatch, StubFeed(synthetic_events(5)))
    code, _, _ = run_cli(capsys, "paper", "--i-understand-this-is-paper-only")
    assert code == 0
    assert seen["assets"] == ("BTC", "ETH") and seen["window_seconds"] == 900
    assert seen["poll_seconds"] == 1.0 and seen["max_events"] is None


def test_paper_stops_cleanly_on_ctrl_c_while_waiting_for_data(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_paper(monkeypatch, StubFeed(synthetic_events(50), interrupt_after=3))
    code, stdout, _ = run_cli(capsys, "paper", "--i-understand-this-is-paper-only")
    assert code == 0
    assert "3 snapshots" in stdout  # the report covers what was seen before the interrupt


def test_paper_with_no_events_exits_4_and_shows_the_last_feed_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    feed = StubFeed([])
    feed.last_error = "book: HTTP 503"
    patch_paper(monkeypatch, feed)
    code, _, stderr = run_cli(capsys, "paper", "--i-understand-this-is-paper-only")
    assert code == 4
    assert "no events were produced; last feed error: book: HTTP 503" in stderr


def test_until_interrupt_passes_events_and_swallows_ctrl_c() -> None:
    assert list(cli._until_interrupt([1, 2, 3])) == [1, 2, 3]  # type: ignore[list-item]
    assert list(cli._until_interrupt(StubFeed(synthetic_events(10), interrupt_after=4))) == (
        synthetic_events(4)
    )


# --------------------------------------------------------------------------- entry points


def test_main_reads_sys_argv_by_default(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["abc-trading", "config"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["sim"]["seed"] == 1


def test_module_entry_point_runs_as_a_subprocess() -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "abc_trading", "backtest", "--windows", "1", "--assets", "BTC",
         "--set", "sim.window_seconds=300"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert BANNER in proc.stdout
    bad = subprocess.run(
        [sys.executable, "-m", "abc_trading", "backtest", "--set", "pair.bogus=1"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert bad.returncode == 2 and "unknown key" in bad.stderr


def test_importing_the_main_module_does_not_run_the_cli() -> None:
    module = importlib.import_module("abc_trading.__main__")
    assert module.main is main


def test_pyproject_script_target_exists() -> None:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'abc-trading = "abc_trading.cli:main"' in text


# --------------------------------------------------------------------------- no live trading


def test_package_imports_only_the_standard_library() -> None:
    """Standard library only, hence no exchange client, signing or wallet library can be in use."""
    allowed = set(sys.stdlib_module_names) | {"abc_trading"}
    offenders: list[str] = []
    for path in (REPO / "src" / "abc_trading").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] not in allowed]
    assert offenders == []


def test_cli_has_no_order_placement_path_other_than_the_paper_exchange() -> None:
    tree = ast.parse((REPO / "src" / "abc_trading" / "cli.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    exchange_modules = {m for m in imported if m.startswith("abc_trading.exchange")}
    assert exchange_modules == set()  # the CLI reaches the exchange only through run_backtest
    assert PaperExchange.__module__ == "abc_trading.exchange.paper"
