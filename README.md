# ABC_Trading

A backtest and paper-trading framework for a **two-sided market-making + full-pair-accumulation +
inventory-rebalancing** strategy on Polymarket's short-dated BTC and ETH **Up/Down** markets.
Python 3.11, standard library only at runtime, fully typed, deterministic.

> ## STATUS & HONESTY
>
> * **Backtest and paper simulation only.** There is **no live trading code** in this project: no
>   order signing, no private keys, no wallets, no real exchange adapter. The `Exchange` protocol
>   has exactly one implementation, the simulated `PaperExchange`. Adding a live one is out of
>   scope on purpose (see [What going live would require](#what-going-live-would-require-and-why-it-is-not-here)).
> * **Synthetic results prove mechanics, not profit.** The bundled simulator contains a deliberate
>   2-second market lag so that the directional model has something to find, so any gain from the
>   model on that feed is circular by construction. On that simulator the default configuration
>   has a *negative* mean PnL (see [Sample results](#sample-results-synthetic-data)); that is not
>   evidence about real markets either.
> * **The fee model is a placeholder.** The taker curve and the zero maker fee / rebate are not
>   Polymarket's verified fee schedule. Verify them before trusting any PnL number.
> * **The read-only data clients are unverified.** Every endpoint path, query parameter and
>   response field name in `data/public_api.py` was written from memory and has never been run
>   against a live endpoint (the build sandbox cannot reach those hosts). They are unit-tested
>   only against fixtures this project made up. `paper` mode has never run successfully.
> * **Defaults are uncalibrated.** Model weights, strategy thresholds and the fill model are
>   illustrative starting points, not tuned or validated values.
> * **The source claim below is unverified.**
> * **Nothing here is financial advice.** It is research code. Do not use it to risk money.

## Contents

1. [What this is](#what-this-is)
2. [How the strategy works](#how-the-strategy-works)
3. [Quick start](#quick-start)
4. [CLI reference](#cli-reference)
5. [Configuration reference](#configuration-reference)
6. [Architecture and module map](#architecture-and-module-map)
7. [The fill model, and where it is pessimistic or optimistic](#the-fill-model-and-where-it-is-pessimistic-or-optimistic)
8. [Sample results (synthetic data)](#sample-results-synthetic-data)
9. [Known limitations and risks](#known-limitations-and-risks)
10. [What going live would require, and why it is not here](#what-going-live-would-require-and-why-it-is-not-here)
11. [Testing and invariants](#testing-and-invariants)
12. [Contributing conventions](#contributing-conventions)

Deeper documents: [`docs/DESIGN.md`](docs/DESIGN.md) is the module contract (the spec this code
implements), and [`docs/RESULTS.md`](docs/RESULTS.md) records what the end-to-end integration pass
observed, including anomalies and their causes.

---

## What this is

A simulator, a strategy engine, a paper exchange with a deliberately pessimistic queue and latency
model and a backtest harness, built to
study the *mechanics* of one particular market-making idea:

* a synthetic **feed** of Up/Down window markets (BTC and ETH, 5 or 15 minute windows, correlated
  spot paths, mirrored UP/DOWN order books, public trade prints),
* a **strategy engine** that rests bids on both tokens, accumulates UP+DOWN pairs, merges them,
  rebalances unpaired inventory and keeps a small, model-driven directional position,
* a **paper exchange** that decides which of the engine's orders would have filled, with latency,
  queue position and no look-ahead,
* a **backtest runner** that drives the two together, checks accounting invariants after every
  event, and prints a report (optionally writing JSON and CSV files),
* an optional, off-by-default, **read-only** paper mode that polls public data and feeds it to
  the same simulated exchange.

### The source claim (UNVERIFIED)

The project was started from a reported claim, not from verified data:

> A Polymarket account **reportedly** made **$129,923** at about **$53 per trade** on BTC/ETH
> Up/Down markets, using bidirectional market making, full-pair accumulation, inventory
> rebalancing and a fair-probability model (spot data, order-book depth, momentum, acceleration,
> volatility) to keep part of the directional exposure.

Treat every figure as **unverified**. Web-search snippets of that account's profile showed
*different* totals at another time, so neither the $129,923 nor the $53 can be relied on. This
repository does not reproduce or confirm the claim: doing so would need real fill data, which it
does not have. The only place the $53 appears in code is a label in the report that prints the
simulated average trade next to it for comparison. The default clip (100 shares, about $50 at 50c)
is of a similar order, but no parameter was tuned to hit the figure, and the simulated average
trade is roughly 37 USDC.

---

## How the strategy works

### Plain-language version

Each window market asks one question: will the underlying finish the window at or above where it
started? It has two tokens, **UP** and **DOWN**. At resolution the winner pays $1 and the loser
pays $0 (ties resolve UP in this project). One UP share plus one DOWN share is a **pair**, and a
pair is worth exactly $1 *whichever way the market resolves*. A pair can also be **merged** back
into $1 at any time before resolution, which frees the cash.

So if you can buy one UP and one DOWN for less than $1 in total, you lock in the difference.

1. **Quote both sides.** The engine rests post-only bids on UP and on DOWN, as a small ladder
   (3 levels by default, each smaller than the last). It never rests asks; exits are taker orders.
   When cash (or the order-rate limit) is scarce, the token that completes pairs is funded first and
   the rest is shared evenly level by level, so neither the UP nor the DOWN label is favoured.
2. **Cap what it will pay.** When flat, each bid is capped at the model's fair probability minus
   half the target margin, so the two caps sum to `1 - target_margin`. Once it holds unpaired
   shares of one token, the bid on the other token is capped at `1 - target_margin - avg_cost`
   of what it already holds, so completing the pair locks at least the target margin.
3. **Accumulate pairs and merge.** When both legs have filled, the pair's profit is locked.
   Pairs are merged in batches (`pair.merge_min_pairs`), turning them back into cash, which is
   then reused for new bids.
4. **Rebalance inventory.** Bids on the heavy side are shifted down and then stopped as the
   imbalance grows. If the unpaired amount gets large, the engine may buy the missing leg as a
   taker, but only if that still locks a net profit after fees (with default settings).
5. **Keep a little direction, only when the model sees edge.** A fair-probability model
   estimates P(UP) from spot versus the window's reference price, realised volatility and time
   left (a driftless random-walk model), nudged by momentum, acceleration and order-book
   imbalance, then shrunk halfway toward the market's own price. If its probability beats the
   price by `directional.min_edge`, the engine tolerates a small, Kelly-sized net position
   (at most `directional.max_directional_shares`, 200 by default).
6. **Run on a clock.** Each window moves through phases: WARMUP (no quotes for 15 s),
   ACCUMULATE, WIND_DOWN (last 90 s: only add the side that completes pairs, or the
   model-favoured side), FLATTEN (last 25 s: cancel quotes, merge every pair, then either hold
   the unpaired remainder to resolution if the model justifies it, or sell it at the bid) and
   DONE.
7. **Stay inside risk limits.** A kill switch, stale-spot and wide-book gates, per-market and
   total capital caps and an order-rate limiter can each stop quoting.

### Worked example: 47c + 52c = 99c

Suppose the UP book is 0.47 bid / 0.48 ask. The DOWN book is its mirror, 0.52 bid / 0.53 ask
(`down_bid = 1 - up_ask`). Resting a bid on each token at the touch means paying the passive
side on both legs:

| | price | shares | cost |
|---|---:|---:|---:|
| UP bid fills | 0.47 | 100 | $47.00 |
| DOWN bid fills | 0.52 | 100 | $52.00 |
| **Total for 100 pairs** | **0.99 per pair** | | **$99.00** |
| Merge (or resolution) pays | $1.00 per pair | | $100.00 |
| **Locked gross profit** | **0.01 per pair** | | **$1.00** |

That is a 1.01% return on $99 of capital, and it is locked in regardless of whether UP or DOWN
wins, *provided both legs fill* and the maker fee is zero (the default placeholder). The code
reproduces these numbers: `MarketInventory.locked_profit()` returns 1.0 and
`capital_at_risk()` returns 99.0 for exactly these two fills.

### Where it breaks

The margin is one cent per pair. Everything that follows is about how easily that cent is lost.

* **Leg risk.** Suppose the UP bid fills (100 shares, $47) and the DOWN bid never does. If UP
  wins you collect $100 (+$53); if DOWN wins you hold worthless shares (-$47). One unhedged
  share that expires at zero costs 47 cents, which is the margin of 47 completed pairs.
* **Adverse selection.** A resting bid is most likely to be hit when the price is moving through
  it, that is, just before it becomes a bad price. The leg that fills tends to be the one the
  market is leaving, and the opposite bid is now further from the market and may never fill. The
  synthetic simulator builds this in on purpose (informed sweeps), and the result is visible in
  the sample run: merges earned +1,723.26 USDC while selling the unpaired remainder at the end of
  windows lost -1,385.95 USDC.
* **Fees.** Completing the missing leg by crossing the spread costs the *ask*, not the bid, and
  pays the taker fee on top. In the example above the DOWN book mirrors the UP book, so once the
  UP bid has filled at 0.47 the DOWN ask is 0.53: buying DOWN there makes the pair cost
  0.47 + 0.53 = 1.00 before any fee, so the spread alone takes the whole 1 cent margin. With the
  *placeholder* curve the taker fee on top is about 0.82 cents per share (0.8222 USDC for 100
  shares at 0.53; about 0.81 cents at 0.52 and 0.73 cents at 0.47), so completing that pair as a
  taker would lock a *loss* of about 0.8 cents per pair. Even a missing leg offered at 0.52 would
  leave only 0.19 cents of the 1 cent after the fee (100 shares at 0.52 pay 0.8099 USDC). Fee
  schedules also change over time. The default engine therefore completes pairs as a taker only
  when the *net* margin after the ask and the fee is still at least `pair.taker_lock_margin`
  (0.5 cent): with the UP leg at 0.47 that needs a DOWN ask of 0.51 or lower, so it refuses the
  0.53 completion above.

---

## Quick start

Requirements: Python 3.11 or newer. There are no runtime dependencies.

### 1. Make the package importable

From the repository root, either put `src` on the path:

```bash
export PYTHONPATH=src
```

or install it editable into a virtual environment (this also provides the `abc-trading` command
and, with `[dev]`, pytest, ruff and mypy):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

`python3 -m pytest` works without either, because `pyproject.toml` already sets
`pythonpath = ["src"]` for pytest.

### 2. Run the tests

```bash
python3 -m pytest
```

At the time of writing this collects 2,076 tests and takes about 30 seconds. Everything runs
offline.

### 3. Run a backtest

```bash
python3 -m abc_trading backtest --seed 1 --windows 12
```

(or `abc-trading backtest --seed 1 --windows 12` after an editable install). This simulates 12
back-to-back 15-minute windows for each of BTC and ETH (24 markets, 21,600 snapshots), checks the
accounting invariants after every event and prints a report. It takes a few seconds. The first
part of the real output:

```text
ABC_Trading backtest report
===========================
SYNTHETIC DATA - mechanics validation only, not evidence of profitability

Run
---
  Source              synthetic (seed=1, assets=BTC,ETH, 12 x 900s windows)
  Events              21,624 (21,600 snapshots, 24 resolutions)
  Time span (unix s)  1,700,000,100 .. 1,700,010,900
  Invariants          checked (DESIGN section 7, 1-4)
  Unresolved markets  none

Result
------
  Initial cash  10,000.00
  Final equity  10,338.98
  Total PnL     +338.98  (+3.39%)
  Max drawdown  440.56  (4.27% of peak; sampled every 60 events)
  Kill switch   not tripped

Per-market PnL
--------------
  Markets resolved                   24 of 24 (24 traded)
  Mean / std per market              +14.12 / 98.37
  Sharpe-like (mean/std*sqrt(n))     +0.70
  Best / worst market                +177.38 / -175.95
  Win rate (traded markets)          50.0%
  PnL from merges                    +1,723.26
  PnL from sells (flatten)           -1,385.95
  PnL at settlement, pairs           +0.79
  PnL at settlement, directional     +0.88
  Unrealised, open inventory at mid  +0.00
  Fees paid (inside the above)       135.30
...
```

The bare `python3 -m abc_trading backtest` uses the default of 8 windows, so its numbers differ
from the 12-window runs shown here and in the sample results.

### 4. Read the output

Every report opens with a banner that states what the numbers are: `SYNTHETIC DATA - mechanics
validation only` for simulated feeds, a paper-mode notice for `paper`, and a generic caveat for
any other source. Then:

| Section | What to look at |
|---|---|
| **Run** | Source label, event counts, whether invariants were checked, unresolved markets. |
| **Result** | Final equity, total PnL, max drawdown (measured on a mark-to-mid equity curve sampled every `--sample-every` events, so event-level drawdown can be larger), and whether the kill switch tripped. |
| **Per-market PnL** | Mean and spread per market (resolved markets only), plus where the PnL came from: merges, flatten sells, settlement of pairs, settlement of directional leftovers and, for markets still open, the unrealised mark-to-mid value of their inventory. These five rows cover every market and add up to Total PnL; fees are already inside them. |
| **Trading activity** | Fill counts, maker versus taker volume, average and median trade size in USDC. |
| **Pairing and inventory** | *Pair completion rate* is the share of bought shares that ended in a pair. *Avg all-in cost per merged pair* below 1.0 is the margin actually earned. *Markets ending unhedged* counts markets that closed with at least 1 unpaired share (for markets that have not resolved: at the end of the stream). |
| **Model calibration** | Brier score of the engine's probability, the raw model and the market's own UP mid, scored at each market's last ACCUMULATE snapshot. Lower is better; always saying 0.5 scores 0.25. |
| **Engine / Exchange / Runner counters** | Quotes placed and cancelled, post-only cancels, merges, rate-limited placements, risk trips, order rejections. |
| **Markets** | One row per market (first 40 shown; the rest are in `markets.csv`). |

A run that ends with **unresolved markets** (every `paper` run that stops inside a window, and
`replay --allow-unresolved` on a truncated recording) is reported differently, and the report says
so:

* *Invariants* reads `checked 1-3; 4 (final accounting) skipped: N unresolved markets`. Invariant 4
  (every market settled, nothing open, the PnL breakdown equals final equity) cannot hold while
  markets are open, so it is not checked.
* Final equity and Total PnL value the open inventory at the last snapshot mids, and *Of which
  unrealised* in the Result block names that part. It is a mark, not realised money.
* In *Per-market PnL*, the mean, spread, Sharpe-like, best / worst and win-rate rows cover the
  resolved markets only (an open market has only realised part of its PnL so far; *Statistics
  cover* says how many). The merge, sell, settlement and fee rows cover every market, with an
  *Unrealised* row so that they add up to Total PnL, and *Markets resolved* reads `N of M (K
  traded)` with K counted among the resolved markets.

To keep the data, add `--out DIR`. This writes `result.json`, `config.json`, `fills.csv`,
`markets.csv` and `equity.csv` (UTF-8, deterministic, full float precision):

```bash
python3 -m abc_trading backtest --seed 1 --windows 2 --out runs/demo
```

### 5. Record a feed and replay it

```bash
python3 -m abc_trading record --seed 1 --windows 2 --out runs/feed.jsonl
python3 -m abc_trading replay runs/feed.jsonl
```

`record` writes the event file plus a `feed.jsonl.meta.json` sidecar that marks it as synthetic,
so `replay` labels the report `synthetic-replay: feed.jsonl` and prints the synthetic banner.
Without the sidecar, `replay` labels the source `replay: <name>` and prints the generic caveat
instead. Replaying a recording gives exactly the same numbers and fills as running the simulation
directly (a tested invariant; only the source label differs).

---

## CLI reference

Run as `python3 -m abc_trading <command>` (with `PYTHONPATH=src`) or `abc-trading <command>`
(after `pip install -e .`).

| Command | What it does |
|---|---|
| `backtest` | Run the strategy on a synthetic feed, check the invariants, print the report. |
| `replay FILE.jsonl` | Run the strategy on a recorded event file. |
| `record --out FILE` | Write the synthetic feed to JSONL, plus a `FILE.meta.json` sidecar. |
| `paper` | **Read-only public data + simulated exchange.** Never places a real order. Unverified endpoints; see below. |
| `config` | Print the effective configuration as JSON. |

### Configuration flags (all commands)

| Flag | Meaning |
|---|---|
| `--config PATH` | TOML config file. Default: the built-in defaults. |
| `--set SECTION.KEY=VALUE` | Override one value. Repeatable. Applied last, in order. |
| `--seed N` | Synthetic feed seed (`sim.seed`). `backtest`, `record`, `config`. |
| `--windows N` | Windows per asset (`sim.n_windows`). `backtest`, `record`, `config`. |
| `--assets A,B` | Assets to simulate, e.g. `BTC` or `BTC,ETH` (`sim.assets`). `backtest`, `record`, `config`. For `paper` it selects the assets to follow. |

Precedence is: the config file (or defaults), then `--seed` / `--windows` / `--assets`, then each
`--set` in order. The result is validated, and an unknown section or key is an error. `--set`
values are coerced using the config field's type: booleans (`true`/`false`/`yes`/`no`/`on`/
`off`/`1`/`0`), ints, floats, optional values (`none`), tuples (`BTC,ETH`) and mappings
(`BTC=55000,ETH=2800`, merged into the existing mapping so one asset can be changed alone).

Values in a TOML file are **not** coerced: each must have the declared type, or the command stops
with exit code 2 and a message naming the key (for example `pair.merge_enabled: expected a
boolean, got str 'false'`). A quoted `"false"` is an error (it would otherwise be a truthy
string), so is `3.0` for an integer key such as `sizing.ladder_levels` or `sim.n_windows`, and so
is `nan` / `inf`. The only conversions are widening an integer to a float (`clip_shares = 50`)
and a list to a tuple (`assets = ["BTC"]`). A loss limit that can never fire
(`risk.max_daily_loss_usd` at or above `exchange.initial_cash`) is legal, since it is how a run
switches the kill switch off, but the CLI prints a `warning:` line on stderr for it.

```bash
python3 -m abc_trading backtest --seed 3 --windows 12 --set pair.target_margin=0.02
python3 -m abc_trading backtest --set directional.enabled=false --set exchange.latency_ticks=2
python3 -m abc_trading config --set sim.spot0=BTC=55000 --set sizing.clip_equity_fraction=none
python3 -m abc_trading backtest --config configs/default.toml
```

`configs/default.toml` lists every key with its default value; loading it gives exactly
`BotConfig()` (a test enforces this). The one omitted key is `sizing.clip_equity_fraction`, whose
default `None` cannot be written in TOML.

### Flags for `backtest`, `replay` and `paper`

| Flag | Meaning |
|---|---|
| `--out DIR` | Write `result.json`, `config.json`, `fills.csv`, `markets.csv`, `equity.csv` into `DIR`. |
| `--no-invariants` | Skip the invariant checks (faster; the report then says `NOT checked`). |
| `--sample-every N` | Equity-curve sampling interval in events (default 60). |
| `--allow-unresolved` | `replay` only. Accept a file that ends before every market resolved (a truncated recording). Without it, such a file is an invariant-4 violation. |

### `paper`

```bash
python3 -m abc_trading paper --i-understand-this-is-paper-only --max-events 600
```

Polls public Polymarket and spot-price (Binance, Coinbase) data and feeds it to the *simulated*
exchange. It requires `--i-understand-this-is-paper-only`; without it the command refuses to run
(exit 2). The flag must be typed in full: no option of any command can be abbreviated, so `paper
--i` is a usage error (exit 2) rather than an acknowledgement. Extra flags: `--assets`, `--window-seconds`, `--poll-seconds` (default 1.0),
`--max-events` (default: until Ctrl-C), `--slug-template` (a guess, default
`{asset_lower}-updown-{minutes}m-{start_ts}`) and `--record FILE` (also tee the events to JSONL).

**Unverified.** The endpoints, field names, market slug scheme, "price to beat" rule and winner
inference are all guesses (see `data/live.py` and `data/public_api.py`), and this mode has never
been run against the real services. In the build sandbox the hosts are unreachable, so the only
behaviour observed is the failure path: exit code 4 with a message saying the public spot
endpoints cannot be reached. The command's own output carries the same caveat, so a run that did
work would not look verified: the `--help` text, the stderr notice, the report banner and the
report's caveats all say the data client is unverified and that the market winners are inferred
from Binance/Coinbase spot prices rather than read from Polymarket's resolution. A window that has
not closed when you stop stays unresolved (see above), and its PnL is a mark-to-mid.

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success. |
| 2 | Usage, configuration or input-data error (including `paper` without its acknowledgement flag, and a missing replay file). |
| 3 | An accounting invariant was violated, or a replay file ended with unresolved markets and no `--allow-unresolved`. |
| 4 | `paper` only: public data unreachable or unusable (the start-up spot-price probe failed or returned something undecodable, or the feed produced no events). |
| 130 | Interrupted (Ctrl-C). |

---

## Configuration reference

Everything lives in one frozen `BotConfig` (`src/abc_trading/config.py`), made of nine sections.
**All defaults are illustrative starting points. None has been calibrated against real Polymarket
data.** Prices are USDC per share in [0, 1], sizes are shares, one tick is 0.01 by default. The
table lists the knobs that matter most; `python3 -m abc_trading config` prints all of them and
`configs/default.toml` documents each one.

| Key | Default | Meaning |
|---|---:|---|
| `fees.taker_fee_rate` | 0.25 | Taker fee = `size * price * rate * (price * (1 - price)) ** exponent`. **Placeholder; verify.** |
| `fees.taker_fee_exponent` | 2.0 | Steepness of the taker curve (larger = concentrated near 50c). |
| `fees.maker_fee_rate` / `fees.maker_rebate_rate` | 0.0 / 0.0 | Fraction of notional charged to / paid back to makers. **Placeholder; verify.** |
| `sizing.clip_shares` | 100.0 | Base size of the first ladder level (about $50 at 50c). |
| `sizing.clip_equity_fraction` | `None` | If set, size clips as `equity * fraction / price` instead of a fixed clip. |
| `sizing.ladder_levels` | 3 | Resting bids per token. |
| `sizing.ladder_spacing_ticks` | 1 | Ticks between ladder levels. |
| `sizing.ladder_size_decay` | 0.7 | Size multiplier per deeper level. |
| `sizing.requote_tolerance_ticks` | 0 | Keep a resting bid if it is at most this many ticks *below* the desired price, never above it (preserves queue position). |
| `pair.target_margin` | 0.01 | Wanted `avg_up_cost + avg_down_cost <= 1 - target_margin`. |
| `pair.taker_lock_margin` | 0.005 | Minimum **net** margin to complete a pair by crossing the spread. |
| `pair.rebalance_max_loss_per_pair` | 0.0 | Extra loss tolerated when completing a pair (0 = only ever at a net profit). |
| `pair.max_net_imbalance_shares` | 300.0 | Cap on `|UP - DOWN|` in hedged-neutral mode. |
| `pair.max_inventory_per_side_shares` | 1500.0 | Cap on shares held per token. |
| `pair.skew_ticks_per_100_shares` | 1.0 | How far bids on the heavy side are pushed down per 100 shares of imbalance. |
| `pair.rebalance_trigger_shares` | 100.0 | Unpaired quantity that triggers taker completion. |
| `pair.merge_enabled` / `pair.merge_min_pairs` | true / 50.0 | Merge pairs back to USDC, in batches of at least this many (in FLATTEN any whole pair). |
| `directional.enabled` | true | Allow a small model-driven net position. |
| `directional.min_edge` | 0.03 | Model probability minus price paid needed to add or hold directional shares. |
| `directional.max_directional_shares` | 200.0 | Maximum deliberate `|UP - DOWN|` per market. |
| `directional.kelly_fraction` | 0.25 | Fraction of full binary Kelly used for the directional target. |
| `directional.hold_margin` | 0.02 | At FLATTEN, hold unpaired shares to resolution only if `p_model - exit_price` is at least this. |
| `timing.warmup_seconds` | 15.0 | No quotes this long after the window starts. |
| `timing.wind_down_seconds` | 90.0 | Before the end: only add the side that completes pairs (or is model-favoured). |
| `timing.flatten_seconds` | 25.0 | Before the end: cancel quotes, merge, resolve the remainder. |
| `timing.min_requote_interval_seconds` | 1.0 | Do not re-quote a market more often than this. |
| `model.shrink_to_market` | 0.5 | 0 = pure model, 1 = pure market mid. |
| `model.w_momentum` / `w_accel` / `w_book_imbalance` | 0.15 / 0.05 / 0.10 | Weights added to the z-score. **Uncalibrated heuristics.** |
| `model.vol_halflife_seconds` | 300.0 | EWMA half-life of the realised-volatility estimate. |
| `model.p_floor` | 0.02 | Model probability is clipped to `[p_floor, 1 - p_floor]`. |
| `risk.max_capital_per_market_usd` | 1500.0 | Cost basis plus resting bids, per market. |
| `risk.max_total_capital_usd` | 5000.0 | The same, across all markets. |
| `risk.max_daily_loss_usd` | 1000.0 | Kill switch, latched for the rest of the run (see the limitations). A limit at or above `exchange.initial_cash` can never fire: that switches it off, and the CLI warns. |
| `risk.max_spot_staleness_seconds` | 5.0 | No quotes when the spot reading is older than this (or missing). |
| `risk.max_orders_per_second` | 20.0 | Order-rate limiter; cancels count toward it. |
| `risk.max_spread_ticks_to_quote` | 6 | Do not quote dislocated or wide books. |
| `exchange.initial_cash` | 10000.0 | Starting USDC of the paper account. |
| `exchange.latency_ticks` | 1 | Snapshots before a new order can fill, and before a cancel takes effect. |
| `exchange.queue_ahead_fraction` | 1.0 | Share of the displayed size at our price assumed to be ahead of us. |
| `exchange.trade_fill_fraction` | 1.0 | Share of a print assumed available to us after the queue. |
| `exchange.taker_slippage_ticks` | 0 | Ticks an IOC fill is moved against us from the displayed level, **never beyond the order's limit**. The engine limits its IOC orders at the touch it saw, so at latency 0 this changes nothing; at latency 1 or more it only moves a fill from the fresh level up to the stale limit. |
| `exchange.enforce_min_order_size` | true | Reject orders below the market's minimum size. |
| `sim.seed` | 1 | RNG seed. Same seed and config give an identical feed. |
| `sim.assets` | `["BTC", "ETH"]` | Underlyings to simulate. |
| `sim.window_seconds` / `sim.n_windows` | 900 / 8 | Window length and windows per asset. |
| `sim.market_lag_seconds` | 2.0 | The simulated market prices off a spot this many seconds stale. **This is what gives the model something to find.** |
| `sim.informed_flow` | 1.0 | 0 disables the sweeps that follow price moves, which removes adverse selection. |
| `sim.uninformed_trades_per_sec` | 0.3 | Random touch prints per second per token side. |
| `sim.pricing_noise_ticks` | 0.7 | AR(1) noise on the market's UP mid, in ticks. |
| `sim.tick_size` / `sim.min_order_size` | 0.01 / 5.0 | Price increment and minimum order size (shares). |
| `sim.spot0`, `sim.annual_vol`, `sim.mean_spread_ticks`, `sim.depth_mean_shares`, `sim.n_levels`, `sim.market_vol_multiplier`, `sim.tick_seconds` | see `configs/default.toml` | Remaining feed parameters. |

`BotConfig.validate()` rejects nonsensical values, naming the key, for example a target margin
outside `[0, 0.5)` or `flatten_seconds` greater than `wind_down_seconds`, and the config
round-trips through a dict, JSON and TOML. It is the single gate for every command and repeats the
checks that the feed, the model, the paper exchange, the risk manager and the fee model make when
they are built, so a bad value fails at once with exit code 2, whichever command you run. It is
also stricter where those components would accept a setting that silently switches a control off
or inverts it: a negative `sizing.requote_tolerance_ticks`, `pair.skew_ticks_per_100_shares`,
`risk.max_spot_staleness_seconds` or `timing.min_requote_interval_seconds`, a
`risk.max_spread_ticks_to_quote` below 1, a fee or rebate rate above 1, a window that leaves no
ACCUMULATE phase after `warmup_seconds + wind_down_seconds`, and a `sim.tick_size` that does not
divide 1 into whole ticks. Every number must be finite.

---

## Architecture and module map

```text
 event sources (each yields the same FeedEvent stream)
 +----------------+  +------------------+  +--------------------------+
 | SyntheticFeed  |  | JSONL file       |  | LiveFeed (paper mode)    |
 | sim/feed.py    |  | data/events.py   |  | data/live.py             |
 | seeded, local  |  | record / replay  |  | data/public_api.py       |
 +-------+--------+  +--------+---------+  | read-only, UNVERIFIED    |
         |                    |            +------------+-------------+
         +--------------------+-------------------------+
                              |  MarketSnapshot | MarketResolved
                              v
                  +------------------------+
                  |     BacktestRunner     |   backtest/runner.py
                  |  event loop, invariant |
                  |  checks, metrics       |
                  +-----+-------------+----+
                        |             |
          snapshot +    |             |   process / submit / cancel /
          open orders   |             |   merge / settle
                        v             v
          +----------------------+  +--------------------------+
          |  MarketMakerEngine   |  |      PaperExchange       |
          |  strategy/engine.py  |  |   exchange/paper.py      |
          |                      |  |   (Exchange protocol:    |
          |  phases    quoter    |  |    exchange/base.py)     |
          |  directional         |  |                          |
          |  rebalance   risk    |  |  latency, queue, print-  |
          |  FairValueModel      |  |  driven fills, cash and  |
          |  Portfolio           |  |  positions               |
          +----------+-----------+  +--------------------------+
                     |
                     |  actions: PlaceOrder | CancelOrder | MergePairs
                     v  (returned to the runner, which executes them)
                  runner --> BacktestResult --> backtest/report.py
                                                (text report, JSON, CSV)
```

The engine never talks to the exchange. It is a pure function of the events it is given and the
actions it returns, which is what makes runs deterministic and testable. For each snapshot the
runner (1) asks the exchange to process the snapshot and hands the resulting fills to the engine,
(2) calls `engine.on_snapshot(snapshot, open_orders)`, (3) executes the returned cancels, merges
and orders on the exchange, (4) passes any immediate fills back to the engine and (5) checks the
invariants. On `MarketResolved` the exchange settles the market and the engine books the payout.

Nothing in the core modules imports `data/`; it is an optional edge layer used only by the CLI.

Everything is under `src/abc_trading/`:

| File | Responsibility |
|---|---|
| `types.py` | Frozen domain types: `Outcome`, `Side`, `Phase`, order books, `MarketSnapshot`, `Trade`, orders, fills, actions, tick helpers. **Frozen contract.** |
| `config.py` | Frozen config dataclasses, validation, TOML load and dict conversion. **Frozen contract.** |
| `fees.py` | `FeeModel`: taker fee curve, maker fee and rebate, all-in buy cost. |
| `inventory.py` | `MarketInventory` and `Portfolio`: all-in average-cost accounting, merges, settlement, PnL breakdown, cash. |
| `model/fair_value.py` | `FairValueModel`: volatility EWMA, z-score, momentum, acceleration, book imbalance, shrinkage; plus `brier_score`, `log_loss`, `fit_logistic_weights`. |
| `sim/feed.py` | `SyntheticFeed`: correlated spot paths, lagged market price, mirrored books, informed and uninformed prints. |
| `data/events.py` | Lossless JSONL (de)serialisation of feed events (`write_jsonl`, `read_jsonl`). |
| `data/public_api.py` | Read-only Polymarket and spot-price HTTP clients behind a `Transport` seam (http(s) only, redirects included; every failure surfaces as a `DataError`, a malformed trade record is skipped and counted). **Unverified.** |
| `data/live.py` | `LiveFeed`: polls the clients and yields feed events (paper mode); `record` tees events to JSONL. |
| `exchange/base.py` | The `Exchange` protocol. |
| `exchange/paper.py` | `PaperExchange`, the only `Exchange` implementation: validation, latency of orders and cancels, queue, fills, merge, settlement. |
| `strategy/phases.py` | `phase_for`: WARMUP / ACCUMULATE / WIND_DOWN / FLATTEN / DONE from the clock. |
| `strategy/quoter.py` | `compute_quotes`: pure construction of the post-only bid ladder. |
| `strategy/directional.py` | Directional overlay: binary Kelly and the signed target net position. |
| `strategy/rebalance.py` | Merge sizing, taker pair completion, end-of-window flatten plan. |
| `strategy/risk.py` | `RiskManager`: kill switch, spot staleness, book sanity, rate limiter, capital caps. |
| `strategy/engine.py` | `MarketMakerEngine`: ties the above together per snapshot; reconciles desired versus resting orders. |
| `backtest/runner.py` | `run_backtest` / `BacktestRunner`: event loop, invariant checks, `BacktestResult`. |
| `backtest/metrics.py` | Pure statistics helpers: drawdown, Brier skill, Sharpe-like statistic. |
| `backtest/report.py` | `format_report` and `write_run_dir`. |
| `cli.py`, `__main__.py` | The command-line interface. |

---

## The fill model, and where it is pessimistic or optimistic

A paper exchange decides which of the bot's orders *would* have filled. Backtests are only as
honest as that decision, so `PaperExchange` is deliberately pessimistic about the things a
market maker cannot know, and the last paragraph of this section lists what it still gets
optimistically:

* **Latency.** An accepted order is *in flight* for `latency_ticks` snapshots of its market
  (default 1) before it is live. It cannot fill from any print that predates the moment it went
  live, so there is no look-ahead.
* **Cancels take the same time.** A cancel is a message to the venue like an order, so it lands
  `latency_ticks` snapshots after it is sent (instantly only at latency 0). Until then the order
  is still at the venue and can fill; it is just no longer reported by `open_orders`, so the engine
  does not cancel it twice. A quote the strategy has already decided to drop is therefore not safe
  from prints. (The first version of this model cancelled instantly, so a quote that was re-priced
  one snapshot after it went live could never be hit; that flattered fills and PnL, see
  [`docs/RESULTS.md`](docs/RESULTS.md) section 6.12.) The funds of an order whose cancel is pending
  are free for the orders sent behind the cancel, so a cash-limited account can replace a ladder
  in one step; to keep cash and positions non-negative a fill of the pending order is then capped
  to the funds the other orders leave free.
* **Queue position.** When an order goes live, the full displayed size at its price is assumed to
  be ahead of it (`queue_ahead_fraction = 1.0`). A print at our price first drains that queue;
  only the remainder fills us. Prints that go *through* our price do not shorten the queue.
* **Through-prints only at our price.** A resting order always fills at its **own** limit price,
  never at a better one. A buy at 0.47 hit by a sell print at 0.46 fills at 0.47.
* **Shared availability.** A print's size (times `trade_fill_fraction`) is shared across all of
  our orders in priority order (better price first, then older), so the same print cannot fill
  two orders in full.
* **Post-only is honoured.** A post-only order that would cross the book when it goes live is
  cancelled, not filled. Post-only orders never fill as takers.
* **Taker orders pay.** IOC orders walk only the *displayed* levels within their limit, consume
  depth that later orders in the same snapshot cannot reuse, pay the taker fee and optionally
  `taker_slippage_ticks`, which can move the fill price against them but never beyond their limit
  (so it does nothing to an IOC limited at the touch it saw, as the engine's are, unless the book
  has moved since: the configuration table says more). The UP and DOWN books are mirrors of one
  liquidity pool, so that depth is shared between them: a BUY of UP at 0.49 and a SELL of DOWN at
  0.51 draw on the same shares.
* **Cash is real.** A resting bid reserves `price * remaining` (and commits any positive maker fee
  on top, which the engine's own reservation accounting mirrors); cash never goes negative and
  reservations never exceed cash (asserted after every mutating call).

Worked example (100-share bid on UP at 0.47, with 400 shares displayed at 0.47 when it goes
live, so 400 are ahead of it). These are the numbers `PaperExchange` actually produces:

| Print (sell aggressor) | Effect | Our fill |
|---|---|---:|
| 30 shares at 0.47 | drains the queue ahead: 400 to 370 | 0 |
| 30 shares at 0.46 (through our price) | fills us; queue not reduced | 30 at 0.47 |
| 500 shares at 0.47 | first 370 drain the queue, 130 left for us, we need 70 | 70 at 0.47 |

**Why it matters, and the caveat.** Fills in a backtest depend heavily on these assumptions. In
the three-seed latency sweep below, going from 0 to 1 snapshot of latency turns a mean PnL of
+187 USDC into -216 USDC, and in the default seed-1 run 79% of the post-only orders were cancelled
by the engine one snapshot after they were placed while 72% of the maker fills came from orders
the engine had already cancelled (their cancel had not landed yet). "Pessimistic" is a design
intent, not a validated property: the model has never been compared with real queues or cancel
acknowledgements, and it can still be optimistic in ways it does not model (it assumes the
displayed books and prints are complete and timely, ignores the bot's own market impact, merges
instantly and for free, never has an order rejected by the venue, lets a cancel fail only by
losing a race to a fill, and, on a cash-limited account, favours a replacement order over the
stale order it replaces).

---

## Sample results (synthetic data)

> **SYNTHETIC DATA. Mechanics validation only. Nothing in this section is evidence that the
> strategy is, or is not, profitable in real markets.** The simulator has a deliberate 2-second
> market lag, and the fee parameters, model weights, thresholds and fill model are uncalibrated
> placeholders.

Copied from [`docs/RESULTS.md`](docs/RESULTS.md), and re-run with the CLI for this README (all
values below reproduced exactly). Defaults throughout (`BotConfig()`), 12 windows of 900 s for
each of BTC and ETH (24 markets), 10,000 USDC initial cash, `latency_ticks=1`. Every run is
deterministic. Nothing was tuned.

### Default configuration, seeds 1 to 6

```bash
python3 -m abc_trading backtest --seed S --windows 12      # S = 1..6
```

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 1255 | 36.01 | 95.7% | +338.98 | 440.6 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 665 | 35.30 | 90.7% | -744.96 | 1070.6 | 0.0931 | 0.0949 | TRIPPED |
| 3 | 12 x 900 s | BTC,ETH | 965 | 33.96 | 93.8% | -242.88 | 551.9 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 942 | 34.68 | 93.7% | -499.95 | 753.7 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 821 | 34.54 | 91.9% | -726.44 | 969.9 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 807 | 35.69 | 91.7% | -894.99 | 1034.5 | 0.0209 | 0.0239 | no |
| **mean** | | | 909 | 35.03 | 92.9% | **-461.71** (s.e. 185) | 803.5 | 0.0711 | 0.0688 | 1 of 6 |

How the PnL arises (same runs, all fully resolved so nothing is unrealised; merge, flatten-sell and
settlement PnL add up to the total, fees are already inside them):

| Seed | Merge PnL | Flatten-sell PnL | Settlement PnL (pair + directional) | Fees paid (inside) | Total PnL |
|---:|---:|---:|---:|---:|---:|
| 1 | +1723.3 | -1386.0 | +1.7 | 135.3 | +339.0 |
| 2 | +738.5 | -1482.6 | -0.8 | 82.7 | -745.0 |
| 3 | +1170.8 | -1396.4 | -17.3 | 104.2 | -242.9 |
| 4 | +1072.7 | -1573.1 | +0.5 | 84.8 | -500.0 |
| 5 | +847.9 | -1575.2 | +0.8 | 87.0 | -726.4 |
| 6 | +852.2 | -1748.2 | +1.0 | 88.9 | -895.0 |

Over **40 seeds** (`--seed 1..40 --windows 12`, same config) the default run gives a mean PnL of
**-538.5 USDC** (standard error 62.5, sd 395.1, median -478.2, range -1,162.4 to
+339.0), positive in 2 of 40 runs, and the kill switch trips in 9 of them (a tripped run
stops trading for the rest of its windows). Average trade 34.7 USDC (range 31.9 to
36.4), so the source claim of about 53 is not reproduced; pair completion 92.7% on average; mean
merge PnL +1,054.4 and mean flatten-sell PnL -1,593.5. The engine's Brier score is 0.0731 against
0.0728 for the market mid, so the model has no measurable skill over the market here. Read this
as: on this simulator and with these uncalibrated defaults, the strategy earns its pair margin
through merges and gives all of it (and a good deal more) back on the inventory it has to dump.

### Variants (seeds 1 to 6, 12 windows each)

| Variant | What changes | Fills (mean) | PnL mean (s.e.) | Seeds with PnL > 0 | Kill switch trips |
|---|---|---:|---:|---:|---:|
| default | nothing | 909 | -462 (185) | 1 of 6 | 1 of 6 |
| btc_only | `--assets BTC` | 494 | -183 (148) | 1 of 6 | 0 of 6 |
| informed0 | `--set sim.informed_flow=0` (no sweeps) | 528 | -22 (59) | 3 of 6 | 0 of 6 |
| lag0 | `--set sim.market_lag_seconds=0` | 916 | -874 (111) | 0 of 6 | 5 of 6 |
| nodir | `--set directional.enabled=false` | 878 | -769 (174) | 1 of 6 | 3 of 6 |
| feeheavy | `--set fees.taker_fee_rate=1.0 --set fees.maker_fee_rate=0.01` | 686 | -865 (80) | 0 of 6 | 3 of 6 |

Standard errors are across the six seeds, so differences of less than about two standard errors
are noise. The directional overlay looks useful here only because it exploits the lag the
simulator builds in on purpose (compare `lag0`). The sensitivity to the paper-exchange latency
(seeds 1 to 3):

| `exchange.latency_ticks` | Fills (mean) | PnL per seed (USDC) | PnL mean |
|---:|---:|---|---:|
| 0 | 1340 | +598, -131, +95 | +187 |
| 1 (default) | 962 | +339, -745, -243 | -216 |
| 2 | 1194 | -324, -1043, -1077 | -815 |
| 3 | 1198 | -493, -1020, -804 | -772 |

With three seeds these PnL means are noise (their standard errors are 153 to 313 USDC). The fill
count no longer collapses as latency grows: a cancel takes as long as an order, so a quote that
is re-priced every snapshot stays hittable for `latency_ticks` more snapshots, and those stale
quotes are the adversely selected ones, which is why PnL falls with latency. (Before cancels
were given a latency this sweep read 1340 / 514 / 227 / 117 fills for latency 0 to 3, "a factor
of 11"; that was largely an artefact of instant cancels, see [`docs/RESULTS.md`](docs/RESULTS.md)
section 6.12.) See that file for the remaining runs, full simulated days, timing and the
root-cause analysis of every anomaly that was looked for.

---

## Known limitations and risks

**Strategy and market risks (these would apply to any real use of the idea):**

* **Leg risk.** A pair only locks profit when *both* legs fill. A single unhedged leg that goes
  the wrong way loses far more than the pair margin earns (47 cents against 1 cent in the example
  above).
* **Adverse selection.** Passive bids fill when the market moves through them. Fills are
  correlated with information, so the inventory you are left holding is skewed toward the
  positions that moved against you. In the sample runs this is the main drain: flatten sells lose
  more than the merges earn, and in the three seeds that were decomposed about 98% of that loss
  was inventory price drift rather than execution cost.
* **Queue position.** Whether a resting order fills depends on its place in a queue that this
  project can only assume. Real fill rates may be very different, in either direction.
* **Fees.** The fee model is a placeholder. A one-cent margin is easily erased or reversed by a
  fee, and fee rules can change. Verify the schedule before trusting any PnL.
* **Latency.** The simulation uses a one-snapshot (one-second) delay by default, for orders and
  for cancels alike (a cancelled quote can still be hit until its cancel lands). Real latency
  includes network, matching and, for merges, on-chain confirmation, and is not constant.
  IOC exits at a stale touch are unreliable in the last seconds of a window, when prices move
  several ticks per second.
* **Capital lock-up.** Resting bids reserve cash (`price * size`, plus any positive maker fee), and
  unmerged pairs and unpaired inventory tie up capital until resolution. Merges are instant and
  free here; in reality they are on-chain actions with cost and delay that are not modelled.
* **Resolution-source mismatch.** The synthetic feed resolves off its own spot path, and
  `LiveFeed` *infers* the winner from Binance or Coinbase spot against the first reading of the
  window. The real market has its own reference ("price to beat") and its own resolution source,
  which may differ, and near the line the difference decides the winner. Any directional exposure,
  and the model's reference price, inherit this error.
* **Polymarket rules and jurisdictional restrictions.** Market structure (window lengths, tick
  and minimum order sizes, fees, how ties and disputes resolve) and the terms of use are set by
  the venue and can change. Access to Polymarket is restricted or prohibited in some
  jurisdictions. Whether and how you may use it is your responsibility; this is not legal advice.
* **API changes.** Endpoints, schemas and rate limits can change without notice. The clients here
  were never verified against the real ones to begin with.

**Limitations of this project:**

* **Quote churn.** With the default parameters most orders are replaced right away: in the
  default seed-1 run 79.1% of the post-only orders were cancelled by the engine one snapshot after
  they were placed. The book moves nearly every second and the ladder is re-priced off the touch
  (`sizing.requote_tolerance_ticks` is 0). Because a cancel takes as long as an order, the
  replaced quotes stay hittable meanwhile: 72% of the maker fills of that run came from orders the
  engine had already cancelled, and those fills are worse than the others (-1.40 against -1.07
  cents from the mid at the moment of the fill). Fills and PnL therefore depend on the latency and
  queue assumptions more than on the strategy.
* **The default strategy loses on this simulator** (mean -538.5 USDC per 12-window run over 40
  seeds, positive in 2 of them, kill switch tripped in 9). Whether that carries over to real
  markets is unknown.
* **The model is uncalibrated** and is no better than the market's mid price on synthetic data
  (Brier 0.0731 against 0.0728). Calibration tools exist (`fit_logistic_weights`, `brier_score`,
  `log_loss`), but there is no real data here to run them on.
* **Synthetic data is simple.** Spot is correlated geometric Brownian motion, with no jumps or fat
  tails, and book depth and spreads are random draws. Real markets are different.
* **Average-cost accounting** pools paired and unpaired shares, so the split between "merge PnL"
  and "sell PnL" in one market is an attribution. Totals are exact (recomputed independently from
  the fills in the integration pass).
* **Soft caps.** The inventory and capital caps bound what the engine rests; an order it has
  cancelled can still fill until the cancel lands, so the simulated net imbalance reached 584
  shares against a cap of 300 plus a 200-share directional allowance.
* **The kill switch** compares equity with the starting equity of the *run*, not per calendar day,
  and once latched it never resets. A multi-day backtest is judged against one
  `max_daily_loss_usd`.
* **`paper` mode is unverified end to end** (see [`paper`](#paper)). The slug template, the
  price-to-beat rule and the winner inference are approximations of the real venue.
* **Single process, no persistence.** A restart of `paper` mode waits for the next window start.

---

## What going live would require, and why it is not here

This project does not contain and will not accept live-trading code. The `Exchange` protocol
exists so that the engine is written against an interface rather than against the simulator, not
so that real money can be moved. Anyone who wanted to take this idea toward real markets would
need, at minimum, a long list of things that nothing here provides:

1. **Evidence of an edge on real data.** Months of recorded real order books and trades, an
   out-of-sample backtest against them, and fill-model and fee calibration from real fills. A
   profitable synthetic run is not that, and the sample results here are not even positive.
2. **Verified venue facts.** The real fee schedule, tick and minimum sizes, market structure,
   reference and resolution sources, rate limits, and verified (not remembered) endpoints and
   schemas.
3. **Execution infrastructure.** Authentication, key and wallet custody, order signing,
   acknowledgements, partial fills, reconnects, reconciliation against the venue's own state and
   on-chain merges. These are a large, security-critical body of work.
4. **Operational safety.** Monitoring, alerting and a kill switch that lives outside the
   strategy process, hard capital limits, and a staged path from paper to tiny size.
5. **Legal and compliance review.** Whether the venue may be used from your jurisdiction at all,
   its terms, and tax treatment.

It is deliberately excluded because the project's purpose is to study mechanics safely and
honestly. There is no verified edge, the inputs that matter (fees, queue behaviour, resolution
rules) are unverified, and code that signs orders or handles keys carries risks (financial,
security and legal) that a research framework should not take on by default.

---

## Testing and invariants

```bash
python3 -m pytest                     # the whole suite (2,076 tests, about 30 s, offline)
python3 -m pytest tests/test_paper_exchange.py -q
ruff check . && ruff format --check . # lint and formatting (line length 100)
mypy src                              # static typing (disallow_untyped_defs)
```

Tests are deterministic and offline: the HTTP clients are exercised through a fake `Transport`,
made-up fixtures in `tests/fixtures/` and a loopback server on 127.0.0.1, never a real host.
Where practical they assert exact numbers worked out by hand, and the invariants below also have
seeded randomised tests.

The strategy tests were also checked by mutation: one-line changes of the engine, the rebalancer
and the quoter (`and` to `or`, `flatten=True` to `False`, a sign, a dropped phase or rate-limit
check) must each fail at least one test. `tests/test_killers_engine.py` is the table of what the
engine does in each phase for each kind of inventory (merge, taker completion, FLATTEN sale,
passive quotes), plus floors on the taker completions of real runs; the per-module files pin the
rest (the completion guard, the order-rate limiter on every taker order, the requote throttle's
hard-stop cancels, the fee band of the hold-to-resolution rule, sizing on current equity).

A second mutation pass looked for changes that swap or ignore one token (the DOWN mid scored as the
market's Brier, the equity marked crosswise, a spread limit on one book, an order-book feature
without the DOWN book, a completion, flatten, clip, WIND_DOWN or queue rule written for UP only) and
for a deleted final-accounting check. They all survived, because most fixtures put the same book on
both tokens. `tests/test_killers_asymmetry.py` pins each rule on both tokens and checks that the
planners are label-symmetric (swap UP and DOWN throughout a randomised state and the answer swaps);
the report metrics, the risk gate, the model's order-book feature and invariant 4 are pinned in
`test_backtest_runner.py`, `test_strategy_risk.py` and `test_model.py` on books that do not mirror
each other. The metrics the invariants do not cover (the Brier scores, the drawdown, the pair
metrics) are checked there against hand-worked numbers and an independent recomputation from the feed.

The ten invariants of [`docs/DESIGN.md`](docs/DESIGN.md) section 7, and where they are enforced:

| # | Invariant | Where |
|---:|---|---|
| 1 | **Reconciliation.** Engine cash equals exchange cash, and per-market positions agree (within 1e-6), after every event. | Runner checks it on every run unless `--no-invariants`; `test_backtest_runner.py`, `test_backtest_e2e.py` |
| 2 | `cash + capital_at_risk - initial_cash == realised_pnl` after every fill and merge. | Runner; `test_inventory.py` (randomised trading) |
| 3 | No negative cash or quantity; `reserved_cash <= cash`. | Runner; the exchange's own always-on self-check; `test_paper_exchange.py` (random walk) |
| 4 | At the end every market is settled, no orders remain, and `sum(PnLBreakdown.total) == final_equity - initial_cash`. | Runner end-of-stream check; `test_backtest_runner.py` (invariant 4 on its own, with its 1e-6 tolerance) |
| 5 | **Determinism.** Same config and seed give an identical result; sim to JSONL to replay gives an identical result. | `test_cli.py`, `test_backtest_e2e.py`, `test_sim_feed.py`, `test_integration.py` (including different hash seeds) |
| 6 | **No look-ahead.** Model and engine see only data with `ts <= now`; fills come only from prints after an order is live. | `test_integration.py` (with a deliberately peeking negative control), `test_model.py`, `test_paper_exchange.py` |
| 7 | Post-only orders never fill as taker; the engine never emits one that crosses the book. | `test_integration.py`, `test_paper_exchange.py` |
| 8 | **Pair-cap property.** Any bid placed while holding unpaired opposite inventory satisfies `bid + avg_cost[opposite] <= 1 - target_margin + EPS`. | Seeded randomised test in `test_strategy_quoter.py`; audited on real runs in `test_integration.py` |
| 9 | No orders (other than merges, flatten sells and cancels) in WARMUP / FLATTEN / DONE, nor with the kill switch latched or the spot stale. | `test_integration.py`, `test_strategy_engine.py`, `test_killers_engine.py` (the phase by inventory decision table) |
| 10 | Configs round-trip; invalid configs are rejected (TOML values must have the declared type, ranges are checked up front, with the key named). | `test_config.py` |

A violated invariant stops the run with exit code 3 and a precise message. Invariants 1 to 4 are
checked inside every backtest unless `--no-invariants` is given, and the report says which ran.
They are accounting identities (cash, positions, realised PnL, final accounting), so they cover
the money numbers, not derived statistics such as the Brier scores, the drawdown (which depends on
`--sample-every`) or the pair-completion rate. A run that ends with unresolved markets (`paper`,
`replay --allow-unresolved`) has invariants 1 to 3 checked; invariant 4 cannot hold there and the
report says it was skipped.

---

## Contributing conventions

These come from section 8 of [`docs/DESIGN.md`](docs/DESIGN.md).

* **Standard library only** at runtime. Do not add dependencies. Python 3.11 or newer.
* **Fully typed** (`disallow_untyped_defs`), small pure functions, dataclasses, no global state.
  Docstrings state units and invariants.
* **Deterministic.** No wall clock, no unseeded randomness. Use a private `random.Random(seed)`.
* **No network or file I/O** outside `data/` and `cli.py`. Tests never touch the network.
* **Fail loudly.** Raise `ValueError` when an invariant is violated (negative size, overselling,
  over-merging, filling a settled market) instead of silently clamping. Compare floats with the
  `EPS` tolerance from `types.py`.
* **Tests** live in `tests/test_<module>.py`, run in under 5 seconds per file, and assert exact
  expected numbers with the arithmetic shown in a comment, not merely that code runs. Include
  seeded randomised tests for any invariant your module is responsible for. Do not rest on
  mirrored fixtures alone: build the UP and DOWN books independently, exercise both tokens and both
  directions of a target, and write the oracle of a randomised test from the spec, not from the code
  under test.
* **`ruff check`, `ruff format` and `mypy src` must pass** on everything you touch.
* **Frozen contract.** `types.py`, `config.py` and `docs/DESIGN.md` are the contract between
  modules. Change them only deliberately, together, and in the same commit; do not edit them
  casually from an unrelated change.
* **No live trading, no secrets.** No order signing, keys, wallets or real exchange adapters, and
  no credentials anywhere in the repository.
* **Be honest in documentation and output.** Anything produced from synthetic data must say so.
  Do not tune parameters toward a target such as the $53 average trade, do not present results
  without their caveats, and label unverified claims as unverified.
