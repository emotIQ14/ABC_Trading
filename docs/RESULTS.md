# Results: synthetic-data mechanics validation

> **SYNTHETIC DATA. Mechanics validation only. Nothing in this file is evidence that the strategy
> is, or is not, profitable in real markets.**
> The simulator (`sim/feed.py`) contains a deliberate 2-second market lag so that the directional
> model has something to find, so any gain from the model on this feed is circular by construction.
> Fee parameters, model weights, strategy thresholds and the paper-exchange fill model are
> uncalibrated placeholders. No live trading exists in this project.

This file records what the integration pass observed when the finished modules were run end to end
(`SyntheticFeed` -> `MarketMakerEngine` -> `PaperExchange` -> `run_backtest`), which anomalies were
looked for, and what caused each one. All numbers below were produced by the commands in section 9
and are exactly as observed (nothing was tuned, rounded in the strategy's favour or re-run to get a
better number). Every run is deterministic, so any row can be reproduced exactly (the wall-time
column of section 5 aside).

## 1. What was run

| Item | Value |
|---|---|
| Code | `HEAD` 0a97c6e plus the uncommitted working tree holding all modules (the lead commits) |
| Interpreter / machine | Python 3.11.15, Linux x86_64, 4-core Intel Xeon 2.1 GHz |
| Config | built-in `BotConfig()` defaults (identical to `configs/default.toml`, tested) unless a row says otherwise |
| Simulation | 12 windows of 900 s per asset, BTC and ETH (24 markets, 21,600 snapshots, 21,624 events) unless a row says otherwise |
| Account | 10,000 USDC initial cash, `latency_ticks=1`, `queue_ahead_fraction=1.0`, `trade_fill_fraction=1.0` |
| Checks | invariants 1-4 of DESIGN section 7 checked after every event in every run (never violated) |
| Test suite | 1,235 tests pass (about 18 s), `ruff check`, `ruff format --check` and `mypy src` clean |

Column definitions used in every table:

* **Windows** is windows per asset (a "12 x 900 s" row has 24 markets when two assets run).
* **Avg trade** is the mean `price * size` over all fills, in USDC (the unverified source claim is
  about 53 USDC; nothing here is tuned to it).
* **Pair completion** is `2 * (pairs merged + pairs held at close) / shares bought`.
* **Max drawdown** is measured on the mark-to-mid equity curve sampled every 60 events (30 simulated
  seconds with two assets). Event-granular drawdown is larger (one 4-window check: 227.7 against
  204.2 sampled).
* **Brier model / market** is the Brier score at the last ACCUMULATE snapshot of each market of the
  engine's `p_up` (the model shrunk 50% toward the market) and of the market's own UP mid, against
  the realised winner. Lower is better; always saying 0.5 scores 0.25. With 24 markets per run these
  are very noisy, and most outcomes are already well determined by the last ACCUMULATE snapshot.
* **Kill switch** is whether the `max_daily_loss_usd=1000` latch tripped.

## 2. Default configuration, seeds 1 to 6

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 597 | 38.01 | 94.6% | -75.97 | 304.2 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 475 | 36.77 | 93.6% | -168.40 | 411.6 | 0.0931 | 0.0949 | no |
| 3 | 12 x 900 s | BTC,ETH | 471 | 35.81 | 94.3% | +126.05 | 216.7 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 507 | 36.62 | 94.5% | +47.58 | 181.7 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 383 | 35.77 | 90.0% | -255.88 | 376.0 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 483 | 35.89 | 92.8% | -303.84 | 558.9 | 0.0209 | 0.0239 | no |
| **mean** | | | 486 | 36.48 | 93.3% | **-105.07** (s.e. 69) | 341.5 | 0.0711 | 0.0688 | 0 of 6 |

How the PnL arises (same runs; merge, flatten-sell and settlement PnL add up to the total, fees are
already inside them):

| Seed | Merge PnL | Flatten-sell PnL | Settlement PnL (pair + directional) | Fees paid (inside) | Total PnL | Merges / flatten sells (orders) | Quotes placed / cancelled | Post-only cancels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +936.3 | -1012.7 | +0.5 | 69.4 | -76.0 | 209 / 20 | 40,609 / 38,058 | 2,172 |
| 2 | +700.7 | -869.9 | +0.8 | 48.2 | -168.4 | 154 / 22 | 34,926 / 33,144 | 1,498 |
| 3 | +752.8 | -628.7 | +2.0 | 48.1 | +126.1 | 152 / 27 | 36,527 / 34,677 | 1,560 |
| 4 | +559.5 | -512.8 | +0.9 | 49.6 | +47.6 | 165 / 32 | 35,694 / 34,079 | 1,329 |
| 5 | +490.0 | -747.0 | +1.2 | 38.7 | -255.9 | 122 / 37 | 32,383 / 30,890 | 1,277 |
| 6 | +537.1 | -841.8 | +0.8 | 50.2 | -303.8 | 156 / 26 | 36,113 / 34,407 | 1,427 |

Over **40 seeds** (`--seed 1..40 --windows 12`, same config) the default run gives a mean PnL
of **-120.0 USDC** (standard error 38.8, sd 245.7, median -111.4, range
-600.8 to 488.9), positive in 15 of 40 runs, and the kill switch never trips
(0 trips). Fills per run 383 to 620; average trade
36.8 USDC (range 34.8 to 39.2, so the source claim of about 53 is not reproduced);
pair completion 93.6% on average (90.0% to 96.8%); mean max drawdown 377.6
(worst 708.4); mean merge PnL 665.3 and mean flatten-sell PnL -786.4; Brier score of
the engine 0.0731, of the raw model 0.0743, of the market mid 0.0728.
Read this as: on this simulator and with these uncalibrated defaults the strategy earns its pair
margin through merges and gives all of it (and a bit more) back on the inventory it has to dump.

## 3. Variants (seeds 1 to 6, 12 windows each)

| Variant | What changes | Fills (mean) | PnL mean (s.e.) | Seeds with PnL > 0 | Kill switch trips |
|---|---|---:|---:|---:|---:|
| default | nothing | 486 | -105 (69) | 2 of 6 | 0 of 6 |
| btc_only | `--assets BTC` | 246 | -2 (48) | 2 of 6 | 0 of 6 |
| informed0 | `sim.informed_flow=0` (no sweeps) | 176 | -24 (35) | 1 of 6 | 0 of 6 |
| lag0 | `sim.market_lag_seconds=0` | 636 | -584 (115) | 0 of 6 | 1 of 6 |
| nodir | `directional.enabled=false` | 538 | -348 (62) | 0 of 6 | 0 of 6 |
| feeheavy | `fees.taker_fee_rate=1.0`, `fees.maker_fee_rate=0.01` | 438 | -390 (85) | 0 of 6 | 0 of 6 |

Standard errors are across the six seeds, so differences between rows of less than about two
standard errors are noise. The Brier columns of `informed0`, `nodir` and `feeheavy` equal the
default's because those knobs change neither the prices of the feed nor the model.

### 3.1 BTC only (`--assets BTC`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC | 276 | 38.89 | 94.9% | -8.39 | 230.7 | 0.1448 | 0.1329 | no |
| 2 | 12 x 900 s | BTC | 245 | 37.11 | 93.7% | -156.77 | 322.3 | 0.0781 | 0.0762 | no |
| 3 | 12 x 900 s | BTC | 249 | 37.34 | 95.4% | +193.00 | 153.8 | 0.1281 | 0.1205 | no |
| 4 | 12 x 900 s | BTC | 241 | 37.43 | 95.8% | +53.27 | 129.5 | 0.0578 | 0.0587 | no |
| 5 | 12 x 900 s | BTC | 178 | 35.30 | 90.6% | -53.98 | 157.1 | 0.0207 | 0.0189 | no |
| 6 | 12 x 900 s | BTC | 287 | 37.02 | 95.2% | -37.84 | 311.2 | 0.0239 | 0.0321 | no |
| **mean** | | | 246 | 37.18 | 94.3% | **-1.78** (s.e. 48) | 217.4 | 0.0756 | 0.0732 | 0 of 6 |

Half the markets, so about half the fills; the mean is indistinguishable from zero.

### 3.2 No adverse selection (`--set sim.informed_flow=0`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 196 | 13.69 | 85.2% | -36.23 | 133.8 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 165 | 14.94 | 75.4% | -73.84 | 128.8 | 0.0931 | 0.0949 | no |
| 3 | 12 x 900 s | BTC,ETH | 193 | 14.20 | 77.7% | -5.78 | 104.4 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 171 | 12.06 | 82.2% | +134.55 | 32.4 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 165 | 14.08 | 78.4% | -47.55 | 165.1 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 169 | 12.28 | 73.8% | -113.63 | 164.8 | 0.0209 | 0.0239 | no |
| **mean** | | | 176 | 13.54 | 78.8% | **-23.75** (s.e. 35) | 121.6 | 0.0711 | 0.0688 | 0 of 6 |

With the informed sweeps switched off the only prints are random touch prints (0.3 per second per
token side, about 30 shares), so the bot is filled about a third as often, in small clips (about
13.5 USDC), and pairs less completely (79%). The mean PnL is indistinguishable from zero. The
average maker BUY fill is still 0.45 cent above the mid at the moment of the fill (section 6.4).

### 3.3 No market lag (`--set sim.market_lag_seconds=0`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 914 | 38.93 | 95.6% | -204.50 | 468.9 | 0.1340 | 0.1357 | no |
| 2 | 12 x 900 s | BTC,ETH | 641 | 38.29 | 93.1% | -526.83 | 668.0 | 0.0911 | 0.0904 | no |
| 3 | 12 x 900 s | BTC,ETH | 457 | 38.21 | 89.6% | -1015.17 | 1114.2 | 0.1120 | 0.1154 | TRIPPED |
| 4 | 12 x 900 s | BTC,ETH | 693 | 39.13 | 94.0% | -493.17 | 755.9 | 0.0756 | 0.0729 | no |
| 5 | 12 x 900 s | BTC,ETH | 508 | 36.41 | 91.4% | -472.64 | 629.1 | 0.0143 | 0.0139 | no |
| 6 | 12 x 900 s | BTC,ETH | 605 | 37.42 | 92.5% | -791.46 | 869.7 | 0.0191 | 0.0194 | no |
| **mean** | | | 636 | 38.07 | 92.7% | **-583.96** (s.e. 115) | 751.0 | 0.0743 | 0.0746 | 1 of 6 |

The market now sees the spot with no delay, so the model has nothing to find and the sweeps still
pick the resting bids off. This is the worst variant, and seed 3 trips the kill switch (equity down
1,015 USDC against the 1,000 USDC limit during the 11th window; the equity curve falls steadily,
there is no single jump, and the 12th window does not trade).

### 3.4 Directional overlay off (`--set directional.enabled=false`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 658 | 37.60 | 95.6% | -203.71 | 303.6 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 536 | 36.76 | 93.8% | -516.86 | 584.0 | 0.0931 | 0.0949 | no |
| 3 | 12 x 900 s | BTC,ETH | 540 | 37.12 | 94.8% | -197.05 | 321.5 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 530 | 36.03 | 94.3% | -250.09 | 361.5 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 466 | 36.01 | 91.5% | -388.50 | 503.1 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 499 | 36.79 | 92.4% | -528.92 | 556.7 | 0.0209 | 0.0239 | no |
| **mean** | | | 538 | 36.72 | 93.7% | **-347.52** (s.e. 62) | 438.4 | 0.0711 | 0.0688 | 0 of 6 |

Worse than the default in all six seeds (paired by seed). The only difference is the overlay, which
uses the 2-second lag that the simulator builds in on purpose, so this gap is a property of the
simulator and not evidence of a real edge.

### 3.5 Fee-heavy (`--set fees.taker_fee_rate=1.0 --set fees.maker_fee_rate=0.01`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 539 | 35.56 | 95.1% | -225.83 | 362.9 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 495 | 35.94 | 94.0% | -203.14 | 273.7 | 0.0931 | 0.0949 | no |
| 3 | 12 x 900 s | BTC,ETH | 421 | 36.07 | 91.6% | -275.35 | 361.5 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 410 | 33.92 | 91.0% | -365.93 | 493.3 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 356 | 32.82 | 89.2% | -538.87 | 668.1 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 408 | 34.99 | 90.1% | -732.40 | 789.3 | 0.0209 | 0.0239 | no |
| **mean** | | | 438 | 34.88 | 91.8% | **-390.25** (s.e. 85) | 491.5 | 0.0711 | 0.0688 | 0 of 6 |

Taker fees four times the default and a 1% maker fee on notional. Fees paid rise from 39-69 to
180-289 USDC per run; the independent recomputation of cash from the raw fills (section 6.10) agrees
to 1e-6 with the fee formula applied independently.

## 4. Sensitivity runs: why the numbers look the way they do (not tuning)

These rows change one assumption at a time to expose a mechanism (section 6.1). They are NOT an
attempt to find better parameters, no default was changed, and with three seeds the PnL columns
are noise (the standard error of a three-seed mean is 59 to 239 USDC across these rows).

| Variant (seeds 1-3) | Fills (mean) | Avg trade (USDC) | Pair completion | PnL per seed (USDC) | PnL mean | Quotes placed per snapshot | Post-only cancels (mean) | Rate-limited placements (mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| default (latency 1, tolerance 0) | 514 | 36.86 | 94.1% | -76, -168, +126 | -39 | 1.73 | 1743 | 121 |
| `exchange.latency_ticks=0` | 1340 | 38.34 | 96.4% | +598, -131, +95 | +187 | 1.33 | 0 | 43 |
| `exchange.latency_ticks=2` | 227 | 36.02 | 92.0% | -47, -209, -18 | -91 | 2.21 | 550 | 304 |
| `exchange.latency_ticks=3` | 117 | 35.81 | 90.7% | -62, -114, +148 | -9 | 2.67 | 157 | 239 |
| `sizing.requote_tolerance_ticks=1` | 609 | 37.28 | 94.2% | +190, -159, -209 | -59 | 1.33 | 1280 | 51 |
| `sizing.requote_tolerance_ticks=2` | 624 | 37.16 | 95.0% | +601, -224, +248 | +208 | 1.11 | 1066 | 21 |
| `sim.window_seconds=300`, 36 windows | 780 | 37.84 | 93.9% | +557, +32, +183 | +257 | 1.86 | 3936 | 140 |

The fill count falls by a factor of 11 as the latency goes from 0 to 3 snapshots, and every number
the backtest prints is therefore dominated by the latency and queue assumptions of the paper
exchange.

## 5. Full simulated days and speed

| Run | Assets | Windows | Markets | Snapshots | Wall time | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BTC day, seed 1 | BTC | 96 x 900 s | 96 | 86,400 | 23.2 s | 2053 | 37.94 | 94.0% | -601.09 | 1012.5 | 0.0518 | 0.0493 | no |
| BTC day, seed 2 | BTC | 96 x 900 s | 96 | 86,400 | 23.2 s | 2108 | 36.64 | 94.9% | -75.60 | 546.8 | 0.1013 | 0.1016 | no |
| BTC day, seed 3 | BTC | 96 x 900 s | 96 | 86,400 | 22.9 s | 2158 | 37.24 | 95.2% | +596.52 | 474.2 | 0.0923 | 0.0895 | no |
| BTC+ETH day, seed 1 | BTC,ETH | 96 x 900 s | 192 | 172,800 | 55.7 s | 4300 | 38.15 | 94.6% | -456.92 | 881.5 | 0.0662 | 0.0640 | no |

One BTC asset-day (96 windows, 86,400 snapshots) takes about 23 s with the invariant checks on
(about 0.27 ms per snapshot); two assets take 55.7 s, so about 28 s per simulated asset-day. The
cost per snapshot does not grow with the length of the run (0.267, 0.259, 0.251 and 0.237 ms per
snapshot for 12, 48, 96 and 192 BTC windows without invariant checks). Before this pass a BTC day
took 29.4 s and a two-asset day 67.4 s (the exchange's self-check rebuilt long error messages on
every call; section 7).

## 6. Review of anomalies and root causes

The pass looked for the problems listed below. Where a problem was found, its cause is given and the
verdict says whether it is a bug (fixed, with a test) or what the strategy does under the
simulator's assumptions (documented, parameters not touched). No functional defect was found in
the modules' integration: the invariant checks, the independent audits of section 6.10, 450
randomised configurations and a corrupted-data run all came back clean.

### 6.1 Quote churn and the fill count (by design under the simulator's assumptions)

* Finding: the default run places 1.88 quotes and cancels 1.76 per snapshot (seed 1: 40,609 placed and
  38,058 cancelled over 21,600 snapshots). Of the 40,286 post-only orders that ended, **80.0% lasted one
  snapshot** (14.9% two, 3.1% three, 2.0% four or more), 94.2% were cancelled by the engine, 5.4% by
  the exchange's post-only re-check, and only 572 (1.4%) ever received a fill.
* Root cause: (1) in this simulator the book moves almost every second. The UP best bid changes on
  73.1% of ticks (84.6% for either side; a move of two ticks or more on about a third of ticks), because
  a binary option's price moves by about `0.4 / sqrt(seconds left)` per second (1.6 ticks per second
  with 600 s left, 4 ticks with 100 s left). (2) The ladder is priced off the touch and
  `requote_tolerance_ticks` is 0, so any move re-prices it. (3) An order placed at snapshot `t`
  goes live at `t+1` and can first match the prints of `t+2`; an order that is re-priced at `t+1`
  therefore never had a chance. Only about 20% of orders live long enough to be matchable.
* Evidence that this is the mechanism (three-seed means, section 4): raising `requote_tolerance_ticks` to
  1 and 2 cuts quotes placed per snapshot from 1.73 to 1.33 and 1.11 and raises fills from 514 to 609
  and 624; and fills fall from 1,340 to 514, 227 and 117 as the latency goes 0, 1, 2, 3.
* Verdict: not a bug. It is how the default parameters interact with a fast book and a 1-snapshot
  latency. Left as is; the rate limiter (20 messages per second, cancels included) clipped 113 of
  40,609 placements (0.3%), all of them in ACCUMULATE, never a flatten sell.

### 6.2 Merges versus flatten sells (by design)

* Finding: in every variant merge PnL is positive and flatten-sell PnL is negative, and the second is
  usually larger (default seed 1: +936.3 and -1,012.7).
* Root cause: pairs complete when the price comes back through the second bid, and they pay a
  margin of 4.2 cents per merged pair here (0.9583 all-in cost, against a 1 cent target); what is
  left unpaired is the inventory whose opposite bid the price never came back to, so it is selected
  to have moved against the bot. Decomposing the flatten sells
  exactly (seeds 1 to 3, 2,575 / 2,375 / 2,059 shares): -1,012.7 / -869.9 / -628.7 USDC is
  -996.4 / -849.9 / -613.2 of price drift (mid at the sale minus average cost), -15.7 / -20.0 / -15.5
  of spread paid against the mid and 0.56 / 0.03 / 0.05 of fees. So about 98% of the loss is
  inventory risk and only about 2% is execution cost.
* Verdict: what the strategy does under these assumptions; no parameter touched.

### 6.3 Pair cost above 1 (three small cases; accounting attribution, not a bug)

* Finding: no run's pooled merged-pair cost exceeds 1 (the default runs merge at about 0.96). Three
  of the 2,328 markets of the matrix and the sweep merged pairs at an average cost above 1:
  `BTC-1700001000-900s` of `informed0` seed 1 (124 pairs at 1.0244, -3.03 USDC),
  `ETH-1700007300-900s` of `latency2` seed 2 (49 pairs at 1.0341, -1.67 USDC) and
  `ETH-1700005500-900s` of default seed 30 (22 pairs at 1.0030, -0.07 USDC).
* Root cause (traced snapshot by snapshot for the first): the bot held 35 UP at 0.61 and 68 DOWN at
  0.361, so DOWN was the heavy token and only UP counted as "completing". It then bought 56.6 more
  DOWN at 0.74, which the rules allow: a token that is not completing is bid at `fair - margin/2`,
  and the pair-cap rule only constrains the completing token. DESIGN 2.1 averages cost over paired
  and unpaired shares, so the pair cap for UP used the blended DOWN cost (0.533) instead of the
  marginal 0.74, and the later UP buys at 0.44 and 0.45 completed pairs that cost 1.048 and 0.983.
* Verdict: a documented approximation of the average-cost method. Total PnL is exact (recomputed
  independently from the fills, section 6.10); only the split between "merge PnL" and "sell PnL" of
  one market is attribution.

### 6.4 Maker fills are picked off even without informed flow (by design)

* Finding: size-weighted maker BUY markouts (mid of the token `h` seconds after the fill, minus the
  fill price, in cents per share) for default seed 1 are -1.10 at 0 s, -0.33 at 1 s, -0.09 at 5 s,
  +0.16 at 10 s, +0.26 at 30 s and +0.01 at 60 s. With `informed_flow=0` they are -0.45, -0.12,
  -0.27, -0.31, -0.80 and -0.75.
* Root cause: the exchange fills a resting order mainly through prints that go THROUGH its price
  (an at-the-touch print first drains the whole displayed queue, about 400 shares, against prints of
  about 30), and a print goes through only after the touch has already moved away from the order.
  This is the conservative queue assumption of DESIGN section 5.
* Verdict: by design.

### 6.5 Zero or very few fills (none unexpected)

* The default config fills 383 to 620 times per run (every one of the 960 markets of the 40-seed
  sweep traded). Lower counts appear exactly where they should: 176 with no sweeps, 227 and 117 at
  latency 2 and 3.
* In the 450-configuration random sweep 109 runs had zero fills. 57 had no prints at all
  (`uninformed_trades_per_sec=0` and `informed_flow=0`). 18 had every ladder level below the
  20-share minimum order (a 10-share clip), so the quoter correctly placed nothing instead of
  sending orders to be rejected. The other 34 had quotes but no fills: 26 had a latency of 2 or 3
  snapshots, and the other 8 (latency 1) each combined a touch that moved on 54% to 97% of ticks with
  an extreme fill model (`trade_fill_fraction=0.2` or `queue_ahead_fraction=2`) or a tight gate (at
  most 2 ticks of spread, or 2 orders per second). To make sure none of them hides a fill the
  exchange should have made, all 109 runs were instrumented: at no moment did a live resting order
  face a print that traded through its price (a through-print must fill); only 19 prints arrived
  exactly at an order's price, and those drain the queue first. Verdict: expected, no bug.

### 6.6 Fills on one side only (not found)

Over the 40-seed sweep the bot bought 771,915 UP and 770,305 DOWN shares (50.1% UP). Per market the
sides are within the net-imbalance caps.

### 6.7 Inventory piling up unhedged (bounded)

* Largest net imbalance seen: 300.0 with the overlay off (cap 300); 470 with it on (cap 300 plus 200
  directional); largest gross inventory 698 shares (cap 1,500 per side); largest capital at risk
  in a market 337 USDC (cap 1,500).
* Markets ending with 1 share or more unpaired: 8 of 960, each holding 2.8 to 5.0 shares (the minimum
  order is 5).
* The `holds` engine counter equals the number of markets in the default runs because it counts any
  unpaired remainder held at FLATTEN, dust included; it does not mean 24 directional bets.
* A larger unhedged exit did occur once (`informed0` seed 3, 40.9 UP shares). Flatten sells are IOC
  orders limited to the bid seen at the decision; in the last 25 seconds of a window the UP mid moves
  several ticks per second, and ten consecutive sells (at 876, 882, 888, 889, 893, 894, 895, 896, 897
  and 898 s) all missed because the bid at the next snapshot was lower than their limit; the window
  ended with the UP bid falling from 0.55 to 0.02 in the last three seconds. Across a default run,
  with a 1-snapshot latency the IOC completion buys fill 53% of their shares and the flatten sells
  83% (seed 1; 49% and 64% at latency 2; 100% and 93% at latency 0). Verdict: latency risk of
  IOC-at-the-touch, by design.

### 6.8 Merges, rejections, hot loops (not found)

* Merges failing: 0 in every run (`runner_stats.merge_failures`); cancel misses: 0.
* Orders rejected: 0 in all 54 matrix runs, all 40 sweep runs, 450 random configurations (cash from
  500 to 100,000, latency 0 to 3, tick 0.001 to 0.02, fees on and off, tiny caps and rate limits)
  and runs with a corrupted feed (dropped snapshots, missing, stale and future-stamped spot, empty,
  one-sided, zero-size and very wide books, enormous prints), with latency 0 and 1.
* Post-only cancels: 2,172 of 40,286 orders (5.4%) in default seed 1. They are not a loop: 2,016 are
  isolated, 78 repeat once at the same price within two snapshots, and the longest run of
  consecutive snapshots with a post-only cancel on one market and token is 6. They occur when the touch
  moves through a resting bid during the 1-snapshot latency and are spread across the whole window.

### 6.9 Cash reserved but unused, kill switch, determinism (no problem)

* Capital use is small: reserved cash in resting bids averages 207 USDC (maximum 420) and positions
  106 USDC (maximum 306) of 10,000, about 3% of the account, because the clip is a fixed 100 shares
  (`clip_equity_fraction` is unset). No capital cap ever binds (`risk_trips_capital` is 0).
* Kill switch: 0 trips in 40 default seeds, 1 in 6 in the no-lag variant (a genuine cumulative loss).
* Determinism: two runs of the same command give byte-identical `result.json`; identical under
  `PYTHONHASHSEED` 0, 1 and 12345 (now a test); `--no-invariants` and `--sample-every` change nothing
  except what they name; the sim -> JSONL -> `replay` result equals the direct run; and all 36 runs
  of the first six variants are identical to the same 36 runs made before this pass's one code change.

### 6.10 Independent audits that agreed with the engine

* Cash and PnL recomputed from `fills.csv` alone (buys, sells, the fee formula applied independently,
  merges at 1 USDC per pair, settlement at 1 USDC per winning share) match `total_pnl` to 1e-6 on 18
  runs spanning six variants.
* Report metrics (Brier, max drawdown, pair completion, average notional) recomputed from the raw
  result match on 12 runs.
* Every fill checked against the feed that produced it (now a test, `tests/test_integration.py`):
  maker fills need a print of that token at or through the fill price in the same snapshot and an
  order that was live for one snapshot before; taker fills need a displayed level with enough size.
  No violations at latency 0, 1, 2.
* Every order the engine emitted checked against DESIGN 7.7 to 7.9 (also now a test): no
  post-only order crosses at emission, every bid placed while holding unpaired opposite inventory
  satisfies `bid + avg_cost[opposite] <= 1 - target_margin`, nothing is placed in WARMUP/DONE, while
  the kill switch is latched or while the spot is stale.
* Changing the future never changes the past (a perturbed or truncated tail leaves every earlier
  fill identical; a deliberately peeking engine is detected), at latency 0 and 1.

### 6.11 Model against the market (Brier)

With all feature weights set to zero the model's Brier score equals the market mid's to within about
0.005 at every horizon, so the units, the volatility and the time-to-expiry handling are consistent with
the simulator (96 markets, 48 windows of BTC and ETH, at 100, 300, 500, 700 and 800 s into the
window: 0.2513, 0.2115, 0.1795, 0.0959, 0.0613 for the model against 0.2504, 0.2167, 0.1812, 0.0948,
0.0624 for the market). The uncalibrated feature weights make the raw model slightly worse at four
of those five horizons (for example 0.2665 at 100 s), partly because the depth imbalance feature is
noise on a simulator whose displayed depth carries no information (and, because the DOWN book is
the exact mirror of the UP book, `up imbalance - down imbalance` is just twice the UP imbalance).
Over 40 default seeds the engine's Brier score is 0.0731 against 0.0728 for the market.
Skill against the market is therefore indistinguishable from zero here. Mechanics only.

## 7. What this pass changed

| Change | File | Why | Test |
|---|---|---|---|
| `PaperExchange._check_invariants` now makes one pass over the open orders and builds its failure message only on failure; `_validate_book` and `_validate_trade` do the same | `src/abc_trading/exchange/paper.py` | profiling showed the self-check was the largest single cost (about 19% of a run): it made five passes over the orders and built several `repr`-heavy f-strings per order on every call. A BTC day went from 29.4 s to about 23 s. Behaviour is identical (36 runs compared before and after: identical `result.json`) | `tests/test_integration.py::test_the_exchange_asserts_its_own_invariants` (new; the five assertion paths of the self-check had no test before) and the whole of `tests/test_paper_exchange.py` |
| New whole-system tests | `tests/test_integration.py` | invariants of DESIGN section 7 that only exist once the modules are connected: 7.7 to 7.9 on real runs including kill-switch and stale-spot runs, fill legitimacy against the feed, no look-ahead (with a negative control), hash-seed determinism | 21 new tests, about 3.4 s |

No other source file was changed. In particular nothing in `types.py`, `config.py` or
`docs/DESIGN.md`, and no default parameter.

## 8. Known limitations and open issues (not fixed)

1. **Quote churn.** 80% of orders are gone one snapshot after placement, so fills depend on the
   latency and queue assumptions more than on the strategy (section 6.1). A tolerance or a
   discretionary-requote rule is a strategy decision, not an integration fix.
2. **The default strategy loses on this simulator** (mean -120.0 USDC per 12-window run over
   40 seeds), because the unpaired inventory it must dump costs more than the pair margin earns
   (section 6.2). Whether that carries over to real markets is unknown.
3. **Average-cost accounting** can attribute a negative margin to a merge (section 6.3) because it pools
   paired and unpaired shares; totals are exact.
4. **IOC at the stale touch** is unreliable under latency in the last seconds of a window (section 6.7).
5. **The model is uncalibrated** and is no better than the market mid here (section 6.11).
6. **Paper mode was never run against real endpoints.** The Polymarket, Binance and Coinbase paths
   and field names in `data/public_api.py` are unverified guesses (the sandbox cannot reach the
   hosts: `paper` exits with status 4 and a clear message). The slug template, the price-to-beat
   rule and the winner inference in `data/live.py` are approximations of the real venue.
7. **Packaging.** `pyproject.toml` names a `README.md` that does not exist (setuptools only warns).
   `python3 -m abc_trading` needs `PYTHONPATH=src` or an editable install; `pip install -e .` works
   in a virtual environment (checked) but `pip wheel .` fails in this sandbox because of the
   distribution's patched setuptools, which is unrelated to the project.
8. In the synthetic feed every print carries its snapshot's timestamp, so the exchange's
   `trade.ts >= live_ts` guard (a print older than the order's go-live moment must not fill it) is
   never exercised end to end; it is covered by two unit tests in `tests/test_paper_exchange.py`
   (removing the guard makes them fail), and live data will exercise it.
9. The kill switch measures equity (cash plus positions marked at the model's probability) against
   the starting equity of the run, not per calendar day, and once latched it never resets. A
   multi-day backtest is therefore judged against one `max_daily_loss_usd` for the whole run, and
   the report's drawdown (marked at market mids) can differ from the equity the switch sees.

## 9. How to reproduce

From the repository root (the package is not installed, so put `src` on the path, or
`pip install -e .` in a virtual environment and use `abc-trading`):

```
export PYTHONPATH=src
python3 -m abc_trading backtest --seed S --windows 12                                   # default, S = 1..6
python3 -m abc_trading backtest --seed S --windows 12 --assets BTC                      # 3.1
python3 -m abc_trading backtest --seed S --windows 12 --set sim.informed_flow=0         # 3.2
python3 -m abc_trading backtest --seed S --windows 12 --set sim.market_lag_seconds=0    # 3.3
python3 -m abc_trading backtest --seed S --windows 12 --set directional.enabled=false   # 3.4
python3 -m abc_trading backtest --seed S --windows 12 \
    --set fees.taker_fee_rate=1.0 --set fees.maker_fee_rate=0.01                        # 3.5
python3 -m abc_trading backtest --seed S --windows 96 --assets BTC                      # section 5
python3 -m abc_trading backtest --seed S --windows 36 --set sim.window_seconds=300      # section 4
```

Add `--out DIR` to write `result.json`, `config.json`, `fills.csv`, `markets.csv` and `equity.csv`.
`python3 -m pytest` runs the tests (about 18 s); `ruff check .`, `ruff format --check .` and
`mypy src` must stay clean.

The tables above were generated from the `result.json` files of these runs by a small script
(not part of the repository), so that no number was typed by hand. The exploratory scripts used for
section 6 (order-lifecycle statistics, markouts, flatten decomposition, the random-configuration and
corrupted-feed sweeps, the independent PnL audit) were scratch tools; their durable parts are the tests in
`tests/test_integration.py`.
