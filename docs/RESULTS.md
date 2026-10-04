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

A later review found that the first version of the paper exchange applied the order latency to
new orders but let a cancel take effect at once, which flattered fills and PnL. A cancel now takes
the same latency as an order (section 6.12). Every table and figure below was regenerated with
that model; section 6.12 keeps the instant-cancel figures they replace, so the size of the effect
can be seen, and says which statements of the first version no longer hold.

## 1. What was run

| Item | Value |
|---|---|
| Code | `HEAD` 0a97c6e plus the uncommitted working tree holding all modules (the lead commits) |
| Interpreter / machine | Python 3.11.15, Linux x86_64, 4-core Intel Xeon 2.1 GHz |
| Config | built-in `BotConfig()` defaults (identical to `configs/default.toml`, tested) unless a row says otherwise |
| Simulation | 12 windows of 900 s per asset, BTC and ETH (24 markets, 21,600 snapshots, 21,624 events) unless a row says otherwise |
| Account | 10,000 USDC initial cash, `latency_ticks=1`, `queue_ahead_fraction=1.0`, `trade_fill_fraction=1.0` |
| Checks | invariants 1-4 of DESIGN section 7 checked after every event in every run (never violated) |
| Test suite | 2,007 tests pass (about 30 s), `ruff check`, `ruff format --check` and `mypy src` clean |

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
| 1 | 12 x 900 s | BTC,ETH | 1255 | 36.01 | 95.7% | +338.98 | 440.6 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 665 | 35.30 | 90.7% | -744.96 | 1070.6 | 0.0931 | 0.0949 | TRIPPED |
| 3 | 12 x 900 s | BTC,ETH | 965 | 33.96 | 93.8% | -242.88 | 551.9 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 942 | 34.68 | 93.7% | -499.95 | 753.7 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 821 | 34.54 | 91.9% | -726.44 | 969.9 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 807 | 35.69 | 91.7% | -894.99 | 1034.5 | 0.0209 | 0.0239 | no |
| **mean** | | | 909 | 35.03 | 92.9% | **-461.71** (s.e. 185) | 803.5 | 0.0711 | 0.0688 | 1 of 6 |

How the PnL arises (same runs, all fully resolved so nothing is unrealised; merge, flatten-sell and
settlement PnL add up to the total, fees are already inside them):

| Seed | Merge PnL | Flatten-sell PnL | Settlement PnL (pair + directional) | Fees paid (inside) | Total PnL | Merges / flatten sells (orders) | Quotes placed / cancelled | Post-only cancels |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +1,723.3 | -1,386.0 | +1.7 | 135.3 | +339.0 | 365 / 20 | 27,504 / 26,327 | 944 |
| 2 | +738.5 | -1,482.6 | -0.8 | 82.7 | -745.0 | 174 / 29 | 16,797 / 16,142 | 550 |
| 3 | +1,170.8 | -1,396.4 | -17.3 | 104.2 | -242.9 | 262 / 48 | 24,919 / 24,029 | 740 |
| 4 | +1,072.7 | -1,573.1 | +0.5 | 84.8 | -500.0 | 261 / 26 | 23,878 / 23,018 | 678 |
| 5 | +847.9 | -1,575.2 | +0.8 | 87.0 | -726.4 | 220 / 32 | 21,965 / 21,247 | 570 |
| 6 | +852.2 | -1,748.2 | +1.0 | 88.9 | -895.0 | 224 / 35 | 23,616 / 22,818 | 636 |

Over **40 seeds** (`--seed 1..40 --windows 12`, same config) the default run gives a mean PnL
of **-538.5 USDC** (standard error 62.5, sd 395.1, median -478.2, range
-1,162.4 to +339.0), positive in 2 of 40 runs, and the kill switch trips in 9 of
them (a tripped run stops trading for the rest of its windows: 38 of the 960 markets have no
fills, all of them in 7 tripped runs). Fills per run 333 to 1,264; average trade
34.7 USDC (range 31.9 to 36.4, so the source claim of about 53 is not reproduced);
pair completion 92.7% on average (87.1% to 95.7%); mean max drawdown 806.7
(worst 1,281.9); mean merge PnL +1,054.4 and mean flatten-sell PnL -1,593.5; Brier score of
the engine 0.0731, of the raw model 0.0743, of the market mid 0.0728.
Read this as: on this simulator and with these uncalibrated defaults the strategy earns its pair
margin through merges and gives all of it (and a good deal more) back on the inventory it has to dump.
With instant cancels (the first version of the exchange) the same 40 seeds gave -120.0 USDC (positive in
15, no kill switch trip); section 6.12 compares the two.

## 3. Variants (seeds 1 to 6, 12 windows each)

| Variant | What changes | Fills (mean) | PnL mean (s.e.) | Seeds with PnL > 0 | Kill switch trips |
|---|---|---:|---:|---:|---:|
| default | nothing | 909 | -462 (185) | 1 of 6 | 1 of 6 |
| btc_only | `--assets BTC` | 494 | -183 (148) | 1 of 6 | 0 of 6 |
| informed0 | `sim.informed_flow=0` (no sweeps) | 528 | -22 (59) | 3 of 6 | 0 of 6 |
| lag0 | `sim.market_lag_seconds=0` | 916 | -874 (111) | 0 of 6 | 5 of 6 |
| nodir | `directional.enabled=false` | 878 | -769 (174) | 1 of 6 | 3 of 6 |
| feeheavy | `fees.taker_fee_rate=1.0`, `fees.maker_fee_rate=0.01` | 686 | -865 (80) | 0 of 6 | 3 of 6 |

Standard errors are across the six seeds, so differences between rows of less than about two
standard errors are noise. The Brier columns of `informed0`, `nodir` and `feeheavy` equal the
default's because those knobs change neither the prices of the feed nor the model.

### 3.1 BTC only (`--assets BTC`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC | 689 | 35.87 | 97.2% | +464.32 | 328.3 | 0.1448 | 0.1329 | no |
| 2 | 12 x 900 s | BTC | 449 | 34.79 | 91.5% | -494.54 | 668.6 | 0.0781 | 0.0762 | no |
| 3 | 12 x 900 s | BTC | 522 | 33.66 | 94.0% | -2.57 | 344.0 | 0.1281 | 0.1205 | no |
| 4 | 12 x 900 s | BTC | 376 | 33.70 | 92.0% | -434.08 | 592.9 | 0.0578 | 0.0587 | no |
| 5 | 12 x 900 s | BTC | 384 | 34.04 | 91.3% | -381.41 | 515.9 | 0.0207 | 0.0189 | no |
| 6 | 12 x 900 s | BTC | 545 | 36.07 | 94.0% | -249.88 | 526.6 | 0.0239 | 0.0321 | no |
| **mean** | | | 494 | 34.69 | 93.3% | **-183.03** (s.e. 148) | 496.0 | 0.0756 | 0.0732 | 0 of 6 |

Half the markets, so about half the fills (494 against 909); the mean is 1.2 standard errors below zero.

### 3.2 No adverse selection (`--set sim.informed_flow=0`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 686 | 15.71 | 92.1% | +99.23 | 297.5 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 546 | 15.41 | 87.5% | -206.96 | 354.4 | 0.0931 | 0.0949 | no |
| 3 | 12 x 900 s | BTC,ETH | 526 | 15.33 | 91.1% | +22.54 | 155.3 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 456 | 13.90 | 90.1% | +140.36 | 117.4 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 436 | 13.33 | 84.4% | -183.45 | 224.0 | 0.0131 | 0.0118 | no |
| 6 | 12 x 900 s | BTC,ETH | 517 | 15.74 | 88.9% | -6.46 | 145.0 | 0.0209 | 0.0239 | no |
| **mean** | | | 528 | 14.91 | 89.0% | **-22.46** (s.e. 59) | 215.6 | 0.0711 | 0.0688 | 0 of 6 |

With the informed sweeps switched off the only prints are random touch prints (0.3 per second per
token side, about 30 shares), so the bot is filled about half as often as in the default run, in small clips
(about 15 USDC), and pairs less completely (89%). The mean PnL is
indistinguishable from zero. The average maker BUY fill is still 0.71 cent above the mid at the moment
of the fill (section 6.4).

### 3.3 No market lag (`--set sim.market_lag_seconds=0`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 1954 | 37.60 | 95.9% | -447.63 | 625.2 | 0.1340 | 0.1357 | no |
| 2 | 12 x 900 s | BTC,ETH | 681 | 37.13 | 92.5% | -1002.14 | 1002.1 | 0.0911 | 0.0904 | TRIPPED |
| 3 | 12 x 900 s | BTC,ETH | 426 | 38.36 | 88.6% | -977.32 | 1221.7 | 0.1120 | 0.1154 | TRIPPED |
| 4 | 12 x 900 s | BTC,ETH | 1369 | 36.85 | 93.8% | -631.85 | 1278.2 | 0.0756 | 0.0729 | TRIPPED |
| 5 | 12 x 900 s | BTC,ETH | 390 | 34.80 | 87.0% | -1159.44 | 1189.6 | 0.0143 | 0.0139 | TRIPPED |
| 6 | 12 x 900 s | BTC,ETH | 679 | 37.04 | 91.8% | -1026.45 | 1129.4 | 0.0191 | 0.0194 | TRIPPED |
| **mean** | | | 916 | 36.96 | 91.6% | **-874.14** (s.e. 111) | 1074.4 | 0.0743 | 0.0746 | 5 of 6 |

The market now sees the spot with no delay, so the model has nothing to find and the sweeps still
pick the resting bids off. This is the worst variant: 5 of the six seeds trip the kill switch (equity 1,000 USDC
below the start), after which the remaining windows are not traded, so a tripped run ends close to
the limit rather than at a larger loss.

### 3.4 Directional overlay off (`--set directional.enabled=false`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 1498 | 35.40 | 96.6% | +11.94 | 465.7 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 950 | 34.26 | 93.1% | -1031.02 | 1031.0 | 0.0931 | 0.0949 | TRIPPED |
| 3 | 12 x 900 s | BTC,ETH | 922 | 35.59 | 93.4% | -908.08 | 1043.1 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 910 | 34.44 | 93.7% | -566.84 | 775.6 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 540 | 34.04 | 88.6% | -1088.09 | 1091.0 | 0.0131 | 0.0118 | TRIPPED |
| 6 | 12 x 900 s | BTC,ETH | 448 | 35.62 | 89.6% | -1034.31 | 1043.2 | 0.0209 | 0.0239 | TRIPPED |
| **mean** | | | 878 | 34.89 | 92.5% | **-769.40** (s.e. 174) | 908.3 | 0.0711 | 0.0688 | 3 of 6 |

Worse than the default in 6 of six seeds (paired by seed). The only difference is the overlay, which
uses the 2-second lag that the simulator builds in on purpose, so this gap is a property of the
simulator and not evidence of a real edge.

### 3.5 Fee-heavy (`--set fees.taker_fee_rate=1.0 --set fees.maker_fee_rate=0.01`)

| Seed | Windows | Assets | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 x 900 s | BTC,ETH | 1147 | 33.23 | 94.8% | -537.75 | 621.6 | 0.1211 | 0.1111 | no |
| 2 | 12 x 900 s | BTC,ETH | 639 | 32.65 | 90.9% | -949.98 | 1084.8 | 0.0931 | 0.0949 | TRIPPED |
| 3 | 12 x 900 s | BTC,ETH | 786 | 32.64 | 91.3% | -977.01 | 1007.4 | 0.1043 | 0.1005 | no |
| 4 | 12 x 900 s | BTC,ETH | 769 | 32.06 | 91.6% | -713.13 | 896.8 | 0.0739 | 0.0706 | no |
| 5 | 12 x 900 s | BTC,ETH | 442 | 30.95 | 86.4% | -1012.01 | 1023.2 | 0.0131 | 0.0118 | TRIPPED |
| 6 | 12 x 900 s | BTC,ETH | 332 | 32.08 | 85.6% | -1000.21 | 1010.7 | 0.0209 | 0.0239 | TRIPPED |
| **mean** | | | 686 | 32.27 | 90.1% | **-865.02** (s.e. 80) | 940.7 | 0.0711 | 0.0688 | 3 of 6 |

Taker fees four times the default and a 1% maker fee on notional. Fees paid rise from 83-135
to 134-545 USDC per run; the independent recomputation of cash from the raw fills (section 6.10)
agrees to 1e-6 with the fee formula applied independently.

## 4. Sensitivity runs: why the numbers look the way they do (not tuning)

These rows change one assumption at a time to expose a mechanism (section 6.1). They are NOT an
attempt to find better parameters, no default was changed, and with three seeds the PnL columns
are noise (the standard error of a three-seed mean is 120 to 313 USDC across these rows).

| Variant (seeds 1-3) | Fills (mean) | Avg trade (USDC) | Pair completion | PnL per seed (USDC) | PnL mean | Quotes placed per snapshot | Post-only cancels (mean) | Rate-limited placements (mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| default (latency 1, tolerance 0) | 962 | 35.09 | 93.4% | +339, -745, -243 | -216 | 1.07 | 745 | 23 |
| `exchange.latency_ticks=0` | 1340 | 38.34 | 96.4% | +598, -131, +95 | +187 | 1.33 | 0 | 43 |
| `exchange.latency_ticks=2` | 1194 | 36.14 | 93.7% | -324, -1043, -1077 | -815 | 0.78 | 911 | 20 |
| `exchange.latency_ticks=3` | 1198 | 36.04 | 92.7% | -493, -1020, -804 | -772 | 0.73 | 1471 | 30 |
| `sizing.requote_tolerance_ticks=1` | 1109 | 35.38 | 94.8% | +487, -195, -420 | -42 | 0.97 | 739 | 8 |
| `sizing.requote_tolerance_ticks=2` | 980 | 35.95 | 94.5% | +388, -322, -424 | -119 | 0.81 | 633 | 5 |
| `sim.window_seconds=300`, 36 windows | 1229 | 34.61 | 90.3% | -658, -1073, -884 | -872 | 0.85 | 1300 | 21 |

The fill count does not collapse as the latency goes from 0 to 3 snapshots (1,340, 962, 1,194,
1,198): a cancel takes as long as an order, so a quote that is re-priced every snapshot stays
hittable for `latency_ticks` snapshots after the engine dropped it. What changes with latency is the
PnL (+187, -216, -815, -772 for latency 0 to 3), because the stale quotes are the ones the price is
leaving. Every number the backtest prints is therefore still dominated by the latency and queue
assumptions of the paper exchange. (With instant cancels this sweep read 1,340 / 514 / 227 / 117
fills, "a factor of 11": largely an artefact, section 6.12.)

## 5. Full simulated days and speed

| Run | Assets | Windows | Markets | Snapshots | Wall time | Fills | Avg trade (USDC) | Pair completion | PnL (USDC) | Max drawdown (USDC) | Brier model | Brier market | Kill switch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BTC day, seed 1 | BTC | 96 x 900 s | 96 | 86,400 | 16.1 s | 2450 | 34.70 | 94.0% | -1073.64 | 1596.9 | 0.0518 | 0.0493 | TRIPPED |
| BTC day, seed 2 | BTC | 96 x 900 s | 96 | 86,400 | 12.4 s | 922 | 35.42 | 91.3% | -919.86 | 1021.1 | 0.1013 | 0.1016 | TRIPPED |
| BTC day, seed 3 | BTC | 96 x 900 s | 96 | 86,400 | 15.1 s | 1992 | 34.20 | 93.4% | -1017.92 | 1195.1 | 0.0923 | 0.0895 | TRIPPED |
| BTC+ETH day, seed 1 | BTC,ETH | 96 x 900 s | 192 | 172,800 | 26.9 s | 2010 | 34.54 | 93.5% | -966.58 | 1698.2 | 0.0662 | 0.0640 | TRIPPED |
| BTC day, seed 1, kill switch off | BTC | 96 x 900 s | 96 | 86,400 | 21.7 s | 4361 | 34.93 | 94.2% | -1227.05 | 1786.4 | 0.0518 | 0.0493 | no |
| BTC day, seed 2, kill switch off | BTC | 96 x 900 s | 96 | 86,400 | 21.2 s | 3959 | 34.86 | 93.5% | -2423.69 | 2552.3 | 0.1013 | 0.1016 | no |
| BTC day, seed 3, kill switch off | BTC | 96 x 900 s | 96 | 86,400 | 22.6 s | 4041 | 34.67 | 93.8% | -1595.52 | 1772.7 | 0.0923 | 0.0895 | no |
| BTC+ETH day, seed 1, kill switch off | BTC,ETH | 96 x 900 s | 192 | 172,800 | 49.9 s | 8461 | 34.89 | 94.2% | -2666.81 | 3709.6 | 0.0662 | 0.0640 | no |

The first four rows are the default configuration. Over a full simulated day the kill switch trips
(its loss limit is measured against the start of the run, not per calendar day, section 8), after
which nothing is traded, so those wall times are shorter than for a day that is traded throughout.
The last four rows switch it off (`--set risk.max_daily_loss_usd=1e12`, a timing device and not a
configuration to trade; the CLI prints a `warning:` on stderr that the kill switch can never fire)
so that the whole day is traded; they lose 1,227 to 2,667 USDC, more than
the 1,000 USDC the default configuration tolerates.

One BTC asset-day traded throughout (96 windows, 86,400 snapshots) takes about 21-23 s with the
invariant checks on (about 0.25 ms per snapshot); two assets take 49.9 s, so about 25 s per
simulated asset-day. The cost per snapshot does not grow with the length of the run (0.269, 0.234,
0.241 and 0.264 ms per snapshot for 12, 48, 96 and 192 BTC windows without invariant checks, kill
switch off). Before the first pass a BTC day took 29.4 s and a two-asset day 67.4 s (the exchange's
self-check rebuilt long error messages on every call; section 7).

That sentence was only true up to a few hundred markets. `Portfolio.capital_at_risk()` and
`Portfolio.equity()` run several times per snapshot and used to walk every inventory the run had ever
created, settled ones included, so a snapshot cost in proportion to the markets already settled and a
run was quadratic in its markets. They now walk only the unsettled inventories (DESIGN 4.1), and
the sums keep their order, so every number of every run is unchanged. The engine's cost per snapshot
over the last 5% of a run against the first 5% (60 s windows, one BTC asset, kill switch off, no
invariant checks, seed 2) was x1.6, x3.3 and x6.3 for 200, 800 and 1,600 markets (the 1,600-market run
took 52 s) and is now x1.0, x1.2 and x1.3 (19 s). What is left of the 20-30% is the interpreter's
cyclic garbage collector walking the events the timing script keeps in memory, not a scan (x0.94 at
800 markets with the collector switched off). A `capital_at_risk() + equity({})` pair after 100,000
settled markets takes 0.0005 ms where it took 21 ms (0.002 ms with 4 markets open). At the 96 and
192 markets of the table the gain is small (a 96-window BTC day with the invariant checks on: 22.5 s
to 21.1 s; BTC and ETH: 53.9 s to 44.5 s on the same machine), and the wall times in the table were
measured before the change. The durable guard is
`test_portfolio_totals_do_not_visit_settled_inventories`; the timings came from a throwaway script
around `engine.on_snapshot`.

## 6. Review of anomalies and root causes

The pass looked for the problems listed below. Where a problem was found, its cause is given and the
verdict says whether it is a bug (fixed, with a test) or what the strategy does under the
simulator's assumptions (documented, parameters not touched). No functional defect was found in
the modules' integration: the invariant checks, the independent audits of section 6.10, 450
randomised configurations and 48 corrupted-data runs all came back clean. (A later review found two
flaws of the paper exchange that none of these checks could see, because neither breaks an
accounting rule: cancels took no time, section 6.12, and the mirrored books did not share IOC
depth, section 6.13. A third review found that the report of a run with open markets mixed resolved-only and
all-market figures and overstated what had been checked, section 6.14. A fourth found that a scarce cash
budget and a binding rate limit were handed out UP-first instead of to the token that completes pairs,
section 6.15.)

### 6.1 Quote churn and the fill count (by design under the simulator's assumptions)

* Finding: the default run places 1.27 quotes and cancels 1.22 per snapshot (seed 1: 27,504 placed and
  26,327 cancelled over 21,600 snapshots). Of the 27,504 post-only orders, **79.1% lasted one snapshot** (the engine
  cancelled them at the next snapshot, or a fill or the exchange's post-only re-check ended them then;
  14.6% lasted two, 3.1% three, 3.2% four or more), 93.4% ended by a cancel that landed, 3.4% by the exchange's
  post-only re-check and 3.2% by a complete fill; 1,035 (3.8%) ever received a fill.
* Root cause: (1) in this simulator the book moves almost every second. The UP best bid changes on
  73.1% of ticks (84.6% for either side; a move of two ticks or more on 37.1% of ticks), because
  a binary option's price moves by about `0.4 / sqrt(seconds left)` per second (1.6 ticks per second
  with 600 s left, 4 ticks with 100 s left). (2) The ladder is priced off the touch and
  `requote_tolerance_ticks` is 0, so any move re-prices it. (3) An order placed at snapshot `t`
  goes live at `t+1`; if the engine re-prices at `t+1`, its cancel lands at `t+2`, so it can still
  meet the prints of `t+2`. Every order that goes live is matchable for at least `latency_ticks`
  snapshots after the engine decided to drop it: of the 26,560 orders that went live, 78.8% were
  matchable for exactly one snapshot, 14.8% for two, 3.1% for three and 3.3% for four or more.
  (With instant cancels the first group was never matchable at all, and the first version of this
  section called that "by design"; section 6.12 explains why it was an artefact.)
* Consequence: 805 of the 1,110 maker fills (72.5%), carrying 68.7% of the maker shares, came from
  orders the engine had already cancelled (seeds 2 and 3: 74.3% and 73.7%). They are worse than the
  other fills: the size-weighted maker BUY markout against the token's mid is -1.40 cents at the moment of
  the fill against -1.07 for the others, -1.00 against -0.04 five seconds later and -0.71 against +0.49
  after thirty (section 6.4).
* Evidence for the mechanism (three-seed means, section 4): raising `requote_tolerance_ticks` to 1 and 2 cuts quotes
  placed per snapshot from 1.07 to 0.97 and 0.81, with fills of 962, 1,109 and 980
  (no trend beyond noise); and the latency sweep gives 1,340, 962, 1,194 and 1,198 fills at latency 0, 1, 2 and 3
  with PnL +187, -216, -815 and -772.
* Verdict: not a bug. It is how the default parameters interact with a fast book and a 1-snapshot
  latency, and it has a price once cancels take time: the quotes it leaves behind are hit, and they
  are the adversely selected ones. Left as is; the rate limiter (20 messages per second, cancels
  included) clipped 35 of 27,504 placements (0.13%), all of them quote placements, never a flatten sell or a
  taker completion.

### 6.2 Merges versus flatten sells (by design)

* Finding: in every variant merge PnL is positive and flatten-sell PnL is negative, and the second is
  usually larger (default seed 1: +1,723.3 and -1,386.0; the second is larger in five of the six seeds of section 2).
* Root cause: pairs complete when the price comes back through the second bid, and they pay a
  margin of 3.8 cents per merged pair here (0.9622 all-in cost, against a 1 cent target); what is
  left unpaired is the inventory whose opposite bid the price never came back to, so it is selected
  to have moved against the bot. Decomposing the flatten sells
  exactly (seeds 1 to 3, 4,085 / 4,517 / 4,224 shares): -1,386.0 / -1,482.6 / -1,396.4 USDC is
  -1,361.3 / -1,441.0 / -1,365.0 of price drift (mid at the sale minus average cost), -24.6 / -38.1 / -30.9
  of spread paid against the mid and 0.003 / 3.56 / 0.45 of fees. So about 98% of the loss is
  inventory risk and only about 2% is execution cost.
* Verdict: what the strategy does under these assumptions; no parameter touched.

### 6.3 Pair cost above 1 (not seen in these runs; accounting attribution, not a bug)

* Finding: none of the 2,184 markets of the 88 runs of this file merged pairs at an average cost
  above 1 (a default run merges at about 0.96). The first version of this file, which ran with instant
  cancels, found three such markets among 2,328: `BTC-1700001000-900s` of `informed0` seed 1 (124 pairs at 1.0244,
  -3.03 USDC), `ETH-1700007300-900s` of `latency2` seed 2 (49 pairs at 1.0341, -1.67 USDC) and
  `ETH-1700005500-900s` of default seed 30 (22 pairs at 1.0030, -0.07 USDC). Different fills mean
  different markets, so their absence here says nothing about the mechanism below.
* Root cause (traced snapshot by snapshot for the first of those): the bot held 35 UP at 0.61 and 68 DOWN at
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
  fill price, in cents per share) for default seed 1 are -1.29 at 0 s, -1.12 at 1 s, -0.70 at 5 s,
  -0.70 at 10 s, -0.33 at 30 s and -0.30 at 60 s. With `informed_flow=0` they are -0.71, -0.66,
  -0.65, -0.60, +0.09 and +0.11.
* Root cause: the exchange fills a resting order mainly through prints that go THROUGH its price
  (an at-the-touch print first drains the whole displayed queue, about 400 shares, against prints
  of about 30), and a print goes through only after the touch has already moved away from the order.
  This is the pessimistic queue assumption of DESIGN section 5. A second effect appeared once
  cancels took time: 72.5% of the default run's maker fills come from orders the engine had already cancelled, and
  their markouts are worse (-1.40 at the fill and -1.00 after 5 s, against -1.07 and -0.04 for the rest).
* Verdict: by design.

### 6.5 Zero or very few fills (none unexpected)

* The default config fills 333 to 1,264 times per run. Of the 960 markets of the 40-seed
  sweep, 922 traded; the other 38 are the windows that a tripped kill switch left
  untraded (7 of the 9 tripped runs have such windows). Lower counts appear exactly where they should: 528 fills with no
  sweeps (six-seed mean) and 494 with a single asset.
* In a fresh 450-configuration random sweep (cash 500 to 100,000, latency 0 to 3, tick 0.001 to 0.02,
  fees on and off, tiny capital caps and rate limits, 2 windows of 300 s; the sweep of the first version
  of this file was not kept, so this is a new one, not a re-run) 85 runs had zero fills. 34 had no prints at all
  (`uninformed_trades_per_sec=0` and `informed_flow=0`). 50 had every ladder level below the 20-share minimum order (a
  10-share clip), so the quoter correctly placed nothing instead of sending orders to be rejected. The remaining one had quotes but no
  fills: latency 3, `queue_ahead_fraction=2`, `trade_fill_fraction=0.5` and a 2-tick spread gate. To make sure none of
  the other runs hides a fill the exchange should have made, all 450 runs were instrumented: whenever a live,
  non-cancelled resting order faced a print that traded through its price, the best-priority such order
  was filled (3,316 occasions, 0 exceptions). Verdict: expected, no bug.

### 6.6 Fills on one side only (not found)

Over the 40-seed sweep the bot bought 1,338,846 UP and 1,337,372 DOWN shares (50.0% UP). Per market the net
imbalance is not bounded by the caps, see 6.7.

### 6.7 Inventory piling up unhedged (bounded softly)

* Largest net imbalance seen: 407.8 with the overlay off (cap 300); 583.5 with it on (cap 300 plus 200
  directional); largest gross inventory 1,106 shares (cap 1,500 per side); largest capital at risk
  in a market 550.5 USDC (cap 1,500). The first two are above their caps (300 and 470 were seen with instant cancels):
  the caps bound what the engine rests, and an order it has cancelled can still fill until its
  cancel lands (6.12).
* Markets ending with 1 share or more unpaired: 14 of 960, 13 of them holding 1.2 to 4.8 shares (less
  than the 5-share minimum order, so they cannot be sold) and one holding 43.2.
* The `holds` engine counter is 900 over the 960 markets of the default sweep because it counts any
  unpaired remainder held at FLATTEN, dust included; it does not mean 900 directional bets.
* The larger unhedged exit (default seed 3, `ETH-1700001900-900s`, 43.2 UP shares): flatten sells are IOC
  orders limited to the bid seen at the decision; in the last 25 seconds of a window the UP mid moves
  several ticks per second, and ten consecutive sells (at 876, 882, 888, 889, 893, 894, 895, 896, 897
  and 898 s) all missed because the bid at the next snapshot was lower than their limit; the window
  ended with the UP bid falling from 0.55 to 0.02 in the last three seconds. Across a default run,
  with a 1-snapshot latency the IOC completion buys fill 46% of their shares and the flatten sells
  88% (seed 1; 42% and 62% at latency 2; 100% and 93% at latency 0). Verdict: latency
  risk of IOC-at-the-touch, by design.

### 6.8 Merges, rejections, hot loops (not found)

* Merges failing: 0 in all 88 runs of this file and in the 450 random configurations
  (`runner_stats.merge_failures`); 1 in the 48 corrupted-feed runs (seed 5, latency 3): a completion buy
  (IOC) was still in flight when FLATTEN began, so the engine sized a flatten sell of the UP shares on the
  pre-fill inventory; when the buy filled, the merge it then planned could not use the UP shares that the
  in-flight sell had reserved and was refused (the shares were sold instead). It involves no cancel and is a
  latency-3 interplay of two in-flight IOC orders in the engine, not an exchange error; not fixed here. Cancel misses
  (the runner asking to cancel an order that was already gone): 0 everywhere, because an order whose cancel
  is pending is not reported by `open_orders`.
* Orders rejected: 0 of 1,884,246 orders in all 88 runs of this file, 0 in the 450 random configurations
  (cash from 500 to 100,000, latency 0 to 3, tick 0.001 to 0.02, fees on and off, tiny caps and rate
  limits) and 0 in 48 runs with a corrupted feed (3 windows of BTC and ETH, latency 0 to 3, with dropped
  snapshots, missing, stale and future-stamped spot, empty, one-sided, zero-size and very wide books and enormous
  prints). Cash-limited accounts rely on the exchange freeing the cash of an order whose cancel is
  pending for the orders sent behind the cancel (DESIGN 5): 420 further random two-asset runs with cash from 60 to
  1,500 USDC, latency 0 to 3, maker fees and rebates and 1 to 5 ladder levels also had no `insufficient_cash`.
* Correction (later audit): the zero-rejection claim above did not hold for a positive
  `fees.maker_fee_rate` combined with an account small enough for cash to bind (about 500 USDC or
  less for two markets; for example `maker_fee_rate=0.001`, cash 300, seed 1, 4 windows gave 1,745
  `insufficient_cash` rejections of 8,824 orders). The engine's belief of the other market's
  reservation counted `price * remaining` but not the maker fee that the exchange also commits, so
  the later market's ladder was sized past the free cash and its last levels were refused every
  snapshot. Fixed in `strategy/engine.py` (the reservation of a resting bid now includes a positive
  maker fee, and a maker fill releases it) and pinned by `tests/test_engine_cash_reservation.py`
  and two cash-limited two-market tests in `tests/test_backtest_e2e.py`. With a zero maker fee, or
  with cash that never binds (every run reported in this document: default, and the fee-heavy
  variant at 10,000 USDC), the change is a no-op: the fills of the six fee-heavy runs of section
  3.5 and of 24 further default, rebate, fee-with-ample-cash and zero-fee tight-cash runs were
  byte-identical before and after that engine change (on the exchange as it was then; the figures of
  this file also carry the later cancel-latency change of 6.12).
* Post-only cancels: 944 of 27,504 orders (3.4%) in default seed 1. They are not a loop: 914 are
  isolated, 15 repeat once at the same price within two snapshots, and the longest run of
  consecutive snapshots with a post-only cancel on one market and token is 4. They occur when the touch
  moves through a resting bid during the 1-snapshot latency and are spread across the whole window.

### 6.9 Cash reserved but unused, kill switch, determinism (no problem)

* Capital use is small: reserved cash in resting bids averages 172 USDC (maximum 423) and positions
  159 USDC (maximum 326) of 10,000 (default seed 1), about 3% of the account, because the clip is a fixed 100 shares
  (`clip_equity_fraction` is unset). No capital cap ever binds (`risk_trips_capital` is 0 in all 88 runs).
* Kill switch: 9 trips in 40 default seeds, 5 in 6 in the no-lag variant (genuine cumulative losses; a tripped run stops quoting,
  keeps merging and flattening, and trades nothing afterwards). With instant cancels the default seeds never tripped it.
* Determinism: two runs of the same command give byte-identical `result.json`; identical under
  `PYTHONHASHSEED` 0, 1 and 12345 (also a test); `--no-invariants` and `--sample-every` change nothing
  except what they name (the same fills, the same totals; the sampled drawdown moves with the sampling); the sim -> JSONL -> `replay` result equals the direct run
  apart from its source label.

### 6.10 Independent audits that agreed with the engine

* Cash and PnL recomputed from `fills.csv` alone (buys, sells, the fee formula applied independently,
  merges at 1 USDC per pair, settlement at 1 USDC per winning share) match `total_pnl` to 1e-6 (largest
  difference 7e-11, fee formula difference 0) on 18 runs spanning six variants.
* Report metrics (Brier, max drawdown, pair completion, average notional) recomputed from the raw
  result match (largest difference 6e-14) on the same 18 runs.
* Every fill checked against the feed that produced it (now a test, `tests/test_integration.py`):
  maker fills need a print of that token at or through the fill price in the same snapshot and an
  order that was live for one snapshot before; taker fills need a displayed level with enough size.
  No violations at latency 0, 1, 2. A second test (`test_a_cancelled_order_can_fill_only_until_its_cancel_lands`)
  checks that an order the engine cancelled is filled at most `latency_ticks` snapshots after the cancel
  was sent, and at latency 0 never afterwards.
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

### 6.12 Cancels took no time (found by a later review; fixed)

* Finding: `PaperExchange.cancel()` removed an order at once, while `submit()` made a new order wait
  `latency_ticks` snapshots before it could go live. The engine sees the snapshot at which an order
  goes live and could cancel a stale quote before any print of the next interval could hit it. A real
  cancel is a message like an order and needs about as long to reach the venue, so the quote can still
  be hit while the cancel is on its way. Nothing disclosed the asymmetry: DESIGN section 5 and the README
  described the latency of new orders only and called the exchange conservative. No invariant or audit
  could see it, because instant cancels break no accounting rule; it is a modelling optimism that
  flatters fills and PnL, and the mechanism behind the first version of 6.1 ("an order that is
  re-priced at t+1 never had a chance").
* Fix (`exchange/paper.py`, contract in `exchange/base.py`, DESIGN 4.4 and 5): with `latency_ticks > 0` a
  cancel lands after `latency_ticks` snapshots of the order's market. Until then the order stays in the
  exchange, matchable against prints (including the prints of the snapshot at which the cancel lands),
  but it is hidden from `open_orders` (so the engine does not cancel it again; the new
  `pending_cancels()` lists it). A cancel never overtakes the order it targets, so an in-flight order
  still activates first and an in-flight IOC still executes. A cancel that finds its order already gone
  is counted in the new `cancel_too_late` statistic; `cancelled` now counts orders removed by a cancel that
  landed. The funds of an order with a pending cancel are free for the orders sent behind it (they
  reach the venue after the cancel), and a fill of the pending order is capped to what the funds left
  by the other orders can pay for, so a cash-limited account can still replace its ladder in one
  snapshot and cash and positions stay non-negative. At latency 0 a cancel is instant, as before.
* Size of the effect. The "instant cancels" columns below were re-measured with the exchange as it was before
  this fix (and before 6.13, which changes none of these runs: checked on seeds 1 to 3 at latency 0, 1 and 2),
  and the others with the current one, so the difference is the cancel latency alone. The instant-cancel figures
  are the ones the first version of this file printed:

| `exchange.latency_ticks` (seeds 1-3) | Fills, instant cancels | PnL mean, instant cancels | Fills, cancels take the latency | PnL mean, cancels take the latency |
|---:|---:|---:|---:|---:|
| 0 | 1340 | +187 | 1340 | +187 |
| 1 (default) | 514 | -39 | 962 | -216 |
| 2 | 227 | -91 | 1194 | -815 |
| 3 | 117 | -9 | 1198 | -772 |

  Default configuration, seeds 1 to 6, PnL (USDC): -76, -168, +126, +48, -256, -304 with instant cancels
  (mean -105.1, kill switch tripped 0 times) against +339, -745, -243, -500, -726, -895 (mean -461.7, tripped
  1 time). Over the 40 seeds the mean goes from -120.0 (s.e. 38.8, positive in
  15, no trips) to -538.5 (s.e. 62.5, positive in 2, 9 trips): the paired difference per seed is
  -418.5 USDC (s.e. 56.2), worse in 36 of 40 seeds, and fills per run go from 498 to 920.
  805 of the 1,110 maker fills of default seed 1 (72.5%) are on orders the engine had already cancelled.
* What no longer holds from the first version of this file: that an order re-priced at `t+1` never had a chance
  and that only about 20% of the orders live long enough to be matchable (6.1); that fills fall by a
  factor of 11 from latency 0 to 3 (section 4 and the README; now 1,340 / 962 / 1,194 / 1,198); that the
  kill switch never trips in the default seeds (section 2, 6.9); that net imbalance respects its caps (6.7); and the
  headline PnL of every table, which is lower in 36 of the 40 default seeds. What holds unchanged: the latency-0 row (an instant
  cancel is correct at latency 0; 1,340 fills and +187 in both columns), the Brier columns (they do not depend on fills),
  the conclusion that the default strategy loses on this simulator (now by more), and every accounting
  invariant.
* Tests (`tests/test_paper_exchange.py`, "cancel latency" section, `tests/test_integration.py` and `tests/test_backtest_e2e.py`): the finding's
  scenario (a through-print after the cancel was sent still fills); a cancel landing after exactly
  1, 2 and 3 snapshots; funds, fees, positions and caps; second cancel refused; too late (filled, IOC,
  post-only); in-flight orders; per-market clock; settle; `pending_cancels` copies; the random-walk invariants with
  cancels pending; and an end-to-end run in which every fill of a cancelled order falls within `latency_ticks`
  snapshots of its cancel, which fails with instant cancels; cash-limited two-asset runs at latency 1 to 3 in which no
  order is refused for cash. Two existing tests encoded the instant cancel
  (`test_cancel_semantics` and `test_reserved_cash_is_sum_of_buy_notional_including_in_flight`) and were rewritten
  (the first for latency 0, the second for the new reservation semantics); the random-walk invariant test gained
  checks on pending cancels.

### 6.13 Mirrored books did not share IOC depth (found by a later review; fixed)

* Finding: the depth an IOC consumes was tracked per `(token, side, level price)`, so a BUY of UP at 0.49 and
  a SELL of DOWN at 0.51 (the same pool: DESIGN 1) could each take the full 100 shares of a 100-share
  level in one snapshot (200 shares out of 100). The same for a SELL of UP at the bid and a BUY of DOWN at the
  mirrored ask.
* Fix (`exchange/paper.py::_pool`): depth is keyed by UP-book coordinates, `(takes the UP ask, UP price)`; a SELL of
  DOWN at `b` maps to the UP ask at `1 - b`, a BUY of DOWN at `a` to the UP bid at `1 - a`.
* Effect on this file: none. The engine never submits both IOC legs of a mirrored pair in one snapshot
  (completion buys and flatten sells are one order per market per snapshot, and run in different phases); the fill
  digests of seeds 1 to 3 at latency 0, 1 and 2 are identical with the old and the new key. It was a latent flaw of
  the exchange, relevant to any strategy that takes both legs in one snapshot.
* Tests: six tests in `tests/test_paper_exchange.py` (both mirrored pairs, either order, the remainder of a
  partly used pool, independence of the other side and other levels, two in-flight IOCs activating together).

### 6.14 The report mixed populations and said more than it had checked (found by a later review; fixed)

* Finding: a run that ends with unresolved markets (every `paper` run that stops inside a window,
  `replay --allow-unresolved`) printed a "Per-market PnL" block that summed merge, sell, settlement and fee PnL over
  the resolved markets only, next to a Total PnL, a pairing section and a markets table that cover all markets. A
  session shorter than one window showed a non-zero Total PnL with every decomposition row at 0.00 and, in the same
  report, thousands of merged pairs at a positive margin. "Markets resolved N (M traded)" paired the resolved count with
  the all-markets traded count, "Markets ending unhedged" counted markets that had not ended, and the Run block read
  "Invariants checked (DESIGN section 7, 1-4)" although invariant 4 is not checked when markets are open. The README said
  every printed number "has passed" the invariants, which also overstated them for clean runs (Brier, drawdown and
  pair completion are not covered). The `paper` banner, stderr line and `--help` did not say that the data client is
  unverified or that winners are inferred, and the caveat called the fee parameters "defaults".
* Fix (`backtest/runner.py::pnl_stats`, `backtest/report.py`): the decomposition rows are sums over every market (an
  open market contributes its realised sell and merge PnL and fees, no settlement), and a new `pnl_unrealised`
  (`final equity - initial cash - Σ market pnl`) makes the rows add up to Total PnL. The mean, spread, Sharpe-like,
  best / worst and win-rate rows stay on the resolved markets and the report says so ("Statistics cover"). "Markets
  resolved" reads `N of M (K traded)` with K among the resolved. An open run's report says `checked 1-3; 4 (final
  accounting) skipped`, names the unrealised part of Total PnL, and calls the unhedged inventory "at close or
  stream end". Paper output carries the unverified-endpoints / inferred-winners caveat, and every report calls the fee
  parameters placeholders, not Polymarket's verified schedule.
* Related (`cli.py`): argparse accepted any unique prefix of an option, so `paper --i` passed the
  `--i-understand-this-is-paper-only` guard. The parser and every subparser now use `allow_abbrev=False`.
* Effect on this file: none on any number. Every run above is fully resolved, where the decomposition sums are
  unchanged and the unrealised part is zero. The report format changed (`24 of 24 (24 traded)`, one more PnL row).
* Tests: `tests/test_backtest_report.py` (a truncated run and a partly resolved run, the rows against the per-market
  records and against Total PnL, the traded count, the invariants, unhedged and caveat wording),
  `tests/test_backtest_runner.py` (`pnl_stats` by hand, the scripted unresolved run), `tests/test_cli.py` (abbreviations
  of the acknowledgement and of other options are usage errors; the paper help and stderr carry the caveat).
  `test_pnl_stats_by_hand` encoded the old behaviour ("unresolved: excluded" from the sums) and was changed.

### 6.15 A scarce cash budget and rate-limit quota went UP-first (found by a later review; fixed)

* Finding: `compute_quotes` spent the cash budget on UP's whole ladder before DOWN's, and the engine ranked the
  placements the rate limiter let through by (level among the still-missing levels, UP before DOWN). Neither looked at
  which token completes pairs. With UP heavy the heavy UP ladder took the whole budget and the completing DOWN bids got
  nothing (a flat state with a binding budget quoted UP only), while the mirror state, DOWN heavy, worked by accident.
  Two states that differ only by the UP / DOWN labels therefore traded differently when cash or quota was scarce:
  with a 60 USDC budget and 150 shares held, the bids were UP 128.3 / DOWN 0 shares when UP was heavy and
  UP 122.9 / DOWN 0 when DOWN was heavy (the completing token starved in the first). On a 300 USDC account the same run
  with the labels swapped (books, prints, winner mirrored, spot mapped to `C^2/spot`) gave -197.6 / -224.4 / -272.2 /
  +52.2 USDC (seeds 1 to 4, 3 windows) against +27.9 / -225.8 / +9.4 / -148.7.
* How much it matters: nothing at the documented operating point. The account is 10,000 USDC and about 3% of it is
  used (6.9), so the budget does not bind; with the rate limiter lifted the fills of default seeds 1 to 4 are
  byte-identical before and after (same `fills_sha256`). It bites on a small account with the fixed 100-share clip:
  over seeds 1 to 6 (4 windows, limiter lifted) the pair completion rate was 43.6% -> 66.2% at 150 USDC, 68.7% -> 82.6%
  at 300 USDC and 90.8% -> 92.2% at 600 USDC (mean PnL -27.2 -> -34.2, -182.5 -> -154.3, -150.9 -> -146.8: noise, six
  seeds of an uncalibrated simulator say nothing about profit).
* Fix (`strategy/quoter.py::compute_quotes`, `strategy/engine.py::_place_quotes`): the budget is handed out in
  priority order, not label order. (1) A token that completes pairs is funded first, all its levels (it reduces risk and
  locks the pair profit). (2) The other tokens are funded level by level, every token's level 0 before any level 1, and
  within a level the remaining budget is split evenly (a bid that costs less than an equal share is funded whole, the
  rest goes to the other token; two bids that cannot both rest a minimum-size order are both left out of that level
  instead of giving all to UP). The returned list is still UP's levels then DOWN's, so with an ample budget every run is
  unchanged. The rate limiter's priority is the same: completing token, then the level's index in the whole desired
  ladder (the old index among the still-missing levels ranked UP's level 1 before DOWN's level 0 whenever UP's level 0
  already rested), then UP before DOWN on an exact tie. DESIGN 2.2(4) and 2.8 say so.
* Label symmetry afterwards: the reviewer's script gives bit-identical PnL for the original and the label-swapped market
  on the unconstrained account, on the default config (seeds 3 and 9) and on 300 USDC (-260.8 / -195.0 / -11.7 / +9.2,
  seeds 1 to 4). The mirror states of the quoter give mirror quotes for every budget tested (tests below). The one exact
  tie that remains is the exchange-side order of two orders of the same snapshot (UP is emitted first); it did not
  change a single result in these runs.
* Effect on this file: none from the cash budget (see above). The rate-limit rank changes which placements are dropped
  where the limiter bites, so a default run that drops placements follows a different path: with the CLI (`backtest
  --seed S --windows 12`) seeds 1, 3 and 4 give +359.90, -335.42 and -472.82 USDC (they were +338.98, -242.88 and -499.95),
  seeds 2, 5 and 6 are identical, the six-seed mean moves from -461.7 to -469.1 (no systematic direction: the standard
  error is 185). The tables of sections 2 to 5 and the 40-seed figures were generated before this change and have not
  been regenerated.
* Tests: `tests/test_strategy_quoter.py` (flat budget split evenly, completing token first and its mirror, max-min fair
  split, no tie-break when the budget cannot rest both sides, 144 parametrised mirror states and an 800-state random
  mirror-symmetry property with a binding budget), `tests/test_strategy_engine.py` (rate limiter: completing token
  first, ladder depth not index among the missing levels). Existing tests that encoded the UP-first allocation were
  changed, not weakened: `test_cash_budget_is_allocated_sequentially_across_levels` (replaced by the even-split test),
  the small-budget cases of `test_zero_or_tiny_budget_yields_no_quotes` and of
  `test_maker_fee_reduces_the_cap_and_is_charged_to_the_budget` (now run on UP alone, which is what they check, and
  assert the two-sided result separately), and in `test_strategy_engine.py` the per-market-capital, total-capital,
  freed-budget, taker-fee and in-flight-IOC tests. The arithmetic they pin (the budget is never exceeded, the fee is
  charged, an in-flight IOC's cash is not reused) is kept; only the expected quotes are the new split.

## 7. What this pass changed

| Change | File | Why | Test |
|---|---|---|---|
| `PaperExchange._check_invariants` now makes one pass over the open orders and builds its failure message only on failure; `_validate_book` and `_validate_trade` do the same | `src/abc_trading/exchange/paper.py` | profiling showed the self-check was the largest single cost (about 19% of a run): it made five passes over the orders and built several `repr`-heavy f-strings per order on every call. A BTC day went from 29.4 s to about 23 s. Behaviour is identical (36 runs compared before and after: identical `result.json`) | `tests/test_integration.py::test_the_exchange_asserts_its_own_invariants` (new; the five assertion paths of the self-check had no test before) and the whole of `tests/test_paper_exchange.py` |
| New whole-system tests | `tests/test_integration.py` | invariants of DESIGN section 7 that only exist once the modules are connected: 7.7 to 7.9 on real runs including kill-switch and stale-spot runs, fill legitimacy against the feed, no look-ahead (with a negative control), hash-seed determinism | 21 new tests, about 3.4 s |
| Cancels take `latency_ticks` like orders; a cancelled order stays matchable until its cancel lands, is hidden from `open_orders` and frees its funds for later orders; `pending_cancels()`, `cancel_too_late` | `src/abc_trading/exchange/paper.py`, `exchange/base.py` | the first version let a cancel take effect at once, which flattered fills and PnL (6.12) | `tests/test_paper_exchange.py` (cancel latency section), `tests/test_integration.py::test_a_cancelled_order_can_fill_only_until_its_cancel_lands` and `tests/test_backtest_e2e.py::test_cash_limited_runs_with_slow_cancels_are_never_refused_for_cash` (new) |
| IOC depth is shared by the mirrored UP and DOWN books | `src/abc_trading/exchange/paper.py` | the two books are one liquidity pool (DESIGN 1); depth was consumed per token (6.13) | six tests in `tests/test_paper_exchange.py` (new) |
| The PnL decomposition covers every market and gains an unrealised row; unresolved runs are labelled as such (invariants 1-3, mark-to-mid, stream end); paper output says unverified / inferred; no option can be abbreviated | `src/abc_trading/backtest/runner.py`, `backtest/report.py`, `cli.py` | the report mixed resolved-only and all-market figures and said "1-4" when invariant 4 was skipped; the paper acknowledgement guard accepted `--i` (6.14) | `tests/test_backtest_report.py`, `tests/test_backtest_runner.py`, `tests/test_cli.py` (new and changed tests) |
| A scarce cash budget goes first to the token that completes pairs, then level by level and evenly between the other tokens; the rate limiter places in the same priority | `src/abc_trading/strategy/quoter.py`, `strategy/engine.py`, DESIGN 2.2(4) and 2.8 | both were handed out UP-first, so a small account was quoted one-sided and the completing token starved when UP was heavy (6.15) | `tests/test_strategy_quoter.py`, `tests/test_strategy_engine.py` (new and changed tests) |
| Mutation-tested strategy: five one-line changes of trading behaviour (a resting bid blocking the taker completion; no completion or no merge in WIND_DOWN; FLATTEN merging only from `merge_min_pairs`; the WIND_DOWN room of the favoured token ignoring what is held) and the dropped order-rate check on FLATTEN sells, the dropped `room` test of a hard stop under the requote throttle, the missing taker fee in the hold-to-resolution rule, and sizing on initial cash instead of equity, or marks at the raw model probability, all passed the whole suite | tests only (no source or default changed) | the existing tests started every completion with no order of ours open, never merged or completed outside ACCUMULATE, used 60 pairs (above `merge_min_pairs`) in FLATTEN, started the WIND_DOWN room tests flat, varied `initial_cash` where equity equals it, and returned `p_up_model == p_up` from the stub | `tests/test_killers_engine.py` (new: phase by inventory decision table, real-run floors, the order-rate limit on a real two-asset run), and new cases in `tests/test_strategy_engine.py`, `test_strategy_rebalance.py`, `test_strategy_quoter.py`; `test_integration.py` now asserts floors (completing bids, taker buys, FLATTEN sells) near the observed counts instead of `> 0`. `StubModel` returns a `p_up_model` different from `p_up` |
| Asymmetry-tested: eleven one-line changes that swapped a token or ignored a token's book or direction (the Brier "market" score read from the DOWN mid; the equity curve marking UP at the DOWN mid and the reverse; `avg_pair_cost_at_close` as a difference and the per-market pair completion rate netting merged against held pairs; the spread limit applied to one book only; the order-book feature without the DOWN book; the taker-completion allowance honoured for UP only; in-flight FLATTEN sells of DOWN not recognised; the compounding clip priced at the UP mid; the WIND_DOWN room of a DOWN-favoured token; the latency-0 queue position read from the UP book) and the deleted or loosened final-accounting check of invariant 4 all passed the whole suite | tests only (no source or default changed) | the fixtures mirrored the UP book on the DOWN token (mid 0.50 on both), gave the DOWN book zero imbalance, held UP-heavy inventories under UP-favoured targets and tested the pair metrics only with no pair held at close; the model's randomised oracle was fed the model's own `book_imbalance` and the quoter property used the implementation's `books_quotable` as its oracle; the one invariant-4 test corrupted cash on both sides, which invariant 2 catches first, and accepted either message | `tests/test_killers_asymmetry.py` (new: a directed test per rule on both tokens, plus label-symmetry properties: swapping UP and DOWN in a whole state swaps the answer of the quoter, the completion and flatten planners, the book gate, the engine in every phase and the paper exchange at latency 0 to 2, on randomised books that do not mirror each other; `types.ceil_price` pinned), and new cases in `test_backtest_runner.py` (metrics on unmirrored books, pairs held at close, invariant 4 on its own and its tolerance), `test_strategy_risk.py`, `test_model.py` (hand-worked DOWN imbalance, independent oracle) and `test_strategy_quoter.py` (independent quotability oracle) |

The first pass changed no other source file: nothing in `types.py`, `config.py` or `docs/DESIGN.md`, and no
default parameter. The later changes recorded in this file are: the engine's reservation of a maker fee
(6.8, `strategy/engine.py`); the cancel latency (6.12) in `exchange/paper.py` and `exchange/base.py`, with
the contract written into DESIGN 4.4 and 5, a docstring note in `strategy/engine.py`, the comment of
`exchange.latency_ticks` in `config.py` and `configs/default.toml`, and the caveat text of the report
(`backtest/report.py`); the mirrored IOC depth (6.13) in `exchange/paper.py`; the report and CLI wording (6.14),
which added two fields to `PnLStats` (`n_markets_resolved_traded`, `pnl_unrealised`); and the priority of a scarce
cash budget and of the rate-limit quota (6.15) in `strategy/quoter.py` and `strategy/engine.py`, with DESIGN 2.2(4) and
2.8; and the asymmetry audit above, which changed tests only and added one test convention to DESIGN 8 and the README.
None of them changed a default parameter.

## 8. Known limitations and open issues (not fixed)

1. **Quote churn.** 79% of orders are cancelled one snapshot after placement and, because a cancel takes as long
   as an order, stay hittable meanwhile (72.5% of the maker fills come from such orders), so fills and PnL depend on the
   latency and queue assumptions more than on the strategy (sections 6.1, 6.12). A tolerance or a
   discretionary-requote rule is a strategy decision, not an integration fix.
2. **The default strategy loses on this simulator** (mean -538.5 USDC per 12-window run over
   40 seeds, positive in 2, kill switch tripped in 9), because the unpaired inventory it must dump costs more than the
   pair margin earns (section 6.2). Whether that carries over to real markets is unknown.
3. **Average-cost accounting** can attribute a negative margin to a merge (section 6.3) because it pools
   paired and unpaired shares; totals are exact.
4. **IOC at the stale touch** is unreliable under latency in the last seconds of a window (section 6.7).
5. **The model is uncalibrated** and is no better than the market mid here (section 6.11).
6. **Paper mode was never run against real endpoints.** The Polymarket, Binance and Coinbase paths
   and field names in `data/public_api.py` are unverified guesses (the sandbox cannot reach the
   hosts: `paper` exits with status 4 and a clear message). The slug template, the price-to-beat
   rule and the winner inference in `data/live.py` are approximations of the real venue.
7. **Packaging.** `python3 -m abc_trading` needs `PYTHONPATH=src` or an editable install; `pip install -e .` works
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
10. **Caps are soft.** The inventory, net-imbalance and capital caps bound what the engine rests; an order it has
    cancelled can still fill until the cancel lands, so they can be exceeded (6.7: net imbalance up to 584 shares).
11. **The cancel model is still an assumption.** A cancel can only fail by losing a race to a fill (there are no
    cancel rejections, reorderings or acknowledgement delays), and on a cash- or position-limited account a fill
    of an order with a pending cancel is capped in favour of its replacement (DESIGN 5); with ample cash that
    never binds.
12. **Two in-flight IOC orders can conflict at latency 3** (6.8): a flatten sell sized before a completion buy
    filled can reserve the shares of the merge planned right after. Seen once in 48 corrupted-feed runs, never in
    the 88 uncorrupted runs; not fixed.

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
python3 -m abc_trading backtest --seed S --windows 96 --assets BTC \
    --set risk.max_daily_loss_usd=1e12                                                   # section 5, whole day traded
python3 -m abc_trading backtest --seed S --windows 36 --set sim.window_seconds=300      # section 4
```

Add `--out DIR` to write `result.json`, `config.json`, `fills.csv`, `markets.csv` and `equity.csv`.
`python3 -m pytest` runs the tests (about 30 s); `ruff check .`, `ruff format --check .` and
`mypy src` must stay clean.

The tables above were generated from the `result.json` files of these runs by a small script
(not part of the repository), so that no number was typed by hand. The exploratory scripts used for
section 6 (order-lifecycle statistics, markouts, flatten decomposition, the random-configuration and
corrupted-feed sweeps, the independent PnL audit) were scratch tools; their durable parts are the tests in
`tests/test_integration.py`.
