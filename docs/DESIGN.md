# ABC Trading — design spec

Backtest / paper-trading framework for a **two-sided market-making + full-pair accumulation +
inventory-rebalancing** strategy on Polymarket's short-dated BTC/ETH **Up/Down** markets.

This document is the contract between modules. `src/abc_trading/types.py` and
`src/abc_trading/config.py` are the frozen code half of that contract.

## 0. Scope and honesty rules

* **No live trading.** There is no order-signing code, no private keys, no wallet, no live
  exchange adapter. `exchange/base.py` defines the `Exchange` protocol so a live adapter *could*
  be written later; writing one is explicitly out of scope.
* **Read-only network code is allowed** (public order books, public spot prices) but is
  optional, off by default, never used in tests, and **unverified against live endpoints**
  (the build sandbox blocks those hosts). Tests must run fully offline.
* **Synthetic results validate mechanics, not profitability.** The simulator contains a
  deliberate market lag so the directional model has something to find; it is circular by
  construction. Nothing here claims an edge exists in real markets. Docs must say so.
* Fee parameters are placeholders to be verified. Defaults are not calibrated.
* Source claim being implemented (unverified): a Polymarket account reportedly ran ~$53
  average-size trades on BTC/ETH Up/Down markets using bidirectional market making + full-pair
  accumulation + inventory rebalancing, with a fair-probability model (spot data, order-book
  depth, momentum, acceleration, volatility) used to keep part of the directional exposure.

## 1. Market mechanics

* A window market (5/15 min) asks "will the underlying close the window at or above where it
  started?". Tokens: **UP** and **DOWN**. At resolution the winner pays $1, the loser $0
  (ties resolve UP).
* **Pair** = 1 UP + 1 DOWN. A pair is worth exactly $1 at resolution *whatever happens*, and
  can be **merged** into $1 any time before resolution (CTF `mergePositions`), freeing capital.
* Hence if you buy UP at `a` and DOWN at `b` (all-in incl. fees) the pair's profit is locked:
  `1 - a - b` per pair. E.g. UP@0.47 + DOWN@0.52 locks 1c gross.
* The two books are mirrors of one liquidity pool: a SELL of UP at x is the same trade as a
  BUY of DOWN at 1-x. The simulator emits both mirrored prints (see `types.Trade`).
* Maker orders add liquidity (fee 0 by default); taker orders pay `FeeConfig` taker fee.
* Cash must be reserved for resting bids (`price * remaining`), so ladders tie up capital.

## 2. Strategy (per market; engine handles many markets and a global risk book)

### 2.1 Accounting (`inventory.py`)
* Cost basis is **all-in**: buys add `price*size + fee`; average cost per share =
  `cost[token] / qty[token]`. Sells remove cost pro-rata (average-cost method) and realise
  `proceeds - fee - removed_cost`.
* `paired_qty = min(qty[UP], qty[DOWN])`; `net_shares = qty[UP] - qty[DOWN]`;
  `unpaired()` = (heavy token, |net|) or (None, 0).
* `locked_profit() = paired_qty * (1 - avg_cost[UP] - avg_cost[DOWN])` (0 if either side empty).
  Average-cost accounting treats paired and unpaired shares as fungible — an acknowledged
  approximation, documented, and exact once the market is flat or fully paired.
* Merge of `s` pairs removes `s` shares from each token at their average cost, credits `$s`.
* Settlement credits `$1 * qty[winner]`; PnL split into pair part
  `paired*(1-cu-cd)` and directional part (the unpaired remainder's payout minus its cost).
* Identity: `cash + capital_at_risk - initial_cash == realised_pnl` at all times (before
  settlement of open positions), where `capital_at_risk = Σ cost basis of open qty`.

### 2.2 Quote construction (`strategy/quoter.py`, pure function)
For each token `T` (opposite `O`) produce a **ladder of post-only bids** (we only *rest* bids;
exits/rebalances are IOC takers):

1. **Cap price** (never pay more than this):
   * if we hold unpaired `O` (qty[O] > qty[T]): `cap_T = 1 - target_margin - avg_cost[O]`
     (completes pairs with ≥ margin locked; "marginal pair" view);
   * else (flat/balanced): `cap_T = fair_T - target_margin/2` where `fair_T` is the model
     probability for T. Then `cap_UP + cap_DOWN = 1 - target_margin` exactly, so two fills at
     their caps lock `target_margin`.
2. **Directional tilt** (only if `directional.enabled` and the model has `valid` data):
   when `target_net` (see 2.4) asks for more UP (> current net), UP is the "favoured" token:
   its cap may rise to `fair_T - directional.min_edge`, and the unfavoured token's bids are
   reduced (size scaled down) / removed once the target is reached. Tilt never exceeds
   `max_directional_shares` of deliberate net exposure.
3. **Inventory skew**: shift bids on the heavy side down by
   `skew_ticks_per_100_shares * |net| / 100` ticks; hard stop (no bid on heavy side) once
   `|net| >= max_net_imbalance_shares` (plus the directional allowance in the favoured
   direction) or `qty[T] >= max_inventory_per_side_shares`.
4. **Ladder**: level `k` (0-based) price = `min(cap, best_bid - k*spacing*tick)` snapped DOWN
   to tick, and at most `best_ask - tick` (post-only; never cross). If `cap < best_bid` the
   quote rests below the touch (patient). Skip levels < 1 tick or ≤ 0 price. Size =
   `clip * ladder_size_decay**k`, floored to `min_order_size` lot, reduced to fit remaining
   per-side inventory room, per-market capital cap and available cash.
5. No quotes when: phase is WARMUP/DONE/FLATTEN, risk gate trips, book empty/crossed/wider
   than `max_spread_ticks_to_quote`, or model invalid *and* no unpaired inventory to complete.
   In WIND_DOWN only the token that **completes pairs** (and, within the directional budget,
   the model-favoured token) is quoted.

### 2.3 Merge
Whenever `paired_qty >= merge_min_pairs` (or any pairs in FLATTEN) emit
`MergePairs(floor(paired_qty))`. Merging realises the locked profit and frees cash, which is
the compounding loop ("repeat the same cycle").

### 2.4 Directional overlay (`strategy/directional.py`)
`edge_T = fair_T - ask_T_effective` (model prob minus price you'd pay, taker-side). If
`edge >= min_edge`, target deliberate net exposure toward T with binary Kelly
`f* = (p - c) / (1 - c)` (p = model prob, c = price), scaled by `kelly_fraction * equity / c`,
clipped to `max_directional_shares`. Returned as a signed `target_net` (UP positive).
Zero when disabled / model invalid / edge < min_edge. No target ever exceeds the caps.

### 2.5 Taker rebalancing (`strategy/rebalance.py`)
If unpaired qty `U >= rebalance_trigger_shares` (and not in a directional hold):
* **Complete the pair** by buying `O` IOC at its ask when the net pair profit per share,
  `1 - avg_cost[T] - (ask_O + taker_fee_per_share(ask_O))`, is
  `>= taker_lock_margin - rebalance_max_loss_per_pair`; size ≤
  min(U minus the deliberate directional allowance, displayed ask size, cash, capital cap).
  With defaults this only ever completes pairs at a net profit.
* Otherwise leave to passive quoting + skew; the FLATTEN logic resolves what is left.

### 2.6 Phases (`strategy/phases.py::phase_for`)
`since = ts - start`, `left = end - ts`:
* `since < warmup_seconds` → WARMUP
* `left <= flatten_seconds` → FLATTEN (if `ts >= end` → DONE)
* `left <= wind_down_seconds` → WIND_DOWN
* otherwise ACCUMULATE.

**FLATTEN**: cancel all resting orders; merge all pairs; for the unpaired remainder `U` of
token `T`: *hold to resolution* iff directional enabled, `valid` model, and
`p_T - (bid_T - taker_fee_per_share(bid_T)) >= hold_margin` and held shares ≤
`max_directional_shares`; otherwise **sell** the excess via IOC at the bid. Selling is skipped
if the bid is empty/≤ 0. DONE: nothing but waiting for `MarketResolved`.

### 2.7 Risk (`strategy/risk.py`)
Global gate evaluated every snapshot (any trip ⇒ cancel quotes for the market, place nothing
except merges/flattening):
* spot missing or older than `max_spot_staleness_seconds` (relative to snapshot ts),
* **kill switch** latched once `equity - starting_equity <= -max_daily_loss_usd`
  (cancel everything, stop trading; merges allowed),
* per-market capital (`cost basis + reserved by open bids`) ≤ `max_capital_per_market_usd`,
  total ≤ `max_total_capital_usd`,
* order-rate limiter ≤ `max_orders_per_second` (cancels count),
* a requote throttle: do not re-quote a market more often than `min_requote_interval_seconds`.

### 2.8 Reconcile
Engine diffs desired quotes against `open_orders`: keep an order whose token/side matches and
price within `requote_tolerance_ticks` (preserves queue position) and size within 20% of
desired; cancel the rest; place missing. Cancels first, then places.

## 3. Fair-value model (`model/fair_value.py`)

Per asset keep an EWMA of squared log-returns (half-life `vol_halflife_seconds`) →
`sigma` per sqrt(second), floored at `vol_floor`. For snapshot at `ts` with `spot`,
`ref_price`, `tau = max(end - ts, min_tau_seconds)`:

```
z      = ln(spot / ref) / (sigma * sqrt(tau))                # driftless GBM, P(UP) = Phi(z)
mom    = ln(spot / spot[ts-L]) / (sigma * sqrt(L)),  L = momentum_lookback_seconds
accel  = mom_short - mom_long  (short = accel_lookback, long = momentum_lookback, both
         normalised the same way)
obi    = up_book.imbalance - down_book.imbalance  clipped to [-1, 1]   (order-book depth)
z_adj  = z + w_momentum*mom + w_accel*accel + w_book_imbalance*obi
p_model = clip(Phi(z_adj), p_floor, 1 - p_floor)
p_up    = (1 - shrink_to_market) * p_model + shrink_to_market * up_mid
```
`valid=False` (and `p_up` = UP mid else 0.5) when spot/ref missing, <2 observations or the
lookback history is not yet available. No look-ahead: only observations with ts ≤ now.
Weights are uncalibrated heuristics; `fit_logistic_weights` + `brier_score`/`log_loss` exist so
calibration quality can be *measured*; backtest reports the model's Brier score.

## 4. Module map and public API (exact)

`src/abc_trading/`:

| file | owner wave | public API |
|---|---|---|
| `types.py`, `config.py` | contract | see files |
| `fees.py` | W1-A | `FeeModel(cfg: FeeConfig)`: `.fee(price, size, is_maker) -> float`, `.taker_fee_per_share(price) -> float`, `.all_in_buy_cost(price, is_maker) -> float` |
| `inventory.py` | W1-A | `PnLBreakdown`, `MarketInventory`, `Portfolio` (below) |
| `model/fair_value.py` | W1-B | `norm_cdf`, `FairValue`, `FairValueModel`, `fit_logistic_weights`, `brier_score`, `log_loss`; re-exported from `model/__init__.py` |
| `sim/feed.py` | W1-E | `SyntheticFeed(cfg: BotConfig)` — iterable of `FeedEvent` |
| `data/events.py` | W1-G | `event_to_dict`, `event_from_dict`, `write_jsonl(path, events)`, `read_jsonl(path) -> Iterator[FeedEvent]` |
| `data/public_api.py` | W1-G | `Transport` protocol, `UrllibTransport`, `PolymarketPublicClient`, `SpotClient` (read-only; parsing unit-tested with fake transport) |
| `data/live.py` | W1-G | `LiveFeed` — polls the public clients, yields `FeedEvent` (paper mode only) |
| `exchange/base.py`, `exchange/paper.py` | W2-D | `Exchange` protocol, `PaperExchange(cfg: BotConfig)` |
| `strategy/phases.py`, `quoter.py`, `directional.py`, `rebalance.py`, `risk.py`, `engine.py` | W2-C | see §4.3 |
| `backtest/runner.py`, `metrics.py`, `report.py` | W3-F | `run_backtest`, `BacktestResult`, `format_report` |
| `cli.py`, `__main__.py` | W3-F | `abc-trading backtest|replay|record|paper` |

### 4.1 `inventory.py`
```python
@dataclass(frozen=True, slots=True)
class PnLBreakdown:
    market_id: str
    sell_pnl: float            # realised on sells (proceeds - fee - removed cost)
    merge_pnl: float           # realised on merges (size - removed cost)
    settle_pair_pnl: float     # paired remainder at settlement: paired*(1 - cu - cd)
    settle_directional_pnl: float  # unpaired remainder: payout - cost
    fees_paid: float           # informational; already inside the pnl numbers above
    total: float               # sell + merge + settle_pair + settle_directional

class MarketInventory:
    market_id: str
    qty: dict[Outcome, float]; cost: dict[Outcome, float]
    fees_paid: float; sell_pnl: float; merge_pnl: float; settled: bool
    def __init__(self, market_id: str) -> None
    def apply_fill(self, fill: Fill) -> None          # BUY adds qty/cost; SELL removes pro-rata (ValueError if > qty)
    def apply_merge(self, size: float) -> float       # returns merge pnl; ValueError if size > paired_qty
    def apply_settlement(self, winner: Outcome) -> PnLBreakdown   # zeroes positions, sets settled
    def avg_cost(self, token: Outcome) -> float | None
    paired_qty: float (property); net_shares: float (property)
    def unpaired(self) -> tuple[Outcome | None, float]
    def locked_profit(self) -> float
    def capital_at_risk(self) -> float
    def value_at(self, p_up: float) -> float

class Portfolio:
    initial_cash: float; cash: float; inventories: dict[str, MarketInventory]
    def __init__(self, initial_cash: float) -> None
    def inventory(self, market_id: str) -> MarketInventory      # get-or-create
    def apply_fill(self, fill: Fill) -> None                    # also moves cash (+/- fee)
    def apply_merge(self, result: MergeResult) -> float
    def apply_settlement(self, s: Settlement) -> PnLBreakdown   # cash += s.payout
    def capital_at_risk(self) -> float
    def realised_pnl(self) -> float                             # cash + capital_at_risk - initial_cash
    def equity(self, marks: Mapping[str, float]) -> float       # cash + Σ value_at(p_up mark); no mark -> valued at cost
    def reserved_cash(self, open_orders: Iterable[OpenOrder]) -> float   # Σ BUY price*remaining
    def available_cash(self, open_orders: Iterable[OpenOrder]) -> float  # cash - reserved
```
Reject nonsense loudly (`ValueError`) rather than silently clamping: negative sizes, selling
more than held, merging more than paired, filling a settled market.

### 4.2 `model/fair_value.py`
```python
@dataclass(frozen=True, slots=True)
class FairValue:
    p_up: float; p_up_model: float
    z: float; z_adj: float; sigma: float; tau: float
    momentum: float; accel: float; book_imbalance: float
    valid: bool
    def p(self, token: Outcome) -> float          # p_up or 1 - p_up

class FairValueModel:
    def __init__(self, cfg: ModelConfig) -> None
    def observe(self, asset: str, ts: float, spot: float) -> None   # idempotent per (asset, ts); ts must be non-decreasing
    def vol(self, asset: str) -> float | None
    def estimate(self, snap: MarketSnapshot) -> FairValue          # does NOT mutate state
def norm_cdf(x: float) -> float
def fit_logistic_weights(samples: Sequence[tuple[Sequence[float], int]], *, l2: float = 1e-3, iters: int = 500, lr: float = 0.1) -> list[float]
def brier_score(preds: Sequence[float], outcomes: Sequence[int]) -> float
def log_loss(preds: Sequence[float], outcomes: Sequence[int], eps: float = 1e-9) -> float
```

### 4.3 `strategy/`
```python
# phases.py
def phase_for(snap_ts: float, market: MarketSpec, cfg: TimingConfig) -> Phase

# quoter.py
@dataclass(frozen=True, slots=True)
class QuoteLevel: token: Outcome; price: float; size: float
def compute_quotes(*, snap: MarketSnapshot, fair: FairValue, inv: MarketInventory, phase: Phase,
                   cfg: BotConfig, fees: FeeModel, clip_shares: float, cash_budget: float,
                   target_net: float) -> list[QuoteLevel]
def max_bid_for_token(*, token: Outcome, fair: FairValue, inv: MarketInventory, cfg: BotConfig,
                      target_net: float) -> float | None     # the cap of 2.2(1)+(2); None = do not bid

# directional.py
def binary_kelly(p: float, price: float) -> float
def directional_target(*, snap: MarketSnapshot, fair: FairValue, inv: MarketInventory,
                       cfg: BotConfig, fees: FeeModel, equity: float) -> float   # signed target net shares

# rebalance.py
def plan_completion(*, snap, inv, fair, cfg, fees, cash_budget, target_net) -> list[OrderRequest-like tuple]
def plan_flatten(*, snap, inv, fair, cfg, fees) -> FlattenPlan   # merge size, IOC sells, hold decision

# risk.py
class RiskManager:
    def __init__(self, cfg: RiskConfig, starting_equity: float) -> None
    def update_equity(self, equity: float) -> None            # latches kill switch
    kill_switch: bool (property)
    def spot_ok(self, snap: MarketSnapshot) -> bool
    def book_ok(self, snap: MarketSnapshot, cfg_tick_limit: int) -> bool
    def allow_orders(self, ts: float, n: int) -> int          # rate limiter; returns how many of n may be sent now
    def capital_ok(self, market_capital: float, total_capital: float, extra: float) -> bool

# engine.py
class MarketMakerEngine:
    def __init__(self, cfg: BotConfig, *, initial_cash: float | None = None) -> None
    portfolio: Portfolio; fees: FeeModel; model: FairValueModel; risk: RiskManager
    stats: dict[str, int]                                       # counters (quotes_placed, ...)
    def on_snapshot(self, snap: MarketSnapshot, open_orders: Sequence[OpenOrder]) -> list[Action]
    def on_fill(self, fill: Fill) -> None
    def on_merge(self, result: MergeResult) -> None
    def on_settlement(self, s: Settlement) -> PnLBreakdown
    def marks(self) -> dict[str, float]                         # last p_up per market (for equity)
    def equity(self) -> float
    def last_fair(self, market_id: str) -> FairValue | None     # read-only, for reporting
    def phase_of(self, market_id: str) -> Phase | None          # read-only, for reporting
```
The exact helper signatures in 4.3 may be refined by the implementer; the **engine API** and
`compute_quotes`/`phase_for`/`RiskManager` shown are fixed. Document any deviation in the
module docstring and the hand-off note.

### 4.4 `exchange/`
```python
class Exchange(Protocol):
    def submit(self, req: OrderRequest, ts: float) -> SubmitResult
    def cancel(self, order_id: str, ts: float) -> bool
    def merge(self, market_id: str, size: float, ts: float) -> MergeResult | None
    def open_orders(self, market_id: str | None = None) -> list[OpenOrder]
    def process(self, snap: MarketSnapshot) -> list[Fill]
    def balance(self) -> float
    def position(self, market_id: str, token: Outcome) -> float

class PaperExchange:           # implements Exchange; also:
    def __init__(self, cfg: BotConfig) -> None
    def settle(self, market_id: str, winner: Outcome, ts: float) -> Settlement
    reserved_cash: float (property); stats: dict[str, int]
```

## 5. Paper exchange fill model (conservative by design)

* `submit` validates: price strictly in (0,1) and tick-aligned; `size > 0` and
  `>= min_order_size` (if enforced); BUY needs `cash - reserved >= price*size` (+ worst-case
  taker fee for IOC); SELL needs held minus already-reserved sells. POST_ONLY that would cross
  the last known book is rejected. Accepted orders are **in flight** for `latency_ticks`
  *snapshots of that market* (`OpenOrder.live=False`), then become live.
* `process(snap)` order of operations (per market):
  1. match **already-live** resting orders against `snap.trades`;
  2. activate in-flight orders whose delay elapsed: POST_ONLY re-checked against `snap` books
     (crossing ⇒ cancelled, counted, no fill); queue position set to
     `queue_ahead_fraction * displayed size at our price` (0 if we improve the touch / level
     absent); IOC executes immediately against `snap` books (walk levels ≤ limit, shared
     per-snapshot depth consumption, taker fee, `taker_slippage_ticks`);
  3. return fills (in order).
  An order never fills from prints that occurred before it was live (no look-ahead).
* Resting BUY at `p` vs SELL-aggressor print at `tp` on the same token:
  `tp < p` → through-fill (we are ahead): fills `min(remaining, print_avail)` at **our** price;
  `tp == p` → print first drains `queue_ahead`, remainder fills us; `tp > p` → no fill.
  Per-print availability `= size * trade_fill_fraction` is shared across our orders (better
  price first, then older first). Symmetric for resting SELL vs BUY-aggressor prints.
* Maker fills pay `fees.fee(.., is_maker=True)` (rebate negative). Fills never exceed
  remaining; partial fills keep the order resting.
* `merge`: requires both positions ≥ size; positions −size, cash +size. `settle`: cancels
  open orders in that market (releases reservations), credits `winner_qty * 1`, zeroes positions.
* Cash never goes negative; reservations never exceed cash (asserted).

## 6. Synthetic market model (`sim/feed.py`)

* Spot per asset: correlated GBM (BTC/ETH correlation 0.8 via a shared factor), per-tick vol
  `annual_vol * sqrt(tick_seconds / 31_557_600)`. Windows back-to-back from a fixed epoch
  aligned to `window_seconds`; `ref_price` = spot at window start; winner UP iff
  `spot_end >= ref`.
* Market UP mid = `Phi(ln(spot_lagged/ref) / (sigma_mkt * sqrt(tau)))` where `spot_lagged` is spot
  `market_lag_seconds` ago and `sigma_mkt = sigma * market_vol_multiplier`, plus AR(1) noise of
  `pricing_noise_ticks`, snapped to tick, clipped to [tick, 1 - tick].
* UP book: spread ≥ 1 tick (random around `mean_spread_ticks`), `n_levels` levels with
  random depth around `depth_mean_shares`; DOWN book is the exact mirror
  (`down_bid = 1 - up_ask`, `down_ask = 1 - up_bid`, sizes mirrored).
* **Trades** (always both mirrored prints, same size): *informed sweeps* — if the UP bid
  fell by k ticks since last tick, SELL prints on UP at each consumed old bid level (and the
  mirrored BUY prints on DOWN); symmetric for ask rises; scaled/gated by `informed_flow`
  (this creates adverse selection: fills happen right before price moves against you).
  *Uninformed flow* — Poisson(`uninformed_trades_per_sec * tick_seconds`) touch prints per token
  side with random sizes. Snapshots carry the prints since the previous snapshot.
* Events are emitted in non-decreasing `ts`; snapshots for `ts ∈ [start, end)`; a
  `MarketResolved` at `end_ts` after the last snapshot. Fully deterministic for a given seed
  (use a private `random.Random(seed)`; no global RNG, no wall clock).

## 7. Invariants (all must be covered by tests)

1. **Reconciliation**: after every event in a backtest the engine `Portfolio.cash` equals
   `PaperExchange.balance()` (≤1e-6) and per-market positions match.
2. `cash + capital_at_risk - initial_cash == realised_pnl` after every fill/merge.
3. No negative cash, no negative quantity, `reserved_cash <= cash`.
4. At the end of a backtest every market is settled, no open orders remain, and
   `Σ PnLBreakdown.total == final_equity - initial_cash` (≤1e-6).
5. **Determinism**: same config+seed ⇒ identical `BacktestResult`; and
   sim → JSONL → replay ⇒ identical result.
6. **No look-ahead**: model/engine only see data with `ts ≤ now`; exchange fills only from
   prints after an order is live.
7. POST_ONLY orders never fill as taker; the engine never emits a POST_ONLY order that
   crosses the book at emission time.
8. **Pair-cap property** (randomised test): for any inventory/book/fair-value state,
   any bid produced while holding unpaired opposite inventory satisfies
   `bid + avg_cost[opposite] <= 1 - target_margin + EPS`.
9. Engine places no orders (other than merges/flatten sells/cancels) in WARMUP, FLATTEN, DONE;
   places none when the kill switch is latched or spot is stale.
10. Config round-trips; invalid configs are rejected.

## 8. Conventions

* Python ≥ 3.11, **standard library only** at runtime. Tests use pytest. `ruff` and `mypy`
  must pass on everything you touch (`ruff check`, `ruff format`, `mypy src`).
* Fully typed (`disallow_untyped_defs`), small pure functions, dataclasses, no global state,
  no wall-clock, no unseeded randomness, no network or file I/O outside `data/` and `cli.py`.
* Tests live in `tests/` (`test_<module>.py`), are deterministic and fast (< 5 s each file).
* Never edit files you don't own; if you need a contract change, say so in your hand-off
  note instead of editing `types.py` / `config.py` / this document.
* No secrets, keys, or credentials anywhere. Do not add dependencies.
