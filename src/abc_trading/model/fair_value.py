"""Fair-probability model for Up/Down window markets (DESIGN sections 3 and 4.2).

Everything here is a heuristic. The weights are uncalibrated, and nothing in this module claims
the resulting probabilities beat the market. ``fit_logistic_weights``, ``brier_score`` and
``log_loss`` exist so calibration quality can be *measured*, not assumed.

Units: timestamps are unix seconds, spot/ref are underlying prices (> 0), ``sigma`` is the
standard deviation of log-returns per sqrt(second), ``tau`` is seconds. ``z`` and the features
are dimensionless (standard-normal units).

Model, for a snapshot at ``ts`` with ``spot``, ``ref`` and ``tau = max(end - ts, min_tau)``::

    z       = ln(spot / ref) / (sigma * sqrt(tau))
    mom     = ln(spot / spot[ts - L_m]) / (sigma * sqrt(L_m))
    accel   = mom_short - mom,   mom_short = ln(spot / spot[ts - L_a]) / (sigma * sqrt(L_a))
    obi     = clip(up_book.imbalance(5) - down_book.imbalance(5), -1, 1)
    z_adj   = z + w_momentum * mom + w_accel * accel + w_book_imbalance * obi
    p_model = clip(Phi(z_adj), p_floor, 1 - p_floor)
    p_up    = (1 - shrink) * p_model + shrink * up_mid        (no shrinkage if no UP mid)

``spot[t]`` is the latest observation at or before ``t`` (a step function, no interpolation).

Volatility: per asset, an EWMA of the per-second squared log-return. For each new observation
with ``dt > 0`` since the previous one: ``r2 = ln(S / S_prev)**2 / dt``,
``alpha = 1 - 0.5 ** (dt / halflife)``, ``var = (1 - alpha) * var + alpha * r2`` (the first
valid ``r2`` initialises ``var``). ``sigma = max(sqrt(var), vol_floor)``.

Warm-up rule (``valid=True`` only if ALL hold, evaluated with observations ``ts <= snap.ts``)
-------------------------------------------------------------------------------------------
1. ``snap.spot`` and ``snap.ref_price`` are present, finite and > 0.
2. At least 2 observations of ``snap.market.asset`` exist with ``ts <= snap.ts`` (so there is at
   least one return, hence a variance estimate).
3. The oldest such observation is at or before ``snap.ts - max(L_m, L_a)``: the history spans
   the LONGER of the two lookbacks, so momentum and accel become valid at the same instant and
   both lookback anchors exist. Until then the model is invalid rather than extrapolating.

Otherwise ``valid=False``, ``p_up = p_up_model =`` UP mid (0.5 if there is none), ``z = z_adj
= momentum = accel = book_imbalance = 0.0``, and ``sigma`` is the asset's sigma as of
``snap.ts`` when one exists (else ``vol_floor``). Consumers must check ``valid`` before using
the probability for directional decisions.

Known limits (documented, not hidden): lookback anchors are compared with exact float ``<=``,
so timestamps should sit on a consistent grid; and a long observation gap makes the anchor
older than ``L``, which inflates momentum. Staleness of the spot feed is the risk gate's job.

Look-ahead safety: ``estimate`` reads only observations with ``ts <= snap.ts`` and never
mutates state. To make sigma "as of ``snap.ts``" without rewinding, each retained observation
stores the EWMA variance as it stood right after that observation (``_Obs.var``). History is
pruned to the longest lookback plus ``_HISTORY_MARGIN_SECONDS``, always keeping one anchor at
or before the cutoff. A snapshot older than the retained window therefore reports invalid
instead of reading data it cannot see (conservative).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from operator import mul
from typing import NamedTuple

from abc_trading.config import ModelConfig
from abc_trading.types import MarketSnapshot, Outcome

# Observations older than (longest lookback + margin) before the newest one are dropped, so
# estimate() may still be asked about a snapshot slightly behind the newest observation.
_HISTORY_MARGIN_SECONDS = 60.0
_OBI_LEVELS = 5


def norm_cdf(x: float) -> float:
    """Standard normal CDF Phi(x) via ``math.erf``. Raises ValueError on NaN."""
    if math.isnan(x):
        raise ValueError("norm_cdf: x is NaN")
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class FairValue:
    """Model output for one snapshot. Probabilities are in [0, 1], z/features dimensionless.

    ``p_up`` is the shrunk probability the strategy should use; ``p_up_model`` is the
    pre-shrinkage model probability. When ``valid`` is False both equal the fallback
    probability (UP mid, else 0.5) and the diagnostics are zero (see module docstring).
    """

    p_up: float
    p_up_model: float
    z: float
    z_adj: float
    sigma: float  # per sqrt(second), floored at vol_floor
    tau: float  # seconds, floored at min_tau_seconds
    momentum: float
    accel: float
    book_imbalance: float  # in [-1, 1]
    valid: bool

    def p(self, token: Outcome) -> float:
        """Probability that ``token`` wins: ``p_up`` for UP, ``1 - p_up`` for DOWN."""
        return self.p_up if token is Outcome.UP else 1.0 - self.p_up


class _Obs(NamedTuple):
    ts: float
    spot: float
    var: float | None  # EWMA of per-second squared log-return after this obs; None for the first


def _is_positive_finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x) and x > 0.0


def _spot_at(obs: Sequence[_Obs], target_ts: float) -> float | None:
    """Spot of the latest observation with ``ts <= target_ts`` (``obs`` oldest first)."""
    for o in reversed(obs):
        if o.ts <= target_ts:
            return o.spot
    return None


def _ewma_var(
    var: float | None, prev_spot: float, spot: float, dt: float, halflife: float
) -> float:
    """One EWMA step of per-second variance; ``dt > 0`` seconds; ``var is None`` initialises."""
    r2 = math.log(spot / prev_spot) ** 2 / dt
    if var is None:
        return r2
    alpha = 1.0 - 0.5 ** (dt / halflife)
    return (1.0 - alpha) * var + alpha * r2


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _validate_model_config(cfg: ModelConfig) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"ModelConfig: {msg}")

    for name in ("vol_halflife_seconds", "vol_floor", "momentum_lookback_seconds"):
        need(_is_positive_finite(getattr(cfg, name)), f"{name} must be finite and > 0")
    need(_is_positive_finite(cfg.accel_lookback_seconds), "accel_lookback_seconds must be > 0")
    need(_is_positive_finite(cfg.min_tau_seconds), "min_tau_seconds must be finite and > 0")
    need(0.0 <= cfg.shrink_to_market <= 1.0, "shrink_to_market must be in [0, 1]")
    need(0.0 < cfg.p_floor < 0.5, "p_floor must be in (0, 0.5)")
    for name in ("w_momentum", "w_accel", "w_book_imbalance"):
        need(math.isfinite(getattr(cfg, name)), f"{name} must be finite")


class FairValueModel:
    """Stateful per-asset volatility/price history plus a pure ``estimate`` per snapshot."""

    def __init__(self, cfg: ModelConfig) -> None:
        _validate_model_config(cfg)
        self._cfg = cfg
        self._horizon = max(cfg.momentum_lookback_seconds, cfg.accel_lookback_seconds)
        self._retain = self._horizon + _HISTORY_MARGIN_SECONDS
        self._history: dict[str, deque[_Obs]] = {}

    # ------------------------------------------------------------------ state updates

    def observe(self, asset: str, ts: float, spot: float) -> None:
        """Record ``spot`` for ``asset`` at ``ts`` (seconds) and update the EWMA variance.

        Idempotent per (asset, ts): a repeat of the latest ``ts`` is ignored (the first value
        wins, even if ``spot`` differs). Raises ValueError if ``ts`` is not finite or goes
        backwards, or ``spot`` is not finite and > 0.
        """
        if not math.isfinite(ts):
            raise ValueError(f"observe({asset}): ts must be finite, got {ts!r}")
        if not _is_positive_finite(spot):
            raise ValueError(f"observe({asset}): spot must be finite and > 0, got {spot!r}")
        hist = self._history.get(asset)
        if hist is None:
            self._history[asset] = deque([_Obs(ts, spot, None)])
            return
        last = hist[-1]
        if ts < last.ts:
            raise ValueError(f"observe({asset}): ts went backwards ({ts!r} < latest {last.ts!r})")
        if ts == last.ts:
            return
        var = _ewma_var(last.var, last.spot, spot, ts - last.ts, self._cfg.vol_halflife_seconds)
        hist.append(_Obs(ts, spot, var))
        self._prune(hist)

    def _prune(self, hist: deque[_Obs]) -> None:
        cutoff = hist[-1].ts - self._retain
        # Keep exactly one observation at or before the cutoff: it anchors the longest lookback.
        while len(hist) >= 2 and hist[1].ts <= cutoff:
            hist.popleft()

    # ------------------------------------------------------------------ queries

    def vol(self, asset: str) -> float | None:
        """Current sigma (per sqrt(second), floored at ``vol_floor``); None if < 2 observations."""
        hist = self._history.get(asset)
        if hist is None or hist[-1].var is None:
            return None
        return self._sigma(hist[-1].var)

    def _sigma(self, var: float) -> float:
        return max(math.sqrt(var), self._cfg.vol_floor)

    def estimate(self, snap: MarketSnapshot) -> FairValue:
        """Fair probability for ``snap``. Pure: does not mutate state; uses only ts <= snap.ts."""
        cfg = self._cfg
        tau = max(snap.seconds_to_end, cfg.min_tau_seconds)
        up_mid = snap.up_book.mid
        fallback = 0.5 if up_mid is None else up_mid
        spot, ref = snap.spot, snap.ref_price

        # Only observations at or before snap.ts are visible (no look-ahead). Sigma is the EWMA
        # as it stood after the latest visible observation; it needs >= 2 of them (>= 1 return).
        visible = [o for o in self._history.get(snap.market.asset, ()) if o.ts <= snap.ts]
        var = visible[-1].var if len(visible) >= 2 else None
        sigma = cfg.vol_floor if var is None else self._sigma(var)

        if (
            spot is None
            or ref is None
            or not (_is_positive_finite(spot) and _is_positive_finite(ref))
        ):
            return self._invalid(fallback, tau, sigma)
        if var is None or visible[0].ts > snap.ts - self._horizon:
            return self._invalid(fallback, tau, sigma)

        anchor_m = _spot_at(visible, snap.ts - cfg.momentum_lookback_seconds)
        anchor_a = _spot_at(visible, snap.ts - cfg.accel_lookback_seconds)
        if anchor_m is None or anchor_a is None:  # unreachable given the horizon check above
            return self._invalid(fallback, tau, sigma)

        z = math.log(spot / ref) / (sigma * math.sqrt(tau))
        mom = math.log(spot / anchor_m) / (sigma * math.sqrt(cfg.momentum_lookback_seconds))
        mom_short = math.log(spot / anchor_a) / (sigma * math.sqrt(cfg.accel_lookback_seconds))
        accel = mom_short - mom
        obi = _clip(
            snap.up_book.imbalance(_OBI_LEVELS) - snap.down_book.imbalance(_OBI_LEVELS), -1.0, 1.0
        )
        z_adj = z + cfg.w_momentum * mom + cfg.w_accel * accel + cfg.w_book_imbalance * obi
        p_model = _clip(norm_cdf(z_adj), cfg.p_floor, 1.0 - cfg.p_floor)
        if up_mid is None:
            p_up = p_model
        else:
            p_up = (1.0 - cfg.shrink_to_market) * p_model + cfg.shrink_to_market * up_mid
        return FairValue(
            p_up=p_up,
            p_up_model=p_model,
            z=z,
            z_adj=z_adj,
            sigma=sigma,
            tau=tau,
            momentum=mom,
            accel=accel,
            book_imbalance=obi,
            valid=True,
        )

    @staticmethod
    def _invalid(fallback: float, tau: float, sigma: float) -> FairValue:
        return FairValue(
            p_up=fallback,
            p_up_model=fallback,
            z=0.0,
            z_adj=0.0,
            sigma=sigma,
            tau=tau,
            momentum=0.0,
            accel=0.0,
            book_imbalance=0.0,
            valid=False,
        )


# --------------------------------------------------------------------------- calibration tools


def _sigmoid(t: float) -> float:
    """Numerically stable logistic function."""
    if t >= 0.0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


def _check_labels_and_preds(preds: Sequence[float], outcomes: Sequence[int], who: str) -> None:
    if len(preds) != len(outcomes):
        raise ValueError(
            f"{who}: preds and outcomes differ in length ({len(preds)} vs {len(outcomes)})"
        )
    if not preds:
        raise ValueError(f"{who}: need at least one sample")
    for p in preds:
        if not 0.0 <= p <= 1.0:  # also rejects NaN
            raise ValueError(f"{who}: prediction {p!r} is not in [0, 1]")
    for y in outcomes:
        if y not in (0, 1):
            raise ValueError(f"{who}: outcome {y!r} is not 0 or 1")


def brier_score(preds: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error ``mean((p - y)**2)`` between probabilities and 0/1 outcomes.

    Lower is better; always predicting 0.5 scores 0.25. Raises ValueError on empty input,
    unequal lengths, predictions outside [0, 1] or outcomes outside {0, 1}.
    """
    _check_labels_and_preds(preds, outcomes, "brier_score")
    return sum((p - y) ** 2 for p, y in zip(preds, outcomes, strict=True)) / len(preds)


def log_loss(preds: Sequence[float], outcomes: Sequence[int], eps: float = 1e-9) -> float:
    """Mean negative log-likelihood ``-mean(y ln p + (1 - y) ln(1 - p))`` in nats.

    Predictions are clipped to ``[eps, 1 - eps]`` first so a confident miss costs a large but
    finite ``-ln(eps)``. Same input validation as ``brier_score``; ``eps`` must be in (0, 0.5).
    """
    if not 0.0 < eps < 0.5:
        raise ValueError(f"log_loss: eps must be in (0, 0.5), got {eps!r}")
    _check_labels_and_preds(preds, outcomes, "log_loss")
    total = 0.0
    for p, y in zip(preds, outcomes, strict=True):
        # Clip the probability assigned to what actually happened. Equivalent to clipping p to
        # [eps, 1 - eps] first, but 1 - p is formed before clipping so p = 1, y = 0 costs
        # exactly -ln(eps) instead of picking up the rounding error of 1 - (1 - eps).
        p_seen = p if y == 1 else 1.0 - p
        total -= math.log(_clip(p_seen, eps, 1.0 - eps))
    return total / len(preds)


def fit_logistic_weights(
    samples: Sequence[tuple[Sequence[float], int]],
    *,
    l2: float = 1e-3,
    iters: int = 500,
    lr: float = 0.1,
) -> list[float]:
    """Fit ``p = sigmoid(w . x)`` (no intercept) by full-batch gradient descent from w = 0.

    Minimises ``mean log-loss + (l2 / 2) * ||w||**2``; each iteration moves ``w`` by
    ``-lr * ((1/n) * sum((p_i - y_i) * x_i) + l2 * w)``. ``samples`` are
    ``(features, label)`` with label in {0, 1} and a constant feature count >= 1. Returns the
    weights, one per feature. Raises ValueError on empty or ragged samples, bad labels,
    non-finite features, ``l2 < 0``, ``iters < 0`` or ``lr <= 0``.
    """
    if not samples:
        raise ValueError("fit_logistic_weights: need at least one sample")
    if l2 < 0.0 or iters < 0 or lr <= 0.0:
        raise ValueError("fit_logistic_weights: need l2 >= 0, iters >= 0, lr > 0")
    dim = len(samples[0][0])
    if dim < 1:
        raise ValueError("fit_logistic_weights: samples need at least one feature")
    for x, y in samples:
        if len(x) != dim:
            raise ValueError("fit_logistic_weights: samples have differing feature counts")
        if y not in (0, 1):
            raise ValueError(f"fit_logistic_weights: label {y!r} is not 0 or 1")
        if not all(math.isfinite(v) for v in x):
            raise ValueError("fit_logistic_weights: non-finite feature value")

    n = len(samples)
    rows = [tuple(x) for x, _ in samples]
    labels = [y for _, y in samples]
    cols = [[row[j] for row in rows] for j in range(dim)]
    w = [0.0] * dim
    for _ in range(iters):
        errs = [_sigmoid(sum(map(mul, w, row))) - y for row, y in zip(rows, labels, strict=True)]
        grad = [sum(map(mul, errs, col)) / n for col in cols]
        w = [wj - lr * (gj + l2 * wj) for wj, gj in zip(w, grad, strict=True)]
    return w
