"""Pure statistics helpers for backtest reports.

Units: money is USDC, probabilities are in [0, 1]; every function is a pure, deterministic
function of its arguments (no state, clock or randomness). Sums use ``math.fsum`` so results do
not depend on accumulation order.

Non-finite inputs (NaN, +-inf) always raise ``ValueError``: a corrupted number must never flow
silently into a report. "Undefined" ratios are reported explicitly: ``ratio`` returns ``None`` and
``sharpe_like`` returns 0.0 (documented below), never NaN or inf, so results stay JSON-safe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from abc_trading.model.fair_value import brier_score
from abc_trading.types import Outcome

ZERO_DISPERSION = 1e-12  # a sample std at or below this (USDC) is treated as "no dispersion"


def _require_finite(values: Sequence[float], what: str) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{what}: values must be finite, got {v!r}")


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean. Raises ValueError on empty or non-finite input."""
    if not values:
        raise ValueError("mean of an empty sequence")
    _require_finite(values, "mean")
    return math.fsum(values) / len(values)


def median(values: Sequence[float]) -> float:
    """Median (mean of the two middle values for an even count). ValueError on empty input."""
    if not values:
        raise ValueError("median of an empty sequence")
    _require_finite(values, "median")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def std(values: Sequence[float], *, ddof: int = 1) -> float:
    """Standard deviation with ``ddof`` degrees of freedom removed (default: sample std).

    Two-pass algorithm. Raises ValueError unless ``len(values) > ddof`` (``ddof >= 0``).
    """
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof!r}")
    n = len(values)
    if n <= ddof:
        raise ValueError(f"std needs more than ddof={ddof} values, got {n}")
    mu = mean(values)
    return math.sqrt(math.fsum((v - mu) ** 2 for v in values) / (n - ddof))


def sharpe_like(values: Sequence[float]) -> float:
    """``mean / std * sqrt(n)`` over per-period PnL values (sample std, ``ddof=1``).

    This is a t-statistic of the mean, not an annualised Sharpe ratio. It is 0.0 when it is
    undefined: fewer than two values, or a std at or below ``ZERO_DISPERSION``.
    """
    n = len(values)
    if n < 2:
        _require_finite(values, "sharpe_like")
        return 0.0
    s = std(values)
    if s <= ZERO_DISPERSION:
        return 0.0
    return mean(values) / s * math.sqrt(n)


def ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is zero."""
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        raise ValueError(f"ratio: operands must be finite, got {numerator!r}, {denominator!r}")
    return None if denominator == 0.0 else numerator / denominator


class Drawdown(NamedTuple):
    """Deepest peak-to-trough fall of an equity series.

    ``depth`` is in USDC (>= 0), ``fraction`` is ``depth / peak`` (0.0 if the peak is not
    positive), ``peak_index`` / ``trough_index`` locate the episode in the series.
    """

    depth: float
    fraction: float
    peak_index: int
    trough_index: int


def max_drawdown(equity: Sequence[float]) -> Drawdown:
    """Maximum drawdown by depth (USDC) of an equity series, with the first deepest episode
    winning ties. A series that never falls below a previous high has depth 0.

    The result depends on the sampling of the series (a dip between two samples is invisible).
    Raises ValueError on an empty or non-finite series.
    """
    if not equity:
        raise ValueError("max_drawdown of an empty series")
    _require_finite(equity, "max_drawdown")
    peak, peak_i = equity[0], 0
    best = Drawdown(0.0, 0.0, 0, 0)
    for i, value in enumerate(equity):
        if value > peak:
            peak, peak_i = value, i
        depth = peak - value
        if depth > best.depth:
            best = Drawdown(depth, depth / peak if peak > 0.0 else 0.0, peak_i, i)
    return best


def brier_up(p_up: Sequence[float], winners: Sequence[Outcome]) -> float:
    """Brier score of P(UP) forecasts against realised winners (UP = 1, DOWN = 0).

    Delegates to ``model.brier_score``: lower is better, always forecasting 0.5 scores 0.25.
    Raises ValueError on empty input, unequal lengths or a probability outside [0, 1].
    """
    if len(p_up) != len(winners):
        raise ValueError(f"brier_up: {len(p_up)} forecasts but {len(winners)} winners")
    return brier_score(p_up, [1 if w is Outcome.UP else 0 for w in winners])


def brier_skill(score: float, reference: float) -> float | None:
    """Skill ``1 - score / reference`` against a reference Brier score (positive = better than
    the reference); ``None`` if the reference is not positive."""
    if not (math.isfinite(score) and math.isfinite(reference)):
        raise ValueError(f"brier_skill: scores must be finite, got {score!r}, {reference!r}")
    return None if reference <= 0.0 else 1.0 - score / reference
