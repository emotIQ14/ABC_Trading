"""Market phase from the clock (DESIGN 2.6).

Units: ``snap_ts`` and the ``MarketSpec`` bounds are unix seconds; the ``TimingConfig`` thresholds
are seconds. With ``since = ts - start`` and ``left = end - ts``::

    ts >= end                    -> DONE         (checked first, see below)
    since < warmup_seconds       -> WARMUP
    left  <= flatten_seconds     -> FLATTEN
    left  <= wind_down_seconds   -> WIND_DOWN
    otherwise                    -> ACCUMULATE

Every threshold is inclusive on the *later* phase: ``left == flatten_seconds`` is FLATTEN,
``left == wind_down_seconds`` is WIND_DOWN, ``since == warmup_seconds`` is ACCUMULATE and
``ts == end`` is DONE. Before the window starts (``since < 0``) the phase is WARMUP.

``ts >= end`` is tested before the warm-up rule. For any sane configuration the order does not
matter (``warmup + flatten < duration``); it only decides a degenerate market whose window is
shorter than its warm-up, where a snapshot at or after the end must still read DONE.
"""

from __future__ import annotations

import math

from abc_trading.config import TimingConfig
from abc_trading.types import MarketSpec, Phase


def phase_for(snap_ts: float, market: MarketSpec, cfg: TimingConfig) -> Phase:
    """Phase of ``market`` at ``snap_ts`` (unix seconds). Raises ValueError on a non-finite ts."""
    if not math.isfinite(snap_ts):
        raise ValueError(f"snapshot ts must be finite, got {snap_ts!r}")
    left = market.end_ts - snap_ts
    if snap_ts >= market.end_ts:
        return Phase.DONE
    if snap_ts - market.start_ts < cfg.warmup_seconds:
        return Phase.WARMUP
    if left <= cfg.flatten_seconds:
        return Phase.FLATTEN
    if left <= cfg.wind_down_seconds:
        return Phase.WIND_DOWN
    return Phase.ACCUMULATE
