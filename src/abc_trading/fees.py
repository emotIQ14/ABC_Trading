"""Fee model: taker fee curve and maker fee / rebate.

Units: prices are USDC per share in [0, 1]; sizes are shares (>= 0); fees are USDC. A positive
fee is a charge on top of ``price * size``; a negative fee is a credit (maker rebate).

Taker::

    fee = size * price * taker_fee_rate * (price * (1 - price)) ** taker_fee_exponent

so the *effective rate on notional* ``fee / (price * size)`` is symmetric under ``p <-> 1 - p``
and peaks at 0.5, while the USDC fee *per share* carries one extra factor ``price`` and peaks at
``(e + 1) / (2e + 1)`` for exponent ``e`` (0.6 for the default ``e = 2``). Both are zero at
``p = 0``; at ``p = 1`` the fee is zero iff the exponent is > 0 (with ``e = 0`` it is a flat
``taker_fee_rate * notional``).

Maker::

    fee = maker_fee_rate * price * size - maker_rebate_rate * price * size

Invariants: the fee is linear in ``size``; the taker fee is never negative; invalid inputs
(price outside [0, 1], negative or non-finite size) raise ``ValueError`` rather than being
clamped. The model is immutable and deterministic (no state, no clock, no randomness).

The fee parameters are placeholders to be verified against Polymarket's published schedule.
"""

from __future__ import annotations

import math

from abc_trading.config import FeeConfig


def _check_price(price: float) -> None:
    if not (math.isfinite(price) and 0.0 <= price <= 1.0):
        raise ValueError(f"price must be a finite number in [0, 1], got {price!r}")


def _check_size(size: float) -> None:
    if not (math.isfinite(size) and size >= 0.0):
        raise ValueError(f"size must be a finite number >= 0, got {size!r}")


def _check_config(cfg: FeeConfig) -> None:
    for name in ("maker_fee_rate", "maker_rebate_rate", "taker_fee_rate", "taker_fee_exponent"):
        value = getattr(cfg, name)
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"FeeConfig.{name} must be a finite number >= 0, got {value!r}")


class FeeModel:
    """Computes USDC fees for fills from a ``FeeConfig`` (see module docstring)."""

    __slots__ = ("_cfg",)

    def __init__(self, cfg: FeeConfig) -> None:
        _check_config(cfg)
        self._cfg = cfg

    @property
    def cfg(self) -> FeeConfig:
        return self._cfg

    def fee(self, price: float, size: float, is_maker: bool) -> float:
        """USDC fee for ``size`` shares at ``price``; negative for a net maker rebate."""
        _check_price(price)
        _check_size(size)
        cfg = self._cfg
        if is_maker:
            return cfg.maker_fee_rate * price * size - cfg.maker_rebate_rate * price * size
        curve: float = (price * (1.0 - price)) ** cfg.taker_fee_exponent
        return size * price * cfg.taker_fee_rate * curve

    def taker_fee_per_share(self, price: float) -> float:
        """Taker fee (USDC) for one share at ``price``."""
        return self.fee(price, 1.0, False)

    def all_in_buy_cost(self, price: float, is_maker: bool) -> float:
        """USDC cost of buying one share at ``price`` including its fee (rebate lowers it)."""
        return price + self.fee(price, 1.0, is_maker)
