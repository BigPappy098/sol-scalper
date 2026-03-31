"""Position sizing strategies."""

from __future__ import annotations

import math


def fixed_risk_size(
    equity: float,
    risk_pct: float,
    stop_distance_pct: float,
    current_price: float,
    max_leverage: float = 5.0,
) -> float:
    """Calculate position size using fixed percentage risk.

    Args:
        equity: Current account equity in USD
        risk_pct: Fraction of equity to risk (e.g. 0.01 = 1%)
        stop_distance_pct: Distance to stop-loss as fraction (e.g. 0.003 = 0.3%)
        current_price: Current asset price
        max_leverage: Maximum allowed leverage

    Returns:
        Position size in asset units (e.g. SOL)
    """
    if equity <= 0 or stop_distance_pct <= 0 or current_price <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    position_value = risk_amount / stop_distance_pct

    # Enforce max leverage
    max_position_value = equity * max_leverage
    position_value = min(position_value, max_position_value)

    return position_value / current_price


def kelly_size(
    equity: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_price: float,
    fraction: float = 0.25,
    max_leverage: float = 5.0,
) -> float:
    """Calculate position size using fractional Kelly criterion.

    Args:
        equity: Current account equity
        win_rate: Historical win rate (0 to 1)
        avg_win: Average winning trade return (as fraction)
        avg_loss: Average losing trade return (as positive fraction)
        current_price: Current asset price
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative)
        max_leverage: Maximum allowed leverage

    Returns:
        Position size in asset units
    """
    if avg_loss <= 0 or equity <= 0 or current_price <= 0:
        return 0.0

    # Kelly formula: f = (p * b - q) / b
    # where p = win_rate, q = 1 - p, b = avg_win / avg_loss
    b = avg_win / avg_loss
    q = 1 - win_rate
    kelly_pct = (win_rate * b - q) / b

    # Apply fractional Kelly
    kelly_pct = max(0, kelly_pct * fraction)

    position_value = equity * kelly_pct
    max_position_value = equity * max_leverage
    position_value = min(position_value, max_position_value)

    return position_value / current_price
