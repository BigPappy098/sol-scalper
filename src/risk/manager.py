"""Risk management — position sizing and trade approval."""

from __future__ import annotations

from datetime import datetime, timezone

from src.config.settings import get_settings
from src.data.schemas import Signal, Side
from src.utils.logging import get_logger

log = get_logger(__name__)


class RiskManager:
    """Manages position sizing and trade approval.

    No trade limits or halts — risk is managed purely through
    position sizing that scales with equity.
    """

    def __init__(self, initial_equity: float | None = None):
        self._settings = get_settings()
        self._equity = initial_equity or 0.0
        self._risk_per_trade = self._settings.risk_per_trade
        self._max_leverage = self._settings.max_leverage
        self._open_positions: dict[str, dict] = {}
        self._daily_pnl = 0.0
        self._peak_equity = self._equity

    def update_equity(self, equity: float) -> None:
        """Update current equity from exchange."""
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def add_pnl(self, pnl: float) -> None:
        """Record PnL from a closed trade."""
        self._daily_pnl += pnl
        self._equity += pnl

    def reset_daily(self) -> None:
        """Reset daily tracking (call at UTC midnight)."""
        self._daily_pnl = 0.0

    def approve_trade(self, signal: Signal, current_price: float) -> tuple[bool, float, str]:
        """Evaluate whether to take a trade and compute position size.

        Returns:
            (approved, quantity, reason)
        """
        if self._equity <= 0:
            return False, 0.0, "zero_equity"

        if current_price <= 0:
            return False, 0.0, "invalid_price"

        # Compute position size based on risk per trade
        risk_amount = self._equity * self._risk_per_trade
        stop_distance = signal.stop_pct

        if stop_distance <= 0:
            return False, 0.0, "invalid_stop"

        # Position value = risk_amount / stop_distance_pct
        position_value = risk_amount / stop_distance

        # Enforce max leverage
        max_position_value = self._equity * self._max_leverage
        position_value = min(position_value, max_position_value)

        # Convert to quantity (SOL units)
        quantity = position_value / current_price

        # Round to reasonable precision for SOL (Bybit typically allows 0.1 SOL minimum)
        quantity = round(quantity, 1)

        if quantity < 0.1:
            return False, 0.0, "quantity_too_small"

        log.info(
            "trade_approved",
            equity=self._equity,
            risk_amount=risk_amount,
            position_value=position_value,
            quantity=quantity,
            leverage=round(position_value / self._equity, 2),
            strategy=signal.strategy_name,
        )

        return True, quantity, "approved"

    def register_position(self, position_id: str, details: dict) -> None:
        """Track an open position."""
        self._open_positions[position_id] = details

    def close_position(self, position_id: str) -> None:
        """Remove a closed position from tracking."""
        self._open_positions.pop(position_id, None)

    def get_exposure(self) -> dict:
        """Get current exposure info."""
        total_exposure = sum(
            abs(p.get("position_value", 0)) for p in self._open_positions.values()
        )
        return {
            "equity": self._equity,
            "open_positions": len(self._open_positions),
            "total_exposure": total_exposure,
            "leverage": total_exposure / self._equity if self._equity > 0 else 0,
            "daily_pnl": self._daily_pnl,
            "drawdown_from_peak": (
                (self._peak_equity - self._equity) / self._peak_equity
                if self._peak_equity > 0
                else 0
            ),
        }

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl
