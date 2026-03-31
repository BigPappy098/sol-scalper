"""Base strategy interface that all strategies must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.data.schemas import Candle, Signal


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        self.weight = config.get("initial_weight", 0.2)
        self._last_signal_time: datetime | None = None

    @abstractmethod
    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        """Process a new candle and optionally return a trading signal.

        Args:
            candle: The latest completed candle
            features: Technical features from the FeatureStore
            orderbook_features: Orderbook-derived features (optional)

        Returns:
            A Signal if the strategy wants to trade, None otherwise
        """
        ...

    def mute(self) -> None:
        """Temporarily disable this strategy (weight = 0)."""
        self.weight = 0.0
        self.enabled = False

    def unmute(self, weight: float | None = None) -> None:
        """Re-enable this strategy."""
        self.weight = weight or self.config.get("initial_weight", 0.2)
        self.enabled = True

    def set_weight(self, weight: float) -> None:
        """Update strategy weight in the ensemble."""
        self.weight = max(0.0, min(1.0, weight))

    @property
    def is_active(self) -> bool:
        return self.enabled and self.weight > 0
