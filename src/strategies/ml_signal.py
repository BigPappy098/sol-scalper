"""Strategy 4: ML Ensemble Signal.

Uses LightGBM + CNN predictions to generate trading signals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.data.schemas import Candle, Signal, SignalDirection
from src.ml.models import ModelManager
from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class MLSignalStrategy(BaseStrategy):
    """ML-based signal generator.

    Entry:
    - Model predicts P(up > 0.1%) > 0.65 for long
    - Model predicts P(up > 0.1%) < 0.35 for short (i.e. P(down) > 0.65)
    - Cooldown after stop-loss to prevent revenge trading

    Exit:
    - Target: 0.1%
    - Stop: 0.08%
    - Time stop: 60 seconds
    - Exit if model confidence drops below 0.5
    """

    def __init__(self, config: dict, model_manager: ModelManager):
        super().__init__("ml_signal", config)
        self._model_manager = model_manager
        self._confidence_threshold = config.get("confidence_threshold", 0.65)
        self._confidence_exit = config.get("confidence_exit_threshold", 0.5)
        self._target_pct = config.get("target_pct", 0.001)
        self._stop_pct = config.get("stop_pct", 0.0008)
        self._time_stop = config.get("time_stop_seconds", 60)
        self._cooldown = config.get("cooldown_after_stop_seconds", 120)
        self._target_timeframe = "5s"  # Predict on 5-second candles

        self._last_stop_time: datetime | None = None

    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        if candle.timeframe != self._target_timeframe:
            return None

        if not self._model_manager.is_ready:
            return None

        # Cooldown check
        if self._last_stop_time:
            elapsed = (candle.timestamp - self._last_stop_time).total_seconds()
            if elapsed < self._cooldown:
                return None

        # Get ML prediction
        prob_up = self._model_manager.predict_lgbm(features)
        if prob_up is None:
            return None

        # --- LONG signal ---
        if prob_up > self._confidence_threshold:
            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.LONG,
                confidence=prob_up,
                target_pct=self._target_pct,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._time_stop,
                metadata={
                    "ml_prob_up": prob_up,
                    "price": features.get("price", candle.close),
                },
            )

        # --- SHORT signal ---
        if prob_up < (1 - self._confidence_threshold):
            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SHORT,
                confidence=1 - prob_up,
                target_pct=self._target_pct,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._time_stop,
                metadata={
                    "ml_prob_up": prob_up,
                    "price": features.get("price", candle.close),
                },
            )

        return None

    def on_stop_loss(self) -> None:
        """Called when a trade from this strategy hits stop-loss."""
        self._last_stop_time = datetime.now(timezone.utc)
