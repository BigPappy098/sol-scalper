"""Strategy 3: Orderbook Imbalance Fade.

Enters when orderbook shows extreme imbalance, betting that price
moves toward the heavy side.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from src.data.schemas import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class OBFadeStrategy(BaseStrategy):
    """Orderbook imbalance fade scalper.

    Entry conditions:
    - Depth-10 imbalance exceeds threshold (0.6)
    - Imbalance sustained for at least 3 seconds
    - Spread within normal range
    - No recent adverse price move

    Exit:
    - Target: 0.08%
    - Stop: 0.05%
    - Time stop: 15 seconds
    """

    def __init__(self, config: dict):
        super().__init__("ob_fade", config)
        self._imbalance_threshold = config.get("imbalance_threshold", 0.6)
        self._sustain_seconds = config.get("imbalance_sustain_seconds", 3)
        self._target_pct = config.get("target_pct", 0.0008)
        self._stop_pct = config.get("stop_pct", 0.0005)
        self._time_stop = config.get("time_stop_seconds", 15)
        self._target_timeframe = config.get("timeframe", "1s")

        # Track imbalance history for sustain check
        self._imbalance_history: deque[tuple[datetime, float]] = deque(maxlen=10)

    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        if candle.timeframe != self._target_timeframe:
            return None

        if not orderbook_features:
            return None

        imbalance = orderbook_features.get("imbalance_10", 0)
        spread_bps = orderbook_features.get("spread_bps", 0)

        # Record imbalance
        self._imbalance_history.append((candle.timestamp, imbalance))

        # Check if imbalance has been sustained
        if not self._is_sustained(imbalance > 0):
            return None

        # Spread check: don't enter when spread is abnormally wide
        # (typical SOL spread is 1-3 bps)
        if spread_bps > 10:
            return None

        # Check that last candle didn't move strongly against us
        if features:
            last_return = features.get("return_1", 0)
        else:
            last_return = 0

        # --- LONG: strong bid imbalance ---
        if imbalance > self._imbalance_threshold:
            # Don't enter if price already moved up significantly
            if last_return > 0.001:
                return None

            confidence = self._compute_confidence(imbalance, spread_bps)
            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.LONG,
                confidence=confidence,
                target_pct=self._target_pct,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._time_stop,
                metadata={
                    "imbalance": imbalance,
                    "spread_bps": spread_bps,
                },
            )

        # --- SHORT: strong ask imbalance ---
        if imbalance < -self._imbalance_threshold:
            if last_return < -0.001:
                return None

            confidence = self._compute_confidence(abs(imbalance), spread_bps)
            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                target_pct=self._target_pct,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._time_stop,
                metadata={
                    "imbalance": imbalance,
                    "spread_bps": spread_bps,
                },
            )

        return None

    def _is_sustained(self, is_positive: bool) -> bool:
        """Check if imbalance direction has been sustained for N seconds."""
        if len(self._imbalance_history) < self._sustain_seconds:
            return False

        recent = list(self._imbalance_history)[-self._sustain_seconds:]
        threshold = self._imbalance_threshold

        if is_positive:
            return all(imb > threshold * 0.7 for _, imb in recent)
        else:
            return all(imb < -threshold * 0.7 for _, imb in recent)

    def _compute_confidence(self, abs_imbalance: float, spread_bps: float) -> float:
        score = 0.5

        # Stronger imbalance = higher confidence
        if abs_imbalance > 0.8:
            score += 0.2
        elif abs_imbalance > 0.7:
            score += 0.1

        # Tighter spread = higher confidence
        if spread_bps < 2:
            score += 0.1
        elif spread_bps < 5:
            score += 0.05

        return min(score, 1.0)
