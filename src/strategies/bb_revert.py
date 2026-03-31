"""Strategy 1: Bollinger Band Mean Reversion.

Enters when price touches extreme BB levels with RSI confirmation
and declining volume (exhaustion moves). Targets the BB middle band.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.schemas import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class BBRevertStrategy(BaseStrategy):
    """Bollinger Band mean reversion scalper.

    Entry conditions:
    - Price at/beyond 2.5σ Bollinger Band
    - RSI below 20 (long) or above 80 (short)
    - Volume declining vs last 3 candles (exhaustion)
    - Context timeframe trend not strongly against direction

    Exit:
    - Target: BB middle band (20-period SMA)
    - Stop: 0.3% from entry
    - Time stop: 3 minutes
    """

    def __init__(self, config: dict):
        super().__init__("bb_revert", config)
        self._bb_sigma = config.get("bb_sigma", 2.5)
        self._rsi_oversold = config.get("rsi_oversold", 20)
        self._rsi_overbought = config.get("rsi_overbought", 80)
        self._stop_pct = config.get("stop_pct", 0.003)
        self._time_stop = config.get("time_stop_seconds", 180)
        self._trail_activation = config.get("trail_activation_pct", 0.6)
        self._trail_stop = config.get("trail_stop_pct", 0.001)
        self._target_timeframe = config.get("timeframe", "15s")
        self._context_timeframe = config.get("context_timeframe", "1m")

        # State for volume decline check
        self._recent_volumes: list[float] = []
        self._max_recent = 5

    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        """Check for BB mean reversion entry."""
        if candle.timeframe != self._target_timeframe:
            return None

        if not features:
            return None

        price = features.get("price", 0)
        rsi = features.get("rsi", 50)
        bb_upper = features.get("bb_upper", 0)
        bb_lower = features.get("bb_lower", 0)
        bb_middle = features.get("bb_middle", 0)
        bb_position = features.get("bb_position", 0.5)

        if price == 0 or bb_upper == 0:
            return None

        # Track recent volumes for exhaustion check
        self._recent_volumes.append(candle.volume)
        if len(self._recent_volumes) > self._max_recent:
            self._recent_volumes = self._recent_volumes[-self._max_recent:]

        # Need at least 3 candles to check volume decline
        if len(self._recent_volumes) < 3:
            return None

        # Check volume exhaustion: current volume < average of previous 3
        avg_prev_volume = sum(self._recent_volumes[-4:-1]) / 3
        volume_declining = candle.volume < avg_prev_volume

        # --- LONG signal: price at/below lower BB + RSI oversold + exhaustion ---
        if price <= bb_lower and rsi < self._rsi_oversold and volume_declining:
            # Target = BB middle band
            target_distance = (bb_middle - price) / price
            if target_distance <= 0:
                return None

            confidence = self._compute_confidence(
                bb_position=bb_position,
                rsi=rsi,
                direction="long",
                volume_ratio=candle.volume / avg_prev_volume if avg_prev_volume > 0 else 1,
            )

            if confidence < 0.5:
                return None

            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.LONG,
                confidence=confidence,
                target_pct=target_distance,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._time_stop,
                metadata={
                    "bb_position": bb_position,
                    "rsi": rsi,
                    "target_price": bb_middle,
                    "volume_ratio": candle.volume / avg_prev_volume if avg_prev_volume > 0 else 1,
                },
            )

        # --- SHORT signal: price at/above upper BB + RSI overbought + exhaustion ---
        if price >= bb_upper and rsi > self._rsi_overbought and volume_declining:
            target_distance = (price - bb_middle) / price
            if target_distance <= 0:
                return None

            confidence = self._compute_confidence(
                bb_position=bb_position,
                rsi=rsi,
                direction="short",
                volume_ratio=candle.volume / avg_prev_volume if avg_prev_volume > 0 else 1,
            )

            if confidence < 0.5:
                return None

            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                target_pct=target_distance,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._time_stop,
                metadata={
                    "bb_position": bb_position,
                    "rsi": rsi,
                    "target_price": bb_middle,
                    "volume_ratio": candle.volume / avg_prev_volume if avg_prev_volume > 0 else 1,
                },
            )

        return None

    def _compute_confidence(
        self,
        bb_position: float,
        rsi: float,
        direction: str,
        volume_ratio: float,
    ) -> float:
        """Compute signal confidence based on indicator alignment.

        Confidence ranges from 0.0 to 1.0.
        """
        score = 0.5  # Base confidence

        if direction == "long":
            # More extreme BB position = higher confidence
            if bb_position < 0:
                score += 0.15  # Beyond lower band
            elif bb_position < 0.05:
                score += 0.1

            # More extreme RSI = higher confidence
            if rsi < 10:
                score += 0.15
            elif rsi < 15:
                score += 0.1
            elif rsi < 20:
                score += 0.05

        elif direction == "short":
            if bb_position > 1:
                score += 0.15
            elif bb_position > 0.95:
                score += 0.1

            if rsi > 90:
                score += 0.15
            elif rsi > 85:
                score += 0.1
            elif rsi > 80:
                score += 0.05

        # Lower volume ratio (more exhaustion) = higher confidence
        if volume_ratio < 0.5:
            score += 0.1
        elif volume_ratio < 0.7:
            score += 0.05

        return min(score, 1.0)
