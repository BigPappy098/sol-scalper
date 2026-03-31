"""Strategy 2: Volume Breakout Momentum.

Enters in the direction of a sharp price move confirmed by a volume spike.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from src.data.schemas import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class VolBreakStrategy(BaseStrategy):
    """Volume breakout momentum scalper.

    Entry conditions:
    - Price moves > 0.15% in 5 seconds
    - Volume in that window > 2x rolling median
    - Orderbook imbalance aligns with direction
    - Confirmation: next candle closes in same direction

    Exit:
    - Target: 0.2%
    - Stop: 0.1%
    - Time stop: 30 seconds
    - Partial exit at target, trail remainder
    """

    def __init__(self, config: dict):
        super().__init__("vol_break", config)
        self._move_threshold = config.get("move_threshold_pct", 0.0015)
        self._volume_spike_mult = config.get("volume_spike_mult", 2.0)
        self._ob_imbalance_threshold = config.get("ob_imbalance_threshold", 0.3)
        self._target_pct = config.get("target_pct", 0.002)
        self._stop_pct = config.get("stop_pct", 0.001)
        self._time_stop = config.get("time_stop_seconds", 30)
        self._target_timeframe = config.get("timeframe", "5s")

        # Rolling volume median
        self._volume_history: deque[float] = deque(maxlen=30)
        self._prev_candle: Candle | None = None

    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        if candle.timeframe != self._target_timeframe:
            return None

        # Track volume history
        self._volume_history.append(candle.volume)

        if self._prev_candle is None:
            self._prev_candle = candle
            return None

        if len(self._volume_history) < 10:
            self._prev_candle = candle
            return None

        # Check for sharp price move
        prev_close = self._prev_candle.close
        if prev_close == 0:
            self._prev_candle = candle
            return None

        price_move = (candle.close - prev_close) / prev_close

        # Check volume spike
        sorted_vols = sorted(self._volume_history)
        median_vol = sorted_vols[len(sorted_vols) // 2]
        volume_spike = candle.volume > median_vol * self._volume_spike_mult if median_vol > 0 else False

        # Check confirmation: candle closed in direction of move
        candle_bullish = candle.close > candle.open
        candle_bearish = candle.close < candle.open

        signal = None

        # --- LONG: sharp upward move + volume spike ---
        if (
            price_move > self._move_threshold
            and volume_spike
            and candle_bullish
        ):
            # Check orderbook alignment if available
            ob_aligned = True
            if orderbook_features:
                imbalance = orderbook_features.get("imbalance_10", 0)
                ob_aligned = imbalance > self._ob_imbalance_threshold

            if ob_aligned:
                confidence = self._compute_confidence(price_move, candle.volume, median_vol, "long")
                signal = Signal(
                    timestamp=candle.timestamp,
                    strategy_name=self.name,
                    direction=SignalDirection.LONG,
                    confidence=confidence,
                    target_pct=self._target_pct,
                    stop_pct=self._stop_pct,
                    time_stop_seconds=self._time_stop,
                    metadata={
                        "price_move": price_move,
                        "volume_ratio": candle.volume / median_vol if median_vol > 0 else 0,
                    },
                )

        # --- SHORT: sharp downward move + volume spike ---
        elif (
            price_move < -self._move_threshold
            and volume_spike
            and candle_bearish
        ):
            ob_aligned = True
            if orderbook_features:
                imbalance = orderbook_features.get("imbalance_10", 0)
                ob_aligned = imbalance < -self._ob_imbalance_threshold

            if ob_aligned:
                confidence = self._compute_confidence(abs(price_move), candle.volume, median_vol, "short")
                signal = Signal(
                    timestamp=candle.timestamp,
                    strategy_name=self.name,
                    direction=SignalDirection.SHORT,
                    confidence=confidence,
                    target_pct=self._target_pct,
                    stop_pct=self._stop_pct,
                    time_stop_seconds=self._time_stop,
                    metadata={
                        "price_move": price_move,
                        "volume_ratio": candle.volume / median_vol if median_vol > 0 else 0,
                    },
                )

        self._prev_candle = candle
        return signal

    def _compute_confidence(
        self, move_size: float, volume: float, median_vol: float, direction: str
    ) -> float:
        score = 0.5

        # Larger move = higher confidence
        if move_size > self._move_threshold * 2:
            score += 0.15
        elif move_size > self._move_threshold * 1.5:
            score += 0.1

        # Larger volume spike = higher confidence
        vol_ratio = volume / median_vol if median_vol > 0 else 1
        if vol_ratio > 4:
            score += 0.15
        elif vol_ratio > 3:
            score += 0.1
        elif vol_ratio > 2:
            score += 0.05

        return min(score, 1.0)
