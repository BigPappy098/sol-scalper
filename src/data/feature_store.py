"""Streaming feature computation for strategies and ML models."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime

import numpy as np

from src.data.schemas import Candle
from src.utils.logging import get_logger

log = get_logger(__name__)


class FeatureStore:
    """Computes and caches technical features from candle data.

    Features are computed incrementally as new candles arrive.
    Maintains rolling windows for all indicators.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self._atr_period = config.get("atr_period", 14)
        self._rsi_period = config.get("rsi_period", 14)
        self._bb_period = config.get("bb_period", 20)
        self._bb_std = config.get("bb_std", 2.5)
        self._ema_fast = config.get("ema_fast", 9)
        self._ema_slow = config.get("ema_slow", 21)
        self._ema_trend = config.get("ema_trend", 55)
        self._vol_windows = config.get("volatility_windows", [10, 30, 60, 200])
        self._return_periods = config.get("return_periods", [1, 3, 5, 10, 20])
        self._zscore_window = config.get("zscore_window", 200)

        # Per-timeframe data stores
        self._candles: dict[str, deque[Candle]] = {}
        self._features_cache: dict[str, dict] = {}
        self._max_candles = 500

    def on_candle(self, candle: Candle) -> dict:
        """Process a new candle and compute all features for its timeframe.

        Returns dict of features for this timeframe.
        """
        tf = candle.timeframe
        if tf not in self._candles:
            self._candles[tf] = deque(maxlen=self._max_candles)

        self._candles[tf].append(candle)
        features = self._compute_features(tf)
        self._features_cache[tf] = features
        return features

    def get_features(self, timeframe: str) -> dict:
        """Get the latest computed features for a timeframe."""
        return self._features_cache.get(timeframe, {})

    def get_candles(self, timeframe: str, count: int = 200) -> list[Candle]:
        """Get recent candles for a timeframe."""
        candles = self._candles.get(timeframe, deque())
        return list(candles)[-count:]

    def has_enough_data(self, timeframe: str, min_candles: int = 50) -> bool:
        """Check if we have enough candles to compute features."""
        return len(self._candles.get(timeframe, deque())) >= min_candles

    def _compute_features(self, timeframe: str) -> dict:
        """Compute all features for a timeframe."""
        candles = list(self._candles[timeframe])
        if len(candles) < 2:
            return {}

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        features: dict = {
            "price": closes[-1],
            "timestamp": candles[-1].timestamp.isoformat(),
        }

        # --- Returns ---
        log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))
        for period in self._return_periods:
            if len(closes) > period:
                features[f"return_{period}"] = float(
                    (closes[-1] / closes[-1 - period]) - 1
                )
            else:
                features[f"return_{period}"] = 0.0

        # --- Volatility ---
        if len(log_returns) > 1:
            for window in self._vol_windows:
                if len(log_returns) >= window:
                    features[f"volatility_{window}"] = float(
                        np.std(log_returns[-window:])
                    )
                else:
                    features[f"volatility_{window}"] = float(np.std(log_returns))

        # --- ATR ---
        features["atr"] = self._compute_atr(highs, lows, closes)

        # --- RSI ---
        features["rsi"] = self._compute_rsi(closes)

        # --- Bollinger Bands ---
        bb = self._compute_bollinger(closes)
        features.update(bb)

        # --- EMAs ---
        features["ema_fast"] = self._ema(closes, self._ema_fast)
        features["ema_slow"] = self._ema(closes, self._ema_slow)
        features["ema_trend"] = self._ema(closes, self._ema_trend)
        features["ema_cross"] = 1.0 if features["ema_fast"] > features["ema_slow"] else -1.0

        # --- VWAP ---
        features["vwap"] = self._compute_vwap(candles)
        if features["atr"] > 0:
            features["vwap_deviation"] = (closes[-1] - features["vwap"]) / features["atr"]
        else:
            features["vwap_deviation"] = 0.0

        # --- Volume features ---
        if len(volumes) >= 30:
            vol_mean = np.mean(volumes[-200:]) if len(volumes) >= 200 else np.mean(volumes)
            vol_std = np.std(volumes[-200:]) if len(volumes) >= 200 else np.std(volumes)
            features["volume_zscore"] = float(
                (volumes[-1] - vol_mean) / vol_std if vol_std > 0 else 0
            )
            features["volume_sma_30"] = float(np.mean(volumes[-30:]))
            features["volume_ratio"] = float(
                volumes[-1] / features["volume_sma_30"]
            ) if features["volume_sma_30"] > 0 else 0
        else:
            features["volume_zscore"] = 0.0
            features["volume_sma_30"] = float(np.mean(volumes)) if len(volumes) > 0 else 0
            features["volume_ratio"] = 1.0

        # --- Time features (cyclical encoding) ---
        ts = candles[-1].timestamp
        hour = ts.hour + ts.minute / 60
        features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        features["dow_sin"] = math.sin(2 * math.pi * ts.weekday() / 7)
        features["dow_cos"] = math.cos(2 * math.pi * ts.weekday() / 7)

        return features

    def _compute_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Compute Average True Range."""
        if len(closes) < 2:
            return 0.0

        n = min(len(closes), self._atr_period + 1)
        tr = np.zeros(n - 1)
        for i in range(1, n):
            idx = len(closes) - n + i
            tr[i - 1] = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - closes[idx - 1]),
                abs(lows[idx] - closes[idx - 1]),
            )

        if len(tr) == 0:
            return 0.0
        return float(np.mean(tr[-self._atr_period:]))

    def _compute_rsi(self, closes: np.ndarray) -> float:
        """Compute Relative Strength Index."""
        if len(closes) < self._rsi_period + 1:
            return 50.0

        deltas = np.diff(closes[-(self._rsi_period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _compute_bollinger(self, closes: np.ndarray) -> dict:
        """Compute Bollinger Band features."""
        if len(closes) < self._bb_period:
            return {
                "bb_upper": closes[-1],
                "bb_middle": closes[-1],
                "bb_lower": closes[-1],
                "bb_position": 0.5,
                "bb_width": 0.0,
            }

        window = closes[-self._bb_period:]
        middle = float(np.mean(window))
        std = float(np.std(window))
        upper = middle + self._bb_std * std
        lower = middle - self._bb_std * std

        width = upper - lower
        position = (closes[-1] - lower) / width if width > 0 else 0.5

        return {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_position": position,
            "bb_width": width / middle if middle > 0 else 0,
        }

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Compute Exponential Moving Average (latest value)."""
        if len(data) < period:
            return float(np.mean(data))

        multiplier = 2 / (period + 1)
        ema = float(np.mean(data[:period]))
        for val in data[period:]:
            ema = (val - ema) * multiplier + ema
        return ema

    def _compute_vwap(self, candles: list[Candle]) -> float:
        """Compute Volume-Weighted Average Price."""
        total_pv = 0.0
        total_v = 0.0
        for c in candles[-200:]:  # Use last 200 candles
            typical_price = (c.high + c.low + c.close) / 3
            total_pv += typical_price * c.volume
            total_v += c.volume

        if total_v == 0:
            return candles[-1].close
        return total_pv / total_v
