"""Builds candles at multiple timeframes from raw trade ticks."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from src.data.schemas import Candle, Tick
from src.utils.logging import get_logger

log = get_logger(__name__)


class CandleBuilder:
    """Aggregates ticks into OHLCV candles at multiple timeframes.

    Supports: 1s, 5s, 15s, 1m, 5m
    Calls on_candle_complete callback when a candle closes.
    """

    TIMEFRAME_SECONDS = {
        "1s": 1,
        "5s": 5,
        "15s": 15,
        "1m": 60,
        "5m": 300,
    }

    def __init__(
        self,
        timeframes: list[str],
        on_candle_complete: Callable[[Candle], None] | None = None,
    ):
        self._timeframes = timeframes
        self._on_candle_complete = on_candle_complete

        # Current building candle per timeframe
        self._current: dict[str, _CandleAccumulator] = {}
        for tf in timeframes:
            if tf not in self.TIMEFRAME_SECONDS:
                raise ValueError(f"Unsupported timeframe: {tf}")

        # Recent completed candles (ring buffer per timeframe)
        self._history: dict[str, list[Candle]] = defaultdict(list)
        self._max_history = 500

    def on_tick(self, tick: Tick) -> list[Candle]:
        """Process a new tick. Returns list of any newly completed candles."""
        completed = []

        for tf in self._timeframes:
            interval = self.TIMEFRAME_SECONDS[tf]
            # Compute the candle bucket this tick belongs to
            epoch = int(tick.timestamp.timestamp())
            bucket_start = epoch - (epoch % interval)
            bucket_ts = datetime.fromtimestamp(bucket_start, tz=timezone.utc)

            if tf not in self._current:
                # First tick for this timeframe
                self._current[tf] = _CandleAccumulator(bucket_ts, tf)

            acc = self._current[tf]

            if bucket_ts > acc.timestamp:
                # New candle period — close the current candle
                candle = acc.to_candle()
                completed.append(candle)
                self._history[tf].append(candle)
                if len(self._history[tf]) > self._max_history:
                    self._history[tf] = self._history[tf][-self._max_history:]

                # Start new accumulator
                self._current[tf] = _CandleAccumulator(bucket_ts, tf)
                acc = self._current[tf]

            acc.update(tick)

        if completed and self._on_candle_complete:
            for candle in completed:
                self._on_candle_complete(candle)

        return completed

    def get_current_candle(self, timeframe: str) -> Candle | None:
        """Get the in-progress (unclosed) candle for a timeframe."""
        acc = self._current.get(timeframe)
        if acc and acc.trade_count > 0:
            return acc.to_candle()
        return None

    def get_history(self, timeframe: str, count: int = 200) -> list[Candle]:
        """Get recent completed candles for a timeframe."""
        history = self._history.get(timeframe, [])
        return history[-count:]


class _CandleAccumulator:
    """Accumulates ticks into a single candle."""

    def __init__(self, timestamp: datetime, timeframe: str):
        self.timestamp = timestamp
        self.timeframe = timeframe
        self.open = 0.0
        self.high = float("-inf")
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0.0
        self.trade_count = 0
        self._total_price_volume = 0.0

    def update(self, tick: Tick) -> None:
        if self.trade_count == 0:
            self.open = tick.price
        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.volume
        self.trade_count += 1
        self._total_price_volume += tick.price * tick.volume

    def to_candle(self) -> Candle:
        vwap = self._total_price_volume / self.volume if self.volume > 0 else self.close
        return Candle(
            timestamp=self.timestamp,
            timeframe=self.timeframe,
            open=self.open,
            high=self.high if self.high != float("-inf") else self.close,
            low=self.low if self.low != float("inf") else self.close,
            close=self.close,
            volume=self.volume,
            trade_count=self.trade_count,
            vwap=vwap,
        )
