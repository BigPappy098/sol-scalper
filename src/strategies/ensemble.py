"""Strategy ensemble — combines signals from multiple strategies."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.schemas import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class StrategyEnsemble:
    """Aggregates signals from multiple strategies using weighted voting.

    Only takes a trade when the weighted consensus exceeds a threshold.
    """

    def __init__(self, strategies: list[BaseStrategy], config: dict):
        self._strategies = {s.name: s for s in strategies}
        self._min_confidence = config.get("min_signal_confidence", 0.55)
        self._min_combined_weight = config.get("min_combined_weight", 0.5)
        self._pending_signals: list[Signal] = []

    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        """Process a candle through all strategies and combine signals.

        Returns the best signal if consensus is reached, None otherwise.
        """
        signals: list[Signal] = []

        for strategy in self._strategies.values():
            if not strategy.is_active:
                continue

            try:
                signal = strategy.on_candle(candle, features, orderbook_features)
                if signal and signal.confidence >= self._min_confidence:
                    signals.append(signal)
            except Exception as e:
                log.error(
                    "strategy_error",
                    strategy=strategy.name,
                    error=str(e),
                )

        if not signals:
            return None

        return self._combine_signals(signals)

    def _combine_signals(self, signals: list[Signal]) -> Signal | None:
        """Combine multiple signals using weighted voting.

        If signals agree on direction, the combined signal uses the
        best individual signal's parameters (target, stop, etc.)
        weighted by strategy weight and confidence.
        """
        # Compute weighted vote for each direction
        long_weight = 0.0
        short_weight = 0.0
        long_signals: list[Signal] = []
        short_signals: list[Signal] = []

        for signal in signals:
            strategy = self._strategies.get(signal.strategy_name)
            if not strategy:
                continue

            weighted_confidence = signal.confidence * strategy.weight

            if signal.direction == SignalDirection.LONG:
                long_weight += weighted_confidence
                long_signals.append(signal)
            elif signal.direction == SignalDirection.SHORT:
                short_weight += weighted_confidence
                short_signals.append(signal)

        # Determine winning direction
        if long_weight > short_weight and long_weight >= self._min_combined_weight:
            return self._pick_best_signal(long_signals, long_weight)
        elif short_weight > long_weight and short_weight >= self._min_combined_weight:
            return self._pick_best_signal(short_signals, short_weight)

        # No consensus — if only one strategy fired, use it if confidence is high enough
        if len(signals) == 1:
            signal = signals[0]
            strategy = self._strategies.get(signal.strategy_name)
            if strategy and signal.confidence * strategy.weight >= self._min_combined_weight * 0.8:
                return signal

        return None

    def _pick_best_signal(self, signals: list[Signal], combined_weight: float) -> Signal:
        """Pick the signal with highest individual confidence as the trade parameters."""
        best = max(signals, key=lambda s: s.confidence)

        # Create a combined signal with the best signal's parameters
        # but with combined confidence
        return Signal(
            timestamp=best.timestamp,
            strategy_name=best.strategy_name,
            direction=best.direction,
            confidence=min(combined_weight, 1.0),
            target_pct=best.target_pct,
            stop_pct=best.stop_pct,
            time_stop_seconds=best.time_stop_seconds,
            metadata={
                **best.metadata,
                "ensemble_weight": combined_weight,
                "contributing_strategies": [s.strategy_name for s in signals],
            },
        )

    def get_strategy(self, name: str) -> BaseStrategy | None:
        return self._strategies.get(name)

    def get_active_strategies(self) -> list[str]:
        return [name for name, s in self._strategies.items() if s.is_active]

    def get_weights(self) -> dict[str, float]:
        return {name: s.weight for name, s in self._strategies.items()}

    def set_weight(self, strategy_name: str, weight: float) -> None:
        if strategy_name in self._strategies:
            self._strategies[strategy_name].set_weight(weight)

    def mute_strategy(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].mute()
            log.info("strategy_muted", strategy=name)

    def unmute_strategy(self, name: str, weight: float | None = None) -> None:
        if name in self._strategies:
            self._strategies[name].unmute(weight)
            log.info("strategy_unmuted", strategy=name, weight=self._strategies[name].weight)
