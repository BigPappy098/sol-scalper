"""Strategy 5: Funding Rate Sentiment.

Trades based on extreme funding rates that predict short-term price moves.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.schemas import Candle, Signal, SignalDirection
from src.execution.bybit_client import HyperliquidClient
from src.strategies.base import BaseStrategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class FundingSentimentStrategy(BaseStrategy):
    """Funding rate sentiment overlay strategy.

    When funding is extremely positive (longs paying shorts), short bias.
    When funding is extremely negative (shorts paying longs), long bias.

    Only trades when aligned with at least one other strategy signal.

    Exit:
    - Target: 0.15%
    - Stop: 0.1%
    - Time stop: 30 min before settlement
    """

    def __init__(self, config: dict, client: HyperliquidClient):
        super().__init__("funding_sent", config)
        self._client = client
        self._funding_threshold = config.get("funding_threshold", 0.0003)
        self._target_pct = config.get("target_pct", 0.0015)
        self._stop_pct = config.get("stop_pct", 0.001)
        self._pre_settlement_minutes = config.get("pre_settlement_exit_minutes", 30)

        self._current_funding: float | None = None
        self._last_funding_check: datetime | None = None
        self._check_interval = 60  # seconds

    def on_candle(
        self,
        candle: Candle,
        features: dict,
        orderbook_features: dict | None = None,
    ) -> Signal | None:
        # Only check on 1m candles (funding changes slowly)
        if candle.timeframe != "1m":
            return None

        # Rate-limit funding rate API calls
        now = candle.timestamp
        if (
            self._last_funding_check
            and (now - self._last_funding_check).total_seconds() < self._check_interval
        ):
            return self._evaluate_signal(candle, features)

        # Fetch current funding rate
        try:
            funding_data = self._client.get_funding_rate("SOL")
            self._current_funding = float(funding_data.get("fundingRate", 0))
            self._last_funding_check = now
        except Exception as e:
            log.debug("funding_rate_fetch_error", error=str(e))
            return None

        return self._evaluate_signal(candle, features)

    def _evaluate_signal(self, candle: Candle, features: dict) -> Signal | None:
        if self._current_funding is None:
            return None

        funding = self._current_funding

        # Check for extreme funding
        if abs(funding) < self._funding_threshold:
            return None

        # Check that price hasn't already moved in expected direction
        return_1h = features.get("return_20", 0)  # Approximate 1h on 1m candles

        if funding > self._funding_threshold:
            # Positive funding = longs overleveraged = short bias
            if return_1h < -0.003:  # Already dropped 0.3%
                return None

            confidence = min(0.5 + abs(funding) / 0.001 * 0.2, 0.85)

            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                target_pct=self._target_pct,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._pre_settlement_minutes * 60,
                metadata={
                    "funding_rate": funding,
                    "price": features.get("price", candle.close),
                },
            )

        elif funding < -self._funding_threshold:
            # Negative funding = shorts overleveraged = long bias
            if return_1h > 0.003:
                return None

            confidence = min(0.5 + abs(funding) / 0.001 * 0.2, 0.85)

            return Signal(
                timestamp=candle.timestamp,
                strategy_name=self.name,
                direction=SignalDirection.LONG,
                confidence=confidence,
                target_pct=self._target_pct,
                stop_pct=self._stop_pct,
                time_stop_seconds=self._pre_settlement_minutes * 60,
                metadata={
                    "funding_rate": funding,
                    "price": features.get("price", candle.close),
                },
            )

        return None
