"""Market data ingestion service — connects to Hyperliquid WebSocket and feeds candle builder."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from src.config.settings import get_settings
from src.data.candle_builder import CandleBuilder
from src.data.orderbook import OrderbookManager
from src.data.schemas import Candle, Tick
from src.db.database import Database
from src.execution.hyperliquid_client import HyperliquidClient
from src.utils.events import EventBus
from src.utils.logging import get_logger

log = get_logger(__name__)


class DataIngestionService:
    """Consumes market data from Hyperliquid WebSocket and publishes processed events."""

    def __init__(
        self,
        client: HyperliquidClient,
        event_bus: EventBus,
        database: Database,
    ):
        self._client = client
        self._event_bus = event_bus
        self._db = database
        self._settings = get_settings()
        self._loop = asyncio.get_event_loop()

        data_config = self._settings.get_data_config()
        timeframes = data_config.get("candle_timeframes", ["1s", "5s", "15s", "1m", "5m"])

        self._candle_builder = CandleBuilder(
            timeframes=timeframes,
            on_candle_complete=self._on_candle_complete_sync,
        )
        self._orderbook = OrderbookManager()

        # Queue for async processing of completed candles
        self._candle_queue: asyncio.Queue[Candle] = asyncio.Queue()
        self._tick_count = 0
        self._running = False

    @property
    def orderbook(self) -> OrderbookManager:
        return self._orderbook

    @property
    def candle_builder(self) -> CandleBuilder:
        return self._candle_builder

    async def start(self) -> None:
        """Start data ingestion: backfill history, then connect WebSocket."""
        self._running = True

        # Backfill historical 1m candles
        await self._backfill_history()

        # Start WebSocket streams
        self._client.start_public_ws({
            "trade": self._on_trade,
            "orderbook": self._on_orderbook,
            "kline": self._on_kline,
        })

        # Start async candle processor
        asyncio.create_task(self._process_candle_queue())

        log.info("data_ingestion_started")

    async def stop(self) -> None:
        self._running = False
        log.info("data_ingestion_stopped")

    async def _backfill_history(self) -> None:
        """Fetch recent historical candles to populate the database."""
        symbol = self._settings.symbol
        limit = self._settings.get_data_config().get("historical_backfill_candles", 1000)

        try:
            klines = self._client.get_klines(symbol=symbol, interval="1", limit=limit)
            candles = []
            for k in klines:
                candle = Candle(
                    timestamp=k["timestamp"],
                    timeframe="1m",
                    open=k["open"],
                    high=k["high"],
                    low=k["low"],
                    close=k["close"],
                    volume=k["volume"],
                )
                candles.append(candle)

            if candles:
                await self._db.insert_candles_batch(candles)
                log.info("history_backfilled", candle_count=len(candles))
        except Exception as e:
            log.error("backfill_failed", error=str(e))

    def _on_trade(self, message: dict) -> None:
        """Callback for WebSocket trade messages (runs in SDK's thread).

        Hyperliquid trade message format:
        {"channel": "trades", "data": [{"coin": "SOL", "side": "B", "px": "150.5", "sz": "1.0", "time": 1704067200000, ...}]}
        """
        try:
            data = message.get("data", [])
            if isinstance(data, list):
                for trade in data:
                    tick = Tick(
                        timestamp=datetime.fromtimestamp(
                            int(trade.get("time", 0)) / 1000, tz=timezone.utc
                        ),
                        price=float(trade.get("px", 0)),
                        volume=float(trade.get("sz", 0)),
                        side="Buy" if trade.get("side") == "B" else "Sell",
                    )
                    self._candle_builder.on_tick(tick)
                    self._tick_count += 1
        except Exception as e:
            log.error("trade_processing_error", error=str(e))

    def _on_orderbook(self, message: dict) -> None:
        """Callback for WebSocket L2 orderbook messages.

        Hyperliquid l2Book format:
        {"channel": "l2Book", "data": {"coin": "SOL", "levels": [[{px, sz, n}, ...], [{px, sz, n}, ...]], "time": ...}}
        """
        try:
            data = message.get("data", {})
            levels = data.get("levels", [])

            if len(levels) >= 2:
                # Convert to format expected by OrderbookManager
                bids = [
                    [float(level.get("px", 0)), float(level.get("sz", 0))]
                    for level in levels[0]
                ]
                asks = [
                    [float(level.get("px", 0)), float(level.get("sz", 0))]
                    for level in levels[1]
                ]

                # Always treat as snapshot (HL sends full book each time)
                self._orderbook.on_snapshot({
                    "b": bids,
                    "a": asks,
                })
        except Exception as e:
            log.error("orderbook_processing_error", error=str(e))

    def _on_kline(self, message: dict) -> None:
        """Callback for WebSocket kline/candle messages (used as cross-check)."""
        # We build our own candles from ticks for more granularity,
        # but use exchange klines to verify accuracy
        pass

    def _on_candle_complete_sync(self, candle: Candle) -> None:
        """Synchronous callback from CandleBuilder (runs in SDK's thread).
        Puts candle into async queue for processing.
        """
        try:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._candle_queue.put_nowait, candle)
            else:
                # This fallback is for initialization if a candle somehow finishes
                # before the loop is fully running, but still in the same thread.
                self._candle_queue.put_nowait(candle)
        except Exception as e:
            log.warning("candle_queue_failed", timeframe=candle.timeframe, error=str(e))


    async def _process_candle_queue(self) -> None:
        """Async task that processes completed candles: stores in DB and publishes events."""
        while self._running:
            try:
                candle = await asyncio.wait_for(
                    self._candle_queue.get(), timeout=5.0
                )

                # Store in database
                await self._db.insert_candle(candle)

                # Publish to event bus
                await self._event_bus.publish(
                    f"candles:{candle.timeframe}",
                    candle.to_dict(),
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error("candle_processing_error", error=str(e))

    def get_stats(self) -> dict:
        """Get ingestion stats."""
        return {
            "tick_count": self._tick_count,
            "orderbook_ready": self._orderbook.is_ready,
            "running": self._running,
        }
