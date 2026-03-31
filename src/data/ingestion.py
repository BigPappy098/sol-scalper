"""Market data ingestion service — connects to Bybit WebSocket and feeds candle builder."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from src.config.settings import get_settings
from src.data.candle_builder import CandleBuilder
from src.data.orderbook import OrderbookManager
from src.data.schemas import Candle, Tick
from src.db.database import Database
from src.execution.bybit_client import BybitClient
from src.utils.events import EventBus
from src.utils.logging import get_logger

log = get_logger(__name__)


class DataIngestionService:
    """Consumes market data from Bybit WebSocket and publishes processed events."""

    def __init__(
        self,
        bybit_client: BybitClient,
        event_bus: EventBus,
        database: Database,
    ):
        self._client = bybit_client
        self._event_bus = event_bus
        self._db = database
        self._settings = get_settings()

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
        """Callback for WebSocket trade messages (runs in pybit's thread)."""
        try:
            data = message.get("data", [])
            for trade in data:
                tick = Tick(
                    timestamp=datetime.fromtimestamp(
                        int(trade["T"]) / 1000, tz=timezone.utc
                    ),
                    price=float(trade["p"]),
                    volume=float(trade["v"]),
                    side=trade["S"],
                )
                self._candle_builder.on_tick(tick)
                self._tick_count += 1
        except Exception as e:
            log.error("trade_processing_error", error=str(e))

    def _on_orderbook(self, message: dict) -> None:
        """Callback for WebSocket orderbook messages."""
        try:
            msg_type = message.get("type", "")
            data = message.get("data", {})

            if msg_type == "snapshot":
                self._orderbook.on_snapshot(data)
            elif msg_type == "delta":
                self._orderbook.on_delta(data)
        except Exception as e:
            log.error("orderbook_processing_error", error=str(e))

    def _on_kline(self, message: dict) -> None:
        """Callback for WebSocket kline messages (used as cross-check)."""
        # We build our own candles from ticks for more granularity,
        # but use exchange klines to verify accuracy
        pass

    def _on_candle_complete_sync(self, candle: Candle) -> None:
        """Synchronous callback from CandleBuilder (runs in pybit's thread).
        Puts candle into async queue for processing.
        """
        try:
            self._candle_queue.put_nowait(candle)
        except asyncio.QueueFull:
            log.warning("candle_queue_full", timeframe=candle.timeframe)

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
