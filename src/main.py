"""Main entry point — starts all services and runs the trading loop."""

from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone

from src.config.settings import get_settings
from src.dashboard import TradingDashboard
from src.data.feature_store import FeatureStore
from src.data.ingestion import DataIngestionService
from src.data.schemas import Candle
from src.db.database import Database
from src.execution.bybit_client import HyperliquidClient
from src.execution.engine import ExecutionEngine
from src.notifications.telegram_bot import TelegramNotifier
from src.risk.manager import RiskManager
from src.self_improve.scheduler import SelfImprovementScheduler
from src.strategies.bb_revert import BBRevertStrategy
from src.strategies.ensemble import StrategyEnsemble
from src.strategies.ob_fade import OBFadeStrategy
from src.strategies.vol_break import VolBreakStrategy
from src.utils.events import EventBus
from src.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


class TradingSystem:
    """Orchestrates all components of the trading system."""

    def __init__(self):
        self._settings = get_settings()
        self._running = False

        # Core infrastructure
        self._db: Database | None = None
        self._event_bus: EventBus | None = None
        self._client: HyperliquidClient | None = None

    async def _wait_for_services(self, timeout: int = 60) -> None:
        """Wait for PostgreSQL and Redis to be ready."""
        import asyncpg
        import redis.asyncio as aioredis

        pg_dsn = self._settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

        for attempt in range(timeout):
            try:
                conn = await asyncpg.connect(pg_dsn)
                await conn.close()
                log.info("postgres_ready", attempts=attempt + 1)
                break
            except Exception:
                if attempt % 10 == 0:
                    log.info("waiting_for_postgres", attempt=attempt)
                await asyncio.sleep(1)
        else:
            raise RuntimeError("PostgreSQL did not become ready")

        for attempt in range(timeout):
            try:
                r = aioredis.from_url(self._settings.redis_url)
                await r.ping()
                await r.aclose()
                log.info("redis_ready", attempts=attempt + 1)
                break
            except Exception:
                if attempt % 10 == 0:
                    log.info("waiting_for_redis", attempt=attempt)
                await asyncio.sleep(1)
        else:
            raise RuntimeError("Redis did not become ready")

        # Components
        self._ingestion: DataIngestionService | None = None
        self._feature_store: FeatureStore | None = None
        self._ensemble: StrategyEnsemble | None = None
        self._execution: ExecutionEngine | None = None
        self._risk: RiskManager | None = None
        self._telegram: TelegramNotifier | None = None
        self._self_improve: SelfImprovementScheduler | None = None
        self._dashboard: TradingDashboard | None = None

    async def start(self) -> None:
        """Initialize and start all components."""
        log.info(
            "system_starting",
            mode=self._settings.trading_mode,
            symbol=self._settings.symbol,
        )

        # 1. Wait for PostgreSQL and Redis to be ready (supervisord starts them)
        await self._wait_for_services()

        # 2. Connect infrastructure
        self._db = Database(self._settings.database_url)
        await self._db.connect()

        self._event_bus = EventBus(self._settings.redis_url)
        await self._event_bus.connect()

        self._client = HyperliquidClient()
        self._client.connect()

        # 2. Initialize risk manager with current equity
        self._risk = RiskManager()
        try:
            equity = self._client.get_equity()
            if equity > 0:
                self._risk.update_equity(equity)
                log.info("initial_equity", equity=equity)
            else:
                log.warning("equity_zero_or_missing", equity=equity,
                            address=self._client.address,
                            hint="Account returned $0 — check logs for direct_api_raw_response details")
                self._risk.update_equity(0.0)
        except Exception as e:
            log.error("equity_fetch_failed", error=str(e),
                      hint="Check network connectivity and API credentials")
            self._risk.update_equity(0.0)

        # 3. Initialize feature store
        self._feature_store = FeatureStore(self._settings.get_feature_config())

        # 4. Initialize strategies
        strategies = self._build_strategies()
        ensemble_config = self._settings.yaml_config.get("ensemble", {})
        self._ensemble = StrategyEnsemble(strategies, ensemble_config)
        log.info("strategies_loaded", active=self._ensemble.get_active_strategies())

        # 5. Initialize execution engine
        self._execution = ExecutionEngine(
            self._client, self._risk, self._event_bus, self._db
        )
        await self._execution.start()

        # 6. Initialize data ingestion
        self._ingestion = DataIngestionService(
            self._client, self._event_bus, self._db
        )
        await self._ingestion.start()

        # 7. Initialize Telegram
        self._telegram = TelegramNotifier()
        self._telegram.set_components(self._execution, self._risk, self._ensemble)
        await self._telegram.start()

        # 8. Initialize self-improvement scheduler
        self._self_improve = SelfImprovementScheduler(
            database=self._db,
            ensemble=self._ensemble,
            risk_manager=self._risk,
            event_bus=self._event_bus,
            telegram=self._telegram,
            config=self._settings.get_self_improve_config(),
        )
        await self._self_improve.start()

        # 9. Start live dashboard
        self._dashboard = TradingDashboard()
        self._dashboard.set_components(
            execution=self._execution,
            risk=self._risk,
            ensemble=self._ensemble,
            ingestion=self._ingestion,
            feature_store=self._feature_store,
        )
        self._dashboard.start_background()

        # 10. Start the main trading loop
        self._running = True
        await self._telegram.send_message(
            f"<b>System Started</b>\n"
            f"Mode: {self._settings.trading_mode}\n"
            f"Symbol: {self._settings.symbol}\n"
            f"Equity: ${self._risk.equity:.2f}\n"
            f"Strategies: {', '.join(self._ensemble.get_active_strategies())}"
        )

        # Start event consumers
        asyncio.create_task(self._trading_loop())
        asyncio.create_task(self._trade_notification_loop())
        asyncio.create_task(self._equity_snapshot_loop())
        asyncio.create_task(self._dashboard_feed_loop())

        log.info("system_started")

    async def stop(self) -> None:
        """Gracefully shut down all components."""
        log.info("system_stopping")
        self._running = False

        if self._telegram:
            await self._telegram.send_message("<b>System Shutting Down</b>")

        if self._dashboard:
            self._dashboard.stop()
        if self._self_improve:
            await self._self_improve.stop()
        if self._ingestion:
            await self._ingestion.stop()
        if self._execution:
            await self._execution.stop()
        if self._telegram:
            await self._telegram.stop()
        if self._client:
            self._client.close()
        if self._event_bus:
            await self._event_bus.close()
        if self._db:
            await self._db.close()

        log.info("system_stopped")

    async def _trading_loop(self) -> None:
        """Main trading loop — listens for candle events and runs the ensemble."""
        # Subscribe to all candle timeframes
        timeframes = self._settings.get_data_config().get(
            "candle_timeframes", ["1s", "5s", "15s", "1m", "5m"]
        )

        # We process candles from the feature store on each event
        async for _msg_id, candle_data in self._event_bus.subscribe("candles:15s"):
            if not self._running:
                break

            try:
                # Reconstruct candle from event data
                candle = Candle(
                    timestamp=datetime.fromisoformat(candle_data["timestamp"]),
                    timeframe=candle_data["timeframe"],
                    open=float(candle_data["open"]),
                    high=float(candle_data["high"]),
                    low=float(candle_data["low"]),
                    close=float(candle_data["close"]),
                    volume=float(candle_data["volume"]),
                    trade_count=int(candle_data.get("trade_count", 0)),
                    vwap=float(candle_data.get("vwap", 0)),
                )

                # Update feature store
                features = self._feature_store.on_candle(candle)

                if not self._feature_store.has_enough_data(candle.timeframe):
                    continue

                # Get orderbook features
                ob_features = None
                if self._ingestion and self._ingestion.orderbook.is_ready:
                    ob_features = self._ingestion.orderbook.get_features()

                # Add current price to features for execution
                if ob_features and ob_features.get("mid_price"):
                    features["price"] = ob_features["mid_price"]

                # Run ensemble
                signal = self._ensemble.on_candle(candle, features, ob_features)

                if signal:
                    # Add price to signal metadata
                    if "price" not in signal.metadata:
                        signal.metadata["price"] = features.get("price", candle.close)

                    await self._execution.execute_signal(signal)

            except Exception as e:
                log.error("trading_loop_error", error=str(e))

        # Also consume other timeframe candles for feature updates
        asyncio.create_task(self._feature_update_loop("1s"))
        asyncio.create_task(self._feature_update_loop("5s"))
        asyncio.create_task(self._feature_update_loop("1m"))
        asyncio.create_task(self._feature_update_loop("5m"))

    async def _feature_update_loop(self, timeframe: str) -> None:
        """Update features from candles at a specific timeframe."""
        async for _msg_id, candle_data in self._event_bus.subscribe(f"candles:{timeframe}"):
            if not self._running:
                break
            try:
                candle = Candle(
                    timestamp=datetime.fromisoformat(candle_data["timestamp"]),
                    timeframe=candle_data["timeframe"],
                    open=float(candle_data["open"]),
                    high=float(candle_data["high"]),
                    low=float(candle_data["low"]),
                    close=float(candle_data["close"]),
                    volume=float(candle_data["volume"]),
                    trade_count=int(candle_data.get("trade_count", 0)),
                    vwap=float(candle_data.get("vwap", 0)),
                )
                self._feature_store.on_candle(candle)

                # Also run ensemble for strategies on this timeframe
                features = self._feature_store.get_features(timeframe)
                ob_features = None
                if self._ingestion and self._ingestion.orderbook.is_ready:
                    ob_features = self._ingestion.orderbook.get_features()

                if features.get("price"):
                    signal = self._ensemble.on_candle(candle, features, ob_features)
                    if signal:
                        signal.metadata["price"] = features.get("price", candle.close)
                        await self._execution.execute_signal(signal)

            except Exception as e:
                log.error("feature_update_error", timeframe=timeframe, error=str(e))

    async def _trade_notification_loop(self) -> None:
        """Listen for trade events and send Telegram notifications."""
        if not self._telegram:
            return

        # Entry notifications
        asyncio.create_task(self._notify_on_stream("trades:entry", self._telegram.notify_trade_entry))
        # Exit notifications
        asyncio.create_task(self._notify_on_stream("trades:exit", self._telegram.notify_trade_exit))

    async def _notify_on_stream(self, stream: str, callback) -> None:
        """Generic stream listener that calls a notification callback."""
        async for _msg_id, data in self._event_bus.subscribe(stream):
            if not self._running:
                break
            try:
                await callback(data)
            except Exception as e:
                log.error("notification_error", stream=stream, error=str(e))

    async def _dashboard_feed_loop(self) -> None:
        """Feed live data to the dashboard by subscribing to Redis events."""
        if not self._dashboard:
            return

        async def price_feed():
            async for _msg_id, data in self._event_bus.subscribe("candles:1s"):
                if not self._running:
                    break
                price = float(data.get("close", 0))
                if price > 0:
                    self._dashboard.update_price(price)

        async def trade_feed():
            async for _msg_id, data in self._event_bus.subscribe("trades:exit", last_id="$"):
                if not self._running:
                    break
                self._dashboard.record_trade(data)

        async def signal_feed():
            async for _msg_id, data in self._event_bus.subscribe("trades:entry", last_id="$"):
                if not self._running:
                    break
                self._dashboard.record_signal()

        async def equity_feed():
            while self._running:
                if self._risk:
                    self._dashboard.update_equity(self._risk.equity)
                await asyncio.sleep(5)

        asyncio.create_task(price_feed())
        asyncio.create_task(trade_feed())
        asyncio.create_task(signal_feed())
        asyncio.create_task(equity_feed())

    async def _equity_snapshot_loop(self) -> None:
        """Periodically snapshot equity for tracking."""
        while self._running:
            try:
                equity = self._client.get_equity()
                if equity > 0:
                    self._risk.update_equity(equity)
                else:
                    log.warning("equity_snapshot_zero", equity=equity)

                await self._db.execute(
                    """
                    INSERT INTO equity_snapshots (ts, equity_usd, open_positions, daily_pnl)
                    VALUES (NOW(), $1, $2, $3)
                    """,
                    equity,
                    self._execution.position_count if self._execution else 0,
                    self._risk.daily_pnl,
                )
            except Exception as e:
                log.error("equity_snapshot_error", error=str(e))

            await asyncio.sleep(60)  # Every minute

    def _build_strategies(self) -> list:
        """Build strategy instances from config."""
        strategies = []

        bb_config = self._settings.get_strategy_config("bb_revert")
        if bb_config.get("enabled", True):
            strategies.append(BBRevertStrategy(bb_config))

        vb_config = self._settings.get_strategy_config("vol_break")
        if vb_config.get("enabled", True):
            strategies.append(VolBreakStrategy(vb_config))

        ob_config = self._settings.get_strategy_config("ob_fade")
        if ob_config.get("enabled", True):
            strategies.append(OBFadeStrategy(ob_config))

        # ML and Funding strategies added in later phases
        # ml_config = self._settings.get_strategy_config("ml_signal")
        # funding_config = self._settings.get_strategy_config("funding_sent")

        return strategies


async def main() -> None:
    """Application entry point."""
    settings = get_settings()
    log_level = settings.yaml_config.get("logging", {}).get("level", "INFO")
    setup_logging(log_level)

    system = TradingSystem()

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()

    def handle_signal(sig):
        log.info("shutdown_signal_received", signal=sig)
        asyncio.create_task(system.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        await system.start()
        # Keep running until stopped
        while system._running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
