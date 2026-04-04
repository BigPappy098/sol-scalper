"""Database connection and query helpers using asyncpg."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from src.data.schemas import Candle, TradeRecord
from src.utils.logging import get_logger

log = get_logger(__name__)


class Database:
    """Async database interface for TimescaleDB."""

    def __init__(self, dsn: str):
        # Convert SQLAlchemy DSN to asyncpg format
        self._dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool and run migrations if needed."""
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        await self._run_migrations()
        log.info("database_connected")

    async def _run_migrations(self) -> None:
        """Run SQL migrations if tables don't exist yet."""
        from pathlib import Path

        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            # Check if the candles table exists
            exists = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'candles')"
            )
            if exists:
                return

            log.info("running_migrations")
            migration_file = Path(__file__).parent / "migrations" / "001_init.sql"
            sql = migration_file.read_text()
            await conn.execute(sql)
            log.info("migrations_complete")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()

    def _ensure_pool(self) -> asyncpg.Pool:
        """Return the pool or raise if not connected."""
        if self._pool is None:
            raise RuntimeError("Database not connected — call connect() first")
        return self._pool

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query."""
        async with self._ensure_pool().acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self._ensure_pool().acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Fetch a single row."""
        async with self._ensure_pool().acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value."""
        async with self._ensure_pool().acquire() as conn:
            return await conn.fetchval(query, *args)

    # --- Candle Operations ---

    async def insert_candle(self, candle: Candle) -> None:
        """Insert a single candle."""
        await self.execute(
            """
            INSERT INTO candles (ts, timeframe, open, high, low, close, volume, trade_count, vwap)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (ts, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                trade_count = EXCLUDED.trade_count,
                vwap = EXCLUDED.vwap
            """,
            candle.timestamp,
            candle.timeframe,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.trade_count,
            candle.vwap,
        )

    async def insert_candles_batch(self, candles: list[Candle]) -> None:
        """Insert multiple candles efficiently."""
        if not candles:
            return
        async with self._ensure_pool().acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO candles (ts, timeframe, open, high, low, close, volume, trade_count, vwap)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (ts, timeframe) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trade_count = EXCLUDED.trade_count,
                    vwap = EXCLUDED.vwap
                """,
                [
                    (c.timestamp, c.timeframe, c.open, c.high, c.low, c.close, c.volume, c.trade_count, c.vwap)
                    for c in candles
                ],
            )

    async def get_candles(
        self,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[dict]:
        """Get candles for a timeframe and time range."""
        rows = await self.fetch(
            """
            SELECT ts, timeframe, open, high, low, close, volume, trade_count, vwap
            FROM candles
            WHERE timeframe = $1 AND ts >= $2 AND ts <= $3
            ORDER BY ts ASC
            LIMIT $4
            """,
            timeframe,
            start,
            end,
            limit,
        )
        return [dict(r) for r in rows]

    async def get_latest_candles(self, timeframe: str, count: int = 200) -> list[dict]:
        """Get the most recent N candles for a timeframe."""
        rows = await self.fetch(
            """
            SELECT ts, timeframe, open, high, low, close, volume, trade_count, vwap
            FROM candles
            WHERE timeframe = $1
            ORDER BY ts DESC
            LIMIT $2
            """,
            timeframe,
            count,
        )
        # Reverse to chronological order
        return [dict(r) for r in reversed(rows)]

    # --- Trade Operations ---

    async def insert_trade(self, trade: TradeRecord) -> int:
        """Insert a completed trade record. Returns the trade ID."""
        import json

        trade_id = await self.fetchval(
            """
            INSERT INTO trades (
                ts_entry, ts_exit, side, entry_price, exit_price,
                quantity, pnl_usd, pnl_pct, strategy_name,
                signal_confidence, exit_reason, fees_usd, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
            RETURNING id
            """,
            trade.entry_time,
            trade.exit_time,
            trade.side,
            trade.entry_price,
            trade.exit_price,
            trade.quantity,
            trade.pnl_usd,
            trade.pnl_pct,
            trade.strategy_name,
            trade.signal_confidence,
            trade.exit_reason,
            trade.fees_usd,
            json.dumps(trade.metadata),
        )
        return trade_id

    async def get_trades(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        strategy_name: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get trade records with optional filters."""
        conditions = []
        params = []
        idx = 1

        if start:
            conditions.append(f"ts_entry >= ${idx}")
            params.append(start)
            idx += 1
        if end:
            conditions.append(f"ts_entry <= ${idx}")
            params.append(end)
            idx += 1
        if strategy_name:
            conditions.append(f"strategy_name = ${idx}")
            params.append(strategy_name)
            idx += 1

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = await self.fetch(
            f"""
            SELECT * FROM trades
            {where}
            ORDER BY ts_entry DESC
            LIMIT ${idx}
            """,
            *params,
            limit,
        )
        return [dict(r) for r in rows]

    async def get_daily_pnl(self, days: int = 7) -> list[dict]:
        """Get daily PnL summary."""
        rows = await self.fetch(
            """
            SELECT
                DATE(ts_entry) as date,
                COUNT(*) as trade_count,
                SUM(pnl_usd) as total_pnl,
                SUM(fees_usd) as total_fees,
                AVG(pnl_pct) as avg_pnl_pct,
                COUNT(*) FILTER (WHERE pnl_usd > 0) as wins,
                COUNT(*) FILTER (WHERE pnl_usd <= 0) as losses
            FROM trades
            WHERE ts_entry >= NOW() - INTERVAL '%s days'
            GROUP BY DATE(ts_entry)
            ORDER BY date DESC
            """,
            days,
        )
        return [dict(r) for r in rows]

    # --- Strategy Metrics ---

    async def insert_strategy_metrics(
        self,
        strategy_name: str,
        window_hours: int,
        metrics: dict,
    ) -> None:
        """Insert strategy performance metrics."""
        await self.execute(
            """
            INSERT INTO strategy_metrics (
                ts, strategy_name, window_hours,
                win_rate, avg_r, sharpe, max_drawdown,
                trade_count, weight
            ) VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8)
            """,
            strategy_name,
            window_hours,
            metrics.get("win_rate", 0),
            metrics.get("avg_r", 0),
            metrics.get("sharpe", 0),
            metrics.get("max_drawdown", 0),
            metrics.get("trade_count", 0),
            metrics.get("weight", 0),
        )

    # --- Model Registry ---

    async def register_model(
        self,
        model_name: str,
        model_version: int,
        artifact_path: str,
        val_metrics: dict,
        config: dict,
    ) -> int:
        """Register a trained model. Returns model ID."""
        import json

        return await self.fetchval(
            """
            INSERT INTO models (
                ts_trained, model_name, model_version,
                artifact_path, val_metrics, is_active, config
            ) VALUES (NOW(), $1, $2, $3, $4::jsonb, FALSE, $5::jsonb)
            RETURNING id
            """,
            model_name,
            model_version,
            artifact_path,
            json.dumps(val_metrics),
            json.dumps(config),
        )

    async def get_active_model(self, model_name: str) -> dict | None:
        """Get the currently active model for a given name."""
        row = await self.fetchrow(
            """
            SELECT * FROM models
            WHERE model_name = $1 AND is_active = TRUE
            ORDER BY ts_trained DESC
            LIMIT 1
            """,
            model_name,
        )
        return dict(row) if row else None

    async def activate_model(self, model_id: int, model_name: str) -> None:
        """Set a model as active (deactivate others with same name)."""
        async with self._ensure_pool().acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE models SET is_active = FALSE WHERE model_name = $1",
                    model_name,
                )
                await conn.execute(
                    "UPDATE models SET is_active = TRUE WHERE id = $1",
                    model_id,
                )
