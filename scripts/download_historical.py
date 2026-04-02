"""Download historical SOL/USD data from Hyperliquid for backtesting."""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timedelta, timezone

from src.config.settings import get_settings
from src.data.schemas import Candle
from src.db.database import Database
from src.execution.bybit_client import HyperliquidClient
from src.utils.logging import setup_logging, get_logger

log = get_logger(__name__)


async def download_historical(days: int = 90, interval: str = "1m") -> None:
    """Download historical kline data and store in TimescaleDB.

    Args:
        days: Number of days of history to download
        interval: Kline interval ("1m", "5m", "15m", "1h", etc.)
    """
    settings = get_settings()

    # Connect
    db = Database(settings.database_url)
    await db.connect()

    client = HyperliquidClient()
    client.connect()

    coin = settings.coin
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # Map interval to timeframe string
    if interval.isdigit():
        interval = f"{interval}m"

    print(f"Downloading {days} days of {interval} {coin} data from Hyperliquid...")

    # Hyperliquid returns max 5000 candles per request
    # For 1m candles: 5000 candles = ~3.5 days
    total_candles = 0
    current_start = start

    while current_start < end:
        start_ms = int(current_start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        try:
            klines = client.get_klines(
                symbol=coin,
                interval=interval,
                limit=5000,
                start=start_ms,
                end=end_ms,
            )

            if not klines:
                break

            candles = []
            for k in klines:
                candles.append(Candle(
                    timestamp=k["timestamp"],
                    timeframe=interval,
                    open=k["open"],
                    high=k["high"],
                    low=k["low"],
                    close=k["close"],
                    volume=k["volume"],
                ))

            await db.insert_candles_batch(candles)
            total_candles += len(candles)

            # Move window forward past the last candle
            newest = max(k["timestamp"] for k in klines)
            current_start = newest + timedelta(seconds=1)

            if total_candles % 1000 == 0:
                print(f"  Downloaded {total_candles} candles, up to {current_start}")

            # Rate limit
            time.sleep(0.2)

        except Exception as e:
            log.error("download_error", error=str(e))
            time.sleep(1)

    print(f"Done! Downloaded {total_candles} candles total.")

    client.close()
    await db.close()


def main():
    parser = argparse.ArgumentParser(description="Download historical data from Hyperliquid")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--interval", default="1m", help="Kline interval (1m, 5m, 15m, 1h)")
    args = parser.parse_args()

    setup_logging("INFO")
    asyncio.run(download_historical(args.days, args.interval))


if __name__ == "__main__":
    main()
