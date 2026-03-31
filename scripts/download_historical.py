"""Download historical SOL/USDT data from Bybit for backtesting."""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timedelta, timezone

from src.config.settings import get_settings
from src.data.schemas import Candle
from src.db.database import Database
from src.execution.bybit_client import BybitClient
from src.utils.logging import setup_logging, get_logger

log = get_logger(__name__)


async def download_historical(days: int = 90, interval: str = "1") -> None:
    """Download historical kline data and store in TimescaleDB.

    Args:
        days: Number of days of history to download
        interval: Kline interval (1, 3, 5, 15, 30, 60, etc.)
    """
    settings = get_settings()

    # Connect
    db = Database(settings.database_url)
    await db.connect()

    client = BybitClient()
    client.connect()

    symbol = settings.symbol
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    print(f"Downloading {days} days of {interval}m {symbol} data...")

    # Bybit returns max 200 candles per request
    # For 1m candles: 200 candles = ~3.3 hours
    total_candles = 0
    current_end = end

    while current_end > start:
        end_ms = int(current_end.timestamp() * 1000)
        start_ms = int((current_end - timedelta(hours=3)).timestamp() * 1000)

        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=200,
                end=end_ms,
            )

            if not klines:
                break

            candles = []
            for k in klines:
                candles.append(Candle(
                    timestamp=k["timestamp"],
                    timeframe=f"{interval}m" if interval.isdigit() else interval,
                    open=k["open"],
                    high=k["high"],
                    low=k["low"],
                    close=k["close"],
                    volume=k["volume"],
                ))

            await db.insert_candles_batch(candles)
            total_candles += len(candles)

            # Move window back
            oldest = min(k["timestamp"] for k in klines)
            current_end = oldest - timedelta(seconds=1)

            if total_candles % 1000 == 0:
                print(f"  Downloaded {total_candles} candles, back to {current_end}")

            # Rate limit
            time.sleep(0.1)

        except Exception as e:
            log.error("download_error", error=str(e))
            time.sleep(1)

    print(f"Done! Downloaded {total_candles} candles total.")

    client.close()
    await db.close()


def main():
    parser = argparse.ArgumentParser(description="Download historical data")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--interval", default="1", help="Kline interval (1=1min)")
    args = parser.parse_args()

    setup_logging("INFO")
    asyncio.run(download_historical(args.days, args.interval))


if __name__ == "__main__":
    main()
