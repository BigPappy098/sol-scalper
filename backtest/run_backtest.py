"""CLI for running backtests."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd

from backtest.engine import BacktestEngine
from src.config.settings import get_settings
from src.data.schemas import Candle
from src.db.database import Database
from src.strategies.bb_revert import BBRevertStrategy
from src.strategies.ob_fade import OBFadeStrategy
from src.strategies.vol_break import VolBreakStrategy
from src.utils.logging import setup_logging


async def run_backtest(
    timeframe: str = "15s",
    days: int = 7,
    initial_equity: float = 1000.0,
) -> None:
    """Run a backtest using data from TimescaleDB."""
    settings = get_settings()

    # Connect to database
    db = Database(settings.database_url)
    await db.connect()

    # Load historical candles
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    print(f"Loading {timeframe} candles from {start} to {end}...")
    rows = await db.get_candles(timeframe, start, end, limit=1_000_000)

    if not rows:
        print("No data found. Run the system first to collect data, or use download_historical.py")
        await db.close()
        return

    print(f"Loaded {len(rows)} candles")

    # Convert to Candle objects
    candles = [
        Candle(
            timestamp=r["ts"],
            timeframe=r["timeframe"],
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            volume=r["volume"],
            trade_count=r.get("trade_count", 0),
            vwap=r.get("vwap", 0),
        )
        for r in rows
    ]

    # Build strategies
    strategies = []

    bb_config = settings.get_strategy_config("bb_revert")
    if bb_config.get("enabled", True):
        strategies.append(BBRevertStrategy(bb_config))

    vb_config = settings.get_strategy_config("vol_break")
    if vb_config.get("enabled", True):
        strategies.append(VolBreakStrategy(vb_config))

    ob_config = settings.get_strategy_config("ob_fade")
    if ob_config.get("enabled", True):
        strategies.append(OBFadeStrategy(ob_config))

    print(f"Running backtest with {len(strategies)} strategies...")

    # Run backtest
    engine = BacktestEngine(
        strategies=strategies,
        initial_equity=initial_equity,
        risk_per_trade=settings.risk_per_trade,
        max_leverage=settings.max_leverage,
    )

    result = engine.run(candles)
    result.print_summary()

    # Show per-strategy breakdown
    strategy_trades: dict[str, list] = {}
    for trade in result.trades:
        if trade.strategy_name not in strategy_trades:
            strategy_trades[trade.strategy_name] = []
        strategy_trades[trade.strategy_name].append(trade)

    for name, trades in strategy_trades.items():
        wins = sum(1 for t in trades if t.pnl_usd > 0)
        total_pnl = sum(t.pnl_usd for t in trades)
        print(f"  {name}: {len(trades)} trades, {wins}/{len(trades)} wins, PnL: ${total_pnl:.2f}")

    # Save results to CSV
    if result.trades:
        df = pd.DataFrame([
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl_usd": t.pnl_usd,
                "pnl_pct": t.pnl_pct,
                "strategy": t.strategy_name,
                "exit_reason": t.exit_reason,
                "fees": t.fees,
            }
            for t in result.trades
        ])
        output_file = f"backtest_results_{timeframe}_{days}d.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    await db.close()


def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--timeframe", default="15s", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=7, help="Days of history")
    parser.add_argument("--equity", type=float, default=1000.0, help="Initial equity")
    args = parser.parse_args()

    setup_logging("INFO")
    asyncio.run(run_backtest(args.timeframe, args.days, args.equity))


if __name__ == "__main__":
    main()
