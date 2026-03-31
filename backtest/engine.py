"""Event-driven backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.data.candle_builder import CandleBuilder
from src.data.feature_store import FeatureStore
from src.data.schemas import Candle, Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.strategies.ensemble import StrategyEnsemble
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime | None = None
    side: str = ""
    entry_price: float = 0
    exit_price: float = 0
    quantity: float = 0
    pnl_usd: float = 0
    pnl_pct: float = 0
    strategy_name: str = ""
    exit_reason: str = ""
    fees: float = 0


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_equity: float = 1000.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        return wins / len(self.trades)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0
        returns = [t.pnl_pct for t in self.trades]
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0
        # Annualize assuming 50 trades/day
        return float((mean / std) * np.sqrt(50 * 365))

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def summary(self) -> dict:
        return {
            "total_trades": len(self.trades),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 2),
            "final_equity": round(self.equity_curve[-1], 2) if self.equity_curve else self.initial_equity,
            "avg_trade_pnl": round(np.mean([t.pnl_usd for t in self.trades]), 4) if self.trades else 0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n=== BACKTEST RESULTS ===")
        for k, v in s.items():
            print(f"  {k}: {v}")
        print("========================\n")


class BacktestEngine:
    """Replays historical candles through strategies to simulate trading."""

    def __init__(
        self,
        strategies: list[BaseStrategy],
        initial_equity: float = 1000.0,
        risk_per_trade: float = 0.01,
        max_leverage: float = 5.0,
        maker_fee: float = 0.0001,  # 0.01%
        taker_fee: float = 0.0006,  # 0.06%
    ):
        self._initial_equity = initial_equity
        self._risk_per_trade = risk_per_trade
        self._max_leverage = max_leverage
        self._maker_fee = maker_fee
        self._taker_fee = taker_fee

        ensemble_config = {"min_signal_confidence": 0.55, "min_combined_weight": 0.5}
        self._ensemble = StrategyEnsemble(strategies, ensemble_config)
        self._feature_store = FeatureStore()

    def run(self, candles: list[Candle]) -> BacktestResult:
        """Run backtest on a list of candles sorted chronologically."""
        result = BacktestResult(initial_equity=self._initial_equity)
        equity = self._initial_equity
        result.equity_curve.append(equity)

        open_trade: BacktestTrade | None = None

        for i, candle in enumerate(candles):
            # Update features
            features = self._feature_store.on_candle(candle)

            if not self._feature_store.has_enough_data(candle.timeframe, 30):
                continue

            # Check if open trade should be closed
            if open_trade:
                close_result = self._check_exit(
                    open_trade, candle, features, equity
                )
                if close_result:
                    open_trade, equity = close_result
                    result.trades.append(open_trade)
                    result.equity_curve.append(equity)
                    open_trade = None

            # Only look for new signals if no open trade
            if open_trade is None:
                signal = self._ensemble.on_candle(candle, features)

                if signal:
                    trade = self._open_trade(signal, candle, equity)
                    if trade:
                        open_trade = trade

        # Close any remaining open trade at last price
        if open_trade and candles:
            last_candle = candles[-1]
            open_trade.exit_time = last_candle.timestamp
            open_trade.exit_price = last_candle.close
            open_trade.exit_reason = "backtest_end"
            self._compute_pnl(open_trade, equity)
            equity += open_trade.pnl_usd
            result.trades.append(open_trade)
            result.equity_curve.append(equity)

        return result

    def _open_trade(
        self, signal: Signal, candle: Candle, equity: float
    ) -> BacktestTrade | None:
        """Open a new trade based on a signal."""
        price = candle.close
        if price <= 0 or equity <= 0:
            return None

        # Position sizing
        risk_amount = equity * self._risk_per_trade
        position_value = risk_amount / signal.stop_pct if signal.stop_pct > 0 else 0
        max_position = equity * self._max_leverage
        position_value = min(position_value, max_position)
        quantity = position_value / price

        if quantity < 0.1:
            return None

        return BacktestTrade(
            entry_time=candle.timestamp,
            side=signal.direction.value,
            entry_price=price,
            quantity=quantity,
            strategy_name=signal.strategy_name,
        )

    def _check_exit(
        self,
        trade: BacktestTrade,
        candle: Candle,
        features: dict,
        equity: float,
    ) -> tuple[BacktestTrade, float] | None:
        """Check if trade should be exited. Returns (closed_trade, new_equity) or None."""
        price = candle.close

        if trade.side == "long":
            pnl_pct = (price - trade.entry_price) / trade.entry_price
            # Stop loss check using candle low
            hit_stop = (candle.low - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - price) / trade.entry_price
            hit_stop = (trade.entry_price - candle.high) / trade.entry_price

        # Get strategy config for this trade
        strategy = self._ensemble.get_strategy(trade.strategy_name)
        if not strategy:
            return None

        stop_pct = strategy.config.get("stop_pct", 0.003)
        target_pct = strategy.config.get("target_pct", 0.002)
        time_stop = strategy.config.get("time_stop_seconds", 180)

        # Stop loss
        if hit_stop <= -stop_pct:
            trade.exit_price = trade.entry_price * (
                (1 - stop_pct) if trade.side == "long" else (1 + stop_pct)
            )
            trade.exit_reason = "stop_loss"
            trade.exit_time = candle.timestamp
            self._compute_pnl(trade, equity)
            return trade, equity + trade.pnl_usd

        # Take profit
        if trade.side == "long":
            hit_tp = (candle.high - trade.entry_price) / trade.entry_price
        else:
            hit_tp = (trade.entry_price - candle.low) / trade.entry_price

        if hit_tp >= target_pct:
            trade.exit_price = trade.entry_price * (
                (1 + target_pct) if trade.side == "long" else (1 - target_pct)
            )
            trade.exit_reason = "take_profit"
            trade.exit_time = candle.timestamp
            self._compute_pnl(trade, equity)
            return trade, equity + trade.pnl_usd

        # Time stop
        if trade.entry_time:
            elapsed = (candle.timestamp - trade.entry_time).total_seconds()
            if elapsed >= time_stop:
                trade.exit_price = price
                trade.exit_reason = "time_stop"
                trade.exit_time = candle.timestamp
                self._compute_pnl(trade, equity)
                return trade, equity + trade.pnl_usd

        return None

    def _compute_pnl(self, trade: BacktestTrade, equity: float) -> None:
        """Compute PnL including fees."""
        if trade.side == "long":
            trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price

        position_value = trade.quantity * trade.entry_price
        trade.fees = position_value * self._taker_fee * 2  # Entry + exit
        trade.pnl_usd = position_value * trade.pnl_pct - trade.fees
