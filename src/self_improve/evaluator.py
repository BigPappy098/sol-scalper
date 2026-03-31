"""Strategy performance evaluator — computes metrics and decides adjustments."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.db.database import Database
from src.strategies.ensemble import StrategyEnsemble
from src.utils.logging import get_logger

log = get_logger(__name__)


class StrategyEvaluator:
    """Evaluates strategy performance and adjusts ensemble weights."""

    def __init__(
        self,
        database: Database,
        ensemble: StrategyEnsemble,
        config: dict,
    ):
        self._db = database
        self._ensemble = ensemble
        self._mute_threshold = config.get("mute_threshold_sharpe_7d", 0.0)
        self._weight_reduce_threshold = config.get("weight_reduce_threshold_sharpe_24h", 0.0)
        self._weight_increase_threshold = config.get("weight_increase_threshold_sharpe_24h", 1.0)

    async def evaluate_all(self) -> dict[str, dict]:
        """Evaluate all strategies and return metrics + adjustments made."""
        results = {}
        now = datetime.now(timezone.utc)

        for strategy_name in self._ensemble.get_weights():
            metrics_24h = await self._compute_metrics(strategy_name, hours=24)
            metrics_7d = await self._compute_metrics(strategy_name, hours=168)

            # Store metrics
            if metrics_24h.get("trade_count", 0) > 0:
                await self._db.insert_strategy_metrics(strategy_name, 24, metrics_24h)
            if metrics_7d.get("trade_count", 0) > 0:
                await self._db.insert_strategy_metrics(strategy_name, 168, metrics_7d)

            # Decide adjustments
            adjustment = self._decide_adjustment(
                strategy_name, metrics_24h, metrics_7d
            )

            results[strategy_name] = {
                "metrics_24h": metrics_24h,
                "metrics_7d": metrics_7d,
                "adjustment": adjustment,
            }

        return results

    async def _compute_metrics(self, strategy_name: str, hours: int) -> dict:
        """Compute performance metrics for a strategy over a time window."""
        start = datetime.now(timezone.utc) - timedelta(hours=hours)
        trades = await self._db.get_trades(
            start=start, strategy_name=strategy_name, limit=10000
        )

        if not trades:
            return {"trade_count": 0, "win_rate": 0, "sharpe": 0, "avg_r": 0, "max_drawdown": 0}

        pnls = [t["pnl_usd"] for t in trades]
        pnl_pcts = [t["pnl_pct"] for t in trades]

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p <= 0)
        win_rate = wins / len(pnls) if pnls else 0

        # Sharpe ratio (annualized from per-trade returns)
        if len(pnl_pcts) > 1:
            mean_return = np.mean(pnl_pcts)
            std_return = np.std(pnl_pcts)
            # Assume ~100 trades/day for scalping, 365 days
            trades_per_year = (len(pnl_pcts) / hours) * 24 * 365
            sharpe = (
                (mean_return / std_return) * np.sqrt(trades_per_year)
                if std_return > 0
                else 0
            )
        else:
            sharpe = 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Average R (using PnL percentage as proxy)
        avg_r = float(np.mean(pnl_pcts)) if pnl_pcts else 0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "trade_count": len(pnls),
            "win_rate": win_rate,
            "sharpe": float(sharpe),
            "avg_r": avg_r,
            "max_drawdown": max_dd,
            "total_pnl": sum(pnls),
            "profit_factor": profit_factor,
            "wins": wins,
            "losses": losses,
            "weight": self._ensemble.get_weights().get(strategy_name, 0),
        }

    def _decide_adjustment(
        self,
        strategy_name: str,
        metrics_24h: dict,
        metrics_7d: dict,
    ) -> str:
        """Decide what adjustment to make based on metrics."""
        # Need minimum trades to make decisions
        if metrics_7d.get("trade_count", 0) < 10:
            return "insufficient_data"

        sharpe_7d = metrics_7d.get("sharpe", 0)
        sharpe_24h = metrics_24h.get("sharpe", 0)

        # Mute if 7-day Sharpe is negative
        if sharpe_7d < self._mute_threshold and metrics_7d["trade_count"] > 20:
            self._ensemble.mute_strategy(strategy_name)
            log.info(
                "strategy_muted",
                strategy=strategy_name,
                sharpe_7d=sharpe_7d,
            )
            return "muted"

        # Reduce weight if 24h is bad but 7d is ok
        if sharpe_24h < self._weight_reduce_threshold and sharpe_7d > 0:
            current_weight = self._ensemble.get_weights().get(strategy_name, 0)
            new_weight = current_weight * 0.5
            self._ensemble.set_weight(strategy_name, new_weight)
            log.info(
                "strategy_weight_reduced",
                strategy=strategy_name,
                old_weight=current_weight,
                new_weight=new_weight,
            )
            return "weight_reduced"

        # Increase weight if both windows are strong
        if (
            sharpe_24h > self._weight_increase_threshold
            and sharpe_7d > 0.5
        ):
            current_weight = self._ensemble.get_weights().get(strategy_name, 0)
            new_weight = min(current_weight * 1.2, 0.4)
            self._ensemble.set_weight(strategy_name, new_weight)
            log.info(
                "strategy_weight_increased",
                strategy=strategy_name,
                old_weight=current_weight,
                new_weight=new_weight,
            )
            return "weight_increased"

        return "no_change"
