"""A/B testing framework for strategy and model variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.db.database import Database
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ABTest:
    """Tracks an A/B test between a champion and challenger."""

    test_id: str
    champion_name: str
    challenger_name: str
    start_time: datetime
    min_trades: int = 50
    min_hours: int = 48
    sharpe_improvement_threshold: float = 0.3

    # Tracked results
    champion_trades: list[float] = field(default_factory=list)
    challenger_trades: list[float] = field(default_factory=list)

    def record_champion_trade(self, pnl_pct: float) -> None:
        self.champion_trades.append(pnl_pct)

    def record_challenger_trade(self, pnl_pct: float) -> None:
        self.challenger_trades.append(pnl_pct)

    @property
    def is_ready_to_evaluate(self) -> bool:
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        has_enough_time = elapsed >= self.min_hours * 3600
        has_enough_trades = len(self.challenger_trades) >= self.min_trades
        return has_enough_time and has_enough_trades

    def evaluate(self) -> dict:
        """Evaluate the test. Returns dict with result and metrics."""
        import numpy as np

        if not self.challenger_trades or not self.champion_trades:
            return {"result": "insufficient_data"}

        champ_sharpe = self._compute_sharpe(self.champion_trades)
        challenger_sharpe = self._compute_sharpe(self.challenger_trades)

        improvement = challenger_sharpe - champ_sharpe
        should_promote = improvement > self.sharpe_improvement_threshold

        return {
            "result": "promote" if should_promote else "discard",
            "champion_sharpe": champ_sharpe,
            "challenger_sharpe": challenger_sharpe,
            "improvement": improvement,
            "champion_trades": len(self.champion_trades),
            "challenger_trades": len(self.challenger_trades),
        }

    def _compute_sharpe(self, returns: list[float]) -> float:
        import numpy as np

        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        # Annualize assuming ~100 trades/day
        trades_per_year = (len(returns) / max(1, self.min_hours)) * 24 * 365
        return float((mean / std) * np.sqrt(trades_per_year))


class ABTestManager:
    """Manages multiple concurrent A/B tests."""

    def __init__(self, database: Database, config: dict):
        self._db = database
        self._tests: dict[str, ABTest] = {}
        self._min_trades = config.get("ab_test_min_trades", 50)
        self._min_hours = config.get("ab_test_min_hours", 48)
        self._sharpe_threshold = config.get("ab_test_sharpe_improvement", 0.3)

    def create_test(
        self,
        test_id: str,
        champion_name: str,
        challenger_name: str,
    ) -> ABTest:
        """Create a new A/B test."""
        test = ABTest(
            test_id=test_id,
            champion_name=champion_name,
            challenger_name=challenger_name,
            start_time=datetime.now(timezone.utc),
            min_trades=self._min_trades,
            min_hours=self._min_hours,
            sharpe_improvement_threshold=self._sharpe_threshold,
        )
        self._tests[test_id] = test
        log.info(
            "ab_test_created",
            test_id=test_id,
            champion=champion_name,
            challenger=challenger_name,
        )
        return test

    def record_trade(self, strategy_name: str, pnl_pct: float) -> None:
        """Record a trade result for any active A/B test involving this strategy."""
        for test in self._tests.values():
            if strategy_name == test.champion_name:
                test.record_champion_trade(pnl_pct)
            elif strategy_name == test.challenger_name:
                test.record_challenger_trade(pnl_pct)

    def evaluate_tests(self) -> list[dict]:
        """Evaluate all tests that are ready. Returns list of results."""
        results = []
        completed = []

        for test_id, test in self._tests.items():
            if test.is_ready_to_evaluate:
                result = test.evaluate()
                result["test_id"] = test_id
                result["champion"] = test.champion_name
                result["challenger"] = test.challenger_name
                results.append(result)
                completed.append(test_id)

                log.info("ab_test_evaluated", **result)

        # Remove completed tests
        for test_id in completed:
            del self._tests[test_id]

        return results

    def get_active_tests(self) -> list[dict]:
        """Get info about all active tests."""
        return [
            {
                "test_id": t.test_id,
                "champion": t.champion_name,
                "challenger": t.challenger_name,
                "champion_trades": len(t.champion_trades),
                "challenger_trades": len(t.challenger_trades),
                "elapsed_hours": (
                    datetime.now(timezone.utc) - t.start_time
                ).total_seconds()
                / 3600,
            }
            for t in self._tests.values()
        ]
