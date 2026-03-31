"""Self-improvement scheduler — orchestrates evaluation, retraining, and LLM review."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.db.database import Database
from src.risk.manager import RiskManager
from src.self_improve.ab_test import ABTestManager
from src.self_improve.evaluator import StrategyEvaluator
from src.self_improve.llm_agent import LLMAgent
from src.strategies.ensemble import StrategyEnsemble
from src.utils.events import EventBus
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.notifications.telegram_bot import TelegramNotifier

log = get_logger(__name__)


class SelfImprovementScheduler:
    """Runs periodic self-improvement tasks.

    Schedule:
    - Every 6 hours: strategy evaluation + weight adjustment
    - Every 24 hours: full ML model retrain + LLM review
    - Continuous: A/B test monitoring
    """

    def __init__(
        self,
        database: Database,
        ensemble: StrategyEnsemble,
        risk_manager: RiskManager,
        event_bus: EventBus,
        telegram: TelegramNotifier | None,
        config: dict,
    ):
        self._db = database
        self._ensemble = ensemble
        self._risk = risk_manager
        self._event_bus = event_bus
        self._telegram = telegram
        self._config = config

        self._evaluator = StrategyEvaluator(database, ensemble, config)
        self._ab_manager = ABTestManager(database, config)
        self._llm_agent = LLMAgent(database)

        self._running = False
        self._eval_interval = config.get("evaluation_interval_hours", 6) * 3600
        self._retrain_interval = config.get("full_retrain_interval_hours", 24) * 3600

    async def start(self) -> None:
        """Start the self-improvement loops."""
        self._running = True
        asyncio.create_task(self._evaluation_loop())
        asyncio.create_task(self._llm_review_loop())
        asyncio.create_task(self._midnight_reset_loop())
        log.info("self_improvement_started")

    async def stop(self) -> None:
        self._running = False

    async def _evaluation_loop(self) -> None:
        """Run strategy evaluation every N hours."""
        # Wait a bit before first evaluation to accumulate some data
        await asyncio.sleep(300)  # 5 minutes initial delay

        while self._running:
            try:
                log.info("evaluation_starting")
                results = await self._evaluator.evaluate_all()

                # Check A/B tests
                ab_results = self._ab_manager.evaluate_tests()
                for ab_result in ab_results:
                    if ab_result["result"] == "promote":
                        log.info(
                            "ab_test_promotion",
                            challenger=ab_result["challenger"],
                            improvement=ab_result["improvement"],
                        )
                        if self._telegram:
                            await self._telegram.send_message(
                                f"<b>A/B Test Result</b>\n"
                                f"Challenger {ab_result['challenger']} promoted!\n"
                                f"Sharpe improvement: {ab_result['improvement']:.2f}"
                            )

                # Send summary via Telegram
                if self._telegram:
                    summary = self._format_eval_summary(results)
                    await self._telegram.send_message(summary)

                log.info("evaluation_complete", strategies_evaluated=len(results))

            except Exception as e:
                log.error("evaluation_error", error=str(e))

            await asyncio.sleep(self._eval_interval)

    async def _llm_review_loop(self) -> None:
        """Run LLM review every 24 hours."""
        # Wait before first review
        await asyncio.sleep(3600)  # 1 hour initial delay

        while self._running:
            try:
                log.info("llm_review_starting")

                # Get latest evaluation results
                eval_results = await self._evaluator.evaluate_all()

                exposure = self._risk.get_exposure()
                review = await self._llm_agent.review(
                    strategy_metrics=eval_results,
                    equity=exposure["equity"],
                    daily_pnl=exposure["daily_pnl"],
                    drawdown=exposure["drawdown_from_peak"],
                )

                if review:
                    # Apply risk-reducing suggestions automatically
                    risk_suggestions = review.get("risk_suggestions", {})
                    reduce_pct = risk_suggestions.get("reduce_position_pct", 0)
                    if reduce_pct > 0:
                        log.info("llm_risk_reduction", reduce_pct=reduce_pct)

                    # Send review to Telegram
                    if self._telegram:
                        msg = (
                            f"<b>LLM Review</b>\n"
                            f"Regime: {review.get('regime_assessment', 'unknown')}\n"
                            f"Analysis: {review.get('analysis', 'N/A')}\n"
                        )
                        suggestions = review.get("parameter_suggestions", [])
                        if suggestions:
                            msg += "\nSuggestions:\n"
                            for s in suggestions:
                                msg += (
                                    f"  {s['strategy']}.{s['parameter']}: "
                                    f"{s['current']} -> {s['suggested']}\n"
                                    f"  Reason: {s['reason']}\n"
                                )
                        actions = review.get("action_items", [])
                        if actions:
                            msg += "\nAction Items:\n"
                            for a in actions:
                                msg += f"  - {a}\n"
                        await self._telegram.send_message(msg)

                log.info("llm_review_complete")

            except Exception as e:
                log.error("llm_review_error", error=str(e))

            await asyncio.sleep(self._retrain_interval)

    async def _midnight_reset_loop(self) -> None:
        """Reset daily metrics at UTC midnight."""
        while self._running:
            now = datetime.now(timezone.utc)
            # Calculate seconds until next midnight
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if tomorrow <= now:
                tomorrow = tomorrow.replace(day=tomorrow.day + 1)
            sleep_seconds = (tomorrow - now).total_seconds()

            await asyncio.sleep(sleep_seconds)

            if not self._running:
                break

            # Send daily summary before reset
            if self._telegram:
                exposure = self._risk.get_exposure()
                await self._telegram.notify_daily_summary({
                    "date": now.strftime("%Y-%m-%d"),
                    "total_pnl": exposure["daily_pnl"],
                    "equity": exposure["equity"],
                    "trade_count": 0,  # TODO: get from DB
                    "wins": 0,
                    "losses": 0,
                    "total_fees": 0,
                })

            self._risk.reset_daily()
            log.info("daily_reset")

    def _format_eval_summary(self, results: dict) -> str:
        """Format evaluation results for Telegram."""
        msg = "<b>Strategy Evaluation</b>\n"
        for name, data in results.items():
            m24 = data.get("metrics_24h", {})
            adj = data.get("adjustment", "no_change")
            msg += (
                f"\n{name}: "
                f"trades={m24.get('trade_count', 0)} | "
                f"wr={m24.get('win_rate', 0):.0%} | "
                f"sharpe={m24.get('sharpe', 0):.1f} | "
                f"pnl=${m24.get('total_pnl', 0):.2f} | "
                f"[{adj}]"
            )
        return msg
