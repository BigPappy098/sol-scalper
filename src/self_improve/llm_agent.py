"""LLM agent that reviews trades and suggests parameter adjustments."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.config.settings import get_settings
from src.db.database import Database
from src.utils.logging import get_logger

log = get_logger(__name__)

REVIEW_PROMPT = """You are an expert quantitative trading analyst reviewing the performance of an autonomous SOL/USDT scalping system.

## Current System State

Equity: ${equity:.2f}
Daily PnL: ${daily_pnl:.2f}
Drawdown from peak: {drawdown:.1%}

## Strategy Performance (last 24h)
{strategy_metrics}

## Recent Trades (last 50)
{recent_trades}

## Feature Importance Shifts
{feature_shifts}

## Instructions

Analyze the trading performance and provide structured recommendations.

Rules:
- Never suggest increasing risk beyond 5x leverage
- Focus on parameter tuning within 20% of current values
- If unsure, recommend no changes
- Consider market regime (trending vs ranging, high vs low volatility)
- Look for patterns in losing trades (time of day, strategy, conditions)

Respond with ONLY valid JSON in this exact format:
{{
    "analysis": "2-3 sentence summary of performance",
    "regime_assessment": "high_volatility | low_volatility | trending | ranging",
    "parameter_suggestions": [
        {{
            "strategy": "strategy_name",
            "parameter": "param_name",
            "current": 0.0,
            "suggested": 0.0,
            "reason": "brief reason"
        }}
    ],
    "risk_suggestions": {{
        "reduce_position_pct": 0
    }},
    "action_items": ["list of actionable items"]
}}"""


class LLMAgent:
    """Uses Claude to review trades and suggest improvements."""

    def __init__(self, database: Database):
        self._db = database
        self._settings = get_settings()
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self._settings.anthropic_api_key
                )
            except ImportError:
                log.warning("anthropic_not_installed")
                return None
        return self._client

    async def review(
        self,
        strategy_metrics: dict[str, dict],
        equity: float,
        daily_pnl: float,
        drawdown: float,
    ) -> dict | None:
        """Run a full trade review and return structured suggestions."""
        client = self._get_client()
        if not client:
            return None

        if not self._settings.anthropic_api_key:
            log.warning("llm_review_skipped", reason="no API key")
            return None

        # Gather recent trades
        trades = await self._db.get_trades(limit=50)

        # Format data for the prompt
        trades_str = self._format_trades(trades)
        metrics_str = self._format_metrics(strategy_metrics)

        prompt = REVIEW_PROMPT.format(
            equity=equity,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            strategy_metrics=metrics_str,
            recent_trades=trades_str,
            feature_shifts="Not yet tracked (Phase 2 feature)",
        )

        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = message.content[0].text

            # Parse JSON response
            result = json.loads(response_text)
            log.info("llm_review_complete", analysis=result.get("analysis", ""))
            return result

        except json.JSONDecodeError as e:
            log.error("llm_response_parse_error", error=str(e))
            return None
        except Exception as e:
            log.error("llm_review_error", error=str(e))
            return None

    def _format_trades(self, trades: list[dict]) -> str:
        """Format trades for the prompt."""
        if not trades:
            return "No recent trades."

        lines = []
        for t in trades[:50]:
            lines.append(
                f"  {t.get('ts_entry', '')} | {t.get('side', '')} | "
                f"strategy={t.get('strategy_name', '')} | "
                f"entry=${t.get('entry_price', 0):.4f} -> exit=${t.get('exit_price', 0):.4f} | "
                f"pnl=${t.get('pnl_usd', 0):.4f} ({t.get('pnl_pct', 0):.2%}) | "
                f"exit={t.get('exit_reason', '')}"
            )
        return "\n".join(lines)

    def _format_metrics(self, metrics: dict[str, dict]) -> str:
        """Format strategy metrics for the prompt."""
        if not metrics:
            return "No strategy metrics available."

        lines = []
        for name, data in metrics.items():
            m24 = data.get("metrics_24h", {})
            lines.append(
                f"  {name}: trades={m24.get('trade_count', 0)} | "
                f"win_rate={m24.get('win_rate', 0):.1%} | "
                f"sharpe={m24.get('sharpe', 0):.2f} | "
                f"pnl=${m24.get('total_pnl', 0):.2f} | "
                f"weight={m24.get('weight', 0):.2f} | "
                f"adjustment={data.get('adjustment', 'none')}"
            )
        return "\n".join(lines)
