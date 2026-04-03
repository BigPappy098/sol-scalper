"""Telegram bot for notifications and commands."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from src.execution.engine import ExecutionEngine
    from src.risk.manager import RiskManager
    from src.strategies.ensemble import StrategyEnsemble


class TelegramNotifier:
    """Sends notifications and handles commands via Telegram.

    Notifications:
    - Trade entries and exits
    - Daily PnL summary
    - System errors
    - Strategy changes

    Commands:
    - /status — equity, positions, active strategies, uptime
    - /pnl — daily/weekly/monthly PnL
    - /trades — last 10 trades
    - /strategies — strategy weights and status
    - /halt — stop all trading
    - /resume — resume trading
    """

    def __init__(self):
        self._settings = get_settings()
        self._bot = None
        self._app = None
        self._running = False

        # Component references (set after initialization)
        self._execution_engine: ExecutionEngine | None = None
        self._risk_manager: RiskManager | None = None
        self._ensemble: StrategyEnsemble | None = None

        self._start_time = datetime.now(timezone.utc)

    def set_components(
        self,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
        ensemble: StrategyEnsemble,
    ) -> None:
        """Set references to other components for command handling."""
        self._execution_engine = execution_engine
        self._risk_manager = risk_manager
        self._ensemble = ensemble

    async def start(self) -> None:
        """Start the Telegram bot."""
        token = self._settings.telegram_bot_token
        if not token:
            log.warning("telegram_bot_disabled", reason="no token configured")
            return

        try:
            from telegram import Update
            from telegram.ext import (
                ApplicationBuilder,
                CommandHandler,
                ContextTypes,
            )

            self._app = ApplicationBuilder().token(token).build()

            # Register command handlers
            self._app.add_handler(CommandHandler("status", self._cmd_status))
            self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
            self._app.add_handler(CommandHandler("trades", self._cmd_trades))
            self._app.add_handler(CommandHandler("strategies", self._cmd_strategies))
            self._app.add_handler(CommandHandler("health", self._cmd_health))

            # Start polling in background
            await self._app.initialize()
            await self._app.start()
            await self._app.updater.start_polling(drop_pending_updates=True)

            self._running = True
            log.info("telegram_bot_started")

        except ImportError:
            log.warning("telegram_library_not_installed")
        except Exception as e:
            log.error("telegram_bot_start_failed", error=str(e))

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        self._running = False

    async def send_message(self, text: str) -> None:
        """Send a message to the configured chat."""
        chat_id = self._settings.telegram_chat_id
        if not chat_id or not self._app:
            return

        try:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as e:
            log.error("telegram_send_failed", error=str(e))

    async def notify_trade_entry(self, trade_data: dict) -> None:
        """Send trade entry notification."""
        msg = (
            f"<b>TRADE OPENED</b>\n"
            f"Strategy: {trade_data.get('strategy', 'unknown')}\n"
            f"Side: {trade_data.get('side', '').upper()}\n"
            f"Entry: ${trade_data.get('entry_price', 0):.4f}\n"
            f"Qty: {trade_data.get('quantity', 0):.2f} SOL\n"
            f"SL: ${trade_data.get('stop_loss', 0):.4f}\n"
            f"TP: ${trade_data.get('take_profit', 0):.4f}\n"
            f"Confidence: {trade_data.get('confidence', 0):.1%}"
        )
        await self.send_message(msg)

    async def notify_trade_exit(self, trade_data: dict) -> None:
        """Send trade exit notification."""
        pnl = trade_data.get("pnl_usd", 0)
        pnl_emoji = "+" if pnl >= 0 else ""

        msg = (
            f"<b>TRADE CLOSED</b>\n"
            f"Strategy: {trade_data.get('strategy', 'unknown')}\n"
            f"Side: {trade_data.get('side', '').upper()}\n"
            f"Entry: ${trade_data.get('entry_price', 0):.4f} -> "
            f"Exit: ${trade_data.get('exit_price', 0):.4f}\n"
            f"PnL: {pnl_emoji}${pnl:.4f} ({pnl_emoji}{trade_data.get('pnl_pct', 0):.2f}%)\n"
            f"Fees: ${trade_data.get('fees_usd', 0):.4f}\n"
            f"Exit: {trade_data.get('exit_reason', 'unknown')}\n"
            f"Duration: {trade_data.get('duration_seconds', 0):.0f}s"
        )
        await self.send_message(msg)

    async def notify_error(self, error_msg: str) -> None:
        """Send error notification."""
        msg = f"<b>ERROR</b>\n{error_msg}"
        await self.send_message(msg)

    async def notify_daily_summary(self, summary: dict) -> None:
        """Send daily PnL summary."""
        pnl = summary.get("total_pnl", 0)
        pnl_emoji = "+" if pnl >= 0 else ""

        msg = (
            f"<b>DAILY SUMMARY</b>\n"
            f"Date: {summary.get('date', 'today')}\n"
            f"Trades: {summary.get('trade_count', 0)}\n"
            f"Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}\n"
            f"PnL: {pnl_emoji}${pnl:.2f}\n"
            f"Fees: ${summary.get('total_fees', 0):.2f}\n"
            f"Equity: ${summary.get('equity', 0):.2f}"
        )
        await self.send_message(msg)

    # --- Command Handlers ---

    async def _cmd_status(self, update, context) -> None:
        """Handle /status command."""
        exposure = self._risk_manager.get_exposure() if self._risk_manager else {}
        positions = self._execution_engine.get_open_positions() if self._execution_engine else []
        active = self._ensemble.get_active_strategies() if self._ensemble else []

        uptime = datetime.now(timezone.utc) - self._start_time
        hours = int(uptime.total_seconds() / 3600)
        minutes = int((uptime.total_seconds() % 3600) / 60)

        msg = (
            f"<b>STATUS</b>\n"
            f"Equity: ${exposure.get('equity', 0):.2f}\n"
            f"Open Positions: {len(positions)}\n"
            f"Leverage: {exposure.get('leverage', 0):.1f}x\n"
            f"Daily PnL: ${exposure.get('daily_pnl', 0):.2f}\n"
            f"Drawdown: {exposure.get('drawdown_from_peak', 0):.1%}\n"
            f"Active Strategies: {', '.join(active) or 'none'}\n"
            f"Uptime: {hours}h {minutes}m"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_pnl(self, update, context) -> None:
        """Handle /pnl command."""
        exposure = self._risk_manager.get_exposure() if self._risk_manager else {}
        peak = f"${self._risk_manager._peak_equity:.2f}" if self._risk_manager else "N/A"
        msg = (
            f"<b>PnL</b>\n"
            f"Daily: ${exposure.get('daily_pnl', 0):.2f}\n"
            f"Equity: ${exposure.get('equity', 0):.2f}\n"
            f"Peak Equity: {peak}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_trades(self, update, context) -> None:
        """Handle /trades command."""
        positions = self._execution_engine.get_open_positions() if self._execution_engine else []
        if not positions:
            await update.message.reply_text("No open positions.")
            return

        msg = "<b>OPEN POSITIONS</b>\n"
        for p in positions:
            msg += (
                f"\n{p['side'].upper()} {p['quantity']} SOL @ ${p['entry_price']:.4f}\n"
                f"  Strategy: {p['strategy_name']}\n"
                f"  SL: ${p['stop_loss_price']:.4f} | TP: ${p['take_profit_price']:.4f}\n"
            )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_strategies(self, update, context) -> None:
        """Handle /strategies command."""
        if not self._ensemble:
            await update.message.reply_text("Ensemble not initialized.")
            return

        weights = self._ensemble.get_weights()
        active = self._ensemble.get_active_strategies()

        msg = "<b>STRATEGIES</b>\n"
        for name, weight in weights.items():
            status = "ACTIVE" if name in active else "MUTED"
            msg += f"\n{name}: weight={weight:.2f} [{status}]"

        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_health(self, update, context) -> None:
        """Handle /health command."""
        msg = (
            f"<b>HEALTH</b>\n"
            f"Bot: running\n"
            f"Mode: {self._settings.trading_mode}\n"
            f"Symbol: {self._settings.symbol}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
