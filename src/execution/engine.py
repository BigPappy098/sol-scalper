"""Execution engine — manages order lifecycle from signal to fill."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from src.config.settings import get_settings
from src.data.schemas import (
    ExitReason,
    Position,
    Side,
    Signal,
    SignalDirection,
    TradeRecord,
)
from src.db.database import Database
from src.execution.bybit_client import HyperliquidClient
from src.risk.manager import RiskManager
from src.utils.events import EventBus
from src.utils.logging import get_logger

log = get_logger(__name__)


class ExecutionEngine:
    """Core execution engine that bridges signals to Hyperliquid orders.

    Handles:
    - Signal → order placement
    - Position tracking with stop-loss and take-profit
    - Time-based exits
    - Trade recording
    """

    def __init__(
        self,
        client: HyperliquidClient,
        risk_manager: RiskManager,
        event_bus: EventBus,
        database: Database,
    ):
        self._client = client
        self._risk = risk_manager
        self._event_bus = event_bus
        self._db = database
        self._settings = get_settings()

        self._positions: dict[str, Position] = {}
        self._running = False
        self._instrument_info: dict = {}

    async def start(self) -> None:
        """Initialize the execution engine."""
        self._running = True

        # Get instrument info for proper rounding
        try:
            self._instrument_info = self._client.get_instrument_info(
                self._settings.symbol
            )
            log.info(
                "instrument_info_loaded",
                coin=self._settings.coin,
                sz_decimals=self._instrument_info.get("szDecimals"),
                lot_size=self._instrument_info.get("lotSizeFilter", {}).get("qtyStep"),
            )
        except Exception as e:
            log.warning("instrument_info_failed", error=str(e))

        # Set leverage
        try:
            self._client.set_leverage(
                self._settings.symbol,
                self._settings.max_leverage,
            )
        except Exception as e:
            # May fail if already set
            log.debug("leverage_set_info", error=str(e))

        # Update equity from exchange
        try:
            equity = self._client.get_equity()
            if equity > 0:
                self._risk.update_equity(equity)
                log.info("initial_equity", equity=equity)
            else:
                log.warning("engine_equity_zero", equity=equity)
        except Exception as e:
            log.error("equity_fetch_failed", error=str(e))

        # Start position monitor (handles SL/TP/time stops since HL doesn't have native SL/TP on market orders)
        asyncio.create_task(self._monitor_positions())

        # Start private WebSocket for fill updates
        self._client.start_private_ws({
            "execution": self._on_execution,
        })

        log.info("execution_engine_started")

    async def stop(self) -> None:
        self._running = False

    async def execute_signal(self, signal: Signal) -> Position | None:
        """Execute a trading signal: check risk, place order, track position."""
        current_price = self._get_current_price(signal)
        if current_price <= 0:
            log.warning("no_price_for_signal", strategy=signal.strategy_name)
            return None

        # Risk check and position sizing
        approved, quantity, reason = self._risk.approve_trade(signal, current_price)

        if not approved:
            log.debug(
                "trade_rejected",
                strategy=signal.strategy_name,
                reason=reason,
            )
            return None

        # Round quantity to valid lot size
        quantity = self._round_quantity(quantity)
        if quantity <= 0:
            return None

        # Compute stop-loss and take-profit prices
        if signal.direction == SignalDirection.LONG:
            side = "Buy"
            stop_price = round(current_price * (1 - signal.stop_pct), 2)
            tp_price = round(current_price * (1 + signal.target_pct), 2)
        else:
            side = "Sell"
            stop_price = round(current_price * (1 + signal.stop_pct), 2)
            tp_price = round(current_price * (1 - signal.target_pct), 2)

        # Place order
        position_id = str(uuid.uuid4())[:8]

        try:
            order_result = self._client.place_order(
                symbol=self._settings.coin,
                side=side,
                qty=quantity,
                order_type="Market",
                order_link_id=position_id,
            )

            if not order_result.get("orderId") and not order_result.get("raw", {}).get("status") == "ok":
                log.error("order_failed", result=order_result)
                return None

        except Exception as e:
            log.error("order_placement_error", error=str(e))
            return None

        # Use fill price if available, otherwise use signal price
        fill_price = float(order_result.get("avgPrice", current_price))

        # Create position record
        position = Position(
            id=position_id,
            symbol=self._settings.coin,
            side=Side.LONG if signal.direction == SignalDirection.LONG else Side.SHORT,
            entry_price=fill_price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc),
            strategy_name=signal.strategy_name,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
        )

        self._positions[position_id] = position
        self._risk.register_position(position_id, {
            "position_value": quantity * fill_price,
            "side": position.side.value,
        })

        # Publish trade event
        await self._event_bus.publish("trades:entry", {
            "position_id": position_id,
            "side": position.side.value,
            "entry_price": fill_price,
            "quantity": quantity,
            "stop_loss": stop_price,
            "take_profit": tp_price,
            "strategy": signal.strategy_name,
            "confidence": signal.confidence,
        })

        log.info(
            "position_opened",
            position_id=position_id,
            side=position.side.value,
            entry_price=fill_price,
            quantity=quantity,
            stop_loss=stop_price,
            take_profit=tp_price,
            strategy=signal.strategy_name,
        )

        return position

    async def close_position(
        self,
        position_id: str,
        reason: ExitReason,
        exit_price: float | None = None,
    ) -> TradeRecord | None:
        """Close a position and record the trade."""
        position = self._positions.get(position_id)
        if not position:
            return None

        # Place closing market order (reduce_only)
        close_side = "Sell" if position.side == Side.LONG else "Buy"

        try:
            result = self._client.place_order(
                symbol=position.symbol,
                side=close_side,
                qty=position.quantity,
                order_type="Market",
                reduce_only=True,
            )
            # Use fill price if available
            if result.get("avgPrice"):
                exit_price = float(result["avgPrice"])
        except Exception as e:
            log.error("close_order_failed", position_id=position_id, error=str(e))
            return None

        # Get actual exit price (use provided or estimate from order)
        if exit_price is None:
            exit_price = self._estimate_current_price()

        return await self._finalize_position(position, exit_price, reason)

    async def _finalize_position(
        self,
        position: Position,
        exit_price: float,
        reason: ExitReason,
    ) -> TradeRecord:
        """Record a closed position."""
        now = datetime.now(timezone.utc)

        # Compute PnL
        if position.side == Side.LONG:
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        position_value = position.quantity * position.entry_price
        pnl_usd = position_value * pnl_pct

        # Estimate fees (Hyperliquid: maker 0.01%, taker 0.035%)
        fee_rate = 0.00035  # 0.035% taker
        fees = position_value * fee_rate * 2  # Entry + exit
        pnl_usd -= fees

        # Update risk manager
        self._risk.add_pnl(pnl_usd)
        self._risk.close_position(position.id)

        # Remove from active positions
        self._positions.pop(position.id, None)

        # Create trade record
        trade = TradeRecord(
            entry_time=position.entry_time,
            exit_time=now,
            symbol=position.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            strategy_name=position.strategy_name,
            signal_confidence=0,
            exit_reason=reason.value,
            fees_usd=fees,
        )

        # Store in database
        try:
            await self._db.insert_trade(trade)
        except Exception as e:
            log.error("trade_insert_failed", error=str(e))

        # Publish exit event
        await self._event_bus.publish("trades:exit", {
            "position_id": position.id,
            "side": position.side.value,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl_usd": round(pnl_usd, 4),
            "pnl_pct": round(pnl_pct * 100, 4),
            "fees_usd": round(fees, 4),
            "exit_reason": reason.value,
            "strategy": position.strategy_name,
            "duration_seconds": (now - position.entry_time).total_seconds(),
        })

        log.info(
            "position_closed",
            position_id=position.id,
            side=position.side.value,
            pnl_usd=round(pnl_usd, 4),
            pnl_pct=round(pnl_pct * 100, 4),
            exit_reason=reason.value,
            strategy=position.strategy_name,
        )

        return trade

    async def _monitor_positions(self) -> None:
        """Background task that monitors open positions for SL/TP/time exits.

        Hyperliquid doesn't support native SL/TP on market orders,
        so we monitor price and close manually.
        """
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                current_price = self._estimate_current_price()

                for pos_id, position in list(self._positions.items()):
                    if position.status != "open":
                        continue

                    elapsed = (now - position.entry_time).total_seconds()

                    # Check stop-loss
                    if current_price > 0 and position.stop_loss_price:
                        if position.side == Side.LONG and current_price <= position.stop_loss_price:
                            log.info("stop_loss_triggered", position_id=pos_id, price=current_price)
                            await self.close_position(pos_id, ExitReason.STOP_LOSS, current_price)
                            continue
                        elif position.side == Side.SHORT and current_price >= position.stop_loss_price:
                            log.info("stop_loss_triggered", position_id=pos_id, price=current_price)
                            await self.close_position(pos_id, ExitReason.STOP_LOSS, current_price)
                            continue

                    # Check take-profit
                    if current_price > 0 and position.take_profit_price:
                        if position.side == Side.LONG and current_price >= position.take_profit_price:
                            log.info("take_profit_triggered", position_id=pos_id, price=current_price)
                            await self.close_position(pos_id, ExitReason.TAKE_PROFIT, current_price)
                            continue
                        elif position.side == Side.SHORT and current_price <= position.take_profit_price:
                            log.info("take_profit_triggered", position_id=pos_id, price=current_price)
                            await self.close_position(pos_id, ExitReason.TAKE_PROFIT, current_price)
                            continue

                    # Check time stop (default 3 minutes)
                    time_stop = 180
                    if elapsed > time_stop:
                        log.info("time_stop_triggered", position_id=pos_id, elapsed=elapsed)
                        await self.close_position(pos_id, ExitReason.TIME_STOP)

                await asyncio.sleep(0.5)  # Check every 500ms for scalping
            except Exception as e:
                log.error("position_monitor_error", error=str(e))
                await asyncio.sleep(5)

    def _on_execution(self, message: dict) -> None:
        """Handle user event updates from Hyperliquid WebSocket."""
        try:
            data = message.get("data", message)
            if isinstance(data, dict):
                fills = data.get("fills", [])
                for fill in fills:
                    coin = fill.get("coin", "")
                    px = fill.get("px", "0")
                    sz = fill.get("sz", "0")
                    log.info(
                        "execution_fill",
                        coin=coin,
                        price=px,
                        size=sz,
                        side=fill.get("side", ""),
                        closed_pnl=fill.get("closedPnl", "0"),
                    )
        except Exception as e:
            log.error("execution_callback_error", error=str(e))

    def _get_current_price(self, signal: Signal) -> float:
        """Get current price from signal metadata or ticker."""
        price = signal.metadata.get("price", 0)
        if price > 0:
            return price

        try:
            ticker = self._client.get_tickers(self._settings.symbol)
            return float(ticker.get("lastPrice", 0))
        except Exception:
            return 0.0

    def _estimate_current_price(self) -> float:
        """Get current price for PnL estimation."""
        try:
            ticker = self._client.get_tickers(self._settings.symbol)
            return float(ticker.get("lastPrice", 0))
        except Exception:
            return 0.0

    def _round_quantity(self, quantity: float) -> float:
        """Round quantity to valid lot size."""
        lot_filter = self._instrument_info.get("lotSizeFilter", {})
        qty_step = float(lot_filter.get("qtyStep", 0.1))
        min_qty = float(lot_filter.get("minOrderQty", 0.1))

        if quantity < min_qty:
            return 0.0

        # Round down to nearest step
        rounded = int(quantity / qty_step) * qty_step
        return round(rounded, 4)

    def get_open_positions(self) -> list[dict]:
        return [p.to_dict() for p in self._positions.values()]

    @property
    def position_count(self) -> int:
        return len(self._positions)
