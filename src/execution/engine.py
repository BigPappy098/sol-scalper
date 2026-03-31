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
from src.execution.bybit_client import BybitClient
from src.risk.manager import RiskManager
from src.utils.events import EventBus
from src.utils.logging import get_logger

log = get_logger(__name__)


class ExecutionEngine:
    """Core execution engine that bridges signals to Bybit orders.

    Handles:
    - Signal → order placement
    - Position tracking with stop-loss and take-profit
    - Time-based exits
    - Trade recording
    """

    def __init__(
        self,
        bybit_client: BybitClient,
        risk_manager: RiskManager,
        event_bus: EventBus,
        database: Database,
    ):
        self._client = bybit_client
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
                symbol=self._settings.symbol,
                tick_size=self._instrument_info.get("priceFilter", {}).get("tickSize"),
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
            self._risk.update_equity(equity)
            log.info("initial_equity", equity=equity)
        except Exception as e:
            log.error("equity_fetch_failed", error=str(e))

        # Start position monitor
        asyncio.create_task(self._monitor_positions())

        # Start private WebSocket for order/position updates
        self._client.start_private_ws({
            "execution": self._on_execution,
            "position": self._on_position_update,
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
                symbol=self._settings.symbol,
                side=side,
                qty=quantity,
                order_type="Market",
                stop_loss=stop_price,
                take_profit=tp_price,
                order_link_id=position_id,
            )

            if not order_result.get("orderId"):
                log.error("order_failed", result=order_result)
                return None

        except Exception as e:
            log.error("order_placement_error", error=str(e))
            return None

        # Create position record
        position = Position(
            id=position_id,
            symbol=self._settings.symbol,
            side=Side.LONG if signal.direction == SignalDirection.LONG else Side.SHORT,
            entry_price=current_price,
            quantity=quantity,
            entry_time=datetime.now(timezone.utc),
            strategy_name=signal.strategy_name,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
        )

        self._positions[position_id] = position
        self._risk.register_position(position_id, {
            "position_value": quantity * current_price,
            "side": position.side.value,
        })

        # Publish trade event
        await self._event_bus.publish("trades:entry", {
            "position_id": position_id,
            "side": position.side.value,
            "entry_price": current_price,
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
            entry_price=current_price,
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

        # Place closing market order
        close_side = "Sell" if position.side == Side.LONG else "Buy"

        try:
            self._client.place_order(
                symbol=position.symbol,
                side=close_side,
                qty=position.quantity,
                order_type="Market",
                reduce_only=True,
            )
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

        # Estimate fees (maker: 0.01%, taker: 0.06% — assume taker for market orders)
        fee_rate = 0.0006  # 0.06% taker
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
        """Background task that monitors open positions for time-based exits."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                for pos_id, position in list(self._positions.items()):
                    if position.status != "open":
                        continue

                    # Check time stop (we rely on exchange for SL/TP)
                    # Time stops need to be managed by us
                    elapsed = (now - position.entry_time).total_seconds()

                    # Default time stop: 3 minutes for all strategies
                    time_stop = 180
                    if elapsed > time_stop:
                        log.info(
                            "time_stop_triggered",
                            position_id=pos_id,
                            elapsed=elapsed,
                        )
                        await self.close_position(pos_id, ExitReason.TIME_STOP)

                await asyncio.sleep(1)
            except Exception as e:
                log.error("position_monitor_error", error=str(e))
                await asyncio.sleep(5)

    def _on_execution(self, message: dict) -> None:
        """Handle execution/fill updates from private WebSocket."""
        try:
            data = message.get("data", [])
            for exec_data in data:
                order_link_id = exec_data.get("orderLinkId", "")
                exec_type = exec_data.get("execType", "")

                if order_link_id in self._positions and exec_type == "Trade":
                    exec_price = float(exec_data.get("execPrice", 0))
                    log.info(
                        "execution_fill",
                        position_id=order_link_id,
                        exec_price=exec_price,
                        exec_qty=exec_data.get("execQty"),
                    )
        except Exception as e:
            log.error("execution_callback_error", error=str(e))

    def _on_position_update(self, message: dict) -> None:
        """Handle position updates from private WebSocket."""
        try:
            data = message.get("data", [])
            for pos_data in data:
                size = float(pos_data.get("size", 0))
                symbol = pos_data.get("symbol", "")

                # If position size went to 0, a stop-loss or take-profit was hit
                if size == 0 and symbol == self._settings.symbol:
                    # Find matching position and close it
                    for pos_id, position in list(self._positions.items()):
                        if position.symbol == symbol and position.status == "open":
                            # Determine exit reason from the close price
                            close_price = float(pos_data.get("avgPrice", 0))
                            if close_price > 0:
                                if position.side == Side.LONG:
                                    pnl = close_price - position.entry_price
                                else:
                                    pnl = position.entry_price - close_price

                                reason = (
                                    ExitReason.TAKE_PROFIT
                                    if pnl > 0
                                    else ExitReason.STOP_LOSS
                                )

                                # Schedule async finalization
                                asyncio.get_event_loop().create_task(
                                    self._finalize_position(position, close_price, reason)
                                )
        except Exception as e:
            log.error("position_update_error", error=str(e))

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
