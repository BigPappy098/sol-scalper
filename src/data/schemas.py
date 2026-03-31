"""Data models for market data and trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class ExitReason(str, Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_EXIT = "signal_exit"
    MANUAL = "manual"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Candle:
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0
    vwap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trade_count": self.trade_count,
            "vwap": self.vwap,
        }


@dataclass
class Tick:
    timestamp: datetime
    price: float
    volume: float
    side: str  # "Buy" or "Sell"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "side": self.side,
        }


@dataclass
class OrderbookLevel:
    price: float
    quantity: float


@dataclass
class OrderbookSnapshot:
    timestamp: datetime
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000

    def imbalance(self, depth: int = 10) -> float:
        """Compute bid-ask volume imbalance at given depth.
        Returns value in [-1, 1]. Positive = more bids (bullish).
        """
        bid_vol = sum(b.quantity for b in self.bids[:depth])
        ask_vol = sum(a.quantity for a in self.asks[:depth])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


@dataclass
class Signal:
    timestamp: datetime
    strategy_name: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    target_pct: float
    stop_pct: float
    time_stop_seconds: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "target_pct": self.target_pct,
            "stop_pct": self.stop_pct,
            "time_stop_seconds": self.time_stop_seconds,
            "metadata": self.metadata,
        }


@dataclass
class Position:
    id: str
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    entry_time: datetime
    strategy_name: str
    stop_loss_price: float
    take_profit_price: float
    stop_order_id: str = ""
    tp_order_id: str = ""
    status: str = "open"  # open, closing, closed
    pnl_usd: float = 0.0
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: ExitReason | None = None
    fees_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "strategy_name": self.strategy_name,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "status": self.status,
            "pnl_usd": self.pnl_usd,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "fees_usd": self.fees_usd,
        }


@dataclass
class TradeRecord:
    """Completed trade for storage in DB."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usd: float
    pnl_pct: float
    strategy_name: str
    signal_confidence: float
    exit_reason: str
    fees_usd: float
    metadata: dict = field(default_factory=dict)
