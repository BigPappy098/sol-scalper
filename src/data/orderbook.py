"""L2 Orderbook manager with imbalance and depth features."""

from __future__ import annotations

from datetime import datetime, timezone

from src.data.schemas import OrderbookLevel, OrderbookSnapshot
from src.utils.logging import get_logger

log = get_logger(__name__)


class OrderbookManager:
    """Maintains a local copy of the L2 orderbook from WebSocket updates."""

    def __init__(self):
        self._bids: dict[float, float] = {}  # price -> qty
        self._asks: dict[float, float] = {}  # price -> qty
        self._last_update: datetime | None = None
        self._initialized = False

    def on_snapshot(self, data: dict) -> None:
        """Handle a full orderbook snapshot from WebSocket."""
        self._bids.clear()
        self._asks.clear()

        for bid in data.get("b", []):
            price, qty = float(bid[0]), float(bid[1])
            if qty > 0:
                self._bids[price] = qty

        for ask in data.get("a", []):
            price, qty = float(ask[0]), float(ask[1])
            if qty > 0:
                self._asks[price] = qty

        self._last_update = datetime.now(timezone.utc)
        self._initialized = True

    def on_delta(self, data: dict) -> None:
        """Handle an incremental orderbook update."""
        for bid in data.get("b", []):
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = qty

        for ask in data.get("a", []):
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

        self._last_update = datetime.now(timezone.utc)

    def get_snapshot(self, depth: int = 50) -> OrderbookSnapshot | None:
        """Get current orderbook state as a snapshot."""
        if not self._initialized:
            return None

        sorted_bids = sorted(self._bids.items(), key=lambda x: -x[0])[:depth]
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])[:depth]

        return OrderbookSnapshot(
            timestamp=self._last_update or datetime.now(timezone.utc),
            bids=[OrderbookLevel(p, q) for p, q in sorted_bids],
            asks=[OrderbookLevel(p, q) for p, q in sorted_asks],
        )

    def get_features(self, depth_levels: list[int] | None = None) -> dict:
        """Compute orderbook features for strategy consumption.

        Returns dict with:
        - mid_price: midpoint between best bid and ask
        - spread_bps: spread in basis points
        - imbalance_N: bid-ask volume imbalance at depth N
        - bid_depth_N: total bid volume at depth N
        - ask_depth_N: total ask volume at depth N
        """
        snapshot = self.get_snapshot()
        if snapshot is None:
            return {}

        if depth_levels is None:
            depth_levels = [5, 10, 20]

        features = {
            "mid_price": snapshot.mid_price,
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "spread_bps": snapshot.spread_bps,
        }

        for depth in depth_levels:
            features[f"imbalance_{depth}"] = snapshot.imbalance(depth)
            features[f"bid_depth_{depth}"] = sum(
                b.quantity for b in snapshot.bids[:depth]
            )
            features[f"ask_depth_{depth}"] = sum(
                a.quantity for a in snapshot.asks[:depth]
            )

        return features

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def mid_price(self) -> float:
        if not self._bids or not self._asks:
            return 0.0
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        return (best_bid + best_ask) / 2
