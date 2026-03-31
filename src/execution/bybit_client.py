"""Bybit API wrapper supporting both testnet and mainnet."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from pybit.unified_trading import HTTP, WebSocket

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class BybitClient:
    """Wrapper around pybit for REST and WebSocket operations."""

    def __init__(self):
        self._settings = get_settings()
        self._http: HTTP | None = None
        self._ws_public: WebSocket | None = None
        self._ws_private: WebSocket | None = None
        self._callbacks: dict[str, list] = {}

    def connect(self) -> None:
        """Initialize HTTP and WebSocket connections."""
        testnet = self._settings.is_paper

        self._http = HTTP(
            testnet=testnet,
            api_key=self._settings.bybit_api_key,
            api_secret=self._settings.bybit_api_secret,
        )

        log.info(
            "bybit_http_connected",
            testnet=testnet,
            endpoint=self._settings.bybit_endpoint,
        )

    def start_public_ws(self, callbacks: dict[str, Any]) -> None:
        """Start public WebSocket for market data.

        callbacks: dict mapping channel type to callback function.
            e.g. {"trade": on_trade, "orderbook": on_orderbook, "kline": on_kline}
        """
        testnet = self._settings.is_paper

        self._ws_public = WebSocket(
            testnet=testnet,
            channel_type="linear",
        )

        symbol = self._settings.symbol

        if "trade" in callbacks:
            self._ws_public.trade_stream(
                symbol=symbol,
                callback=callbacks["trade"],
            )
            log.info("ws_subscribed", channel="trade", symbol=symbol)

        if "orderbook" in callbacks:
            depth = self._settings.yaml_config.get("data", {}).get("orderbook_depth", 50)
            self._ws_public.orderbook_stream(
                depth=depth,
                symbol=symbol,
                callback=callbacks["orderbook"],
            )
            log.info("ws_subscribed", channel="orderbook", symbol=symbol, depth=depth)

        if "kline" in callbacks:
            self._ws_public.kline_stream(
                interval=1,
                symbol=symbol,
                callback=callbacks["kline"],
            )
            log.info("ws_subscribed", channel="kline_1m", symbol=symbol)

    def start_private_ws(self, callbacks: dict[str, Any]) -> None:
        """Start private WebSocket for account/order updates."""
        testnet = self._settings.is_paper

        self._ws_private = WebSocket(
            testnet=testnet,
            channel_type="private",
            api_key=self._settings.bybit_api_key,
            api_secret=self._settings.bybit_api_secret,
        )

        if "order" in callbacks:
            self._ws_private.order_stream(callback=callbacks["order"])
            log.info("ws_subscribed", channel="order")

        if "position" in callbacks:
            self._ws_private.position_stream(callback=callbacks["position"])
            log.info("ws_subscribed", channel="position")

        if "execution" in callbacks:
            self._ws_private.execution_stream(callback=callbacks["execution"])
            log.info("ws_subscribed", channel="execution")

    # --- REST API Methods ---

    def get_account_balance(self) -> dict:
        """Get unified account balance."""
        result = self._http.get_wallet_balance(accountType="UNIFIED")
        return result.get("result", {})

    def get_equity(self) -> float:
        """Get total equity in USD."""
        balance = self.get_account_balance()
        coins = balance.get("list", [{}])[0].get("coin", [])
        for coin in coins:
            if coin.get("coin") == "USDT":
                return float(coin.get("equity", 0))
        return 0.0

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        order_link_id: str = "",
    ) -> dict:
        """Place an order on Bybit."""
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if side.lower() in ("long", "buy") else "Sell",
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
        }

        if price is not None and order_type == "Limit":
            params["price"] = str(price)

        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)

        if take_profit is not None:
            params["takeProfit"] = str(take_profit)

        if order_link_id:
            params["orderLinkId"] = order_link_id

        result = self._http.place_order(**params)
        log.info(
            "order_placed",
            symbol=symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            result_code=result.get("retCode"),
        )
        return result.get("result", {})

    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an order."""
        result = self._http.cancel_order(
            category="linear",
            symbol=symbol,
            orderId=order_id,
        )
        log.info("order_cancelled", symbol=symbol, order_id=order_id)
        return result.get("result", {})

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders for a symbol."""
        result = self._http.cancel_all_orders(
            category="linear",
            symbol=symbol,
        )
        log.info("all_orders_cancelled", symbol=symbol)
        return result.get("result", {})

    def get_open_orders(self, symbol: str) -> list[dict]:
        """Get all open orders for a symbol."""
        result = self._http.get_open_orders(
            category="linear",
            symbol=symbol,
        )
        return result.get("result", {}).get("list", [])

    def get_positions(self, symbol: str) -> list[dict]:
        """Get current positions for a symbol."""
        result = self._http.get_positions(
            category="linear",
            symbol=symbol,
        )
        return result.get("result", {}).get("list", [])

    def set_leverage(self, symbol: str, leverage: float) -> dict:
        """Set leverage for a symbol."""
        result = self._http.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )
        log.info("leverage_set", symbol=symbol, leverage=leverage)
        return result.get("result", {})

    def get_klines(
        self,
        symbol: str,
        interval: str = "1",
        limit: int = 200,
        start: int | None = None,
        end: int | None = None,
    ) -> list[dict]:
        """Get historical kline/candlestick data.

        interval: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        """
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        result = self._http.get_kline(**params)
        raw_list = result.get("result", {}).get("list", [])

        candles = []
        for item in raw_list:
            candles.append({
                "timestamp": datetime.fromtimestamp(int(item[0]) / 1000, tz=timezone.utc),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "turnover": float(item[6]),
            })

        # Bybit returns newest first, reverse for chronological order
        candles.reverse()
        return candles

    def get_tickers(self, symbol: str) -> dict:
        """Get current ticker info."""
        result = self._http.get_tickers(
            category="linear",
            symbol=symbol,
        )
        tickers = result.get("result", {}).get("list", [])
        return tickers[0] if tickers else {}

    def get_funding_rate(self, symbol: str) -> dict:
        """Get current funding rate info."""
        result = self._http.get_funding_rate_history(
            category="linear",
            symbol=symbol,
            limit=1,
        )
        rates = result.get("result", {}).get("list", [])
        return rates[0] if rates else {}

    def get_instrument_info(self, symbol: str) -> dict:
        """Get instrument specifications (tick size, lot size, etc.)."""
        result = self._http.get_instruments_info(
            category="linear",
            symbol=symbol,
        )
        instruments = result.get("result", {}).get("list", [])
        return instruments[0] if instruments else {}

    def close(self) -> None:
        """Close all connections."""
        if self._ws_public:
            self._ws_public.exit()
        if self._ws_private:
            self._ws_private.exit()
        log.info("bybit_client_closed")
