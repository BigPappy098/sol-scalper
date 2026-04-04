"""Hyperliquid DEX client — REST and WebSocket for perp trading."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from src.config.settings import get_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


def _retry(func, retries: int = 3, base_delay: float = 1.0):
    """Call *func* with exponential-backoff retries on failure or None response."""
    last_err = None
    for attempt in range(retries):
        try:
            result = func()
            if result is not None:
                return result
            log.warning("api_returned_none", attempt=attempt + 1, func=getattr(func, "__qualname__", str(func)))
        except Exception as e:
            last_err = e
            log.warning("api_call_failed", attempt=attempt + 1, error=str(e))
        if attempt < retries - 1:
            time.sleep(base_delay * (2 ** attempt))
    if last_err:
        log.error("api_call_exhausted_retries", error=str(last_err))
    return None


class HyperliquidClient:
    """Wrapper around the Hyperliquid Python SDK for REST and WebSocket operations.

    Despite the filename (kept for import compatibility), this is a Hyperliquid client.
    """

    def __init__(self):
        self._settings = get_settings()
        self._exchange: Exchange | None = None
        self._info: Info | None = None
        self._info_ws: Info | None = None
        self._wallet = None
        self._address: str = ""
        self._callbacks: dict[str, list[Callable]] = {}
        self._ws_running = False
        self._meta: dict = {}  # Asset metadata (sz_decimals, etc.)
        self._asset_index: dict[str, int] = {}  # coin -> index mapping

    @property
    def address(self) -> str:
        return self._address

    def connect(self) -> None:
        """Initialize HTTP connections (Exchange + Info)."""
        base_url = self._settings.hl_base_url

        # Create wallet from private key (API wallet — used for signing)
        self._wallet = eth_account.Account.from_key(self._settings.hl_private_key)
        signing_addr = self._wallet.address

        # HL_WALLET_ADDRESS = main wallet where funds live.
        # API wallet signs on its behalf but has no funds itself.
        configured_addr = self._settings.hl_wallet_address
        if configured_addr and configured_addr.lower() != signing_addr.lower():
            self._address = configured_addr
            log.info(
                "api_wallet_setup",
                api_wallet=signing_addr,
                main_wallet=configured_addr,
                hint="API wallet signs trades; main wallet holds funds",
            )
        else:
            self._address = signing_addr

        # Exchange client — account_address is the main wallet we trade on behalf of
        self._exchange = Exchange(
            wallet=self._wallet,
            base_url=base_url,
            account_address=self._address,
        )

        # Info client (for reading data — no wallet needed)
        self._info = Info(base_url, skip_ws=True)

        # Load asset metadata
        self._load_meta()

        log.info(
            "hyperliquid_connected",
            address=self._address,
            testnet=self._settings.is_paper,
            base_url=base_url,
        )

        # Health check: verify we can read the account and it has funds
        self._health_check()

    def _health_check(self) -> None:
        """Verify API connectivity and account state after connect."""
        import requests as _requests

        try:
            # 1. Check perps clearinghouse balance
            resp = _requests.post(
                f"{self._settings.hl_base_url}/info",
                json={"type": "clearinghouseState", "user": self._address},
                timeout=10,
            )
            raw = resp.json()

            cross = raw.get("crossMarginSummary", {})
            margin = raw.get("marginSummary", {})
            equity_cross = float(cross.get("accountValue", 0)) if cross else 0
            equity_margin = float(margin.get("accountValue", 0)) if margin else 0
            withdrawable = raw.get("withdrawable", "?")

            log.info(
                "health_check_perps",
                address=self._address,
                equity_cross=equity_cross,
                equity_margin=equity_margin,
                withdrawable=withdrawable,
                http_status=resp.status_code,
            )

            # 2. Check spot balance for USDC
            spot_resp = _requests.post(
                f"{self._settings.hl_base_url}/info",
                json={"type": "spotClearinghouseState", "user": self._address},
                timeout=10,
            )
            spot_raw = spot_resp.json()
            spot_usdc = 0.0
            for bal in spot_raw.get("balances", []):
                if bal.get("coin") == "USDC":
                    spot_usdc = float(bal.get("total", 0))
                    break

            log.info(
                "health_check_spot",
                address=self._address,
                spot_usdc=spot_usdc,
            )

            # 3. Report total equity (unified mode: spot USDC is usable for perps margin)
            total = equity_cross + spot_usdc
            log.info(
                "health_check_total",
                perps_equity=equity_cross,
                spot_usdc=spot_usdc,
                total_equity=total,
            )

        except Exception as e:
            log.error("health_check_failed", error=str(e))

    def _load_meta(self) -> None:
        """Load exchange metadata (asset specs, size decimals, etc.)."""
        try:
            meta = _retry(lambda: self._info.meta())
            if meta is None:
                log.error("meta_load_failed", error="API returned None after retries")
                return
            self._meta = meta
            for i, asset in enumerate(meta.get("universe", [])):
                self._asset_index[asset["name"]] = i
            log.info("meta_loaded", assets=len(self._asset_index))
        except Exception as e:
            log.error("meta_load_failed", error=str(e))

    def _get_sz_decimals(self, coin: str) -> int:
        """Get the size decimal precision for a coin."""
        for asset in self._meta.get("universe", []):
            if asset["name"] == coin:
                return asset.get("szDecimals", 2)
        return 2

    def start_public_ws(self, callbacks: dict[str, Any]) -> None:
        """Start public WebSocket for market data.

        callbacks: dict mapping channel type to callback function.
            e.g. {"trade": on_trade, "orderbook": on_orderbook, "kline": on_kline}
        """
        base_url = self._settings.hl_base_url
        coin = self._settings.coin

        self._info_ws = Info(base_url, skip_ws=False)
        self._ws_running = True

        if "trade" in callbacks:
            self._info_ws.subscribe(
                {"type": "trades", "coin": coin},
                callbacks["trade"],
            )
            log.info("ws_subscribed", channel="trades", coin=coin)

        if "orderbook" in callbacks:
            self._info_ws.subscribe(
                {"type": "l2Book", "coin": coin},
                callbacks["orderbook"],
            )
            log.info("ws_subscribed", channel="l2Book", coin=coin)

        if "kline" in callbacks:
            self._info_ws.subscribe(
                {"type": "candle", "coin": coin, "interval": "1m"},
                callbacks["kline"],
            )
            log.info("ws_subscribed", channel="candle_1m", coin=coin)

    def start_private_ws(self, callbacks: dict[str, Any]) -> None:
        """Start private WebSocket for user fills and order updates.

        Hyperliquid uses a 'userEvents' subscription keyed by address.
        """
        if self._info_ws is None:
            self._info_ws = Info(self._settings.hl_base_url, skip_ws=False)

        if "execution" in callbacks or "position" in callbacks:
            def _user_event_handler(msg):
                # Route to appropriate callback
                if "execution" in callbacks:
                    callbacks["execution"](msg)
                if "position" in callbacks:
                    callbacks["position"](msg)

            self._info_ws.subscribe(
                {"type": "userEvents", "user": self._address},
                _user_event_handler,
            )
            log.info("ws_subscribed", channel="userEvents", address=self._address)

    # --- REST API Methods ---

    def get_account_balance(self) -> dict | None:
        """Get account state including margin summary."""
        return _retry(lambda: self._info.user_state(self._address))

    def get_equity(self) -> float:
        """Get total account value in USD (perps + spot USDC for unified accounts)."""
        equity = 0.0

        # 1. Check perps clearinghouse
        state = self.get_account_balance()
        if state is not None:
            summary = state.get("crossMarginSummary") or state.get("marginSummary") or {}
            equity = float(summary.get("accountValue", 0))

        # 2. Add spot USDC balance (unified mode: spot USDC is available for perps margin)
        spot_usdc = self._get_spot_usdc()
        if spot_usdc > 0:
            equity += spot_usdc
            log.info("equity_includes_spot", perps_equity=equity - spot_usdc, spot_usdc=spot_usdc, total=equity)

        if equity == 0:
            log.warning("get_equity_zero", address=self._address)

        return equity

    def _get_spot_usdc(self) -> float:
        """Get USDC balance from spot wallet."""
        try:
            spot_state = _retry(lambda: self._info.spot_user_state(self._address))
            if spot_state is None:
                return 0.0
            for bal in spot_state.get("balances", []):
                if bal.get("coin") == "USDC":
                    return float(bal.get("total", 0))
        except Exception as e:
            log.error("spot_usdc_check_failed", error=str(e))
        return 0.0

    def get_available_margin(self) -> float:
        """Get available margin for new trades."""
        state = self.get_account_balance()
        if state is None:
            log.error("get_available_margin_failed", error="account balance returned None")
            return 0.0
        summary = state.get("crossMarginSummary") or state.get("marginSummary") or {}
        return float(summary.get("totalRawUsd", 0))

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        time_in_force: str = "Gtc",
        reduce_only: bool = False,
        order_link_id: str = "",
    ) -> dict:
        """Place an order on Hyperliquid.

        Args:
            symbol: Coin name (e.g. "SOL", not "SOLUSDT")
            side: "Buy"/"Long" or "Sell"/"Short"
            qty: Order size
            order_type: "Market" or "Limit"
            price: Limit price (required for limit orders)
            stop_loss: Not directly supported — managed by engine
            take_profit: Not directly supported — managed by engine
            time_in_force: "Gtc", "Ioc", or "Alo"
            reduce_only: Whether this is a reduce-only order
            order_link_id: Client order ID (cloid)
        """
        coin = self._settings.coin
        is_buy = side.lower() in ("long", "buy")
        sz_decimals = self._get_sz_decimals(coin)

        # Round quantity to valid precision
        qty = round(qty, sz_decimals)

        if order_type == "Market":
            # Use market_open for new positions, market_close for reduce-only
            if reduce_only:
                result = self._exchange.market_close(
                    coin=coin,
                    sz=qty,
                    slippage=0.01,  # 1% slippage tolerance
                )
            else:
                result = self._exchange.market_open(
                    coin=coin,
                    is_buy=is_buy,
                    sz=qty,
                    slippage=0.01,
                )
        else:
            # Limit order
            if price is None:
                raise ValueError("Price required for limit orders")

            order_type_spec = {"limit": {"tif": time_in_force}}
            result = self._exchange.order(
                coin=coin,
                is_buy=is_buy,
                sz=qty,
                limit_px=price,
                order_type=order_type_spec,
                reduce_only=reduce_only,
            )

        if result is None:
            log.error("order_returned_none", coin=coin, side=side, qty=qty)
            return {"raw": None, "error": "API returned None"}

        log.info(
            "order_placed",
            coin=coin,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            status=result.get("status"),
        )

        # Normalize result to include orderId for compatibility
        normalized = {"raw": result}
        if result.get("status") == "ok":
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            for s in statuses:
                if "filled" in s:
                    normalized["orderId"] = s["filled"].get("oid")
                    normalized["avgPrice"] = s["filled"].get("avgPx")
                    normalized["filledQty"] = s["filled"].get("totalSz")
                elif "resting" in s:
                    normalized["orderId"] = s["resting"].get("oid")
                elif "error" in s:
                    normalized["error"] = s["error"]
                    log.error("order_error", error=s["error"])

        return normalized

    def cancel_order(self, symbol: str, order_id: int | str) -> dict:
        """Cancel an order by order ID."""
        coin = self._settings.coin
        result = self._exchange.cancel(coin, int(order_id))
        log.info("order_cancelled", coin=coin, order_id=order_id)
        return result

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders for the coin."""
        coin = self._settings.coin
        open_orders = self.get_open_orders(symbol)
        results = []
        for order in open_orders:
            try:
                r = self._exchange.cancel(coin, order["oid"])
                results.append(r)
            except Exception as e:
                log.error("cancel_error", oid=order["oid"], error=str(e))
        log.info("all_orders_cancelled", coin=coin, count=len(results))
        return {"cancelled": len(results)}

    def get_open_orders(self, symbol: str) -> list[dict]:
        """Get all open orders."""
        result = _retry(lambda: self._info.open_orders(self._address))
        return result if result is not None else []

    def get_positions(self, symbol: str) -> list[dict]:
        """Get current positions."""
        state = _retry(lambda: self._info.user_state(self._address))
        if state is None:
            log.error("get_positions_failed", error="user_state returned None")
            return []
        positions = []
        for pos in state.get("assetPositions", []):
            p = pos.get("position", pos)
            if float(p.get("szi", 0)) != 0:
                positions.append(p)
        return positions

    def set_leverage(self, symbol: str, leverage: float) -> dict:
        """Set leverage for the coin."""
        coin = self._settings.coin
        result = self._exchange.update_leverage(
            leverage=int(leverage),
            name=coin,
            is_cross=True,
        )
        log.info("leverage_set", coin=coin, leverage=leverage)
        return result

    def get_klines(
        self,
        symbol: str,
        interval: str = "1",
        limit: int = 200,
        start: int | None = None,
        end: int | None = None,
    ) -> list[dict]:
        """Get historical kline/candlestick data.

        interval: "1m", "5m", "15m", "1h", etc.
        Note: Hyperliquid returns max 5000 candles per request.
        """
        coin = self._settings.coin

        # Map interval format: "1" -> "1m", "5" -> "5m", etc.
        if interval.isdigit():
            interval = f"{interval}m"

        start_time = start or 0
        end_time = end or 0

        raw = _retry(lambda: self._info.candles_snapshot(
            name=coin,
            interval=interval,
            startTime=start_time,
            endTime=end_time,
        ))
        if not raw:
            log.error("get_klines_failed", error="candles_snapshot returned None/empty")
            return []

        candles = []
        for item in raw[-limit:]:  # Respect limit
            candles.append({
                "timestamp": datetime.fromtimestamp(
                    int(item["t"]) / 1000, tz=timezone.utc
                ),
                "open": float(item["o"]),
                "high": float(item["h"]),
                "low": float(item["l"]),
                "close": float(item["c"]),
                "volume": float(item["v"]),
                "turnover": 0.0,  # Not provided by HL
            })

        return candles

    def get_tickers(self, symbol: str) -> dict:
        """Get current mid-price and recent data for the coin."""
        coin = self._settings.coin

        # Get all mid prices
        all_mids = _retry(lambda: self._info.all_mids())
        if all_mids is None:
            log.error("get_tickers_failed", error="all_mids returned None")
            return {"lastPrice": "0", "symbol": coin}
        mid_price = float(all_mids.get(coin, 0))

        return {
            "lastPrice": str(mid_price),
            "symbol": coin,
        }

    def get_funding_rate(self, symbol: str) -> dict:
        """Get current funding rate info."""
        coin = self._settings.coin
        state = _retry(lambda: self._info.meta_and_asset_ctxs())

        # state is [meta, [asset_ctx, ...]]
        if state and len(state) >= 2:
            meta = state[0]
            ctxs = state[1]
            for i, asset in enumerate(meta.get("universe", [])):
                if asset["name"] == coin and i < len(ctxs):
                    ctx = ctxs[i]
                    return {
                        "fundingRate": ctx.get("funding", "0"),
                        "markPrice": ctx.get("markPx", "0"),
                        "openInterest": ctx.get("openInterest", "0"),
                    }
        return {}

    def get_instrument_info(self, symbol: str) -> dict:
        """Get instrument specifications (sz_decimals, max_leverage, etc.)."""
        coin = self._settings.coin

        for asset in self._meta.get("universe", []):
            if asset["name"] == coin:
                sz_decimals = asset.get("szDecimals", 2)
                # Construct a response compatible with the execution engine
                qty_step = 10 ** (-sz_decimals)
                return {
                    "coin": coin,
                    "szDecimals": sz_decimals,
                    "maxLeverage": asset.get("maxLeverage", 50),
                    # Compatibility fields for existing code
                    "lotSizeFilter": {
                        "qtyStep": str(qty_step),
                        "minOrderQty": str(qty_step),
                    },
                    "priceFilter": {
                        "tickSize": "0.01",
                    },
                }
        return {}

    def get_user_fills(self, limit: int = 100) -> list[dict]:
        """Get recent fills/trades for the user."""
        fills = _retry(lambda: self._info.user_fills(self._address))
        if fills is None:
            log.error("get_user_fills_failed", error="user_fills returned None")
            return []
        return fills[:limit]

    def close(self) -> None:
        """Close all connections."""
        self._ws_running = False
        if self._info_ws:
            try:
                self._info_ws.ws_manager.close()
            except Exception:
                pass
        log.info("hyperliquid_client_closed")


# Alias for backward compatibility with imports
BybitClient = HyperliquidClient
