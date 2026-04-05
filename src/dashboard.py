"""Real-time terminal dashboard using Rich Live display.

Provides a visual overview of the trading system:
- Price with sparkline chart
- Open positions
- Recent trades with PnL
- Strategy weights and status
- Account equity, daily PnL, system health

Run standalone: python3 -m src.dashboard
Or launched automatically by main.py in a background thread.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from src.data.feature_store import FeatureStore
    from src.data.ingestion import DataIngestionService
    from src.execution.engine import ExecutionEngine
    from src.risk.manager import RiskManager
    from src.strategies.ensemble import StrategyEnsemble


# Unicode sparkline blocks (8 levels)
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 40) -> str:
    """Create a Unicode sparkline from a list of values."""
    if not values:
        return ""
    # Take last `width` values
    vals = values[-width:]
    if len(vals) < 2:
        return SPARK_CHARS[4] * len(vals)
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return SPARK_CHARS[4] * len(vals)
    return "".join(
        SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in vals
    )


class TradingDashboard:
    """Rich-based live terminal dashboard for the trading system."""

    def __init__(self):
        self._console = Console()
        self._running = False

        # Data sources (set via set_components)
        self._execution: ExecutionEngine | None = None
        self._risk: RiskManager | None = None
        self._ensemble: StrategyEnsemble | None = None
        self._ingestion: DataIngestionService | None = None
        self._feature_store: FeatureStore | None = None

        # Dashboard state
        self._price_history: deque[float] = deque(maxlen=60)
        self._recent_trades: deque[dict] = deque(maxlen=20)
        self._last_price: float = 0.0
        self._prev_price: float = 0.0
        self._equity_history: deque[float] = deque(maxlen=60)
        self._start_time = datetime.now(timezone.utc)
        self._trade_count_today = 0
        self._win_count_today = 0
        self._loss_count_today = 0
        self._signals_today = 0
        self._symbol = "SOL/USDT"
        self._mode = "PAPER"

    def set_components(
        self,
        execution: ExecutionEngine | None = None,
        risk: RiskManager | None = None,
        ensemble: StrategyEnsemble | None = None,
        ingestion: DataIngestionService | None = None,
        feature_store: FeatureStore | None = None,
        symbol: str = "SOL/USDT",
        mode: str = "PAPER",
    ) -> None:
        self._execution = execution
        self._risk = risk
        self._ensemble = ensemble
        self._ingestion = ingestion
        self._feature_store = feature_store
        self._symbol = symbol
        self._mode = mode.upper()

    def update_price(self, price: float) -> None:
        """Call this whenever we get a new price."""
        self._prev_price = self._last_price
        self._last_price = price
        self._price_history.append(price)

    def record_trade(self, trade_data: dict) -> None:
        """Record a completed trade for display."""
        self._recent_trades.appendleft(trade_data)
        self._trade_count_today += 1
        pnl = trade_data.get("pnl_usd", 0)
        if pnl > 0:
            self._win_count_today += 1
        else:
            self._loss_count_today += 1

    def record_signal(self) -> None:
        """Record that a signal was generated."""
        self._signals_today += 1

    def update_equity(self, equity: float) -> None:
        """Update equity for the equity sparkline."""
        self._equity_history.append(equity)

    def start_background(self) -> None:
        """Start the dashboard in a background thread."""
        self._running = True
        thread = Thread(target=self._run_live, daemon=True)
        thread.start()

    def stop(self) -> None:
        self._running = False

    def _run_live(self) -> None:
        """Main render loop running in a background thread."""
        try:
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=2,
                screen=True,
            ) as live:
                while self._running:
                    live.update(self._build_layout())
                    time.sleep(0.5)
        except Exception:
            # If terminal doesn't support Live (e.g. logging mode), silently stop
            self._running = False

    def _build_layout(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(self._render_header())

        # Main area: split into left (price + positions) and right (trades + strategies)
        layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )

        layout["left"].split_column(
            Layout(name="price", size=9),
            Layout(name="positions", ratio=1),
        )

        layout["right"].split_column(
            Layout(name="trades", ratio=2),
            Layout(name="strategies", ratio=1),
        )

        layout["price"].update(self._render_price_panel())
        layout["positions"].update(self._render_positions())
        layout["trades"].update(self._render_recent_trades())
        layout["strategies"].update(self._render_strategies())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render the header bar."""
        mode = self._mode
        equity = self._risk.equity if self._risk else 0
        daily_pnl = self._risk.daily_pnl if self._risk else 0
        pnl_color = "green" if daily_pnl >= 0 else "red"
        pnl_sign = "+" if daily_pnl >= 0 else ""

        uptime = datetime.now(timezone.utc) - self._start_time
        hours = int(uptime.total_seconds() / 3600)
        minutes = int((uptime.total_seconds() % 3600) / 60)

        header = Text()
        header.append(" SOL SCALPER ", style="bold white on blue")
        header.append(f"  [{mode}]  ", style="bold yellow" if mode == "PAPER" else "bold red")
        header.append(f"  Equity: ", style="dim")
        header.append(f"${equity:,.2f}", style="bold white")
        header.append(f"  Daily: ", style="dim")
        header.append(f"{pnl_sign}${daily_pnl:,.2f}", style=f"bold {pnl_color}")
        header.append(f"  Trades: ", style="dim")
        header.append(f"{self._trade_count_today}", style="bold white")
        header.append(f" ({self._win_count_today}W/{self._loss_count_today}L)", style="dim")
        header.append(f"  Up: ", style="dim")
        header.append(f"{hours}h{minutes}m", style="white")

        return Panel(Align.center(header), style="blue")

    def _render_price_panel(self) -> Panel:
        """Render current price with sparkline chart."""
        price = self._last_price
        price_change = price - self._prev_price if self._prev_price > 0 else 0
        pct_change = (price_change / self._prev_price * 100) if self._prev_price > 0 else 0

        color = "green" if price_change >= 0 else "red"
        arrow = "▲" if price_change >= 0 else "▼"
        sign = "+" if price_change >= 0 else ""

        # Price display
        price_text = Text()
        price_text.append(f"  {self._symbol}  ", style="bold white")
        price_text.append(f"${price:,.4f}", style=f"bold {color}")
        price_text.append(f"  {arrow} {sign}{price_change:,.4f} ({sign}{pct_change:.2f}%)", style=color)

        # Sparkline
        chart = sparkline(list(self._price_history), width=min(60, self._console.width - 10))
        spark_text = Text()
        spark_text.append("  ")
        spark_text.append(chart, style=color)

        # Price range
        if self._price_history:
            lo = min(self._price_history)
            hi = max(self._price_history)
            range_text = Text()
            range_text.append(f"  L: ${lo:,.4f}  H: ${hi:,.4f}  ", style="dim")
            range_text.append(f"(last {len(self._price_history)} ticks)", style="dim italic")
        else:
            range_text = Text(f"  Waiting for {self._symbol} data...", style="dim italic")

        return Panel(
            Group(price_text, Text(""), spark_text, range_text),
            title="[bold]Price[/bold]",
            border_style="cyan",
        )

    def _render_positions(self) -> Panel:
        """Render open positions table."""
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Side", style="bold", width=6)
        table.add_column("Entry", justify="right", width=12)
        table.add_column("Current", justify="right", width=12)
        table.add_column("Qty", justify="right", width=8)
        table.add_column("PnL", justify="right", width=10)
        table.add_column("Strategy", width=12)
        table.add_column("Age", justify="right", width=8)

        positions = self._execution.get_open_positions() if self._execution else []

        if not positions:
            return Panel(
                Align.center(Text("No open positions", style="dim italic")),
                title="[bold]Positions[/bold]",
                border_style="yellow",
            )

        now = datetime.now(timezone.utc)
        for p in positions:
            side = p.get("side", "")
            side_style = "green bold" if side == "long" else "red bold"
            side_arrow = "▲ LONG" if side == "long" else "▼ SHORT"

            entry = p.get("entry_price", 0)
            current = self._last_price
            qty = p.get("quantity", 0)

            if side == "long":
                pnl_pct = (current - entry) / entry if entry > 0 else 0
            else:
                pnl_pct = (entry - current) / entry if entry > 0 else 0

            pnl_usd = qty * entry * pnl_pct
            pnl_color = "green" if pnl_usd >= 0 else "red"
            pnl_sign = "+" if pnl_usd >= 0 else ""

            entry_time = datetime.fromisoformat(p.get("entry_time", now.isoformat()))
            age = now - entry_time
            age_str = f"{int(age.total_seconds())}s"

            table.add_row(
                Text(side_arrow, style=side_style),
                f"${entry:,.4f}",
                f"${current:,.4f}",
                f"{qty:.2f}",
                Text(f"{pnl_sign}${pnl_usd:,.2f}", style=f"bold {pnl_color}"),
                p.get("strategy_name", ""),
                age_str,
            )

        return Panel(table, title="[bold]Positions[/bold]", border_style="yellow")

    def _render_recent_trades(self) -> Panel:
        """Render recent trades table."""
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Time", width=8)
        table.add_column("Side", width=6)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Exit", justify="right", width=10)
        table.add_column("PnL", justify="right", width=10)
        table.add_column("Exit", width=8)
        table.add_column("Strategy", width=10)

        if not self._recent_trades:
            return Panel(
                Align.center(Text("No trades yet", style="dim italic")),
                title="[bold]Recent Trades[/bold]",
                border_style="green",
            )

        for trade in list(self._recent_trades)[:15]:
            pnl = trade.get("pnl_usd", 0)
            pnl_color = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else ""

            side = trade.get("side", "")
            side_style = "green" if side == "long" else "red"

            # Format time
            ts = trade.get("entry_time", "")
            if isinstance(ts, str) and len(ts) > 11:
                time_str = ts[11:19]  # HH:MM:SS
            else:
                time_str = str(ts)[:8]

            table.add_row(
                time_str,
                Text(side.upper(), style=side_style),
                f"${trade.get('entry_price', 0):,.2f}",
                f"${trade.get('exit_price', 0):,.2f}",
                Text(f"{pnl_sign}${pnl:,.2f}", style=f"bold {pnl_color}"),
                trade.get("exit_reason", "")[:8],
                trade.get("strategy", "")[:10],
            )

        return Panel(table, title="[bold]Recent Trades[/bold]", border_style="green")

    def _render_strategies(self) -> Panel:
        """Render strategy status panel."""
        if not self._ensemble:
            return Panel(
                Text("Ensemble not initialized", style="dim"),
                title="[bold]Strategies[/bold]",
                border_style="magenta",
            )

        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("Strategy", width=12)
        table.add_column("Weight", justify="right", width=8)
        table.add_column("Status", width=8)
        table.add_column("", width=15)  # Weight bar

        weights = self._ensemble.get_weights()
        active = self._ensemble.get_active_strategies()

        for name, weight in weights.items():
            is_active = name in active
            status_text = Text("ACTIVE", style="green") if is_active else Text("MUTED", style="red")

            # Visual weight bar
            bar_width = 12
            filled = int(weight / 0.4 * bar_width)  # 0.4 = max weight
            bar = "█" * min(filled, bar_width) + "░" * max(bar_width - filled, 0)
            bar_color = "cyan" if is_active else "dim"

            table.add_row(
                name,
                f"{weight:.2f}",
                status_text,
                Text(bar, style=bar_color),
            )

        return Panel(table, title="[bold]Strategies[/bold]", border_style="magenta")

    def _render_footer(self) -> Panel:
        """Render the status bar."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # System stats
        ingestion_stats = self._ingestion.get_stats() if self._ingestion else {}
        tick_count = ingestion_stats.get("tick_count", 0)
        ob_ready = ingestion_stats.get("orderbook_ready", False)

        # Feature store status
        feature_ready = False
        if self._feature_store:
            feature_ready = self._feature_store.has_enough_data("15s", 30)

        exposure = self._risk.get_exposure() if self._risk else {}
        leverage = exposure.get("leverage", 0)
        dd = exposure.get("drawdown_from_peak", 0)

        footer = Text()
        footer.append(f" {now} ", style="dim")
        footer.append(" │ ", style="dim")
        footer.append("Ticks: ", style="dim")
        footer.append(f"{tick_count:,}", style="white")
        footer.append(" │ ", style="dim")
        footer.append("OB: ", style="dim")
        footer.append("✓" if ob_ready else "✗", style="green" if ob_ready else "red")
        footer.append(" │ ", style="dim")
        footer.append("Features: ", style="dim")
        footer.append("✓" if feature_ready else "…", style="green" if feature_ready else "yellow")
        footer.append(" │ ", style="dim")
        footer.append("Leverage: ", style="dim")
        footer.append(f"{leverage:.1f}x", style="white")
        footer.append(" │ ", style="dim")
        footer.append("DD: ", style="dim")
        footer.append(f"{dd:.1%}", style="red" if dd > 0.05 else "green")
        footer.append(" │ ", style="dim")
        footer.append("Signals: ", style="dim")
        footer.append(f"{self._signals_today}", style="white")

        return Panel(footer, style="dim blue")


# ============================================================
# Standalone mode — connect to Redis and display live data
# ============================================================

async def _standalone_dashboard() -> None:
    """Run dashboard standalone by subscribing to Redis events."""
    from src.config.settings import get_settings
    from src.utils.events import EventBus

    settings = get_settings()
    event_bus = EventBus(settings.redis_url)
    await event_bus.connect()

    dashboard = TradingDashboard()
    dashboard.set_components(
        symbol=settings.symbol,
        mode=settings.trading_mode,
    )
    dashboard.start_background()

    print("Dashboard started. Listening for events...")
    print("Press Ctrl+C to stop.\n")

    # Listen for price updates and trade events
    async def price_listener():
        async for _msg_id, data in event_bus.subscribe("candles:1s"):
            price = float(data.get("close", 0))
            if price > 0:
                dashboard.update_price(price)

    async def trade_listener():
        async for _msg_id, data in event_bus.subscribe("trades:exit"):
            dashboard.record_trade(data)

    async def signal_listener():
        async for _msg_id, data in event_bus.subscribe("trades:entry"):
            dashboard.record_signal()

    try:
        await asyncio.gather(
            price_listener(),
            trade_listener(),
            signal_listener(),
        )
    except KeyboardInterrupt:
        dashboard.stop()
        await event_bus.close()


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging("WARNING")  # Suppress logs in dashboard mode
    asyncio.run(_standalone_dashboard())
