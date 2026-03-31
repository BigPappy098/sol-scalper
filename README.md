# sol-scalper

A Solana (SOL/USDT) perpetual futures scalping bot for Bybit. Combines multiple rule-based strategies with machine learning and a Claude-powered self-improvement loop. Designed to run on RunPod (GPU cloud) via Docker.

## What it does

- Trades SOL/USDT perpetual futures on Bybit using an ensemble of strategies
- Ingests live orderbook and candle data across multiple timeframes (1s, 5s, 15s, 1m, 5m)
- Manages risk per-trade (default: 1% equity, max 5x leverage)
- Sends trade notifications via Telegram
- Retrains ML models and runs A/B tests automatically every 6–24 hours
- Displays a live terminal dashboard (rich)

---

## Quick Start

**Recommended: start in paper trading mode before going live.**

### 1. Get API keys

| Service | Purpose | Where to get it |
|---|---|---|
| Bybit (testnet) | Paper trading | testnet.bybit.com |
| Bybit (live) | Live trading | bybit.com |
| Telegram Bot | Trade notifications | @BotFather on Telegram |
| Anthropic | Self-improvement agent | console.anthropic.com |

### 2. Build the Docker image

```bash
git clone <repo-url> sol-scalper
cd sol-scalper
docker build -t sol-scalper .
```

### 3. Run in paper mode

```bash
docker run -it \
  -e TRADING_MODE=paper \
  -e BYBIT_API_KEY=your_testnet_key \
  -e BYBIT_API_SECRET=your_testnet_secret \
  -e TELEGRAM_BOT_TOKEN=your_bot_token \
  -e TELEGRAM_CHAT_ID=your_chat_id \
  -e ANTHROPIC_API_KEY=your_anthropic_key \
  -v /path/to/local/data:/data \
  sol-scalper
```

The container starts PostgreSQL, Redis, and the bot via supervisord. On first run it initializes the database automatically.

---

## Running on RunPod

1. Create a **Network Volume** (recommended: 20GB+) and mount it at `/data` — this persists the database, models, and logs across pod restarts.
2. Set all environment variables in the pod's **Environment Variables** section (no `.env` file needed).
3. Use the Docker image directly or build it in the pod.

Without a `/data` mount, data will not survive pod restarts. The bot will warn you at startup.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `TRADING_MODE` | yes | `paper` | `paper` or `live` |
| `BYBIT_API_KEY` | yes | — | Bybit API key |
| `BYBIT_API_SECRET` | yes | — | Bybit API secret |
| `TELEGRAM_BOT_TOKEN` | yes | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | yes | — | Telegram chat ID for notifications |
| `ANTHROPIC_API_KEY` | yes | — | Claude API key for self-improvement agent |
| `SYMBOL` | no | `SOLUSDT` | Trading pair |
| `RISK_PER_TRADE` | no | `0.01` | Fraction of equity risked per trade |
| `MAX_LEVERAGE` | no | `5.0` | Maximum leverage |
| `DATABASE_URL` | no | `postgresql+asyncpg://postgres@localhost:5432/scalper` | PostgreSQL connection string |
| `REDIS_URL` | no | `redis://localhost:6379` | Redis connection string |

---

## Configuration

Strategy parameters and system settings live in `config/`:

| File | Purpose |
|---|---|
| `config/default.yaml` | All defaults — edit this to tune strategies |
| `config/paper.yaml` | Paper mode overrides (DEBUG logging) |
| `config/live.yaml` | Live mode overrides |

Environment variables take precedence over YAML config.

Key settings in `default.yaml`:

```yaml
trading:
  risk_per_trade: 0.01   # 1% equity per trade
  max_leverage: 5.0

strategies:
  bb_revert:
    enabled: true        # Bollinger Band mean reversion
  vol_break:
    enabled: true        # Volatility breakout
  ob_fade:
    enabled: true        # Order book imbalance fade
  ml_signal:
    enabled: false       # Enable after Phase 2 (ML training)
  funding_sent:
    enabled: false       # Enable after Phase 3

ensemble:
  min_signal_confidence: 0.55
  min_combined_weight: 0.5
```

---

## Running Without Docker

Requires: Python 3.11, PostgreSQL 16 + TimescaleDB, Redis.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database URLs

# Start PostgreSQL and Redis manually, then:
python -m src.main
```

---

## Backtesting

The backtest engine replays historical candles through the same strategy logic used in live trading.

**Step 1 — Download historical data** (if you haven't collected live data yet):

```bash
python scripts/download_historical.py --days 90 --interval 1
```

**Step 2 — Run the backtest:**

```bash
python backtest/run_backtest.py --timeframe 15s --days 7 --equity 1000
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--timeframe` | `15s` | Candle timeframe to backtest on |
| `--days` | `7` | How many days of history to replay |
| `--equity` | `1000.0` | Starting equity in USD |

Results are printed to the terminal (per-strategy breakdown) and saved to a CSV: `backtest_results_15s_7d.csv`.

---

## ML Training

ML strategies are disabled by default (Phase 2). Enable them after collecting sufficient live data and training models.

```bash
# Train LightGBM model
python -m src.ml.train_lgbm

# Train CNN model (requires GPU)
python -m src.ml.train_cnn
```

Then set `ml_signal.enabled: true` in `config/default.yaml` (or your override config).

The self-improvement scheduler handles incremental retraining automatically once running (every 6h by default, full retrain every 24h).

---

## Monitoring & Logs

**Inside the container:**

```bash
# Check all process status
supervisorctl status

# Stream bot logs
supervisorctl tail -f scalper

# Restart just the bot (keeps DB and Redis running)
supervisorctl restart scalper
```

**Log files** (in `/data/logs/`):

| File | Contents |
|---|---|
| `scalper.log` | Bot output (trades, signals, errors) |
| `postgres.log` | PostgreSQL stdout |
| `redis.log` | Redis stdout |
| `supervisord.log` | Process manager events |

**Terminal dashboard** — the bot renders a live Rich dashboard showing price, open positions, recent trades, equity, and active strategies.

**Prometheus metrics** — exposed for external scraping if you connect a Prometheus instance.

---

## Trading Phases

The bot is designed to be rolled out in phases:

| Phase | Strategies active | How to enable |
|---|---|---|
| 1 (default) | `bb_revert`, `vol_break`, `ob_fade` | Active out of the box |
| 2 | + `ml_signal` | Train ML models, set `ml_signal.enabled: true` |
| 3 | + `funding_sent` | Set `funding_sent.enabled: true` |

---

## Project Structure

```
sol-scalper/
├── backtest/              # Backtesting engine and CLI
├── config/                # YAML configuration files
├── data/                  # Model artifacts and local data
├── scripts/               # Utility scripts
│   └── download_historical.py
├── src/
│   ├── main.py            # Entry point / system orchestrator
│   ├── dashboard.py       # Rich terminal dashboard
│   ├── config/            # Pydantic settings
│   ├── data/              # Candle builder, feature store, orderbook, ingestion
│   ├── db/                # AsyncPG + SQLAlchemy database layer
│   ├── execution/         # Bybit client and order execution engine
│   ├── ml/                # Feature engineering, LGBM and CNN model training
│   ├── notifications/     # Telegram bot
│   ├── risk/              # Risk manager and position sizer
│   ├── self_improve/      # A/B testing, LLM evaluator, retraining scheduler
│   ├── strategies/        # Individual strategy implementations
│   └── utils/             # Redis event bus, structlog logging
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile             # CUDA 12.2 + Python 3.11 + PostgreSQL 16 + Redis
├── entrypoint.sh          # DB init + supervisord startup
├── supervisord.conf       # Process management config
└── .env.example           # Environment variable template
```

---

## Testing

```bash
pytest tests/
```

Integration tests require a live PostgreSQL and Redis connection. Unit tests run standalone.

---

## Risk Warning

This software trades real money when `TRADING_MODE=live`. Always:

- Test thoroughly in paper mode (`TRADING_MODE=paper`) before going live
- Use testnet Bybit keys for paper trading
- Understand the strategies and risk parameters before enabling live trading
- Start with a small equity allocation
