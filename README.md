# SOL Scalper

Autonomous SOL/USD perpetual futures scalping bot on Hyperliquid. Runs 24/7 on RunPod as a single Docker container.

## Deploy on RunPod

### 1. Get API Keys

| Service | Purpose | How |
|---|---|---|
| Hyperliquid wallet | Trading | Create an Ethereum wallet (MetaMask, etc.) and deposit USDC via Arbitrum |
| Hyperliquid testnet | Paper trading | Use testnet at [app.hyperliquid-testnet.xyz](https://app.hyperliquid-testnet.xyz) |
| Telegram | Trade alerts | [@BotFather](https://t.me/BotFather) |
| Anthropic | Self-improvement agent | [console.anthropic.com](https://console.anthropic.com) |

### 2. Create a RunPod GPU Pod

1. Go to [runpod.io](https://runpod.io) → **GPU Pods** → **Deploy**
2. Pick any GPU pod (RTX 3090 / A4000 or better recommended)
3. Set the **Docker Image** to:
   ```
   ghcr.io/bigpappy098/sol-scalper:latest
   ```
4. Create a **Network Volume** (20GB+) and mount it at `/data` — this persists the database, ML models, and logs across pod restarts
5. Add these **Environment Variables**:

   | Variable | Required | Description |
   |---|---|---|
   | `TRADING_MODE` | yes | `paper` or `live` |
   | `HL_PRIVATE_KEY` | yes | Wallet private key (hex, starts with `0x`) |
   | `HL_WALLET_ADDRESS` | yes | Wallet public address |
   | `TELEGRAM_BOT_TOKEN` | yes | Telegram bot token from @BotFather |
   | `TELEGRAM_CHAT_ID` | yes | Your Telegram chat ID for notifications |
   | `ANTHROPIC_API_KEY` | yes | Claude API key for self-improvement agent |
   | `SYMBOL` | no | Trading pair (default: `SOL`) |
   | `RISK_PER_TRADE` | no | Fraction of equity per trade (default: `0.01`) |
   | `MAX_LEVERAGE` | no | Max leverage (default: `5.0`) |

6. Click **Deploy** — the container starts PostgreSQL, Redis, and the trading bot automatically via supervisord

### 3. Connect & Monitor

SSH into the pod or use the RunPod web terminal. You'll see:

```
==========================================
  SOL Scalper Trading System
==========================================

  Mode:   paper
  Symbol: SOL

  Commands:
    dashboard       - Live trading dashboard (full screen)
    logs            - Follow scalper logs
    logs-err        - Follow error logs
    status          - Service status
    restart-scalper - Restart the trading bot
    stop-scalper    - Stop the trading bot
    psql-scalper    - Open database shell
    backtest        - Run backtest
    download-history - Download historical data

==========================================
```

**Key commands:**

```bash
# Live visual dashboard (price, positions, trades, equity, strategy weights)
dashboard

# Check all services are running
status

# Stream live trading logs
logs

# Restart just the bot (keeps DB/Redis running)
restart-scalper
```

## Why Hyperliquid

- **No KYC** — connect any Ethereum wallet, no identity verification
- **No geo-restrictions** — works from the US (decentralized exchange)
- **Lower fees** — 0.01% maker / 0.035% taker (cheaper than Bybit/Binance)
- **SOL/USD perp** — high liquidity, tight spreads
- **Testnet available** — full paper trading support
- **Deposit:** bridge USDC to Arbitrum, then deposit to Hyperliquid

## What's Inside

The container runs three services managed by supervisord:

- **PostgreSQL 16 + TimescaleDB** — stores candles, trades, model artifacts
- **Redis** — event bus between components, caching
- **Scalper** — the trading bot itself

### Strategies

| Strategy | Description | Trades/day |
|---|---|---|
| BB_REVERT | Bollinger Band mean reversion on 15s candles | 20-40 |
| VOL_BREAK | Volume breakout momentum on 5s detection | 10-20 |
| OB_FADE | Orderbook imbalance fade on tick-level data | High frequency |
| ML_SIGNAL | LightGBM + CNN ensemble (enable after training) | Variable |
| FUNDING_SENT | Funding rate sentiment overlay | 2-5 |

Strategies are combined via weighted ensemble. Weights auto-adjust based on rolling 24h Sharpe ratio. Underperformers get muted automatically.

### Self-Improvement

- **Every 6h** — evaluate strategy performance, adjust weights, mute losers
- **Every 24h** — full ML model retrain on last 14 days of data
- **A/B testing** — new models run in shadow mode for 48h before promotion
- **Daily LLM review** — Claude analyzes trade logs and suggests parameter changes

### Risk Management

- 1% of equity risked per trade (scales with account size)
- Max 5x leverage
- No circuit breakers — the system trades continuously, adjusting strategy weights instead of halting

## Network Volume (`/data`)

All persistent data lives on the network volume mounted at `/data`:

```
/data/
├── postgres/     # Database files
├── redis/        # Redis persistence
├── models/       # Trained ML model artifacts
└── logs/         # Application logs
    ├── scalper.log
    ├── scalper_error.log
    ├── postgres.log
    └── redis.log
```

Without `/data` mounted, nothing persists across pod restarts. The bot will warn you at startup if `/data` is missing.

## Configuration

Strategy parameters live in `config/default.yaml` inside the container. To override:

```bash
# SSH into the pod
nano /root/sol-scalper/config/default.yaml
restart-scalper
```

Or override via environment variables (takes precedence over YAML).

## Backtesting

```bash
# Download 90 days of historical data
download-history

# Run backtest
backtest
```

## ML Training

ML strategies are disabled by default. Enable after collecting live data:

```bash
# Train LightGBM model (uses GPU)
python3 -m src.ml.train_lgbm

# Train CNN model (uses GPU)
python3 -m src.ml.train_cnn
```

Then edit `config/default.yaml` and set `ml_signal.enabled: true`, then `restart-scalper`.

## Docker Image

The image is automatically built and pushed to GHCR on every commit to `master`:

```
ghcr.io/bigpappy098/sol-scalper:latest
```

Base: `nvidia/cuda:12.2.0-runtime-ubuntu22.04` with Python 3.11, PostgreSQL 16 + TimescaleDB, Redis, and all Python dependencies pre-installed.
