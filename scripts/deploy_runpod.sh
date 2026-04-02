#!/bin/bash
# ============================================================
# SOL Scalper - RunPod Deployment Guide
# ============================================================
#
# This image is designed to run as a single container on RunPod.
#
# SETUP:
#
# 1. Push this repo to GitHub and let GitHub Actions build the image:
#    Image will be at: ghcr.io/<your-username>/sol-scalper:latest
#
# 2. On RunPod, create a new GPU Pod:
#    - Docker Image: ghcr.io/<your-username>/sol-scalper:latest
#    - GPU: Any (RTX 3090, A4000, etc.) — cheapest is fine
#    - Network Volume: Create one (20GB minimum), mount at /data
#    - Environment Variables:
#        TRADING_MODE=paper
#        HL_PRIVATE_KEY=<your wallet private key>
#        HL_WALLET_ADDRESS=<your wallet address>
#        TELEGRAM_BOT_TOKEN=<your bot token>
#        TELEGRAM_CHAT_ID=<your chat id>
#        ANTHROPIC_API_KEY=<your claude api key>
#
# 3. Start the pod. The system will automatically:
#    - Initialize PostgreSQL + TimescaleDB on first boot
#    - Start Redis
#    - Start the trading bot
#    - Begin paper trading SOL/USD on Hyperliquid testnet
#
# 4. SSH or Web Terminal into the pod to monitor:
#    $ logs            # Follow trading logs
#    $ status          # Check service status
#    $ dashboard       # Live visual dashboard
#    $ psql-scalper    # Open database shell
#    $ backtest        # Run backtest on collected data
#
# 5. To go live, stop the pod and recreate with:
#    TRADING_MODE=live
#    (same wallet keys — Hyperliquid uses mainnet API automatically)
#
# ============================================================

echo "This is a documentation script. See the comments above for deployment instructions."
echo ""
echo "To build and push the Docker image locally:"
echo ""
echo "  docker build -t ghcr.io/<your-username>/sol-scalper:latest ."
echo "  docker push ghcr.io/<your-username>/sol-scalper:latest"
echo ""
echo "Or push to GitHub and let GitHub Actions handle it automatically."
