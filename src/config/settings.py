"""Application settings loaded from env vars and YAML config files."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict:
    """Load and merge YAML config files (default + mode overlay)."""
    config_dir = Path(__file__).parent.parent.parent / "config"

    # Load default config
    default_path = config_dir / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)

    # Overlay mode-specific config
    mode = os.getenv("TRADING_MODE", "paper")
    overlay_path = config_dir / f"{mode}.yaml"
    if overlay_path.exists():
        with open(overlay_path) as f:
            overlay = yaml.safe_load(f) or {}
        config = _deep_merge(config, overlay)

    return config


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Settings(BaseSettings):
    """Main application settings."""

    # Trading mode
    trading_mode: str = Field(default="paper", alias="TRADING_MODE")

    # Hyperliquid
    hl_private_key: str = Field(default="", alias="HL_PRIVATE_KEY")
    hl_wallet_address: str = Field(default="", alias="HL_WALLET_ADDRESS")

    # Database (localhost — all services run in same container)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres@localhost:5432/scalper",
        alias="DATABASE_URL",
    )

    # Redis (localhost — same container)
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # LLM
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Trading params
    symbol: str = Field(default="SOL", alias="SYMBOL")
    risk_per_trade: float = Field(default=0.01, alias="RISK_PER_TRADE")
    max_leverage: float = Field(default=5.0, alias="MAX_LEVERAGE")

    # YAML config (loaded separately)
    yaml_config: dict = Field(default_factory=dict)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.yaml_config = _load_yaml_config()

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def coin(self) -> str:
        """Get the coin name for Hyperliquid (e.g. 'SOL')."""
        # Strip USDT/USD/PERP suffixes if present
        s = self.symbol.upper()
        for suffix in ("USDT", "USD", "PERP", "-PERP"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s

    @property
    def hl_base_url(self) -> str:
        """Hyperliquid API base URL for order execution."""
        if self.is_paper:
            return "https://api.hyperliquid-testnet.xyz"
        return "https://api.hyperliquid.xyz"

    @property
    def hl_data_url(self) -> str:
        """Hyperliquid API base URL for market data — always mainnet.

        Testnet returns broken spot_meta that crashes the SDK, and paper
        trading should use real market prices anyway.
        """
        return "https://api.hyperliquid.xyz"

    def get_strategy_config(self, strategy_name: str) -> dict:
        """Get strategy-specific config from YAML."""
        return self.yaml_config.get("strategies", {}).get(strategy_name, {})

    def get_ml_config(self) -> dict:
        """Get ML config from YAML."""
        return self.yaml_config.get("ml", {})

    def get_feature_config(self) -> dict:
        """Get feature engineering config from YAML."""
        return self.yaml_config.get("features", {})

    def get_data_config(self) -> dict:
        """Get data pipeline config from YAML."""
        return self.yaml_config.get("data", {})

    def get_self_improve_config(self) -> dict:
        """Get self-improvement config from YAML."""
        return self.yaml_config.get("self_improve", {})


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
