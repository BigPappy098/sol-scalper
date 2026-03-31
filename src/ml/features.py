"""Batch feature engineering for ML model training."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def compute_training_features(candles_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Compute all features from a DataFrame of candles for training.

    Args:
        candles_df: DataFrame with columns [ts, open, high, low, close, volume]
        config: Feature config dict

    Returns:
        DataFrame with all features, indexed by timestamp
    """
    config = config or {}
    df = candles_df.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # --- Returns ---
    for period in config.get("return_periods", [1, 3, 5, 10, 20]):
        df[f"return_{period}"] = df["close"].pct_change(period)

    # --- Log returns ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Volatility ---
    for window in config.get("volatility_windows", [10, 30, 60, 200]):
        df[f"volatility_{window}"] = df["log_return"].rolling(window).std()

    # --- ATR ---
    atr_period = config.get("atr_period", 14)
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift(1)).abs(),
        "lc": (df["low"] - df["close"].shift(1)).abs(),
    })
    df["true_range"] = tr.max(axis=1)
    df["atr"] = df["true_range"].rolling(atr_period).mean()

    # --- RSI ---
    rsi_period = config.get("rsi_period", 14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # --- Bollinger Bands ---
    bb_period = config.get("bb_period", 20)
    bb_std = config.get("bb_std", 2.5)
    df["bb_middle"] = df["close"].rolling(bb_period).mean()
    bb_rolling_std = df["close"].rolling(bb_period).std()
    df["bb_upper"] = df["bb_middle"] + bb_std * bb_rolling_std
    df["bb_lower"] = df["bb_middle"] - bb_std * bb_rolling_std
    bb_width = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_width.replace(0, np.nan)
    df["bb_width"] = bb_width / df["bb_middle"].replace(0, np.nan)

    # --- EMAs ---
    for period in [config.get("ema_fast", 9), config.get("ema_slow", 21), config.get("ema_trend", 55)]:
        df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
    df["ema_cross"] = (df[f"ema_{config.get('ema_fast', 9)}"] > df[f"ema_{config.get('ema_slow', 21)}"]).astype(float) * 2 - 1

    # --- VWAP ---
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["atr"].replace(0, np.nan)

    # --- Volume features ---
    df["volume_sma_30"] = df["volume"].rolling(30).mean()
    df["volume_sma_200"] = df["volume"].rolling(200).mean()
    vol_std = df["volume"].rolling(200).std()
    df["volume_zscore"] = (df["volume"] - df["volume_sma_200"]) / vol_std.replace(0, np.nan)
    df["volume_ratio"] = df["volume"] / df["volume_sma_30"].replace(0, np.nan)

    # --- Time features ---
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"])
        hour = ts.dt.hour + ts.dt.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # --- Momentum features ---
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["momentum_20"] = df["close"] - df["close"].shift(20)

    # --- Price relative to range ---
    for window in [10, 30, 60]:
        rolling_high = df["high"].rolling(window).max()
        rolling_low = df["low"].rolling(window).min()
        range_width = rolling_high - rolling_low
        df[f"price_position_{window}"] = (
            (df["close"] - rolling_low) / range_width.replace(0, np.nan)
        )

    # Drop NaN rows from rolling calculations
    df = df.dropna()

    return df


def get_feature_columns(config: dict | None = None) -> list[str]:
    """Get list of feature column names for model input."""
    config = config or {}
    cols = []

    for p in config.get("return_periods", [1, 3, 5, 10, 20]):
        cols.append(f"return_{p}")

    cols.append("log_return")

    for w in config.get("volatility_windows", [10, 30, 60, 200]):
        cols.append(f"volatility_{w}")

    cols.extend(["atr", "rsi", "bb_position", "bb_width", "ema_cross"])
    cols.extend(["vwap_deviation", "volume_zscore", "volume_ratio"])
    cols.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    cols.extend(["momentum_5", "momentum_10", "momentum_20"])
    cols.extend(["price_position_10", "price_position_30", "price_position_60"])

    return cols
