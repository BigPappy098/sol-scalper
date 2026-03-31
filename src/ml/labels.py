"""Label generation for ML training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def generate_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """Generate labels for ML training.

    For each row, look forward `horizon` rows and determine:
    - label = 1: price went UP by >= threshold (0.1%)
    - label = -1: price went DOWN by >= threshold
    - label = 0: flat (neither threshold hit)

    Args:
        df: DataFrame with 'close' column
        horizon: Number of rows to look forward (e.g., 12 x 5s = 60s)
        threshold: Price change threshold (0.001 = 0.1%)

    Returns:
        DataFrame with added 'label' and 'max_up'/'max_down' columns
    """
    closes = df["close"].values
    n = len(closes)

    labels = np.zeros(n, dtype=np.int32)
    max_up = np.zeros(n)
    max_down = np.zeros(n)

    for i in range(n - horizon):
        future_prices = closes[i + 1 : i + 1 + horizon]
        current_price = closes[i]

        if current_price == 0:
            continue

        future_returns = (future_prices - current_price) / current_price
        max_up[i] = float(np.max(future_returns))
        max_down[i] = float(np.min(future_returns))

        # Label based on which threshold is hit first
        up_hit = np.where(future_returns >= threshold)[0]
        down_hit = np.where(future_returns <= -threshold)[0]

        if len(up_hit) > 0 and len(down_hit) > 0:
            # Both thresholds hit — label by which came first
            if up_hit[0] < down_hit[0]:
                labels[i] = 1
            else:
                labels[i] = -1
        elif len(up_hit) > 0:
            labels[i] = 1
        elif len(down_hit) > 0:
            labels[i] = -1
        # else: label stays 0 (flat)

    df = df.copy()
    df["label"] = labels
    df["max_up"] = max_up
    df["max_down"] = max_down

    # Remove last `horizon` rows (no future data)
    df = df.iloc[:-horizon]

    return df


def get_label_distribution(df: pd.DataFrame) -> dict:
    """Get distribution of labels."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    return {
        "total": total,
        "up": counts.get(1, 0),
        "down": counts.get(-1, 0),
        "flat": counts.get(0, 0),
        "up_pct": counts.get(1, 0) / total if total > 0 else 0,
        "down_pct": counts.get(-1, 0) / total if total > 0 else 0,
        "flat_pct": counts.get(0, 0) / total if total > 0 else 0,
    }
