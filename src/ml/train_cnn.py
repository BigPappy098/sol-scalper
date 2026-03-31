"""1D-CNN training pipeline for temporal pattern recognition."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.db.database import Database
from src.ml.labels import generate_labels
from src.utils.logging import get_logger

log = get_logger(__name__)


class PriceCNN(nn.Module):
    """1D-CNN for price movement prediction from raw candle data."""

    def __init__(
        self,
        input_length: int = 120,
        input_channels: int = 5,  # OHLCV
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
    ):
        super().__init__()
        channels = channels or [32, 64, 64]
        kernel_sizes = kernel_sizes or [5, 5, 3]

        layers = []
        in_ch = input_channels
        for out_ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class CNNTrainer:
    """Trains 1D-CNN models for price prediction."""

    def __init__(self, database: Database, config: dict):
        self._db = database
        self._config = config.get("cnn", {})
        self._model_dir = Path("/data/models")
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def train(self, candles_df: pd.DataFrame | None = None) -> dict:
        """Train a new CNN model."""
        if candles_df is None:
            from datetime import timedelta
            start = datetime.now(timezone.utc) - timedelta(days=14)
            end = datetime.now(timezone.utc)
            rows = await self._db.get_candles("1s", start, end, limit=2_000_000)
            if not rows:
                return {"error": "no_data"}
            candles_df = pd.DataFrame(rows)

        # Generate labels
        labeled_df = generate_labels(candles_df, horizon=60, threshold=0.001)
        binary_df = labeled_df[labeled_df["label"] != 0].copy()
        binary_df["target"] = (binary_df["label"] == 1).astype(int)

        if len(binary_df) < 500:
            return {"error": "insufficient_data"}

        # Create sequences
        input_length = self._config.get("input_length", 120)
        X, y = self._create_sequences(binary_df, input_length)

        if len(X) < 200:
            return {"error": "insufficient_sequences"}

        # Temporal split
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # Create data loaders
        batch_size = self._config.get("batch_size", 256)
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
            ),
            batch_size=batch_size,
        )

        # Build model
        model = PriceCNN(
            input_length=input_length,
            channels=self._config.get("channels", [32, 64, 64]),
            kernel_sizes=self._config.get("kernel_sizes", [5, 5, 3]),
        ).to(self._device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._config.get("learning_rate", 0.001),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._config.get("epochs", 50)
        )
        criterion = nn.BCELoss()

        # Train
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        epochs = self._config.get("epochs", 50)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self._device)
                    y_batch = y_batch.to(self._device)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log.info("early_stopping", epoch=epoch)
                break

            if epoch % 10 == 0:
                log.info(
                    "cnn_epoch",
                    epoch=epoch,
                    train_loss=train_loss / len(train_loader),
                    val_loss=val_loss,
                )

        # Load best model and evaluate on test set
        model.load_state_dict(best_state)
        model.eval()

        X_test_t = torch.FloatTensor(X_test).to(self._device)
        with torch.no_grad():
            test_pred = model(X_test_t).cpu().numpy()

        test_accuracy = float(np.mean((test_pred > 0.5).astype(int) == y_test))

        from sklearn.metrics import roc_auc_score
        test_auc = float(roc_auc_score(y_test, test_pred))

        log.info("cnn_trained", test_auc=test_auc, test_accuracy=test_accuracy)

        # Save model
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = str(self._model_dir / f"cnn_{timestamp}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": self._config,
            "input_length": input_length,
        }, model_path)

        # Register in DB
        model_id = await self._db.register_model(
            model_name="cnn_price_predictor",
            model_version=int(timestamp.replace("_", "")),
            artifact_path=model_path,
            val_metrics={
                "test_auc": test_auc,
                "test_accuracy": test_accuracy,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
            config=self._config,
        )

        return {
            "model_id": model_id,
            "model_path": model_path,
            "test_auc": test_auc,
            "test_accuracy": test_accuracy,
        }

    def _create_sequences(
        self, df: pd.DataFrame, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create input sequences from candle data.

        Each sequence is [open, high, low, close, volume] normalized.
        """
        ohlcv = df[["open", "high", "low", "close", "volume"]].values
        targets = df["target"].values

        # Normalize each sequence independently
        sequences = []
        labels = []

        for i in range(length, len(ohlcv)):
            seq = ohlcv[i - length : i].copy()

            # Normalize prices relative to last close
            last_close = seq[-1, 3]
            if last_close > 0:
                seq[:, :4] = seq[:, :4] / last_close - 1  # Price cols
                vol_mean = seq[:, 4].mean()
                if vol_mean > 0:
                    seq[:, 4] = seq[:, 4] / vol_mean  # Volume col

            sequences.append(seq.T)  # Transpose to (channels, length)
            labels.append(targets[i])

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

    def load_model(self, model_path: str) -> PriceCNN:
        """Load a saved CNN model."""
        checkpoint = torch.load(model_path, map_location=self._device)
        model = PriceCNN(
            input_length=checkpoint.get("input_length", 120),
            channels=checkpoint.get("config", {}).get("channels", [32, 64, 64]),
            kernel_sizes=checkpoint.get("config", {}).get("kernel_sizes", [5, 5, 3]),
        ).to(self._device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
