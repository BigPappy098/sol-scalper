"""LightGBM training pipeline with GPU support."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from src.db.database import Database
from src.ml.features import compute_training_features, get_feature_columns
from src.ml.labels import generate_labels, get_label_distribution
from src.utils.logging import get_logger

log = get_logger(__name__)


class LGBMTrainer:
    """Trains and manages LightGBM models for price prediction."""

    def __init__(self, database: Database, config: dict, feature_config: dict):
        self._db = database
        self._config = config.get("lgbm", {})
        self._feature_config = feature_config
        self._model_dir = Path("/data/models")
        self._model_dir.mkdir(parents=True, exist_ok=True)

    async def train(
        self,
        candles_df: pd.DataFrame | None = None,
        lookback_days: int = 14,
    ) -> dict:
        """Train a new LightGBM model.

        Args:
            candles_df: Optional pre-loaded DataFrame. If None, loads from DB.
            lookback_days: Days of historical data to use.

        Returns:
            Dict with model path, metrics, and model ID.
        """
        # Load data if not provided
        if candles_df is None:
            from datetime import timedelta
            start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            end = datetime.now(timezone.utc)
            rows = await self._db.get_candles("5s", start, end, limit=1_000_000)
            if not rows:
                log.warning("no_training_data")
                return {"error": "no_data"}
            candles_df = pd.DataFrame(rows)

        log.info("training_data_loaded", rows=len(candles_df))

        # Compute features
        featured_df = compute_training_features(candles_df, self._feature_config)
        log.info("features_computed", rows=len(featured_df))

        # Generate labels
        labeled_df = generate_labels(featured_df, horizon=12, threshold=0.001)
        distribution = get_label_distribution(labeled_df)
        log.info("labels_generated", **distribution)

        # Filter out flat labels for binary classification (up vs down)
        binary_df = labeled_df[labeled_df["label"] != 0].copy()
        binary_df["target"] = (binary_df["label"] == 1).astype(int)

        if len(binary_df) < 100:
            log.warning("insufficient_labeled_data", count=len(binary_df))
            return {"error": "insufficient_data"}

        # Temporal train/val/test split
        n = len(binary_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        feature_cols = get_feature_columns(self._feature_config)
        # Only use columns that exist
        feature_cols = [c for c in feature_cols if c in binary_df.columns]

        X_train = binary_df.iloc[:train_end][feature_cols]
        y_train = binary_df.iloc[:train_end]["target"]
        X_val = binary_df.iloc[train_end:val_end][feature_cols]
        y_val = binary_df.iloc[train_end:val_end]["target"]
        X_test = binary_df.iloc[val_end:][feature_cols]
        y_test = binary_df.iloc[val_end:]["target"]

        log.info(
            "data_split",
            train=len(X_train),
            val=len(X_val),
            test=len(X_test),
        )

        # Create LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Training parameters
        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "num_leaves": self._config.get("num_leaves", 63),
            "max_depth": self._config.get("max_depth", 7),
            "learning_rate": self._config.get("learning_rate", 0.05),
            "verbose": -1,
            "seed": 42,
        }

        # Use GPU if available
        device = self._config.get("device", "cpu")
        if device == "gpu":
            params["device"] = "gpu"
            params["gpu_use_dp"] = False

        # Train
        callbacks = [
            lgb.early_stopping(
                self._config.get("early_stopping_rounds", 50),
                verbose=False,
            ),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=self._config.get("num_rounds", 500),
            valid_sets=[dval],
            callbacks=callbacks,
        )

        # Evaluate on test set
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_accuracy = accuracy_score(y_test, y_pred)

        log.info(
            "model_trained",
            test_auc=test_auc,
            test_accuracy=test_accuracy,
            best_iteration=model.best_iteration,
        )

        # Feature importance
        importance = dict(
            zip(feature_cols, model.feature_importance(importance_type="gain"))
        )
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        log.info("top_features", features=top_features)

        # Save model
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = str(self._model_dir / f"lgbm_{timestamp}.txt")
        model.save_model(model_path)

        # Register in DB
        val_metrics = {
            "test_auc": float(test_auc),
            "test_accuracy": float(test_accuracy),
            "best_iteration": model.best_iteration,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "label_distribution": distribution,
            "top_features": top_features,
        }

        model_id = await self._db.register_model(
            model_name="lgbm_price_predictor",
            model_version=int(timestamp.replace("_", "")),
            artifact_path=model_path,
            val_metrics=val_metrics,
            config=params,
        )

        return {
            "model_id": model_id,
            "model_path": model_path,
            "test_auc": test_auc,
            "test_accuracy": test_accuracy,
            "feature_importance": top_features,
        }

    def load_model(self, model_path: str) -> lgb.Booster:
        """Load a saved model for inference."""
        return lgb.Booster(model_file=model_path)

    def predict(self, model: lgb.Booster, features: dict) -> float:
        """Run inference on a single feature vector.

        Returns probability of upward move.
        """
        feature_cols = get_feature_columns(self._feature_config)
        values = [features.get(col, 0.0) for col in feature_cols]
        arr = np.array([values])
        return float(model.predict(arr)[0])
