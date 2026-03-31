"""Model loading and inference wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.db.database import Database
from src.ml.features import get_feature_columns
from src.utils.logging import get_logger

log = get_logger(__name__)


class ModelManager:
    """Manages model loading and inference for the ML_SIGNAL strategy."""

    def __init__(self, database: Database, feature_config: dict, ml_config: dict):
        self._db = database
        self._feature_config = feature_config
        self._ml_config = ml_config
        self._lgbm_model = None
        self._cnn_model = None
        self._feature_cols = get_feature_columns(feature_config)

    async def load_active_models(self) -> None:
        """Load the currently active models from the registry."""
        # Load LightGBM
        lgbm_info = await self._db.get_active_model("lgbm_price_predictor")
        if lgbm_info:
            try:
                import lightgbm as lgb
                self._lgbm_model = lgb.Booster(
                    model_file=lgbm_info["artifact_path"]
                )
                log.info(
                    "lgbm_model_loaded",
                    path=lgbm_info["artifact_path"],
                    metrics=lgbm_info.get("val_metrics", {}),
                )
            except Exception as e:
                log.error("lgbm_load_failed", error=str(e))

        # Load CNN
        cnn_info = await self._db.get_active_model("cnn_price_predictor")
        if cnn_info:
            try:
                import torch
                from src.ml.train_cnn import PriceCNN

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(
                    cnn_info["artifact_path"], map_location=device
                )
                config = checkpoint.get("config", {})
                model = PriceCNN(
                    input_length=checkpoint.get("input_length", 120),
                    channels=config.get("channels", [32, 64, 64]),
                    kernel_sizes=config.get("kernel_sizes", [5, 5, 3]),
                ).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self._cnn_model = model
                log.info("cnn_model_loaded", path=cnn_info["artifact_path"])
            except Exception as e:
                log.error("cnn_load_failed", error=str(e))

    def predict_lgbm(self, features: dict) -> float | None:
        """Get LightGBM prediction (probability of upward move)."""
        if self._lgbm_model is None:
            return None

        values = [features.get(col, 0.0) for col in self._feature_cols]
        arr = np.array([values])

        try:
            return float(self._lgbm_model.predict(arr)[0])
        except Exception as e:
            log.error("lgbm_predict_error", error=str(e))
            return None

    def predict_cnn(self, candle_sequence: np.ndarray) -> float | None:
        """Get CNN prediction from a sequence of candles.

        Args:
            candle_sequence: numpy array of shape (5, 120) — 5 channels, 120 timesteps
        """
        if self._cnn_model is None:
            return None

        try:
            import torch
            device = next(self._cnn_model.parameters()).device
            x = torch.FloatTensor(candle_sequence).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = self._cnn_model(x)
            return float(pred.cpu().item())
        except Exception as e:
            log.error("cnn_predict_error", error=str(e))
            return None

    def predict_ensemble(self, features: dict, candle_sequence: np.ndarray | None = None) -> float | None:
        """Get ensemble prediction combining LightGBM and CNN.

        Returns probability of upward move (0 to 1).
        """
        predictions = []
        weights = []

        lgbm_pred = self.predict_lgbm(features)
        if lgbm_pred is not None:
            predictions.append(lgbm_pred)
            weights.append(0.6)  # LightGBM gets more weight

        if candle_sequence is not None:
            cnn_pred = self.predict_cnn(candle_sequence)
            if cnn_pred is not None:
                predictions.append(cnn_pred)
                weights.append(0.4)

        if not predictions:
            return None

        # Weighted average
        total_weight = sum(weights[: len(predictions)])
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        return weighted_sum / total_weight

    @property
    def has_lgbm(self) -> bool:
        return self._lgbm_model is not None

    @property
    def has_cnn(self) -> bool:
        return self._cnn_model is not None

    @property
    def is_ready(self) -> bool:
        return self.has_lgbm  # At minimum need LightGBM
