"""Model versioning and deployment management."""

from __future__ import annotations

from datetime import datetime, timezone

from src.db.database import Database
from src.ml.models import ModelManager
from src.utils.logging import get_logger

log = get_logger(__name__)


class ModelRegistry:
    """Manages model lifecycle: train, evaluate, promote, retire."""

    def __init__(
        self,
        database: Database,
        model_manager: ModelManager,
    ):
        self._db = database
        self._model_manager = model_manager

    async def promote_model(self, model_id: int, model_name: str) -> None:
        """Promote a model to active (deactivates the previous active model)."""
        await self._db.activate_model(model_id, model_name)
        await self._model_manager.load_active_models()
        log.info("model_promoted", model_id=model_id, model_name=model_name)

    async def get_active_models(self) -> dict:
        """Get info about currently active models."""
        lgbm = await self._db.get_active_model("lgbm_price_predictor")
        cnn = await self._db.get_active_model("cnn_price_predictor")
        return {
            "lgbm": dict(lgbm) if lgbm else None,
            "cnn": dict(cnn) if cnn else None,
        }

    async def compare_models(self, model_id_a: int, model_id_b: int) -> dict:
        """Compare two models by their validation metrics."""
        # Fetch both models from DB
        a = await self._db.fetchrow("SELECT * FROM models WHERE id = $1", model_id_a)
        b = await self._db.fetchrow("SELECT * FROM models WHERE id = $1", model_id_b)

        if not a or not b:
            return {"error": "model_not_found"}

        return {
            "model_a": {
                "id": a["id"],
                "name": a["model_name"],
                "metrics": a["val_metrics"],
            },
            "model_b": {
                "id": b["id"],
                "name": b["model_name"],
                "metrics": b["val_metrics"],
            },
        }
