# ml/predictor.py
"""
Strategy Selector Predictor — Phase 5.

Loads the registered model from MLflow and predicts which trading strategy
will perform best given the current (or a specified) market feature snapshot.

Flow:
  1. Load model + class labels from MLflow
  2. Fetch the latest feature row for the ticker from the features table
  3. Run inference → {predicted_strategy, confidence, probabilities}
  4. Persist the prediction to the ml_predictions table
  5. Return the result dict

Usage:
    from ml.predictor import StrategyPredictor

    predictor = StrategyPredictor()
    result = predictor.predict("AAPL")
    print(result["predicted_strategy"], result["confidence"])
"""

import json
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select

from data.database.connection import get_db_session
from data.database.models import Feature, MLPrediction
from ml.mlflow_utils import (
    DEFAULT_MODEL_NAME,
    get_best_run_id,
    get_run_params,
    load_model_from_run,
    load_registered_model,
    setup_mlflow,
)

# Feature columns must match trainer.FEATURE_COLS exactly
FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_hist",
    "volatility_20",
    "volume_zscore",
    "regime_encoded",
    "ma50_ma200_ratio",
]

REGIME_MAP = {"bull": 1, "sideways": 0, "bear": -1}


class StrategyPredictor:
    """
    Loads the Production model from MLflow and generates per-ticker predictions.

    Args:
        model_name: MLflow registered model name (default: DEFAULT_MODEL_NAME).
        model_stage: Registry stage to load ("Production", "Staging", etc.).
    """

    def __init__(
        self,
        model_name:  str = DEFAULT_MODEL_NAME,
        model_stage: str = "Production",
    ):
        self.model_name  = model_name
        self.model_stage = model_stage
        self._model:   Optional[Any]   = None
        self._classes: Optional[list]  = None
        self._run_id:  Optional[str]   = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-load the model and class labels on first use."""
        if self._model is not None:
            return

        setup_mlflow()

        # Try the registry first; fall back to best run
        try:
            self._model = load_registered_model(self.model_name, self.model_stage)
            self._run_id, self._classes = self._resolve_run_meta_from_registry()
        except Exception as exc:
            logger.warning(f"Registry load failed ({exc}). Falling back to best run.")
            self._run_id = get_best_run_id(metric="val_accuracy")
            if self._run_id is None:
                raise RuntimeError(
                    "No trained model found. Run StrategyTrainer.train() first."
                ) from exc
            self._model   = load_model_from_run(self._run_id)
            self._classes = self._resolve_classes_from_run(self._run_id)

        logger.info(
            f"Model ready | run_id={self._run_id} | classes={self._classes}"
        )

    def _resolve_run_meta_from_registry(self) -> tuple[str, list]:
        """
        Find the run ID that produced the Production model version,
        then extract the class labels from its logged params.
        """
        import mlflow
        client   = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(self.model_name, stages=[self.model_stage])
        if not versions:
            raise RuntimeError(f"No '{self.model_stage}' version of '{self.model_name}' found.")

        run_id  = versions[0].run_id
        classes = self._resolve_classes_from_run(run_id)
        return run_id, classes

    def _resolve_classes_from_run(self, run_id: str) -> list:
        """
        Read the 'classes' param logged during training.
        Falls back to a sensible default list if the param is missing.
        """
        try:
            params      = get_run_params(run_id)
            classes_raw = params.get("classes", "")
            if classes_raw:
                return json.loads(classes_raw)
        except Exception as exc:
            logger.warning(f"Could not read classes param from run {run_id}: {exc}")

        # Default strategy names in case metadata is missing
        return ["Strategy_A", "Strategy_B", "Strategy_C"]

    # ── Feature retrieval ─────────────────────────────────────────────────────

    def _fetch_features(self, ticker: str, as_of: date) -> Optional[pd.DataFrame]:
        """
        Retrieve the most recent feature row for ticker at or before as_of.
        Returns a single-row DataFrame shaped for model input, or None.
        """
        with get_db_session() as session:
            row = session.execute(
                select(Feature)
                .where(Feature.ticker == ticker, Feature.date <= as_of)
                .order_by(Feature.date.desc())
                .limit(1)
            ).scalar_one_or_none()

        if row is None:
            logger.warning(f"No features found for {ticker} on/before {as_of}.")
            return None

        ma50  = row.ma_50  or 1.0
        ma200 = row.ma_200 or 1.0

        raw = {
            "rsi_14":          row.rsi_14        or 50.0,
            "macd":            row.macd           or 0.0,
            "macd_hist":       row.macd_hist      or 0.0,
            "volatility_20":   row.volatility_20  or 0.0,
            "volume_zscore":   row.volume_zscore  or 0.0,
            "regime_encoded":  REGIME_MAP.get(row.market_regime or "sideways", 0),
            "ma50_ma200_ratio": ma50 / ma200 if ma200 != 0 else 1.0,
            # Keep raw values for the audit snapshot saved to DB
            "_market_regime":  row.market_regime or "sideways",
            "_feature_date":   row.date,
        }
        return raw

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_prediction(
        self,
        ticker:     str,
        pred_date:  date,
        strategy:   str,
        confidence: float,
        raw:        dict,
    ) -> None:
        """Upsert a prediction record into the ml_predictions table."""
        with get_db_session() as session:
            pred = MLPrediction(
                ticker             = ticker,
                date               = pred_date,
                predicted_strategy = strategy,
                confidence         = confidence,
                model_name         = self.model_name,
                mlflow_run_id      = self._run_id,
                rsi                = raw.get("rsi_14"),
                macd               = raw.get("macd"),
                volatility         = raw.get("volatility_20"),
                market_regime      = raw.get("_market_regime"),
                volume_zscore      = raw.get("volume_zscore"),
            )
            session.add(pred)

        logger.info(
            f"Prediction saved | {ticker} {pred_date} → {strategy} "
            f"(confidence={confidence:.4f})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        ticker:      str,
        as_of_date:  Optional[date] = None,
        save_to_db:  bool = True,
    ) -> dict:
        """
        Predict the best strategy for a ticker given current market features.

        Args:
            ticker:     Stock ticker symbol (must have data in the features table).
            as_of_date: Date to use for feature lookup (defaults to today).
            save_to_db: Whether to persist the prediction to ml_predictions.

        Returns:
            dict with:
              predicted_strategy, confidence, probabilities (per class),
              feature_date, ticker, as_of_date, run_id
        """
        self._ensure_model()

        target_date = as_of_date or date.today()
        logger.info(f"Predicting for {ticker} as of {target_date}")

        raw = self._fetch_features(ticker, target_date)
        if raw is None:
            raise ValueError(
                f"No feature data found for {ticker} on/before {target_date}. "
                "Ensure the feature engineering pipeline has been run."
            )

        feature_date = raw.pop("_feature_date")
        raw.pop("_market_regime", None)   # remove audit keys before inference

        X = pd.DataFrame([{col: raw.get(col, 0.0) for col in FEATURE_COLS}])

        # Inference
        proba_arr  = self._model.predict_proba(X)[0]     # shape (n_classes,)
        pred_idx   = int(np.argmax(proba_arr))
        strategy   = self._classes[pred_idx]
        confidence = float(proba_arr[pred_idx])

        probabilities = {
            cls: round(float(p), 4)
            for cls, p in zip(self._classes, proba_arr)
        }

        logger.info(
            f"  → {strategy} | confidence={confidence:.4f} | probs={probabilities}"
        )

        if save_to_db:
            # Rebuild raw dict for snapshot (feature_date already popped)
            raw_snapshot = {col: raw.get(col, 0.0) for col in FEATURE_COLS}
            raw_snapshot["_market_regime"] = self._fetch_market_regime(ticker, target_date)
            self._save_prediction(ticker, feature_date, strategy, confidence, raw_snapshot)

        return {
            "ticker":             ticker,
            "as_of_date":         str(target_date),
            "feature_date":       str(feature_date),
            "predicted_strategy": strategy,
            "confidence":         round(confidence, 4),
            "probabilities":      probabilities,
            "run_id":             self._run_id,
        }

    def predict_batch(
        self,
        tickers:     list,
        as_of_date:  Optional[date] = None,
        save_to_db:  bool = True,
    ) -> list:
        """
        Run predict() for each ticker. Returns a list of result dicts.
        Failed tickers are included with an "error" key instead of crashing.
        """
        self._ensure_model()   # load once, reuse for all tickers

        results = []
        for ticker in tickers:
            try:
                result = self.predict(ticker, as_of_date, save_to_db)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")
                results.append({"ticker": ticker, "error": str(e)})

        return results

    def _fetch_market_regime(self, ticker: str, as_of: date) -> str:
        """Helper to retrieve market_regime for the audit snapshot."""
        with get_db_session() as session:
            row = session.execute(
                select(Feature.market_regime)
                .where(Feature.ticker == ticker, Feature.date <= as_of)
                .order_by(Feature.date.desc())
                .limit(1)
            ).scalar_one_or_none()
        return row or "sideways"


# ─── Standalone run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json

    predictor = StrategyPredictor()

    # Single ticker
    try:
        result = predictor.predict("AAPL", save_to_db=False)
        print("\n=== Prediction (AAPL) ===")
        print(_json.dumps(result, indent=2))
    except Exception as e:
        print(f"Prediction failed (expected if DB/MLflow not running): {e}")

    # Batch
    try:
        results = predictor.predict_batch(["AAPL", "MSFT", "GOOGL"], save_to_db=False)
        print("\n=== Batch Predictions ===")
        for r in results:
            if "error" in r:
                print(f"  {r['ticker']}: ERROR — {r['error']}")
            else:
                print(
                    f"  {r['ticker']}: {r['predicted_strategy']} "
                    f"(conf={r['confidence']:.4f})"
                )
    except Exception as e:
        print(f"Batch failed: {e}")
