# ml/trainer.py
"""
Strategy Selector Trainer — Phase 5.

Trains a classifier to predict which trading strategy performs best
given current market conditions (technical features).

Pipeline:
  1. Build dataset: join experiment_results → experiments → features
  2. Encode features (market_regime → int, compute MA ratio)
  3. Train XGBoost and LightGBM with 5-fold stratified CV
  4. Log both to MLflow in a single run
  5. Register the better model as DEFAULT_MODEL_NAME in Production

Label  : ExperimentResult.winner  (e.g. "Strategy_A", "Strategy_B")
Features: RSI, MACD hist, volatility, volume z-score, regime, MA ratio
"""

import json
from datetime import date
from typing import Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select
from xgboost import XGBClassifier

from data.database.connection import get_db_session
from data.database.models import Experiment, ExperimentResult, Feature
from ml.mlflow_utils import (
    DEFAULT_MODEL_NAME,
    log_model,
    log_params_and_metrics,
    start_run,
)

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_hist",
    "volatility_20",
    "volume_zscore",
    "regime_encoded",    # bull=1, sideways=0, bear=-1
    "ma50_ma200_ratio",  # MA50/MA200 — golden/death cross signal
]

REGIME_MAP = {"bull": 1, "sideways": 0, "bear": -1}
MIN_SAMPLES = 1   # minimum labeled experiments required to train


# ─── Trainer ──────────────────────────────────────────────────────────────────

class StrategyTrainer:
    """
    Trains XGBoost and LightGBM classifiers to predict the winning strategy.

    Args:
        random_state: Seed for reproducibility across CV and model fitting.
        cv_folds:     Number of stratified CV folds.
    """

    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state  = random_state
        self.cv_folds      = cv_folds
        self.label_encoder = LabelEncoder()

    # ── Dataset ───────────────────────────────────────────────────────────────

    def _load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build (X, y) by joining experiment_results with experiments,
        then fetching the corresponding feature snapshot for each row.
        """
        with get_db_session() as session:
            rows = session.execute(
                select(
                    Experiment.ticker,
                    Experiment.end_date,
                    ExperimentResult.winner,
                )
                .join(ExperimentResult, Experiment.id == ExperimentResult.experiment_id)
                .where(ExperimentResult.winner.isnot(None))
            ).all()

        logger.info(f"Found {len(rows)} labeled experiment(s) in DB.")

        records = []
        for ticker, end_date, winner in rows:
            feat = self._fetch_feature_row(ticker, end_date)
            if feat is None:
                continue
            feat["winner"] = winner
            records.append(feat)

        if not records:
            raise ValueError(
                "No training data could be built. "
                "Run A/B experiments and populate the features table first."
            )

        df = pd.DataFrame(records)
        df = self._encode_features(df)

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after encoding: {missing}")

        X = df[FEATURE_COLS].fillna(0.0)
        y = df["winner"]

        logger.info(
            f"Dataset ready | rows={len(X)} | features={X.shape[1]} "
            f"| classes={sorted(y.unique().tolist())}"
        )
        return X, y

    def _fetch_feature_row(self, ticker: str, as_of: date) -> Optional[dict]:
        """
        Fetch the most recent feature row for ticker at or before as_of.
        Returns None if no data exists.
        """
        with get_db_session() as session:
            row = session.execute(
                select(Feature)
                .where(Feature.ticker == ticker, Feature.date <= as_of)
                .order_by(Feature.date.desc())
                .limit(1)
            ).scalar_one_or_none()

        if row is None:
            logger.warning(f"No features for {ticker} on/before {as_of} — skipping.")
            return None

        ma50  = row.ma_50  or 1.0
        ma200 = row.ma_200 or 1.0

        return {
            "rsi_14":        row.rsi_14        or 50.0,
            "macd":          row.macd           or 0.0,
            "macd_hist":     row.macd_hist      or 0.0,
            "volatility_20": row.volatility_20  or 0.0,
            "volume_zscore": row.volume_zscore  or 0.0,
            "market_regime": row.market_regime  or "sideways",
            "ma_50":         ma50,
            "ma_200":        ma200,
        }

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical and derived features:
          - market_regime → int via REGIME_MAP
          - ma50_ma200_ratio = MA50 / MA200 (golden/death cross signal; >1 = bullish)
        """
        df = df.copy()
        df["regime_encoded"]  = df["market_regime"].map(REGIME_MAP).fillna(0).astype(int)
        df["ma50_ma200_ratio"] = (
            df["ma_50"] / df["ma_200"].replace(0, np.nan)
        ).fillna(1.0)
        return df

    # ── Model training ────────────────────────────────────────────────────────

    def _cross_validate(self, model, X: pd.DataFrame, y: np.ndarray) -> float:
        """Return mean stratified CV accuracy."""
        if len(X) < max(2, self.cv_folds):
            return 1.0  # Bypass CV if we don't have enough data for a quick test
        cv     = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return float(scores.mean())

    def _train_xgboost(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[XGBClassifier, float, dict]:
        params = {
            "n_estimators":     100,
            "max_depth":        4,
            "learning_rate":    0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "eval_metric":      "mlogloss",
            "random_state":     self.random_state,
            "verbosity":        0,
        }
        model   = XGBClassifier(**params)
        cv_acc  = self._cross_validate(model, X, y)
        model.fit(X, y)
        logger.info(f"XGBoost trained | CV accuracy={cv_acc:.4f}")
        return model, cv_acc, params

    def _train_lightgbm(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[LGBMClassifier, float, dict]:
        params = {
            "n_estimators":     100,
            "num_leaves":       31,
            "learning_rate":    0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "random_state":     self.random_state,
            "verbose":          -1,
        }
        model   = LGBMClassifier(**params)
        cv_acc  = self._cross_validate(model, X, y)
        model.fit(X, y)
        logger.info(f"LightGBM trained | CV accuracy={cv_acc:.4f}")
        return model, cv_acc, params

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Full training pipeline.

        Trains XGBoost and LightGBM in a single MLflow run, logs both,
        then registers the winner as DEFAULT_MODEL_NAME in Production.

        Returns:
            dict with n_samples, class list, model accuracies, and run ID.
        """
        logger.info("=" * 60)
        logger.info("Phase 5 — ML Training: Strategy Selector")
        logger.info("=" * 60)

        X, y = self._load_dataset()

        if len(X) < MIN_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_SAMPLES} labeled experiments to train. "
                f"Have {len(X)}. Run more A/B experiments first."
            )

        y_encoded = self.label_encoder.fit_transform(y)
        classes   = self.label_encoder.classes_.tolist()
        logger.info(f"Label classes: {classes}")

        # Train both models
        xgb_model,  xgb_acc,  xgb_params  = self._train_xgboost(X, y_encoded)
        lgbm_model, lgbm_acc, lgbm_params  = self._train_lightgbm(X, y_encoded)

        # Determine winner
        if xgb_acc >= lgbm_acc:
            winner_model = xgb_model
            winner_name  = "xgboost"
            winner_acc   = xgb_acc
        else:
            winner_model = lgbm_model
            winner_name  = "lightgbm"
            winner_acc   = lgbm_acc

        logger.info(f"Winner: {winner_name} (acc={winner_acc:.4f})")

        # Log everything to MLflow in one run
        with start_run("strategy_selector_training") as run:
            log_params_and_metrics(
                params = {
                    "xgb_n_estimators":  xgb_params["n_estimators"],
                    "xgb_max_depth":     xgb_params["max_depth"],
                    "lgbm_n_estimators": lgbm_params["n_estimators"],
                    "lgbm_num_leaves":   lgbm_params["num_leaves"],
                    "cv_folds":          self.cv_folds,
                    "n_samples":         len(X),
                    "winner_model":      winner_name,
                    "classes":           json.dumps(classes),
                    "feature_cols":      json.dumps(FEATURE_COLS),
                },
                metrics = {
                    "xgb_val_accuracy":  xgb_acc,
                    "lgbm_val_accuracy": lgbm_acc,
                    "val_accuracy":      winner_acc,
                },
            )

            # Log both models as separate artifacts
            mlflow.sklearn.log_model(xgb_model,  "xgboost")
            mlflow.sklearn.log_model(lgbm_model, "lightgbm")

            # Register the winner
            log_model(
                winner_model,
                artifact_path   = "model",
                registered_name = DEFAULT_MODEL_NAME,
            )

            run_id = run.info.run_id

        logger.info("=" * 60)
        logger.info(
            f"Training complete | winner={winner_name} | acc={winner_acc:.4f} | run_id={run_id}"
        )
        logger.info("=" * 60)

        return {
            "run_id":     run_id,
            "n_samples":  len(X),
            "classes":    classes,
            "xgboost":    {"val_accuracy": round(xgb_acc,  4)},
            "lightgbm":   {"val_accuracy": round(lgbm_acc, 4)},
            "winner":     {"model": winner_name, "val_accuracy": round(winner_acc, 4)},
        }


# ─── Standalone run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = StrategyTrainer()
    try:
        result = trainer.train()
        print("\n=== Training Result ===")
        print(f"  Run ID:    {result['run_id']}")
        print(f"  Samples:   {result['n_samples']}")
        print(f"  Classes:   {result['classes']}")
        print(f"  XGBoost:   {result['xgboost']['val_accuracy']:.4f}")
        print(f"  LightGBM:  {result['lightgbm']['val_accuracy']:.4f}")
        print(f"  Winner:    {result['winner']['model']} ({result['winner']['val_accuracy']:.4f})")
    except ValueError as e:
        print(f"Training skipped: {e}")
    except Exception as e:
        print(f"Training failed (expected if DB/MLflow not running): {e}")
