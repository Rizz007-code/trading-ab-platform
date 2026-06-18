# api/routers/predictions.py
"""
/api/v1/predictions

Endpoints:
  POST /              — trigger a new ML prediction for one ticker
  POST /batch         — trigger predictions for multiple tickers
  GET  /{ticker}      — latest saved prediction from ml_predictions table
  GET  /{ticker}/history — paginated prediction history for a ticker
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    PredictionHistoryItem,
    PredictionOut,
    PredictionRequest,
)
from data.database.connection import get_db
from data.database.models import MLPrediction
from ml.predictor import StrategyPredictor

router = APIRouter()

# ── Singleton predictor ───────────────────────────────────────────────────────
# Lazy-initialised on first request so the model loads once per process.
_predictor: StrategyPredictor | None = None


def _get_predictor() -> StrategyPredictor:
    global _predictor
    if _predictor is None:
        _predictor = StrategyPredictor()
    return _predictor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _orm_to_prediction_out(row: MLPrediction) -> PredictionOut:
    """Convert a saved MLPrediction ORM row back into PredictionOut."""
    return PredictionOut(
        ticker             = row.ticker,
        as_of_date         = str(row.date),
        feature_date       = str(row.date),
        predicted_strategy = row.predicted_strategy,
        confidence         = row.confidence or 0.0,
        probabilities      = {},        # raw probabilities not stored in DB
        run_id             = row.mlflow_run_id,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model = PredictionOut,
    summary        = "Trigger an ML prediction for one ticker",
    responses      = {
        404: {"description": "No feature data found for ticker"},
        503: {"description": "MLflow model not available"},
    },
)
def predict(req: PredictionRequest) -> PredictionOut:
    """
    Load the Production ML model and predict the best strategy for a ticker.

    The model uses the latest technical feature snapshot (RSI, MACD, volatility,
    market regime, etc.) stored in the features table.
    If `save_to_db` is True the prediction is persisted to ml_predictions.
    """
    predictor = _get_predictor()
    logger.info(f"API: predict | ticker={req.ticker} as_of={req.as_of_date}")

    try:
        result = predictor.predict(
            ticker     = req.ticker,
            as_of_date = req.as_of_date,
            save_to_db = req.save_to_db,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionOut(**result)


@router.post(
    "/batch",
    response_model = list[BatchPredictionItem],
    summary        = "Trigger ML predictions for multiple tickers",
)
def predict_batch(req: BatchPredictionRequest) -> list[BatchPredictionItem]:
    """
    Run predictions for up to 20 tickers in a single call.
    Tickers that fail (missing features, etc.) return an `error` field
    instead of crashing the entire batch.
    """
    predictor = _get_predictor()
    logger.info(f"API: predict_batch | tickers={req.tickers}")

    raw_results = predictor.predict_batch(
        tickers    = req.tickers,
        as_of_date = req.as_of_date,
        save_to_db = req.save_to_db,
    )

    output = []
    for r in raw_results:
        if "error" in r:
            output.append(BatchPredictionItem(ticker=r["ticker"], error=r["error"]))
        else:
            output.append(
                BatchPredictionItem(
                    ticker             = r["ticker"],
                    predicted_strategy = r["predicted_strategy"],
                    confidence         = r["confidence"],
                    probabilities      = r["probabilities"],
                )
            )
    return output


@router.get(
    "/{ticker}",
    response_model = PredictionOut,
    summary        = "Get the latest saved prediction for a ticker",
    responses      = {404: {"description": "No prediction found for ticker"}},
)
def get_latest_prediction(
    ticker: str,
    db:     Session = Depends(get_db),
) -> PredictionOut:
    """
    Return the most recent prediction stored in the ml_predictions table.
    Does NOT trigger a new inference — use POST / for that.
    """
    row = db.execute(
        select(MLPrediction)
        .where(MLPrediction.ticker == ticker)
        .order_by(MLPrediction.date.desc(), MLPrediction.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"No prediction found for '{ticker}'. Run POST /predictions/ first.",
        )
    return _orm_to_prediction_out(row)


@router.get(
    "/{ticker}/history",
    response_model = list[PredictionHistoryItem],
    summary        = "Paginated prediction history for a ticker",
)
def get_prediction_history(
    ticker: str,
    skip:   int     = Query(0,  ge=0),
    limit:  int     = Query(20, ge=1, le=100),
    db:     Session = Depends(get_db),
) -> list[PredictionHistoryItem]:
    """
    Return historical predictions for a ticker, newest first.
    Useful for tracking how the model's recommendation changes over time.
    """
    rows = db.execute(
        select(MLPrediction)
        .where(MLPrediction.ticker == ticker)
        .order_by(MLPrediction.date.desc(), MLPrediction.created_at.desc())
        .offset(skip)
        .limit(limit)
    ).scalars().all()

    return [PredictionHistoryItem.model_validate(r) for r in rows]
