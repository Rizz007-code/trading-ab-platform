# api/routers/experiments.py
"""
/api/v1/experiments

Endpoints:
  GET  /              — paginated list of all experiments
  GET  /{id}          — full experiment detail with result
  POST /run           — run a new A/B backtest and return the result
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from ab_testing.engine import ABTestEngine
from api.schemas import (
    ExperimentDetailOut,
    ExperimentListItem,
    ExperimentResultDetail,
    ExperimentRunResponse,
    RunExperimentRequest,
)
from data.database.connection import get_db
from data.database.models import Experiment, ExperimentResult
from strategies.strategy_a import MACrossoverStrategy
from strategies.strategy_b import MARSIStrategy
from strategies.strategy_c import MACDStrategy

router = APIRouter()

# ── Strategy registry (same as strategies.py, kept local to avoid circular import)
STRATEGY_REGISTRY: dict = {
    "MACrossoverStrategy": MACrossoverStrategy,
    "MARSIStrategy":       MARSIStrategy,
    "MACDStrategy":        MACDStrategy,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_detail(exp: Experiment) -> ExperimentDetailOut:
    """Convert ORM Experiment (+ optional result) into the detail schema."""
    result_schema = None
    if exp.experiment_result:
        result_schema = ExperimentResultDetail.model_validate(exp.experiment_result)

    return ExperimentDetailOut(
        id           = exp.id,
        name         = exp.name,
        strategy_a   = exp.strategy_a,
        strategy_b   = exp.strategy_b,
        ticker       = exp.ticker,
        start_date   = exp.start_date,
        end_date     = exp.end_date,
        status       = exp.status,
        created_at   = exp.created_at,
        completed_at = exp.completed_at,
        result       = result_schema,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model = list[ExperimentListItem],
    summary        = "List all A/B experiments",
)
def list_experiments(
    skip:  int     = Query(0,  ge=0,         description="Rows to skip"),
    limit: int     = Query(20, ge=1, le=100, description="Max rows to return"),
    db:    Session = Depends(get_db),
) -> list[ExperimentListItem]:
    """
    Return a paginated list of experiments ordered by most recent first.
    The `winner` field is populated from the linked experiment_result (if completed).
    """
    rows = db.execute(
        select(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(skip)
        .limit(limit)
    ).scalars().all()

    result = []
    for exp in rows:
        winner = exp.experiment_result.winner if exp.experiment_result else None
        result.append(
            ExperimentListItem(
                id         = exp.id,
                name       = exp.name,
                strategy_a = exp.strategy_a,
                strategy_b = exp.strategy_b,
                ticker     = exp.ticker,
                start_date = exp.start_date,
                end_date   = exp.end_date,
                status     = exp.status,
                created_at = exp.created_at,
                winner     = winner,
            )
        )
    return result


@router.get(
    "/{experiment_id}",
    response_model = ExperimentDetailOut,
    summary        = "Get a single experiment with full result",
    responses      = {404: {"description": "Experiment not found"}},
)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
) -> ExperimentDetailOut:
    """Return the full experiment record plus its statistical result (if completed)."""
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")
    return _build_detail(exp)


@router.post(
    "/run",
    response_model = ExperimentRunResponse,
    status_code    = 200,
    summary        = "Run a new A/B experiment",
    responses      = {
        400: {"description": "Unknown strategy name"},
        422: {"description": "Backtest failed — no data for ticker/dates"},
        500: {"description": "Unexpected engine error"},
    },
)
def run_experiment(req: RunExperimentRequest) -> ExperimentRunResponse:
    """
    Run a full A/B backtest comparing two strategies.

    1. Validates strategy names against the registry
    2. Calls ABTestEngine.run() (synchronous, runs in FastAPI's thread pool)
    3. Returns the complete result including metrics and statistical tests

    Note: this endpoint can take 5-30 seconds depending on the date range.
    """
    if req.strategy_a not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code = 400,
            detail      = {
                "error":     f"Unknown strategy_a: '{req.strategy_a}'",
                "available": list(STRATEGY_REGISTRY.keys()),
            },
        )
    if req.strategy_b not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code = 400,
            detail      = {
                "error":     f"Unknown strategy_b: '{req.strategy_b}'",
                "available": list(STRATEGY_REGISTRY.keys()),
            },
        )

    logger.info(
        f"API: run experiment | {req.strategy_a} vs {req.strategy_b} "
        f"| {req.ticker} | {req.start_date} → {req.end_date}"
    )

    engine = ABTestEngine(
        initial_capital  = req.initial_capital,
        confidence_level = req.confidence_level,
        n_bootstrap      = req.n_bootstrap,
    )

    try:
        result = engine.run(
            strategy_a      = STRATEGY_REGISTRY[req.strategy_a](),
            strategy_b      = STRATEGY_REGISTRY[req.strategy_b](),
            ticker          = req.ticker,
            start_date      = req.start_date,
            end_date        = req.end_date,
            experiment_name = req.experiment_name,
            save_to_db      = req.save_to_db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Engine error: {exc}")
        raise HTTPException(status_code=500, detail=f"Experiment failed: {exc}") from exc

    return ExperimentRunResponse(**result)
