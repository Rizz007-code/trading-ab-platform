# api/schemas.py
"""
Pydantic v2 request and response schemas for the Trading A/B Platform API.

Organised in three sections:
  1. Request bodies  — what callers send IN
  2. Sub-schemas     — reusable nested models
  3. Response models — what the API sends OUT
"""

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── 1. Request bodies ─────────────────────────────────────────────────────────

class RunExperimentRequest(BaseModel):
    """Run a new A/B backtest comparing two strategies."""

    strategy_a: str = Field(
        ...,
        description="Strategy class name: 'MACrossoverStrategy' | 'MARSIStrategy' | 'MACDStrategy'",
        examples=["MACrossoverStrategy"],
    )
    strategy_b: str = Field(
        ...,
        description="Strategy class name for the challenger",
        examples=["MARSIStrategy"],
    )
    ticker: str = Field(..., description="Stock ticker symbol", examples=["AAPL"])
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)", examples=["2022-01-01"])
    end_date:   str = Field(..., description="Backtest end date (YYYY-MM-DD)",   examples=["2024-01-01"])
    experiment_name:  Optional[str]   = Field(None, description="Optional human-readable name")
    initial_capital:  float           = Field(100_000.0, gt=0, description="Starting capital in USD")
    confidence_level: float           = Field(0.95, ge=0.80, le=0.99, description="Statistical confidence level")
    n_bootstrap:      int             = Field(500, ge=100, le=5000, description="Bootstrap resamples for Sharpe CI")
    save_to_db:       bool            = Field(True, description="Persist experiment and results to PostgreSQL")


class PredictionRequest(BaseModel):
    """Trigger an ML prediction for a single ticker."""

    ticker:     str           = Field(..., description="Stock ticker symbol", examples=["AAPL"])
    as_of_date: Optional[date] = Field(None, description="Feature snapshot date (defaults to today)")
    save_to_db: bool           = Field(True, description="Persist prediction to ml_predictions table")


class BatchPredictionRequest(BaseModel):
    """Trigger ML predictions for multiple tickers in one call."""

    tickers:    list[str]     = Field(..., min_length=1, max_length=20, examples=[["AAPL", "MSFT", "GOOGL"]])
    as_of_date: Optional[date] = Field(None, description="Feature snapshot date (defaults to today)")
    save_to_db: bool           = Field(True)


# ── 2. Sub-schemas ────────────────────────────────────────────────────────────

class StrategyMetricsOut(BaseModel):
    """Performance metrics for one strategy arm."""
    name:          str
    description:   str
    sharpe:        float
    annual_return: float
    volatility:    float
    max_drawdown:  float
    win_rate:      float
    total_return:  float
    num_trades:    int


class StatTestOut(BaseModel):
    """Result of a single statistical hypothesis test."""
    statistic:      float
    p_value:        float
    is_significant: bool


class BootstrapCIOut(BaseModel):
    """Bootstrap confidence interval for Sharpe ratio difference."""
    lower:         float
    upper:         float
    n_bootstrap:   int
    excludes_zero: bool


class StatisticalTestsOut(BaseModel):
    """All three statistical tests run by the A/B engine."""
    t_test:                   StatTestOut
    mann_whitney:             StatTestOut
    bootstrap_ci_sharpe_diff: BootstrapCIOut


class ExperimentResultDetail(BaseModel):
    """Aggregated statistics from the experiment_results table (ORM → schema)."""
    winner:           Optional[str]
    lift_pct:         Optional[float]
    p_value:          Optional[float]
    is_significant:   Optional[bool]
    confidence_level: float
    sharpe_a:         Optional[float]
    annual_return_a:  Optional[float]
    volatility_a:     Optional[float]
    max_drawdown_a:   Optional[float]
    win_rate_a:       Optional[float]
    sharpe_b:         Optional[float]
    annual_return_b:  Optional[float]
    volatility_b:     Optional[float]
    max_drawdown_b:   Optional[float]
    win_rate_b:       Optional[float]
    ci_lower:         Optional[float]
    ci_upper:         Optional[float]
    test_method:      Optional[str]

    model_config = {"from_attributes": True}


# ── 3. Response models ────────────────────────────────────────────────────────

class ExperimentRunResponse(BaseModel):
    """
    Full result returned immediately after POST /experiments/run.
    Mirrors the dict produced by ABTestEngine.run().
    """
    experiment_name:  str
    ticker:           str
    start_date:       str
    end_date:         str
    winner:           Optional[str]
    is_significant:   bool
    lift_pct:         Optional[float]
    confidence_level: float
    strategy_a:       StrategyMetricsOut
    strategy_b:       StrategyMetricsOut
    statistical_tests: StatisticalTestsOut


class ExperimentListItem(BaseModel):
    """Compact experiment row for GET /experiments list view."""
    id:         int
    name:       str
    strategy_a: str
    strategy_b: str
    ticker:     str
    start_date: date
    end_date:   date
    status:     str
    created_at: Optional[datetime]
    winner:     Optional[str] = None

    model_config = {"from_attributes": True}


class ExperimentDetailOut(BaseModel):
    """Full experiment record including result (if completed)."""
    id:           int
    name:         str
    strategy_a:   str
    strategy_b:   str
    ticker:       str
    start_date:   date
    end_date:     date
    status:       str
    created_at:   Optional[datetime]
    completed_at: Optional[datetime]
    result:       Optional[ExperimentResultDetail] = None

    model_config = {"from_attributes": True}


class PredictionOut(BaseModel):
    """
    ML strategy prediction returned by the predictor.
    Also used when reading back the latest saved prediction.
    """
    ticker:             str
    as_of_date:         str
    feature_date:       str
    predicted_strategy: str
    confidence:         float
    probabilities:      dict[str, float]
    run_id:             Optional[str] = None


class PredictionHistoryItem(BaseModel):
    """One row from the ml_predictions table."""
    id:                 int
    ticker:             str
    date:               date
    predicted_strategy: str
    confidence:         Optional[float]
    model_name:         Optional[str]
    market_regime:      Optional[str]

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class BatchPredictionItem(BaseModel):
    """Single entry in a batch prediction response."""
    ticker: str
    predicted_strategy: Optional[str] = None
    confidence:         Optional[float] = None
    probabilities:      Optional[dict[str, float]] = None
    error:              Optional[str] = None


class StrategyOut(BaseModel):
    """Strategy metadata returned by GET /strategies."""
    name:        str
    version:     str
    description: str
    parameters:  dict[str, Any]


class HealthOut(BaseModel):
    """API health check response."""
    status:       str
    db_connected: bool
    timestamp:    datetime
    version:      str = "1.0.0"
