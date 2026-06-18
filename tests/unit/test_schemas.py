# tests/unit/test_schemas.py
"""Unit tests for api/schemas.py — Pydantic validation, no DB or network."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BootstrapCIOut,
    ExperimentDetailOut,
    ExperimentListItem,
    ExperimentResultDetail,
    ExperimentRunResponse,
    HealthOut,
    PredictionHistoryItem,
    PredictionOut,
    PredictionRequest,
    RunExperimentRequest,
    StatTestOut,
    StatisticalTestsOut,
    StrategyMetricsOut,
    StrategyOut,
)


# ── RunExperimentRequest ──────────────────────────────────────────────────────

class TestRunExperimentRequest:
    def _valid(self, **overrides):
        data = {
            "strategy_a": "MACrossoverStrategy",
            "strategy_b": "MARSIStrategy",
            "ticker":     "AAPL",
            "start_date": "2022-01-01",
            "end_date":   "2024-01-01",
        }
        data.update(overrides)
        return RunExperimentRequest(**data)

    def test_valid_minimal(self):
        r = self._valid()
        assert r.ticker == "AAPL"
        assert r.initial_capital == 100_000.0
        assert r.save_to_db is True

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            RunExperimentRequest(strategy_a="A", strategy_b="B")   # no ticker/dates

    def test_initial_capital_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._valid(initial_capital=0)
        with pytest.raises(ValidationError):
            self._valid(initial_capital=-100)

    def test_confidence_level_bounds(self):
        with pytest.raises(ValidationError):
            self._valid(confidence_level=0.79)
        with pytest.raises(ValidationError):
            self._valid(confidence_level=1.0)
        r = self._valid(confidence_level=0.95)
        assert r.confidence_level == 0.95

    def test_n_bootstrap_bounds(self):
        with pytest.raises(ValidationError):
            self._valid(n_bootstrap=99)
        with pytest.raises(ValidationError):
            self._valid(n_bootstrap=5001)
        r = self._valid(n_bootstrap=1000)
        assert r.n_bootstrap == 1000

    def test_optional_experiment_name(self):
        r = self._valid(experiment_name="My Test")
        assert r.experiment_name == "My Test"
        r2 = self._valid()
        assert r2.experiment_name is None


# ── PredictionRequest ─────────────────────────────────────────────────────────

class TestPredictionRequest:
    def test_valid_ticker_only(self):
        r = PredictionRequest(ticker="MSFT")
        assert r.ticker == "MSFT"
        assert r.as_of_date is None
        assert r.save_to_db is True

    def test_with_date(self):
        r = PredictionRequest(ticker="GOOGL", as_of_date=date(2024, 6, 1))
        assert r.as_of_date == date(2024, 6, 1)

    def test_missing_ticker_raises(self):
        with pytest.raises(ValidationError):
            PredictionRequest()


# ── BatchPredictionRequest ────────────────────────────────────────────────────

class TestBatchPredictionRequest:
    def test_valid(self):
        r = BatchPredictionRequest(tickers=["AAPL", "MSFT"])
        assert len(r.tickers) == 2

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(tickers=[])

    def test_too_many_tickers_raises(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(tickers=[f"T{i}" for i in range(21)])


# ── StrategyMetricsOut ────────────────────────────────────────────────────────

class TestStrategyMetricsOut:
    def _make(self, **overrides):
        data = {
            "name":          "Strategy_A",
            "description":   "50-day MA",
            "sharpe":        1.2,
            "annual_return": 0.15,
            "volatility":    0.18,
            "max_drawdown":  -0.12,
            "win_rate":      0.55,
            "total_return":  0.30,
            "num_trades":    42,
        }
        data.update(overrides)
        return StrategyMetricsOut(**data)

    def test_valid(self):
        m = self._make()
        assert m.name == "Strategy_A"
        assert m.num_trades == 42

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            StrategyMetricsOut(name="A", description="d")


# ── StatTestOut ───────────────────────────────────────────────────────────────

class TestStatTestOut:
    def test_valid(self):
        s = StatTestOut(statistic=2.5, p_value=0.03, is_significant=True)
        assert s.is_significant is True

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            StatTestOut(statistic=1.0, p_value=0.1)   # missing is_significant


# ── BootstrapCIOut ────────────────────────────────────────────────────────────

class TestBootstrapCIOut:
    def test_valid(self):
        ci = BootstrapCIOut(lower=0.1, upper=0.5, n_bootstrap=500, excludes_zero=True)
        assert ci.excludes_zero is True


# ── ExperimentListItem ────────────────────────────────────────────────────────

class TestExperimentListItem:
    def _make(self, **overrides):
        data = {
            "id":         1,
            "name":       "exp-1",
            "strategy_a": "MACrossoverStrategy",
            "strategy_b": "MARSIStrategy",
            "ticker":     "AAPL",
            "start_date": date(2022, 1, 1),
            "end_date":   date(2024, 1, 1),
            "status":     "completed",
            "created_at": datetime(2024, 1, 1, 12, 0),
        }
        data.update(overrides)
        return ExperimentListItem(**data)

    def test_valid(self):
        item = self._make()
        assert item.id == 1
        assert item.winner is None   # optional defaults to None

    def test_with_winner(self):
        item = self._make(winner="MACrossoverStrategy")
        assert item.winner == "MACrossoverStrategy"

    def test_from_attributes_config(self):
        assert ExperimentListItem.model_config.get("from_attributes") is True


# ── PredictionOut ─────────────────────────────────────────────────────────────

class TestPredictionOut:
    def _make(self, **overrides):
        data = {
            "ticker":             "AAPL",
            "as_of_date":         "2024-06-01",
            "feature_date":       "2024-05-31",
            "predicted_strategy": "MACrossoverStrategy",
            "confidence":         0.82,
            "probabilities":      {"MACrossoverStrategy": 0.82, "MARSIStrategy": 0.18},
        }
        data.update(overrides)
        return PredictionOut(**data)

    def test_valid(self):
        p = self._make()
        assert p.confidence == 0.82

    def test_optional_run_id(self):
        p = self._make(run_id="abc123")
        assert p.run_id == "abc123"
        p2 = self._make()
        assert p2.run_id is None


# ── PredictionHistoryItem ─────────────────────────────────────────────────────

class TestPredictionHistoryItem:
    def test_model_name_field_allowed(self):
        # Protected-namespace override — must not raise UserWarning or error
        item = PredictionHistoryItem(
            id=1,
            ticker="AAPL",
            date=date(2024, 6, 1),
            predicted_strategy="MACrossoverStrategy",
            confidence=0.80,
            model_name="xgboost-v1",
            market_regime="bull",
        )
        assert item.model_name == "xgboost-v1"

    def test_protected_namespaces_empty(self):
        assert "protected_namespaces" in PredictionHistoryItem.model_config

    def test_optional_fields_default_none(self):
        # In Pydantic v2, Optional[X] without a default must still be supplied.
        # The schema mirrors nullable DB columns — callers always pass them (even as None).
        item = PredictionHistoryItem(
            id=2,
            ticker="MSFT",
            date=date(2024, 6, 1),
            predicted_strategy="MARSIStrategy",
            confidence=None,
            model_name=None,
            market_regime=None,
        )
        assert item.confidence is None
        assert item.model_name is None
        assert item.market_regime is None


# ── BatchPredictionItem ───────────────────────────────────────────────────────

class TestBatchPredictionItem:
    def test_success_case(self):
        item = BatchPredictionItem(
            ticker="AAPL",
            predicted_strategy="MACrossoverStrategy",
            confidence=0.75,
            probabilities={"MACrossoverStrategy": 0.75, "MARSIStrategy": 0.25},
        )
        assert item.error is None

    def test_error_case(self):
        item = BatchPredictionItem(ticker="FAIL", error="No data available")
        assert item.predicted_strategy is None
        assert item.error == "No data available"


# ── StrategyOut ───────────────────────────────────────────────────────────────

class TestStrategyOut:
    def test_valid(self):
        s = StrategyOut(
            name="MACrossoverStrategy",
            version="1.0",
            description="50-day MA Crossover",
            parameters={"ma_window": 50},
        )
        assert s.parameters["ma_window"] == 50


# ── HealthOut ─────────────────────────────────────────────────────────────────

class TestHealthOut:
    def test_valid(self):
        h = HealthOut(
            status="healthy",
            db_connected=True,
            timestamp=datetime.now(),
        )
        assert h.version == "1.0.0"
        assert h.status == "healthy"

    def test_custom_version(self):
        h = HealthOut(
            status="ok",
            db_connected=False,
            timestamp=datetime.now(),
            version="2.0.0",
        )
        assert h.version == "2.0.0"
