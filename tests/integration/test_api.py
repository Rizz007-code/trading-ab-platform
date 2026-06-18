# tests/integration/test_api.py
"""
Integration tests for the FastAPI application.

Requires a live PostgreSQL database (set DATABASE_URL env var).
Run via CI integration-tests job or locally with:
    DATABASE_URL=postgresql://trading_user:trading_pass@localhost:5432/trading_db \
    pytest tests/integration/ -v
"""

import os
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from api.main import app
from data.database.connection import get_db
from data.database.models import Base, Experiment, ExperimentResult

# ── Database setup ─────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://trading_user:trading_pass@localhost:5432/trading_db",
)

_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
_TestingSessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def _create_tables():
    Base.metadata.create_all(bind=_engine)


def _drop_test_rows():
    """Remove experiment rows inserted by tests (keep real data intact)."""
    with _engine.connect() as conn:
        conn.execute(text("DELETE FROM experiment_results WHERE experiment_id IN "
                          "(SELECT id FROM experiments WHERE name LIKE 'test-%')"))
        conn.execute(text("DELETE FROM experiments WHERE name LIKE 'test-%'"))
        conn.commit()


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    _create_tables()
    yield
    _drop_test_rows()


@pytest.fixture()
def db():
    session = _TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db):
    """TestClient with DB dependency overridden to use the test session."""
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ── Seed data helper ──────────────────────────────────────────────────────────

def _seed_experiment(db, name: str = "test-exp-1") -> Experiment:
    exp = Experiment(
        name       = name,
        strategy_a = "MACrossoverStrategy",
        strategy_b = "MARSIStrategy",
        ticker     = "AAPL",
        start_date = date(2022, 1, 1),
        end_date   = date(2023, 1, 1),
        status     = "completed",
        created_at = datetime.now(timezone.utc),
    )
    db.add(exp)
    db.flush()   # get exp.id without committing

    result = ExperimentResult(
        experiment_id    = exp.id,
        winner           = "MACrossoverStrategy",
        lift_pct         = 5.2,
        p_value          = 0.03,
        is_significant   = True,
        confidence_level = 0.95,
        sharpe_a         = 1.2,
        sharpe_b         = 0.9,
        annual_return_a  = 0.15,
        annual_return_b  = 0.10,
        volatility_a     = 0.18,
        volatility_b     = 0.20,
        max_drawdown_a   = -0.12,
        max_drawdown_b   = -0.18,
        win_rate_a       = 0.55,
        win_rate_b       = 0.50,
        ci_lower         = 0.05,
        ci_upper         = 0.30,
        test_method      = "welch_t_test",
    )
    db.add(result)
    db.commit()
    db.refresh(exp)
    return exp


# ── Root & health ─────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        data = resp = client.get("/")
        assert "docs" in resp.json()

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "db_connected" in data
        assert "timestamp" in data
        assert "version" in data

    def test_health_db_connected_when_db_up(self, client):
        data = client.get("/health").json()
        # DB is reachable in integration test environment
        assert data["db_connected"] is True
        assert data["status"] == "healthy"


# ── Strategies endpoints ───────────────────────────────────────────────────────

class TestStrategiesEndpoints:
    def test_list_strategies_200(self, client):
        resp = client.get("/api/v1/strategies/")
        assert resp.status_code == 200

    def test_list_strategies_returns_three(self, client):
        data = client.get("/api/v1/strategies/").json()
        assert len(data) == 3

    def test_strategy_names_present(self, client):
        data = client.get("/api/v1/strategies/").json()
        names = {s["name"] for s in data}
        assert "Strategy_A" in names
        assert "Strategy_B" in names
        assert "Strategy_C" in names

    def test_get_single_strategy_200(self, client):
        resp = client.get("/api/v1/strategies/MACrossoverStrategy")
        assert resp.status_code == 200

    def test_get_single_strategy_schema(self, client):
        data = client.get("/api/v1/strategies/MACrossoverStrategy").json()
        for field in ("name", "version", "description", "parameters"):
            assert field in data

    def test_get_strategy_ma_crossover_parameters(self, client):
        data = client.get("/api/v1/strategies/MACrossoverStrategy").json()
        assert "ma_window" in data["parameters"]

    def test_get_unknown_strategy_404(self, client):
        resp = client.get("/api/v1/strategies/UnknownStrategy")
        assert resp.status_code == 404

    def test_get_macd_strategy(self, client):
        data = client.get("/api/v1/strategies/MACDStrategy").json()
        assert "fast_period" in data["parameters"]
        assert "slow_period" in data["parameters"]


# ── Experiments list & detail ─────────────────────────────────────────────────

class TestExperimentsListDetail:
    def test_list_experiments_200(self, client):
        resp = client.get("/api/v1/experiments/")
        assert resp.status_code == 200

    def test_list_experiments_returns_list(self, client):
        data = client.get("/api/v1/experiments/").json()
        assert isinstance(data, list)

    def test_list_experiments_pagination(self, client, db):
        _seed_experiment(db, name="test-pag-1")
        _seed_experiment(db, name="test-pag-2")
        resp = client.get("/api/v1/experiments/?skip=0&limit=1")
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_list_experiments_invalid_limit_422(self, client):
        resp = client.get("/api/v1/experiments/?limit=0")
        assert resp.status_code == 422

    def test_list_experiments_items_have_required_fields(self, client, db):
        _seed_experiment(db, name="test-list-fields")
        data = client.get("/api/v1/experiments/?limit=1").json()
        if data:
            item = data[0]
            for field in ("id", "name", "strategy_a", "strategy_b", "ticker", "status"):
                assert field in item

    def test_get_experiment_detail_200(self, client, db):
        exp = _seed_experiment(db, name="test-detail-1")
        resp = client.get(f"/api/v1/experiments/{exp.id}")
        assert resp.status_code == 200

    def test_get_experiment_detail_schema(self, client, db):
        exp = _seed_experiment(db, name="test-detail-schema")
        data = client.get(f"/api/v1/experiments/{exp.id}").json()
        for field in ("id", "name", "ticker", "status", "result"):
            assert field in data

    def test_get_experiment_detail_with_result(self, client, db):
        exp = _seed_experiment(db, name="test-detail-result")
        data = client.get(f"/api/v1/experiments/{exp.id}").json()
        assert data["result"] is not None
        assert data["result"]["winner"] == "MACrossoverStrategy"

    def test_get_nonexistent_experiment_404(self, client):
        resp = client.get("/api/v1/experiments/999999")
        assert resp.status_code == 404


# ── Experiments run (validation path — no live data fetch) ────────────────────

class TestExperimentsRunValidation:
    def test_unknown_strategy_a_returns_400(self, client):
        payload = {
            "strategy_a": "NonExistentStrategy",
            "strategy_b": "MARSIStrategy",
            "ticker":     "AAPL",
            "start_date": "2022-01-01",
            "end_date":   "2023-01-01",
        }
        resp = client.post("/api/v1/experiments/run", json=payload)
        assert resp.status_code == 400

    def test_unknown_strategy_b_returns_400(self, client):
        payload = {
            "strategy_a": "MACrossoverStrategy",
            "strategy_b": "Ghost",
            "ticker":     "AAPL",
            "start_date": "2022-01-01",
            "end_date":   "2023-01-01",
        }
        resp = client.post("/api/v1/experiments/run", json=payload)
        assert resp.status_code == 400

    def test_missing_required_field_returns_422(self, client):
        resp = client.post("/api/v1/experiments/run", json={"strategy_a": "MACrossoverStrategy"})
        assert resp.status_code == 422

    def test_invalid_capital_returns_422(self, client):
        payload = {
            "strategy_a":      "MACrossoverStrategy",
            "strategy_b":      "MARSIStrategy",
            "ticker":          "AAPL",
            "start_date":      "2022-01-01",
            "end_date":        "2023-01-01",
            "initial_capital": -100,
        }
        resp = client.post("/api/v1/experiments/run", json=payload)
        assert resp.status_code == 422

    def test_invalid_confidence_level_returns_422(self, client):
        payload = {
            "strategy_a":      "MACrossoverStrategy",
            "strategy_b":      "MARSIStrategy",
            "ticker":          "AAPL",
            "start_date":      "2022-01-01",
            "end_date":        "2023-01-01",
            "confidence_level": 0.5,
        }
        resp = client.post("/api/v1/experiments/run", json=payload)
        assert resp.status_code == 422


# ── Experiments run (engine mocked — tests happy path response shape) ─────────

class TestExperimentsRunMocked:
    _FAKE_RESULT = {
        "experiment_name":  "mocked-exp",
        "ticker":           "AAPL",
        "start_date":       "2022-01-01",
        "end_date":         "2023-01-01",
        "winner":           "MACrossoverStrategy",
        "is_significant":   True,
        "lift_pct":         4.5,
        "confidence_level": 0.95,
        "strategy_a": {
            "name": "Strategy_A", "description": "50-day MA",
            "sharpe": 1.2, "annual_return": 0.15, "volatility": 0.18,
            "max_drawdown": -0.12, "win_rate": 0.55, "total_return": 0.30, "num_trades": 42,
        },
        "strategy_b": {
            "name": "Strategy_B", "description": "MA + RSI",
            "sharpe": 0.9, "annual_return": 0.10, "volatility": 0.20,
            "max_drawdown": -0.18, "win_rate": 0.50, "total_return": 0.20, "num_trades": 30,
        },
        "statistical_tests": {
            "t_test":         {"statistic": 2.5,  "p_value": 0.03, "is_significant": True},
            "mann_whitney":   {"statistic": 1800, "p_value": 0.04, "is_significant": True},
            "bootstrap_ci_sharpe_diff": {
                "lower": 0.05, "upper": 0.30, "n_bootstrap": 500, "excludes_zero": True,
            },
        },
    }

    def test_run_returns_200_with_valid_payload(self, client):
        with patch("api.routers.experiments.ABTestEngine") as mock_engine_cls:
            mock_engine_cls.return_value.run.return_value = self._FAKE_RESULT
            payload = {
                "strategy_a": "MACrossoverStrategy",
                "strategy_b": "MARSIStrategy",
                "ticker":     "AAPL",
                "start_date": "2022-01-01",
                "end_date":   "2023-01-01",
                "save_to_db": False,
            }
            resp = client.post("/api/v1/experiments/run", json=payload)
        assert resp.status_code == 200

    def test_run_response_has_expected_fields(self, client):
        with patch("api.routers.experiments.ABTestEngine") as mock_engine_cls:
            mock_engine_cls.return_value.run.return_value = self._FAKE_RESULT
            payload = {
                "strategy_a": "MACrossoverStrategy",
                "strategy_b": "MARSIStrategy",
                "ticker":     "AAPL",
                "start_date": "2022-01-01",
                "end_date":   "2023-01-01",
                "save_to_db": False,
            }
            data = client.post("/api/v1/experiments/run", json=payload).json()
        for field in ("winner", "is_significant", "strategy_a", "strategy_b", "statistical_tests"):
            assert field in data

    def test_run_value_error_returns_422(self, client):
        with patch("api.routers.experiments.ABTestEngine") as mock_engine_cls:
            mock_engine_cls.return_value.run.side_effect = ValueError("no data")
            payload = {
                "strategy_a": "MACrossoverStrategy",
                "strategy_b": "MARSIStrategy",
                "ticker":     "INVALID_TICKER_ZZZZZ",
                "start_date": "2022-01-01",
                "end_date":   "2023-01-01",
                "save_to_db": False,
            }
            resp = client.post("/api/v1/experiments/run", json=payload)
        assert resp.status_code == 422


# ── Predictions endpoints ─────────────────────────────────────────────────────

class TestPredictionsEndpoints:
    def test_get_ticker_no_prediction_returns_404_or_200(self, client):
        resp = client.get("/api/v1/predictions/NOSUCHTICKERXYZ")
        # Either 404 (no prediction) or 200 with empty is acceptable
        assert resp.status_code in (200, 404)

    def test_get_ticker_history_returns_200(self, client):
        resp = client.get("/api/v1/predictions/AAPL/history?limit=5")
        assert resp.status_code == 200

    def test_get_ticker_history_returns_list(self, client):
        data = client.get("/api/v1/predictions/AAPL/history?limit=5").json()
        assert isinstance(data, list)

    def test_batch_prediction_missing_tickers_422(self, client):
        with patch("api.routers.predictions._get_predictor") as _:
            resp = client.post("/api/v1/predictions/batch", json={})
        assert resp.status_code == 422

    def test_batch_prediction_empty_list_422(self, client):
        resp = client.post("/api/v1/predictions/batch", json={"tickers": []})
        assert resp.status_code == 422

    def test_single_prediction_missing_ticker_422(self, client):
        resp = client.post("/api/v1/predictions/", json={})
        assert resp.status_code == 422
