# api/main.py
"""
Trading A/B Platform — FastAPI Application Entry Point

Routers mounted:
  /api/v1/strategies   — list and inspect available trading strategies
  /api/v1/experiments  — run A/B backtests and browse past experiments
  /api/v1/predictions  — trigger ML predictions and view history

Start locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routers import experiments, predictions, strategies
from api.schemas import HealthOut
from data.database.connection import check_db_connection

load_dotenv()

# ── App metadata ──────────────────────────────────────────────────────────────

API_VERSION = "1.0.0"
API_TITLE   = "Trading A/B Platform API"
API_DESC    = """
REST API for the **Trading A/B Platform**.

### What you can do:
* **Strategies** — inspect the available trading strategy implementations
* **Experiments** — run A/B backtests comparing two strategies on a ticker,
  browse past experiment results with statistical significance metrics
* **Predictions** — ask the ML model which strategy is best for current
  market conditions, and view the history of past predictions

### Typical workflow:
1. `GET /api/v1/strategies` — pick two strategies
2. `POST /api/v1/experiments/run` — backtest them on historical data
3. `POST /api/v1/predictions/` — get today's ML recommendation
"""


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")

    db_ok = check_db_connection()
    if not db_ok:
        logger.warning(
            "Database not reachable on startup. "
            "Endpoints that need the DB will return 503 until it comes up."
        )

    yield   # ← app is live from here until shutdown

    logger.info(f"{API_TITLE} shutting down.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = API_TITLE,
    description = API_DESC,
    version     = API_VERSION,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────

_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = _CORS_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {"detail": "An unexpected server error occurred."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(
    strategies.router,
    prefix = "/api/v1/strategies",
    tags   = ["Strategies"],
)
app.include_router(
    experiments.router,
    prefix = "/api/v1/experiments",
    tags   = ["Experiments"],
)
app.include_router(
    predictions.router,
    prefix = "/api/v1/predictions",
    tags   = ["Predictions"],
)


# ── Root & health ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"], summary="API info")
def root() -> dict:
    """Return basic API metadata and links to docs."""
    return {
        "name":    API_TITLE,
        "version": API_VERSION,
        "docs":    "/docs",
        "redoc":   "/redoc",
        "health":  "/health",
    }


@app.get(
    "/health",
    response_model = HealthOut,
    tags           = ["Root"],
    summary        = "Health check",
)
def health() -> HealthOut:
    """
    Verify the API is up and the database is reachable.

    `status` is `"healthy"` when the DB is connected, `"degraded"` otherwise.
    """
    db_ok = check_db_connection()
    return HealthOut(
        status       = "healthy" if db_ok else "degraded",
        db_connected = db_ok,
        timestamp    = datetime.now(timezone.utc),
        version      = API_VERSION,
    )
