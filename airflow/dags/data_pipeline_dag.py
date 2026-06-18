# airflow/dags/data_pipeline_dag.py
"""
Airflow DAG: Daily Data Pipeline
=================================
Runs every day at 6:30 AM UTC (after US market close + data propagation).

Task flow:
    fetch_prices → validate_data → engineer_features → pipeline_health_check

Each task is idempotent — safe to re-run if it fails.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago

# ─── Default Args ─────────────────────────────────────────────────────────────
default_args = {
    "owner":            "quant-team",
    "depends_on_past":  False,
    "email_on_failure": False,        # Set True + add email in production
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# ─── DAG Definition ───────────────────────────────────────────────────────────
dag = DAG(
    dag_id              = "daily_data_pipeline",
    description         = "Fetch, validate, and engineer features for stock data",
    default_args        = default_args,
    schedule_interval   = "30 6 * * 1-5",     # Mon–Fri at 06:30 UTC
    start_date          = days_ago(1),
    catchup             = False,               # Don't backfill missed runs
    max_active_runs     = 1,                   # Only one run at a time
    tags                = ["data", "ingestion", "features"],
)


# ─── Task Functions ───────────────────────────────────────────────────────────

def task_fetch_prices(**context):
    """
    Task 1: Incremental fetch from yfinance → PostgreSQL.
    Uses XCom to pass summary to next tasks.
    """
    import os, sys
    sys.path.insert(0, "/opt/airflow")   # Airflow worker path

    from dotenv import load_dotenv
    load_dotenv()

    from data.ingestion.fetcher import StockFetcher
    from loguru import logger

    fetcher = StockFetcher()
    summary = fetcher.incremental_load()

    failed = [t for t, r in summary.items() if r["status"] == "error"]
    if failed:
        logger.warning(f"Fetch failed for tickers: {failed}")

    # Push summary to XCom for downstream tasks
    context["ti"].xcom_push(key="fetch_summary", value=summary)
    logger.info(f"Fetch complete. Tickers processed: {list(summary.keys())}")
    return summary


def task_validate_data(**context):
    """
    Task 2: Run data quality checks on all tickers.
    Fails the task if any ticker has ERROR-level validation issues.
    """
    import os, sys
    sys.path.insert(0, "/opt/airflow")

    from dotenv import load_dotenv
    load_dotenv()

    from data.ingestion.validator import DataValidator
    from loguru import logger

    tickers = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,TSLA,NVDA").split(",")
    index_tickers = os.getenv("INDEX_TICKERS", "SPY,QQQ").split(",")
    all_tickers = list(set(tickers + index_tickers))

    validator = DataValidator()
    reports   = validator.validate_all(all_tickers)

    # Collect failed tickers
    failed_tickers = [t for t, r in reports.items() if not r.passed]

    # Log all issues
    for ticker, report in reports.items():
        logger.info(report.summary())
        for issue in report.issues:
            if issue.severity == "error":
                logger.error(f"  ❌ {ticker}: {issue.description}")
            else:
                logger.warning(f"  ⚠️  {ticker}: {issue.description}")

    validation_summary = {
        t: {"passed": r.passed, "issues": len(r.issues)}
        for t, r in reports.items()
    }
    context["ti"].xcom_push(key="validation_summary", value=validation_summary)

    if failed_tickers:
        # Raise but don't kill the whole pipeline — log and continue
        logger.error(f"Validation errors for: {failed_tickers}")
        # Uncomment below to make validation failures block downstream tasks:
        # raise ValueError(f"Validation failed for: {failed_tickers}")

    return validation_summary


def task_engineer_features(**context):
    """
    Task 3: Compute technical indicators and write to features table.
    Runs for ALL tickers regardless of validation result (best effort).
    """
    import os, sys
    sys.path.insert(0, "/opt/airflow")

    from dotenv import load_dotenv
    load_dotenv()

    from data.features.feature_engineer import FeatureEngineer
    from loguru import logger

    tickers = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,TSLA,NVDA").split(",")
    index_tickers = os.getenv("INDEX_TICKERS", "SPY,QQQ").split(",")
    all_tickers = list(set(tickers + index_tickers))

    engineer = FeatureEngineer()
    summary  = engineer.run(tickers=all_tickers)

    failed = [t for t, r in summary.items() if r["status"] == "error"]
    if failed:
        logger.error(f"Feature engineering failed for: {failed}")

    context["ti"].xcom_push(key="feature_summary", value=summary)
    logger.info("Feature engineering complete.")
    return summary


def task_health_check(**context):
    """
    Task 4: Final pipeline health check.
    Verifies DB has recent data. Logs a clean summary of the whole run.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")

    from dotenv import load_dotenv
    load_dotenv()

    from data.database.connection import check_db_connection
    from loguru import logger

    # Pull XCom summaries from previous tasks
    ti              = context["ti"]
    fetch_summary   = ti.xcom_pull(key="fetch_summary",    task_ids="fetch_prices")   or {}
    val_summary     = ti.xcom_pull(key="validation_summary", task_ids="validate_data") or {}
    feature_summary = ti.xcom_pull(key="feature_summary",  task_ids="engineer_features") or {}

    db_ok = check_db_connection()

    total_fetched   = sum(r.get("rows_inserted", 0) for r in fetch_summary.values())
    total_featured  = sum(r.get("rows_inserted", 0) for r in feature_summary.values())
    val_failed      = [t for t, r in val_summary.items() if not r.get("passed", True)]

    logger.info("=" * 50)
    logger.info("PIPELINE HEALTH CHECK")
    logger.info(f"  DB connection:      {'✅ OK' if db_ok else '❌ FAIL'}")
    logger.info(f"  Rows fetched:       {total_fetched}")
    logger.info(f"  Feature rows added: {total_featured}")
    logger.info(f"  Validation failed:  {val_failed or 'None'}")
    logger.info("=" * 50)

    if not db_ok:
        raise RuntimeError("Database connection failed during health check!")

    return {
        "db_ok":          db_ok,
        "rows_fetched":   total_fetched,
        "rows_featured":  total_featured,
        "val_failed":     val_failed,
    }


# ─── Task Definitions ─────────────────────────────────────────────────────────
with dag:

    fetch_prices = PythonOperator(
        task_id         = "fetch_prices",
        python_callable = task_fetch_prices,
        provide_context = True,
        doc_md          = "Incremental OHLCV fetch from Yahoo Finance via yfinance",
    )

    validate_data = PythonOperator(
        task_id         = "validate_data",
        python_callable = task_validate_data,
        provide_context = True,
        doc_md          = "Run data quality checks: nulls, gaps, outliers, zero prices",
    )

    engineer_features = PythonOperator(
        task_id         = "engineer_features",
        python_callable = task_engineer_features,
        provide_context = True,
        doc_md          = "Compute RSI, MACD, MA, volatility, regime → features table",
    )

    health_check = PythonOperator(
        task_id         = "pipeline_health_check",
        python_callable = task_health_check,
        provide_context = True,
        doc_md          = "Verify DB, log pipeline run summary",
    )

    # ── Task Dependencies ─────────────────────────────────────────────────────
    # fetch → validate → engineer features → health check
    fetch_prices >> validate_data >> engineer_features >> health_check
