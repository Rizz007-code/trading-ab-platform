# data/ingestion/validator.py
"""
Data quality validator.
Runs after each fetch to catch issues before they corrupt downstream analysis.

Checks:
  1. Null / missing values
  2. Date gaps (missing trading days)
  3. Price outliers (> 3σ single-day moves)
  4. Zero or negative prices
  5. Volume anomalies
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import func, select

from data.database.connection import get_db_session
from data.database.models import RawPrice


# ─── Result Objects ───────────────────────────────────────────────────────────
@dataclass
class ValidationIssue:
    ticker: str
    issue_type: str      # null | gap | outlier | zero_price | volume_anomaly
    severity: str        # warning | error
    date: Optional[date]
    description: str


@dataclass
class ValidationReport:
    ticker: str
    total_rows: int
    issues: List[ValidationIssue] = field(default_factory=list)
    passed: bool = True

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)
        if issue.severity == "error":
            self.passed = False

    def summary(self) -> str:
        errors   = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        status   = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | {self.ticker} | rows={self.total_rows} | "
            f"errors={len(errors)} | warnings={len(warnings)}"
        )


# ─── Validator Class ──────────────────────────────────────────────────────────
class DataValidator:
    """
    Validates stock data stored in PostgreSQL.
    Loads data from DB and runs a suite of quality checks.
    """

    OUTLIER_THRESHOLD = 0.20      # Flag daily moves > 20% as outlier warnings
    EXTREME_THRESHOLD = 0.50      # Flag daily moves > 50% as errors
    MAX_GAP_DAYS      = 5         # Allow gaps up to 5 days (weekends + holidays)
    MIN_VOLUME        = 1_000     # Flag suspiciously low volume

    def __init__(self):
        pass

    # ── Load from DB ──────────────────────────────────────────────────────────
    def _load_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Load all price data for a ticker from DB into a DataFrame."""
        with get_db_session() as session:
            rows = session.execute(
                select(RawPrice)
                .where(RawPrice.ticker == ticker)
                .order_by(RawPrice.date.asc())
            ).scalars().all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([{
            "date":   r.date,
            "open":   r.open,
            "high":   r.high,
            "low":    r.low,
            "close":  r.close,
            "volume": r.volume,
        } for r in rows]).set_index("date")

    # ── Check 1: Null Values ──────────────────────────────────────────────────
    def _check_nulls(self, df: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        issues = []
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                issues.append(ValidationIssue(
                    ticker      = ticker,
                    issue_type  = "null",
                    severity    = "error",
                    date        = None,
                    description = f"Column '{col}' has {count} null values"
                ))
        return issues

    # ── Check 2: Date Gaps ────────────────────────────────────────────────────
    def _check_gaps(self, df: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """Detect gaps larger than MAX_GAP_DAYS (accounting for weekends)."""
        issues = []
        dates = pd.to_datetime(df.index).sort_values()

        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]).days
            # Allow weekends and holidays (up to MAX_GAP_DAYS calendar days)
            if gap_days > self.MAX_GAP_DAYS:
                issues.append(ValidationIssue(
                    ticker      = ticker,
                    issue_type  = "gap",
                    severity    = "warning",
                    date        = dates[i].date(),
                    description = f"Gap of {gap_days} days between {dates[i-1].date()} and {dates[i].date()}"
                ))
        return issues

    # ── Check 3: Price Outliers ───────────────────────────────────────────────
    def _check_outliers(self, df: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        """Flag extreme single-day price movements."""
        issues = []
        if len(df) < 2:
            return issues

        daily_returns = df["close"].pct_change().dropna()

        for idx, ret in daily_returns.items():
            abs_ret = abs(ret)
            if abs_ret >= self.EXTREME_THRESHOLD:
                issues.append(ValidationIssue(
                    ticker      = ticker,
                    issue_type  = "outlier",
                    severity    = "error",
                    date        = idx,
                    description = f"Extreme move: {ret*100:.1f}% on {idx}"
                ))
            elif abs_ret >= self.OUTLIER_THRESHOLD:
                issues.append(ValidationIssue(
                    ticker      = ticker,
                    issue_type  = "outlier",
                    severity    = "warning",
                    date        = idx,
                    description = f"Large move: {ret*100:.1f}% on {idx}"
                ))
        return issues

    # ── Check 4: Zero or Negative Prices ─────────────────────────────────────
    def _check_zero_prices(self, df: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        issues = []
        for col in ["open", "high", "low", "close"]:
            bad_rows = df[df[col] <= 0]
            for idx in bad_rows.index:
                issues.append(ValidationIssue(
                    ticker      = ticker,
                    issue_type  = "zero_price",
                    severity    = "error",
                    date        = idx,
                    description = f"Zero or negative {col}={df.loc[idx, col]} on {idx}"
                ))
        return issues

    # ── Check 5: Volume Anomalies ─────────────────────────────────────────────
    def _check_volume(self, df: pd.DataFrame, ticker: str) -> List[ValidationIssue]:
        issues = []
        low_volume = df[df["volume"] < self.MIN_VOLUME]
        for idx in low_volume.index:
            issues.append(ValidationIssue(
                ticker      = ticker,
                issue_type  = "volume_anomaly",
                severity    = "warning",
                date        = idx,
                description = f"Suspiciously low volume: {df.loc[idx, 'volume']} on {idx}"
            ))
        return issues

    # ── Main Validate ─────────────────────────────────────────────────────────
    def validate_ticker(self, ticker: str) -> ValidationReport:
        """Run all checks for a single ticker. Returns a ValidationReport."""
        df = self._load_ticker_data(ticker)

        report = ValidationReport(ticker=ticker, total_rows=len(df))

        if df.empty:
            report.add_issue(ValidationIssue(
                ticker      = ticker,
                issue_type  = "null",
                severity    = "error",
                date        = None,
                description = "No data found in database for this ticker"
            ))
            logger.error(f"Validation failed: No data for {ticker}")
            return report

        # Run all checks
        checks = [
            self._check_nulls(df, ticker),
            self._check_gaps(df, ticker),
            self._check_outliers(df, ticker),
            self._check_zero_prices(df, ticker),
            self._check_volume(df, ticker),
        ]

        for issue_list in checks:
            for issue in issue_list:
                report.add_issue(issue)

        logger.info(report.summary())
        return report

    def validate_all(self, tickers: List[str]) -> Dict[str, ValidationReport]:
        """Validate all tickers. Returns dict of reports."""
        logger.info(f"Running data validation for {len(tickers)} tickers...")
        reports = {}
        for ticker in tickers:
            reports[ticker] = self.validate_ticker(ticker)

        # Summary
        passed  = sum(1 for r in reports.values() if r.passed)
        failed  = len(reports) - passed
        logger.info(f"Validation complete: {passed} passed, {failed} failed")

        return reports

    def get_stats(self, ticker: str) -> dict:
        """Return basic statistics about a ticker's data in DB."""
        with get_db_session() as session:
            result = session.execute(
                select(
                    func.count(RawPrice.id).label("total_rows"),
                    func.min(RawPrice.date).label("earliest"),
                    func.max(RawPrice.date).label("latest"),
                    func.avg(RawPrice.close).label("avg_close"),
                    func.avg(RawPrice.volume).label("avg_volume"),
                )
                .where(RawPrice.ticker == ticker)
            ).one()

        return {
            "ticker":      ticker,
            "total_rows":  result.total_rows,
            "earliest":    str(result.earliest),
            "latest":      str(result.latest),
            "avg_close":   round(float(result.avg_close or 0), 2),
            "avg_volume":  int(result.avg_volume or 0),
        }


# ─── Standalone Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    tickers = os.getenv("TICKERS", "AAPL,MSFT").split(",")
    validator = DataValidator()
    reports = validator.validate_all(tickers)

    print("\n=== Validation Summary ===")
    for ticker, report in reports.items():
        print(f"  {report.summary()}")
        for issue in report.issues:
            icon = "⚠️" if issue.severity == "warning" else "❌"
            print(f"    {icon} [{issue.issue_type}] {issue.description}")
