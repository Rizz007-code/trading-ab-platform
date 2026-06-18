# data/ingestion/fetcher.py
"""
Stock data fetcher using yfinance.
Supports full historical load and incremental daily updates.
Writes directly to PostgreSQL via SQLAlchemy.
"""

import os
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

from data.database.connection import get_db_session
from data.database.models import RawPrice

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,TSLA,NVDA").split(",")
INDEX_TICKERS = os.getenv("INDEX_TICKERS", "SPY,QQQ").split(",")
ALL_TICKERS = list(set(TICKERS + INDEX_TICKERS))
HISTORICAL_YEARS = int(os.getenv("HISTORICAL_YEARS", 3))
INCREMENTAL_DAYS = int(os.getenv("INCREMENTAL_DAYS", 5))


# ─── Fetcher Class ────────────────────────────────────────────────────────────
class StockFetcher:
    """
    Fetches OHLCV data from Yahoo Finance and persists it to PostgreSQL.

    Two modes:
    - full_load:        Pull N years of history (first run)
    - incremental_load: Pull last N days only (daily pipeline)
    """

    def __init__(self):
        self.tickers = ALL_TICKERS

    # ── Core Download ─────────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download OHLCV data from yfinance with retry logic."""
        logger.debug(f"Downloading {ticker} from {start} to {end}")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
        return df

    # ── Latest Date in DB ─────────────────────────────────────────────────────
    def _get_latest_date(self, ticker: str) -> Optional[date]:
        """Check the most recent date stored in the DB for this ticker."""
        with get_db_session() as session:
            result = session.execute(
                select(RawPrice.date)
                .where(RawPrice.ticker == ticker)
                .order_by(RawPrice.date.desc())
                .limit(1)
            ).scalar_one_or_none()
        return result

    # ── Upsert ────────────────────────────────────────────────────────────────
    def _upsert_rows(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Insert rows, skipping any ticker+date combos already in DB.
        Returns number of rows inserted.
        """
        if df.empty:
            return 0

        df = df.copy()
        df.index = pd.to_datetime(df.index).date   # Convert index to date objects

        # Flatten MultiIndex columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        inserted = 0
        with get_db_session() as session:
            for row_date, row in df.iterrows():
                # Check if already exists
                exists = session.execute(
                    select(RawPrice.id)
                    .where(RawPrice.ticker == ticker, RawPrice.date == row_date)
                ).scalar_one_or_none()

                if exists:
                    continue  # Skip — incremental load, don't overwrite

                price = RawPrice(
                    ticker    = ticker,
                    date      = row_date,
                    open      = float(row.get("open", 0)),
                    high      = float(row.get("high", 0)),
                    low       = float(row.get("low", 0)),
                    close     = float(row.get("close", 0)),
                    adj_close = float(row.get("close", 0)),  # auto_adjust=True so close = adj_close
                    volume    = int(row.get("volume", 0)),
                )
                session.add(price)
                inserted += 1

        logger.info(f"✅ {ticker}: inserted {inserted} new rows")
        return inserted

    # ── Full Load ─────────────────────────────────────────────────────────────
    def full_load(self, tickers: Optional[List[str]] = None) -> dict:
        """
        Load N years of historical data for all tickers.
        Skips dates already in DB (safe to re-run).

        Returns: summary dict with counts per ticker.
        """
        tickers = tickers or self.tickers
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=365 * HISTORICAL_YEARS)).strftime("%Y-%m-%d")

        logger.info(f"Starting FULL LOAD: {len(tickers)} tickers from {start_date} to {end_date}")
        summary = {}

        for ticker in tickers:
            try:
                df = self._download(ticker, start_date, end_date)
                count = self._upsert_rows(ticker, df)
                summary[ticker] = {"status": "ok", "rows_inserted": count}
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                summary[ticker] = {"status": "error", "error": str(e)}

        logger.info(f"Full load complete. Summary: {summary}")
        return summary

    # ── Incremental Load ──────────────────────────────────────────────────────
    def incremental_load(self, tickers: Optional[List[str]] = None) -> dict:
        """
        Pull only the latest N days for each ticker.
        Designed for daily Airflow runs.

        Returns: summary dict with counts per ticker.
        """
        tickers = tickers or self.tickers
        end_date = datetime.today().strftime("%Y-%m-%d")
        summary = {}

        logger.info(f"Starting INCREMENTAL LOAD: {len(tickers)} tickers")

        for ticker in tickers:
            try:
                latest = self._get_latest_date(ticker)

                if latest is None:
                    # No data yet — fall back to full load for this ticker
                    logger.warning(f"{ticker} has no data — running full load for this ticker")
                    start_date = (datetime.today() - timedelta(days=365 * HISTORICAL_YEARS)).strftime("%Y-%m-%d")
                else:
                    # Start from day after latest stored date
                    start_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")

                if start_date >= end_date:
                    logger.info(f"{ticker}: already up to date")
                    summary[ticker] = {"status": "up_to_date", "rows_inserted": 0}
                    continue

                df = self._download(ticker, start_date, end_date)
                count = self._upsert_rows(ticker, df)
                summary[ticker] = {"status": "ok", "rows_inserted": count}

            except Exception as e:
                logger.error(f"Incremental load failed for {ticker}: {e}")
                summary[ticker] = {"status": "error", "error": str(e)}

        logger.info(f"Incremental load complete. Summary: {summary}")
        return summary

    # ── Fetch for Strategy (in-memory, no DB) ─────────────────────────────────
    def fetch_for_strategy(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Read price data from DB (not yfinance) for strategy/backtest use.
        Returns a clean DataFrame with DatetimeIndex.
        """
        from sqlalchemy import and_
        with get_db_session() as session:
            rows = session.execute(
                select(RawPrice)
                .where(
                    and_(
                        RawPrice.ticker == ticker,
                        RawPrice.date >= start,
                        RawPrice.date <= end,
                    )
                )
                .order_by(RawPrice.date.asc())
            ).scalars().all()

        if not rows:
            logger.warning(f"No DB data for {ticker} between {start} and {end}")
            return pd.DataFrame()

        data = [
            {
                "date":      r.date,
                "open":      r.open,
                "high":      r.high,
                "low":       r.low,
                "close":     r.close,
                "volume":    r.volume,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df


# ─── Standalone Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetcher = StockFetcher()

    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "incremental"

    if mode == "full":
        summary = fetcher.full_load()
    else:
        summary = fetcher.incremental_load()

    print("\n=== Fetch Summary ===")
    for ticker, result in summary.items():
        print(f"  {ticker}: {result}")
