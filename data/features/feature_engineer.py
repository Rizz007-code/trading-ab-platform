# data/features/feature_engineer.py
"""
Feature Engineering Pipeline.
Reads raw prices from DB → computes technical indicators → writes to features table.

Indicators computed:
  - MA 50, MA 200
  - RSI (14)
  - MACD, Signal Line, Histogram
  - Volatility (20-day rolling std of returns)
  - ATR (14)
  - Volume Z-Score
  - Market Regime (bull / bear / sideways)
  - Relative Strength vs SPY
"""

import os
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import and_, select

from data.database.connection import get_db_session
from data.database.models import Feature, RawPrice

load_dotenv()

TICKERS = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,TSLA,NVDA").split(",")
INDEX_TICKERS = os.getenv("INDEX_TICKERS", "SPY,QQQ").split(",")


# ─── Feature Engineer ─────────────────────────────────────────────────────────
class FeatureEngineer:
    """
    Computes all technical features from raw price data.
    Writes results to the features table (upsert style).
    """

    def __init__(self):
        self.tickers = TICKERS
        self.spy_cache: Optional[pd.DataFrame] = None   # Cache SPY data for rel_strength

    # ── Load Raw Prices from DB ───────────────────────────────────────────────
    def _load_prices(self, ticker: str) -> pd.DataFrame:
        """Load all raw prices for a ticker from DB. Returns DataFrame sorted by date."""
        with get_db_session() as session:
            rows = session.execute(
                select(RawPrice)
                .where(RawPrice.ticker == ticker)
                .order_by(RawPrice.date.asc())
            ).scalars().all()

        if not rows:
            logger.warning(f"No raw prices found for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date":   r.date,
            "open":   r.open,
            "high":   r.high,
            "low":    r.low,
            "close":  r.close,
            "volume": r.volume,
        } for r in rows])

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    # ── MA ────────────────────────────────────────────────────────────────────
    def _compute_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ma_50"]  = df["close"].rolling(window=50,  min_periods=1).mean()
        df["ma_200"] = df["close"].rolling(window=200, min_periods=1).mean()
        return df

    # ── RSI ───────────────────────────────────────────────────────────────────
    def _compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta  = df["close"].diff()
        gain   = delta.clip(lower=0)
        loss   = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs          = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df["rsi_14"].clip(0, 100)
        return df

    # ── MACD ──────────────────────────────────────────────────────────────────
    def _compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        df["macd"]        = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]
        return df

    # ── Volatility ────────────────────────────────────────────────────────────
    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        daily_returns       = df["close"].pct_change()
        df["volatility_20"] = daily_returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)
        return df

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()

        true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.ewm(com=period - 1, min_periods=period).mean()
        return df

    # ── Volume Z-Score ────────────────────────────────────────────────────────
    def _compute_volume_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_mean            = df["volume"].rolling(window=20, min_periods=5).mean()
        vol_std             = df["volume"].rolling(window=20, min_periods=5).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
        return df

    # ── Market Regime ─────────────────────────────────────────────────────────
    def _compute_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple regime classification:
          bull     → price > MA50 > MA200
          bear     → price < MA50 < MA200
          sideways → everything else
        """
        conditions = [
            (df["close"] > df["ma_50"]) & (df["ma_50"] > df["ma_200"]),
            (df["close"] < df["ma_50"]) & (df["ma_50"] < df["ma_200"]),
        ]
        choices = ["bull", "bear"]
        df["market_regime"] = np.select(conditions, choices, default="sideways")
        return df

    # ── Relative Strength vs SPY ──────────────────────────────────────────────
    def _compute_rel_strength(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Relative strength = (ticker return over 20 days) / (SPY return over 20 days).
        Values > 1 mean the stock is outperforming SPY.
        """
        if ticker == "SPY":
            df["rel_strength"] = 1.0
            return df

        if self.spy_cache is None:
            self.spy_cache = self._load_prices("SPY")

        if self.spy_cache.empty:
            df["rel_strength"] = np.nan
            return df

        spy_returns    = self.spy_cache["close"].pct_change(20)
        ticker_returns = df["close"].pct_change(20)

        # Align on dates
        aligned = ticker_returns.to_frame("ticker").join(
            spy_returns.to_frame("spy"), how="left"
        )
        df["rel_strength"] = aligned["ticker"] / aligned["spy"].replace(0, np.nan)
        return df

    # ── Compute All Features ──────────────────────────────────────────────────
    def compute_features(self, ticker: str) -> pd.DataFrame:
        """
        Run the full feature pipeline for a ticker.
        Returns a DataFrame with all features.
        """
        df = self._load_prices(ticker)
        if df.empty:
            return pd.DataFrame()

        logger.info(f"Computing features for {ticker} ({len(df)} rows)...")

        df = self._compute_ma(df)
        df = self._compute_rsi(df)
        df = self._compute_macd(df)
        df = self._compute_volatility(df)
        df = self._compute_atr(df)
        df = self._compute_volume_zscore(df)
        df = self._compute_regime(df)
        df = self._compute_rel_strength(df, ticker)

        return df

    # ── Write to DB ───────────────────────────────────────────────────────────
    def _upsert_features(self, ticker: str, df: pd.DataFrame) -> int:
        """Write computed features to DB. Skips rows already present."""
        if df.empty:
            return 0

        inserted = 0
        with get_db_session() as session:
            for row_date, row in df.iterrows():
                row_date_val = row_date.date() if hasattr(row_date, "date") else row_date

                exists = session.execute(
                    select(Feature.id).where(
                        and_(
                            Feature.ticker == ticker,
                            Feature.date   == row_date_val,
                        )
                    )
                ).scalar_one_or_none()

                def safe(val):
                    """Convert numpy types to Python float, handle NaN."""
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return None
                    return float(val)

                if exists:
                    # Update existing row
                    session.execute(
                        select(Feature).where(Feature.id == exists)
                    )
                    feature = session.get(Feature, exists)
                    if feature:
                        feature.ma_50         = safe(row.get("ma_50"))
                        feature.ma_200        = safe(row.get("ma_200"))
                        feature.rsi_14        = safe(row.get("rsi_14"))
                        feature.macd          = safe(row.get("macd"))
                        feature.macd_signal   = safe(row.get("macd_signal"))
                        feature.macd_hist     = safe(row.get("macd_hist"))
                        feature.volatility_20 = safe(row.get("volatility_20"))
                        feature.atr_14        = safe(row.get("atr_14"))
                        feature.volume_zscore = safe(row.get("volume_zscore"))
                        feature.market_regime = str(row.get("market_regime", "sideways"))
                        feature.rel_strength  = safe(row.get("rel_strength"))
                else:
                    feature = Feature(
                        ticker        = ticker,
                        date          = row_date_val,
                        ma_50         = safe(row.get("ma_50")),
                        ma_200        = safe(row.get("ma_200")),
                        rsi_14        = safe(row.get("rsi_14")),
                        macd          = safe(row.get("macd")),
                        macd_signal   = safe(row.get("macd_signal")),
                        macd_hist     = safe(row.get("macd_hist")),
                        volatility_20 = safe(row.get("volatility_20")),
                        atr_14        = safe(row.get("atr_14")),
                        volume_zscore = safe(row.get("volume_zscore")),
                        market_regime = str(row.get("market_regime", "sideways")),
                        rel_strength  = safe(row.get("rel_strength")),
                    )
                    session.add(feature)
                    inserted += 1

        logger.info(f"✅ {ticker}: {inserted} feature rows inserted")
        return inserted

    # ── Run Full Pipeline ─────────────────────────────────────────────────────
    def run(self, tickers: Optional[List[str]] = None) -> dict:
        """
        Run feature engineering for all tickers.
        Returns summary dict.
        """
        tickers = tickers or self.tickers
        summary = {}

        for ticker in tickers:
            try:
                df    = self.compute_features(ticker)
                count = self._upsert_features(ticker, df)
                summary[ticker] = {"status": "ok", "rows_processed": len(df), "rows_inserted": count}
            except Exception as e:
                logger.error(f"Feature engineering failed for {ticker}: {e}")
                summary[ticker] = {"status": "error", "error": str(e)}

        return summary


# ─── Standalone Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    engineer = FeatureEngineer()
    summary  = engineer.run()

    print("\n=== Feature Engineering Summary ===")
    for ticker, result in summary.items():
        print(f"  {ticker}: {result}")
