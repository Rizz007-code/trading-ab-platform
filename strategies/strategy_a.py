# strategies/strategy_a.py
"""
Strategy A: 50-Day Moving Average Crossover
============================================
Logic:
  BUY  → price crosses ABOVE the 50-day MA
  SELL → price crosses BELOW the 50-day MA
  HOLD → no crossover event

This is the simplest trend-following strategy.
Used as the baseline in all A/B experiments.
"""

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal, StrategyInfo


class MACrossoverStrategy(BaseStrategy):
    """
    50-day Simple Moving Average Crossover Strategy.

    Parameters:
        ma_window (int): Moving average window. Default 50.
    """

    def __init__(self, ma_window: int = 50):
        self.ma_window = ma_window

    @property
    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name        = "Strategy_A",
            version     = "1.0",
            description = f"{self.ma_window}-day MA Crossover",
            parameters  = {"ma_window": self.ma_window},
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Compute MA if not already present (feature_engineer may have done it)
        if f"ma_{self.ma_window}" not in df.columns:
            df[f"ma_{self.ma_window}"] = (
                df["close"].rolling(window=self.ma_window, min_periods=1).mean()
            )
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crossover detection:
          - Yesterday: close was BELOW MA → Today: close is ABOVE MA = BUY
          - Yesterday: close was ABOVE MA → Today: close is BELOW MA = SELL
          - No crossover = HOLD
        """
        ma_col = f"ma_{self.ma_window}"

        # Boolean: is price above MA?
        above_ma      = df["close"] > df[ma_col]
        prev_above_ma = above_ma.shift(1).fillna(False)

        conditions = [
            (~prev_above_ma) & above_ma,   # Was below, now above → BUY
            (prev_above_ma)  & ~above_ma,  # Was above, now below → SELL
        ]
        choices = [Signal.BUY, Signal.SELL]

        df["signal"]   = np.select(conditions, choices, default=Signal.HOLD)
        df["position"] = np.where(above_ma, 1, 0)   # 1 = in market, 0 = flat

        # Log summary
        counts = df["signal"].value_counts().to_dict()
        logger.debug(f"[{self.info.name}] Signals: {counts}")

        return df
