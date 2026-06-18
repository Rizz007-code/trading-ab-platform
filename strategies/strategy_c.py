# strategies/strategy_c.py
"""
Strategy C: MACD-Based Strategy
================================
Uses the MACD line and Signal line crossover for trade signals.

Logic:
  BUY  → MACD line crosses ABOVE the Signal line (bullish momentum)
  SELL → MACD line crosses BELOW the Signal line (bearish momentum)
  HOLD → no crossover

Additionally uses the MACD histogram for confirmation:
  BUY  is stronger when histogram is rising (momentum accelerating)
  SELL is stronger when histogram is falling (momentum decelerating)

Parameters (standard):
  Fast EMA:   12 days
  Slow EMA:   26 days
  Signal EMA:  9 days
"""

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal, StrategyInfo


class MACDStrategy(BaseStrategy):
    """
    MACD Crossover Strategy.

    Parameters:
        fast_period   (int): Fast EMA window. Default 12.
        slow_period   (int): Slow EMA window. Default 26.
        signal_period (int): Signal line EMA window. Default 9.
    """

    def __init__(
        self,
        fast_period:   int = 12,
        slow_period:   int = 26,
        signal_period: int = 9,
    ):
        self.fast_period   = fast_period
        self.slow_period   = slow_period
        self.signal_period = signal_period

    @property
    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name        = "Strategy_C",
            version     = "1.0",
            description = (
                f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) Crossover"
            ),
            parameters  = {
                "fast_period":   self.fast_period,
                "slow_period":   self.slow_period,
                "signal_period": self.signal_period,
            },
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Compute MACD components if not present
        if "macd" not in df.columns:
            ema_fast = df["close"].ewm(span=self.fast_period,   adjust=False).mean()
            ema_slow = df["close"].ewm(span=self.slow_period,   adjust=False).mean()
            df["macd"]        = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=self.signal_period, adjust=False).mean()
            df["macd_hist"]   = df["macd"] - df["macd_signal"]

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Signal generation:
          - MACD crosses above signal line → BUY
          - MACD crosses below signal line → SELL
          - Histogram confirms direction
        """
        macd        = df["macd"]
        macd_signal = df["macd_signal"]
        macd_hist   = df["macd_hist"]

        # Is MACD above signal line right now?
        macd_above      = macd > macd_signal
        prev_macd_above = macd_above.shift(1).fillna(False)

        # Crossover events
        bullish_cross = (~prev_macd_above) & macd_above   # MACD just crossed above
        bearish_cross = (prev_macd_above)  & ~macd_above  # MACD just crossed below

        # Histogram momentum confirmation
        hist_rising  = macd_hist > macd_hist.shift(1)     # Histogram increasing
        hist_falling = macd_hist < macd_hist.shift(1)     # Histogram decreasing

        conditions = [
            bullish_cross,   # BUY on bullish MACD crossover
            bearish_cross,   # SELL on bearish MACD crossover
        ]
        choices = [Signal.BUY, Signal.SELL]

        df["signal"]   = np.select(conditions, choices, default=Signal.HOLD)
        df["position"] = np.where(macd_above, 1, 0)

        # Extra column: signal strength based on histogram
        df["signal_strength"] = macd_hist.abs()

        counts = df["signal"].value_counts().to_dict()
        logger.debug(f"[{self.info.name}] Signals: {counts}")

        return df
