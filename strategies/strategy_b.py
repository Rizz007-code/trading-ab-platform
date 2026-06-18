# strategies/strategy_b.py
"""
Strategy B: 50-Day MA Crossover + RSI Filter
=============================================
Builds on Strategy A by adding an RSI filter to reduce false signals.

Logic:
  BUY  → price crosses ABOVE MA50 AND RSI < 70 (not overbought)
  SELL → price crosses BELOW MA50 OR  RSI > 70 (overbought — take profit)
  HOLD → no trigger condition met

The RSI filter prevents buying into overbought conditions,
which is a common cause of whipsaws in pure MA strategies.
"""

import pandas as pd
import numpy as np
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal, StrategyInfo


class MARSIStrategy(BaseStrategy):
    """
    50-day MA Crossover with RSI(14) filter.

    Parameters:
        ma_window    (int):   Moving average window. Default 50.
        rsi_period   (int):   RSI period. Default 14.
        rsi_overbought (int): RSI level above which we avoid buying. Default 70.
        rsi_oversold   (int): RSI level below which signal is stronger. Default 30.
    """

    def __init__(
        self,
        ma_window:      int = 50,
        rsi_period:     int = 14,
        rsi_overbought: int = 70,
        rsi_oversold:   int = 30,
    ):
        self.ma_window      = ma_window
        self.rsi_period     = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold   = rsi_oversold

    @property
    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name        = "Strategy_B",
            version     = "1.0",
            description = f"{self.ma_window}-day MA Crossover + RSI({self.rsi_period}) Filter",
            parameters  = {
                "ma_window":      self.ma_window,
                "rsi_period":     self.rsi_period,
                "rsi_overbought": self.rsi_overbought,
                "rsi_oversold":   self.rsi_oversold,
            },
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Compute MA if not present
        ma_col = f"ma_{self.ma_window}"
        if ma_col not in df.columns:
            df[ma_col] = df["close"].rolling(window=self.ma_window, min_periods=1).mean()

        # Compute RSI if not present
        if "rsi_14" not in df.columns:
            delta    = df["close"].diff()
            gain     = delta.clip(lower=0)
            loss     = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
            avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
            rs       = avg_gain / avg_loss.replace(0, np.nan)
            df["rsi_14"] = (100 - (100 / (1 + rs))).clip(0, 100)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ma_col = f"ma_{self.ma_window}"

        above_ma      = df["close"] > df[ma_col]
        prev_above_ma = above_ma.shift(1).fillna(False)
        rsi           = df["rsi_14"]

        # Crossover events
        bullish_cross = (~prev_above_ma) & above_ma   # Price just crossed above MA
        bearish_cross = (prev_above_ma)  & ~above_ma  # Price just crossed below MA

        # RSI filters
        not_overbought = rsi < self.rsi_overbought    # Safe to buy
        overbought     = rsi > self.rsi_overbought    # Take profit signal

        conditions = [
            bullish_cross & not_overbought,           # BUY: crossover + not overbought
            bearish_cross | overbought,               # SELL: bearish cross OR overbought
        ]
        choices = [Signal.BUY, Signal.SELL]

        df["signal"]   = np.select(conditions, choices, default=Signal.HOLD)
        df["position"] = np.where(above_ma & not_overbought, 1, 0)

        counts = df["signal"].value_counts().to_dict()
        logger.debug(f"[{self.info.name}] Signals: {counts}")

        return df
