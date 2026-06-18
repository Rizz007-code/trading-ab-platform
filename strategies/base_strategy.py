# strategies/base_strategy.py
"""
Abstract Base Strategy.
Every strategy must extend this class and implement generate_signals().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ─── Signal Constants ─────────────────────────────────────────────────────────
class Signal:
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ─── Strategy Metadata ────────────────────────────────────────────────────────
@dataclass
class StrategyInfo:
    name:        str
    version:     str
    description: str
    parameters:  dict


# ─── Base Strategy ────────────────────────────────────────────────────────────
class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement:
        - info()              → StrategyInfo
        - generate_signals()  → pd.DataFrame with 'signal' column

    Subclasses may override:
        - preprocess()        → called before generate_signals()
    """

    @property
    @abstractmethod
    def info(self) -> StrategyInfo:
        """Return strategy metadata."""
        ...

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional preprocessing step.
        Override to add custom cleaning before signal generation.
        Default: return df as-is.
        """
        return df.copy()

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Core strategy logic.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                and any features already computed (rsi_14, macd, ma_50, etc.)
                Index must be DatetimeIndex.

        Returns:
            DataFrame with original columns PLUS:
                - signal:        'BUY' | 'SELL' | 'HOLD'
                - position:      1 (long) | 0 (flat) | -1 (short, if supported)
        """
        ...

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: preprocess → generate_signals.
        This is what the simulator calls.
        """
        df = self.preprocess(df)
        df = self.generate_signals(df)

        # Validate output
        if "signal" not in df.columns:
            raise ValueError(f"[{self.info.name}] generate_signals() must return a 'signal' column")

        valid_signals = {Signal.BUY, Signal.SELL, Signal.HOLD}
        bad = set(df["signal"].unique()) - valid_signals
        if bad:
            raise ValueError(f"[{self.info.name}] Invalid signals found: {bad}")

        return df

    def __repr__(self):
        return f"<Strategy: {self.info.name} v{self.info.version}>"
