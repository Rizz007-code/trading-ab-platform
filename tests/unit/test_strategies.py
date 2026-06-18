# tests/unit/test_strategies.py
"""Unit tests for trading strategies — pure pandas/numpy, no DB or network."""

import numpy as np
import pandas as pd
import pytest

from strategies.base_strategy import BaseStrategy, Signal, StrategyInfo
from strategies.strategy_a import MACrossoverStrategy
from strategies.strategy_b import MARSIStrategy
from strategies.strategy_c import MACDStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)   # keep positive
    df = pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    return df


def _trending_up(n: int = 200) -> pd.DataFrame:
    """Steadily rising price — should produce mostly BUY signals."""
    close = np.linspace(50, 150, n)
    df = pd.DataFrame({
        "open":  close * 0.99,
        "high":  close * 1.02,
        "low":   close * 0.97,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    return df


def _up_then_down(n: int = 200, peak_frac: float = 0.4) -> pd.DataFrame:
    """
    Price rises to a peak then falls sharply.
    Guarantees at least one bearish crossover (price crosses from above to below MA).
    """
    peak = int(n * peak_frac)
    rise = np.linspace(80, 150, peak)
    fall = np.linspace(150, 30, n - peak)
    close = np.concatenate([rise, fall])
    df = pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.02,
        "low":    close * 0.97,
        "close":  close,
        "volume": np.full(n, 1_000_000.0),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    return df


# ── Signal constants ──────────────────────────────────────────────────────────

class TestSignalConstants:
    def test_values(self):
        assert Signal.BUY  == "BUY"
        assert Signal.SELL == "SELL"
        assert Signal.HOLD == "HOLD"


# ── BaseStrategy.run() validation ─────────────────────────────────────────────

class TestBaseStrategyValidation:
    def test_missing_signal_column_raises(self):
        class BadStrategy(BaseStrategy):
            @property
            def info(self):
                return StrategyInfo("Bad", "0", "desc", {})
            def generate_signals(self, df):
                return df   # deliberately omits 'signal' column

        s = BadStrategy()
        with pytest.raises(ValueError, match="signal"):
            s.run(_make_ohlcv(50))

    def test_invalid_signal_value_raises(self):
        class WeirdStrategy(BaseStrategy):
            @property
            def info(self):
                return StrategyInfo("Weird", "0", "desc", {})
            def generate_signals(self, df):
                df = df.copy()
                df["signal"] = "UNKNOWN"
                return df

        s = WeirdStrategy()
        with pytest.raises(ValueError, match="Invalid signals"):
            s.run(_make_ohlcv(50))


# ── MACrossoverStrategy ───────────────────────────────────────────────────────

class TestMACrossoverStrategy:
    def test_repr(self):
        s = MACrossoverStrategy()
        assert "Strategy_A" in repr(s)

    def test_info_fields(self):
        s = MACrossoverStrategy(ma_window=20)
        assert s.info.name == "Strategy_A"
        assert s.info.parameters["ma_window"] == 20

    def test_run_returns_dataframe(self):
        s = MACrossoverStrategy()
        result = s.run(_make_ohlcv())
        assert isinstance(result, pd.DataFrame)

    def test_signal_column_present(self):
        s = MACrossoverStrategy()
        result = s.run(_make_ohlcv())
        assert "signal" in result.columns

    def test_position_column_present(self):
        s = MACrossoverStrategy()
        result = s.run(_make_ohlcv())
        assert "position" in result.columns

    def test_signals_are_valid_values(self):
        s = MACrossoverStrategy()
        result = s.run(_make_ohlcv())
        valid = {Signal.BUY, Signal.SELL, Signal.HOLD}
        assert set(result["signal"].unique()).issubset(valid)

    def test_position_is_binary(self):
        s = MACrossoverStrategy()
        result = s.run(_make_ohlcv())
        assert set(result["position"].unique()).issubset({0, 1})

    def test_uptrend_has_buy_signals(self):
        s = MACrossoverStrategy(ma_window=20)
        result = s.run(_trending_up())
        assert (result["signal"] == Signal.BUY).any()

    def test_downtrend_has_sell_signals(self):
        s = MACrossoverStrategy(ma_window=20)
        result = s.run(_up_then_down())
        assert (result["signal"] == Signal.SELL).any()

    def test_custom_ma_window_used(self):
        s = MACrossoverStrategy(ma_window=10)
        df = _make_ohlcv()
        result = s.run(df)
        assert "ma_10" in result.columns

    def test_preexisting_ma_column_not_overwritten(self):
        s = MACrossoverStrategy(ma_window=50)
        df = _make_ohlcv()
        df["ma_50"] = 999.0   # sentinel value
        result = s.run(df)
        # preprocess should detect existing column and skip recomputation
        assert result["ma_50"].iloc[0] == 999.0

    def test_output_length_unchanged(self):
        df = _make_ohlcv(150)
        result = MACrossoverStrategy().run(df)
        assert len(result) == 150


# ── MARSIStrategy ─────────────────────────────────────────────────────────────

class TestMARSIStrategy:
    def test_info_fields(self):
        s = MARSIStrategy()
        assert s.info.name == "Strategy_B"
        assert "ma_window" in s.info.parameters
        assert "rsi_period" in s.info.parameters

    def test_run_returns_dataframe(self):
        s = MARSIStrategy()
        assert isinstance(s.run(_make_ohlcv()), pd.DataFrame)

    def test_signal_column_valid(self):
        s = MARSIStrategy()
        result = s.run(_make_ohlcv())
        valid = {Signal.BUY, Signal.SELL, Signal.HOLD}
        assert set(result["signal"].unique()).issubset(valid)

    def test_rsi_column_computed(self):
        s = MARSIStrategy()
        result = s.run(_make_ohlcv())
        assert "rsi_14" in result.columns

    def test_rsi_values_in_range(self):
        s = MARSIStrategy()
        result = s.run(_make_ohlcv())
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_preexisting_rsi_not_overwritten(self):
        s = MARSIStrategy()
        df = _make_ohlcv()
        df["rsi_14"] = 55.0   # sentinel
        result = s.run(df)
        assert result["rsi_14"].iloc[-1] == 55.0

    def test_overbought_filter_reduces_buys_vs_crossover_only(self):
        # RSI filter should suppress some BUY signals relative to pure MA crossover
        df = _make_ohlcv(seed=7)
        buys_ma  = (MACrossoverStrategy().run(df)["signal"] == Signal.BUY).sum()
        buys_rsi = (MARSIStrategy().run(df)["signal"] == Signal.BUY).sum()
        # RSI strategy BUYs ≤ pure MA BUYs (filter can only suppress, never add)
        assert buys_rsi <= buys_ma

    def test_output_length_unchanged(self):
        df = _make_ohlcv(100)
        assert len(MARSIStrategy().run(df)) == 100


# ── MACDStrategy ──────────────────────────────────────────────────────────────

class TestMACDStrategy:
    def test_info_fields(self):
        s = MACDStrategy()
        assert s.info.name == "Strategy_C"
        assert s.info.parameters["fast_period"] == 12

    def test_run_returns_dataframe(self):
        assert isinstance(MACDStrategy().run(_make_ohlcv()), pd.DataFrame)

    def test_signal_column_valid(self):
        s = MACDStrategy()
        result = s.run(_make_ohlcv())
        valid = {Signal.BUY, Signal.SELL, Signal.HOLD}
        assert set(result["signal"].unique()).issubset(valid)

    def test_macd_columns_computed(self):
        s = MACDStrategy()
        result = s.run(_make_ohlcv())
        for col in ("macd", "macd_signal", "macd_hist"):
            assert col in result.columns

    def test_preexisting_macd_not_overwritten(self):
        s = MACDStrategy()
        df = _make_ohlcv()
        # preprocess checks only for "macd"; must also set sibling columns it would create
        df["macd"]        = 0.5
        df["macd_signal"] = 0.4
        df["macd_hist"]   = 0.1
        result = s.run(df)
        assert result["macd"].iloc[0] == 0.5

    def test_signal_strength_column_present(self):
        s = MACDStrategy()
        result = s.run(_make_ohlcv())
        assert "signal_strength" in result.columns

    def test_signal_strength_is_non_negative(self):
        s = MACDStrategy()
        result = s.run(_make_ohlcv())
        assert (result["signal_strength"] >= 0).all()

    def test_uptrend_has_buy_signals(self):
        s = MACDStrategy(fast_period=5, slow_period=15, signal_period=4)
        result = s.run(_trending_up(200))
        assert (result["signal"] == Signal.BUY).any()

    def test_downtrend_has_sell_signals(self):
        # Up-then-down ensures MACD first crosses above then below signal line (SELL)
        s = MACDStrategy(fast_period=5, slow_period=15, signal_period=4)
        result = s.run(_up_then_down(200))
        assert (result["signal"] == Signal.SELL).any()

    def test_output_length_unchanged(self):
        df = _make_ohlcv(120)
        assert len(MACDStrategy().run(df)) == 120

    def test_position_binary(self):
        s = MACDStrategy()
        result = s.run(_make_ohlcv())
        assert set(result["position"].unique()).issubset({0, 1})
