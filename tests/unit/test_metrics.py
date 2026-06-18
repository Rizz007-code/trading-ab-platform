# tests/unit/test_metrics.py
"""Unit tests for ab_testing/metrics.py — no external dependencies required."""

import numpy as np
import pandas as pd
import pytest

from ab_testing.metrics import (
    TRADING_DAYS,
    StrategyMetrics,
    compute_annual_return,
    compute_max_drawdown,
    compute_metrics,
    compute_sharpe,
    compute_total_return,
    compute_volatility,
    compute_win_rate,
)


# ── compute_sharpe ────────────────────────────────────────────────────────────

class TestComputeSharpe:
    def test_positive_returns_give_positive_sharpe(self):
        returns = pd.Series([0.001] * 252)
        assert compute_sharpe(returns) > 0

    def test_negative_returns_give_negative_sharpe(self):
        returns = pd.Series([-0.001] * 252)
        assert compute_sharpe(returns) < 0

    def test_zero_returns_give_zero_sharpe(self):
        returns = pd.Series([0.0] * 100)
        assert compute_sharpe(returns) == 0.0

    def test_empty_series_returns_zero(self):
        assert compute_sharpe(pd.Series([], dtype=float)) == 0.0

    def test_single_value_returns_zero(self):
        assert compute_sharpe(pd.Series([0.01])) == 0.0

    def test_annualisation_factor(self):
        # Constant daily return r → Sharpe = r/0 but with std=0 gives 0
        # Test that scaling is by sqrt(252)
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = compute_sharpe(returns)
        # Manual calculation
        excess = returns - 0.0
        expected = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)
        assert abs(sharpe - expected) < 1e-10

    def test_risk_free_rate_reduces_sharpe(self):
        returns = pd.Series([0.001] * 252)
        sharpe_zero_rf = compute_sharpe(returns, risk_free_rate=0.0)
        sharpe_high_rf = compute_sharpe(returns, risk_free_rate=0.05)
        assert sharpe_zero_rf > sharpe_high_rf

    def test_nan_values_are_dropped(self):
        returns = pd.Series([0.001, float("nan"), 0.001, 0.001])
        # Should not raise, should compute on non-NaN values
        result = compute_sharpe(returns)
        assert isinstance(result, float)


# ── compute_max_drawdown ──────────────────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_declining_portfolio_has_large_drawdown(self):
        # 100 → 50 = -50% drawdown
        values = pd.Series([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
        dd = compute_max_drawdown(values)
        assert abs(dd - (-0.5)) < 1e-6

    def test_rising_portfolio_has_zero_drawdown(self):
        values = pd.Series([100.0, 110.0, 120.0, 130.0])
        dd = compute_max_drawdown(values)
        assert dd == 0.0

    def test_peak_then_recovery(self):
        # 100 → 120 (peak) → 80 (trough) → 130
        values = pd.Series([100.0, 120.0, 80.0, 130.0])
        dd = compute_max_drawdown(values)
        # Max drawdown from 120 to 80 = (80-120)/120 ≈ -0.333
        assert abs(dd - (-1 / 3)) < 1e-6

    def test_result_is_non_positive(self):
        np.random.seed(42)
        values = pd.Series(np.random.lognormal(0, 0.02, 100).cumprod() * 100)
        assert compute_max_drawdown(values) <= 0.0

    def test_empty_series_returns_zero(self):
        assert compute_max_drawdown(pd.Series([], dtype=float)) == 0.0


# ── compute_win_rate ──────────────────────────────────────────────────────────

class TestComputeWinRate:
    def test_all_positive_returns_win_rate_one(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        assert compute_win_rate(returns) == 1.0

    def test_all_negative_returns_win_rate_zero(self):
        returns = pd.Series([-0.01, -0.02, -0.03])
        assert compute_win_rate(returns) == 0.0

    def test_mixed_returns(self):
        # 3 positive, 2 negative → win rate = 0.6
        returns = pd.Series([0.01, 0.02, 0.01, -0.01, -0.02])
        assert abs(compute_win_rate(returns) - 0.6) < 1e-10

    def test_zero_returns_are_excluded(self):
        # Zeros are flat days, not counted as wins or losses
        returns = pd.Series([0.01, 0.0, 0.0, -0.01, 0.02])
        # 2 positive, 1 negative (ignoring 2 zeros) → 2/3
        assert abs(compute_win_rate(returns) - (2 / 3)) < 1e-10

    def test_all_zeros_returns_zero(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        assert compute_win_rate(returns) == 0.0

    def test_empty_series_returns_zero(self):
        assert compute_win_rate(pd.Series([], dtype=float)) == 0.0


# ── compute_annual_return ─────────────────────────────────────────────────────

class TestComputeAnnualReturn:
    def test_doubling_in_one_year(self):
        # $100 → $200 over exactly 252 days ≈ 100% CAGR
        values = pd.Series([100.0 + i * (100.0 / 252) for i in range(252 + 1)])
        annual = compute_annual_return(values)
        assert 0.90 < annual < 1.10   # roughly 100%

    def test_flat_portfolio_returns_zero(self):
        values = pd.Series([100.0] * 252)
        assert abs(compute_annual_return(values)) < 1e-6

    def test_declining_portfolio_returns_negative(self):
        values = pd.Series([100.0] + [90.0] * 252)
        assert compute_annual_return(values) < 0

    def test_empty_series_returns_zero(self):
        assert compute_annual_return(pd.Series([], dtype=float)) == 0.0

    def test_single_value_returns_zero(self):
        assert compute_annual_return(pd.Series([100.0])) == 0.0

    def test_zero_start_returns_zero(self):
        values = pd.Series([0.0, 10.0, 20.0])
        assert compute_annual_return(values) == 0.0


# ── compute_volatility ────────────────────────────────────────────────────────

class TestComputeVolatility:
    def test_constant_returns_zero_volatility(self):
        returns = pd.Series([0.001] * 100)
        # Constant series has effectively zero variance (floating-point epsilon near 0)
        assert compute_volatility(returns) < 1e-10

    def test_volatile_series_has_positive_vol(self):
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        vol = compute_volatility(returns)
        assert vol > 0
        # Annualised vol ≈ 0.02 * sqrt(252) ≈ 0.317
        assert abs(vol - 0.02 * np.sqrt(252)) < 0.05

    def test_empty_or_single_returns_zero(self):
        assert compute_volatility(pd.Series([], dtype=float)) == 0.0
        assert compute_volatility(pd.Series([0.01])) == 0.0


# ── compute_total_return ──────────────────────────────────────────────────────

class TestComputeTotalReturn:
    def test_doubling_returns_one(self):
        values = pd.Series([100.0, 200.0])
        assert abs(compute_total_return(values) - 1.0) < 1e-10

    def test_no_change_returns_zero(self):
        values = pd.Series([100.0, 100.0])
        assert compute_total_return(values) == 0.0

    def test_loss_returns_negative(self):
        values = pd.Series([100.0, 50.0])
        assert abs(compute_total_return(values) - (-0.5)) < 1e-10

    def test_zero_start_returns_zero(self):
        assert compute_total_return(pd.Series([0.0, 100.0])) == 0.0

    def test_empty_returns_zero(self):
        assert compute_total_return(pd.Series([], dtype=float)) == 0.0


# ── compute_metrics (integration of all metrics) ──────────────────────────────

class TestComputeMetrics:
    def _make_df(self, n: int = 252, daily_ret: float = 0.001):
        np.random.seed(42)
        returns   = pd.Series(np.random.normal(daily_ret, 0.015, n))
        portfolio = (1 + returns).cumprod() * 100_000
        signals   = ["BUY"] * 5 + ["HOLD"] * (n - 5)
        return pd.DataFrame({
            "daily_return":    returns,
            "portfolio_value": portfolio,
            "signal":          signals,
        })

    def test_returns_strategy_metrics_dataclass(self):
        df = self._make_df()
        m  = compute_metrics(df)
        assert isinstance(m, StrategyMetrics)

    def test_all_fields_are_finite(self):
        df = self._make_df()
        m  = compute_metrics(df)
        assert np.isfinite(m.sharpe_ratio)
        assert np.isfinite(m.annual_return)
        assert np.isfinite(m.volatility)
        assert np.isfinite(m.max_drawdown)
        assert np.isfinite(m.win_rate)
        assert np.isfinite(m.total_return)

    def test_num_trades_counts_buy_signals(self):
        df = self._make_df(n=252, daily_ret=0.001)
        m  = compute_metrics(df)
        assert m.num_trades == 5

    def test_missing_signal_column_gives_zero_trades(self):
        df = self._make_df()
        df = df.drop(columns=["signal"])
        m  = compute_metrics(df)
        assert m.num_trades == 0

    def test_max_drawdown_is_non_positive(self):
        df = self._make_df()
        m  = compute_metrics(df)
        assert m.max_drawdown <= 0.0

    def test_win_rate_between_zero_and_one(self):
        df = self._make_df()
        m  = compute_metrics(df)
        assert 0.0 <= m.win_rate <= 1.0
