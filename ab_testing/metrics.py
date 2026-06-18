# ab_testing/metrics.py
"""
Performance metrics for backtested strategies.

Metrics computed:
  - Sharpe Ratio      (annualized, risk-free adjusted)
  - Annual Return     (CAGR)
  - Volatility        (annualized std of daily returns)
  - Max Drawdown      (peak-to-trough, as a negative fraction)
  - Win Rate          (fraction of non-zero return days that are positive)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ─── Metrics Container ────────────────────────────────────────────────────────
@dataclass
class StrategyMetrics:
    sharpe_ratio:  float
    annual_return: float
    volatility:    float
    max_drawdown:  float
    win_rate:      float
    total_return:  float
    num_trades:    int


# ─── Individual Metric Functions ──────────────────────────────────────────────

def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    risk_free_rate is annual; converted to daily internally.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0

    daily_rf = risk_free_rate / TRADING_DAYS
    excess   = returns - daily_rf
    std      = excess.std()

    if std == 0.0 or np.isnan(std):
        return 0.0

    return float((excess.mean() / std) * np.sqrt(TRADING_DAYS))


def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Maximum drawdown as a negative fraction (e.g. -0.23 = -23%).
    Uses running peak as reference.
    """
    portfolio_values = portfolio_values.dropna()
    if portfolio_values.empty:
        return 0.0

    rolling_max = portfolio_values.cummax()
    drawdown    = (portfolio_values - rolling_max) / rolling_max.replace(0, np.nan)
    return float(drawdown.min())


def compute_win_rate(returns: pd.Series) -> float:
    """
    Fraction of days with a positive return.
    Ignores days with zero return (flat/no position days).
    """
    active = returns.dropna()
    active = active[active != 0.0]

    if active.empty:
        return 0.0

    return float((active > 0).sum() / len(active))


def compute_annual_return(portfolio_values: pd.Series) -> float:
    """
    Compound Annual Growth Rate (CAGR).
    Uses number of rows as proxy for trading days.
    """
    portfolio_values = portfolio_values.dropna()
    if len(portfolio_values) < 2:
        return 0.0

    years        = len(portfolio_values) / TRADING_DAYS
    start_val    = portfolio_values.iloc[0]
    end_val      = portfolio_values.iloc[-1]

    if start_val <= 0 or years == 0:
        return 0.0

    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def compute_volatility(returns: pd.Series) -> float:
    """Annualized volatility (std of daily returns × sqrt(252))."""
    returns = returns.dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def compute_total_return(portfolio_values: pd.Series) -> float:
    """Simple total return from first to last portfolio value."""
    portfolio_values = portfolio_values.dropna()
    if len(portfolio_values) < 2 or portfolio_values.iloc[0] == 0:
        return 0.0
    return float((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0)


# ─── Main Compute Function ────────────────────────────────────────────────────

def compute_metrics(result_df: pd.DataFrame, risk_free_rate: float = 0.0) -> StrategyMetrics:
    """
    Compute all performance metrics from a simulator result DataFrame.

    Args:
        result_df:       Output of Simulator.run() — must have columns:
                         daily_return, portfolio_value, signal
        risk_free_rate:  Annual risk-free rate (default 0.0)

    Returns:
        StrategyMetrics dataclass with all computed values.
    """
    returns          = result_df["daily_return"].dropna()
    portfolio_values = result_df["portfolio_value"].dropna()

    num_trades = int((result_df["signal"] == "BUY").sum()) if "signal" in result_df.columns else 0

    return StrategyMetrics(
        sharpe_ratio  = compute_sharpe(returns, risk_free_rate),
        annual_return = compute_annual_return(portfolio_values),
        volatility    = compute_volatility(returns),
        max_drawdown  = compute_max_drawdown(portfolio_values),
        win_rate      = compute_win_rate(returns),
        total_return  = compute_total_return(portfolio_values),
        num_trades    = num_trades,
    )


# ─── Standalone Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    # Simulate a strategy result DataFrame
    n = 252
    daily_rets = pd.Series(np.random.normal(0.0005, 0.015, n))
    portfolio  = (1 + daily_rets).cumprod() * 100_000

    mock_df = pd.DataFrame({
        "daily_return":    daily_rets,
        "portfolio_value": portfolio,
        "signal":          ["BUY"] * 10 + ["HOLD"] * (n - 10),
    })

    metrics = compute_metrics(mock_df)
    print("\n=== Metrics Test ===")
    print(f"  Sharpe Ratio:  {metrics.sharpe_ratio:.4f}")
    print(f"  Annual Return: {metrics.annual_return * 100:.2f}%")
    print(f"  Volatility:    {metrics.volatility * 100:.2f}%")
    print(f"  Max Drawdown:  {metrics.max_drawdown * 100:.2f}%")
    print(f"  Win Rate:      {metrics.win_rate * 100:.2f}%")
    print(f"  Total Return:  {metrics.total_return * 100:.2f}%")
    print(f"  Num Trades:    {metrics.num_trades}")
