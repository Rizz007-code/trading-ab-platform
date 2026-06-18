# strategies/simulator.py
"""
Strategy Simulator (Backtester).
==================================
Takes a strategy + price data → runs the backtest → returns daily returns
and portfolio value series.

Features:
  - Transaction cost modeling (commission + slippage)
  - Position sizing (fixed fractional)
  - Max drawdown tracking
  - Stores results to PostgreSQL
  - Returns clean DataFrame for A/B engine to consume
"""

from datetime import date
from typing import Optional, Type

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select

from data.database.connection import get_db_session
from data.database.models import BacktestResult, Experiment, RawPrice, Feature
from strategies.base_strategy import BaseStrategy, Signal


# ─── Simulator Config ─────────────────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL = 100_000.0   # $100k starting capital
DEFAULT_COMMISSION      = 0.001       # 0.1% per trade (realistic for retail)
DEFAULT_SLIPPAGE        = 0.0005      # 0.05% slippage per trade


class Simulator:
    """
    Event-driven backtester.
    Iterates day-by-day through signals and tracks portfolio value.

    Args:
        initial_capital (float): Starting portfolio value in dollars.
        commission      (float): Fraction of trade value charged as commission.
        slippage        (float): Fraction of price lost to slippage on entry/exit.
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        commission:      float = DEFAULT_COMMISSION,
        slippage:        float = DEFAULT_SLIPPAGE,
    ):
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage

    # ── Load Data from DB ─────────────────────────────────────────────────────
    def _load_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load price + feature data from DB for a ticker/date range.
        Merges raw_prices and features tables.
        """
        with get_db_session() as session:
            price_rows = session.execute(
                select(RawPrice)
                .where(
                    RawPrice.ticker >= ticker,
                    RawPrice.ticker <= ticker,
                    RawPrice.date   >= start_date,
                    RawPrice.date   <= end_date,
                )
                .order_by(RawPrice.date.asc())
            ).scalars().all()

            feature_rows = session.execute(
                select(Feature)
                .where(
                    Feature.ticker >= ticker,
                    Feature.ticker <= ticker,
                    Feature.date   >= start_date,
                    Feature.date   <= end_date,
                )
                .order_by(Feature.date.asc())
            ).scalars().all()

        if not price_rows:
            logger.warning(f"No price data for {ticker} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Build price DataFrame
        prices_df = pd.DataFrame([{
            "date":   r.date, "open": r.open, "high": r.high,
            "low":    r.low,  "close": r.close, "volume": r.volume,
        } for r in price_rows])
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        prices_df = prices_df.set_index("date")

        # Build features DataFrame
        if feature_rows:
            features_df = pd.DataFrame([{
                "date":          r.date,
                "ma_50":         r.ma_50,
                "ma_200":        r.ma_200,
                "rsi_14":        r.rsi_14,
                "macd":          r.macd,
                "macd_signal":   r.macd_signal,
                "macd_hist":     r.macd_hist,
                "volatility_20": r.volatility_20,
                "volume_zscore": r.volume_zscore,
                "market_regime": r.market_regime,
            } for r in feature_rows])
            features_df["date"] = pd.to_datetime(features_df["date"])
            features_df = features_df.set_index("date")
            df = prices_df.join(features_df, how="left")
        else:
            df = prices_df

        return df.sort_index()

    # ── Core Backtest Loop ────────────────────────────────────────────────────
    def run(
        self,
        strategy:      BaseStrategy,
        ticker:        str,
        start_date:    str,
        end_date:      str,
        experiment_id: Optional[int] = None,
        save_to_db:    bool = True,
    ) -> pd.DataFrame:
        """
        Run a full backtest.

        Args:
            strategy:      Any BaseStrategy subclass instance.
            ticker:        Stock ticker symbol.
            start_date:    'YYYY-MM-DD' string.
            end_date:      'YYYY-MM-DD' string.
            experiment_id: If provided, results are saved to backtest_results table.
            save_to_db:    Whether to persist results to DB.

        Returns:
            DataFrame with columns:
                date, close, signal, position, daily_return,
                strategy_return, portfolio_value, drawdown
        """
        logger.info(
            f"Running backtest | {strategy.info.name} | {ticker} | {start_date} → {end_date}"
        )

        # Load data
        df = self._load_data(ticker, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data available for {ticker} between {start_date} and {end_date}")

        # Generate signals
        df = strategy.run(df)

        # ── Portfolio Simulation ──────────────────────────────────────────────
        capital         = self.initial_capital
        position        = 0          # 0 = flat, 1 = long
        shares_held     = 0.0
        portfolio_values = []
        daily_returns    = []
        trade_log        = []

        for i, (idx, row) in enumerate(df.iterrows()):
            price  = row["close"]
            signal = row["signal"]

            # Execute trades based on signal
            if signal == Signal.BUY and position == 0:
                # Enter long position
                effective_price = price * (1 + self.slippage)
                commission_cost = capital * self.commission
                capital        -= commission_cost
                shares_held     = capital / effective_price
                capital         = 0.0
                position        = 1
                trade_log.append({"date": idx, "action": "BUY", "price": effective_price})

            elif signal == Signal.SELL and position == 1:
                # Exit long position
                effective_price = price * (1 - self.slippage)
                capital         = shares_held * effective_price
                commission_cost = capital * self.commission
                capital        -= commission_cost
                shares_held     = 0.0
                position        = 0
                trade_log.append({"date": idx, "action": "SELL", "price": effective_price})

            # Current portfolio value
            portfolio_val = capital + (shares_held * price)
            portfolio_values.append(portfolio_val)

            # Daily return
            if i == 0:
                daily_return = 0.0
            else:
                prev_val     = portfolio_values[i - 1]
                daily_return = (portfolio_val - prev_val) / prev_val if prev_val != 0 else 0.0
            daily_returns.append(daily_return)

        df["portfolio_value"] = portfolio_values
        df["daily_return"]    = daily_returns

        # ── Drawdown ─────────────────────────────────────────────────────────
        rolling_max   = df["portfolio_value"].cummax()
        df["drawdown"] = (df["portfolio_value"] - rolling_max) / rolling_max

        logger.info(
            f"Backtest complete | {strategy.info.name} | "
            f"Final value: ${df['portfolio_value'].iloc[-1]:,.2f} | "
            f"Trades: {len(trade_log)}"
        )

        # ── Save to DB ────────────────────────────────────────────────────────
        if save_to_db and experiment_id is not None:
            self._save_results(df, strategy.info.name, ticker, experiment_id)

        df.attrs["trade_log"]      = trade_log
        df.attrs["strategy_name"]  = strategy.info.name
        df.attrs["ticker"]         = ticker

        return df

    # ── Save Results to DB ────────────────────────────────────────────────────
    def _save_results(
        self,
        df:            pd.DataFrame,
        strategy_name: str,
        ticker:        str,
        experiment_id: int,
    ) -> None:
        """Persist daily backtest results to the backtest_results table."""
        with get_db_session() as session:
            for idx, row in df.iterrows():
                result = BacktestResult(
                    experiment_id   = experiment_id,
                    strategy_name   = strategy_name,
                    ticker          = ticker,
                    date            = idx.date() if hasattr(idx, "date") else idx,
                    daily_return    = float(row["daily_return"]),
                    portfolio_value = float(row["portfolio_value"]),
                    signal          = str(row["signal"]),
                )
                session.add(result)
        logger.info(f"Saved {len(df)} backtest rows for {strategy_name} / {ticker}")

    # ── Run Multiple Strategies ───────────────────────────────────────────────
    def run_multiple(
        self,
        strategies:    list[BaseStrategy],
        ticker:        str,
        start_date:    str,
        end_date:      str,
        experiment_id: Optional[int] = None,
        save_to_db:    bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Run backtest for multiple strategies on the same ticker/period.
        Returns dict of {strategy_name: result_df}.
        """
        results = {}
        for strategy in strategies:
            try:
                df = self.run(
                    strategy      = strategy,
                    ticker        = ticker,
                    start_date    = start_date,
                    end_date      = end_date,
                    experiment_id = experiment_id,
                    save_to_db    = save_to_db,
                )
                results[strategy.info.name] = df
            except Exception as e:
                logger.error(f"Backtest failed for {strategy.info.name}: {e}")
                results[strategy.info.name] = pd.DataFrame()

        return results


# ─── Standalone Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from strategies.strategy_a import MACrossoverStrategy
    from strategies.strategy_b import MARSIStrategy
    from strategies.strategy_c import MACDStrategy

    simulator = Simulator()
    strategies = [
        MACrossoverStrategy(),
        MARSIStrategy(),
        MACDStrategy(),
    ]

    results = simulator.run_multiple(
        strategies  = strategies,
        ticker      = "AAPL",
        start_date  = "2022-01-01",
        end_date    = "2024-01-01",
        save_to_db  = False,
    )

    print("\n=== Backtest Results ===")
    for name, df in results.items():
        if df.empty:
            print(f"  {name}: NO DATA")
            continue
        final_val    = df["portfolio_value"].iloc[-1]
        total_return = (final_val - 100_000) / 100_000 * 100
        max_dd       = df["drawdown"].min() * 100
        print(f"  {name}:")
        print(f"    Final Value:  ${final_val:>12,.2f}")
        print(f"    Total Return: {total_return:>+8.2f}%")
        print(f"    Max Drawdown: {max_dd:>+8.2f}%")
        print(f"    Trades:       {len(df.attrs.get('trade_log', []))}")
