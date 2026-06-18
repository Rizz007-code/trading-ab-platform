# ab_testing/engine.py
"""
A/B Test Engine — the main orchestrator for Phase 4.

Flow:
  1. Create Experiment record in DB
  2. Run both strategies through Simulator (backtests)
  3. Compute performance metrics for each
  4. Run statistical tests (t-test, Mann-Whitney, Bootstrap CI)
  5. Determine winner by Sharpe ratio (only if tests are significant)
  6. Save ExperimentResult to DB
  7. Return JSON-serializable result dict

Usage:
    from ab_testing.engine import ABTestEngine
    from strategies.strategy_a import MACrossoverStrategy
    from strategies.strategy_b import MARSIStrategy

    engine = ABTestEngine()
    result = engine.run(
        strategy_a = MACrossoverStrategy(),
        strategy_b = MARSIStrategy(),
        ticker     = "AAPL",
        start_date = "2022-01-01",
        end_date   = "2024-01-01",
    )
    print(result["winner"])
"""

import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from loguru import logger

from ab_testing.metrics import StrategyMetrics, compute_metrics
from ab_testing.statistical_tests import (
    BootstrapCI,
    TestResult,
    bootstrap_sharpe_ci,
    mann_whitney,
    ttest,
)
from data.database.connection import get_db_session
from data.database.models import Experiment, ExperimentResult
from strategies.base_strategy import BaseStrategy
from strategies.simulator import Simulator


class ABTestEngine:
    """
    Orchestrates a full A/B experiment between two trading strategies.

    Args:
        initial_capital:  Starting portfolio value for each backtest.
        commission:       Per-trade commission fraction.
        slippage:         Per-trade slippage fraction.
        confidence_level: Statistical confidence level (default 0.95 → α=0.05).
        n_bootstrap:      Number of bootstrap resamples for CI.
    """

    def __init__(
        self,
        initial_capital:  float = 100_000.0,
        commission:       float = 0.001,
        slippage:         float = 0.0005,
        confidence_level: float = 0.95,
        n_bootstrap:      int   = 1000,
    ):
        self.simulator        = Simulator(initial_capital, commission, slippage)
        self.confidence_level = confidence_level
        self.n_bootstrap      = n_bootstrap
        self.alpha            = 1.0 - confidence_level

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        strategy_a:       BaseStrategy,
        strategy_b:       BaseStrategy,
        ticker:           str,
        start_date:       str,
        end_date:         str,
        experiment_name:  Optional[str] = None,
        save_to_db:       bool = True,
    ) -> dict:
        """
        Run a full A/B experiment.

        Args:
            strategy_a:       First strategy instance (treated as "control").
            strategy_b:       Second strategy instance (treated as "challenger").
            ticker:           Stock ticker symbol (must have data in DB).
            start_date:       Backtest start date 'YYYY-MM-DD'.
            end_date:         Backtest end date 'YYYY-MM-DD'.
            experiment_name:  Optional name; auto-generated if not provided.
            save_to_db:       Persist experiment and results to PostgreSQL.

        Returns:
            dict with keys: experiment_name, ticker, winner, is_significant,
                            lift_pct, strategy_a, strategy_b, statistical_tests
        """
        name = experiment_name or (
            f"{strategy_a.info.name}_vs_{strategy_b.info.name}_{ticker}_"
            f"{start_date[:4]}_{end_date[:4]}"
        )

        logger.info("=" * 60)
        logger.info(f"A/B Test: {name}")
        logger.info(f"  Strategies : {strategy_a.info.name}  vs  {strategy_b.info.name}")
        logger.info(f"  Ticker     : {ticker}")
        logger.info(f"  Period     : {start_date} → {end_date}")
        logger.info("=" * 60)

        # Step 1 — Create DB record
        experiment_id = None
        if save_to_db:
            experiment_id = self._create_experiment(
                name, strategy_a, strategy_b, ticker, start_date, end_date
            )

        # Step 2 — Run backtests
        logger.info("Step 1/4: Running backtests...")
        backtest_results = self.simulator.run_multiple(
            strategies    = [strategy_a, strategy_b],
            ticker        = ticker,
            start_date    = start_date,
            end_date      = end_date,
            experiment_id = experiment_id,
            save_to_db    = save_to_db and experiment_id is not None,
        )

        df_a = backtest_results.get(strategy_a.info.name, pd.DataFrame())
        df_b = backtest_results.get(strategy_b.info.name, pd.DataFrame())

        if df_a.empty or df_b.empty:
            self._mark_failed(experiment_id)
            raise ValueError(
                f"Backtest returned empty results for one or both strategies. "
                f"Ensure {ticker} data exists in DB for {start_date} → {end_date}."
            )

        # Step 3 — Compute metrics
        logger.info("Step 2/4: Computing performance metrics...")
        metrics_a = compute_metrics(df_a)
        metrics_b = compute_metrics(df_b)

        logger.info(
            f"  {strategy_a.info.name}: Sharpe={metrics_a.sharpe_ratio:.4f} "
            f"| AnnReturn={metrics_a.annual_return*100:.2f}% "
            f"| MaxDD={metrics_a.max_drawdown*100:.2f}%"
        )
        logger.info(
            f"  {strategy_b.info.name}: Sharpe={metrics_b.sharpe_ratio:.4f} "
            f"| AnnReturn={metrics_b.annual_return*100:.2f}% "
            f"| MaxDD={metrics_b.max_drawdown*100:.2f}%"
        )

        # Step 4 — Statistical tests
        logger.info("Step 3/4: Running statistical tests...")
        returns_a = df_a["daily_return"].dropna()
        returns_b = df_b["daily_return"].dropna()

        t_result  = ttest(returns_a, returns_b, self.alpha)
        mw_result = mann_whitney(returns_a, returns_b, self.alpha)
        ci        = bootstrap_sharpe_ci(
            returns_a, returns_b, self.n_bootstrap, self.confidence_level
        )

        logger.info(
            f"  t-test   : p={t_result.p_value:.4f}  significant={t_result.is_significant}"
        )
        logger.info(
            f"  MW-U     : p={mw_result.p_value:.4f}  significant={mw_result.is_significant}"
        )
        logger.info(
            f"  Bootstrap: CI=[{ci.lower:.4f}, {ci.upper:.4f}]  "
            f"excludes_zero={ci.excludes_zero}"
        )

        # Step 5 — Determine winner
        is_significant = t_result.is_significant or mw_result.is_significant
        winner, lift_pct = self._determine_winner(
            strategy_a.info.name, strategy_b.info.name,
            metrics_a, metrics_b, is_significant,
        )

        # Build result dict
        logger.info("Step 4/4: Building result...")
        result = self._build_result(
            name, ticker, start_date, end_date,
            winner, is_significant, lift_pct,
            strategy_a, strategy_b,
            metrics_a, metrics_b,
            t_result, mw_result, ci,
        )

        # Step 6 — Persist
        if save_to_db and experiment_id is not None:
            self._save_result(experiment_id, result, metrics_a, metrics_b, t_result, ci)

        logger.info("=" * 60)
        logger.info(
            f"RESULT: Winner = {winner or 'No significant winner'} "
            f"(significant={is_significant})"
        )
        if lift_pct:
            logger.info(f"  Sharpe lift: {lift_pct:.2f}%")
        logger.info("=" * 60)

        return result

    # ── DB Helpers ────────────────────────────────────────────────────────────

    def _create_experiment(
        self,
        name:       str,
        strategy_a: BaseStrategy,
        strategy_b: BaseStrategy,
        ticker:     str,
        start_date: str,
        end_date:   str,
    ) -> int:
        with get_db_session() as session:
            experiment = Experiment(
                name       = name,
                strategy_a = strategy_a.info.name,
                strategy_b = strategy_b.info.name,
                ticker     = ticker,
                start_date = start_date,
                end_date   = end_date,
                status     = "running",
            )
            session.add(experiment)
            session.flush()
            exp_id = experiment.id
        logger.info(f"Experiment created: id={exp_id}")
        return exp_id

    def _mark_failed(self, experiment_id: Optional[int]) -> None:
        if experiment_id is None:
            return
        with get_db_session() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment:
                experiment.status = "failed"

    def _save_result(
        self,
        experiment_id: int,
        result:        dict,
        metrics_a:     StrategyMetrics,
        metrics_b:     StrategyMetrics,
        t_result:      TestResult,
        ci:            BootstrapCI,
    ) -> None:
        with get_db_session() as session:
            experiment = session.get(Experiment, experiment_id)
            if experiment:
                experiment.status       = "completed"
                experiment.completed_at = datetime.now(timezone.utc)

            exp_result = ExperimentResult(
                experiment_id    = experiment_id,
                winner           = result["winner"],
                lift_pct         = result["lift_pct"],
                p_value          = t_result.p_value,
                is_significant   = result["is_significant"],
                confidence_level = result["confidence_level"],
                sharpe_a         = metrics_a.sharpe_ratio,
                annual_return_a  = metrics_a.annual_return,
                volatility_a     = metrics_a.volatility,
                max_drawdown_a   = metrics_a.max_drawdown,
                win_rate_a       = metrics_a.win_rate,
                sharpe_b         = metrics_b.sharpe_ratio,
                annual_return_b  = metrics_b.annual_return,
                volatility_b     = metrics_b.volatility,
                max_drawdown_b   = metrics_b.max_drawdown,
                win_rate_b       = metrics_b.win_rate,
                ci_lower         = ci.lower,
                ci_upper         = ci.upper,
                test_method      = "welch_t_test + mann_whitney_u + bootstrap_ci",
            )
            session.add(exp_result)

        logger.info(f"Experiment {experiment_id} result saved to DB.")

    # ── Logic Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _determine_winner(
        name_a:         str,
        name_b:         str,
        metrics_a:      StrategyMetrics,
        metrics_b:      StrategyMetrics,
        is_significant: bool,
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Winner is the strategy with the higher Sharpe ratio,
        but only declared if the difference is statistically significant.
        Lift is expressed as % improvement in Sharpe over the loser.
        """
        if not is_significant:
            return None, None

        sharpe_diff = metrics_a.sharpe_ratio - metrics_b.sharpe_ratio

        if sharpe_diff > 0:
            winner    = name_a
            base      = abs(metrics_b.sharpe_ratio) if metrics_b.sharpe_ratio != 0 else 1.0
            lift_pct  = abs(sharpe_diff / base * 100)
        elif sharpe_diff < 0:
            winner    = name_b
            base      = abs(metrics_a.sharpe_ratio) if metrics_a.sharpe_ratio != 0 else 1.0
            lift_pct  = abs(sharpe_diff / base * 100)
        else:
            return None, None

        return winner, round(lift_pct, 2)

    @staticmethod
    def _build_result(
        name:           str,
        ticker:         str,
        start_date:     str,
        end_date:       str,
        winner:         Optional[str],
        is_significant: bool,
        lift_pct:       Optional[float],
        strategy_a:     BaseStrategy,
        strategy_b:     BaseStrategy,
        metrics_a:      StrategyMetrics,
        metrics_b:      StrategyMetrics,
        t_result:       TestResult,
        mw_result:      TestResult,
        ci:             BootstrapCI,
    ) -> dict:
        return {
            "experiment_name":  name,
            "ticker":           ticker,
            "start_date":       start_date,
            "end_date":         end_date,
            "winner":           winner,
            "is_significant":   is_significant,
            "lift_pct":         lift_pct,
            "confidence_level": 1.0 - (1.0 - 0.95),   # echoes the engine's alpha
            "strategy_a": {
                "name":          strategy_a.info.name,
                "description":   strategy_a.info.description,
                "sharpe":        round(metrics_a.sharpe_ratio,  4),
                "annual_return": round(metrics_a.annual_return, 4),
                "volatility":    round(metrics_a.volatility,    4),
                "max_drawdown":  round(metrics_a.max_drawdown,  4),
                "win_rate":      round(metrics_a.win_rate,      4),
                "total_return":  round(metrics_a.total_return,  4),
                "num_trades":    metrics_a.num_trades,
            },
            "strategy_b": {
                "name":          strategy_b.info.name,
                "description":   strategy_b.info.description,
                "sharpe":        round(metrics_b.sharpe_ratio,  4),
                "annual_return": round(metrics_b.annual_return, 4),
                "volatility":    round(metrics_b.volatility,    4),
                "max_drawdown":  round(metrics_b.max_drawdown,  4),
                "win_rate":      round(metrics_b.win_rate,      4),
                "total_return":  round(metrics_b.total_return,  4),
                "num_trades":    metrics_b.num_trades,
            },
            "statistical_tests": {
                "t_test": {
                    "statistic":      round(t_result.statistic,  6),
                    "p_value":        round(t_result.p_value,    6),
                    "is_significant": t_result.is_significant,
                },
                "mann_whitney": {
                    "statistic":      round(mw_result.statistic, 6),
                    "p_value":        round(mw_result.p_value,   6),
                    "is_significant": mw_result.is_significant,
                },
                "bootstrap_ci_sharpe_diff": {
                    "lower":          round(ci.lower, 4),
                    "upper":          round(ci.upper, 4),
                    "n_bootstrap":    ci.n_bootstrap,
                    "excludes_zero":  ci.excludes_zero,
                },
            },
        }


# ─── Standalone Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from strategies.strategy_a import MACrossoverStrategy
    from strategies.strategy_b import MARSIStrategy

    try:
        result = ABTestEngine(n_bootstrap=500).run(
            strategy_a      = MACrossoverStrategy(),
            strategy_b      = MARSIStrategy(),
            ticker          = "AAPL",
            start_date      = "2022-01-01",
            end_date        = "2024-01-01",
            experiment_name = "AAPL_StratA_vs_StratB_test",
            save_to_db      = False,
        )

        print("\n=== A/B Test Result ===")
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Test failed (expected if DB not running): {e}")
