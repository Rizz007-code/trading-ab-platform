# ab_testing/statistical_tests.py
"""
Statistical tests for comparing two strategy return distributions.

Tests:
  1. Welch's t-test        — parametric, unequal variances OK
  2. Mann-Whitney U test   — non-parametric, no normality assumption
  3. Bootstrap CI          — confidence interval for Sharpe ratio difference

All tests are two-sided. A result is significant when p_value < alpha (default 0.05).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


# ─── Result Containers ────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_name:        str
    statistic:        float
    p_value:          float
    is_significant:   bool
    confidence_level: float


@dataclass
class BootstrapCI:
    lower:        float
    upper:        float
    n_bootstrap:  int
    excludes_zero: bool   # True → difference is meaningful at the chosen confidence level


# ─── Test 1: Welch's t-test ───────────────────────────────────────────────────

def ttest(
    returns_a: pd.Series,
    returns_b: pd.Series,
    alpha:     float = 0.05,
) -> TestResult:
    """
    Welch's t-test for difference in mean daily returns.
    Does not assume equal variance between the two series.

    H0: mean(A) == mean(B)
    H1: mean(A) != mean(B)
    """
    a = returns_a.dropna().values
    b = returns_b.dropna().values

    if len(a) < 2 or len(b) < 2:
        logger.warning("t-test skipped: insufficient data points.")
        return TestResult("welch_t_test", 0.0, 1.0, False, 1 - alpha)

    stat, p = stats.ttest_ind(a, b, equal_var=False)

    return TestResult(
        test_name        = "welch_t_test",
        statistic        = float(stat),
        p_value          = float(p),
        is_significant   = bool(p < alpha),
        confidence_level = 1.0 - alpha,
    )


# ─── Test 2: Mann-Whitney U ───────────────────────────────────────────────────

def mann_whitney(
    returns_a: pd.Series,
    returns_b: pd.Series,
    alpha:     float = 0.05,
) -> TestResult:
    """
    Mann-Whitney U test — non-parametric alternative to t-test.
    Tests whether one distribution is stochastically greater than the other.
    Robust to non-normal return distributions (common in trading strategies).

    H0: P(A > B) == 0.5
    H1: P(A > B) != 0.5
    """
    a = returns_a.dropna().values
    b = returns_b.dropna().values

    if len(a) < 2 or len(b) < 2:
        logger.warning("Mann-Whitney skipped: insufficient data points.")
        return TestResult("mann_whitney_u", 0.0, 1.0, False, 1 - alpha)

    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")

    return TestResult(
        test_name        = "mann_whitney_u",
        statistic        = float(stat),
        p_value          = float(p),
        is_significant   = bool(p < alpha),
        confidence_level = 1.0 - alpha,
    )


# ─── Test 3: Bootstrap CI for Sharpe Difference ───────────────────────────────

def bootstrap_sharpe_ci(
    returns_a:        pd.Series,
    returns_b:        pd.Series,
    n_bootstrap:      int   = 1000,
    confidence_level: float = 0.95,
    risk_free_rate:   float = 0.0,
    random_seed:      int   = 42,
) -> BootstrapCI:
    """
    Bootstrap confidence interval for (Sharpe_A - Sharpe_B).

    Resamples both return series with replacement n_bootstrap times,
    computes the Sharpe difference each time, and returns the CI.

    If the CI excludes zero → the Sharpe difference is statistically meaningful.
    If it includes zero    → we cannot confidently say one strategy beats the other.
    """
    from ab_testing.metrics import compute_sharpe

    a = returns_a.dropna().values
    b = returns_b.dropna().values

    if len(a) < 10 or len(b) < 10:
        logger.warning("Bootstrap CI skipped: fewer than 10 data points.")
        return BootstrapCI(lower=0.0, upper=0.0, n_bootstrap=0, excludes_zero=False)

    rng   = np.random.default_rng(random_seed)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample_a  = pd.Series(rng.choice(a, size=len(a), replace=True))
        sample_b  = pd.Series(rng.choice(b, size=len(b), replace=True))
        sharpe_a  = compute_sharpe(sample_a, risk_free_rate)
        sharpe_b  = compute_sharpe(sample_b, risk_free_rate)
        diffs[i]  = sharpe_a - sharpe_b

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1.0 - alpha / 2)))

    excludes_zero = not (lower <= 0.0 <= upper)

    logger.debug(
        f"Bootstrap CI ({confidence_level*100:.0f}%): "
        f"[{lower:.4f}, {upper:.4f}] | excludes_zero={excludes_zero}"
    )

    return BootstrapCI(
        lower         = lower,
        upper         = upper,
        n_bootstrap   = n_bootstrap,
        excludes_zero = excludes_zero,
    )


# ─── Run All Tests Helper ─────────────────────────────────────────────────────

def run_all_tests(
    returns_a:        pd.Series,
    returns_b:        pd.Series,
    alpha:            float = 0.05,
    n_bootstrap:      int   = 1000,
    confidence_level: float = 0.95,
) -> dict:
    """
    Run t-test, Mann-Whitney, and Bootstrap CI in one call.
    Returns a dict suitable for JSON serialization.
    """
    t   = ttest(returns_a, returns_b, alpha)
    mw  = mann_whitney(returns_a, returns_b, alpha)
    ci  = bootstrap_sharpe_ci(returns_a, returns_b, n_bootstrap, confidence_level)

    return {
        "t_test": {
            "statistic":      round(t.statistic, 6),
            "p_value":        round(t.p_value, 6),
            "is_significant": t.is_significant,
        },
        "mann_whitney": {
            "statistic":      round(mw.statistic, 6),
            "p_value":        round(mw.p_value, 6),
            "is_significant": mw.is_significant,
        },
        "bootstrap_ci_sharpe_diff": {
            "lower":          round(ci.lower, 4),
            "upper":          round(ci.upper, 4),
            "n_bootstrap":    ci.n_bootstrap,
            "excludes_zero":  ci.excludes_zero,
        },
    }


# ─── Standalone Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    np.random.seed(42)

    # Strategy A: slightly better mean return
    returns_a = pd.Series(np.random.normal(0.0007, 0.015, 252))
    # Strategy B: baseline
    returns_b = pd.Series(np.random.normal(0.0003, 0.015, 252))

    print("\n=== Statistical Tests ===")
    results = run_all_tests(returns_a, returns_b)
    print(json.dumps(results, indent=2))
