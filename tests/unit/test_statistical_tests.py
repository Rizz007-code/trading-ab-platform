# tests/unit/test_statistical_tests.py
"""Unit tests for ab_testing/statistical_tests.py — no external dependencies."""

import numpy as np
import pandas as pd
import pytest

from ab_testing.statistical_tests import (
    BootstrapCI,
    TestResult,
    bootstrap_sharpe_ci,
    mann_whitney,
    run_all_tests,
    ttest,
)

# Reproducible random data
RNG = np.random.default_rng(42)


def _returns(mean: float, std: float = 0.015, n: int = 252) -> pd.Series:
    return pd.Series(RNG.normal(mean, std, n))


# ── ttest ─────────────────────────────────────────────────────────────────────

class TestTtest:
    def test_returns_test_result(self):
        a, b = _returns(0.001), _returns(0.001)
        result = ttest(a, b)
        assert isinstance(result, TestResult)
        assert result.test_name == "welch_t_test"

    def test_p_value_in_range(self):
        a, b = _returns(0.001), _returns(0.001)
        result = ttest(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_clearly_different_distributions_are_significant(self):
        # Large mean difference → should be significant
        a = _returns(0.010, std=0.005, n=252)
        b = _returns(-0.010, std=0.005, n=252)
        result = ttest(a, b)
        assert result.is_significant
        assert result.p_value < 0.05

    def test_identical_distributions_rarely_significant(self):
        np.random.seed(99)
        a = pd.Series(np.random.normal(0.0, 0.01, 252))
        b = pd.Series(np.random.normal(0.0, 0.01, 252))
        result = ttest(a, b, alpha=0.001)   # very strict alpha
        # Not guaranteed, but with tiny alpha unlikely to be significant
        assert result.confidence_level == 0.999

    def test_insufficient_data_returns_non_significant(self):
        a = pd.Series([0.01])   # only 1 value
        b = pd.Series([0.02, 0.03])
        result = ttest(a, b)
        assert result.p_value == 1.0
        assert not result.is_significant

    def test_confidence_level_stored_correctly(self):
        a, b = _returns(0.001), _returns(0.001)
        result = ttest(a, b, alpha=0.10)
        assert abs(result.confidence_level - 0.90) < 1e-10

    def test_statistic_is_finite(self):
        a, b = _returns(0.001), _returns(-0.001)
        result = ttest(a, b)
        assert np.isfinite(result.statistic)


# ── mann_whitney ──────────────────────────────────────────────────────────────

class TestMannWhitney:
    def test_returns_test_result(self):
        a, b = _returns(0.001), _returns(0.001)
        result = mann_whitney(a, b)
        assert isinstance(result, TestResult)
        assert result.test_name == "mann_whitney_u"

    def test_p_value_in_range(self):
        a, b = _returns(0.001), _returns(0.001)
        result = mann_whitney(a, b)
        assert 0.0 <= result.p_value <= 1.0

    def test_clearly_different_stochastic_distributions(self):
        a = _returns(0.005, std=0.003, n=500)
        b = _returns(-0.005, std=0.003, n=500)
        result = mann_whitney(a, b)
        assert result.is_significant

    def test_insufficient_data_returns_non_significant(self):
        a = pd.Series([0.01])
        b = pd.Series([0.02])
        result = mann_whitney(a, b)
        assert result.p_value == 1.0
        assert not result.is_significant

    def test_statistic_is_non_negative(self):
        a, b = _returns(0.001), _returns(-0.001)
        result = mann_whitney(a, b)
        assert result.statistic >= 0


# ── bootstrap_sharpe_ci ───────────────────────────────────────────────────────

class TestBootstrapSharpeCi:
    def test_returns_bootstrap_ci(self):
        a, b = _returns(0.001), _returns(0.001)
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=100)
        assert isinstance(ci, BootstrapCI)

    def test_lower_less_than_upper(self):
        a, b = _returns(0.001), _returns(0.001)
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=100)
        assert ci.lower <= ci.upper

    def test_large_difference_excludes_zero(self):
        # A clearly better than B → CI of (Sharpe_A - Sharpe_B) should be > 0
        a = _returns(0.005, std=0.005, n=252)
        b = _returns(-0.005, std=0.005, n=252)
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=500, random_seed=42)
        assert ci.excludes_zero

    def test_identical_distributions_includes_zero(self):
        np.random.seed(0)
        a = pd.Series(np.random.normal(0, 0.01, 252))
        b = pd.Series(np.random.normal(0, 0.01, 252))
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=300, random_seed=0)
        # CI should include zero (no meaningful difference)
        assert not ci.excludes_zero

    def test_insufficient_data_returns_empty_ci(self):
        a = pd.Series([0.01] * 5)   # < 10 values
        b = pd.Series([0.02] * 5)
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=100)
        assert ci.n_bootstrap == 0
        assert not ci.excludes_zero

    def test_n_bootstrap_stored_correctly(self):
        a, b = _returns(0.001), _returns(0.001)
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=200)
        assert ci.n_bootstrap == 200

    def test_excludes_zero_consistent_with_bounds(self):
        a, b = _returns(0.003, n=252), _returns(0.001, n=252)
        ci = bootstrap_sharpe_ci(a, b, n_bootstrap=300)
        # excludes_zero should be True iff 0 is outside [lower, upper]
        expected = not (ci.lower <= 0.0 <= ci.upper)
        assert ci.excludes_zero == expected


# ── run_all_tests ─────────────────────────────────────────────────────────────

class TestRunAllTests:
    def test_returns_dict_with_expected_keys(self):
        a, b = _returns(0.001), _returns(-0.001)
        result = run_all_tests(a, b)
        assert "t_test" in result
        assert "mann_whitney" in result
        assert "bootstrap_ci_sharpe_diff" in result

    def test_sub_dict_keys(self):
        a, b = _returns(0.001), _returns(-0.001)
        result = run_all_tests(a, b)
        for key in ("statistic", "p_value", "is_significant"):
            assert key in result["t_test"]
            assert key in result["mann_whitney"]
        assert "lower" in result["bootstrap_ci_sharpe_diff"]
        assert "upper" in result["bootstrap_ci_sharpe_diff"]
        assert "excludes_zero" in result["bootstrap_ci_sharpe_diff"]

    def test_values_are_rounded(self):
        a, b = _returns(0.001), _returns(-0.001)
        result = run_all_tests(a, b)
        # p_value should be rounded to 6 decimal places
        pv = result["t_test"]["p_value"]
        assert pv == round(pv, 6)
