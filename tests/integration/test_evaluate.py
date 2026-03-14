"""
tests/integration/test_evaluate.py — Metric computation and evaluate() smoke tests.

Two test categories:
  1. Unit-style tests for the four pure metric functions (known inputs → known outputs).
  2. Smoke tests for evaluate() using the synthetic fixture CSV.

No live data is fetched.  Metric tests use hard-coded numpy arrays with
analytically verifiable expected values.
"""

import numpy as np
import pytest

from agent import DQNAgent
from config import INITIAL_CAPITAL
from data import build_features, load_csv, prepare_arrays
from evaluate import (
    EvalResult,
    compute_calmar,
    compute_max_drawdown,
    compute_roi,
    compute_sharpe,
    evaluate,
)


# ===========================================================================
# 1. Metric function tests (pure functions, no environment)
# ===========================================================================

class TestComputeROI:
    """ROI = (final - initial) / initial."""

    def test_simple_gain(self):
        pv = np.array([100.0, 150.0])
        assert compute_roi(pv) == pytest.approx(0.5)

    def test_simple_loss(self):
        pv = np.array([100.0, 50.0])
        assert compute_roi(pv) == pytest.approx(-0.5)

    def test_breakeven(self):
        pv = np.array([100.0, 100.0])
        assert compute_roi(pv) == pytest.approx(0.0)

    def test_multi_step_uses_first_and_last(self):
        # Intermediate values must not affect the result.
        pv = np.array([100.0, 200.0, 50.0, 120.0])
        assert compute_roi(pv) == pytest.approx(0.20)

    def test_returns_float(self):
        pv = np.array([100.0, 110.0])
        assert isinstance(compute_roi(pv), float)

    def test_large_gain(self):
        pv = np.array([100.0, 1000.0])
        assert compute_roi(pv) == pytest.approx(9.0)


class TestComputeMaxDrawdown:
    """MDD = max over t of (running_peak - value) / running_peak."""

    def test_simple_trough(self):
        # Peak 100, trough 80 → MDD = 0.20.
        pv = np.array([100.0, 90.0, 80.0, 90.0, 100.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.20)

    def test_monotone_increase_has_zero_mdd(self):
        pv = np.array([100.0, 110.0, 120.0, 130.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.0)

    def test_monotone_decrease(self):
        # Entire sequence is a drawdown from the initial value.
        # Peak is always 100; final value 60 → MDD = 0.40.
        pv = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.40)

    def test_mdd_uses_running_peak(self):
        # Peak advances to 110 before dropping to 90.
        # MDD = (110 - 90) / 110 ≈ 0.1818.
        pv = np.array([100.0, 110.0, 90.0, 105.0])
        assert compute_max_drawdown(pv) == pytest.approx(20.0 / 110.0, abs=1e-6)

    def test_is_non_negative(self):
        pv = np.array([100.0, 80.0, 60.0, 70.0, 50.0])
        assert compute_max_drawdown(pv) >= 0.0

    def test_returns_float(self):
        pv = np.array([100.0, 90.0])
        assert isinstance(compute_max_drawdown(pv), float)

    def test_single_trough_at_end(self):
        # Peak 100, then falls to 50 → MDD = 0.50.
        # Using a monotone decline after peak to keep the running max at 100.
        pv = np.array([100.0, 90.0, 50.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.50, abs=1e-6)


class TestComputeSharpe:
    """Annualized Sharpe ratio with zero risk-free rate."""

    def test_zero_std_returns_zero(self):
        # All returns identical → std = 0 → Sharpe = 0.
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        assert compute_sharpe(pv) == pytest.approx(0.0)

    def test_positive_returns_positive_sharpe(self):
        # Non-uniform positive returns → positive mean, positive std → positive Sharpe.
        pv = np.array([100.0, 105.0, 108.0, 115.0, 120.0])
        assert compute_sharpe(pv) > 0.0

    def test_negative_returns_negative_sharpe(self):
        # Declining portfolio with variance → negative mean, positive std → negative Sharpe.
        pv = np.array([100.0, 95.0, 92.0, 87.0, 80.0])
        assert compute_sharpe(pv) < 0.0

    def test_returns_float(self):
        pv = np.array([100.0, 110.0, 105.0, 115.0])
        assert isinstance(compute_sharpe(pv), float)

    def test_too_few_returns_gives_zero(self):
        # Only 1 daily return → cannot compute sample std → returns 0.0.
        pv = np.array([100.0, 110.0])
        assert compute_sharpe(pv) == pytest.approx(0.0)

    def test_annualization_scales_result(self):
        """Sharpe with 252 periods should be sqrt(252) times the per-period ratio."""
        pv = np.array([100.0, 101.0, 100.0, 101.5, 102.0, 101.0])
        sharpe_252 = compute_sharpe(pv, periods_per_year=252)
        sharpe_1 = compute_sharpe(pv, periods_per_year=1)
        assert sharpe_252 == pytest.approx(sharpe_1 * np.sqrt(252), rel=1e-6)


class TestComputeCalmar:
    """Calmar ratio = annualized return / max drawdown."""

    def test_zero_drawdown_returns_zero(self):
        # Monotone increase → MDD = 0 → Calmar = 0 (our convention, A-Ev4).
        pv = np.array([100.0, 110.0, 120.0, 130.0])
        assert compute_calmar(pv) == pytest.approx(0.0)

    def test_profitable_with_drawdown_is_positive(self):
        # Ends higher than it starts AND has a drawdown → Calmar > 0.
        pv = np.array([100.0, 110.0, 90.0, 120.0, 150.0])
        assert compute_calmar(pv) > 0.0

    def test_losing_strategy_with_drawdown_is_negative(self):
        # Ends lower than it starts AND has a drawdown → Calmar < 0.
        pv = np.array([100.0, 110.0, 80.0, 70.0, 60.0])
        assert compute_calmar(pv) < 0.0

    def test_returns_float(self):
        pv = np.array([100.0, 110.0, 90.0, 120.0])
        assert isinstance(compute_calmar(pv), float)

    def test_relationship_roi_mdd(self):
        """For a short series where annualization ≈ 1, Calmar ≈ ROI / MDD."""
        # Use exactly 252 steps so annualized_return ≈ ROI directly.
        rng = np.random.default_rng(seed=0)
        returns = rng.standard_normal(252) * 0.01 + 0.0005  # small positive drift
        pv = np.cumprod(np.concatenate([[100.0], 1 + returns]))
        calmar = compute_calmar(pv, periods_per_year=252)
        mdd = compute_max_drawdown(pv)
        roi = compute_roi(pv)
        # With n_days=252 and periods_per_year=252, annualized_roi = (1+ROI)^1 - 1 = ROI.
        if mdd > 0:
            assert calmar == pytest.approx(roi / mdd, rel=1e-6)


# ===========================================================================
# 2. evaluate() smoke tests (full pipeline on fixture data)
# ===========================================================================

@pytest.fixture(scope="module")
def test_arrays(sample_csv_path):
    """Prepared test feature and close price arrays from the fixture CSV."""
    raw = load_csv(sample_csv_path)
    feature_df = build_features(raw)
    _, test_features, _, test_close, _ = prepare_arrays(feature_df)
    return test_features, test_close


class TestEvaluateSmoke:
    """Structural correctness checks for evaluate() on fixture data."""

    @pytest.fixture(autouse=True)
    def _result(self, test_arrays):
        features, close = test_arrays
        agent = DQNAgent(seed=0)
        self.result = evaluate(agent, features, close, ticker="EVAL")
        self.n_steps = len(features) - 1  # TradingEnv.n_steps

    def test_returns_eval_result(self):
        assert isinstance(self.result, EvalResult)

    def test_ticker_preserved(self):
        assert self.result.ticker == "EVAL"

    def test_roi_is_float(self):
        assert isinstance(self.result.roi, float)

    def test_sharpe_is_float(self):
        assert isinstance(self.result.sharpe, float)

    def test_max_drawdown_is_float(self):
        assert isinstance(self.result.max_drawdown, float)

    def test_calmar_is_float(self):
        assert isinstance(self.result.calmar, float)

    def test_portfolio_values_is_numpy(self):
        assert isinstance(self.result.portfolio_values, np.ndarray)

    def test_portfolio_values_length(self):
        # One entry after reset + one per step = n_steps + 1.
        assert len(self.result.portfolio_values) == self.n_steps + 1

    def test_initial_portfolio_value(self):
        # First entry must equal INITIAL_CAPITAL.
        assert self.result.portfolio_values[0] == pytest.approx(INITIAL_CAPITAL)

    def test_portfolio_values_all_positive(self):
        assert (self.result.portfolio_values > 0).all()

    def test_max_drawdown_is_non_negative(self):
        assert self.result.max_drawdown >= 0.0

    def test_max_drawdown_at_most_one(self):
        # Cannot lose more than 100% with binary sizing and no leverage.
        assert self.result.max_drawdown <= 1.0

    def test_no_nan_in_metrics(self):
        for metric in [
            self.result.roi,
            self.result.sharpe,
            self.result.max_drawdown,
            self.result.calmar,
        ]:
            assert not np.isnan(metric), f"NaN metric: {metric}"

    def test_greedy_policy_used(self, test_arrays):
        """Evaluate must not modify agent's epsilon or episode counter."""
        features, close = test_arrays
        agent = DQNAgent(seed=0)
        eps_before = agent.epsilon
        ep_before = agent._episode
        evaluate(agent, features, close, ticker="GREEDY")
        assert agent.epsilon == eps_before, "evaluate() must not change epsilon."
        assert agent._episode == ep_before, "evaluate() must not advance episode counter."
