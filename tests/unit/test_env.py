"""
tests/unit/test_env.py — Unit tests for src/env.py (TradingEnv).

Tests verify:
  - Gymnasium API contract: reset() → (obs, info),
    step() → (obs, reward, terminated, truncated, info)
  - Observation space shape and dtype
  - Action space cardinality
  - Reward formula: position × daily_return  [assumption A-E1]
  - Position transitions: buy → long, sell → flat, hold → unchanged  [assumption A-E4]
  - Terminal condition fires at the correct step  [assumptions A-E6, A-E7]
  - Portfolio accounting (cash, shares, portfolio_value)
  - Edge cases: redundant buy, redundant sell, hold from flat

Synthetic fixtures (defined in conftest.py) are used throughout.
No CSV loading or real data is required for these tests.
"""

import numpy as np
import pytest

from config import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    INITIAL_CAPITAL,
    N_ACTIONS,
    OBS_DIM,
)
from env import TradingEnv

# ---------------------------------------------------------------------------
# Fixture: small env with known close prices
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_env(synthetic_features, synthetic_close_prices):
    """TradingEnv with 10 days of data.

    close_prices = [100, 110, 100, 110, 100, 110, 100, 110, 100, 110]
    daily_returns (step 0→1): +0.10
    daily_returns (step 1→2): -0.090909...
    Episode length (n_steps): 9
    """
    return TradingEnv(
        features=synthetic_features,
        close_prices=synthetic_close_prices,
        ticker="SYNTH",
    )


@pytest.fixture
def reset_env(simple_env):
    """TradingEnv after a single reset(), returned as (env, obs, info)."""
    obs, info = simple_env.reset()
    return simple_env, obs, info


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self, simple_env):
        assert simple_env.observation_space.shape == (OBS_DIM,)

    def test_observation_space_dtype(self, simple_env):
        assert simple_env.observation_space.dtype == np.float32

    def test_action_space_n(self, simple_env):
        assert simple_env.action_space.n == N_ACTIONS

    def test_action_buy_in_space(self, simple_env):
        assert simple_env.action_space.contains(ACTION_BUY)

    def test_action_hold_in_space(self, simple_env):
        assert simple_env.action_space.contains(ACTION_HOLD)

    def test_action_sell_in_space(self, simple_env):
        assert simple_env.action_space.contains(ACTION_SELL)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_obs_shape(self, reset_env):
        _, obs, _ = reset_env
        assert obs.shape == (OBS_DIM,)

    def test_obs_dtype(self, reset_env):
        _, obs, _ = reset_env
        assert obs.dtype == np.float32

    def test_obs_is_first_feature_row(self, synthetic_features, synthetic_close_prices):
        env = TradingEnv(features=synthetic_features, close_prices=synthetic_close_prices)
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs, synthetic_features[0])

    def test_position_starts_flat(self, reset_env):
        env, _, _ = reset_env
        assert env.position == 0

    def test_portfolio_value_equals_initial_capital(self, reset_env):
        env, _, info = reset_env
        assert info["portfolio_value"] == pytest.approx(INITIAL_CAPITAL)

    def test_info_keys(self, reset_env):
        _, _, info = reset_env
        expected_keys = {"step", "position", "cash", "shares", "portfolio_value", "close", "ticker"}
        assert set(info.keys()) == expected_keys

    def test_reset_is_idempotent(self, simple_env, synthetic_features):
        """Calling reset() twice must return the same initial observation."""
        obs1, _ = simple_env.reset()
        obs2, _ = simple_env.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_n_steps(self, simple_env):
        """n_steps = len(features) - 1 = 10 - 1 = 9."""
        assert simple_env.n_steps == 9


# ---------------------------------------------------------------------------
# step(): position transitions  [assumption A-E4]
# ---------------------------------------------------------------------------

class TestPositionTransitions:
    def test_hold_when_flat_stays_flat(self, reset_env):
        env, _, _ = reset_env
        env.step(ACTION_HOLD)
        assert env.position == 0

    def test_buy_when_flat_goes_long(self, reset_env):
        env, _, _ = reset_env
        env.step(ACTION_BUY)
        assert env.position == 1

    def test_sell_when_long_goes_flat(self, reset_env):
        env, _, _ = reset_env
        env.step(ACTION_BUY)
        env.step(ACTION_SELL)
        assert env.position == 0

    def test_hold_when_long_stays_long(self, reset_env):
        env, _, _ = reset_env
        env.step(ACTION_BUY)
        env.step(ACTION_HOLD)
        assert env.position == 1

    def test_buy_when_already_long_stays_long(self, reset_env):
        """Redundant buy (already long) must be treated as hold."""
        env, _, _ = reset_env
        env.step(ACTION_BUY)
        env.step(ACTION_BUY)
        assert env.position == 1

    def test_sell_when_already_flat_stays_flat(self, reset_env):
        """Redundant sell (already flat) must be treated as hold; no short-selling."""
        env, _, _ = reset_env
        env.step(ACTION_SELL)
        assert env.position == 0


# ---------------------------------------------------------------------------
# step(): reward formula  [assumption A-E1]
# ---------------------------------------------------------------------------

class TestRewardFormula:
    """
    close_prices = [100, 110, 100, 110, ...]
    daily_return[0→1] = (110 - 100) / 100 = 0.10
    daily_return[1→2] = (100 - 110) / 110 = -1/11 ≈ -0.09090909...
    """

    def test_hold_from_flat_reward_is_zero(self, reset_env):
        """position=0 after HOLD → reward = 0 × daily_return = 0."""
        env, _, _ = reset_env
        _, reward, _, _, _ = env.step(ACTION_HOLD)
        assert reward == pytest.approx(0.0)

    def test_sell_from_flat_reward_is_zero(self, reset_env):
        """position=0 after SELL (noop) → reward = 0."""
        env, _, _ = reset_env
        _, reward, _, _, _ = env.step(ACTION_SELL)
        assert reward == pytest.approx(0.0)

    def test_buy_reward_is_next_day_return(self, reset_env):
        """BUY at step 0: position=1, daily_return = (110-100)/100 = 0.10."""
        env, _, _ = reset_env
        _, reward, _, _, _ = env.step(ACTION_BUY)
        assert reward == pytest.approx(0.10, rel=1e-6)

    def test_hold_long_reward_is_next_day_return(self, reset_env):
        """After BUY at step 0, HOLD at step 1: daily_return[1→2] = -1/11."""
        env, _, _ = reset_env
        env.step(ACTION_BUY)          # step 0: position → 1
        _, reward, _, _, _ = env.step(ACTION_HOLD)   # step 1: reward = 1 × (100-110)/110
        expected = (100.0 - 110.0) / 110.0
        assert reward == pytest.approx(expected, rel=1e-6)

    def test_sell_from_long_reward_is_zero(self, reset_env):
        """SELL at step 1: position → 0 immediately, so reward = 0 × daily_return = 0."""
        env, _, _ = reset_env
        env.step(ACTION_BUY)   # step 0
        _, reward, _, _, _ = env.step(ACTION_SELL)  # step 1: position→0 before reward
        assert reward == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# step(): observation advancement
# ---------------------------------------------------------------------------

class TestObservationAdvancement:
    def test_step_returns_next_feature_row(self, synthetic_features, synthetic_close_prices):
        """Observation returned by step() at step t should be features[t+1]."""
        env = TradingEnv(features=synthetic_features, close_prices=synthetic_close_prices)
        env.reset()
        obs, _, _, _, _ = env.step(ACTION_HOLD)
        np.testing.assert_array_equal(obs, synthetic_features[1])

    def test_obs_dtype_after_step(self, reset_env):
        env, _, _ = reset_env
        obs, _, _, _, _ = env.step(ACTION_HOLD)
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# step(): terminal condition  [assumptions A-E6, A-E7]
# ---------------------------------------------------------------------------

class TestTerminalCondition:
    def test_not_terminated_before_last_step(self, reset_env):
        env, _, _ = reset_env
        for _ in range(env.n_steps - 1):   # all but the last step
            _, _, terminated, _, _ = env.step(ACTION_HOLD)
            assert not terminated

    def test_terminated_at_last_step(self, reset_env):
        env, _, _ = reset_env
        terminated = False
        for _ in range(env.n_steps):
            _, _, terminated, truncated, _ = env.step(ACTION_HOLD)
        assert terminated
        assert not truncated

    def test_truncated_is_always_false(self, reset_env):
        env, _, _ = reset_env
        for _ in range(env.n_steps):
            _, _, _, truncated, _ = env.step(ACTION_HOLD)
            assert not truncated

    def test_step_after_termination_raises(self, reset_env):
        env, _, _ = reset_env
        for _ in range(env.n_steps):
            env.step(ACTION_HOLD)
        with pytest.raises(AssertionError):
            env.step(ACTION_HOLD)


# ---------------------------------------------------------------------------
# Portfolio accounting  [assumptions A-E3, A-E4]
# ---------------------------------------------------------------------------

class TestPortfolioAccounting:
    def test_buy_depletes_cash(self, reset_env):
        env, _, _ = reset_env
        env.step(ACTION_BUY)
        assert env._cash == pytest.approx(0.0)

    def test_buy_acquires_shares(self, reset_env):
        """After BUY at close=100: shares = initial_capital / 100 = 100."""
        env, _, _ = reset_env
        env.step(ACTION_BUY)
        expected_shares = INITIAL_CAPITAL / 100.0   # close[0] = 100.0
        assert env._shares == pytest.approx(expected_shares, rel=1e-6)

    def test_sell_recovers_portfolio_value(self, reset_env):
        """BUY at 100, price rises to 110, SELL at 110 → portfolio value = 11,000."""
        env, _, _ = reset_env
        env.step(ACTION_BUY)   # buy at close[0] = 100 → shares = 100
        env.step(ACTION_SELL)  # sell at close[1] = 110 → cash = 100 × 110 = 11000
        assert env._cash == pytest.approx(INITIAL_CAPITAL * 1.10, rel=1e-6)

    def test_portfolio_value_when_flat(self, reset_env):
        env, _, info = reset_env
        assert info["portfolio_value"] == pytest.approx(INITIAL_CAPITAL)

    def test_portfolio_value_when_long(self, reset_env):
        """After BUY at 100, portfolio_value = shares × current_close."""
        env, _, _ = reset_env
        _, _, _, _, info = env.step(ACTION_BUY)   # now at step 1, close = 110
        # shares = 10000 / 100 = 100; portfolio_value = 100 × 110 = 11000
        assert info["portfolio_value"] == pytest.approx(11_000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_feature_width_raises(self, synthetic_close_prices):
        bad_features = np.ones((10, 5), dtype=np.float32)   # wrong n_cols
        with pytest.raises(ValueError, match="shape"):
            TradingEnv(features=bad_features, close_prices=synthetic_close_prices)

    def test_mismatched_lengths_raise(self, synthetic_features):
        bad_close = np.ones(5, dtype=np.float64)   # wrong length
        with pytest.raises(ValueError, match="length"):
            TradingEnv(features=synthetic_features, close_prices=bad_close)

    def test_too_few_rows_raises(self, synthetic_close_prices):
        one_row = np.ones((1, OBS_DIM), dtype=np.float32)
        one_close = synthetic_close_prices[:1]
        with pytest.raises(ValueError, match="at least 2"):
            TradingEnv(features=one_row, close_prices=one_close)

    def test_negative_capital_raises(self, synthetic_features, synthetic_close_prices):
        with pytest.raises(ValueError, match="positive"):
            TradingEnv(features=synthetic_features, close_prices=synthetic_close_prices,
                       initial_capital=-100.0)
