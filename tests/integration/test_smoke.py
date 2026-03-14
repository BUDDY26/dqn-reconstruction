"""
tests/integration/test_smoke.py — Smoke test: CSV → features → environment → episode.

Verifies that the full first-pass pipeline runs end-to-end without errors:
  1. Load the synthetic fixture CSV (data.load_csv)
  2. Build features (data.build_features)
  3. Split into train / test sets (data.split_by_date)
  4. Scale features (data.fit_scaler, data.apply_scaler)
  5. Extract raw close prices
  6. Instantiate TradingEnv for the training split
  7. Run a complete episode using random actions from the action space
  8. Verify all returned values have the expected types and shapes throughout

This test does not assert on specific numeric values — it asserts on structural
correctness (shapes, types, flag types, non-NaN values).  Numeric correctness
is covered by the unit tests.

No live data is fetched.  All data comes from tests/fixtures/sample_ohlcv.csv.
"""

import numpy as np
import pytest

from config import INITIAL_CAPITAL, OBS_DIM
from data import build_features, load_csv, prepare_arrays
from env import TradingEnv


class TestSmokeEndToEnd:
    """Full pipeline smoke tests."""

    @pytest.fixture(autouse=True)
    def _setup(self, sample_csv_path):
        """Load and prepare data once for all tests in this class."""
        raw_df = load_csv(sample_csv_path)
        feature_df = build_features(raw_df)

        (
            self.train_features,
            self.test_features,
            self.train_close,
            self.test_close,
            self.scaler,
        ) = prepare_arrays(feature_df)

        self.env = TradingEnv(
            features=self.train_features,
            close_prices=self.train_close,
            ticker="SMOKE",
        )

    # -----------------------------------------------------------------------
    # Data pipeline output shapes
    # -----------------------------------------------------------------------

    def test_train_features_shape(self):
        assert self.train_features.ndim == 2
        assert self.train_features.shape[1] == OBS_DIM

    def test_test_features_shape(self):
        assert self.test_features.ndim == 2
        assert self.test_features.shape[1] == OBS_DIM

    def test_train_close_length(self):
        assert len(self.train_close) == len(self.train_features)

    def test_no_nan_in_features(self):
        assert not np.isnan(self.train_features).any()
        assert not np.isnan(self.test_features).any()

    def test_no_nan_in_close_prices(self):
        assert not np.isnan(self.train_close).any()
        assert not np.isnan(self.test_close).any()

    # -----------------------------------------------------------------------
    # Environment initialisation
    # -----------------------------------------------------------------------

    def test_env_is_gymnasium_env(self):
        import gymnasium

        assert isinstance(self.env, gymnasium.Env)

    def test_env_observation_space(self):
        assert self.env.observation_space.shape == (OBS_DIM,)

    def test_env_action_space(self):
        assert self.env.action_space.n == 3

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def test_reset_obs_shape(self):
        obs, info = self.env.reset()
        assert obs.shape == (OBS_DIM,)

    def test_reset_obs_dtype(self):
        obs, _ = self.env.reset()
        assert obs.dtype == np.float32

    def test_reset_info_portfolio_value(self):
        _, info = self.env.reset()
        assert info["portfolio_value"] == pytest.approx(INITIAL_CAPITAL)

    # -----------------------------------------------------------------------
    # Full episode with random actions
    # -----------------------------------------------------------------------

    def test_full_episode_runs_without_error(self):
        """Run a complete episode using actions sampled from the action space."""
        obs, info = self.env.reset()
        terminated = False
        step_count = 0

        while not terminated:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            step_count += 1

            # Structural assertions at every step.
            assert obs.shape == (OBS_DIM,), f"Unexpected obs shape at step {step_count}"
            assert obs.dtype == np.float32
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert not truncated, "truncated should always be False"
            assert not np.isnan(reward), f"NaN reward at step {step_count}"
            assert not np.isnan(obs).any(), f"NaN in observation at step {step_count}"

        # Episode should last exactly n_steps transitions.
        assert step_count == self.env.n_steps

    def test_full_episode_terminates(self):
        """Verify the episode terminates (does not loop forever)."""
        obs, _ = self.env.reset()
        terminated = False
        max_steps = self.env.n_steps + 10  # safety margin

        for _ in range(max_steps):
            action = self.env.action_space.sample()
            _, _, terminated, _, _ = self.env.step(action)
            if terminated:
                break

        assert terminated, "Episode did not terminate within expected step count."

    def test_portfolio_value_is_positive_throughout(self):
        """Portfolio value must remain positive with binary sizing and no leverage."""
        self.env.reset()
        terminated = False

        while not terminated:
            action = self.env.action_space.sample()
            _, _, terminated, _, info = self.env.step(action)
            assert (
                info["portfolio_value"] > 0.0
            ), f"Portfolio value went non-positive: {info['portfolio_value']}"

    def test_buy_hold_sell_sequence_runs(self):
        """Deterministic buy-hold-hold-sell sequence should produce no errors."""
        from config import ACTION_BUY, ACTION_HOLD, ACTION_SELL

        obs, _ = self.env.reset()
        _, r1, t1, _, _ = self.env.step(ACTION_BUY)
        _, r2, t2, _, _ = self.env.step(ACTION_HOLD)
        _, r3, t3, _, _ = self.env.step(ACTION_HOLD)
        _, r4, t4, _, _ = self.env.step(ACTION_SELL)

        for reward in [r1, r2, r3, r4]:
            assert isinstance(reward, float)
            assert not np.isnan(reward)
