"""
tests/integration/test_training_loop.py — Short training run end-to-end.

Verifies that the full training pipeline runs without error over a small number
of episodes using the synthetic fixture CSV.  Does not assert on specific
numeric values — it asserts on structural correctness (types, lengths, ranges).

No live data is fetched.  All data comes from tests/fixtures/sample_ohlcv.csv.
"""

import numpy as np
import pytest

from agent import DQNAgent
from data import build_features, load_csv, prepare_arrays
from train import TrainResult, train

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_arrays(sample_csv_path):
    """Prepared training feature and close price arrays from the fixture CSV."""
    raw = load_csv(sample_csv_path)
    feature_df = build_features(raw)
    train_features, _, train_close, _, _ = prepare_arrays(feature_df)
    return train_features, train_close


# ---------------------------------------------------------------------------
# TrainResult structural correctness
# ---------------------------------------------------------------------------

class TestTrainResultStructure:
    """Verify train() returns a well-formed TrainResult for a 2-episode run."""

    N_EPISODES = 2

    @pytest.fixture(autouse=True)
    def _result(self, train_arrays):
        features, close = train_arrays
        self.result = train(
            features, close, ticker="SMOKE", n_episodes=self.N_EPISODES, seed=42
        )

    def test_returns_train_result_instance(self):
        assert isinstance(self.result, TrainResult)

    def test_ticker_preserved(self):
        assert self.result.ticker == "SMOKE"

    def test_n_episodes_matches_request(self):
        assert self.result.n_episodes == self.N_EPISODES

    def test_episode_rewards_length(self):
        assert len(self.result.episode_rewards) == self.N_EPISODES

    def test_episode_final_values_length(self):
        assert len(self.result.episode_final_values) == self.N_EPISODES

    def test_episode_rewards_are_floats(self):
        for r in self.result.episode_rewards:
            assert isinstance(r, float)

    def test_episode_final_values_are_positive(self):
        """Portfolio value must remain positive (binary sizing, no leverage)."""
        for v in self.result.episode_final_values:
            assert v > 0.0, f"Non-positive portfolio value: {v}"

    def test_agent_is_dqn_agent(self):
        assert isinstance(self.result.agent, DQNAgent)

    def test_agent_episode_counter_advanced(self):
        """decay_epsilon() is called once per episode; counter should equal n_episodes."""
        assert self.result.agent._episode == self.N_EPISODES


# ---------------------------------------------------------------------------
# Epsilon and target network schedule
# ---------------------------------------------------------------------------

class TestTrainingSchedule:
    """Verify epsilon decay and target update schedule fire correctly."""

    @pytest.fixture(autouse=True)
    def _result(self, train_arrays):
        features, close = train_arrays
        self.result = train(
            features, close, ticker="SCHED", n_episodes=3, seed=0
        )

    def test_epsilon_decreases_after_training(self):
        """Epsilon at episode 3 should be less than at episode 0 (1.0)."""
        assert self.result.agent.epsilon < 1.0

    def test_epsilon_stays_above_epsilon_end(self):
        from config import EPSILON_END
        assert self.result.agent.epsilon >= EPSILON_END


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

class TestCheckpointSaving:
    """Verify that checkpoint_path causes a state_dict file to be written."""

    def test_checkpoint_file_created(self, tmp_path, train_arrays):
        features, close = train_arrays
        ckpt = tmp_path / "model.pt"
        train(features, close, ticker="CKPT", n_episodes=1, checkpoint_path=ckpt)
        assert ckpt.exists(), "Checkpoint file was not created."

    def test_checkpoint_loadable(self, tmp_path, train_arrays):
        """State dict saved during training must be loadable into a fresh QNetwork."""
        import torch

        from network import QNetwork

        features, close = train_arrays
        ckpt = tmp_path / "model.pt"
        train(features, close, ticker="CKPT", n_episodes=1, checkpoint_path=ckpt)

        net = QNetwork()
        net.load_state_dict(torch.load(ckpt, weights_only=True))  # must not raise


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Two runs with the same seed should produce identical episode rewards."""

    def test_same_seed_same_rewards(self, train_arrays):
        features, close = train_arrays
        r1 = train(features, close, ticker="REP", n_episodes=2, seed=7)
        r2 = train(features, close, ticker="REP", n_episodes=2, seed=7)
        np.testing.assert_array_equal(r1.episode_rewards, r2.episode_rewards)
