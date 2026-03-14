"""
tests/unit/test_agent.py — Unit tests for src/agent.py.

Verifies:
  - Epsilon schedule: correct values at episode 0, 100, 200.
  - decay_epsilon increments the episode counter.
  - select_action returns a valid integer in [0, n_actions).
  - select_action is fully random at epsilon=1.0.
  - select_action is greedy at epsilon=0.0.
  - train_step returns None when buffer is below batch_size.
  - train_step returns a non-negative float loss when buffer is full.
  - update_target_network copies online weights to target network.
  - store_transition delegates correctly to the replay buffer.
"""

import numpy as np
import pytest
import torch

from agent import DQNAgent
from config import (
    BATCH_SIZE,
    EPSILON_END,
    EPSILON_START,
    N_ACTIONS,
    OBS_DIM,
    TRAINING_EPISODES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(**kwargs) -> DQNAgent:
    """Construct an agent with a fixed seed for reproducibility."""
    return DQNAgent(seed=42, **kwargs)


def _random_state(obs_dim: int = OBS_DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(obs_dim).astype(np.float32)


def _fill_agent_buffer(agent: DQNAgent, n: int):
    """Push n synthetic transitions into the agent's replay buffer."""
    obs_dim = agent._obs_dim
    for i in range(n):
        rng = np.random.default_rng(i)
        state = rng.standard_normal(obs_dim).astype(np.float32)
        next_state = rng.standard_normal(obs_dim).astype(np.float32)
        action = int(rng.integers(0, N_ACTIONS))
        reward = float(rng.standard_normal())
        done = bool(rng.integers(0, 2))
        agent.store_transition(state, action, reward, next_state, done)


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------


class TestEpsilonSchedule:

    def test_epsilon_at_episode_0(self):
        agent = _make_agent()
        assert agent.epsilon == pytest.approx(EPSILON_START)

    def test_epsilon_at_episode_200(self):
        agent = _make_agent()
        for _ in range(TRAINING_EPISODES):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(EPSILON_END)

    def test_epsilon_at_episode_100(self):
        agent = _make_agent()
        for _ in range(100):
            agent.decay_epsilon()
        # Linear decay: 1.0 - 0.00495 * 100 = 0.505
        assert agent.epsilon == pytest.approx(0.505, abs=1e-4)

    def test_epsilon_does_not_go_below_epsilon_end(self):
        agent = _make_agent()
        for _ in range(TRAINING_EPISODES + 50):
            agent.decay_epsilon()
        assert agent.epsilon >= EPSILON_END

    def test_decay_epsilon_increments_episode(self):
        agent = _make_agent()
        assert agent._episode == 0
        agent.decay_epsilon()
        assert agent._episode == 1
        agent.decay_epsilon()
        assert agent._episode == 2

    def test_epsilon_decreases_monotonically(self):
        agent = _make_agent()
        prev = agent.epsilon
        for _ in range(TRAINING_EPISODES):
            agent.decay_epsilon()
            assert agent.epsilon <= prev
            prev = agent.epsilon


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


class TestSelectAction:

    def test_returns_int(self):
        agent = _make_agent()
        action = agent.select_action(_random_state())
        assert isinstance(action, int)

    def test_action_in_valid_range(self):
        agent = _make_agent()
        for i in range(50):
            action = agent.select_action(_random_state(seed=i))
            assert 0 <= action < N_ACTIONS

    def test_greedy_action_is_argmax(self):
        """At epsilon=0.0, the agent must select the argmax Q-value action."""
        agent = _make_agent(epsilon_start=0.0, epsilon_end=0.0)

        state = _random_state(seed=7)
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        with torch.no_grad():
            q_values = agent.online_net(state_t)
        expected_action = int(q_values.argmax(dim=1).item())

        actual_action = agent.select_action(state)
        assert actual_action == expected_action

    def test_random_action_at_epsilon_1(self):
        """At epsilon=1.0, all actions across many trials should appear.

        With 3 actions and 300 trials, the probability that any single action
        never appears is (2/3)^300 ≈ 10^{-53}, negligible.
        """
        agent = _make_agent(epsilon_start=1.0)
        np.random.seed(99)
        seen = set()
        for i in range(300):
            seen.add(agent.select_action(_random_state(seed=i)))
        assert seen == {0, 1, 2}

    def test_state_dtype_does_not_raise(self):
        """select_action must handle float32 numpy arrays without dtype errors."""
        agent = _make_agent()
        state = np.ones(OBS_DIM, dtype=np.float32)
        agent.select_action(state)  # Must not raise.


# ---------------------------------------------------------------------------
# store_transition
# ---------------------------------------------------------------------------


class TestStoreTransition:

    def test_buffer_length_increases(self):
        agent = _make_agent()
        assert len(agent.replay_buffer) == 0
        state = _random_state()
        agent.store_transition(state, 0, 1.0, state, False)
        assert len(agent.replay_buffer) == 1

    def test_multiple_transitions_stored(self):
        agent = _make_agent()
        _fill_agent_buffer(agent, 10)
        assert len(agent.replay_buffer) == 10


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------


class TestTrainStep:

    def test_returns_none_when_buffer_empty(self):
        agent = _make_agent()
        result = agent.train_step()
        assert result is None

    def test_returns_none_when_buffer_below_batch_size(self):
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE - 1)
        result = agent.train_step()
        assert result is None

    def test_returns_float_when_buffer_full(self):
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE)
        result = agent.train_step()
        assert isinstance(result, float)

    def test_loss_is_non_negative(self):
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE)
        loss = agent.train_step()
        assert loss >= 0.0

    def test_multiple_train_steps_run_without_error(self):
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE * 4)
        for _ in range(10):
            agent.train_step()  # Must not raise.

    def test_online_weights_change_after_train_step(self):
        """Training must update at least one parameter of the online network."""
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE)

        # Snapshot online weights before.
        before = {name: param.clone() for name, param in agent.online_net.named_parameters()}

        agent.train_step()

        changed = any(
            not torch.equal(before[name], param)
            for name, param in agent.online_net.named_parameters()
        )
        assert changed, "No online network parameters were updated after train_step."

    def test_target_weights_unchanged_after_train_step(self):
        """train_step must NOT modify the target network."""
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE)

        target_before = {name: param.clone() for name, param in agent.target_net.named_parameters()}

        agent.train_step()

        for name, param in agent.target_net.named_parameters():
            assert torch.equal(
                target_before[name], param
            ), f"Target network parameter '{name}' changed during train_step."


# ---------------------------------------------------------------------------
# update_target_network
# ---------------------------------------------------------------------------


class TestUpdateTargetNetwork:

    def test_target_matches_online_after_update(self):
        agent = _make_agent()
        # Diverge the networks by doing some training.
        _fill_agent_buffer(agent, BATCH_SIZE)
        agent.train_step()

        agent.update_target_network()

        for (name, online_param), (_, target_param) in zip(
            agent.online_net.named_parameters(),
            agent.target_net.named_parameters(),
        ):
            assert torch.equal(
                online_param, target_param
            ), f"Parameter '{name}' differs between online and target after update."

    def test_target_initially_matches_online(self):
        """Fresh agent: target and online should have identical weights."""
        agent = _make_agent()
        for (_, op), (_, tp) in zip(
            agent.online_net.named_parameters(),
            agent.target_net.named_parameters(),
        ):
            assert torch.equal(op, tp)

    def test_target_diverges_after_train_step(self):
        """After training without an update, online and target should differ."""
        agent = _make_agent()
        _fill_agent_buffer(agent, BATCH_SIZE)
        agent.train_step()

        any_different = any(
            not torch.equal(op, tp)
            for op, tp in zip(
                agent.online_net.parameters(),
                agent.target_net.parameters(),
            )
        )
        assert (
            any_different
        ), "Expected online and target to diverge after train_step, but they are still equal."
