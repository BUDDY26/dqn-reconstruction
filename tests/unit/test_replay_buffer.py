"""
tests/unit/test_replay_buffer.py — Unit tests for src/replay_buffer.py.

Verifies:
  - Buffer starts empty and grows correctly.
  - Ring-buffer overwrite behaviour at capacity.
  - sample() returns tensors of the correct shape and dtype.
  - ValueError raised when sampling from an under-full buffer.
  - Stored values round-trip correctly through push/sample.
"""

import numpy as np
import pytest
import torch

from replay_buffer import ReplayBuffer
from config import OBS_DIM, BATCH_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(obs_dim: int = OBS_DIM, seed: int = 0):
    """Return a deterministic (state, action, reward, next_state, done) tuple."""
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(obs_dim).astype(np.float32)
    action = int(rng.integers(0, 3))
    reward = float(rng.standard_normal())
    next_state = rng.standard_normal(obs_dim).astype(np.float32)
    done = bool(rng.integers(0, 2))
    return state, action, reward, next_state, done


def _fill_buffer(buf: ReplayBuffer, n: int, obs_dim: int = OBS_DIM):
    """Push n distinct transitions into buf."""
    for i in range(n):
        buf.push(*_make_transition(obs_dim=obs_dim, seed=i))


# ---------------------------------------------------------------------------
# Construction and length
# ---------------------------------------------------------------------------

class TestReplayBufferConstruction:

    def test_starts_empty(self):
        buf = ReplayBuffer()
        assert len(buf) == 0

    def test_custom_capacity_and_obs_dim(self):
        buf = ReplayBuffer(capacity=50, obs_dim=5)
        assert len(buf) == 0

    def test_len_after_push(self):
        buf = ReplayBuffer()
        buf.push(*_make_transition())
        assert len(buf) == 1

    def test_len_grows_with_pushes(self):
        buf = ReplayBuffer()
        for i in range(10):
            buf.push(*_make_transition(seed=i))
        assert len(buf) == 10

    def test_len_capped_at_capacity(self):
        capacity = 20
        buf = ReplayBuffer(capacity=capacity)
        _fill_buffer(buf, capacity + 5)
        assert len(buf) == capacity


# ---------------------------------------------------------------------------
# Ring-buffer overflow
# ---------------------------------------------------------------------------

class TestRingBufferOverflow:

    def test_ptr_wraps_around(self):
        """After filling and overflowing, the internal pointer should reset."""
        capacity = 10
        buf = ReplayBuffer(capacity=capacity)
        _fill_buffer(buf, capacity + 3)
        # ptr should be 3 (number of overwrites modulo capacity).
        assert buf._ptr == 3

    def test_size_stays_at_capacity(self):
        capacity = 10
        buf = ReplayBuffer(capacity=capacity)
        _fill_buffer(buf, capacity * 2)
        assert len(buf) == capacity

    def test_oldest_entry_overwritten(self):
        """The slot at _ptr before overflow holds a newer transition."""
        capacity = 4
        obs_dim = 2
        buf = ReplayBuffer(capacity=capacity, obs_dim=obs_dim)

        # Fill exactly at capacity with known state values.
        for i in range(capacity):
            state = np.full(obs_dim, float(i), dtype=np.float32)
            buf.push(state, 0, 0.0, state, False)

        # Overwrite slot 0 with a new transition (value=99).
        new_state = np.full(obs_dim, 99.0, dtype=np.float32)
        buf.push(new_state, 0, 0.0, new_state, False)

        # Slot 0 should now contain the new state.
        np.testing.assert_array_equal(buf._states[0], new_state)


# ---------------------------------------------------------------------------
# sample() output shapes and dtypes
# ---------------------------------------------------------------------------

class TestReplayBufferSample:

    @pytest.fixture(autouse=True)
    def _buf(self):
        self.buf = ReplayBuffer()
        _fill_buffer(self.buf, BATCH_SIZE * 2)  # Ensure enough data.

    def test_sample_returns_5_tuple(self):
        result = self.buf.sample(BATCH_SIZE)
        assert len(result) == 5

    def test_states_shape(self):
        states, *_ = self.buf.sample(BATCH_SIZE)
        assert states.shape == (BATCH_SIZE, OBS_DIM)

    def test_actions_shape(self):
        _, actions, *_ = self.buf.sample(BATCH_SIZE)
        assert actions.shape == (BATCH_SIZE,)

    def test_rewards_shape(self):
        _, _, rewards, *_ = self.buf.sample(BATCH_SIZE)
        assert rewards.shape == (BATCH_SIZE,)

    def test_next_states_shape(self):
        _, _, _, next_states, _ = self.buf.sample(BATCH_SIZE)
        assert next_states.shape == (BATCH_SIZE, OBS_DIM)

    def test_dones_shape(self):
        *_, dones = self.buf.sample(BATCH_SIZE)
        assert dones.shape == (BATCH_SIZE,)

    def test_states_dtype_float32(self):
        states, *_ = self.buf.sample(BATCH_SIZE)
        assert states.dtype == torch.float32

    def test_actions_dtype_int64(self):
        _, actions, *_ = self.buf.sample(BATCH_SIZE)
        assert actions.dtype == torch.int64

    def test_rewards_dtype_float32(self):
        _, _, rewards, *_ = self.buf.sample(BATCH_SIZE)
        assert rewards.dtype == torch.float32

    def test_dones_dtype_bool(self):
        *_, dones = self.buf.sample(BATCH_SIZE)
        assert dones.dtype == torch.bool

    def test_all_tensors(self):
        result = self.buf.sample(BATCH_SIZE)
        for t in result:
            assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# ValueError when buffer is under-full
# ---------------------------------------------------------------------------

class TestReplayBufferSampleError:

    def test_raises_when_empty(self):
        buf = ReplayBuffer()
        with pytest.raises(ValueError):
            buf.sample(BATCH_SIZE)

    def test_raises_when_below_batch_size(self):
        buf = ReplayBuffer()
        _fill_buffer(buf, BATCH_SIZE - 1)
        with pytest.raises(ValueError):
            buf.sample(BATCH_SIZE)

    def test_no_error_at_exactly_batch_size(self):
        buf = ReplayBuffer()
        _fill_buffer(buf, BATCH_SIZE)
        result = buf.sample(BATCH_SIZE)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Round-trip value correctness
# ---------------------------------------------------------------------------

class TestReplayBufferValues:

    def test_stored_state_recoverable(self):
        """A specific state pushed into the buffer should be recoverable via sample
        when the buffer contains only that one transition (batch_size=1)."""
        obs_dim = OBS_DIM
        buf = ReplayBuffer(obs_dim=obs_dim)
        known_state = np.arange(obs_dim, dtype=np.float32)
        buf.push(known_state, 1, 0.5, np.zeros(obs_dim, dtype=np.float32), False)

        states, actions, rewards, _, dones = buf.sample(batch_size=1)

        np.testing.assert_array_almost_equal(states.numpy()[0], known_state)
        assert actions.item() == 1
        assert rewards.item() == pytest.approx(0.5)
        assert dones.item() is False

    def test_done_true_stored_correctly(self):
        buf = ReplayBuffer()
        state = np.zeros(OBS_DIM, dtype=np.float32)
        buf.push(state, 0, 1.0, state, done=True)
        *_, dones = buf.sample(batch_size=1)
        assert dones.item() is True
