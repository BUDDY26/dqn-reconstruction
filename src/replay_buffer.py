"""
replay_buffer.py — Experience replay memory for the DQN reconstruction.

Design (all values CONFIRMED):
    Capacity:   100,000 transitions  [CONFIRMED]
    Batch size: 64                   [CONFIRMED]
    Sampling:   Uniform random       [ASSUMED: no prioritisation confirmed — standard DQN]
    Structure:  Ring buffer (oldest transitions overwritten when at capacity)

Transitions are stored as NumPy arrays pre-allocated at construction time to
avoid repeated heap allocation during training.  Sampling returns PyTorch
tensors, matching the dtype expectations of agent.py and network.py.

Reference: docs/implementation-plan.md §replay_buffer.py
"""

from __future__ import annotations

import numpy as np
import torch

from config import BATCH_SIZE, OBS_DIM, REPLAY_BUFFER_SIZE


class ReplayBuffer:
    """Fixed-capacity ring buffer storing (s, a, r, s', done) transitions.

    Args:
        capacity: Maximum number of transitions to store.  Default: REPLAY_BUFFER_SIZE.
        obs_dim:  Dimensionality of each observation vector.  Default: OBS_DIM.
    """

    def __init__(
        self,
        capacity: int = REPLAY_BUFFER_SIZE,
        obs_dim: int = OBS_DIM,
    ) -> None:
        self._capacity = capacity
        self._obs_dim = obs_dim

        # Pre-allocate storage arrays.
        self._states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)

        self._ptr = 0   # Write pointer (next slot to overwrite).
        self._size = 0  # Number of valid transitions currently stored.

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition.

        Overwrites the oldest transition once the buffer is at capacity.

        Args:
            state:      Observation at time t, shape (obs_dim,), float32.
            action:     Integer action taken at time t.
            reward:     Scalar reward received after taking action.
            next_state: Observation at time t+1, shape (obs_dim,), float32.
            done:       True if the episode terminated after this transition.
        """
        self._states[self._ptr] = state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr] = done

        # Advance write pointer; wrap around at capacity (ring buffer).
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int = BATCH_SIZE) -> tuple[
        torch.Tensor,  # states      (batch, obs_dim)
        torch.Tensor,  # actions     (batch,)
        torch.Tensor,  # rewards     (batch,)
        torch.Tensor,  # next_states (batch, obs_dim)
        torch.Tensor,  # dones       (batch,) — bool
    ]:
        """Sample a random mini-batch of transitions without replacement.

        Args:
            batch_size: Number of transitions to sample.  Default: BATCH_SIZE.

        Returns:
            Five tensors — states, actions, rewards, next_states, dones.

        Raises:
            ValueError: If the buffer does not contain at least batch_size transitions.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {self._size}."
            )

        indices = np.random.choice(self._size, size=batch_size, replace=False)

        return (
            torch.from_numpy(self._states[indices]),
            torch.from_numpy(self._actions[indices]),
            torch.from_numpy(self._rewards[indices]),
            torch.from_numpy(self._next_states[indices]),
            torch.from_numpy(self._dones[indices]),
        )

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return self._size
