"""
agent.py — DQN agent for the DQN reconstruction.

Implements:
    - Epsilon-greedy action selection (linear decay — ASSUMED: A-T1)
    - Experience storage (delegates to ReplayBuffer)
    - TD learning step with MSE loss (ASSUMED: A-T5)
    - Hard target network update every N episodes (ASSUMED: A-T2/A-T3)

Hyperparameter sources:
    LEARNING_RATE   0.0003  [CONFIRMED]
    GAMMA           0.99    [CONFIRMED]
    EPSILON_START   1.0     [CONFIRMED]
    EPSILON_END     0.01    [CONFIRMED]
    EPSILON_DECAY   linear  [ASSUMED: A-T1]
    OPTIMIZER       Adam    [ASSUMED: A-T4]
    LOSS            MSE     [ASSUMED: A-T5]
    TARGET_UPDATE   10 eps  [ASSUMED: A-T2/A-T3]

Reference: docs/adr/ADR-002-reconstruction-assumptions.md
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import (
    BATCH_SIZE,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    LEARNING_RATE,
    N_ACTIONS,
    OBS_DIM,
    REPLAY_BUFFER_SIZE,
    TRAINING_EPISODES,
)
from network import QNetwork
from replay_buffer import ReplayBuffer
from utils import epsilon_for_episode, get_logger

logger = get_logger(__name__)


class DQNAgent:
    """DQN agent with epsilon-greedy policy and experience replay.

    Maintains an online Q-network (updated every train_step) and a target
    Q-network (copied from online every TARGET_UPDATE_FREQ episodes via
    update_target_network).

    The agent does not own the training loop.  train.py is responsible for
    calling select_action / store_transition / train_step / decay_epsilon /
    update_target_network at the appropriate times.

    Args:
        obs_dim:          Observation dimensionality.         Default: OBS_DIM.
        n_actions:        Number of discrete actions.         Default: N_ACTIONS.
        learning_rate:    Adam learning rate.                 Default: LEARNING_RATE.
        gamma:            Discount factor.                    Default: GAMMA.
        epsilon_start:    Initial epsilon.                    Default: EPSILON_START.
        epsilon_end:      Final epsilon.                      Default: EPSILON_END.
        total_episodes:   Total training episodes for decay.  Default: TRAINING_EPISODES.
        batch_size:       Mini-batch size for train_step.     Default: BATCH_SIZE.
        buffer_capacity:  Replay buffer capacity.             Default: REPLAY_BUFFER_SIZE.
        seed:             Optional RNG seed for torch.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        total_episodes: int = TRAINING_EPISODES,
        batch_size: int = BATCH_SIZE,
        buffer_capacity: int = REPLAY_BUFFER_SIZE,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)

        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._gamma = gamma
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._total_episodes = total_episodes
        self._batch_size = batch_size

        # Q-networks (online updated every step; target updated every N episodes).
        self.online_net = QNetwork(obs_dim, n_actions)
        self.target_net = QNetwork(obs_dim, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly.

        # Optimizer — Adam (ASSUMED: A-T4).
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # Experience replay buffer.
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim)

        # Episode counter drives the epsilon schedule.
        self._episode: int = 0

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current epsilon value based on the linear decay schedule (A-T1)."""
        return epsilon_for_episode(
            self._episode,
            self._epsilon_start,
            self._epsilon_end,
            self._total_episodes,
        )

    def decay_epsilon(self) -> None:
        """Advance the episode counter, reducing epsilon by one step.

        Call once at the end of each training episode.
        """
        self._episode += 1

    # ------------------------------------------------------------------
    # Interaction with environment
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Choose an action using the epsilon-greedy policy.

        With probability epsilon, selects a uniformly random action.
        Otherwise, selects the action with the highest Q-value (greedy).

        Args:
            state: Observation vector, shape (obs_dim,), dtype float32.

        Returns:
            Integer action in [0, n_actions).
        """
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self._n_actions))

        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0)  # (1, obs_dim)
            q_values = self.online_net(state_t)                      # (1, n_actions)
        self.online_net.train()

        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer.

        Args:
            state:      Observation at time t.
            action:     Action taken at time t.
            reward:     Reward received.
            next_state: Observation at time t+1.
            done:       True if the episode ended after this transition.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def train_step(self) -> float | None:
        """Sample a mini-batch and perform one gradient update on the online network.

        Skips the update and returns None if the replay buffer does not yet
        contain enough transitions (< batch_size).

        TD target (standard DQN, no double-DQN):
            y = r + γ · max_a Q_target(s', a) · (1 - done)

        Loss: MSE(Q_online(s)[a], y)  [ASSUMED: A-T5]

        Returns:
            Scalar loss value if an update was performed; None otherwise.
        """
        if len(self.replay_buffer) < self._batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self._batch_size
        )

        # Q-values for actions actually taken: shape (batch,).
        self.online_net.train()
        q_pred = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD targets computed under the frozen target network.
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values  # (batch,)
            # Zero out future reward for terminal transitions.
            q_target = rewards + self._gamma * q_next * (1.0 - dones.float())

        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target_network(self) -> None:
        """Hard-copy online network weights to the target network.

        Called by train.py every TARGET_UPDATE_FREQ episodes (ASSUMED: A-T2/A-T3).
        """
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Target network updated at episode %d.", self._episode)
