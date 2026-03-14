"""
train.py — Training loop entry point for the DQN reconstruction.

Runs the outer episode loop for one asset.  Each episode is a full traversal
of the training data sequence (A-E6).  The caller is responsible for loading
and preprocessing data (data.py) and can loop over assets externally.

Training structure (from docs/implementation-plan.md):
    for episode in range(TRAINING_EPISODES):       # 200 confirmed
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)    # epsilon-greedy
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) >= BATCH_SIZE:
                agent.learn(replay_buffer.sample(BATCH_SIZE))
            state = next_state
        agent.decay_epsilon()

Key assumption references:
    A-T1  Linear epsilon decay per episode (utils.epsilon_for_episode)
    A-T2  Hard target network copy (not soft update)
    A-T3  Target update frequency = 10 episodes
    A-E6  Episode = full traversal of the data (no random start)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from agent import DQNAgent
from config import TARGET_UPDATE_FREQ, TRAINING_EPISODES
from env import TradingEnv
from utils import get_logger, set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Container for training outcomes from a single-asset training run.

    Attributes:
        agent:                 The trained DQNAgent instance (online network contains
                               the final learned weights).
        ticker:                Ticker symbol used for this training run.
        episode_rewards:       Sum of step rewards per episode.  Approximates the
                               episodic return (sum of position * daily_return).
        episode_final_values:  Portfolio value at the end of each episode, in USD.
        n_episodes:            Number of episodes actually completed.
    """
    agent: DQNAgent
    ticker: str
    episode_rewards: list[float]
    episode_final_values: list[float]
    n_episodes: int


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(env: TradingEnv, agent: DQNAgent) -> tuple[float, float]:
    """Run one complete episode and return (total_reward, final_portfolio_value).

    Interaction loop (one step per trading day):
      1. Agent selects action (epsilon-greedy).
      2. Environment steps: returns next_obs, reward, terminated, truncated, info.
      3. Transition is stored in agent's replay buffer.
      4. Agent performs one learning step (skipped if buffer is too small).
      5. Repeat until terminated.

    Args:
        env:   Freshly reset TradingEnv (reset is called internally here).
        agent: DQNAgent instance; epsilon and buffer state carry over between episodes.

    Returns:
        total_reward:         Sum of step rewards over the episode.
        final_portfolio_value: Portfolio value in USD at the terminal step.
    """
    state, info = env.reset()
    terminated = False
    total_reward = 0.0

    while not terminated:
        action = agent.select_action(state)
        next_state, reward, terminated, _, info = env.step(action)

        # Store transition; train_step returns None if buffer is under-full.
        agent.store_transition(state, action, float(reward), next_state, terminated)
        agent.train_step()

        state = next_state
        total_reward += reward

    return total_reward, float(info["portfolio_value"])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    features: np.ndarray,
    close_prices: np.ndarray,
    ticker: str = "unknown",
    n_episodes: int = TRAINING_EPISODES,
    seed: Optional[int] = None,
    checkpoint_path: Optional[Path | str] = None,
) -> TrainResult:
    """Train a DQN agent on one asset's feature array.

    Creates a fresh TradingEnv and DQNAgent, then runs the episode loop for
    n_episodes episodes.  Target network is updated every TARGET_UPDATE_FREQ
    episodes (hard copy, ASSUMED: A-T2/A-T3).  Epsilon decays linearly after
    each episode (ASSUMED: A-T1).

    Args:
        features:        Scaled feature array of shape (n_days, OBS_DIM), float32.
                         Produced by data.prepare_arrays().
        close_prices:    Raw close prices of shape (n_days,), float64.
                         Aligned with features; used by TradingEnv for rewards.
        ticker:          Ticker label for logging.
        n_episodes:      Number of training episodes.  Default: TRAINING_EPISODES (200).
        seed:            Optional integer seed for reproducibility.  Seeds NumPy,
                         Python random, and PyTorch.
        checkpoint_path: If provided, saves the online network's state_dict to this
                         path after training completes.  Basic model persistence only.

    Returns:
        TrainResult with the trained agent and per-episode statistics.
    """
    if seed is not None:
        set_seed(seed)

    env = TradingEnv(features=features, close_prices=close_prices, ticker=ticker)
    agent = DQNAgent(seed=seed)

    episode_rewards: list[float] = []
    episode_final_values: list[float] = []

    logger.info("[%s] Starting training: %d episodes.", ticker, n_episodes)

    for episode in range(n_episodes):
        ep_reward, final_value = _run_episode(env, agent)
        episode_rewards.append(ep_reward)
        episode_final_values.append(final_value)

        # Epsilon decays linearly over episodes (ASSUMED: A-T1).
        agent.decay_epsilon()

        # Hard target network copy every TARGET_UPDATE_FREQ episodes (ASSUMED: A-T2/A-T3).
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        if (episode + 1) % 20 == 0 or episode == 0:
            logger.info(
                "[%s] episode %3d/%d  reward=%.4f  portfolio=%.2f  ε=%.4f",
                ticker, episode + 1, n_episodes,
                ep_reward, final_value, agent.epsilon,
            )

    logger.info(
        "[%s] Training complete. Final ε=%.4f. Last portfolio value=%.2f.",
        ticker, agent.epsilon, episode_final_values[-1],
    )

    # Optional checkpoint — basic model persistence (implementation-plan.md §train.py).
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.online_net.state_dict(), path)
        logger.info("[%s] Checkpoint saved → %s", ticker, path)

    return TrainResult(
        agent=agent,
        ticker=ticker,
        episode_rewards=episode_rewards,
        episode_final_values=episode_final_values,
        n_episodes=n_episodes,
    )
