"""
utils.py — Shared utility functions for the DQN reconstruction.

Provides:
  - Reproducible seeding (set_seed)
  - Consistent logging configuration (get_logger)

Training utilities (epsilon decay, episode logging, metric helpers) are deferred
to later modules (agent.py, train.py, evaluate.py) as they depend on components
not yet implemented in this first coding pass.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for NumPy and Python's random module.

    PyTorch seeds (torch.manual_seed) are set in agent.py when the network
    is instantiated.  They are deferred here to avoid importing torch before
    it is confirmed as an installed dependency.

    Args:
        seed: Non-negative integer seed value.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative; got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    logging.getLogger(__name__).debug("set_seed: seed=%d applied to numpy and random.", seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Return a named logger with a StreamHandler if none are attached.

    Calling get_logger multiple times with the same name returns the same
    logger instance (Python logging is module-level singleton by name).

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level.  Default: logging.INFO.
        fmt: Log format string.  Default: '[%(levelname)s] %(name)s: %(message)s'.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt or "[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Epsilon schedule (episodic linear decay)
# ---------------------------------------------------------------------------

def epsilon_for_episode(episode: int, epsilon_start: float, epsilon_end: float,
                         total_episodes: int) -> float:
    """Compute the epsilon value for the given episode using linear decay.

    ASSUMED (A-T1): Linear decay over total_episodes episodes.
      epsilon(e) = max(epsilon_end, epsilon_start - rate × e)
      rate = (epsilon_start - epsilon_end) / total_episodes

    This function is a pure helper; the DQN agent (agent.py) calls it to
    determine the exploration probability before each episode.

    Args:
        episode: Current 0-based episode index.
        epsilon_start: Initial epsilon value (e.g. 1.0).
        epsilon_end: Final epsilon value (e.g. 0.01).
        total_episodes: Total number of training episodes (e.g. 200).

    Returns:
        Epsilon value in [epsilon_end, epsilon_start].

    Example:
        >>> epsilon_for_episode(0, 1.0, 0.01, 200)
        1.0
        >>> epsilon_for_episode(100, 1.0, 0.01, 200)
        0.505
        >>> epsilon_for_episode(200, 1.0, 0.01, 200)
        0.01
    """
    rate = (epsilon_start - epsilon_end) / total_episodes
    return max(epsilon_end, epsilon_start - rate * episode)
