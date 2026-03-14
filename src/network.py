"""
network.py — Q-Network architecture for the DQN reconstruction.

Architecture (all values CONFIRMED from paper/report):
    Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 3)

Reference: docs/adr/ADR-002-reconstruction-assumptions.md (derived obs_dim = 11)
"""

import torch
import torch.nn as nn

from config import HIDDEN_SIZE, N_ACTIONS, OBS_DIM


class QNetwork(nn.Module):
    """Two-hidden-layer Q-network mapping observations to Q-values.

    Architecture:
        Linear(obs_dim, 256) → ReLU
        Linear(256, 256)     → ReLU
        Linear(256, n_actions)

    All layer widths are confirmed from the paper.  obs_dim defaults to OBS_DIM (11),
    which is a derived quantity documented in evidence-ledger.md and ADR-002.

    Args:
        obs_dim:   Dimensionality of the input observation vector.  Default: OBS_DIM.
        n_actions: Number of discrete actions (buy / hold / sell).   Default: N_ACTIONS.
        hidden_size: Width of each hidden layer.                      Default: HIDDEN_SIZE.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of observations.

        Args:
            x: Tensor of shape (batch_size, obs_dim), dtype float32.

        Returns:
            Tensor of shape (batch_size, n_actions) containing raw Q-values.
        """
        return self.net(x)
