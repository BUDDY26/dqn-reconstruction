"""
config.py — Hyperparameter registry for the DQN reconstruction.

Every constant in this module is annotated with its source:
  [CONFIRMED]     Value taken directly from the paper or final project report.
                  Must not be changed without an ADR explaining the deviation.
  [ASSUMED: A-Xn] Reconstruction assumption declared in
                  docs/adr/ADR-002-reconstruction-assumptions.md.
                  Must not be changed without annotating the corresponding Category 3 entry
                  in docs/evidence-ledger.md and creating a new ADR if architecturally significant.
  [DERIVED]       Computed from confirmed and assumed values; not independently confirmed.

Reference: docs/adr/ADR-002-reconstruction-assumptions.md
"""

from typing import Final, List

# ---------------------------------------------------------------------------
# Assets and Date Ranges
# ---------------------------------------------------------------------------

ASSETS: Final[List[str]] = ["AAPL", "INTC", "META", "TQQQ", "TSLA"]
"""[CONFIRMED] The five equity assets used in training and evaluation."""

START_DATE: Final[str] = "2017-01-01"
"""[CONFIRMED] First date of the full data download range."""

END_DATE: Final[str] = "2024-06-01"
"""[CONFIRMED] Last date of the full data download range (inclusive)."""

TRAIN_END: Final[str] = "2022-12-31"
"""[ASSUMED: A-D3] Last day of the training period.
The paper confirms the test period starts 2023-01-01; this date follows as the training cutoff.
No validation split is used (absence of documentation treated as evidence of none).
"""

TEST_START: Final[str] = "2023-01-01"
"""[CONFIRMED] First date of the held-out test period."""

TEST_END: Final[str] = "2024-01-01"
"""[CONFIRMED] Last date of the held-out test period (inclusive upper bound)."""

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

PCT_CHANGE_PERIOD: Final[int] = 1
"""[ASSUMED: A-D1] Period for percentage change features (1-day).
Applied to each of the five OHLCV columns. 1-day is the default for daily-bar data.
"""

FEATURE_COLUMNS: Final[List[str]] = [
    "open",
    "high",
    "low",
    "close",
    "volume",  # [CONFIRMED] raw OHLCV
    "vwap",  # [CONFIRMED] VWAP (approximated — A-D2)
    "pct_open",
    "pct_high",
    "pct_low",  # [ASSUMED: A-D1] 1-day pct changes
    "pct_close",
    "pct_volume",  # [ASSUMED: A-D1]
]
"""Canonical ordered list of feature columns produced by data.build_features().
The network input layer is indexed by this order. Do not reorder without updating OBS_DIM
and regenerating any cached features.
"""

OBS_DIM: Final[int] = 11
"""[DERIVED] Observation space dimensionality.
= 5 raw OHLCV + 1 VWAP + 5 pct_changes = 11.
Derived from A-D1 (pct_change period), A-D2 (VWAP), A-E2 (single timestep), A-D4 (per-asset).
Any change to FEATURE_COLUMNS must be reflected here.
"""

assert len(FEATURE_COLUMNS) == OBS_DIM, (
    f"FEATURE_COLUMNS length ({len(FEATURE_COLUMNS)}) does not match OBS_DIM ({OBS_DIM}). "
    "Update both together."
)

# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

ACTION_BUY: Final[int] = 0
"""[CONFIRMED] Integer index for the buy action."""

ACTION_HOLD: Final[int] = 1
"""[CONFIRMED] Integer index for the hold action."""

ACTION_SELL: Final[int] = 2
"""[CONFIRMED] Integer index for the sell action."""

N_ACTIONS: Final[int] = 3
"""[CONFIRMED] Number of discrete actions: buy / hold / sell."""

# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------

HIDDEN_SIZE: Final[int] = 256
"""[CONFIRMED] Width of each hidden layer in the Q-network (two layers of this size)."""

# ---------------------------------------------------------------------------
# Training Hyperparameters (all CONFIRMED unless noted)
# ---------------------------------------------------------------------------

LEARNING_RATE: Final[float] = 0.0003
"""[CONFIRMED] Optimizer learning rate. Value is a strong contextual indicator of Adam (A-T4)."""

GAMMA: Final[float] = 0.99
"""[CONFIRMED] Discount factor for future rewards."""

EPSILON_START: Final[float] = 1.0
"""[CONFIRMED] Initial epsilon for epsilon-greedy exploration."""

EPSILON_END: Final[float] = 0.01
"""[CONFIRMED] Final epsilon after full decay."""

EPSILON_DECAY_RATE: Final[float] = (EPSILON_START - EPSILON_END) / 200
"""[ASSUMED: A-T1] Per-episode linear decay rate for epsilon.
= (1.0 - 0.01) / 200 = 0.00495.
Epsilon reaches EPSILON_END exactly at episode 200 (the final episode).
"""

BATCH_SIZE: Final[int] = 64
"""[CONFIRMED] Mini-batch size sampled from the replay buffer per learning step."""

REPLAY_BUFFER_SIZE: Final[int] = 100_000
"""[CONFIRMED] Maximum number of transitions stored in the experience replay buffer."""

TRAINING_EPISODES: Final[int] = 200
"""[CONFIRMED] Total number of training episodes (each = full traversal of training data)."""

TARGET_UPDATE_FREQ: Final[int] = 10
"""[ASSUMED: A-T2/A-T3] Hard-copy target network update frequency in episodes.
Every 10 episodes ≈ every 15,000 steps at ~1,500 steps per training episode.
Analogous to the original DQN paper's 10,000-step update frequency.
"""

# ---------------------------------------------------------------------------
# Environment Parameters
# ---------------------------------------------------------------------------

INITIAL_CAPITAL: Final[float] = 10_000.0
"""[ASSUMED: A-E3] Starting portfolio cash in USD.
All four evaluation metrics (ROI, Sharpe, MDD, Calmar) are scale-invariant under
proportional sizing, so this value does not affect any reported metric.
"""

TRANSACTION_COST: Final[float] = 0.0
"""[ASSUMED: A-E8] Transaction cost rate applied per trade.
Zero — frictionless model. Not mentioned in paper or report; absence of specification
is the standard basis for a frictionless assumption in academic simulations.
Known limitation: overestimates achievable returns for TQQQ (3× leveraged ETF).
"""
