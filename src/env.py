"""
env.py — Custom single-asset trading environment for the DQN reconstruction.

This environment implements the gymnasium.Env interface over pre-processed daily
price feature arrays.  It is designed for one asset at a time; five independent
instances are created for the five-asset experiment.

CONFIRMED behaviours (sourced from paper + report):
  - Action space: Discrete(3) — buy / hold / sell
  - Assets are traded independently (inferred from scalar action space)

ASSUMED behaviours (declared in docs/adr/ADR-002-reconstruction-assumptions.md):
  A-D4  Per-asset independent environment (one env instance per ticker)
  A-E1  Reward = position × daily_return, where position ∈ {0, 1}
  A-E2  Single-timestep observation (no rolling window); obs shape = (OBS_DIM,)
  A-E3  Initial capital = $10,000
  A-E4  Binary all-in / all-out position sizing (no fractional sizing)
  A-E6  Episode = full traversal of the data sequence (no sliding window or random start)
  A-E7  Terminal condition = last trading day of the data sequence reached
  A-E8  Zero transaction costs (frictionless execution model)

Reward formulation (A-E1):
  At step t, after the agent takes action a_t:
    position_t+1 ∈ {0, 1}  (flat or long)
    daily_return_t = (close[t+1] - close[t]) / close[t]
    reward = position_t+1 × daily_return_t

  The agent earns the next-day return proportional to its updated position.
  This is a dense signal — zero only when the agent is flat (position=0).

Episode boundaries:
  - reset(): step=0, returns features[0] as initial observation.
  - step() at step t: uses close[t] and close[t+1] to compute reward.
  - terminated when step advances to len(features) - 1 (last valid index).
  - The episode contains exactly len(features) - 1 transitions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from config import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    INITIAL_CAPITAL,
    N_ACTIONS,
    OBS_DIM,
    TRANSACTION_COST,
)

logger = logging.getLogger(__name__)


class TradingEnv(gymnasium.Env):
    """Single-asset DQN trading environment.

    The environment wraps a pre-scaled feature array and the corresponding raw
    close price array.  It does not perform any data loading or preprocessing;
    those responsibilities belong to data.py.

    Args:
        features: Scaled feature array of shape (n_days, OBS_DIM).
                  Produced by data.apply_scaler().  Row t is the observation
                  returned at the start of step t.
        close_prices: Raw (unscaled) closing prices, shape (n_days,).
                      Aligned 1-to-1 with rows in `features`.
                      Used for reward computation (A-E1) and portfolio tracking.
        initial_capital: Starting cash balance in USD.  Default: config.INITIAL_CAPITAL.
                         Does not affect any scale-invariant metric (ROI, Sharpe, MDD, Calmar).
        ticker: Optional ticker label for logging and info dict.  Not used in computation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        initial_capital: float = INITIAL_CAPITAL,
        ticker: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Validate inputs.
        if features.ndim != 2 or features.shape[1] != OBS_DIM:
            raise ValueError(
                f"features must have shape (n_days, {OBS_DIM}); "
                f"got {features.shape}."
            )
        if close_prices.ndim != 1 or len(close_prices) != len(features):
            raise ValueError(
                f"close_prices must be a 1-D array of length {len(features)} "
                f"(same as features rows); got shape {close_prices.shape}."
            )
        if len(features) < 2:
            raise ValueError(
                f"Environment requires at least 2 rows in features to form one transition; "
                f"got {len(features)}."
            )
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive; got {initial_capital}.")

        self._features = features.astype(np.float32)
        self._close_prices = close_prices.astype(np.float64)
        self._initial_capital = initial_capital
        self._ticker = ticker or "unknown"

        # Number of valid steps in one episode: n_days - 1
        # (each step uses close[t] and close[t+1]; the last step uses close[-2] and close[-1]).
        self._n_steps: int = len(features) - 1

        # Define spaces.
        # Observation: 11-feature vector, unbounded (StandardScaler can produce any value).
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Action: 0=buy, 1=hold, 2=sell  [CONFIRMED]
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Internal state (initialised in reset()).
        self._step: int = 0
        self._position: int = 0        # 0=flat, 1=long  [ASSUMED: A-E4]
        self._cash: float = initial_capital
        self._shares: float = 0.0
        self._episode_return: float = 0.0

    # -----------------------------------------------------------------------
    # gymnasium.Env interface
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the start of the data sequence.

        ASSUMED (A-E6): Each episode starts at the first trading day.
        ASSUMED (A-E3): Initial capital is config.INITIAL_CAPITAL.
        ASSUMED (A-E4): Portfolio begins flat (no shares held).

        Args:
            seed: Optional RNG seed (passed to super for reproducibility).
            options: Unused; accepted for API compatibility.

        Returns:
            observation: Feature vector for day 0, shape (OBS_DIM,).
            info: Initial portfolio state.
        """
        super().reset(seed=seed)

        self._step = 0
        self._position = 0      # Start flat.  [ASSUMED: A-E4]
        self._cash = self._initial_capital
        self._shares = 0.0
        self._episode_return = 0.0

        observation = self._features[0]
        info = self._build_info()

        logger.debug("[%s] reset — %d steps per episode.", self._ticker, self._n_steps)
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one trading day.

        Execution order:
          1. Execute trade based on action (A-E4: binary all-in/all-out).
          2. Compute reward using updated position and next-day close price (A-E1).
          3. Advance timestep.
          4. Return next observation, reward, terminated flag, and info.

        Reward (A-E1):
          reward = position × (close[t+1] - close[t]) / close[t]
          where position is the UPDATED position after the action.

        Transaction costs (A-E8): zero.

        Args:
            action: Integer in {0=buy, 1=hold, 2=sell}.

        Returns:
            observation: Feature vector for the next day (step + 1), shape (OBS_DIM,).
            reward: Float reward for the current transition.
            terminated: True when the last trading day has been reached.
            truncated: Always False (no time limit applied).
            info: Portfolio state after the step.

        Raises:
            AssertionError: If the episode has already terminated.
        """
        assert self._step < self._n_steps, (
            f"step() called after episode termination (step={self._step}, "
            f"n_steps={self._n_steps}). Call reset() first."
        )

        current_close = self._close_prices[self._step]
        next_close = self._close_prices[self._step + 1]

        # ── 1. Execute trade ──────────────────────────────────────────────
        self._execute_action(action, current_close)

        # ── 2. Compute reward (A-E1) ──────────────────────────────────────
        daily_return = (next_close - current_close) / current_close
        reward = float(self._position * daily_return)
        # Transaction cost term: 0.0 (A-E8). Included explicitly so the formula is
        # visible in code and easy to update if assumption A-E8 is revised.
        reward -= TRANSACTION_COST

        self._episode_return += reward

        # ── 3. Advance timestep ───────────────────────────────────────────
        self._step += 1
        terminated = self._step >= self._n_steps

        # ── 4. Observation and info ───────────────────────────────────────
        observation = self._features[self._step]
        info = self._build_info()

        if terminated:
            logger.debug(
                "[%s] episode done — total return: %.4f", self._ticker, self._episode_return
            )

        return observation, reward, terminated, False, info

    def render(self) -> None:
        """Rendering is not implemented in this reconstruction pass."""
        pass

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _execute_action(self, action: int, current_close: float) -> None:
        """Update position, cash, and shares based on the chosen action.

        ASSUMED (A-E4): Binary all-in / all-out position sizing.
          BUY  — invest all available cash at current_close; noop if already long.
          SELL — liquidate all shares at current_close; noop if already flat.
          HOLD — no trade, position unchanged.

        ASSUMED (A-E8): No transaction cost applied.

        Args:
            action: Integer action in {ACTION_BUY, ACTION_HOLD, ACTION_SELL}.
            current_close: Raw closing price for the current timestep.
        """
        if action == ACTION_BUY:
            if self._position == 0:
                # Buy: invest all cash.
                self._shares = self._cash / current_close
                self._cash = 0.0
                self._position = 1
            # If already long, BUY is treated as HOLD (cannot increase position).

        elif action == ACTION_SELL:
            if self._position == 1:
                # Sell: liquidate all shares.
                self._cash = self._shares * current_close
                self._shares = 0.0
                self._position = 0
            # If already flat, SELL is treated as HOLD (no short-selling).

        # ACTION_HOLD: no change to position, cash, or shares.

    def _build_info(self) -> Dict[str, Any]:
        """Build the info dictionary for the current state.

        Portfolio value when long = cash + shares × close[step].
        Portfolio value when flat = cash.
        """
        close = self._close_prices[self._step]
        portfolio_value = self._cash + self._shares * close
        return {
            "step": self._step,
            "position": self._position,
            "cash": self._cash,
            "shares": self._shares,
            "portfolio_value": portfolio_value,
            "close": close,
            "ticker": self._ticker,
        }

    # -----------------------------------------------------------------------
    # Properties (read-only access to internal state for tests and logging)
    # -----------------------------------------------------------------------

    @property
    def position(self) -> int:
        """Current position: 0=flat, 1=long."""
        return self._position

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value in USD."""
        close = self._close_prices[self._step]
        return self._cash + self._shares * close

    @property
    def n_steps(self) -> int:
        """Number of steps per episode (= len(features) - 1)."""
        return self._n_steps
