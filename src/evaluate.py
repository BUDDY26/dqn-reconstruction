"""
evaluate.py — Evaluation metrics for the DQN reconstruction.

Runs the trained agent on the test period in fully greedy mode (ε=0) and
computes the four confirmed evaluation metrics:

    ROI            — Total return over the test period              [CONFIRMED]
    Sharpe Ratio   — Annualized risk-adjusted return                [CONFIRMED]
    Max Drawdown   — Largest peak-to-trough portfolio decline       [CONFIRMED]
    Calmar Ratio   — Annualized return divided by max drawdown      [CONFIRMED]

Metric implementation assumptions (not specified in paper; standard finance):
    A-Ev1  Risk-free rate = 0  (standard for short academic backtests)
    A-Ev2  Annualization factor = 252 trading days per year
    A-Ev3  Sharpe uses sample standard deviation (ddof=1); returns 0 if std=0
    A-Ev4  Calmar returns 0 when max_drawdown=0 (undefined ratio treated conservatively)
    A-Ev5  Portfolio value series starts at INITIAL_CAPITAL (reset value) and
           includes one entry per step (n_steps + 1 entries total)

All assumptions are consistent with ADR-002 declared assumptions and do not
introduce new architectural decisions.

Reference: docs/implementation-plan.md §evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agent import DQNAgent
from env import TradingEnv
from network import QNetwork
from utils import get_logger

logger = get_logger(__name__)

# Annualization factor (A-Ev2): standard 252 trading days per year.
_TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Container for evaluation outcomes from a single-asset evaluation run.

    Attributes:
        ticker:          Ticker symbol evaluated.
        roi:             Total return over the test period as a decimal fraction.
                         e.g., 0.15 = 15% gain, -0.10 = 10% loss.
        sharpe:          Annualized Sharpe ratio (risk-free rate = 0, A-Ev1/A-Ev2).
        max_drawdown:    Maximum peak-to-trough portfolio decline as a positive fraction.
                         e.g., 0.20 = a 20% drawdown from peak.
        calmar:          Annualized return divided by max_drawdown.
                         Returns 0.0 when max_drawdown = 0 (A-Ev4).
        portfolio_values: Array of portfolio values in USD, one entry per step
                          (length = n_test_steps + 1, includes the initial value).
    """

    ticker: str
    roi: float
    sharpe: float
    max_drawdown: float
    calmar: float
    portfolio_values: np.ndarray


# ---------------------------------------------------------------------------
# Pure metric functions (public; tested independently)
# ---------------------------------------------------------------------------


def compute_roi(portfolio_values: np.ndarray) -> float:
    """Compute total return over the evaluation period.

    ROI = (final_value - initial_value) / initial_value

    Args:
        portfolio_values: Array of portfolio values in USD, at least 2 entries.

    Returns:
        ROI as a decimal fraction (e.g., 0.15 for a 15% gain).
    """
    return float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0])


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Compute the maximum peak-to-trough decline in portfolio value.

    Max Drawdown = max over all t of (peak_t - value_t) / peak_t

    where peak_t = max(portfolio_values[:t+1]).

    The result is a positive fraction (e.g., 0.20 for a 20% drawdown).
    A monotonically increasing portfolio has MDD = 0.

    Args:
        portfolio_values: Array of portfolio values in USD, at least 1 entry.

    Returns:
        Maximum drawdown in [0, 1].
    """
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    return float(np.max(drawdowns))


def compute_sharpe(
    portfolio_values: np.ndarray,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute the annualized Sharpe ratio with zero risk-free rate.

    Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(periods_per_year)

    where daily_returns[t] = portfolio_values[t+1] / portfolio_values[t] - 1.

    ASSUMED (A-Ev1): risk-free rate = 0.
    ASSUMED (A-Ev2): annualization factor = 252.
    ASSUMED (A-Ev3): sample std (ddof=1); returns 0 if fewer than 2 returns
                     or if std = 0 (all returns identical).

    Args:
        portfolio_values: Array of portfolio values in USD, at least 2 entries.
        periods_per_year: Trading periods per year.  Default: 252.

    Returns:
        Annualized Sharpe ratio.  Returns 0.0 if std is zero or undefined.
    """
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    if len(daily_returns) < 2:
        # Cannot compute sample std with fewer than 2 observations.
        return 0.0

    std = float(np.std(daily_returns, ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0

    return float(np.mean(daily_returns) / std * np.sqrt(periods_per_year))


def compute_calmar(
    portfolio_values: np.ndarray,
    periods_per_year: int = _TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute the Calmar ratio: annualized return divided by maximum drawdown.

    Calmar = annualized_return / max_drawdown

    where annualized_return = (1 + ROI)^(periods_per_year / n_days) - 1.

    ASSUMED (A-Ev4): returns 0.0 when max_drawdown = 0 (the ratio is otherwise
    undefined or infinite; 0.0 is the conservative representation).

    Args:
        portfolio_values: Array of portfolio values in USD, at least 2 entries.
        periods_per_year: Trading periods per year.  Default: 252.

    Returns:
        Calmar ratio.  Returns 0.0 if max_drawdown = 0 or n_days = 0.
    """
    n_days = len(portfolio_values) - 1
    if n_days <= 0:
        return 0.0

    roi = compute_roi(portfolio_values)
    mdd = compute_max_drawdown(portfolio_values)

    if mdd == 0.0:
        # No drawdown: ratio is undefined (A-Ev4).
        return 0.0

    annualized_return = float((1.0 + roi) ** (periods_per_year / n_days) - 1.0)
    return float(annualized_return / mdd)


# ---------------------------------------------------------------------------
# Greedy action selection (evaluation only — no exploration)
# ---------------------------------------------------------------------------


def _greedy_action(net: QNetwork, state: np.ndarray) -> int:
    """Select the action with the highest Q-value (ε=0 policy).

    This bypasses the DQNAgent's epsilon schedule to enforce fully greedy
    evaluation, as specified in the implementation plan.

    Args:
        net:   Trained QNetwork (agent.online_net).
        state: Observation vector, shape (OBS_DIM,), dtype float32.

    Returns:
        Integer action in [0, n_actions).
    """
    device = next(net.parameters()).device
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1, obs_dim)
        return int(net(state_t).argmax(dim=1).item())


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate(
    agent: DQNAgent,
    features: np.ndarray,
    close_prices: np.ndarray,
    ticker: str = "unknown",
) -> EvalResult:
    """Evaluate a trained agent on the test feature array in greedy mode (ε=0).

    Runs a single episode through the test environment, recording portfolio
    value at each step.  Computes the four confirmed evaluation metrics from
    the resulting portfolio value series.

    Args:
        agent:        Trained DQNAgent.  Only agent.online_net is used.
        features:     Scaled test feature array, shape (n_days, OBS_DIM), float32.
        close_prices: Raw test close prices, shape (n_days,), float64.
        ticker:       Ticker label for logging and the result.

    Returns:
        EvalResult with all four metrics and the full portfolio value series.
    """
    env = TradingEnv(features=features, close_prices=close_prices, ticker=ticker)
    net = agent.online_net
    net.eval()

    portfolio_values: list[float] = []

    obs, info = env.reset()
    portfolio_values.append(float(info["portfolio_value"]))

    terminated = False
    while not terminated:
        action = _greedy_action(net, obs)
        obs, _, terminated, _, info = env.step(action)
        portfolio_values.append(float(info["portfolio_value"]))

    pv = np.array(portfolio_values, dtype=np.float64)

    roi = compute_roi(pv)
    sharpe = compute_sharpe(pv)
    mdd = compute_max_drawdown(pv)
    calmar = compute_calmar(pv)

    logger.info(
        "[%s] Evaluation — ROI=%.4f  Sharpe=%.4f  MDD=%.4f  Calmar=%.4f",
        ticker,
        roi,
        sharpe,
        mdd,
        calmar,
    )

    return EvalResult(
        ticker=ticker,
        roi=roi,
        sharpe=sharpe,
        max_drawdown=mdd,
        calmar=calmar,
        portfolio_values=pv,
    )
