"""
conftest.py — Shared pytest fixtures for the DQN reconstruction test suite.

Fixtures defined here are available to all test files without explicit imports.
"""

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# File path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_csv_path() -> Path:
    """Absolute path to the synthetic OHLCV fixture CSV."""
    path = FIXTURES_DIR / "sample_ohlcv.csv"
    assert path.exists(), f"Fixture file not found: {path}"
    return path


# ---------------------------------------------------------------------------
# Synthetic array fixtures (used by unit tests that bypass file I/O)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features() -> np.ndarray:
    """Small synthetic feature array of shape (10, 11), dtype float32.

    Values are arbitrary; used to test environment mechanics without real data.
    Seeded for reproducibility.
    """
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((10, 11)).astype(np.float32)


@pytest.fixture
def synthetic_close_prices() -> np.ndarray:
    """Synthetic close price sequence of length 10 for environment tests.

    Prices are chosen to produce known, easy-to-verify daily returns:
      [100, 110, 100, 110, 100, 110, 100, 110, 100, 110]

    daily_returns (indices 0→1, 1→2, ...):
      +0.10, -0.0909..., +0.10, -0.0909..., +0.10, -0.0909..., +0.10, -0.0909..., +0.10
    """
    return np.array(
        [100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0],
        dtype=np.float64,
    )
