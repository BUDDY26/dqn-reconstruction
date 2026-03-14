"""
data.py — Data loading, feature engineering, and preprocessing pipeline.

CONFIRMED behaviours (sourced from paper + report):
  - Assets: AAPL, INTC, META, TQQQ, TSLA
  - Features: OHLCV + VWAP + percentage changes
  - Preprocessing: StandardScaler

ASSUMED behaviours (declared in docs/adr/ADR-002-reconstruction-assumptions.md):
  A-D1  Percentage changes: 1-day pct_change on each OHLCV column
  A-D2  VWAP approximation: (High + Low + Close) / 3
  A-D3  Training period ends 2022-12-31; no validation split
  A-D4  Per-asset observation (features contain only one asset's data)

Data source for this first pass: local CSV files only.
Live data download (e.g. via yfinance) is deferred to a later pass.

Expected CSV format:
    Date,Open,High,Low,Close,Volume
    2017-01-03,115.80,116.33,114.76,116.15,28781865
    ...
Column names are case-insensitive; they are normalized to lowercase on load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, PCT_CHANGE_PERIOD, TEST_END, TEST_START, TRAIN_END

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path | str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file into a DataFrame with a DatetimeIndex.

    Args:
        path: Path to a CSV file with columns: Date, Open, High, Low, Close, Volume.
              Column names are case-insensitive. The Date column is used as the index.

    Returns:
        DataFrame with DatetimeIndex and lowercase columns: open, high, low, close, volume.
        Rows are sorted ascending by date.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing after normalisation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    # Normalise column names to lowercase for consistent access.
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values, then drop any remaining NaN rows.

    For daily OHLCV data, forward-fill propagates the previous trading day's values
    across any gaps (e.g. data errors, non-standard holidays). This is conservative
    and keeps the time index intact.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with no NaN values.
    """
    n_before = len(df)
    df = df.ffill()
    n_after_ffill = len(df.dropna())
    n_dropped = n_before - n_after_ffill

    if n_dropped > 0:
        logger.warning(
            "fill_missing: %d row(s) still contain NaN after forward-fill and will be dropped.",
            n_dropped,
        )
    return df.dropna()


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute the daily VWAP approximation as (High + Low + Close) / 3.

    ASSUMED (A-D2): True VWAP requires intraday tick data, which is unavailable
    for daily bars.  The 'typical price' (H + L + C) / 3 is the standard
    daily-bar approximation in technical analysis and academic literature.

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns.

    Returns:
        Series named 'vwap' aligned with df's index.
    """
    vwap = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap.name = "vwap"
    return vwap


def compute_pct_changes(df: pd.DataFrame, period: int = PCT_CHANGE_PERIOD) -> pd.DataFrame:
    """Compute percentage changes for each OHLCV column.

    ASSUMED (A-D1): 1-day (period=1) percentage change applied to each of the five
    OHLCV columns.  The first row will contain NaN and must be dropped downstream.

    Args:
        df: DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
        period: Look-back period for pct_change.  Default is PCT_CHANGE_PERIOD (1).

    Returns:
        DataFrame with columns: pct_open, pct_high, pct_low, pct_close, pct_volume.
        The first `period` rows are NaN.
    """
    pct_cols = {}
    for col in ["open", "high", "low", "close", "volume"]:
        pct_cols[f"pct_{col}"] = df[col].pct_change(period)
    return pd.DataFrame(pct_cols, index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full 11-column feature matrix from raw OHLCV data.

    Column order matches config.FEATURE_COLUMNS (canonical; do not reorder):
        open, high, low, close, volume,               (raw OHLCV — CONFIRMED)
        vwap,                                          (A-D2 approximation)
        pct_open, pct_high, pct_low, pct_close, pct_volume  (A-D1, 1-day)

    The first row is always dropped because pct_change yields NaN for it.
    Any additional NaN rows (from gaps after forward-fill) are also dropped.

    Args:
        df: Raw OHLCV DataFrame as returned by load_csv / fill_missing.

    Returns:
        DataFrame with exactly OBS_DIM (11) columns and (len(df) - 1) rows.
        Index is a DatetimeIndex of trading days, starting from the second input day.

    Raises:
        ValueError: If the resulting DataFrame has fewer than 2 rows (too little data
                    to form even one environment transition).
    """
    df = fill_missing(df)

    vwap = compute_vwap(df)
    pct = compute_pct_changes(df)

    features = pd.concat([df, vwap, pct], axis=1)

    # Drop rows where any feature is NaN (always includes the first row due to pct_change).
    features = features.dropna()

    # Enforce canonical column order from config.
    features = features[FEATURE_COLUMNS]

    if len(features) < 2:
        raise ValueError(
            f"build_features produced only {len(features)} row(s); at least 2 are required "
            "to form a single environment transition."
        )

    logger.debug("build_features: produced %d rows × %d features.", *features.shape)
    return features


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def split_by_date(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature DataFrame into training and test sets by date.

    ASSUMED (A-D3): train_end = 2022-12-31, test_start = 2023-01-01,
    test_end = 2024-01-01.  No validation split is created.

    The split is applied to the DatetimeIndex of the feature DataFrame (which has
    already had its first row dropped by build_features).  Rows between train_end
    and test_start are excluded if any such gap exists in the data.

    Args:
        df: Feature DataFrame with DatetimeIndex (output of build_features).
        train_end: Last date of the training period (inclusive), ISO format.
        test_start: First date of the test period (inclusive), ISO format.
        test_end: Last date of the test period (inclusive), ISO format.

    Returns:
        (train_df, test_df) — two non-overlapping DataFrames.

    Raises:
        ValueError: If either split is empty.
    """
    train_df = df[df.index <= train_end]
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]

    if len(train_df) == 0:
        raise ValueError(
            f"Training split is empty for train_end={train_end!r}. "
            "Check that the data covers the training period."
        )
    if len(test_df) == 0:
        raise ValueError(
            f"Test split is empty for test_start={test_start!r}, test_end={test_end!r}. "
            "Check that the data covers the test period."
        )

    logger.debug(
        "split_by_date: %d train rows (%s to %s), %d test rows (%s to %s).",
        len(train_df), train_df.index[0].date(), train_df.index[-1].date(),
        len(test_df), test_df.index[0].date(), test_df.index[-1].date(),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training feature DataFrame.

    The scaler is fit on the training set only.  It must then be used to transform
    both the training and test sets.  Fitting on the full dataset would constitute
    look-ahead bias.

    Args:
        train_df: Training feature DataFrame (output of split_by_date).

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    logger.debug("fit_scaler: fitted on %d training rows.", len(train_df))
    return scaler


def apply_scaler(scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    """Transform a feature DataFrame using a pre-fitted StandardScaler.

    Args:
        scaler: A StandardScaler that has already been fitted (via fit_scaler).
        df: Feature DataFrame to transform.

    Returns:
        Float32 NumPy array of shape (len(df), OBS_DIM).
    """
    return scaler.transform(df.values).astype(np.float32)


# ---------------------------------------------------------------------------
# Full Pipeline Helper
# ---------------------------------------------------------------------------

def prepare_arrays(
    feature_df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Run the full preprocessing pipeline: split → scale → extract close prices.

    Raw close prices are extracted BEFORE scaling because the environment needs
    unscaled prices to compute the reward signal (A-E1).

    Args:
        feature_df: Full feature DataFrame (output of build_features).
        train_end: Last date of the training period.
        test_start: First date of the test period.
        test_end: Last date of the test period.

    Returns:
        train_features: Float32 array of shape (n_train, 11) — scaled training observations.
        test_features:  Float32 array of shape (n_test, 11) — scaled test observations.
        train_close:    Float64 array of shape (n_train,) — raw training close prices.
        test_close:     Float64 array of shape (n_test,) — raw test close prices.
        scaler:         Fitted StandardScaler (fit on training data only).
    """
    train_df, test_df = split_by_date(feature_df, train_end, test_start, test_end)

    # Extract raw close prices before scaling destroys their magnitude.
    # The 'close' column index in FEATURE_COLUMNS is 3.
    train_close = train_df["close"].values.astype(np.float64)
    test_close = test_df["close"].values.astype(np.float64)

    scaler = fit_scaler(train_df)
    train_features = apply_scaler(scaler, train_df)
    test_features = apply_scaler(scaler, test_df)

    return train_features, test_features, train_close, test_close, scaler
