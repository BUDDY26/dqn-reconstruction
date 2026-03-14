"""
tests/unit/test_data.py — Unit tests for src/data.py.

Tests verify:
  - load_csv: correct shape, index type, column names
  - compute_vwap: correct formula (H + L + C) / 3  [assumption A-D2]
  - compute_pct_changes: correct 1-day computation, NaN on first row  [assumption A-D1]
  - build_features: correct output shape (n-1 rows, 11 cols), no NaN
  - split_by_date: correct row counts for training / test splits  [assumption A-D3]
  - fit_scaler / apply_scaler: zero-mean unit-variance transformation
  - prepare_arrays: pipeline produces correct shapes and raw close prices

All tests use the synthetic fixture CSV (tests/fixtures/sample_ohlcv.csv).
No live data is fetched.
"""

import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLUMNS, OBS_DIM
from data import (
    apply_scaler,
    build_features,
    compute_pct_changes,
    compute_vwap,
    fit_scaler,
    load_csv,
    prepare_arrays,
    split_by_date,
)

# ---------------------------------------------------------------------------
# Fixture: loaded raw DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df(sample_csv_path):
    """Raw OHLCV DataFrame loaded from the synthetic fixture."""
    return load_csv(sample_csv_path)


@pytest.fixture
def feature_df(raw_df):
    """Full feature DataFrame after build_features."""
    return build_features(raw_df)


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------

class TestLoadCsv:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_index_is_datetime(self, raw_df):
        assert isinstance(raw_df.index, pd.DatetimeIndex)

    def test_columns_are_lowercase_ohlcv(self, raw_df):
        assert set(raw_df.columns) == {"open", "high", "low", "close", "volume"}

    def test_row_count_matches_csv(self, raw_df):
        # The synthetic CSV has 34 rows (header excluded).
        assert len(raw_df) == 34

    def test_sorted_ascending(self, raw_df):
        assert raw_df.index.is_monotonic_increasing

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Date,Open,High,Low\n2023-01-03,100,101,99\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(bad_csv)


# ---------------------------------------------------------------------------
# compute_vwap  [assumption A-D2: VWAP = (H + L + C) / 3]
# ---------------------------------------------------------------------------

class TestComputeVwap:
    def test_formula_is_hlc_over_3(self, raw_df):
        """VWAP for each row = (high + low + close) / 3."""
        vwap = compute_vwap(raw_df)
        expected = (raw_df["high"] + raw_df["low"] + raw_df["close"]) / 3.0
        pd.testing.assert_series_equal(vwap, expected, check_names=False)

    def test_first_row_known_value(self, raw_df):
        """Fixture row 0: H=101.50, L=99.00, C=100.80 → VWAP = 100.4333..."""
        vwap = compute_vwap(raw_df)
        expected = (101.50 + 99.00 + 100.80) / 3.0   # 100.43333...
        assert np.isclose(vwap.iloc[0], expected, rtol=1e-6)

    def test_returns_series_named_vwap(self, raw_df):
        vwap = compute_vwap(raw_df)
        assert isinstance(vwap, pd.Series)
        assert vwap.name == "vwap"

    def test_no_nan_when_input_is_clean(self, raw_df):
        vwap = compute_vwap(raw_df)
        assert not vwap.isna().any()


# ---------------------------------------------------------------------------
# compute_pct_changes  [assumption A-D1: 1-day pct_change on all OHLCV]
# ---------------------------------------------------------------------------

class TestComputePctChanges:
    def test_first_row_is_nan(self, raw_df):
        """pct_change(1) always produces NaN for the first row."""
        pct = compute_pct_changes(raw_df)
        assert pct.iloc[0].isna().all()

    def test_second_row_close_value(self, raw_df):
        """Row 1 pct_close = (101.50 - 100.80) / 100.80 ≈ 0.006944."""
        pct = compute_pct_changes(raw_df)
        expected = (101.50 - 100.80) / 100.80
        assert np.isclose(pct["pct_close"].iloc[1], expected, rtol=1e-6)

    def test_second_row_open_value(self, raw_df):
        """Row 1 pct_open = (100.80 - 100.00) / 100.00 = 0.008."""
        pct = compute_pct_changes(raw_df)
        expected = (100.80 - 100.00) / 100.00
        assert np.isclose(pct["pct_open"].iloc[1], expected, rtol=1e-6)

    def test_second_row_volume_value(self, raw_df):
        """Row 1 pct_volume = (1050000 - 1000000) / 1000000 = 0.05."""
        pct = compute_pct_changes(raw_df)
        expected = (1_050_000 - 1_000_000) / 1_000_000
        assert np.isclose(pct["pct_volume"].iloc[1], expected, rtol=1e-6)

    def test_output_columns(self, raw_df):
        pct = compute_pct_changes(raw_df)
        expected_cols = {"pct_open", "pct_high", "pct_low", "pct_close", "pct_volume"}
        assert set(pct.columns) == expected_cols

    def test_same_length_as_input(self, raw_df):
        pct = compute_pct_changes(raw_df)
        assert len(pct) == len(raw_df)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_output_has_11_columns(self, feature_df):
        assert feature_df.shape[1] == OBS_DIM

    def test_column_names_match_config(self, feature_df):
        assert list(feature_df.columns) == FEATURE_COLUMNS

    def test_no_nan_values(self, feature_df):
        assert not feature_df.isna().any().any()

    def test_row_count_is_n_minus_1(self, raw_df, feature_df):
        """First row is dropped due to NaN pct_change; result = 34 - 1 = 33 rows."""
        assert len(feature_df) == len(raw_df) - 1

    def test_first_row_is_dropped(self, raw_df, feature_df):
        """The date '2022-11-28' (first CSV row) must not appear in output."""
        assert pd.Timestamp("2022-11-28") not in feature_df.index

    def test_index_is_datetime(self, feature_df):
        assert isinstance(feature_df.index, pd.DatetimeIndex)

    def test_raises_on_insufficient_data(self, tmp_path):
        """A single-row CSV (0 features after NaN drop) must raise ValueError."""
        tiny = tmp_path / "tiny.csv"
        tiny.write_text("Date,Open,High,Low,Close,Volume\n2023-01-03,100,101,99,100,1000000\n")
        df = load_csv(tiny)
        with pytest.raises(ValueError, match="at least 2"):
            build_features(df)


# ---------------------------------------------------------------------------
# split_by_date  [assumption A-D3]
# ---------------------------------------------------------------------------

class TestSplitByDate:
    def test_train_ends_before_test_starts(self, feature_df):
        train_df, test_df = split_by_date(feature_df)
        assert train_df.index.max() < test_df.index.min()

    def test_train_count(self, feature_df):
        """Fixture: 23 rows with date ≤ 2022-12-30 (after dropping first CSV row)."""
        train_df, _ = split_by_date(feature_df)
        assert len(train_df) == 23

    def test_test_count(self, feature_df):
        """Fixture: 10 rows with date ≥ 2023-01-03."""
        _, test_df = split_by_date(feature_df)
        assert len(test_df) == 10

    def test_no_overlap(self, feature_df):
        train_df, test_df = split_by_date(feature_df)
        overlap = train_df.index.intersection(test_df.index)
        assert len(overlap) == 0

    def test_empty_train_raises(self, feature_df):
        with pytest.raises(ValueError, match="Training split is empty"):
            split_by_date(feature_df, train_end="2000-01-01")

    def test_empty_test_raises(self, feature_df):
        with pytest.raises(ValueError, match="Test split is empty"):
            split_by_date(feature_df, test_start="2030-01-01", test_end="2031-01-01")


# ---------------------------------------------------------------------------
# fit_scaler / apply_scaler
# ---------------------------------------------------------------------------

class TestScaling:
    def test_train_features_zero_mean(self, feature_df):
        """After StandardScaler fit+transform on train, column means ≈ 0."""
        train_df, _ = split_by_date(feature_df)
        scaler = fit_scaler(train_df)
        train_scaled = apply_scaler(scaler, train_df)
        col_means = train_scaled.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=1e-6), f"Column means not near 0: {col_means}"

    def test_train_features_unit_variance(self, feature_df):
        """After StandardScaler fit+transform on train, column stds ≈ 1."""
        train_df, _ = split_by_date(feature_df)
        scaler = fit_scaler(train_df)
        train_scaled = apply_scaler(scaler, train_df)
        col_stds = train_scaled.std(axis=0)
        assert np.allclose(col_stds, 1.0, atol=1e-6), f"Column stds not near 1: {col_stds}"

    def test_output_dtype_is_float32(self, feature_df):
        train_df, test_df = split_by_date(feature_df)
        scaler = fit_scaler(train_df)
        assert apply_scaler(scaler, train_df).dtype == np.float32
        assert apply_scaler(scaler, test_df).dtype == np.float32

    def test_output_shape(self, feature_df):
        train_df, test_df = split_by_date(feature_df)
        scaler = fit_scaler(train_df)
        train_arr = apply_scaler(scaler, train_df)
        test_arr = apply_scaler(scaler, test_df)
        assert train_arr.shape == (23, OBS_DIM)
        assert test_arr.shape == (10, OBS_DIM)


# ---------------------------------------------------------------------------
# prepare_arrays (pipeline integration within data.py)
# ---------------------------------------------------------------------------

class TestPrepareArrays:
    def test_output_shapes(self, feature_df):
        train_f, test_f, train_c, test_c, scaler = prepare_arrays(feature_df)
        assert train_f.shape == (23, OBS_DIM)
        assert test_f.shape == (10, OBS_DIM)
        assert train_c.shape == (23,)
        assert test_c.shape == (10,)

    def test_train_close_matches_raw(self, feature_df):
        """Raw close prices must equal the unscaled 'close' column from the train split."""
        train_df, _ = split_by_date(feature_df)
        _, _, train_c, _, _ = prepare_arrays(feature_df)
        expected = train_df["close"].values
        assert np.allclose(train_c, expected)

    def test_close_prices_are_float64(self, feature_df):
        _, _, train_c, test_c, _ = prepare_arrays(feature_df)
        assert train_c.dtype == np.float64
        assert test_c.dtype == np.float64

    def test_scaler_not_fitted_on_test(self, feature_df):
        """Verify the scaler was fit only on training data by checking test set mean ≠ 0."""
        _, test_f, _, _, _ = prepare_arrays(feature_df)
        # Test features should NOT have zero mean (scaler was not fit on them).
        # This is a soft check — with a small test set the mean won't be exactly 0 anyway.
        assert not np.allclose(test_f.mean(axis=0), 0.0, atol=1e-2)
