"""
scripts/download_market_data.py — Download historical OHLCV data for all five assets.

Uses yfinance to fetch daily price data for the date range confirmed in the original
paper (2017-01-01 to 2024-06-01) and saves one CSV file per asset into data/.

The asset list and date range are read from src/config.py to maintain a single
source of truth with the rest of the pipeline.

Usage:
    python scripts/download_market_data.py
    python scripts/download_market_data.py --force   # overwrite existing files

Output:
    data/AAPL.csv
    data/INTC.csv
    data/META.csv
    data/TQQQ.csv
    data/TSLA.csv

CSV columns (standard Yahoo Finance format):
    Date, Open, High, Low, Close, Adj Close, Volume

Requires yfinance (not in requirements.txt — data download is optional tooling):
    pip install yfinance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add src/ to sys.path so config.py is importable on direct invocation.
# Mirrors the pattern used in scripts/run_verification.py.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

try:
    import yfinance as yf  # noqa: E402
except ImportError:
    print("ERROR: yfinance is not installed.")
    print("Install it with:  pip install yfinance")
    sys.exit(1)

import pandas as pd  # noqa: E402

from config import ASSETS, END_DATE, START_DATE  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download daily OHLCV data for all five assets and save to data/. "
            "Existing files are skipped by default."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV files.  By default, existing files are skipped.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by recent yfinance versions.

    yfinance >= 0.2.x may return a MultiIndex when downloading a single ticker
    (e.g. ('Open', 'AAPL') instead of 'Open').  This function collapses the
    MultiIndex to a plain Index, keeping only the first level (the OHLCV name).

    Args:
        df: DataFrame as returned by yf.download for a single ticker.

    Returns:
        DataFrame with a simple column Index.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download_ticker(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV data for one ticker from Yahoo Finance.

    Args:
        ticker: Ticker symbol (e.g. 'AAPL').

    Returns:
        DataFrame with DatetimeIndex named 'Date' and columns:
        Open, High, Low, Close, Adj Close, Volume.

    Raises:
        RuntimeError: If yfinance returns an empty DataFrame.
    """
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,  # preserve Adj Close alongside raw Close
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}.")

    df = _flatten_columns(df)

    # Ensure the index is named 'Date' so df.to_csv() writes 'Date' as the
    # first column — matching the schema expected by data.load_csv().
    df.index.name = "Date"

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    data_dir = _REPO_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"Data directory : {data_dir}")
    print(f"Date range     : {START_DATE} → {END_DATE}")
    print(f"Assets         : {', '.join(ASSETS)}")
    print()

    for ticker in ASSETS:
        csv_path = data_dir / f"{ticker}.csv"

        if csv_path.exists() and not args.force:
            print(f"Skipping {ticker} (file exists)")
            continue

        print(f"Downloading {ticker}...")

        try:
            df = _download_ticker(ticker)
        except RuntimeError as exc:
            print(f"WARNING: {exc}  Skipping.")
            continue

        df.to_csv(csv_path)
        print(f"Saved → data/{ticker}.csv  ({len(df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
