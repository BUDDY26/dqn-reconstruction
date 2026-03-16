"""
scripts/run_optimized.py — Optimized run runner for the DQN reconstruction.

Trains and evaluates the DQN agent on each of the five confirmed assets using
existing pipeline functions only.  No files in src/ are modified.

Differences from run_verification.py:
    - Output: results/optimized_results.json  (never overwrites verification_results.json)
    - Checkpoints: results/checkpoints/optimized/{ticker}.pt

The optimized src/ includes all five improvements over the baseline:
    1. GPU / CUDA device support
    2. Reproducibility seed fix (all four RNG sources seeded)
    3. Gradient clipping (max_norm=1.0)
    4. Huber loss / SmoothL1 (replaces MSE)
    5. Double DQN target selection

Usage:
    python scripts/run_optimized.py --data-dir data/
    python scripts/run_optimized.py --data-dir data/ --seed 7

Expected CSV layout (one file per asset, named {TICKER}.csv):
    {data_dir}/AAPL.csv
    {data_dir}/INTC.csv
    {data_dir}/META.csv
    {data_dir}/TQQQ.csv
    {data_dir}/TSLA.csv

CSV format (column names are case-insensitive):
    Date,Open,High,Low,Close,Volume
    2017-01-03,115.80,116.33,114.76,116.15,28781865
    ...

The data must cover 2017-01-01 through at least 2024-01-01 so that both the
training split (≤ 2022-12-31) and the test split (2023-01-01 – 2024-01-01)
are non-empty.  See src/config.py for the confirmed date boundaries.

Output:
    results/optimized_results.json              — per-asset metrics and run metadata
    results/checkpoints/optimized/{ticker}.pt   — trained model weights per asset
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Add src/ to sys.path so pipeline modules are importable when this script is
# invoked directly (e.g. python scripts/run_optimized.py ...).
# pytest handles this automatically via pythonpath = ["src"] in pyproject.toml;
# direct execution requires the path to be set explicitly.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402 — must follow sys.path setup

from config import ASSETS  # noqa: E402
from data import build_features, load_csv, prepare_arrays  # noqa: E402
from evaluate import evaluate  # noqa: E402
from train import train  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized run: train and evaluate the DQN agent "
            "on all five confirmed assets and write results to "
            "results/optimized_results.json."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Directory containing one CSV file per asset, named {TICKER}.csv "
            "(e.g. data/AAPL.csv).  All five asset files must be present."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Integer random seed passed to train() for reproducibility (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


def _validate_data_files(data_dir: Path, assets: list[str]) -> list[Path]:
    """Confirm every expected CSV file exists before any training begins.

    Args:
        data_dir: Directory that should contain one CSV per asset.
        assets:   Ordered list of ticker symbols from config.ASSETS.

    Returns:
        Ordered list of Path objects (same order as `assets`) if all files
        are present.

    Side effects:
        Prints a detailed error summary and exits with status 1 if any file
        is missing.  Partial execution is never allowed.
    """
    paths = {ticker: data_dir / f"{ticker}.csv" for ticker in assets}
    missing = [ticker for ticker in assets if not paths[ticker].exists()]

    if missing:
        print("ERROR: one or more required data files are missing.")
        print(f"  Expected directory : {data_dir.resolve()}")
        print(f"  Missing ({len(missing)}/{len(assets)}):")
        for ticker in missing:
            print(f"    {paths[ticker]}")
        print()
        print("Each file must contain daily OHLCV data with columns:")
        print("  Date, Open, High, Low, Close, Volume  (case-insensitive)")
        print("covering 2017-01-01 through at least 2024-01-01.")
        sys.exit(1)

    return [paths[ticker] for ticker in assets]


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------


def _prepare_output_dirs(repo_root: Path) -> tuple[Path, Path]:
    """Create results/ and results/checkpoints/optimized/ directories if absent.

    Returns:
        (results_dir, checkpoints_dir) as absolute Path objects.
    """
    results_dir = repo_root / "results"
    checkpoints_dir = results_dir / "checkpoints" / "optimized"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, checkpoints_dir


# ---------------------------------------------------------------------------
# Per-asset pipeline
# ---------------------------------------------------------------------------


def _run_asset(
    ticker: str,
    csv_path: Path,
    seed: int,
    checkpoint_path: Path,
) -> dict[str, float]:
    """Run the full train-and-evaluate pipeline for one asset.

    Calls existing functions in order:
        load_csv → build_features → prepare_arrays → train → evaluate

    Args:
        ticker:          Asset ticker symbol (used for logging).
        csv_path:        Path to the asset's OHLCV CSV file.
        seed:            Random seed forwarded to train().
        checkpoint_path: Destination path for the trained model checkpoint.

    Returns:
        Dict containing roi, sharpe, max_drawdown, calmar as plain floats.
    """
    print(f"\n[{ticker}] Loading data from {csv_path} ...")
    raw_df = load_csv(csv_path)
    feature_df = build_features(raw_df)
    train_features, test_features, train_close, test_close, _ = prepare_arrays(feature_df)

    n_train = len(train_features)
    n_test = len(test_features)
    print(f"[{ticker}] Data ready — {n_train} train rows, {n_test} test rows.")

    print(f"[{ticker}] Training (200 episodes) ...")
    train_result = train(
        features=train_features,
        close_prices=train_close,
        ticker=ticker,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    print(f"[{ticker}] Training complete. Checkpoint → {checkpoint_path}")

    print(f"[{ticker}] Evaluating on test period ...")
    eval_result = evaluate(
        agent=train_result.agent,
        features=test_features,
        close_prices=test_close,
        ticker=ticker,
    )

    print(
        f"[{ticker}] "
        f"ROI={eval_result.roi:.4f}  "
        f"Sharpe={eval_result.sharpe:.4f}  "
        f"MDD={eval_result.max_drawdown:.4f}  "
        f"Calmar={eval_result.calmar:.4f}"
    )

    return {
        "roi": eval_result.roi,
        "sharpe": eval_result.sharpe,
        "max_drawdown": eval_result.max_drawdown,
        "calmar": eval_result.calmar,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # Validate all five asset files before touching any training logic.
    # If any file is absent the script exits here and does not proceed.
    csv_paths = _validate_data_files(args.data_dir, ASSETS)

    results_dir, checkpoints_dir = _prepare_output_dirs(_REPO_ROOT)

    print(f"Starting optimized run — seed={args.seed}, assets={ASSETS}")

    asset_results: dict[str, dict[str, float]] = {}
    for ticker, csv_path in zip(ASSETS, csv_paths):
        checkpoint_path = checkpoints_dir / f"{ticker}.pt"
        asset_results[ticker] = _run_asset(ticker, csv_path, args.seed, checkpoint_path)

    # Assemble the output document.
    output = {
        "run_metadata": {
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "assets": ASSETS,
        },
        "results": asset_results,
    }

    results_file = results_dir / "optimized_results.json"
    results_file.write_text(json.dumps(output, indent=2))

    print(f"\nResults written → {results_file}")
    print("Optimized run complete.")


if __name__ == "__main__":
    main()
