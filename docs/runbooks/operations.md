# Operations Runbook

---

## Prerequisites

- **Python 3.11 or 3.12** (both tested in CI)
- **pip** (bundled with Python)
- Approximately **2 GB disk space** for OHLCV data files and trained model checkpoints

All Python package dependencies are declared in `requirements.txt`:

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
gymnasium>=0.29.0
torch>=2.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/BUDDY26/dqn-reconstruction.git
cd dqn-reconstruction

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Data Preparation

The verification runner reads daily OHLCV data from local CSV files.
No automatic download is performed.

### File placement

Place one CSV file per asset in a directory of your choosing.
The default expected layout when using `--data-dir data/` is:

```
dqn-reconstruction/
└── data/
    ├── AAPL.csv
    ├── INTC.csv
    ├── META.csv
    ├── TQQQ.csv
    └── TSLA.csv
```

The `data/` directory is not committed to the repository and is not created
automatically.  Create it and place the files before running verification.

### Required date range

Each CSV file must cover **2017-01-01 through at least 2024-01-01** so that
both splits are non-empty:

| Split | Date range | Approximate rows |
|-------|-----------|-----------------|
| Training | 2017-01-01 – 2022-12-31 | ~1,500 trading days |
| Test (evaluation) | 2023-01-01 – 2024-01-01 | ~252 trading days |

Rows outside these ranges are downloaded but not used in training or evaluation.
The full confirmed download range is 2017-01-01 to 2024-06-01 (see `src/config.py`).

### Required CSV schema

Column names are **case-insensitive** and normalised to lowercase on load.

```
Date,Open,High,Low,Close,Volume
2017-01-03,115.80,116.33,114.76,116.15,28781865
2017-01-04,116.15,116.98,115.75,116.02,20555200
...
```

| Column | Type | Notes |
|--------|------|-------|
| `Date` | ISO date string | Used as the DataFrame index |
| `Open` | float | Raw opening price |
| `High` | float | Intraday high |
| `Low` | float | Intraday low |
| `Close` | float | Raw closing price (unscaled; used for reward computation) |
| `Volume` | integer | Daily trading volume |

The pipeline will raise a descriptive `ValueError` if required columns are
absent or if either split resolves to zero rows.

---

## Running Verification

The verification runner trains and evaluates the DQN agent on all five assets
and writes the results to `results/verification_results.json`.

```bash
python scripts/run_verification.py --data-dir data/ --seed 42
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-dir DIR` | Yes | — | Directory containing the five asset CSV files |
| `--seed N` | No | `42` | Integer random seed for reproducible runs |

### What the script does

For each asset in order (AAPL, INTC, META, TQQQ, TSLA):

1. Loads `{data-dir}/{TICKER}.csv` via `data.load_csv`
2. Engineers VWAP and percentage-change features via `data.build_features`
3. Splits and scales the data via `data.prepare_arrays`
4. Trains the DQN agent for 200 episodes via `train.train`
5. Evaluates the trained agent on the test period via `evaluate.evaluate`
6. Collects ROI, Sharpe Ratio, Maximum Drawdown, and Calmar Ratio

All five CSV files are validated before any training begins.  If any file
is missing, the script prints a complete list of absent files and exits with
status code `1` without processing any asset.

### Expected console output (per asset)

```
[AAPL] Loading data from data/AAPL.csv ...
[AAPL] Data ready — 1509 train rows, 252 test rows.
[AAPL] Training (200 episodes) ...
[AAPL] Training complete. Checkpoint → results/checkpoints/AAPL.pt
[AAPL] Evaluating on test period ...
[AAPL] ROI=...  Sharpe=...  MDD=...  Calmar=...
```

Row counts will vary slightly depending on the exact trading days in the
source data.  The values printed for each metric are the actual computed
results and will differ between runs with different seeds.

---

## Verification Outputs

### `results/verification_results.json` (primary tracked artifact)

Written after all five assets complete.  Contains run metadata and the four
confirmed evaluation metrics for each asset.

```json
{
  "run_metadata": {
    "seed": 42,
    "timestamp": "<ISO 8601 UTC>",
    "python_version": "<sys.version>",
    "torch_version": "<torch.__version__>",
    "assets": ["AAPL", "INTC", "META", "TQQQ", "TSLA"]
  },
  "results": {
    "AAPL": { "roi": ..., "sharpe": ..., "max_drawdown": ..., "calmar": ... },
    "INTC": { "roi": ..., "sharpe": ..., "max_drawdown": ..., "calmar": ... },
    "META": { "roi": ..., "sharpe": ..., "max_drawdown": ..., "calmar": ... },
    "TQQQ": { "roi": ..., "sharpe": ..., "max_drawdown": ..., "calmar": ... },
    "TSLA": { "roi": ..., "sharpe": ..., "max_drawdown": ..., "calmar": ... }
  }
}
```

This file is **committed to the repository** once produced.  It is the
single verification artifact that documents the reconstruction results.

### `results/checkpoints/{ticker}.pt` (not committed)

One PyTorch `state_dict` file is saved per asset after training completes.
These are **gitignored** by the existing `.gitignore` rules (`*.pt` and
`checkpoints/`) and are not tracked in the repository.

To reproduce the evaluation metrics from a saved checkpoint without
retraining, load the state dict into a `QNetwork` instance using
`torch.load` and call `evaluate.evaluate` directly.

---

## Artifact Tracking Policy

| Artifact | Location | Tracked in git |
|----------|----------|---------------|
| Verification metrics | `results/verification_results.json` | **Yes** — commit after each verified run |
| Model checkpoints | `results/checkpoints/*.pt` | **No** — gitignored; regenerate by rerunning |
| Raw data files | `data/*.csv` | **No** — not committed; obtain separately |
| Coverage report | `coverage.xml` | **No** — gitignored; generated by CI |

The `.gitignore` rules `*.pt` and `checkpoints/` are already in place and
cover the checkpoint directory.  No `.gitignore` modifications are needed
to enforce this policy.

---

## Running Tests

```bash
# Run the full test suite with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite uses a synthetic fixture CSV (`tests/fixtures/sample_ohlcv.csv`)
for all tests.  No real asset data is required to run tests.

**Current state:** 204 tests, 0 failures, verified on Python 3.11 and 3.12.

---

## Linting and Formatting

```bash
# Lint (ruff)
ruff check src/ tests/

# Format check (black)
black --check src/ tests/

# Apply formatting
black src/ tests/
```

Configuration is in `pyproject.toml`:

- `line-length = 100`
- ruff rules: `E`, `F`, `W`, `I` (errors, pyflakes, warnings, import sort)
- target Python: 3.11

---

## Validate Repository Structure

```bash
bash scripts/validate-structure.sh
```

---

## CI/CD

The CI pipeline runs on every push and pull request to `main`.
See `.github/workflows/ci.yml` for the full pipeline definition.

Jobs in order:
1. **Lint & Format** — `ruff check` + `black --check` on `src/` and `tests/`
2. **Tests** — `pytest` with coverage on Python 3.11 and 3.12 (matrix)
3. **Validate Repo Structure** — `bash scripts/validate-structure.sh`
4. **Security Scan** — `bandit -r src/` (push to `main` only)

The verification runner (`scripts/run_verification.py`) is **not** run in CI
because it requires real asset CSV files that are not committed to the repository.

---

## Troubleshooting

### `ModuleNotFoundError` when running tests or scripts
**Cause:** Virtual environment not activated, or `src/` not on the import path.
**Fix (tests):** Activate the venv and run `pytest` from the repository root.
`pyproject.toml` sets `pythonpath = ["src"]` for pytest automatically.
**Fix (scripts):** `scripts/run_verification.py` adds `src/` to `sys.path`
at startup — no manual path configuration is needed.

### `FileNotFoundError: CSV not found: data/AAPL.csv`
**Cause:** The `data/` directory or one of the asset CSV files is missing.
**Fix:** See the **Data Preparation** section above.  The script will list
all missing files before exiting.

### `ValueError: Training split is empty` or `Test split is empty`
**Cause:** The CSV file does not cover the required date range.
**Fix:** Ensure data covers 2017-01-01 through at least 2024-01-01.
The split boundaries are defined in `src/config.py`.

---

*Last updated: 2026-03-15*
