# dqn-reconstruction

Reconstruction of a Deep Q-Network (DQN) reinforcement learning system for discrete-action equity trading.

This repository restores a working implementation of a previously completed research project after the original implementation repository became unavailable. The reconstruction was performed using only the surviving technical documentation: the final project report and the associated research paper.

The goal of this project is reproducibility, not reinterpretation. The implementation attempts to faithfully recreate the documented experiment and clearly separates confirmed details from reconstruction assumptions.

---

# Project Overview

The system trains a Deep Q-Network agent to make buy / hold / sell decisions for individual equity assets using historical daily market data.

Each asset is trained independently using the same architecture and hyperparameters. The trained policy is then evaluated on a held-out test period.

The reconstructed pipeline performs the following stages:

1. Market data acquisition  
2. Feature engineering  
3. Environment simulation  
4. DQN training  
5. Policy evaluation  
6. Metric reporting  

Evaluation metrics are consistent with those used in the original paper.

---

# Assets and Data

Assets used in the experiment:

AAPL  
INTC  
META  
TQQQ  
TSLA  

Data source: Yahoo Finance

Date range:

2017-01-01 → 2024-06-01

Dataset split:

| Period | Purpose |
|------|------|
| 2017-01-01 → 2022-12-31 | Training |
| 2023-01-01 → 2024-01-01 | Evaluation |

---

# Reinforcement Learning Setup

The trading problem is framed as a Markov Decision Process.

Action space:

0 = Sell  
1 = Hold  
2 = Buy  

The agent observes engineered market features and learns a policy that maximizes cumulative reward over the training horizon.

---

# Confirmed Hyperparameters

The following parameters are taken directly from the paper and report.

| Parameter | Value |
|------|------|
| Hidden layers | 256 → 256 |
| Activation | ReLU |
| Learning rate | 0.0003 |
| Discount factor (γ) | 0.99 |
| Replay buffer | 100,000 |
| Batch size | 64 |
| Epsilon schedule | 1.0 → 0.01 |
| Training episodes | 200 |
| Evaluation metrics | ROI, Sharpe, Max Drawdown, Calmar |

No hyperparameters were tuned during the reconstruction phase.

---

# Repository Structure

dqn-reconstruction

├── data/                     # Market datasets (downloaded locally)  
├── docs/  
│   ├── architecture.md  
│   ├── evidence-ledger.md  
│   ├── implementation-plan.md  
│   ├── qa/  
│   └── runbooks/  
│  
├── results/  
│   ├── checkpoints/          # trained models  
│   ├── paper_reported_metrics.md  
│   └── verification_results.json  
│  
├── scripts/  
│   ├── download_market_data.py  
│   └── run_verification.py  
│  
├── src/                      # training pipeline  
├── tests/  
└── .github/workflows/  

---

# Setup

Create a virtual environment:

python -m venv .venv

Activate:

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

# Download Market Data

pip install yfinance  
python scripts/download_market_data.py

This downloads the datasets into:

data/

---

# Run the Verification Experiment

python scripts/run_verification.py --data-dir data --seed 42

This performs:

• model training  
• evaluation  
• metric computation  

Results are saved to:

results/verification_results.json

---

# Evaluation Metrics

The reconstructed model reports the following metrics:

| Metric | Description |
|------|------|
| ROI | Return on Investment |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Largest peak-to-trough loss |
| Calmar Ratio | Return / drawdown ratio |

Paper-reported metrics are stored in:

results/paper_reported_metrics.md

Reconstructed metrics are stored in:

results/verification_results.json

---

# Documentation

| File | Purpose |
|------|------|
| docs/architecture.md | System design |
| docs/evidence-ledger.md | Confirmed details vs reconstruction assumptions |
| docs/implementation-plan.md | Reconstruction process |
| docs/runbooks/operations.md | Setup and execution instructions |

---

# Project Status

Reconstruction phase complete.

Current repository contains:

• working training pipeline  
• reproducible data ingestion  
• evaluation scripts  
• experiment verification outputs  

Future work may include performance optimizations and experimental extensions, but these will be clearly separated from the baseline reconstruction.
