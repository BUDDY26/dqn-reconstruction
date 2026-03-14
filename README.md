# dqn-reconstruction

A reconstruction of a Deep Q-Network (DQN) agent for discrete-action stock trading, rebuilt from
the surviving final project report and original research paper after the implementation repository
became unavailable.

---

## Project Overview

This repository reconstructs a reinforcement learning system in which a DQN agent learns to issue
buy, hold, or sell decisions for five equity assets using daily OHLCV data. The original system was
developed as a collaborative research project. When the implementation became inaccessible, this
reconstruction was initiated to restore a working, reproducible version of the experiment using only
the documented technical record.

The reconstruction adheres to a strict evidence discipline: every implementation choice is either
directly sourced from the paper or report, or explicitly flagged as a reconstruction assumption in
`docs/evidence-ledger.md`. No implementation details are invented or inferred without documentation.

---

## Reconstruction Context

| Item | Status |
|------|--------|
| Original implementation | Unavailable |
| Final project report | Available — primary source |
| Research paper | Available — primary source |
| Hyperparameters | Confirmed from paper/report |
| Architecture details | Confirmed from paper/report |
| Unresolved implementation gaps | Documented in `docs/evidence-ledger.md` |

This is not a re-implementation from scratch based on general DQN knowledge. It is a
evidence-bounded reconstruction from a specific prior experiment. Where the paper and report are
silent, decisions are deferred or explicitly declared as assumptions.

---

## Research Context

The original project applied deep reinforcement learning to financial trading, framing portfolio
management as a discrete Markov Decision Process. The agent observes engineered features derived
from daily price data and learns a policy that maximizes cumulative return over the training period.

**Assets:** AAPL, INTC, META, TQQQ, TSLA
**Data range:** 2017-01-01 to 2024-06-01
**Action space:** {buy, hold, sell} — three discrete actions
**State features:** OHLCV + VWAP + percentage changes, normalized with StandardScaler

---

## Confirmed Technical Specifications

These values are taken directly from the paper and report. They are not estimates.

| Parameter | Value | Source |
|-----------|-------|--------|
| Assets | AAPL, INTC, META, TQQQ, TSLA | Paper/Report |
| Date range | 2017-01-01 to 2024-06-01 | Paper/Report |
| Test period | 2023-01-01 to 2024-01-01 | Paper/Report |
| Input features | OHLCV + VWAP + pct changes | Paper/Report |
| Preprocessing | StandardScaler | Paper/Report |
| Hidden layers | 256 → 256 | Paper/Report |
| Activation | ReLU | Paper/Report |
| Actions | buy / hold / sell | Paper/Report |
| Learning rate | 0.0003 | Paper/Report |
| Discount factor (γ) | 0.99 | Paper/Report |
| Epsilon range | 1.0 → 0.01 | Paper/Report |
| Batch size | 64 | Paper/Report |
| Replay buffer size | 100,000 | Paper/Report |
| Training episodes | 200 | Paper/Report |
| Evaluation metrics | ROI, Sharpe Ratio, Max Drawdown, Calmar Ratio | Paper/Report |

For a complete list of confirmed details, unresolved gaps, and declared reconstruction assumptions,
see [`docs/evidence-ledger.md`](docs/evidence-ledger.md).

---

## Repository Structure

```
dqn-reconstruction/
├── docs/
│   ├── architecture.md          # System design and component breakdown
│   ├── evidence-ledger.md       # Confirmed facts, gaps, and assumptions
│   ├── implementation-plan.md   # Planned modules and pipeline
│   ├── adr/                     # Architecture Decision Records
│   ├── qa/                      # Test strategy
│   └── runbooks/                # Setup and operational guide
├── src/                         # Implementation (added during coding phase)
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── bootstrap.sh
│   └── validate-structure.sh
└── .github/
    └── workflows/
        └── ci.yml               # Lint + test + security pipeline
```

---

## Setup

```bash
# Clone
git clone https://github.com/BUDDY26/dqn-reconstruction.git
cd dqn-reconstruction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies (once requirements.txt is populated)
pip install -r requirements.txt
```

---

## Running

```bash
# Train the agent (once src/train.py exists)
python src/train.py

# Run tests
pytest tests/ -v

# Validate repo structure
bash scripts/validate-structure.sh
```

---

## Reproducibility Goals

1. **Hyperparameter fidelity** — all confirmed hyperparameters from the paper are reproduced exactly;
   no values are tuned away from the documented source.
2. **Evidence traceability** — every module decision references either the paper/report or a declared
   assumption in `docs/evidence-ledger.md`.
3. **Assumption transparency** — reconstruction assumptions are not buried in code comments; they are
   centrally tracked and versioned.
4. **No silent substitutions** — if a documented value produces unexpected behavior during
   reconstruction, the behavior is documented rather than silently corrected.

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [`docs/architecture.md`](docs/architecture.md) | Full system design |
| [`docs/evidence-ledger.md`](docs/evidence-ledger.md) | Confirmed facts, gaps, assumptions |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | Module and pipeline plan |
| [`docs/adr/ADR-001-template.md`](docs/adr/ADR-001-template.md) | Reconstruction philosophy ADR |
| [`docs/qa/qa-plan.md`](docs/qa/qa-plan.md) | Test strategy |
| [`docs/runbooks/operations.md`](docs/runbooks/operations.md) | Setup and operational procedures |

---

*Portfolio project — UT Austin MSCS application.*
*Reconstruction initiated: 2026-03-14.*
