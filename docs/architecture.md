# Architecture Overview

**Project:** dqn-reconstruction
**Status:** Documentation phase — architecture is planned, not yet implemented.
**Last updated:** 2026-03-14

---

## System Purpose

This system reconstructs a Deep Q-Network (DQN) agent that learns to trade five equity assets
(AAPL, INTC, META, TQQQ, TSLA) by issuing discrete buy/hold/sell decisions based on engineered
daily price features. The agent is trained over 200 episodes on data from 2017-01-01 to 2023-01-01
and evaluated on a held-out test period (2023-01-01 to 2024-01-01).

The design follows the standard DQN formulation: an online Q-network is trained via experience
replay against a periodically-updated target network, with epsilon-greedy exploration decaying
from 1.0 to 0.01 over the course of training.

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Loop (train.py)                     │
│                                                                       │
│   ┌───────────┐   state    ┌───────────┐   action   ┌────────────┐  │
│   │           │ ─────────> │           │ ─────────> │            │  │
│   │ Trading   │            │  DQN      │            │  Trading   │  │
│   │ Env       │ <───────── │  Agent    │ <───────── │  Env       │  │
│   │ (env.py)  │  reward,   │ (agent.py)│  new state │  (env.py)  │  │
│   │           │  done      │           │            │            │  │
│   └───────────┘            └─────┬─────┘            └────────────┘  │
│                                  │                                    │
│                          experience (s,a,r,s',done)                  │
│                                  │                                    │
│                                  ▼                                    │
│                        ┌──────────────────┐                          │
│                        │  Replay Buffer   │                          │
│                        │ (replay_buffer.py│                          │
│                        │  cap: 100,000)   │                          │
│                        └────────┬─────────┘                          │
│                                  │  sample batch (size 64)           │
│                                  ▼                                    │
│                   ┌──────────────────────────┐                       │
│                   │     DQN Agent Learn       │                       │
│                   │                           │                       │
│                   │  Online Q-Network         │                       │
│                   │  (network.py)             │                       │
│                   │  in: obs_dim              │                       │
│                   │  256 → ReLU               │                       │
│                   │  256 → ReLU               │                       │
│                   │  out: 3 Q-values          │                       │
│                   │            ↕ periodic     │                       │
│                   │  Target Q-Network (copy)  │                       │
│                   └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline (data.py)                       │
│                                                                       │
│  OHLCV Data ──> Feature Engineering ──> StandardScaler ──> Arrays   │
│  (5 assets,      (VWAP, pct changes)    (fit on train,    (train /  │
│  2017–2024)                              transform all)    test)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Evaluation (evaluate.py)                        │
│                                                                       │
│  Trained agent (ε=0) runs on test period 2023-01-01 to 2024-01-01  │
│  Reports: ROI, Sharpe Ratio, Maximum Drawdown, Calmar Ratio         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### `src/config.py` — Hyperparameter Registry
**Responsibility:** Single source of truth for all confirmed hyperparameters and constants.
No magic numbers are permitted elsewhere in `src/`.
**Key values:** assets, date ranges, LR=0.0003, γ=0.99, ε∈[0.01,1.0], batch=64, buffer=100k, episodes=200

---

### `src/data.py` — Data Loading and Feature Engineering
**Responsibility:** Fetch raw OHLCV data, compute VWAP and percentage change features,
apply `StandardScaler` normalization, and produce train/test arrays.
**Inputs:** asset list, date range
**Outputs:** normalized feature arrays (train split / test split)
**Confirmed:** OHLCV + VWAP + pct changes → StandardScaler
**Pending:** exact pct change periods, VWAP window, multi-asset concatenation strategy

---

### `src/env.py` — Custom Gymnasium Trading Environment
**Responsibility:** Implement the `gymnasium.Env` interface over preprocessed price data.
Accepts discrete actions {0=buy, 1=hold, 2=sell}, returns observations and rewards.
**Action space:** `Discrete(3)`
**Observation space:** TBD — depends on feature count and window size (evidence-ledger.md gap)
**Pending:** reward function, position sizing, observation window size, multi-asset scope

---

### `src/network.py` — Q-Network
**Responsibility:** Define `QNetwork(nn.Module)` implementing the confirmed architecture.
**Architecture (confirmed):**
```
Input: obs_dim (TBD)
  Linear(obs_dim, 256) → ReLU
  Linear(256, 256)     → ReLU
  Linear(256, 3)       → Q-values for {buy, hold, sell}
```

---

### `src/replay_buffer.py` — Experience Replay Memory
**Responsibility:** Ring-buffer storing transitions `(s, a, r, s', done)`.
Supports uniform random mini-batch sampling.
**Capacity:** 100,000 transitions (confirmed)
**Batch size:** 64 (confirmed)

---

### `src/agent.py` — DQN Agent
**Responsibility:** Owns the online and target Q-networks. Implements epsilon-greedy action
selection, learning step (sample → TD target → loss → optimizer step), epsilon decay,
and target network updates.
**Confirmed:** dual-network (online + target), ε: 1.0→0.01, LR=0.0003, γ=0.99
**Pending:** optimizer, loss function, epsilon decay schedule, target update frequency

---

### `src/train.py` — Training Entry Point
**Responsibility:** Outer training loop over 200 episodes. Wires together env, agent,
and replay buffer. Logs episode metrics and saves final model checkpoint.
**Episodes:** 200 (confirmed)

---

### `src/evaluate.py` — Evaluation
**Responsibility:** Runs the trained agent in greedy mode (ε=0) over the test period
and computes the four confirmed metrics.
**Test period:** 2023-01-01 to 2024-01-01 (confirmed)
**Metrics:** ROI, Sharpe Ratio, Maximum Drawdown, Calmar Ratio (all confirmed)

---

## Data Flow

```
1. data.py loads OHLCV for AAPL, INTC, META, TQQQ, TSLA (2017-01-01 to 2024-06-01)
2. data.py engineers VWAP and percentage change features
3. data.py fits StandardScaler on training data; transforms train and test sets
4. env.py wraps the training data as a Gymnasium environment
5. train.py runs 200 episodes:
     a. env.reset() → initial observation
     b. agent.select_action(obs) → epsilon-greedy action
     c. env.step(action) → next_obs, reward, done
     d. replay_buffer.push(obs, action, reward, next_obs, done)
     e. if buffer ≥ batch_size: agent.learn(replay_buffer.sample(64))
     f. periodic: copy online network weights to target network
6. evaluate.py loads trained model, runs on test env (ε=0)
7. evaluate.py reports ROI, Sharpe, Max Drawdown, Calmar Ratio
```

---

## Key Design Decisions

See `docs/adr/` for formally documented decisions.

| Decision | Status | ADR |
|----------|--------|-----|
| Evidence-bounded reconstruction (no invented details) | Accepted | ADR-001 |
| Documentation-first implementation order | Accepted | ADR-001 |
| Target network (separate from online network) | Confirmed from paper | — |
| Experience replay buffer | Confirmed from paper | — |
| Optimizer | Pending — gap in evidence-ledger.md | — |
| Loss function | Pending — gap in evidence-ledger.md | — |
| Reward function | Pending — gap in evidence-ledger.md | — |
| Observation window | Pending — gap in evidence-ledger.md | — |

---

## External Dependencies (Planned)

| Dependency | Purpose | Version |
|------------|---------|---------|
| `torch` | Q-network implementation | TBD |
| `gymnasium` | Trading environment interface | TBD |
| `numpy` | Array operations | TBD |
| `pandas` | Time-series data handling | TBD |
| `scikit-learn` | StandardScaler | TBD |
| `yfinance` or CSV | OHLCV data source | TBD |

---

*Last updated: 2026-03-14*
*Architecture status: Designed — not yet implemented*
