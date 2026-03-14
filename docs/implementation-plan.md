# Implementation Plan

**Status:** Documentation phase — no code written yet.
**Constraint:** This plan describes intended structure only. All modules referencing unresolved
gaps from `docs/evidence-ledger.md` must have those gaps resolved (or assumptions declared)
before implementation begins.

---

## Planned Module Structure

```
src/
├── config.py           # Hyperparameter registry — all confirmed values live here
├── data.py             # Data loading, feature engineering, preprocessing
├── env.py              # Custom Gymnasium trading environment
├── network.py          # Q-network architecture (256 → 256 ReLU)
├── replay_buffer.py    # Experience replay memory
├── agent.py            # DQN agent: epsilon-greedy policy, learning step
├── train.py            # Training loop entry point
└── evaluate.py         # Evaluation metrics: ROI, Sharpe, Max Drawdown, Calmar

tests/
├── unit/
│   ├── test_data.py            # Feature engineering, scaling, shapes
│   ├── test_env.py             # Environment step, reset, action space
│   ├── test_network.py         # Forward pass output shapes
│   ├── test_replay_buffer.py   # Sample correctness, capacity behavior
│   └── test_agent.py           # Epsilon decay, action selection
└── integration/
    ├── test_training_loop.py   # Short training run end-to-end
    └── test_evaluate.py        # Metric computation on known data
```

---

## Module Descriptions

### `src/config.py` — Hyperparameter Registry

Central location for all confirmed hyperparameters. No magic numbers anywhere else in `src/`.
Imports from this file wherever a hyperparameter is needed.

**Confirmed values to register:**

| Key | Value | Status |
|-----|-------|--------|
| `ASSETS` | `["AAPL", "INTC", "META", "TQQQ", "TSLA"]` | Confirmed |
| `START_DATE` | `"2017-01-01"` | Confirmed |
| `END_DATE` | `"2024-06-01"` | Confirmed |
| `TEST_START` | `"2023-01-01"` | Confirmed |
| `TEST_END` | `"2024-01-01"` | Confirmed |
| `LEARNING_RATE` | `0.0003` | Confirmed |
| `GAMMA` | `0.99` | Confirmed |
| `EPSILON_START` | `1.0` | Confirmed |
| `EPSILON_END` | `0.01` | Confirmed |
| `BATCH_SIZE` | `64` | Confirmed |
| `REPLAY_BUFFER_SIZE` | `100000` | Confirmed |
| `TRAINING_EPISODES` | `200` | Confirmed |
| `HIDDEN_SIZE` | `256` | Confirmed |
| `ACTION_COUNT` | `3` | Confirmed |
| `EPSILON_DECAY` | TBD | Gap — see evidence-ledger.md |
| `TARGET_UPDATE_FREQ` | TBD | Gap — see evidence-ledger.md |
| `OPTIMIZER` | TBD | Gap — see evidence-ledger.md |

---

### `src/data.py` — Data Loading and Feature Engineering

**Responsibilities:**
- Fetch daily OHLCV data for all five assets over 2017-01-01 to 2024-06-01
- Engineer VWAP and percentage change features (exact period TBD — see evidence-ledger.md)
- Apply `StandardScaler` normalization
- Split into train and test sets (train: up to 2023-01-01; test: 2023-01-01 to 2024-01-01)

**Key design decisions pending:**
- Data source (yfinance, pandas-datareader, CSV files?)
- Scaler fit location (fit on train only, transform train and test separately)
- Feature concatenation strategy across assets

**Inputs:** date range, asset list
**Outputs:** normalized feature arrays, train/test split indices

---

### `src/env.py` — Custom Gymnasium Trading Environment

**Responsibilities:**
- Implement `gymnasium.Env` interface: `reset()`, `step(action)`, `observation_space`, `action_space`
- Accept action ∈ {0=buy, 1=hold, 2=sell}
- Return observation vector for current timestep
- Compute and return reward signal (formulation TBD — see evidence-ledger.md)
- Track episode state: current position, portfolio value, timestep

**Key design decisions pending:**
- Reward function (log return, PnL delta, Sharpe-based?)
- Observation window size (single timestep vs. rolling window)
- Whether environment manages one asset or all five simultaneously
- Position sizing rules (all-in, fractional, fixed units)
- Transaction cost modeling

**Observation space:** depends on feature count and window size (gap)
**Action space:** `gymnasium.spaces.Discrete(3)`

---

### `src/network.py` — Q-Network

**Responsibilities:**
- Define `QNetwork(nn.Module)` with the confirmed architecture
- Architecture: `Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 3)`
- No additional regularization unless confirmed from paper

**Confirmed architecture:**
```
Input: observation vector (dim TBD pending env design)
  └─ Linear → 256 → ReLU
       └─ Linear → 256 → ReLU
            └─ Linear → 3 (Q-values for buy/hold/sell)
```

**Parameters pending:** input dimension depends on state space resolution (evidence-ledger.md gap)

---

### `src/replay_buffer.py` — Experience Replay

**Responsibilities:**
- Store transitions `(state, action, reward, next_state, done)` up to capacity 100,000
- Support uniform random sampling of mini-batches of size 64
- Overwrite oldest transitions when at capacity (ring buffer)

**All design details for this module are either confirmed or standard:**
- Capacity: 100,000 (confirmed)
- Batch size: 64 (confirmed)
- Sampling: uniform random (standard DQN; no prioritization confirmed)

---

### `src/agent.py` — DQN Agent

**Responsibilities:**
- Maintain online Q-network and target Q-network (both `QNetwork` instances)
- Implement epsilon-greedy action selection
- Implement learning step: sample from replay buffer, compute TD targets, update online network
- Decay epsilon from 1.0 → 0.01 (schedule TBD)
- Periodically copy online network weights to target network (frequency TBD)

**Confirmed:**
- Two networks: online (updated every step) and target (updated periodically)
- Epsilon range: 1.0 → 0.01
- Optimizer: TBD (gap)
- Loss: TBD (gap)
- Target update frequency: TBD (gap)

---

### `src/train.py` — Training Loop Entry Point

**Responsibilities:**
- Instantiate data, environment, agent, replay buffer
- Run outer episode loop (200 episodes confirmed)
- For each episode: reset environment, run until terminal, collect experience, train agent
- Log training metrics per episode
- Save model checkpoint after training

**Episode loop structure (planned):**
```
for episode in range(TRAINING_EPISODES):       # 200 confirmed
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)    # epsilon-greedy
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) >= BATCH_SIZE:
            agent.learn(replay_buffer.sample(BATCH_SIZE))
        state = next_state
    agent.decay_epsilon()                      # schedule TBD
```

---

### `src/evaluate.py` — Evaluation Metrics

**Responsibilities:**
- Run trained agent on the test period (2023-01-01 to 2024-01-01) in evaluation mode (ε=0)
- Compute and report:
  - **ROI** — total return over test period
  - **Sharpe Ratio** — risk-adjusted return (annualized)
  - **Maximum Drawdown** — largest peak-to-trough portfolio decline
  - **Calmar Ratio** — annualized return divided by maximum drawdown

All four metrics are confirmed from the paper/report. Benchmark comparison strategy TBD.

---

## Training Pipeline Data Flow

```
Raw OHLCV data (yfinance or CSV)
  └─ data.py: engineer VWAP + pct_change features
       └─ data.py: StandardScaler fit (train) / transform (train + test)
            └─ env.py: TradingEnv wraps scaled data
                 └─ agent.py: DQN agent interacts with env
                      └─ replay_buffer.py: stores transitions
                           └─ agent.py: learns from sampled batches
                                └─ train.py: logs metrics per episode
                                     └─ evaluate.py: reports test metrics
```

---

## Experiment Workflow

1. Confirm all evidence-ledger.md gaps are resolved (assumption or source)
2. Implement `config.py` — no other module written until this exists
3. Implement `data.py` and verify feature shapes with unit tests
4. Implement `env.py` and verify Gymnasium contract with unit tests
5. Implement `network.py` and verify output shapes with unit tests
6. Implement `replay_buffer.py` and verify sampling behavior with unit tests
7. Implement `agent.py` integrating network and replay buffer
8. Implement `train.py` — run integration test with 5 episodes before full training
9. Run full 200-episode training
10. Implement `evaluate.py` and report test metrics

---

## Unresolved Gaps Blocking Implementation

The following gaps in `docs/evidence-ledger.md` must be resolved before the indicated module
can be written. Do not begin implementation of a blocked module.

| Gap | Blocks |
|-----|--------|
| Reward function formulation | `env.py` |
| Observation window size | `env.py`, `network.py` (input dim) |
| Epsilon decay schedule | `agent.py` |
| Target network update frequency | `agent.py` |
| Optimizer choice | `agent.py` |
| Percentage change periods | `data.py`, `env.py` |
| VWAP calculation method | `data.py` |
| Position sizing rules | `env.py` |
| Train/test split method | `data.py` |
| Multi-asset observation structure | `env.py`, `network.py` |

---

*Last updated: 2026-03-14*
*Status: Documentation phase — awaiting gap resolution before implementation*
