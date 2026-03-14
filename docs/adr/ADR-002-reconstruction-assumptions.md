# ADR-002: Initial Implementation Assumptions for First Coding Pass

**Date:** 2026-03-14
**Status:** Accepted
**Deciders:** Ruben Aleman (BUDDY26)
**Supersedes:** Nothing
**Governed by:** ADR-001 (Evidence-Bounded Reconstruction Philosophy)

---

## Context

ADR-001 established the evidence-bounded reconstruction philosophy: confirmed values are
constraints; gaps must be declared as assumptions before dependent code is written. As of
the documentation phase completion, 13 confirmed gaps remained in `docs/evidence-ledger.md`
Category 2. None could be resolved from the surviving paper or report.

This ADR documents the complete set of reconstruction assumptions adopted for the first
implementation pass. Each assumption is the minimum-intervention choice: the most common,
most standard, most architecturally consistent default supported by DQN literature. No
assumption constitutes an improvement over base DQN — the goal is a faithful reconstruction,
not an optimized one.

**All 13 gaps are resolved here, yielding 15 discrete assumptions (some gaps share a decision).**

---

## Decision

We adopt the 15 assumptions below as the operational basis for the first implementation pass.
These assumptions are not permanent. If new evidence surfaces from the original paper, report,
or any other credible source, the relevant assumption is annotated in `docs/evidence-ledger.md`
and a new ADR is written if the change is architecturally significant.

---

## Data Assumptions

### A-D1 — Percentage Change: 1-Day, Applied to All OHLCV Columns

**Resolves:** Gap D1
**Value:** 1-day percentage change (`pct_change(1)`) on each of the five OHLCV columns.
Together with raw OHLCV (5) and VWAP (1), this yields **11 features per asset per timestep**.
**Why this over alternatives:** "Percentage changes" without a period qualifier defaults to
1-day in daily-bar financial modeling. Multi-period variants require a window hyperparameter
not present in the source documents.

---

### A-D2 — VWAP Approximation: (High + Low + Close) / 3

**Resolves:** Gap D2
**Value:** `vwap_t = (high_t + low_t + close_t) / 3`
**Why this over alternatives:** True VWAP is uncomputable from daily bars alone. The typical
price `(H + L + C) / 3` is the standard and most cited daily VWAP approximation in technical
analysis and academic literature. No other daily-bar approximation has comparable prevalence.
An alternative — `(O + H + L + C) / 4` — is less common and would make no material difference
after StandardScaler normalization, but introduces an unjustified deviation from convention.

---

### A-D3 — Training Period: 2017-01-01 to 2022-12-31; No Validation Split

**Resolves:** Gap D3
**Value:**
- Train: 2017-01-01 to 2022-12-31 (inclusive)
- Test: 2023-01-01 to 2024-01-01 (confirmed — not an assumption)
- Excluded from evaluation: 2024-01-01 to 2024-06-01 (downloaded but unused)
- No validation split

**Why this over alternatives:** The test period start (2023-01-01) defines the training
cutoff as 2022-12-31. A validation split would require a confirmed holdout date not present
in the source documents. Absence of documentation for validation is treated as evidence of none.

---

### A-D4 / A-E5 — Independent Per-Asset Environments; 11-Feature Single-Asset Observations

**Resolves:** Gaps D4, E5
**Value:** Five independent Gymnasium environments (one per asset). The observation vector
at each step contains only the 11 features of the current asset. One DQN model is trained
per asset (five separate training runs, one per asset ticker).
**Why this over alternatives:** The confirmed action space is {buy, hold, sell} — a scalar
action with no asset selector. This is most naturally interpreted as a single-asset decision.
A joint 5-asset observation (55 features) would require defining action semantics that the
confirmed action space cannot express cleanly (e.g., does "buy" apply to one asset or all
five?). Independent per-asset training is the standard approach in academic DQN trading work
with multiple tickers and a scalar action space.

---

## Environment Assumptions

### A-E1 — Reward: Position × Daily Return (Binary Long/Flat Model)

**Resolves:** Gap E1
**Value:** `reward_t = position_t × (close_{t+1} - close_t) / close_t`
where `position_t ∈ {1, 0}`. Buy sets position to 1; sell sets position to 0; hold maintains
current position.
**Why this over alternatives:**

| Alternative | Reason Not Chosen |
|-------------|------------------|
| Log return: `log(close_{t+1} / close_t) × position_t` | Mathematically equivalent for small returns; provides no reconstruction advantage |
| Realized PnL only (sparse) | Would yield zero reward until position closed; impractical for long hold sequences |
| Sharpe-based reward shaping | Introduces a non-confirmed reward modification; violates reconstruction discipline |
| Portfolio delta with sizing | Requires confirmed position size (not available without A-E4) |

The position × daily_return formulation is the most widely cited in DQN trading papers with
a three-action discrete space and binary position model. It is dense, non-sparse, and directly
optimizes for what the evaluation metrics measure.

---

### A-E2 — Observation: Single Timestep (No Rolling Window)

**Resolves:** Gap E2
**Value:** Observation at step t = feature vector for day t only (11 values after scaling).
No rolling window. No temporal stacking.
**Why this over alternatives:** The confirmed architecture is a feed-forward MLP with no
recurrent or convolutional structure. A rolling window is possible with an MLP (by flattening
`window × features` into the input vector) but introduces a window-size hyperparameter with
no source basis. The percentage-change features already encode temporal movement direction
within the single-step observation. Adopting a window without a confirmed size would expand
the network input dimension (and thus the effective architecture) beyond what the source
documents describe.

---

### A-E3 — Initial Capital: $10,000

**Resolves:** Gap E3
**Value:** $10,000 in cash at the start of each episode; zero shares held.
**Why this over alternatives:** All four confirmed evaluation metrics (ROI, Sharpe, MDD, Calmar)
are scale-invariant under proportional sizing. This assumption carries no risk for metric
computation. $10,000 is the most common initial capital in academic simulations.

---

### A-E4 — Position Sizing: Binary All-In / All-Out

**Resolves:** Gap E4
**Value:** Buy = invest 100% of available cash. Sell = liquidate 100% of position to cash.
Hold = no trade. No fractional sizing, no fixed-unit sizing, no leverage.
**Why this over alternatives:** Binary all-in/all-out is the simplest sizing model consistent
with a three-action space and requires no additional sizing hyperparameter. Fractional or
fixed-unit sizing would require a confirmed fraction or unit size not present in the source
documents. Binary sizing is the default assumption in academic DQN trading papers that specify
only the action labels without a sizing rule.

---

### A-E6 / A-E7 — Episode: Full Training Sequence; Terminal = Last Training Day

**Resolves:** Gaps E6, E7
**Value:**
- Episode start: first trading day of training period (≥ 2017-01-01)
- Episode end (terminal): last trading day of training period (≤ 2022-12-31)
- Each of 200 episodes is a full traversal of the training sequence
- No sliding window, no random start offset, no early termination

**Why this over alternatives:** 200 episodes over ~1,500 training days gives a full-sequence
structure that naturally exposes the agent to all market regimes (2018 correction, 2020
pandemic, 2021 recovery, 2022 bear market) in every episode. Sliding windows or random starts
would require a confirmed window length not present in the source documents. Early termination
is unnecessary because binary all-in sizing with no leverage prevents bankruptcy.

---

### A-E8 — Transaction Costs: Zero (Frictionless)

**Resolves:** Gap E8
**Value:** No transaction cost applied to any buy or sell action. Reward signal contains no
friction term.
**Why this over alternatives:** Not mentioned in paper or report. Absence of specification
is the standard basis for a frictionless assumption in academic simulations. Introducing a
specific cost rate (0.1%, 0.2%) would constitute an unconstrained choice with no source basis.
**Known limitation:** TQQQ is a 3× leveraged ETF with meaningful daily expense ratio and spread.
A frictionless model will overestimate achievable TQQQ returns. This is a known reconstruction
limitation, not a modeling error — it is documented rather than corrected.

---

## Training Assumptions

### A-T1 — Epsilon Decay: Linear Over 200 Episodes

**Resolves:** Gap T1
**Value:** `ε(e) = max(0.01, 1.0 − 0.00495 × e)` where e is the 0-based episode index.
Decay rate = (1.0 − 0.01) / 200 = 0.00495 per episode.
**Why this over alternatives:**

| Alternative | Reason Not Chosen |
|-------------|------------------|
| Exponential: `ε(e) = 0.01 + 0.99 × exp(−e / τ)` | Requires decay constant τ with no source basis |
| Step decay (halve every N episodes) | Requires step size N with no source basis |
| Per-step decay | More common in step-counted environments; episodic decay matches episodic structure |

Linear over 200 episodes is the minimum-parameter schedule consistent with the two confirmed
endpoints. The agent reaches near-greedy behavior (ε = 0.01) exactly at the last episode.

---

### A-T2 / A-T3 — Target Network: Hard Copy Every 10 Episodes

**Resolves:** Gaps T2, T3
**Value:** Full hard copy of online network weights to target network every 10 training
episodes (≈ every 15,000 environment steps at ~1,500 steps per training episode).
**Why this over alternatives:**

| Alternative | Reason Not Chosen |
|-------------|------------------|
| Soft update (τ-weighted): `θ_target ← τθ_online + (1−τ)θ_target` | Characteristic of DDPG/TD3, not standard DQN |
| Every episode | Potentially too frequent; target changes before agent has converged |
| Every 50 episodes | Too infrequent; target becomes stale relative to learned policy |

Hard copy is specified in the original DQN paper (Mnih et al. 2015). Every 10 episodes
(~15,000 steps) is directly analogous to the original's 10,000-step update frequency given
the training episode length in this setting.

---

### A-T4 — Optimizer: Adam with lr=0.0003

**Resolves:** Gap T4
**Value:** `torch.optim.Adam(network.parameters(), lr=0.0003)` with default PyTorch
parameters (β₁=0.9, β₂=0.999, ε=1e-8).
**Why this over alternatives:**

| Alternative | Reason Not Chosen |
|-------------|------------------|
| RMSProp (original DQN) | Original DQN used lr=0.00025; the confirmed lr=0.0003 is atypical for RMSProp but canonical for Adam |
| SGD | No momentum, poor convergence without careful tuning; not used in modern DQN |

The confirmed learning rate (0.0003 = 3e-4) is widely recognized as "the Adam learning rate
for reinforcement learning." Its presence is a strong contextual indicator that Adam was used.

---

### A-T5 — Loss Function: MSE

**Resolves:** Gap T5
**Value:** `torch.nn.MSELoss()` between predicted Q-values and Bellman TD targets.
**Why this over alternatives:**

| Alternative | Reason Not Chosen |
|-------------|------------------|
| Huber loss / SmoothL1 | An improvement over base DQN (clips large gradients); adopting it without evidence means reconstructing a variant, not the original |
| MAE | Not used in DQN literature |

MSE matches the original DQN paper. Huber loss is a well-known improvement, but improvements
are excluded from the first reconstruction pass under ADR-001.

---

## Derived Architectural Consequence

The combination of assumptions A-D1, A-D2, A-E2, and A-D4 fully determines the network
input dimension:

```
obs_dim = n_raw_features + n_pct_change_features + n_vwap_features
        = 5 (OHLCV) + 5 (1-day pct_change of OHLCV) + 1 (VWAP)
        = 11

Network: Linear(11, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 3)
```

This is a derived fact. Any change to A-D1, A-D2, A-E2, or A-D4 requires recomputing `obs_dim`
and updating `config.py` accordingly.

---

## Consequences

### Positive
- All 13 gaps are resolved. Implementation may begin according to `docs/implementation-plan.md`.
- Every assumption is traceable, justified, and carries an explicit risk statement.
- The reconstruction is complete without inventing undocumented details.
- The obs_dim = 11 is fully determined; no blocking unknowns remain for `config.py`.

### Negative / Trade-offs
- **A-D4 / A-E5** carries the highest architectural uncertainty. If the original used a joint
  multi-asset observation, the input space and policy structure would differ substantially.
- **A-E1** (reward) is the highest behavioral uncertainty. A different reward signal could
  produce an entirely different learned policy even with identical hyperparameters.
- **A-E8** (frictionless) will overestimate TQQQ returns. This is documented and accepted.
- **A-T1** (linear decay) may explore too much in early episodes or too little in late ones
  depending on the original schedule. The learned policy at episode 200 will be correct, but
  the convergence path may differ.

### Neutral
- These assumptions apply only to the first implementation pass. If the trained model produces
  metrics that diverge significantly from any values reported in the paper, the assumptions
  should be revisited systematically, starting with A-E1 (reward), A-D4 (multi-asset scope),
  and A-E2 (window size), in that order.
- The document is not re-written when assumptions are revised. New ADRs (ADR-003, etc.) are
  created to document deviations.

---

## References

- `docs/evidence-ledger.md` — full assumption entries with risk statements (Category 3)
- `docs/adr/ADR-001-template.md` — reconstruction philosophy governing this ADR
- `docs/implementation-plan.md` — module dependency order using these assumptions
- Mnih et al. (2015), "Human-level control through deep reinforcement learning" — original DQN
- Kingma & Ba (2014), "Adam: A method for stochastic optimization" — optimizer reference

---

*ADRs are immutable once Accepted. To revise any assumption here, create ADR-003 referencing
the specific assumption ID (e.g., "Revises A-E1 from ADR-002") and update the corresponding
entry in docs/evidence-ledger.md Category 3 with an annotation.*
