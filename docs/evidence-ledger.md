# Evidence Ledger

**Purpose:** Central record of all technical facts, gaps, and reconstruction assumptions for this
project. Every implementation decision must be traceable to one of the three categories below.

**Discipline:** Implementation details that appear in neither the confirmed nor the assumption
columns must not be silently chosen. They must first be added here before any dependent code
is written.

---

## Category 1 — Confirmed Evidence

These details are taken directly from the surviving final project report and/or the original
research paper. They are treated as ground truth and must not be changed during reconstruction
without a new ADR explaining the deviation.

### Data

| Detail | Value | Source |
|--------|-------|--------|
| Assets | AAPL, INTC, META, TQQQ, TSLA | Paper + Report |
| Full date range | 2017-01-01 to 2024-06-01 | Paper + Report |
| Test period | 2023-01-01 to 2024-01-01 | Paper + Report |
| Data frequency | Daily | Paper + Report |
| Raw input features | Open, High, Low, Close, Volume (OHLCV) | Paper + Report |
| Engineered feature: VWAP | Volume-weighted average price (approximated — see A-D2) | Paper + Report |
| Engineered feature: pct changes | Percentage changes (period declared in assumption A-D1) | Paper + Report |
| Preprocessing | StandardScaler (zero mean, unit variance) | Paper + Report |

### Environment

| Detail | Value | Source |
|--------|-------|--------|
| Action space | Discrete: buy / hold / sell | Paper + Report |
| Action cardinality | 3 | Paper + Report |

### Q-Network Architecture

| Detail | Value | Source |
|--------|-------|--------|
| Hidden layer 1 | 256 units | Paper + Report |
| Hidden layer 2 | 256 units | Paper + Report |
| Activation function | ReLU | Paper + Report |
| Output layer | One Q-value per action (3 outputs) | Inferred from action cardinality |

### Training Hyperparameters

| Detail | Value | Source |
|--------|-------|--------|
| Learning rate (α) | 0.0003 | Paper + Report |
| Discount factor (γ) | 0.99 | Paper + Report |
| Epsilon initial | 1.0 | Paper + Report |
| Epsilon final | 0.01 | Paper + Report |
| Batch size | 64 | Paper + Report |
| Replay buffer capacity | 100,000 | Paper + Report |
| Training episodes | 200 | Paper + Report |

### Evaluation

| Detail | Value | Source |
|--------|-------|--------|
| Test period | 2023-01-01 to 2024-01-01 | Paper + Report |
| Metric 1 | ROI (total return) | Paper + Report |
| Metric 2 | Sharpe Ratio | Paper + Report |
| Metric 3 | Maximum Drawdown | Paper + Report |
| Metric 4 | Calmar Ratio | Paper + Report |

---

## Category 2 — Confirmed Gaps

**All 13 gaps have been resolved for the first implementation pass via declared assumptions in
Category 3. Status column indicates resolution type and assumption ID.**

### Data Gaps

| ID | Gap | Status | Resolution |
|----|-----|--------|------------|
| D1 | Exact percentage change periods | **ASSUMED** | → A-D1 |
| D2 | VWAP calculation window or method | **ASSUMED** | → A-D2 |
| D3 | Train/test split boundary (train end date) | **ASSUMED** | → A-D3 |
| D4 | Multi-asset feature structure: concatenated or per-asset | **ASSUMED** | → A-D4 |

### Environment Gaps

| ID | Gap | Status | Resolution |
|----|-----|--------|------------|
| E1 | Reward function formulation | **ASSUMED** | → A-E1 |
| E2 | Observation window size | **ASSUMED** | → A-E2 |
| E3 | Initial portfolio capital | **ASSUMED** | → A-E3 |
| E4 | Position sizing rule | **ASSUMED** | → A-E4 |
| E5 | Multi-asset: independent or joint environment | **ASSUMED** | → A-D4 (same decision) |
| E6 | Episode structure | **ASSUMED** | → A-E6 |
| E7 | Terminal condition | **ASSUMED** | → A-E6 (same decision) |
| E8 | Transaction cost modeling | **ASSUMED** | → A-E8 |

### Training Gaps

| ID | Gap | Status | Resolution |
|----|-----|--------|------------|
| T1 | Epsilon decay schedule | **ASSUMED** | → A-T1 |
| T2 | Target network update rule (hard vs. soft) | **ASSUMED** | → A-T2 |
| T3 | Target network update frequency | **ASSUMED** | → A-T3 |
| T4 | Optimizer | **ASSUMED** | → A-T4 |
| T5 | Loss function | **ASSUMED** | → A-T5 |

---

## Category 3 — Reconstruction Assumptions

All 15 assumptions below are declared for the first implementation pass. Each is traceable to a
specific gap, justified by DQN literature conventions or first-principles reasoning, and carries
an explicit risk statement. All are documented in full in `docs/adr/ADR-002-reconstruction-assumptions.md`.

---

### A-D1: 1-Day Percentage Change for All OHLCV Columns

**Value chosen:** 1-day percentage change (`pct_change(1)`) applied independently to each of the
five OHLCV columns (Open, High, Low, Close, Volume).
**Resolves gap:** D1
**Justification:** "Percentage changes" in financial DQN literature almost universally refers to
daily (1-period) returns. This is the minimal, most interpretable choice and requires no
additional window hyperparameter. Multi-period variants (e.g., 5-day) are not the default and
would require explicit documentation. Together with the 5 raw OHLCV values and 1 VWAP value,
this yields 11 features per asset per timestep.
**Risk:** If the original used multi-period changes (e.g., 5-day), the feature count and
normalization behavior would differ. Detectable if reproduced results diverge significantly from
reported metrics.
**ADR:** ADR-002

---

### A-D2: VWAP as Daily Typical Price — (High + Low + Close) / 3

**Value chosen:** VWAP is approximated as `(High + Low + Close) / 3` for each trading day.
**Resolves gap:** D2
**Justification:** True VWAP requires intraday tick data and cannot be computed from daily OHLCV
bars. The standard and universally cited approximation for daily bar data is the "typical price":
`(H + L + C) / 3`. This is the conventional daily VWAP proxy in technical analysis and is
reproduced across academic and practitioner literature. No other daily-bar VWAP approximation has
comparable prevalence.
**Risk:** If the original used a different approximation (e.g., cumulative rolling VWAP or a
streaming window), the feature values would differ. The direction and magnitude of any divergence
would be small given StandardScaler normalization.
**ADR:** ADR-002

---

### A-D3: Training Period Is 2017-01-01 to 2022-12-31 (Inclusive)

**Value chosen:** Training data = 2017-01-01 through 2022-12-31. Test data = 2023-01-01 through
2024-01-01 (confirmed). Data from 2024-01-01 to 2024-06-01 is downloaded but excluded from
training and evaluation. No separate validation split is created.
**Resolves gap:** D3
**Justification:** The test period is confirmed (2023-01-01 to 2024-01-01). The training period
must end before the test period begins. 2022-12-31 is the last calendar day before the test
period. The absence of a documented validation split is treated as evidence that none was used
(validation sets would typically be mentioned in the experimental description if present).
**Risk:** If a validation period was held out within the training range (e.g., 2022 used for
validation), the effective training data would be smaller. The model would then have seen fewer
episodes of the 2022 data, which included significant market volatility (the 2022 bear market).
**ADR:** ADR-002

---

### A-D4 / A-E5: Independent Per-Asset Environments, Per-Asset Observations

**Value chosen:** Each of the five assets is modeled as a separate, independent Gymnasium
environment. The agent observes features for one asset at a time. One DQN model is trained
per asset (five independent training runs). The observation vector contains only the features
of the current asset — 11 values (5 OHLCV + 5 pct_changes + 1 VWAP) — not a concatenation
across all five assets.
**Resolves gaps:** D4, E5
**Justification:** A joint observation over 5 assets would yield 55 input features and require
defining an action semantics (does the action apply to all assets simultaneously? to one
selected asset?). The confirmed action space is simply {buy, hold, sell} with no asset selector,
which is most naturally interpreted as applying to a single asset. The 256-unit hidden layers
are also more consistent with an 11-dimensional input than a 55-dimensional one (though both
are feasible). Independent per-asset training is the standard simplification in academic DQN
trading projects.
**Risk:** If the original used a joint multi-asset observation, the input dimensionality, state
space, and generalization behavior would all differ. This is the most architecturally uncertain
assumption in the reconstruction.
**ADR:** ADR-002

---

### A-E1: Reward = Position × Daily Return (Binary Long/Flat Model)

**Value chosen:** At each timestep t, reward = `position_t × (close[t+1] - close[t]) / close[t]`,
where `position_t ∈ {1, 0}`. A buy action sets position = 1. A sell action sets position = 0.
A hold action maintains current position. No transaction cost is applied to the reward signal
(see A-E8).
**Resolves gap:** E1
**Justification:** This is the canonical reward formulation for single-asset DQN trading agents
with a three-action discrete action space. It directly aligns the agent's reward signal with
portfolio return, provides a dense (non-sparse) signal at every step, and requires no additional
hyperparameters. It is the most widely reproduced formulation in the DQN trading literature
(e.g., resembles formulations in Carta et al. 2021, Liu et al. 2020, and similar work).
**Risk:** If the original used a different reward (e.g., log return, Sharpe-based shaping,
or realized-only PnL), the learned policy would differ. Log return is a common alternative;
under Standard scaling, behavioral difference would be small but present. Detectable from
training curve shape and final metric values.
**ADR:** ADR-002

---

### A-E2: Single-Timestep Observation (No Rolling Window)

**Value chosen:** The observation at step t is the feature vector for day t only:
`obs_t = [open_t, high_t, low_t, close_t, volume_t, vwap_t, pct_open_t, pct_high_t, pct_low_t, pct_close_t, pct_vol_t]`
(11 values after StandardScaler normalization). No rolling window is applied.
**Resolves gap:** E2
**Justification:** The confirmed architecture is a feed-forward MLP (256→256). MLPs process
fixed-size inputs without inherent temporal structure. A rolling window is possible with an MLP
(flattening window×features into the input) but would require specifying the window size as an
additional hyperparameter not present in the source documents. The single-timestep formulation
requires no additional parameters and is architecturally consistent with a plain MLP. Temporal
context is partially encoded in the percentage change features, which already capture day-over-day
movement direction.
**Risk:** If the original used a rolling window, the network input dimension would be
`window_size × 11` instead of 11, and the learned representations would differ substantially.
A rolling window of 10 or 20 days is the most common alternative; if results diverge this should
be investigated first.
**ADR:** ADR-002

---

### A-E3: Initial Portfolio Capital of $10,000

**Value chosen:** Each episode begins with $10,000 in cash, zero shares held.
**Resolves gap:** E3
**Justification:** $10,000 is the most common initial capital in academic trading simulations.
All four confirmed evaluation metrics (ROI, Sharpe Ratio, Maximum Drawdown, Calmar Ratio) are
scale-invariant under proportional position sizing, so the exact initial capital does not affect
any reported metric value. This assumption carries essentially zero reconstruction risk for the
evaluation phase.
**Risk:** Negligible for metric computation. Relevant only if absolute portfolio values are
reported, which none of the four confirmed metrics require.
**ADR:** ADR-002

---

### A-E4: Binary All-In / All-Out Position Sizing

**Value chosen:** When action = buy: invest 100% of available cash in the current asset.
When action = sell: liquidate 100% of the current position to cash. When action = hold:
no trade executed. No fractional sizing, no fixed unit sizing. Leverage not permitted.
**Resolves gap:** E4
**Justification:** Binary all-in/all-out is the simplest position model consistent with a
three-action discrete space. It requires no additional sizing hyperparameter and is the default
assumption in academic DQN trading papers that use buy/hold/sell without specifying a sizing rule.
Fractional or fixed-unit sizing would require an additional confirmed parameter (fraction,
unit size) not present in the source documents.
**Risk:** If the original used fractional sizing (e.g., 50% of capital per trade), portfolio
volatility and drawdown characteristics would differ. Buy/sell frequency in the learned policy
may also differ if the agent's effective reward magnitude depends on position size.
**ADR:** ADR-002

---

### A-E6 / A-E7: Episode = Full Training Sequence; Terminal = Last Day

**Value chosen:** Each episode begins at the first trading day of the training period
(2017-01-01 or nearest trading day) and ends at the last trading day of the training period
(2022-12-31 or nearest trading day). There is no sliding window, no random start, and no
early termination. 200 episodes = 200 complete traversals of the training sequence.
**Resolves gaps:** E6, E7
**Justification:** With 200 confirmed training episodes and approximately 1,500 training days
(2017–2022), a full-sequence episode structure is the natural interpretation. It provides the
agent with complete exposure to all market conditions in each episode, including high-volatility
periods (2020 pandemic, 2022 bear market). Sliding windows or random episode starts would
require specifying an additional window length parameter not present in the source documents.
**Risk:** If the original used shorter episode windows (e.g., random 252-day windows within
the training range), the agent would see more diverse starting conditions but fewer steps per
episode. This would change sample diversity and potentially convergence speed. Early stopping
on portfolio bankruptcy is not modeled — under binary all-in sizing without leverage, bankruptcy
is not reachable.
**ADR:** ADR-002

---

### A-E8: Zero Transaction Costs (Frictionless Model)

**Value chosen:** No transaction cost is applied when buy or sell actions are executed.
The reward signal contains no trading friction term.
**Resolves gap:** E8
**Justification:** Transaction costs are not mentioned in either the paper or the report.
Absence of specification is the standard reason to assume frictionless execution in academic
trading simulations. Introducing a transaction cost without a confirmed value would require
choosing a rate (typically 0.1%–0.2%) that has no source basis. The frictionless assumption is
also more favorable to the agent and therefore constitutes a conservative reconstruction choice
(it cannot inflate difficulty, only reduce it).
**Risk:** TQQQ is a 3× leveraged ETF with non-trivial spread and management fees. A frictionless
model will overestimate achievable returns for TQQQ specifically. If the original modeled
costs for TQQQ, the learned policy would trade less aggressively on that asset. This risk is
noted but not correctable without a confirmed cost value.
**ADR:** ADR-002

---

### A-T1: Linear Epsilon Decay Over 200 Episodes

**Value chosen:** Epsilon decays linearly from 1.0 to 0.01 over the course of 200 training
episodes. Rate = (1.0 − 0.01) / 200 = 0.00495 per episode.
Decay rule: `ε(e) = max(0.01, 1.0 − 0.00495 × e)` where e is the current episode index (0-based).
**Resolves gap:** T1
**Justification:** Linear decay is the simplest schedule consistent with two confirmed endpoints.
It requires no additional hyperparameters (no time constant τ, no decay fraction). Exponential
decay is an equally common alternative but requires a decay rate not present in the source
documents. Linear over the full training horizon ensures the agent transitions from full
exploration to near-greedy behavior smoothly and deterministically.
**Risk:** If the original used faster decay (e.g., reaching 0.01 by episode 100), the agent
would exploit earlier, potentially resulting in earlier convergence or different final policy.
Exponential decay would result in faster early decay and slower final approach to 0.01. Either
alternative would produce different exploration behavior in the first half of training.
**ADR:** ADR-002

---

### A-T2 / A-T3: Hard Target Network Update Every 10 Episodes

**Value chosen:** Target network weights are updated by a hard copy (full parameter copy)
from the online network. The copy is performed once every 10 training episodes
(approximately every 15,000 environment steps given ~1,500 steps per training episode).
**Resolves gaps:** T2, T3
**Justification:** Hard copy matches the original DQN paper (Mnih et al. 2015). Soft updates
(τ-weighted averaging) are characteristic of DDPG and similar actor-critic methods — not
standard DQN. For frequency: the original DQN updated every 10,000 steps; with ~1,500 steps
per episode, every 10 episodes yields approximately 15,000 steps, which is directly analogous.
**Risk:** If the original used soft updates (τ ≈ 0.005), training would be smoother but slower
to diverge from a bad target. If the update frequency was higher (every episode) or lower
(every 50 episodes), training stability characteristics would differ. Every-episode updates
are a common alternative for short-episode settings and would be the first thing to try if
training instability is observed.
**ADR:** ADR-002

---

### A-T4: Adam Optimizer with Confirmed Learning Rate 0.0003

**Value chosen:** Adam optimizer (Kingma & Ba, 2014) with learning rate 0.0003 and default
PyTorch parameters (β₁=0.9, β₂=0.999, ε=1e-8).
**Resolves gap:** T4
**Justification:** The confirmed learning rate of 0.0003 is a strong contextual indicator of
Adam. This value is the most common default Adam learning rate in PyTorch deep RL implementations
(3e-4 is often cited as "the Adam learning rate for RL"). The original DQN paper used RMSProp
at 0.00025 — a different learning rate — making Adam the more likely choice given the confirmed
value. Modern DQN reconstructions overwhelmingly use Adam.
**Risk:** If the original used RMSProp or SGD, gradient magnitudes and convergence behavior
would differ despite the same learning rate. RMSProp with lr=0.0003 would also be a defensible
choice but is less standard in current PyTorch DQN work.
**ADR:** ADR-002

---

### A-T5: MSE Loss Function

**Value chosen:** Mean Squared Error (MSE) loss between predicted Q-values and Bellman
TD targets. PyTorch: `nn.MSELoss()`.
**Resolves gap:** T5
**Justification:** MSE is the loss function used in the original DQN paper (Mnih et al. 2015)
and represents the correct reconstruction baseline when no modification is documented. Huber
loss (SmoothL1) is a common improvement that clips gradients for large TD errors, but it
constitutes an algorithmic modification to the base DQN. Adopting Huber loss without evidence
would be reconstructing an improved variant rather than the original. MSE is the conservative,
source-faithful default.
**Risk:** If large TD errors occur (e.g., from high-volatility TQQQ episodes), MSE will produce
larger gradient magnitudes than Huber loss. Training instability, if observed, should be
documented before switching to Huber loss — switching without documentation would violate the
reconstruction philosophy.
**ADR:** ADR-002

---

## Derived Consequence: Observation Space Dimension

Given the above assumptions, the observation space dimension is fully determined:

- 5 raw OHLCV values + 5 one-day pct_changes + 1 VWAP value = **11 features per asset**
- Single-timestep observation (no window multiplier)
- Per-asset observation (no cross-asset concatenation)
- **Final obs_dim = 11**
- **Network input layer: Linear(11, 256)**

This value flows directly from A-D1, A-D2, A-E2, and A-D4. It is a derived fact, not an
independent assumption. Any change to A-D1, A-D2, A-E2, or A-D4 requires recomputing obs_dim.

---

## Ledger Integrity Rules

1. Do not implement a module that depends on an unresolved gap without first declaring an
   assumption in Category 3.
2. Do not change a confirmed value (Category 1) without creating an ADR explaining the deviation.
3. Do not mark a gap as resolved without specifying the source (paper, report, or assumption ID).
4. The ledger is append-only for Category 1. Corrections require an inline note with the source.
5. Category 3 entries are permanent once written. If an assumption is superseded by confirmed
   evidence, the assumption entry is annotated — not deleted.

---

## Resolution Status Summary

| Category | Count | Notes |
|----------|-------|-------|
| Confirmed (Cat 1) | 24 | All confirmed values are ready for `config.py` |
| Open gaps (Cat 2) | 0 | All 13 gaps resolved by declaration in Category 3 |
| Declared assumptions (Cat 3) | 15 | Cover 16 gap IDs (D4/E5 share one assumption; E6/E7 share one; T2/T3 share one) |
| Derived consequences | 1 | `obs_dim = 11` derived from A-D1, A-D2, A-E2, A-D4 |

All gaps resolved. Implementation may proceed according to `docs/implementation-plan.md`.

---

*Last updated: 2026-03-14*
*Reconstruction phase: Gap resolution complete — ready for implementation*
