# Reconstruction Notes

## Project Background

This repository reconstructs a Deep Q-Network (DQN) reinforcement learning system used for stock trading.

The original implementation was created as part of a team project. The original repository is no longer accessible, so this version was rebuilt using:

- the final project report
- the original research paper
- architectural evidence extracted from those sources

Because the available materials did not contain a complete implementation, this repository reconstructs the system so that it can run end-to-end.

The goal of this project is to provide a transparent and operational reconstruction of the original system.

---

## Reconstruction Methodology

The reconstruction followed a structured engineering process:

1. Extract confirmed architecture and hyperparameters from the report and paper.
2. Document missing details as explicit assumptions in ADR files.
3. Create a fixed implementation plan before writing code.
4. Implement the system incrementally according to the plan.
5. Validate behavior through automated tests.

This process separates **confirmed evidence** from **reconstruction assumptions**.

---

## Operational Completion

The original documentation did not include a complete runnable implementation.

This repository therefore completes the operational components required for the system to run:

- data loading and preprocessing
- trading environment implementation
- DQN agent implementation
- training pipeline
- evaluation metrics
- automated testing

Where implementation details were not specified in the source materials, reasonable engineering assumptions were made and documented.

---

## Fidelity to the Original System

The reconstruction preserves the following confirmed characteristics of the original design:

- DQN network architecture: **11 → 256 → 256 → 3**
- Adam optimizer
- MSE loss function
- epsilon-greedy exploration strategy
- replay buffer training
- target network updates

These characteristics were derived directly from the available project documentation.

---

## Repository Governance

To ensure the reconstruction remained disciplined, the repository enforced a strict rule:

**`docs/implementation-plan.md` was treated as authoritative and read-only during coding passes.**

This prevented architectural drift during implementation.

---

## Current Implementation Status

The repository now contains a complete operational implementation of the reconstructed system.

Implemented modules include:

- `src/config.py`
- `src/data.py`
- `src/env.py`
- `src/utils.py`
- `src/network.py`
- `src/replay_buffer.py`
- `src/agent.py`
- `src/train.py`
- `src/evaluate.py`

The repository currently contains:

- **204 automated tests**
- **0 failures**
- CI verified through GitHub Actions.

---

## Reconstruction Outcomes and Improvements

The original research paper described the model architecture and training approach but did not
provide a complete runnable implementation.

During reconstruction, several operational components had to be engineered to allow the system
to run end-to-end, including:

- data preprocessing and feature generation
- environment dynamics
- training loop implementation
- evaluation metric calculation
- automated testing

While implementing these components, minor engineering adjustments were required in order to
produce a stable and reproducible training pipeline.

After the system was operational, the reconstructed implementation achieved evaluation results
that exceeded the performance values reported in the original project documentation.

These improvements were not the result of architectural changes to the DQN itself, but rather
from completing and stabilizing the operational pipeline around the model.

The repository therefore represents both:

1. A reconstruction of the original system as described in the available source materials.
2. A demonstration that a disciplined, evidence-bounded engineering process can produce a
   stable and reproducible implementation from incomplete documentation.

---

## Future Work

### Improvement Branch

The `main` branch of this repository represents the **completed reconstruction and operational
implementation** of the DQN trading system described in the original research paper and project
report.

This branch preserves the reconstructed baseline and should remain stable.

Future experimentation and algorithmic improvements will be developed in a **separate branch**
so that the reconstructed baseline remains unchanged.

The improvement work will begin in the following branch:

`improvement-experiments`

This separation ensures:

- the reconstructed system remains reproducible
- improvements can be compared against the baseline
- architectural changes do not overwrite the reconstruction

---

## Planned Improvements

Future experimentation may explore improvements such as:

- Huber loss instead of MSE
- gradient clipping for training stability
- Double DQN target computation
- soft target network updates
- improved training metrics and logging

These changes are intended to be evaluated against the reconstructed baseline to measure their
individual contribution to training stability and evaluation performance.

---

## Evaluation Assumptions

The evaluation metrics implemented in `src/evaluate.py` follow the assumptions below:

- risk-free rate assumed to be **0**
- annualization uses **252 trading days**
- Sharpe ratio uses **sample standard deviation (ddof = 1)**
- Calmar ratio returns **0.0** when maximum drawdown is **0** (undefined case handled conservatively)
