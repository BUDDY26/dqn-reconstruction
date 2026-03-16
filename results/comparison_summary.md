# Experiment Results — Three-Way Comparison

**Test period:** 2023-01-01 → 2024-01-01
**Source paper:** Innovative Portfolio Optimization Using Deep Q-Network Reinforcement Learning, NLPIR 2024
DOI: https://doi.org/10.1145/3711542.3711567

---

## Notes on Units

Paper-reported values appear to use a different unit convention than this implementation.
ROI and MDD values in the paper are signed percentages (e.g. ROI 4.11 = 4.11%, MDD -2.55 = -2.55%).
This implementation reports ROI and MDD as decimal fractions (e.g. ROI 0.2964 = 29.64%, MDD 0.0349 = 3.49% peak-to-trough decline expressed as a positive fraction).
The columns are reproduced as-reported for each source without unit conversion.

---

## ROI

| Asset | Paper (%) | Reconstructed Baseline | Optimized |
|-------|-----------|------------------------|-----------|
| AAPL  | 4.11      | 0.2964                 | 0.5394    |
| INTC  | -4.24     | 0.1031                 | 0.2956    |
| META  | 4.06      | 0.4865                 | 0.7488    |
| TQQQ  | 11.24     | 0.6419                 | 1.0901    |
| TSLA  | 4.55      | -0.0025                | -0.0166   |

## Sharpe Ratio

| Asset | Paper | Reconstructed Baseline | Optimized |
|-------|-------|------------------------|-----------|
| AAPL  | 0.74  | 2.3223                 | 2.2897    |
| INTC  | -0.56 | 0.5248                 | 1.2298    |
| META  | 0.41  | 2.2969                 | 2.1627    |
| TQQQ  | 0.78  | 1.3968                 | 2.1121    |
| TSLA  | 0.40  | 0.1891                 | 0.1558    |

## Max Drawdown

| Asset | Paper (%) | Reconstructed Baseline | Optimized |
|-------|-----------|------------------------|-----------|
| AAPL  | -2.55     | 0.0349                 | 0.1505    |
| INTC  | -4.43     | 0.1984                 | 0.0989    |
| META  | -7.74     | 0.0749                 | 0.0771    |
| TQQQ  | -6.41     | 0.3046                 | 0.2015    |
| TSLA  | -6.67     | 0.4394                 | 0.2977    |

## Calmar Ratio

| Asset | Paper | Reconstructed Baseline | Optimized |
|-------|-------|------------------------|-----------|
| AAPL  | 0.78  | 8.6208                 | 3.6379    |
| INTC  | -0.47 | 0.5264                 | 3.0315    |
| META  | 0.25  | 6.5881                 | 9.8674    |
| TQQQ  | 0.83  | 2.1398                 | 5.5012    |
| TSLA  | 0.33  | -0.0057                | -0.0565   |

---

## Summary

**INTC** improved materially in the optimized run: ROI rose from 0.1031 to 0.2956, Sharpe from 0.52 to 1.23, and MDD halved from 0.1984 to 0.0989. Calmar improved from 0.53 to 3.03.

**META** improved materially: ROI rose from 0.4865 to 0.7488, and Calmar improved from 6.59 to 9.87. Drawdown was essentially unchanged (0.0749 vs 0.0771).

**TQQQ** improved materially: ROI rose from 0.6419 to 1.0901, Sharpe from 1.40 to 2.11, and MDD decreased from 0.3046 to 0.2015. Calmar improved from 2.14 to 5.50.

**AAPL** improved ROI substantially (0.2964 → 0.5394) but drawdown increased from 0.0349 to 0.1505, which drove Calmar down from 8.62 to 3.64. A higher-return, higher-volatility trade-off.

**TSLA** remained difficult in both runs. ROI was near zero or slightly negative in both cases. The optimized run reduced MDD from 0.4394 to 0.2977 but ROI became marginally more negative (-0.0025 → -0.0166).

---

## Optimizations Applied (Baseline → Optimized)

The five changes applied between the baseline and optimized runs:

1. GPU / CUDA device support
2. Full reproducibility seeding (all four RNG sources: Python, NumPy, PyTorch CPU, PyTorch CUDA)
3. Gradient clipping (`max_norm=1.0`) on online network parameters
4. Huber loss / SmoothL1 replacing MSE for TD error computation
5. Double DQN target selection (online network selects action, target network evaluates it)

---

## Result Files

| Run | Output file |
|-----|-------------|
| Reconstructed baseline | `results/verification_results.json` |
| Optimized run | `results/optimized_results.json` |
| Paper-reported values | `results/paper_reported_metrics.md` |
