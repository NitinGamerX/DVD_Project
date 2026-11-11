# Model comparison (validation set, deliverable 4)

R² of log10|I|, pooled median / 95th-percentile relative error over Id, Ig, Is, Ib.
Best configuration of each type; **bold** = chosen (lowest validation p95).

| Case | Polynomial ridge | Random forest | Gradient boosting | Neural network (MLP) |
|---|---|---|---|---|
| NMOS ON | 0.9976 / 6.47% / 22.9% | 0.9817 / 15.83% / 88.7% | 0.9972 / 3.77% / 33.6% | **0.9997 / 0.45% / 2.7%** |
| NMOS OFF | 0.6588 / 10.86% / 609.7% | 0.8743 / 12.21% / 100.3% | 0.9284 / 5.01% / 99.8% | **0.9855 / 0.46% / 19.9%** |
| PMOS ON | 0.9963 / 3.89% / 20.9% | 0.9848 / 10.90% / 77.3% | 0.9969 / 2.99% / 30.0% | **0.9999 / 0.37% / 2.1%** |
| PMOS OFF | 0.9434 / 4.23% / 117.6% | 0.9425 / 7.84% / 76.2% | 0.9866 / 2.89% / 33.9% | **0.9994 / 0.32% / 4.7%** |

# Chosen models (test set, deliverable 5)

| Case | Model | Median | p95 | Id med/p95 | Ig med/p95 | Is med/p95 | Ib med/p95 |
|---|---|---|---|---|---|---|---|
| NMOS ON | Neural network (MLP) | 0.458% | 2.75% | 0.45% / 2.6% | 0.36% / 4.0% | 0.45% / 2.5% | 0.61% / 2.7% |
| NMOS OFF | Neural network (MLP) | 0.442% | 20.02% | 0.35% / 6.0% | 0.22% / 1.0% | 0.32% / 4.9% | 5.79% / 84.9% |
| PMOS ON | Neural network (MLP) | 0.364% | 2.18% | 0.28% / 1.8% | 0.39% / 2.6% | 0.28% / 1.8% | 0.60% / 2.6% |
| PMOS OFF | Neural network (MLP) | 0.336% | 4.90% | 0.35% / 5.2% | 0.24% / 1.0% | 0.33% / 4.7% | 1.75% / 9.4% |
