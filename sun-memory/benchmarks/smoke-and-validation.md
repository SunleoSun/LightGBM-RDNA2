---
description: Smoke, production, and stress suite roles plus the correctness gates required after ROCm kernel changes.
---

# Smoke and validation

After a histogram/kernel modification, run the smallest correctness gate first: `python .\benchmarks\windows_rocm\run_matrix.py --suite smoke`. The optimized smoke suite uses `h64`, `h128`, `h64_scale16`, and `h128_regression`, compares only LightGBM 4.7 CPU against ROCm, trains 6 trees on the full 40k x 3000 training shape, and uses 5000 validation rows. Once validation caches exist, the historical wall time is about 14 seconds on the target machine.

If smoke fails, do not use heavier production/stress results as performance evidence. Diagnose the correctness failure first.

If smoke passes, use `--suite production` for H64/H128 performance at 100 trees across all four backends. Use `--suite stress` for integration-level coverage across the strict twelve-profile matrix at 20 trees and all four backends; historically this took roughly 3.5 minutes.

Binary correctness gates include finite/nonconstant probabilities, `[0,1]` range, requested tree count, tight prediction agreement, Pearson correlation, AUC agreement, hard labels at 0.5, confusion matrix, confident tails, and tree structural signature. Regression gates include finite/nonconstant predictions, requested tree count, prediction agreement/correlation, tree structure, and RMSE agreement. Typical binary prediction tolerances are `atol=1e-6`, `rtol=1e-6`, AUC tolerance `5e-8`, and Pearson at least `0.99999999`.

The strict matrix exercises baseline H64/H128 plus no-bagging, feature_fraction=0.5, scale_pos_weight=16, strong regularization, and regression_l2 variants. This matrix is specifically meant to catch histogram correctness failures that a single binary benchmark can miss.
