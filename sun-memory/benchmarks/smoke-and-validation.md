---
description: Smoke, production, and stress suite roles plus the correctness gates required after RDNA2 kernel changes.
---

# Smoke and validation

After an RDNA2 histogram/kernel modification, run the smallest correctness gate first: `python .\benchmarks\windows_rocm\run_matrix.py --suite smoke`. The optimized smoke suite uses `h64`, `h128`, `h64_scale16`, and `h128_regression`, compares LightGBM 4.7 CPU against `device_type=rdna2`, trains 6 trees on the full 40k x 3000 training shape, and uses 5000 validation rows.

As of the initial RDNA2 backend boundary, all four smoke profiles pass with prediction max diff `0`, exact tree text/structure, matching binary metrics and matching regression RMSE. This baseline is intentionally CPU/Serial histogram construction inside `RDNA2TreeLearner`; any HIP offload must preserve the gate.

`run_benchmarks.py --modes v470` runs the CPU reference plus `v470_rdna2`. `--modes all` additionally keeps the legacy CPU/OpenCL backends and `v470_cuda_diagnostic`; the CUDA learner is diagnostic only and is not correctness-gated for the RDNA2 project path.

If smoke fails, do not use heavier production/stress results as performance evidence. Diagnose the correctness failure first.

If smoke passes, use `--suite production` for H64/H128 performance at 100 trees. Use `--suite stress` for integration-level coverage across the strict twelve-profile matrix at 20 trees.

Binary correctness gates include finite/nonconstant probabilities, `[0,1]` range, requested tree count, tight prediction agreement, Pearson correlation, AUC agreement, hard labels at 0.5, confusion matrix, confident tails relative to what the CPU reference itself exhibits, and tree structural signature. A short smoke model is not failed merely because the CPU reference has no predictions below 0.1 or above 0.9. Regression gates include finite/nonconstant predictions, requested tree count, prediction agreement/correlation, tree structure, and RMSE agreement. Typical binary prediction tolerances are `atol=1e-6`, `rtol=1e-6`, AUC tolerance `5e-8`, and Pearson at least `0.99999999`.

The strict matrix exercises baseline H64/H128 plus no-bagging, feature_fraction=0.5, scale_pos_weight=16, strong regularization, and regression_l2 variants. This matrix is specifically meant to catch histogram correctness failures that a single binary benchmark can miss.
