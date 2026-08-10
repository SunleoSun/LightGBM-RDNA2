---
description: Smoke, Optuna-envelope, compatibility, production, and long-horizon correctness gates for RDNA2 changes.
---

# Smoke and validation

After an RDNA2 histogram/kernel modification, run the smallest correctness gate first: `python .\benchmarks\windows_rocm\run_matrix.py --suite smoke`. The smoke suite uses `h64`, `h128`, `h64_scale16`, and `h128_regression`, compares LightGBM 4.7 CPU against `device_type=rdna2`, trains 6 trees on the full 40k x 3000 training shape, and uses 5000 validation rows. The initial RDNA2 Serial-semantics baseline passes all four with prediction max diff `0` and exact tree text/structure.

Use `--suite optuna` after smoke for the production hyperparameter envelope. It runs 20 trees on ten representative profiles spanning depth `2/3/4/5/6/8`, leaves `4/8/16/32/40/120`, max-bin `63/64/127`, all regularization-profile families, learning rates from `0.085` down to `0.02`, and class-weight extremes together with the smoke scale-16 probe. The current RDNA2 Serial baseline passed every Optuna profile with prediction max diff `0` and exact tree structure.

Use `--suite optuna_compat` when a CPU/OpenCL/RDNA2 cross-check is useful. It samples depth 2, depth 4, and depth 8 with `--modes all`. RDNA2 is still gated against LightGBM 4.7 CPU. Legacy OpenCL is reported but not used as the universal correctness oracle because it matched CPU structure at depths 2 and 4 yet diverged at depth 8 / 120 leaves (prediction max diff about `0.00627` and different tree structure). This is evidence that an RDNA2 HIP kernel may borrow the OpenCL architecture without being required to reproduce every OpenCL floating-point split decision.

Use `--suite optuna_long` only at milestone/pre-merge boundaries. It encodes all eight production learning-rate/tree-count pairs: `80@0.085`, `120@0.075`, `200@0.06`, `300@0.05`, `450@0.04`, `650@0.03`, `850@0.025`, and `1050@0.02`. Representative 80-, 120-, 450-, and 1050-tree runs have already passed CPU/RDNA2 exactness. The complete long suite remains intentionally expensive and should be run before accepting a major H64/H128/SuperTile stage rather than after every small kernel edit.

`run_benchmarks.py --modes v470` runs the CPU reference plus `v470_rdna2`. `--modes all` additionally keeps legacy CPU/OpenCL and `v470_cuda_diagnostic`; the CUDA learner is diagnostic only and is not correctness-gated for the RDNA2 path.

If smoke fails, do not use heavier suite results as performance evidence. If smoke passes, `--suite production` provides H64/H128 performance at 100 trees and `--suite stress` provides the older integration matrix at 20 trees (no-bagging, feature_fraction=0.5, scale_pos_weight=16, strong regularization, and regression probes). The Optuna suite complements rather than replaces stress by covering the actual depth/leaf/max-bin/boost envelope.

Binary correctness gates include finite/nonconstant probabilities, `[0,1]` range, requested tree count, tight prediction agreement, Pearson correlation, AUC agreement, hard labels at 0.5, confusion matrix, class-balance/tail sanity relative to what the CPU reference itself exhibits, and tree structural signature. Both the minimum predicted-class fraction and confident-tail fraction are capped by the CPU reference value; a two-tree reference that predicts only one hard class must not make an otherwise bit-identical RDNA2 run fail. Regression gates include finite/nonconstant predictions, requested tree count, prediction agreement/correlation, tree structure, and RMSE agreement. Typical binary prediction tolerances are `atol=1e-6`, `rtol=1e-6`, AUC tolerance `5e-8`, and Pearson at least `0.99999999`.
