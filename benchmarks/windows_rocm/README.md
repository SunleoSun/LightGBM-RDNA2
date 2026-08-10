# Windows CPU / OpenCL / ROCm benchmark

This harness builds LightGBM 4.7.0 and runs the same deterministic binary-classification workload through four modes: legacy DLL CPU, legacy DLL OpenCL `gpu`, LightGBM 4.7.0 CPU, and native Windows HIP/ROCm using `device_type=cuda`.

The default dataset is intentionally very wide: 40,000 training rows, 50,000 validation rows, and 3,000 features. The harness now has two production-oriented histogram profiles instead of the old `max_bin=255` microbenchmark.

- `h64` models Stage 1: `max_bin=63`, `max_depth=6`, `num_leaves=40`, `min_data_in_leaf=50`, `feature_fraction=0.9`, `bagging_fraction=0.7`, `bagging_freq=1`, and `learning_rate=0.075`. `num_leaves=40` is a representative resolved value matching the `balanced_small` depth-6 profile; it can be changed later if the Stage 1 resolver commonly chooses another value.
- `h128` models a representative Stage 2 depth-6 case: `max_bin=127`, `max_depth=6`, `num_leaves=40`, `min_data_in_leaf=50`, `feature_fraction=1.0`, `bagging_fraction=0.9`, `bagging_freq=1`, `lambda_l1=lambda_l2=0.1`, and `learning_rate=0.05`. Stage 2 is dynamic, so this profile is a stable H128 architecture benchmark rather than a claim that every Optuna trial uses these exact regularization values.

Run both profiles from PowerShell:

```powershell
.\benchmarks\windows_rocm\build_and_benchmark.ps1
```

Run only one histogram width:

```powershell
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Profile h64
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Profile h128
```

The generated LightGBM binary dataset is keyed by `max_bin` (`..._maxbin63.bin` and `..._maxbin127.bin`). This is required because bin boundaries are encoded when the binary dataset is created; reusing a `max_bin=255` binary would not exercise the H64/H128 kernels correctly. All four modes consume the same binary dataset within each profile.

Artifacts are separated into `artifacts/h64/` and `artifacts/h128/`. Each `summary.json` records the complete profile, timings, AUC, prediction ranges, prediction differences against the legacy CPU reference, tree counts, exact tree-text equality, and a structural tree signature.

For binary classification, saved predictions are probabilities. The correctness gate requires finite, non-constant probabilities in `[0, 1]`, exactly the requested number of trees, matching tree structure, probability agreement with the legacy CPU reference, near-perfect Pearson correlation, AUC agreement within `5e-8`, identical hard labels at threshold `0.5`, and an identical confusion matrix. Exact serialized tree equality is reported but is not a hard failure because CPU, OpenCL, and HIP reductions can differ by tiny floating-point values while retaining the same structure and predictions.

Native Windows ROCm currently uses the CLI for the fourth mode because the experimental HIP `_lightgbm.dll` still has a C-API initialization crash on `device_type=cuda`; the CLI executes the same LightGBM CUDA/HIP training path.
