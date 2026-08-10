# Windows CPU / OpenCL / ROCm benchmark

This harness validates and times the same LightGBM training workload through legacy CPU, legacy OpenCL GPU, LightGBM 4.7 CPU, and native Windows HIP/ROCm. The default dataset shape is 40,000 training rows, 50,000 validation rows, and 3,000 features.

## Canonical reference

Correctness of the ROCm backend is gated against **LightGBM 4.7 CPU**, because it is the same source version as the HIP backend. The legacy CPU/OpenCL modes remain in every profile as compatibility and performance references, but differences caused by the older LightGBM version do not fail the ROCm correctness gate.

For binary objectives the gate requires finite, non-constant probabilities in `[0, 1]`, requested tree count, matching tree structure, `allclose` predictions, near-perfect Pearson correlation, AUC agreement, identical hard labels at 0.5, and identical confusion matrix. For L2 regression it requires finite non-constant predictions, requested tree count, matching tree structure, `allclose` predictions, near-perfect correlation, and RMSE agreement.

## Production profiles

- `h64`: Stage-1-like `max_bin=63`, depth 6, 40 leaves, `min_data_in_leaf=50`, `feature_fraction=0.9`, `bagging_fraction=0.7`, `bagging_freq=1`, learning rate 0.075.
- `h128`: representative Stage-2 `max_bin=127`, depth 6, 40 leaves, `min_data_in_leaf=50`, `feature_fraction=1.0`, `bagging_fraction=0.9`, light L1/L2 regularization, learning rate 0.05.

Binary LightGBM datasets are keyed by objective and `max_bin`, because binning is encoded when the dataset is serialized. H64, H128, binary, and regression runs therefore never silently reuse an incompatible `.bin` file.

## Strict correctness matrix

The strict matrix deliberately changes dimensions that exercise different histogram/index/gradient behavior:

- baseline H64 and H128;
- no-bagging variants (`subsample=1.0`);
- feature sampling (`feature_fraction=0.5`);
- strong regularization (`min_data_in_leaf=100`, L1/L2=1, `min_gain_to_split=0.005`, `path_smooth=1`);
- strong binary class weighting (`scale_pos_weight=16`);
- `regression_l2` for both H64 and H128, which changes gradient/Hessian semantics and removes binary-probability-specific checks.

Suites:

```powershell
# Fast development gate: four representative profiles, 6 trees each.
# Runs only LightGBM 4.7 CPU + ROCm and uses 5,000 validation rows by default.
# Training remains production-sized at 40,000 x 3,000.
python .\benchmarks\windows_rocm\run_matrix.py --suite smoke

# Real timing: H64 + H128, 100 trees each.
python .\benchmarks\windows_rocm\run_matrix.py --suite production

# Full correctness sweep: every variant, 20 trees each.
python .\benchmarks\windows_rocm\run_matrix.py --suite stress
```

The build wrapper can build and run the same suites:

```powershell
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite smoke
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite production
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite stress

# Backward-compatible direct H64/H128 run with explicit tree count.
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite single -Profile h64 -Iterations 100
```

`-MatrixIterations N` overrides the suite default when using the build wrapper. Smoke defaults to 6 trees, stress to 20, and production to 100.

Runtime datasets, models, predictions, logs/results, and matrix summaries are kept under `C:\Temp\lightgbm-rdna2-temp` by default instead of cluttering `C:\Temp` or the repository. Set `LIGHTGBM_RDNA2_TEMP` to override that root. Profile artifacts live under `<temp-root>/artifacts/<profile>/`; matrix summaries live under `<temp-root>/artifacts/matrix_*.json`. The first smoke run can be slower while its 5,000-row validation files are generated; subsequent runs reuse that cache.

Native Windows ROCm currently uses the CLI because the experimental HIP `_lightgbm.dll` still crashes during C-API booster creation with `device_type=cuda`; the CLI executes the same CUDA/HIP training path.
