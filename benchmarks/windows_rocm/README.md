# Windows CPU / OpenCL / ROCm benchmark

This harness validates and times the same LightGBM training workload through legacy CPU, legacy OpenCL GPU, LightGBM 4.7 CPU, and native Windows HIP/ROCm. The default dataset shape is 40,000 training rows, 50,000 validation rows, and 3,000 features.

## Canonical reference

Correctness of the ROCm backend is gated against a **pristine upstream LightGBM v4.7.0 CPU DLL** built from the official `microsoft/LightGBM` tag `v4.7.0` at commit `8f7036f03627054d5a54a6f965b13f4b9ff2cb63`. The CPU reference is built from a separate external source checkout under the benchmark work root, never from the RDNA2 fork checkout, so fork-side Serial/RDNA2 changes cannot contaminate the oracle. The legacy CPU/OpenCL modes remain compatibility and performance references only.

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

The Google Drive checkout is source-only. All generated state lives under `C:\Temp\LightGBM-RDNA2` by default: CMake/Ninja trees under `build\cpu` and `build\rocm`, benchmark binaries under `benches\bin`, generated datasets under `benches\data`, and models/predictions/logs/matrix summaries under `benches\artifacts`. `build_and_benchmark.ps1 -WorkRoot <path>` overrides the complete external work root. Standalone Python benchmark runs use `LIGHTGBM_RDNA2_TEMP` for the benches root and `LIGHTGBM_RDNA2_BIN` for binaries. The first smoke run can be slower while its 5,000-row validation files are generated; subsequent runs reuse that cache.

Native Windows ROCm currently uses the CLI because the experimental HIP `_lightgbm.dll` still crashes during C-API booster creation with `device_type=cuda`; the CLI executes the same CUDA/HIP training path.

## Dataset-pipeline benchmark

`run_dataset_benchmarks.py` benchmarks split-local raw feature handling without reusing bin boundaries from another split. The pristine upstream v4.7.0 CPU DLL and the fork candidate DLL both receive the same C-contiguous `float32` training matrix through `LGBM_DatasetCreateFromMat`, set labels, and serialize independent datasets. Candidate dataset correctness is isolated by training both serialized datasets with the pristine CPU oracle and applying the normal prediction/tree-structure gate.

Stages are `dataset_create`, `dataset_to_rdna2`, `end_to_train`, and `end_to_end`. `--split-offset` shifts the generated train/validation window while constructing bins from the training slice only, so time-series-style split changes do not share future-derived binning state. Dataset reports include input conversion, dataset construction, label attachment, serialization, and process peak working-set measurements; RDNA2 stages additionally run a one-iteration initialization probe and parse the packed-dataset pack/allocation/H2D components. The probe wall time is explicitly labeled as including that one iteration; the reported logical `end_to_train` total excludes training and binary serialization. `end_to_end` adds the measured boosting time to that logical pre-train path and applies the normal pristine-v4.7 CPU vs RDNA2 prediction/tree correctness gate.

```powershell
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage dataset_create --profile h64
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage dataset_to_rdna2 --profile h128 --split-offset 5000
```

