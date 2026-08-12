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

`run_dataset_benchmarks.py` benchmarks split-local raw feature handling without reusing bin boundaries from another split. It now starts from a full phase-like C-contiguous `float32` feature matrix, extracts the current training interval, then passes only that interval to `LGBM_DatasetCreateFromMat`. `--gap-rows` keeps validation separated from the training boundary; `--split-offset` can shift the whole train window, while an expanding-prefix `TimeSeriesSplit` shape is modeled with `--split-offset 0` and increasing `--train-rows`. No candidate path sees validation/future rows while constructing bins.

Two extraction modes are available. `pipeline_fancy` reproduces the current Python pipeline shape (`features[train_idx]`) and therefore materializes a dense copy plus the integer index array. `contiguous_slice` uses the equivalent contiguous time range as a NumPy view. `--split-extraction both` runs both representations, records layout/strides/ownership and working-set changes, and then trains both pristine-v4.7 serialized datasets with the pristine CPU oracle to prove that the extraction representation did not change Dataset/tree/prediction semantics. Candidate-vs-pristine Dataset correctness is gated separately for every extraction mode.

Stages remain `dataset_create`, `dataset_to_rdna2`, `end_to_train`, and `end_to_end`. Reports attribute phase input conversion, split materialization, model-input conversion, Dataset construction, label attachment, serialization, and process peak working set. RDNA2 stages add a one-iteration initialization probe and parse Feature4 packing/allocation/H2D components. Logical `end_to_train` includes split extraction through device-ready RDNA2 state but excludes training and benchmark-only binary serialization. `end_to_end` adds boosting time and applies the normal pristine-v4.7 CPU vs RDNA2 correctness gate. A Phase-1 two-side projection is reported because the production pipeline currently constructs independent long and short Datasets from the same fold feature matrix; it is a throughput estimate, not a semantic reuse implementation.

On the 40k x 3000 H64 production shape, `pipeline_fancy` split extraction copies about 457.8 MiB before Dataset creation. With pristine v4.7 this path is about 3.0 s for Dataset construction and peaks about 2.0 GiB above the resident phase matrix. The fork's dense row-major float32 CPU builder reduces construction to roughly 0.9-1.05 s in warm same-process runs. The optional `--candidate-dataset-device rdna2` path keeps CPU `BinMapper` boundaries canonical but streams row-major float32 through the RX 6800 XT in bounded chunks for value-to-bin population, then writes those canonical bins directly into the LightGBM dense storage. Warm same-process 40k x 3000 construction is roughly 0.56-0.58 s for both H64/H128; a fresh process still pays roughly 0.5 s one-time ROCm context/allocation startup. Persistent pinned/device staging removes repeated allocation cost after the first Dataset. H64/H128 production binaries are SHA-256 identical to pristine v4.7, and dedicated dense-zero/NaN, sparse-EFB fallback, categorical fallback, and reference-Dataset probes are also byte-identical. The 100-tree H64/H128 end-to-end correctness gates pass. A 500k x 4000 population microbenchmark uses 8192-row streaming chunks rather than making the ~7.45 GiB raw float32 matrix resident in VRAM; chunking itself was effectively neutral on the 40k x 3000 shape.

```powershell
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage dataset_create --profile h64 --split-extraction both
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage dataset_to_rdna2 --profile h128 --gap-rows 64 --split-extraction contiguous_slice
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage end_to_end --profile h64 --iterations 100 --split-extraction contiguous_slice --candidate-dataset-device rdna2
```

