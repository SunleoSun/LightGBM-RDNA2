# Windows CPU / OpenCL / RDNA2 benchmark

This directory contains the correctness and performance harness used for the native Windows RDNA2 / ROCm LightGBM fork. The target machine is an AMD Ryzen 9 5950X with a Radeon RX 6800 XT (`gfx1030`) running ROCm 6.2. The production benchmark shape is 40,000 training rows, 50,000 validation rows, 3,000 `float32` features, and 100 boosting iterations.

## Correctness authority

The correctness oracle is a **pristine upstream LightGBM v4.7.0 CPU build** from commit `8f7036f03627054d5a54a6f965b13f4b9ff2cb63`. It is built from a separate source checkout, so Serial/RDNA2 changes in this fork cannot contaminate the reference.

The legacy OpenCL backend is retained as a compatibility and performance reference. It is useful for answering “is the new backend actually faster than the old GPU path?”, but it is not the semantic oracle. The experimental upstream CUDA-on-HIP diagnostic is also benchmarked in the full production suite, but it is not accepted as a correctness baseline because it diverges on this gfx1030 target.

For binary objectives, the gate requires the requested tree count, matching tree structure, finite and non-constant probabilities, `allclose` predictions, near-perfect correlation, AUC agreement, identical hard labels, and an identical confusion matrix. For regression objectives it requires the requested tree count, matching structure, finite non-constant predictions, `allclose`, near-perfect correlation, and metric agreement. Quantile regression additionally compares the pinball loss for the requested `alpha`.

## Production profiles

- `h64`: Stage-1-like `max_bin=63`, depth 6, 40 leaves, `min_data_in_leaf=50`, `feature_fraction=0.9`, `bagging_fraction=0.7`, `bagging_freq=1`, learning rate 0.075.
- `h128`: representative Stage-2 `max_bin=127`, depth 6, 40 leaves, `min_data_in_leaf=50`, `feature_fraction=1.0`, `bagging_fraction=0.9`, light L1/L2 regularization, learning rate 0.05.

Binary Dataset files are keyed by target family (binary versus continuous-target regression/quantile) and `max_bin`, because binning is already encoded in the serialized Dataset. H64/H128 therefore never silently reuse an incompatible binary, while L2 and quantile may intentionally share the same continuous-target Dataset when rows, labels, and binning parameters are identical.

## Implemented RDNA2 optimizations

### Training path

The accepted training architecture is correctness-first. RDNA2 changes physical execution, while canonical LightGBM structures remain semantic owners.

- Exact gfx1030 HIP histogram kernels for H64 and H128.
- Persistent canonical Feature4 packing: four byte-sized feature groups are packed into one aligned `uint32` representation once per Dataset/learner lifecycle instead of being repacked per tree.
- Direct histogram readback into the Serial-owned canonical histogram buffer using host registration when available, removing an intermediate staging memcpy.
- H128 device-resident histogram pool with canonical parent-minus-smaller-child subtraction.
- Batched H128 pair best-split work and explicit event handoff between histogram completion and selector work.
- H64 keeps canonical CPU `FeatureHistogram::FindBestThreshold` split authority. Finite GPU top-K nomination was tested but did not deliver a strict-correctness speedup.
- H128 uses GPU approximate top-2 nomination only as a candidate reducer; compact nominated slices are finalized by CPU `Dataset::FixHistogram` and canonical `FeatureHistogram::FindBestThreshold`. CPU remains the final `SplitInfo` authority.
- Pinned Feature4 activity-mask staging and a read-only H128 nomination scheduling sidecar for the supported full-feature path.
- Unsupported semantics, including shapes that cannot preserve the canonical contract, fall back to the Serial path rather than silently changing results.

Several experiments were deliberately rejected after measurement: H128 top-1 nomination diverged over longer horizons, larger nomination counts lost the speedup, H64 finite top-K did not improve strict-correctness wall time, two-bank H128 was slower, and direct in-place/fused histogram ownership variants either slowed the workload or broke canonical correctness.

### Dataset construction path

Dataset creation was originally the dominant pre-train bottleneck and transient-memory consumer. The dense C-row-major `float32` path now contains a sequence of accepted optimizations, all gated against pristine v4.7 semantics.

1. Specialized single-matrix row-major `float32` construction removes the old per-row `std::vector<double>` materialization and avoids keeping all feature samples as wide nested vectors. Generic layouts continue to use the generic path.
2. Numerical `BinMapper` construction is feature-parallel on the 5950X while canonical LightGBM boundary logic remains authoritative.
3. Canonical EFB behavior is preserved. Dense features that are mathematically proven pairwise unbundleable can take the singleton-group shortcut; sparse/bundleable inputs use the canonical fallback.
4. Numerical float32 samples use a stable IEEE-key radix presort instead of widening the entire sample to `double` and then invoking `stable_sort<double>`. Categorical/generic callers retain their original path.
5. `FindBinFromSortedFloat32` constructs canonical distinct values/counts directly from sorted finite float32 samples plus an explicit NaN count. It then delegates forced-bin handling, bin-boundary decisions, prefiltering, default/most-frequent bins, and sparse-rate semantics to the same canonical BinMapper finalizer.
6. Gather and radix scratch buffers are reused per OpenMP thread instead of being allocated per feature.
7. Sixteen adjacent float32 numerical features are gathered per row-major pass. Sixteen floats are one 64-byte cache line, replacing the former 12 KB stride between adjacent rows of a single feature on a 3,000-feature matrix.
8. Regular value-to-bin population is offloaded to the RX 6800 XT only after CPU-authoritative BinMapper boundaries exist. Unsupported categorical, EFB/multi-feature, sparse, raw-storage, or wider-bin cases fall back to CPU population.
9. GPU population is bounded and streaming. Raw data is never required to reside in VRAM as one giant matrix; this is important for shapes such as 500,000 x 4,000, where raw float32 input alone is about 7.45 GiB.
10. Pinned host/device staging and boundary metadata are process-persistent, avoiding repeated allocation cost for later Datasets.
11. First-process ROCm/staging initialization is overlapped with canonical CPU BinMapper construction when the persistent context is not yet large enough.
12. The final population pipeline uses **two 4,096-row slots** instead of one 8,192-row slot. Host staging, H2D/kernel/D2H, and canonical host DenseBin loading overlap while keeping approximately the same bounded staging footprint. On the 40k x 3000 shape, the population section dropped from roughly 72-80 ms to about 55-61 ms, with only about 7-10 ms of observed GPU wait.

The resulting production H64/H128 serialized Datasets are SHA-256 identical to pristine v4.7. Additional byte-identical probes cover zeros/NaNs, sparse data exercising EFB fallback, categorical features including negative-to-missing handling, and Dataset construction from a reference Dataset.

## Production results — 40k x 3000, 100 trees

The following measurements were regenerated on 2026-08-12 after the final Dataset-population pipeline was accepted. `end_to_end` here is the training-process path through Dataset/binary load, learner initialization, and boosting; prediction is measured separately. Memory is **peak host process working set**, not GPU VRAM.

| Profile | Backend | End-to-end (s) | Speedup vs v4.7 CPU | Peak host WS (MiB) | Iteration (ms) |
|---|---|---:|---:|---:|---:|
| H64 | v4.7 CPU | 5.477 | 1.00x | 293.0 | 53.05 |
| H64 | legacy OpenCL | 2.915 | 1.88x | 357.0 | 19.77 |
| H64 | RDNA2 / ROCm | **2.611** | **2.10x** | 499.0 | 26.11 |
| H128 | v4.7 CPU | 8.551 | 1.00x | 412.2 | 83.77 |
| H128 | legacy OpenCL | 5.202 | 1.64x | 479.1 | 42.73 |
| H128 | RDNA2 / ROCm | **2.538** | **3.37x** | 631.9 | **25.38** |

RDNA2 is about 1.12x faster than legacy OpenCL end-to-end on H64 and about 2.05x faster on H128. H64 legacy OpenCL still has a lower pure boosting iteration time in this particular profile, but RDNA2 wins the complete measured training process and preserves the pristine-v4.7 tree/prediction contract exactly.

![Production end-to-end comparison](images/production_end_to_end.png)

![Production peak host-memory comparison](images/production_peak_host_memory.png)

The training host-memory chart should not be confused with Dataset-construction memory. RDNA2 deliberately keeps extra host/device-facing staging and persistent learner state, so its peak host working set during the complete 100-tree process is higher than CPU/OpenCL in this benchmark. The raw-to-canonical **Dataset construction** itself is dramatically smaller than pristine v4.7, as shown below. GPU VRAM is not included in these host working-set measurements.

## Dataset creation results

Fresh-process, zero-copy contiguous-slice benchmark, 40k x 3000, 32 CPU threads:

| Profile | Builder | DatasetCreate (s) | Peak delta above resident phase matrix (MiB) |
|---|---|---:|---:|
| H64 | pristine v4.7 CPU | 3.491 | 1543.5 |
| H64 | current RDNA2 path | **0.774** | **302.1** |
| H128 | pristine v4.7 CPU | 2.955 | 1543.5 |
| H128 | current RDNA2 path | **0.775** | **304.2** |

That is roughly **4.5x faster / 5.1x lower transient peak** for H64 and **3.8x faster / 5.1x lower transient peak** for H128 in a fresh worker. In a long-lived process with the ROCm population context already warm, 40k x 3000 Dataset construction is commonly about **0.29-0.35 s**.

The current production Python shape `features[train_idx]` can still materialize an additional approximately **457.8 MiB** copy for a 40k x 3000 fold. The benchmark-proven equivalent contiguous slice is a zero-copy view and does not change canonical Dataset/tree/prediction semantics. Removing that copy belongs to the pipeline-side work after this LightGBM optimization phase.

## Quantile regression correctness

The strict matrix now includes `objective=quantile` at `alpha=0.1`, `0.5`, and `0.9` for both H64 and H128. The suite uses pristine v4.7 CPU as the oracle and checks tree count/structure, prediction equality, correlation, RMSE agreement, and pinball-loss agreement.

Production-sized 40k x 3000, 20-tree correctness run:

The same six profiles were also run through `run_dataset_benchmarks.py --stage end_to_end --candidate-dataset-device rdna2`, so the optimized raw-float32 Dataset builder, serialized Dataset, and RDNA2 training path are all covered together; all six end-to-end Dataset-pipeline gates passed.

| Profile | alpha | Pinball loss | Max prediction diff vs CPU | Structure | Result |
|---|---:|---:|---:|---|---|
| H64 | 0.1 | 0.3683008724 | 0 | match | PASS |
| H64 | 0.5 | 0.7008871445 | 0 | match | PASS |
| H64 | 0.9 | 0.3659010242 | 0 | match | PASS |
| H128 | 0.1 | 0.4330241263 | 0 | match | PASS |
| H128 | 0.5 | 0.8341953569 | 0 | match | PASS |
| H128 | 0.9 | 0.4351474008 | 8.88e-16 | match | PASS |

Run it with:

```powershell
python .\benchmarks\windows_rocm\run_matrix.py --suite quantile
```

## Correctness and performance suites

```powershell
# Fast representative correctness gate.
python .\benchmarks\windows_rocm\run_matrix.py --suite smoke

# H64 + H128, 100 trees, CPU/OpenCL/RDNA2 performance comparison.
python .\benchmarks\windows_rocm\run_matrix.py --suite production

# Quantile alpha=0.1/0.5/0.9 for H64 and H128.
python .\benchmarks\windows_rocm\run_matrix.py --suite quantile

# Broader non-Optuna correctness sweep.
python .\benchmarks\windows_rocm\run_matrix.py --suite stress

# Feature-fraction and production Optuna envelopes.
python .\benchmarks\windows_rocm\run_matrix.py --suite feature_fraction
python .\benchmarks\windows_rocm\run_matrix.py --suite optuna
python .\benchmarks\windows_rocm\run_matrix.py --suite optuna_long
python .\benchmarks\windows_rocm\run_matrix.py --suite optuna_compat
```

The build wrapper can build and run the same suites:

```powershell
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite smoke
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite production
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite stress
```

## Dataset-pipeline benchmark

`run_dataset_benchmarks.py` starts from a full phase-like C-contiguous `float32` matrix and models the production time-series boundary before calling `LGBM_DatasetCreateFromMat`. It never constructs bin boundaries from validation/future rows.

Two split representations are supported:

- `pipeline_fancy` reproduces `features[train_idx]`, including the dense copy and integer index array.
- `contiguous_slice` uses the same contiguous time interval as a NumPy view.

`--split-extraction both` trains the pristine-v4.7 outputs from both representations and gates them against each other, so the zero-copy representation is a measured correctness result rather than an assumption. Candidate-vs-pristine Dataset correctness is also gated separately.

Stages are `dataset_create`, `dataset_to_rdna2`, `end_to_train`, and `end_to_end`. Reports attribute input conversion, split materialization, Dataset construction, label attachment, serialization, process peak memory, RDNA2 Feature4 packing/allocation/H2D, and boosting time.

```powershell
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage dataset_create --profile h64 --split-extraction both --candidate-dataset-device rdna2
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage dataset_to_rdna2 --profile h128 --gap-rows 64 --split-extraction contiguous_slice --candidate-dataset-device rdna2
python .\benchmarks\windows_rocm\run_dataset_benchmarks.py --stage end_to_end --profile h64 --iterations 100 --split-extraction contiguous_slice --candidate-dataset-device rdna2
```

## Reproducing the charts

The checked-in benchmark snapshot is `results/production_2026-08-12.json`. Charts are generated from that file rather than hand-entered values:

```powershell
python .\benchmarks\windows_rocm\plot_benchmark_results.py
```

This writes:

- `images/production_end_to_end.png`
- `images/production_peak_host_memory.png`

## Generated state

The Google Drive checkout is source-only. Build trees, binaries, generated datasets, models, predictions, and raw benchmark reports live under `C:\Temp\LightGBM-RDNA2` by default. The repository only retains benchmark source, the compact result snapshot used by the README, and generated documentation charts.

`LIGHTGBM_RDNA2_TEMP` changes the external benchmark root and `LIGHTGBM_RDNA2_BIN` changes the benchmark binary directory. `ROCM_PATH` selects the ROCm installation used by the native Windows runner.
