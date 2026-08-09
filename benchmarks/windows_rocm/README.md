# Windows CPU / OpenCL / ROCm benchmark

This harness builds LightGBM 4.7.0 and runs the same deterministic binary-classification workload through four modes:

1. legacy DLL on CPU;
2. legacy DLL on the OpenCL `gpu` backend;
3. LightGBM 4.7.0 on CPU;
4. LightGBM 4.7.0 built with native Windows HIP/ROCm and run with `device_type=cuda`.

The default workload uses `objective=binary`, `metric=auc`, 100 boosting iterations (`n_estimators=100`), `max_depth=8`, and `num_leaves=4`. The generated dataset and all seeds are deterministic.

Run from PowerShell:

```powershell
.\benchmarks\windows_rocm\build_and_benchmark.ps1
```

The script copies the binaries used for the run into `benchmarks/windows_rocm/bin/`, generated data into `data/`, and outputs into `artifacts/`. It creates the training bins once with the legacy DLL and saves a common LightGBM binary dataset, so all four training modes consume exactly the same bin boundaries and rows. For each mode it saves a model text file and prediction file. `summary.json` records timings, AUC, prediction ranges, prediction differences against the legacy CPU reference, tree counts, exact tree-text equality, and a structural tree signature.

The correctness gate requires finite, non-constant predictions, exactly the requested number of trees, matching tree structure, and predictions matching the legacy CPU reference within configurable tolerances. Exact serialized tree equality is reported but is not a hard failure because CPU, OpenCL, and HIP reductions can differ by tiny floating-point values in gains, thresholds, weights, or leaf outputs while retaining the same structure and predictions.

Native Windows ROCm currently uses the CLI for the fourth mode because the experimental HIP `_lightgbm.dll` still has a C-API initialization crash on `device_type=cuda`; the compiled ROCm CLI executes the same LightGBM CUDA/HIP training path.
