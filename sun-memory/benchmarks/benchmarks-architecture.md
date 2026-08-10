---
description: Windows ROCm benchmark harness architecture, backend roles, production profiles, and dataset/correctness invariants.
---

# Benchmark architecture

The benchmark harness lives under `benchmarks/windows_rocm/`. Its main entry points are `build_and_benchmark.ps1`, `run_benchmarks.py`, `run_worker.py`, and `run_matrix.py`. The deterministic dataset seed is `20260809`.

The harness compares four modes when running full production/stress coverage: legacy Windows DLL CPU, legacy Windows DLL OpenCL GPU, LightGBM 4.7 CPU, and LightGBM 4.7 native Windows ROCm/HIP. The ROCm mode currently uses the CLI because the HIP shared `_lightgbm.dll` still crashes during C-API booster creation with `device_type=cuda`; CPU C API and HIP CLI work. `device_type=gpu` is the legacy OpenCL backend, while `device_type=cuda` selects the CUDA/HIP backend.

ROCm correctness is primarily compared to LightGBM 4.7 CPU because they share source-version semantics. The legacy CPU/OpenCL modes are compatibility and performance references.

Production profiles are H64 (`max_bin=63`, Stage-1-like, feature_fraction 0.9, bagging_fraction 0.7) and H128 (`max_bin=127`, representative Stage-2 depth-6 case, feature_fraction 1.0, bagging_fraction 0.9 with light L1/L2 regularization). Binary LightGBM `.bin` datasets must be keyed by objective and `max_bin`; bin boundaries are serialized, so one binary dataset must not be reused across different `max_bin` values.

The old OpenCL backend is an architectural and performance reference. On production-shaped workloads its historical measurements are roughly 18 ms/tree for H64 and 38 ms/tree for H128, while old H256 synthetic numbers around 5 ms/tree are not representative of current production targets.
