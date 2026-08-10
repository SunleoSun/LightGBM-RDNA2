---
description: Windows ROCm benchmark harness architecture, backend roles, production/Optuna coverage, and dataset/correctness invariants.
---

# Benchmark architecture

The benchmark harness lives under `benchmarks/windows_rocm/`. Its main entry points are `build_and_benchmark.ps1`, `run_benchmarks.py`, `run_worker.py`, and `run_matrix.py`. The deterministic dataset seed is `20260809`.

The canonical correctness comparison is LightGBM 4.7 CPU versus `device_type=rdna2`. Full/compatibility runs can additionally report the legacy Windows CPU/OpenCL DLL and `v470_cuda_diagnostic`. The native ROCm backend uses the CLI because the HIP shared `_lightgbm.dll` still crashes during C-API booster creation on the CUDA-style path; CPU C API and native CLI work. `device_type=gpu` is the legacy OpenCL backend, `device_type=cuda` is retained as a diagnostic CUDA/HIP learner, and `device_type=rdna2` is the fork-specific correctness/performance target.

CPU 4.7 is the source of truth for RDNA2 tree/prediction correctness. Legacy OpenCL remains an architectural/performance compatibility reference, not a universal exactness oracle: in Optuna-compat probes it matched CPU structure at depth 2 and depth 4 with prediction differences around `3.1e-7` and `5.6e-7`, but at depth 8 / 120 leaves it selected a different structure with prediction max difference about `0.00627`. RDNA2 remained bit-for-bit identical to the CPU reference in all three probes.

Production profiles are H64 (`max_bin=63`, Stage-1-like, feature_fraction 0.9, bagging_fraction 0.7) and H128 (`max_bin=127`, representative Stage-2 depth-6 case, feature_fraction 1.0, bagging_fraction 0.9 with light L1/L2 regularization). Binary LightGBM `.bin` datasets are keyed by objective and `max_bin`; bin boundaries are serialized, so max-bin 63, 64, and 127 use distinct canonical binaries.

The `optuna` suite is a representative envelope rather than a Cartesian product. It covers depths `2/3/4/5/6/8`, leaf limits `4/8/16/32/40/120`, Stage-1 `max_bin=63`, Stage-2 `max_bin=64/127`, all production regularization-profile shapes from very-light through ultra-extreme, bagging fractions `0.7/0.85/0.9/1.0`, and learning rates `0.085/0.075/0.06/0.05/0.04/0.03/0.025/0.02`. The regular smoke suite separately covers `scale_pos_weight=16`; the Optuna envelope covers the upper bound `64`.

`optuna_long` maps all eight production boost profiles to their actual tree horizons: 80/0.085, 120/0.075, 200/0.06, 300/0.05, 450/0.04, 650/0.03, 850/0.025, and 1050/0.02. This suite is a milestone/pre-merge correctness gate, not a per-kernel smoke test. The 80-, 120-, 450-, and 1050-tree representative runs have already been observed with exact CPU/RDNA2 predictions and structures; the remaining configured long horizons should be exercised before a major optimized backend stage is accepted.

The old OpenCL backend is also a performance reference. On production-shaped workloads historical measurements are roughly 18 ms/tree for H64 and 38 ms/tree for H128, while old H256 synthetic numbers around 5 ms/tree are not representative of current production targets. The first exact RDNA2 HIP references measured about `31.804 ms/tree` for H64 versus CPU `49.832`, and `45.525 ms/tree` for H128 versus CPU `72.007`, each over 100 production-shaped trees with prediction diff `0` and exact structure. Short six-tree RDNA2 timings are dominated by one-time ROCm initialization and should not be compared directly with amortized per-tree targets.
