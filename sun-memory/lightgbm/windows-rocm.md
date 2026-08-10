---
description: Windows ROCm LightGBM optimization context, production H64/H128 workloads, RX 6800 XT constraints, benchmark rules, and architecture roadmap.
---

# Windows ROCm LightGBM optimization

## Goal and hardware

Optimize LightGBM 4.7.0 native Windows ROCm/HIP training for an AMD Radeon RX 6800 XT (Navi21, gfx1030, RDNA2) while preserving LightGBM model correctness. The host is Windows 11 with ROCm/HIP SDK 6.2 and Visual Studio 2022. HIP device code is compiled with AMD clang targeting gfx1030.

The historical baseline is the user's older Windows LightGBM DLL, which supports CPU and OpenCL GPU. Native Windows HIP CLI works; the experimental HIP shared DLL still crashes during C-API booster creation with device_type=cuda, so benchmark ROCm mode uses the CLI.

## Production-shaped workloads

The main dataset shape is approximately 40,000 training rows by 3,000 features, with about 50,000 validation rows. The workload is very wide relative to row count, so histogram architecture should emphasize feature parallelism and reuse of row indices, gradients, and Hessians across feature groups.

Stage 1 uses binary/AUC, learning_rate 0.075, max_depth 6, min_child_samples 50, feature_fraction 0.9, subsample 0.7 with subsample_freq 1, max_bin 63, force_col_wise true, and typically a resolved leaf count around the depth-6 profile.

Stage 2 is dynamic. Tree profiles span depth 2/4 leaves, depth 3/8 leaves, depth 4/16 leaves, depth 5/32 leaves, depth 6/40 leaves, and depth 8/120 leaves. max_bin is 64 for depth <=2, 64 or 127 for depth 3, and 127 for depth >=4. feature_fraction is 1.0. subsample is typically 0.85, 0.9, or 1.0 depending on the regularization profile.

Therefore the optimization targets are specialized H64 and H128 histogram engines, not the old synthetic max_bin=255 path.

## Benchmark harness

Benchmark sources live under benchmarks/windows_rocm/. The harness compares four modes on the same LightGBM binary dataset within a profile: old DLL CPU, old DLL OpenCL GPU, LightGBM 4.7 CPU, and LightGBM 4.7 ROCm/HIP GPU.

The harness has production-oriented profiles:

- h64: Stage-1-like, max_bin=63, max_depth=6, num_leaves=40, min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, learning_rate=0.075.
- h128: representative Stage-2 depth-6 case, max_bin=127, max_depth=6, num_leaves=40, min_data_in_leaf=50, feature_fraction=1.0, bagging_fraction=0.9, bagging_freq=1, light L1/L2 regularization, learning_rate=0.05.

Binary datasets are keyed by max_bin because bin boundaries are encoded in the LightGBM binary dataset. Never reuse a max_bin=255 binary dataset when measuring H64/H128 behavior.

Correctness is mandatory before accepting performance work. ROCm is gated against LightGBM 4.7 CPU, not the older DLL, because CPU and HIP must share the same source-version semantics. The legacy CPU/OpenCL modes remain compatibility and performance references. Binary checks cover finite probability outputs, probability range/non-constancy, requested tree count, tree structure, AUC, tight prediction agreement/correlation, identical hard labels at 0.5, and identical confusion matrix. Regression checks cover finite/nonconstant predictions, tree count/structure, prediction agreement/correlation, and RMSE.

A strict matrix now covers H64/H128 baselines, no-bagging variants, feature_fraction=0.5, strong regularization, scale_pos_weight=16, and regression_l2. Smoke uses six representative profiles at 20 trees; stress covers all twelve at 20 trees; production uses H64/H128 at 100 trees. On the current gfx1030 feature4 implementation all twelve production-sized stress profiles fail against v4.7 CPU, including regression_l2, so the defect is not specific to binary probability conversion or class weighting. H128 with scale_pos_weight=16 can degenerate to a one-tree ROCm model. This makes restoration of H64/H128 correctness the first requirement of the new histogram architecture.

## Confirmed architecture observations

The dominant ROCm cost has been histogram construction. Earlier profiling on the old max_bin=255 workload showed ConstructHistogramForLeaf consuming roughly two thirds of train time. Micro-optimizations to synchronization, grid sizing, and initialization helped, but did not approach the old OpenCL implementation.

The older OpenCL implementation contains highly specialized histogram16/histogram64/histogram256 kernels. histogram64 is especially relevant to Stage 1. It uses packed four-feature loads, small bank-aware LDS histograms, feature rotation across lanes, prefetching, temporal accumulation, root/no-index specialization, and AMD-specific layout decisions.

The current CUDA/HIP implementation was designed generically and has different work decomposition, synchronization, and shared-memory behavior. A gfx1030 feature4 fast path was added earlier, but it was designed around max_bin=255 and is not the final architecture for production H64/H128.

## Optimization roadmap

The architectural sequence is:

1. Exact OpenCL-style H64/H128 reference architecture in HIP. Reproduce the proven histogram64 design closely enough to establish a fair HIP control. H128 should be a dedicated kernel rather than falling back to histogram256 semantics.
2. Feature SuperTile H64/H128 plus native wave32 decomposition. For H64, a promising target is roughly 16 features per workgroup with four wave32s and about 8 KB of base histogram LDS; H128 likely uses 8 or 16 features depending on LDS/VGPR occupancy measurements. The purpose is to reuse row index, gradient, and Hessian state across multiple feature4 groups instead of launching hundreds of independent feature4 partitions.
3. Wave-level equal-bin aggregation. Before updating LDS, combine lanes in a wave32 that target the same bin so LDS atomic pressure scales with unique bins per wave rather than rows. This is especially promising for 64 bins. Preserve deterministic/correct numerical behavior; previous changes to floating-point accumulation order have changed tree structure.
4. Histogram/BestSplit fusion. Once the histogram engine is fast and correct, evaluate split gains while the histogram is hot and fuse subtraction with larger-child best-split evaluation where practical. This is a later architectural step because it changes ownership boundaries between histogram construction and best-split search.

The final engine should likely expose separate compile-time-specialized H64 and H128 kernels, with distinct root and indexed-leaf paths, rather than one universal GPU histogram kernel.

## Prior experiments and constraints

Useful accepted optimizations include gfx1030 histogram grid sizing, tuned data-per-thread, reduced data-partition synchronization, reduced leaf-initialization synchronization, histogram reset queued on the histogram stream, and stream-scoped partition synchronization.

Experiments that were rejected include naive concurrent smaller/larger best-split kernels because they broke correctness, host MSVC plus clang HIP object mixing because it produced incorrect models due to cross-compiler C++ ABI/semantic issues, direct replacement of histogram atomics with different scopes when no speedup resulted, aggressive register accumulation that changed split decisions, and several sub-histogram/coalescing experiments that did not improve performance.

A previous active striped-row experiment may exist as an uncommitted transaction; always inspect Git status and active transactions before modifying code. Preserve unrelated user work.

## Build/runtime facts

Use native Windows HIP, not WSL, for RX 6800 XT. The WSL path could enumerate gfx1030 but GPU execution stalled. The stable native HIP toolchain uses ROCm 6.2 AMD clang with gfx1030 target and VS2022 environment. The LightGBM build uses device_type=cuda for the HIP backend; device_type=gpu is the legacy OpenCL backend.
