---
description: Confirmed H64/H128 feature4 correctness root cause, accepted optimizations, and rejected histogram experiments.
---

# Histogram correctness and experiment history

The H64/H128 correctness failure in the current gfx1030 feature4 path is now source-confirmed. `CUDARowData` sets `shared_hist_size_ = 2048` and enables `use_gfx1030_feature4_` for gfx1030 dense single-precision ROCm data. `DivideCUDAFeatureGroups()` derives `max_num_bin_per_partition = shared_hist_size_ / 2`, i.e. 1024 histogram bins per partition. That naturally packs about 16 H64 features or 8 H128 features into one partition.

`CUDAConstructHistogramDenseFeature4Gfx1030Kernel`, however, is hard-coded to four feature slots (`GFX1030_FEATURE4_NUM_FEATURES = 4`) and only accumulates/writes those first four columns. The dispatcher selects this kernel for the entire dense uint8 gfx1030 path without checking that every partition actually contains at most four columns. H256 happened to be safe because roughly four 256-bin columns fit into the 1024-bin partition budget, which explains why the same path passed H256 correctness but fails H64/H128 broadly.

Therefore the immediate correctness action is to contain the old feature4 path to layouts that really satisfy its four-column invariant, or route H64/H128 back through the generic kernel until dedicated kernels are ready. No performance measurement from the broken H64/H128 feature4 path is valid.

The strict matrix had already shown failures in all twelve H64/H128 scenarios, including `regression_l2`, with large prediction/tree-structure divergence and an H128 `scale_pos_weight=16` case degenerating to one tree. The source-level partition mismatch now provides a concrete production-path explanation for those failures.

Accepted optimizations from the earlier path include gfx1030 histogram grid sizing, tuned data-per-thread, reduced CUDA data-partition synchronization, reduced leaf-initialization synchronization, histogram reset queued on the histogram stream, and stream-scoped partition synchronization.

Rejected experiments include naive concurrent smaller/larger best-split execution because shared state broke correctness; register accumulation that changed floating-point accumulation order and tree structure; direct AMD LDS atomic builtin substitution when it was slower; ordered gradient/Hessian gathers with no speedup; H/G staggering with no clear win; per-grid-y subhistograms with deterministic reduction but no speedup; unsafe atomic/wave compiler flags that regressed performance; and MSVC-host plus AMD-clang HIP C++ object mixing that produced incorrect models across the compiler ABI/semantic boundary.

Do not repeat rejected experiments without a new reason or a materially different ownership boundary.
