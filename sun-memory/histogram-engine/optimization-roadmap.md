---
description: Source-grounded optimization order for RX 6800 XT gfx1030, from correctness containment through H64/H128, SuperTile, wave32, subtraction, and split fusion.
---

# RX 6800 XT optimization roadmap

This ordering follows the actual production path and keeps each performance stage behind a correctness gate. The dominant owner is `CUDAHistogramConstructor`; `CUDARowData` owns dense row packing/feature partitions, and `CUDABestSplitFinder` remains a later consumer of the durable histograms.

## Phase 0 - restore a trustworthy baseline

Contain the existing gfx1030 feature4 kernel to its real invariant: at most four columns in every selected partition. H64/H128 currently produce wider partitions because the 2048-float shared-hist budget maps to 1024 bins, allowing roughly 16 H64 or 8 H128 columns. Until a specialized kernel exists, H64/H128 should use a correctness-safe fallback. Run smoke immediately after this containment and then the strict matrix before using ROCm timing numbers.

Add narrow diagnostics or assertions around partition width, bin count, and selected kernel path so future specialization cannot silently receive an incompatible row layout. The kernel-dispatch contract should encode H64/H128/H256 meaning explicitly instead of inferring semantics only from `bit_type == 8` and architecture.

## Phase 1 - H64 OpenCL-style reference

Implement a dedicated dense H64 HIP kernel for gfx1030. Preserve existing LightGBM histogram offsets, missing/default-bin fixup, smaller-child ownership, and subtraction semantics. Use the proven OpenCL `histogram64.cl` as the architectural reference: 256 threads, packed four-feature reads, four LDS banks, bank-aware `(bin, bank, G/H, feature)` placement, lane feature rotation, software prefetch, root/no-index specialization, and optional constant-Hessian handling.

The first H64 milestone is smoke correctness, not speed. Then run production H64 and compare both total tree time and histogram timer against LightGBM 4.7 CPU and the legacy OpenCL reference.

## Phase 2 - dedicated H128 reference

Create an independent H128 specialization rather than stretching the H64 or H256 kernel. Select workgroup size, bank count, and features-per-workgroup from measured LDS/VGPR occupancy on gfx1030. H128 must pass the same binary/regression correctness gates and histogram subtraction lifecycle before any architecture experiments are layered on top.

## Phase 3 - packed dataset and Feature SuperTile

Move from four-feature partitions toward a row-state-reuse design. The production matrix has about 3000 features but only about 40k rows, so repeatedly loading the same leaf index, gradient, and Hessian for hundreds of feature groups is the major architectural waste.

For H64, evaluate approximately 16 features per workgroup with four native wave32s. The base G/H storage for `16 * 64 * 2 * 4 bytes` is about 8 KB LDS before banking/auxiliary state. This reduces conceptual feature groups from about 750 four-feature groups to about 188 sixteen-feature groups and gives substantially more reuse of row-state. H128 should test 8 versus 16 features per workgroup based on occupancy and bank behavior rather than assuming the H64 geometry.

If needed, add an explicit GPU packed representation such as `uint32 feature4[group][row]` so aligned four-byte feature loads become the canonical dense H64/H128 input instead of reinterpret-casting arbitrary partition rows.

## Phase 4 - root, large-leaf, and constant-Hessian specializations

The root leaf can avoid index gathers entirely. Large leaves may benefit from a linear scan path while indexed gathers remain the default for smaller children. Benchmark this as a separate dispatch decision; do not mix membership semantics into the core histogram math.

For objectives with constant Hessian, evaluate gradient-plus-count histograms and derive Hessian from count times the constant Hessian. `regression_l2` is the most useful correctness profile for this specialization.

## Phase 5 - native wave32 equal-bin aggregation

Once SuperTile H64/H128 is correctness-stable, use RDNA2 native wave32 primitives to combine lanes that target the same bin before LDS atomics. H64 is the strongest candidate because 32 lanes mapped into 64 bins create frequent collisions. Preserve the numerical contract carefully: previous accumulation-order changes have altered split selection. Any aggregation method must pass smoke and stress before performance is accepted.

## Phase 6 - histogram merge, fixup, and subtraction

After construction is fast, profile the global write/merge path. Current kernels emit durable `hist_in_leaf` data and later run `FixHistogramKernel` followed by `SubtractHistogramKernel` for the larger child. Optimize these only after preserving the canonical histogram layout and most-frequent-bin reconstruction semantics. Potential work includes reducing global atomics between workgroups, specializing the single-workgroup/root case, and fusing reduction with durable write where deterministic behavior is retained.

Histogram subtraction is especially important for Stage-2 trees with many small leaves; do not replace the smaller-child/subtract strategy with rebuilding both children.

## Phase 7 - best-split scheduling and optional fusion

`CUDABestSplitFinder::FindBestSplitsForLeaf` consumes the completed smaller/larger histograms and synchronizes after split kernels; `FindBestFromAllSplits` has another device synchronization. Only after histogram construction, merge, and subtraction stop dominating should these boundaries be optimized.

Possible later work is feature-tile histogram -> prefix/gain evaluation while data is hot, and parent-minus-smaller subtraction combined with larger-child split evaluation. This changes ownership between histogram construction and split finding, so it should be done only with a clear durable-histogram contract and boundary regression tests. Do not repeat the earlier naive concurrent smaller/larger split attempt.

## Validation order for every phase

1. Focused H64 or H128 smoke profile after each kernel/dispatch change.
2. Full smoke suite (`h64`, `h128`, `h64_scale16`, `h128_regression`).
3. Production H64/H128 only when smoke passes.
4. Full strict stress matrix before merging a completed architectural stage.
5. Compare kernel timers, total training time, tree structure, predictions, AUC/RMSE, and tree count. Performance without correctness is discarded.

The immediate next engineering task is Phase 0 followed by Phase 1: fix the feature4 dispatch invariant, establish a correct ROCm H64 baseline, then implement the dedicated H64 reference kernel.
