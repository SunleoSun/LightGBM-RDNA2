---
description: Source-grounded optimization order for RX 6800 XT gfx1030, from safe dispatch through H64/H128 reference kernels, SuperTile, wave32, subtraction, and split fusion.
---

# RX 6800 XT optimization roadmap

This ordering follows the actual production path and keeps each performance stage behind a correctness gate. The dominant owner is `CUDAHistogramConstructor`; `CUDARowData` owns dense row packing/feature partitions, and `CUDABestSplitFinder` remains a later consumer of the durable histograms.

## Phase 0 - contain broken specialization and establish the diagnostic baseline

The old gfx1030 feature4 kernel is valid only when the selected partition contains at most four columns. Enforce that at dispatch and assert it at launch. This preserves the proven H256 fast path while H64/H128 bypass it. Do not force H64/H128 partitions down to four columns merely to make them enter feature4; that reuses the wrong architecture instead of fixing the production workload.

The generic HIP fallback is a diagnostic baseline, not a strict-correctness reference. After bypassing feature4 it removes the catastrophic H64/H128 corruption, but strict CPU equality still fails on late splits. H128 regression is especially informative: the first five trees match CPU and the sixth differs only at its final split. Use this baseline to distinguish gross mapping/layout defects from smaller numerical or split-selection differences.

The August 9 synchronization reductions and gfx1030 grid/data-per-thread tuning have been rollback-tested and do not change this residual mismatch, so they remain enabled. A separate CUDA binary-objective class-weight indexing bug has been fixed and should stay independent of histogram work.

## Phase 1 - H64 OpenCL-style reference

Implement a dedicated dense H64 HIP kernel for gfx1030. Preserve existing LightGBM histogram offsets, missing/default-bin fixup, smaller-child ownership, and subtraction semantics. Use the proven OpenCL `histogram64.cl` as the architectural reference: 256 threads, packed four-feature reads, four LDS banks, bank-aware `(bin, bank, G/H, feature)` placement, lane feature rotation, software prefetch, root/no-index specialization, and optional constant-Hessian handling.

The first H64 milestone is correctness against the LightGBM 4.7 CPU reference and comparison against legacy OpenCL, not peak speed. Because generic HIP can differ on late near-tie splits, inspect the first divergent split rather than assuming every non-bit-identical result is a layout bug. The specialized reference should aim to reproduce the proven OpenCL accumulation/layout behavior closely enough to pass the strict project gate.

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
5. Retain an H256 regression check while the historical feature4 path remains reachable.
6. Compare kernel timers, total training time, tree structure, predictions, AUC/RMSE, and tree count. Performance without correctness is discarded.

The immediate next engineering task is Phase 1: build a dedicated H64 OpenCL-style reference path on top of the now-contained feature4 dispatch, using the generic HIP results only as a diagnostic control.
