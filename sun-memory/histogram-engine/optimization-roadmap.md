---
description: Current RDNA2 optimization roadmap for RX 6800 XT/gfx1030 after exact H64/H128 HIP histograms and stream/pinned transfer cleanup.
---

# RX 6800 XT optimization roadmap

The production owner is `device_type=rdna2`: `RDNA2TreeLearner` preserves `SerialTreeLearner` objective, score, split-selection, leaf bookkeeping, and subtraction semantics while `RDNA2HistogramEngine` owns gfx1030 data layout and HIP histogram construction. `device_type=cuda` remains diagnostic; `device_type=gpu` remains legacy OpenCL. CPU LightGBM 4.7 is the correctness oracle.

## Completed foundation

Phase 0 is complete. `device_type=rdna2` is an explicit single-machine backend, uses the canonical CPU-created `Dataset` / `BinMapper`, and does not route through CUDA objective, score updater, best-split finder, or data partition semantics.

The packed dataset foundation is complete for the dense production path. Exact canonical bin IDs are packed once as persistent tuple-major `uint32 feature4[group][row]` storage. For 40k x 3000 this is about 114.44 MiB. The representation is derived from canonical bins and never owns binning semantics.

## Completed H64 and H128 reference kernels

H64 and H128 now have exact HIP histogram producers. Both consume persistent Feature4 data, use 256-thread workgroups and four double-precision LDS banks, support the root and indexed smaller-leaf paths, and write directly into the canonical Serial histogram layout. Gradients/Hessians are uploaded once per tree. Indexed smaller-leaf row IDs are uploaded when needed. Parent-minus-smaller subtraction and best-split evaluation stay on the Serial host path. Cases that require independently constructing both children still use the Serial histogram fallback.

The smoke suite and representative Optuna envelope pass with prediction max diff `0` and exact CPU 4.7 tree structure. The first production 100-tree baselines were about `31.804 ms/tree` H64 versus CPU `49.832`, and `45.525 ms/tree` H128 versus CPU `72.007`.

## Completed transfer/synchronization cleanup

The histogram pipeline now owns one persistent HIP stream. Gradient/Hessian and indexed-leaf H2D copies, histogram memset, kernel launch, and D2H are ordered on that stream. The old device-wide synchronizations around every histogram readback were replaced by one stream synchronization. Histogram D2H targets reusable pinned host staging memory and is then copied into the exact Serial-owned histogram buffer.

This preserves bit-for-bit CPU correctness. Representative 100-tree measurements after the change put H64 around `30.7 ms/tree`, roughly a 3% improvement over the first exact H64 reference. H128 stayed approximately neutral around `46 ms/tree` versus the earlier `45.5 ms/tree` observation, so transfer cleanup is not the main H128 bottleneck. A two-LDS-bank H128 geometry was also neutral and was reverted to four banks.

## Immediate Phase 3 - Feature SuperTile

The largest remaining architectural waste is repeated row-state loading. With about 3000 features the current four-feature kernel creates roughly 750 Feature4 workgroups, and each tuple independently reloads the same row index, gradient, and Hessian. The next task is SuperTile reuse while retaining the current canonical packed dataset and double-precision accumulation semantics.

For H64, start with 16 features per workgroup (four adjacent Feature4 tuples), 256 threads, and four double LDS banks. Base shared histogram storage is `16 * 64 * 2 * 8 * 4 = 64 KiB`, permitting two such workgroups per 128 KiB LDS CU in the LDS-only limit. The grid drops from about 750 tuple workgroups to about 188 SuperTiles, while each row loads gradient/Hessian once for sixteen features instead of four. Validate exact CPU structure/predictions before tuning feature rotation, prefetch, or temporal accumulation.

For H128, test an independent eight-feature SuperTile first. `8 * 128 * 2 * 8 * 4` is also 64 KiB. Do not automatically use the H64 tile width; measured LDS/VGPR occupancy and atomic pressure decide the final geometry.

## Later phases

After SuperTile is exact and faster, tune root/no-index specialization, workgroup and bank geometry separately for H64/H128, safe prefetch/temporal aggregation, and constant-Hessian handling. The float-LDS experiment is rejected because it exceeded the strict prediction tolerance and was slower.

Native wave32 equal-bin aggregation comes only after SuperTile is stable. Accumulation-order changes are correctness-sensitive and must pass the full gate.

Dataset construction optimization remains a parallel end-to-end objective: structured timing/peak-RAM measurement, persistent packed caches, fewer host copies, and only later GPU-assisted canonical bin construction if binning itself remains dominant.

Best-split fusion or moving split finding onto GPU remains late-stage work. It changes ownership and numerical semantics and must not be used to recover performance before the histogram/data path is exhausted.

## Validation order

1. Focused H64/H128 profile after each kernel change.
2. Full `smoke` suite.
3. Representative `optuna` envelope after a completed optimization.
4. 100-tree production H64/H128 performance comparison only after correctness passes.
5. `optuna_long` before accepting a major architectural stage such as SuperTile/wave32.
6. Keep H256 as a regression check for the separate historical CUDA diagnostic Feature4 path.
