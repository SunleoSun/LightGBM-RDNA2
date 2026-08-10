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

## Phase 3 - adaptive Feature SuperTile

The H64 SuperTile experiment is now implemented and accepted only as an adaptive large-leaf path. The best tested shape is eight features per 256-thread workgroup (two adjacent Feature4 tuples) with four double-precision LDS banks. Wider H64 tiles were slower: 12 features was about `31.6 ms/tree`, 16 features about `31.4`, and eight features with eight banks about `31.3`. The eight-feature/four-bank path was best, but only for large leaves. A leaf-size threshold of 16384 rows gave the best production result around `30.15 ms/tree`; 8192 was about `30.67` and 24576 regressed to about `31.84`. Smaller leaves therefore remain on the original four-feature kernel.

The first H128 eight-feature SuperTile was exact but slower at about `47.6 ms/tree`, so H128 remains on the four-feature reference kernel. This confirms that row-state reuse must be balanced against LDS occupancy/atomic pressure rather than applied uniformly.

Bottleneck attribution is now measured on 100-tree production profiles with a dedicated `USE_TIMETAG=ON` build. H64: RDNA2 histogram dispatch ~18.66 ms/tree, split scan ~4.31, full tree training ~24.05. Inside histogram dispatch: index H2D ~3.46, memset ~1.74, HIP kernel ~5.12, histogram D2H ~4.18, pinned-staging-to-canonical host memcpy ~4.13. H128: histogram dispatch ~28.33 ms/tree, split scan ~10.28, full tree training ~39.79; index H2D ~3.76, memset ~2.00, kernel ~6.50, D2H ~7.33, host memcpy ~8.72. Gradient/Hessian H2D is only ~0.36-0.39 ms/tree. Production H64/H128 had zero Serial histogram fallbacks. CPU reference split-scan times are nearly the same (~4.65 H64, ~10.84 H128), confirming split semantics are not the reason RDNA2 differs from CPU.

Direct D2H into the canonical Serial histogram is now implemented. `RDNA2HistogramEngine` lazily host-registers each canonical histogram-pool destination, writes the device histogram directly into that pinned buffer, and unregisters all buffers before the base `HistogramPool` is destroyed. If registration fails, the previous pinned-staging path remains as a correctness fallback. TIMETAG confirms `host_copy_ms=0`; production 100-tree results improved to about `28.25 ms/tree` H64 and `41.30 ms/tree` H128 with prediction diff `0` and exact CPU structure.

Canonical active-feature suppression is now implemented. `RDNA2TreeLearner` passes the exact per-dispatch `is_feature_used` vector into `RDNA2HistogramEngine`; the engine maps dense feature groups back to canonical feature indices, uploads the scattered mask, and suppresses both LDS atomics and output writes for disabled features. There is no assumed fraction/count/range. The dedicated `feature_fraction` matrix at 0.5/0.75/0.9/1.0 passes with prediction diff `0` and exact CPU structure. On 100-tree runs, H64 at feature_fraction=0.5 measured about `25.49 ms/tree` versus `28.30` with all features, and H128 at 0.5 about `34.67` versus `42.32` with all features. Full-grid memset/readback remain, so this stage intentionally suppresses feature compute before introducing compact chunk descriptors.

Two CPU/RDNA2 overlap prototypes were tested and rejected in their current form. Feature-chunk histogram/D2H plus CPU split overlap stayed exact but added enough launch, transfer, and OpenMP-region overhead to regress: roughly `65/68 ms/tree` at 64 tuples per chunk, `34/46` at 256 tuples, and `30/44` at 512 tuples for H64/H128, all worse than the monolithic path. Concurrent feature ownership was also exact but slower. The stock CPU histogram constructor paid full ordered-gradient/Hessian gather and OpenMP overhead even for a small CPU feature share; a direct selected-feature CPU histogram avoided that gather but still regressed representative 100-tree runs to roughly `32.1 ms/tree` H64 and `47.9` H128. Do not reintroduce either overlap shape without a materially different batching/ownership mechanism.

The next accepted boundary optimization is pinning canonical `DataPartition::indices()` host storage. `RDNA2HistogramEngine` registers the whole stable index array after Serial initialization and unregisters it before any training-data resize/destruction; indexed smaller-leaf H2D then reads from an already pinned subrange. TIMETAG reduced aggregate index-copy time from roughly `3.46` to `1.84 ms/tree` on H64 and from `3.76` to `2.02 ms/tree` on H128 while preserving exact smoke correctness. Normal wall-clock runs are noisy enough that this micro-optimization should be judged by attributed transfer time rather than a single end-to-end sample.

The first minimal partition-mirror step is now implemented without adopting CUDA partition semantics. After `SerialTreeLearner::Split` finishes and initializes the next smaller leaf, RDNA2 immediately enqueues that leaf's pinned `DataPartition` index subrange into the persistent device index buffer. The next histogram consumes the preloaded device indices on the same stream instead of issuing a critical-path copy. Bagged roots are preloaded after `BeforeTrain`. TIMETAG observed preload hits on all 2516 H64 and 2589 H128 histogram calls, zero fallback copies, and `index_h2d_ms=0` inside `ConstructHistogram`; smoke, feature_fraction, and the full representative Optuna envelope remain prediction-diff `0` with exact CPU structure. This is intentionally still a host-canonical mirror: Serial owns partition order and split decisions.

The immediate optimization order is now: (1) reduce full-grid histogram memset/D2H for scattered active features through compact active tuple/range descriptors while retaining monolithic launches where they win; (2) revisit root and constant-Hessian specialization; (3) profile/tune H64/H128 kernel internals again with the host/device boundary overhead reduced. A deeper GPU-side `DataPartition` implementation is no longer the immediate next step because early preloading removes the index copy from the histogram critical path without taking partition ownership. Moving objective or best-split wholesale to GPU remains later because their current costs do not justify the semantic risk.

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
