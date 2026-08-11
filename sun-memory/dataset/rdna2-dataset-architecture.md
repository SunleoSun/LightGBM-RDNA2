---
description: RDNA2 canonical-bin ownership, implemented packed GPU representation, measured packing/upload costs, cache lifecycle, and dataset-construction optimization stages.
---

# RDNA2 dataset architecture

The RDNA2 backend separates semantic dataset ownership from the physical layout used by HIP kernels. LightGBM `Dataset`, `FeatureGroup`, and `BinMapper` remain the canonical source of feature/bin meaning. RDNA2 must not invent different cut points or rebucket raw values while learner correctness is being established.

The ownership chain is:

`raw input -> canonical LightGBM Dataset/BinMapper -> RDNA2DatasetView -> RDNA2PackedDataset -> RDNA2HistogramEngine`

`RDNA2DatasetView` is the semantic adapter. It exposes canonical feature-to-bin mappings, missing/default-bin semantics, feature-group metadata, row count, max-bin specialization, and the row/column access needed to build the packed representation. It must not redefine bin boundaries.

The first `RDNA2HistogramEngine` dataset layer is implemented and is now the production input for the exact H64/H128 HIP histogram paths. It accepts only gfx1030, enumerates dense single-feature groups, reads their exact canonical bin IDs through `DenseBinIterator`, packs four byte-sized groups into one `uint32`, and uploads tuple-major `[feature4][row]` storage to persistent device memory. It deliberately does not apply legacy OpenCL bin multipliers/random bank offsets because the packed values must remain canonical LightGBM bins. H64/H128 eligibility is detected from canonical per-group bin counts. `RDNA2TreeLearner` constructs this representation during initialization; only unsupported histogram cases fall back to Serial construction.

For the 40k x 3000 production-shaped H64 dataset, the packed device image is about 114.44 MiB (`750 feature4 tuples * 40000 rows * 4 bytes`). A representative gfx1030 probe measured about 25 ms for CPU packing and about 12 ms for the 114 MiB H2D copy after allocation. The first ROCm device allocation/context path cost about 456 ms in that process, which dominates one-time RDNA2 initialization; this cost is distinct from the actual packing and transfer and should not be mistaken for per-tree work. The temporary host packed vector is released when initialization returns; the device representation persists for the learner lifetime.

The packing loop has a production fast path for four ordinary 8-bit dense groups: it reads the four canonical iterators in one row pass and writes one aligned `uint32`, avoiding four read/modify/write passes over the 114 MiB staging buffer. Dense 4-bit groups and incomplete final tuples use a correctness fallback.

Future `RDNA2PackedDataset` work may change the physical representation as long as every packed value maps back to the same canonical bin. The existing aligned `uint32 feature4[group][row]` layout is sufficient for the first SuperTile implementation: a SuperTile can read several adjacent Feature4 tuple bases for the same row without repacking the dataset. H64 and H128 may diverge physically later if profiling proves a dedicated tile layout worthwhile. Repacking per tree or boosting iteration is forbidden.

Histogram readback now targets the Serial-owned canonical histogram buffer directly. `RDNA2HistogramEngine` lazily host-registers each histogram-pool destination and performs async D2H into that pinned memory on the persistent HIP stream, then synchronizes once before CPU consumers run. Registered buffers are unregistered before the base histogram pool is destroyed. Reusable pinned staging remains only as a fallback when host registration fails. This preserves canonical ownership and removes the former staging-to-histogram memcpy.

Dataset construction remains an explicit optimization surface because production observations show it can consume comparable or greater wall time and peak RAM than training. Measure raw parse/load, sampling for bin construction, `BinMapper` construction, canonical bin population, RDNA2 packing, device allocation/context initialization, H2D upload, and training separately. Peak host RAM and device RAM belong in the benchmark contract alongside time.

Optimization order for dataset work:

1. Preserve canonical CPU-created bins and persistent one-time packed GPU storage. This foundation is implemented.
2. Add structured dataset timing and peak-memory measurements so initialization costs are attributed correctly.
3. Reduce the transient host staging footprint with chunked/pinned transfer only if profiling shows it matters; preserve large/coalesced transfers rather than replacing them with hundreds of tiny H2D copies.
4. Add a persistent RDNA2 packed cache for repeated Stage-1/Stage-2 training on identical data and binning configuration. Cache identity must include binning-relevant configuration, especially `max_bin`; H64/H128 caches are not interchangeable.
5. Consider memory mapping and overlapped packing/H2D when the lifetime and ownership contracts are stable.
6. Only if canonical bin construction itself remains dominant, investigate `RDNA2DatasetBuilder` or GPU-assisted bin construction. Any accelerated builder must reproduce canonical `BinMapper` behavior for sampling, missing values, categorical handling, forced bins, `min_data_in_bin`, and feature prefiltering.

The long-term component relationship remains:

`RDNA2TreeLearner(Serial semantics) -> RDNA2HistogramEngine -> RDNA2DatasetView -> RDNA2PackedDataset`

A production-shape split-local matrix benchmark now establishes the dataset baseline against pristine upstream v4.7 CPU. On 40k x 3000 C-contiguous float32 input with 32 threads, H64 measured about 2.956 s for pristine v4.7 dataset construction versus about 2.616 s for the current fork candidate, and H128 about 3.069 s versus 2.700 s. Both paths peak roughly 1.54-1.55 GiB above the already-resident ~487 MiB input process working set, confirming transient dataset construction memory as the dominant initialization footprint. Candidate and canonical datasets passed the isolated CPU-oracle semantic gate. A production H64 RDNA2 initialization probe measured about 21.993 ms CPU Feature4 packing, 14.007 ms allocation/context work, and 8.972 ms H2D for the 114.44 MiB packed image; logical feature-input-to-train-ready time was about 2.662 s, so canonical dataset construction dominates pre-train latency. The first optimization target should therefore preserve v4.7 `BinMapper` semantics while eliminating wide-matrix temporary materialization/allocation; GPU-assisted value-to-bin population is a later target only after CPU-side construction is decomposed and re-profiled.
