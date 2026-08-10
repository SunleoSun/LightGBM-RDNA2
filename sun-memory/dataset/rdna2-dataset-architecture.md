---
description: Planned RDNA2 dataset ownership, canonical binning contract, packed GPU representation, cache lifecycle, and dataset-construction optimization stages.
---

# RDNA2 dataset architecture

The RDNA2 backend must separate semantic dataset ownership from the physical layout used by HIP kernels. LightGBM `Dataset`, `FeatureGroup`, and `BinMapper` remain the canonical source of feature/bin meaning. The first RDNA2 backend must not invent different cut points or rebucket raw values, because that would mix learner correctness with dataset correctness.

The intended ownership chain is:

`raw input -> canonical LightGBM Dataset/BinMapper -> RDNA2DatasetView -> RDNA2PackedDataset -> RDNA2HistogramEngine`

`RDNA2DatasetView` is a zero-copy or low-copy semantic adapter where possible. It exposes canonical feature-to-bin mappings, missing/default-bin semantics, feature-group metadata, row count, max-bin specialization, and the row/column access needed to build the packed representation. It must not redefine bin boundaries.

`RDNA2PackedDataset` is a derived physical representation owned by the RDNA2 backend. It may use layouts that differ completely from CPU/OpenCL/CUDA storage as long as every packed value maps back to the same canonical bin. H64 and H128 should be independent packed specializations. Candidate layouts include byte-packed bins and aligned `uint32 feature4[group][row]` tiles, evolving toward SuperTile-friendly `[feature tile][row]` storage that reuses row state across 8-16 features.

The first implementation should build the packed representation after canonical Dataset construction and keep it resident for the lifetime of the training Dataset. Repacking per tree or per boosting iteration is forbidden. Train and validation caches must be keyed by dataset identity plus binning-relevant configuration, especially `max_bin`; H64 and H128 caches are not interchangeable.

Dataset construction is an explicit later optimization surface because production observations show it can consume comparable or greater wall time and peak RAM than training. Measure raw parse/load, sampling for bin construction, `BinMapper` construction, canonical bin population, RDNA2 packing, H2D upload, and training separately. Peak host RAM and device RAM belong in the benchmark contract alongside time.

Optimization order for dataset work:

1. Preserve canonical CPU-created bins while introducing `RDNA2DatasetView` and one-time packed GPU storage.
2. Add dataset timing and peak-memory measurements so the dominant stages are known rather than inferred from total initialization time.
3. Remove avoidable copies and repeated row/column transforms while packing; pre-size final buffers and parallelize independent feature/tile packing.
4. Add a persistent RDNA2 packed cache for repeated Stage-1/Stage-2 training on identical data and binning configuration. Loading a compatible cache should bypass repacking and ideally allow direct or staged H2D upload.
5. Consider memory mapping and overlapped packing/H2D when the lifetime and ownership contracts are stable.
6. Only if canonical bin construction itself remains dominant, investigate `RDNA2DatasetBuilder` or GPU-assisted bin construction. Any accelerated builder must reproduce the canonical `BinMapper` contract for sampling, missing values, categorical handling, forced bins, `min_data_in_bin`, and feature prefiltering before it can replace CPU binning.

The long-term component relationship is:

`RDNA2TreeLearner(Serial semantics) -> RDNA2HistogramEngine -> RDNA2DatasetView -> RDNA2PackedDataset`

The dataset layer is therefore optimized early enough to matter to end-to-end latency, but only after the minimal `device_type=rdna2` semantic boundary and first correct histogram offload are established.
