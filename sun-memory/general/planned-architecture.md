---
description: Planned LightGBM-RDNA2 architecture and the ordered H64/H128 optimization roadmap.
---

# Planned architecture

The architectural sequence is intentionally staged so every performance experiment sits on a correctness-safe reference.

1. Build an OpenCL-style HIP H64 reference kernel and then a separate H128 specialization. The first goal is semantic correctness and a fair architectural control against the proven OpenCL histogram implementation, not peak speed.
2. Build an RDNA2 Feature SuperTile engine that reuses row indices, gradients, and Hessians across more features per workgroup. H64 is expected to fit roughly 16 features per workgroup with native wave32 decomposition and about 8 KB of base histogram LDS; H128 should choose 8 or 16 features based on measured LDS/VGPR occupancy.
3. Add wave32 equal-bin aggregation only after the SuperTile path is correct. Combine lanes targeting the same bin before LDS updates to reduce atomic pressure, while guarding against floating-point accumulation-order changes that alter splits.
4. Consider histogram / best-split fusion only after histogram correctness and performance are established. This changes ownership across histogram construction, subtraction, and best-split evaluation and therefore comes last.

The expected long-term component shape is conceptually:

`RDNA2HistogramEngine -> PackedDataset, H64, H128, Root, IndexedLeaf, WaveHistogramPrimitive, Merge/Subtract, optional BestSplitFusion`.

Generic CUDA behavior remains the fallback for other GPUs. The H64 and H128 kernels should be compile-time specialized rather than routed through a universal 256-bin implementation.
