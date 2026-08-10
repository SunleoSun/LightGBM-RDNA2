---
description: Planned LightGBM-RDNA2 architecture and the ordered H64/H128 optimization roadmap.
---

# Planned architecture

The architectural sequence is intentionally staged so every performance experiment sits on a correctness-safe reference. The preferred production direction is now a distinct `rdna2` backend instead of continuing to patch generic `device_type=cuda` semantics. The reason is source- and benchmark-grounded: the current benchmark feeds CPU, legacy OpenCL, and ROCm from the same canonical LightGBM 4.7 CPU-produced `.bin`, while legacy `device_type=gpu` preserves CPU tree structure on H64 and H128 regression and the ROCm/CUDA learner does not. The remaining strict mismatch therefore belongs to the CUDA learner path, not to dataset binning.

1. Introduce an `rdna2` device type with its own routing contract. Keep upstream `cpu`, `gpu`, and `cuda` behavior unchanged. `rdna2` should select an RDNA2 learner whose host-side semantics follow `GPUTreeLearner` / `SerialTreeLearner`: CPU objective and score lifecycle, CPU/serial split selection, canonical leaf bookkeeping and subtraction semantics.
2. Offload only histogram construction first. Reuse the canonical Dataset/BinMapper representation and build an RDNA2 HIP histogram engine from those bins, rather than inheriting CUDA-specific objective, best-split, score-updater, and partition semantics. This isolates the GPU optimization surface to the component that materially benefits from RX 6800 XT acceleration.
3. Build an OpenCL-style HIP H64 reference kernel and then a separate H128 specialization. The first goal is semantic correctness and a fair architectural control against the proven OpenCL histogram implementation, not peak speed.
4. Build an RDNA2 Feature SuperTile engine that reuses row indices, gradients, and Hessians across more features per workgroup. H64 is expected to fit roughly 16 features per workgroup with native wave32 decomposition and about 8 KB of base histogram LDS; H128 should choose 8 or 16 features based on measured LDS/VGPR occupancy.
5. Add wave32 equal-bin aggregation only after the SuperTile path is correct. Combine lanes targeting the same bin before LDS updates to reduce atomic pressure, while guarding against floating-point accumulation-order changes that alter splits.
6. Consider histogram / best-split fusion only after the RDNA2 histogram engine is correctness-stable and fast. Fusion would deliberately cross the new ownership boundary and therefore comes last.

The expected long-term component shape is conceptually:

`RDNA2TreeLearner(Serial semantics) -> RDNA2HistogramEngine -> PackedDataset, H64, H128, Root, IndexedLeaf, WaveHistogramPrimitive, Merge/Subtract`

Generic CUDA behavior remains available for `device_type=cuda`; legacy OpenCL remains `device_type=gpu`. The fork-specific `rdna2` path should not silently change either upstream contract. H64 and H128 kernels should be compile-time specialized rather than routed through a universal 256-bin implementation. Dataset ownership and future packing/cache work are defined separately in `dataset/rdna2-dataset-architecture`; canonical `BinMapper` semantics remain authoritative even when the physical RDNA2 layout diverges.
