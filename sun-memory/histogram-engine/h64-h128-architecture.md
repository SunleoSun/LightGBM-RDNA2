---
description: Exact H64/H128 HIP histogram architecture, transfer pipeline, RX 6800 XT constraints, and immediate SuperTile work.
---

# H64/H128 histogram engine

The production target is RX 6800 XT / gfx1030 on roughly 40k rows by 3000 features. CPU LightGBM 4.7 is the correctness oracle. The device has native wave32 execution and the RDNA2 LDS/bank structure makes aligned/coalesced loads, bank-aware LDS placement, and occupancy/VGPR control important.

The RDNA2 packed-input layer stores exact canonical bins as tuple-major `uint32 feature4[group][row]` in persistent device memory. For 40k x 3000 this is about 114.44 MiB. No OpenCL rebucketing is applied; every byte remains the canonical LightGBM dense-bin ID.

The first production H64 HIP histogram path is implemented. It is enabled only when every feature group is a dense single-feature group and every group has at most 64 bins. One 256-thread workgroup owns each four-feature tuple. The kernel reads aligned packed `uint32` values, uses four LDS histogram banks with lane-dependent feature rotation, accumulates gradient/Hessian pairs in double-precision `hist_t`, reduces the banks, and writes directly into the canonical `GroupBinBoundary` histogram layout expected by `SerialTreeLearner`. Gradients and Hessians are copied to device once per tree; indexed smaller-leaf row IDs are copied as needed. Root and indexed smaller leaves use HIP, while cases that require constructing both children still fall back to Serial construction. Parent-minus-smaller subtraction and CPU split finding remain canonical Serial semantics.

The full smoke and Optuna-envelope suites remain bit-for-bit identical to CPU: prediction max diff `0` and exact tree structure. A 100-tree production-shaped H64 run measured about `31.804 ms/tree` for RDNA2 versus `49.832 ms/tree` for CPU, with identical AUC, predictions, and tree structure. Short smoke timings are dominated by one-time ROCm context/device-allocation and packed-dataset setup and are not representative of amortized training speed.

Double-precision LDS is currently a correctness requirement for the strict gate. A float-LDS variant kept the same six-tree H64 structure but produced max prediction difference about `1.48e-6`, exceeding the project's `1e-6` prediction tolerance, and it was slower in that probe; it was reverted.

Legacy OpenCL `histogram64` remains an architectural/performance reference, not a correctness oracle. Useful ideas include 256 threads, packed four-feature reads, banked LDS, feature rotation, temporal accumulation, prefetch, root specialization, and constant-Hessian handling. The current H64 path intentionally starts simpler while keeping exact CPU semantics.

The first H128 HIP reference is also implemented as a separate compile-time `NUM_BINS=128` instantiation of the same canonical Feature4 producer. It keeps 256 threads and four double-precision LDS banks, requiring about 32 KiB LDS per workgroup versus about 16 KiB for H64, and uses two output rounds because four features times 128 bins exceeds 256 threads. H128 smoke, regression, and the representative Optuna envelope remain prediction-diff `0` with exact CPU structure. A 100-tree production H128 run measured about `45.525 ms/tree` versus CPU `72.007 ms/tree`, with identical AUC, predictions, and structure.

The transfer path now uses one persistent HIP stream: gradients/Hessians and indexed row IDs are queued with async H2D, histogram memset and kernel work are ordered on the same stream, and D2H goes into reusable pinned staging followed by one stream synchronization and a host memcpy into the Serial-owned histogram. This removed per-histogram device-wide synchronization. H64 improved from the first exact reference around `31.8` to about `30.7 ms/tree` in a representative 100-tree run; H128 stayed approximately neutral around `46 ms/tree`. An H128 two-bank LDS trial was neutral and was reverted to four banks.

The immediate performance task is SuperTile row-state reuse. H64 should first test 16 features per 256-thread workgroup by consuming four existing Feature4 tuple bases, with four double LDS banks (64 KiB total histogram state). H128 should independently start at eight features for the same 64 KiB LDS footprint. This avoids a dataset repack and reduces repeated gradient/Hessian/index loads before more fragile wave32 or accumulation-order optimizations.
