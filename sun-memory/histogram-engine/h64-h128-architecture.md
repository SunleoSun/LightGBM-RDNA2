---
description: H64/H128 histogram-engine contract, OpenCL reference properties, and RDNA2-specific design constraints.
---

# H64/H128 histogram engine

The near-term production target is a correct specialized HIP H64 kernel for gfx1030, followed by H128. The workload is approximately 40k rows by 3000 features, so feature parallelism and reuse of row-state matter more than the old H256 tuning assumptions.

The OpenCL histogram64 implementation is the architectural reference. Relevant properties include packed four-feature loads, four independent LDS histogram banks, bank-aware LDS layout, feature rotation across lanes, gradient/Hessian staggering, temporal accumulation, software prefetch, root/no-index specialization, constant-Hessian specialization where applicable, and dynamic workgroup counts.

Before optimizing, preserve the mathematical contract: feature-to-bin mapping, row/index mapping, gradient/Hessian histogram layout, missing/default-bin semantics, durable histogram representation, and subtraction requirements. A new kernel must integrate with the existing histogram producer/consumer lifecycle rather than redefining those semantics.

For the SuperTile stage, H64 can plausibly map four native wave32s to 16 features per workgroup. `16 features * 64 bins * 2 (G/H) * 4 bytes` is about 8 KB LDS, allowing row index / gradient / Hessian state to be reused across substantially more features than the current feature4 decomposition. H128 should be independently specialized and selected using real occupancy measurements.

Root and indexed-leaf paths should remain distinct when that avoids unnecessary index gathers. Histogram subtraction remains important for deeper Stage-2 trees because many leaves are small and the smaller-child histogram is the durable work product used to derive the larger child.
