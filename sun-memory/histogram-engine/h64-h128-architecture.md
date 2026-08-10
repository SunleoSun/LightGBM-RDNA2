---
description: H64/H128 histogram-engine contract, RX 6800 XT hardware facts, packed-input foundation, OpenCL reference properties, and RDNA2 design constraints.
---

# H64/H128 histogram engine

The near-term production target is a correct specialized HIP H64 kernel for RX 6800 XT / gfx1030, followed by H128. The workload is approximately 40k rows by 3000 features, so feature parallelism and reuse of row-state matter more than the old H256 tuning assumptions.

The RX 6800 XT target has 72 compute units, native wave32 execution, 128 KiB LDS per CU, 128 MiB Infinity Cache, and a 4 MiB L2. RDNA LDS is organized as 32 banks of 32 bits. Kernel design should therefore use wave32-aware work distribution, coalesced aligned global loads, bank-aware LDS placement, and explicit control of VGPR/LDS occupancy. Workgroups should remain multiples of 64; 256 threads is the initial H64 reference geometry because it gives eight native waves while matching the proven OpenCL implementation.

The OpenCL `histogram64` implementation remains the architectural reference. Relevant properties include packed four-feature loads, four independent LDS histogram banks, bank-aware LDS layout, feature rotation across lanes, gradient/Hessian staggering, temporal accumulation, software prefetch, root/no-index specialization, constant-Hessian specialization where applicable, and dynamic workgroup counts. It is not the correctness oracle: LightGBM 4.7 CPU is authoritative, and legacy OpenCL can make different late split choices at deep/high-leaf configurations.

The first RDNA2 packed-input layer now stores exact canonical bins as tuple-major `uint32 feature4[group][row]` in persistent device memory. For 40k x 3000 it is about 114.44 MiB. A representative probe measured roughly 25 ms host packing and 12 ms H2D after allocation; first-process ROCm device allocation/context initialization was roughly 456 ms. The H64 HIP producer should consume this aligned representation directly rather than reconstructing four-byte tuples in every tree.

Before optimizing, preserve the mathematical contract: feature-to-bin mapping, row/index mapping, gradient/Hessian histogram layout, missing/default-bin semantics, durable histogram representation, and subtraction requirements. A new kernel must integrate with the Serial learner's histogram producer/consumer boundary rather than importing CUDA best-split/objective semantics.

For the SuperTile stage, H64 can plausibly map four native wave32s to 16 features per workgroup. `16 features * 64 bins * 2 (G/H) * 4 bytes` is about 8 KiB of unbanked base histogram state before collision-reduction/padding, leaving substantial LDS headroom. H128 should be independently specialized and selected using measured LDS/VGPR occupancy.

Root and indexed-leaf paths should remain distinct when that avoids unnecessary index gathers. Histogram subtraction remains important for deeper Stage-2 trees because many leaves are small and the smaller-child histogram is the durable work product used to derive the larger child.
