---
description: LightGBM-RDNA2 project scope, target hardware, production workload, and correctness-first optimization objective.
---

# Project scope

LightGBM-RDNA2 is an experimental LightGBM fork for native Windows ROCm/HIP training on AMD RDNA2, initially targeting Radeon RX 6800 XT / Navi21 / gfx1030. The current base is LightGBM 4.7.0.

The production-shaped workload is very wide: about 40,000 training rows and 3,000 features, with approximately 50,000 validation rows. Stage 1 is primarily H64 (`max_bin=63`); Stage 2 is primarily H64/H128 (`max_bin=63/127`) with deeper trees and 40-120 leaves. H256 is historical and is not the primary optimization target.

The project rule is correctness first, performance second. ROCm performance numbers are not accepted when tree structure, prediction agreement, AUC/RMSE, hard labels, or requested tree count fail against the LightGBM 4.7 CPU reference.

The intended end state is a dedicated RDNA2 histogram engine rather than endless parameter tuning of the generic CUDA histogram path. See `general/planned-architecture`, `histogram-engine/h64-h128-architecture`, and `benchmarks/smoke-and-validation`.
