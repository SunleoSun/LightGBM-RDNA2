---
description: Native Windows HIP toolchain, runtime limitations, and external build layout for gfx1030 LightGBM-RDNA2.
---

# Windows ROCm toolchain

The target environment is Windows 11, AMD Radeon RX 6800 XT / Navi21 / gfx1030 with native wave32, AMD ROCm/HIP SDK 6.2, Visual Studio 2022 with MSVC 14.44, CMake 3.31.6, and Ninja 1.12.1. Use the VS2022 environment. A newer VS18 Insiders toolchain was incompatible with the bundled HIP clang 19 path.

Use native Windows HIP rather than WSL. WSL could enumerate gfx1030 and allocate memory, but the execution/completion path stalled on the first host-to-device copy. Native Windows HIP smoke testing succeeded through device discovery, allocation, H2D, kernel launch, D2H, and result validation.

The reliable HIP compilation path is direct AMD `clang++.exe -x hip --offload-arch=gfx1030 --rocm-path=...` after `vcvars64`; `hipcc.bat` is inconvenient in this environment. Upstream ROCm CMake assumes Linux/RCCL in places, so this fork carries an experimental native Windows single-GPU path.

All build output must stay outside the repository under `C:\Temp\LightGBM-RDNA2`. Default build roots are `build\cpu` and `build\rocm`, with output directories beneath them. Benchmark binaries/data/artifacts belong under the sibling `benches` tree.

The HIP CLI is usable, but the HIP shared DLL is not production-ready for Python/SunTrader: `_lightgbm.dll` still access-violates during `LGBM_BoosterCreate` with `device_type=cuda`, likely in Windows HIP shared-DLL kernel registration/static initialization/linking. Treat the CLI as the ROCm benchmark execution path until that boundary is separately fixed.
