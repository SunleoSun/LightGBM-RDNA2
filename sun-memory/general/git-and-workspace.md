---
description: Git branch authority and source-only workspace rules for the LightGBM-RDNA2 repository.
---

# Git and workspace

The active repository is `C:\Drive\MyPrograms\LightGBM-RDNA2`. It lives in Google Drive and must remain source-only. Build products, generated datasets, models, predictions, CMake caches, object files, DLLs, executables, and other temporary artifacts must not be written into the checkout.

Generated work belongs under `C:\Temp\LightGBM-RDNA2`, with `build\cpu` and `build\rocm` for builds and `benches\bin`, `benches\data`, `benches\artifacts`, and `benches\legacy-pre-repo` for benchmark state and historical artifacts.

As of 2026-08-10, `main` is at merge commit `0352e482` (`Merge RDNA2 Windows ROCm development into main`) and contains `rocm-gfx1030-optimizations` commit `f59ea91a` as an ancestor. `main` tracks `origin/main`. Therefore new integrated work should start from `main`; a separate `dev` branch is not required merely to recover the already-merged optimization history. Feature branches may still be created from `main` for isolated architectural work.

`rocm-gfx1030-optimizations` remains useful as historical reference. `windows-rocm-4.7.0` represents the earlier Windows ROCm compatibility baseline.
