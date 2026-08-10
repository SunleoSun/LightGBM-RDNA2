# LightGBM-RDNA2

Experimental fork of [LightGBM](https://github.com/lightgbm-org/LightGBM) focused on high-performance native Windows ROCm/HIP training for AMD RDNA2 GPUs.

The initial target is **Radeon RX 6800 XT / Navi21 / gfx1030** on Windows 11 with AMD ROCm/HIP SDK 6.2. The current RDNA2 development baseline is derived from LightGBM 4.7.0 while `main` also retains the newer upstream commits already present in the fork.

> [!WARNING]
> This is an experimental performance project, not an official LightGBM distribution. The current H64/H128 ROCm histogram path does not yet pass the strict correctness matrix and should not be used as a production replacement. Correctness is the gate for every optimization.

## Project goal

The goal is to build a dedicated RDNA2 histogram engine for wide, moderate-row workloads where the generic CUDA/HIP histogram architecture leaves substantial performance on the table. The main workload is approximately 40,000 training rows by 3,000 features and primarily uses `max_bin=63` or `max_bin=127`.

The architecture roadmap is:

1. **OpenCL-style H64/H128 reference kernels** — reproduce the important properties of LightGBM's proven AMD OpenCL histogram design in HIP and restore correctness.
2. **RDNA2 Feature SuperTile + native wave32** — reuse row indices, gradients, and Hessians across wider feature tiles instead of repeatedly processing them per small feature group.
3. **Wave-level equal-bin aggregation** — reduce LDS atomic pressure by combining equal-bin updates inside a wave32 while preserving acceptable numerical behavior.
4. **Histogram / BestSplit fusion** — reduce global-memory traffic and synchronization once the histogram engine itself is correct and fast.

The generic LightGBM CPU/CUDA paths remain the fallback. RDNA2-specific kernels are intended to be explicit specializations rather than hidden changes to generic GPU behavior.

## Current status

Native Windows HIP execution on gfx1030 works. LightGBM can be built with the Windows ROCm compatibility changes and the CLI executes the HIP backend using `device_type=cuda`. The experimental HIP shared DLL still has a C-API booster-creation crash, so ROCm benchmark runs currently use the CLI.

A strict benchmark harness compares:

- legacy Windows LightGBM CPU;
- legacy OpenCL GPU;
- LightGBM 4.7 CPU;
- LightGBM 4.7 native Windows ROCm/HIP.

ROCm correctness is gated against **LightGBM 4.7 CPU**. The suite covers H64/H128 baselines, bagging/subsampling, feature sampling, regularization, class weighting, and `regression_l2`. The current gfx1030 feature4 path fails these production-shaped H64/H128 checks, including regression, which is why the next work is architectural correctness rather than further micro-tuning of that path.

## Benchmark suites

The benchmark harness is under `benchmarks/windows_rocm/`.

```powershell
# Fast development gate: 4 representative profiles, 6 trees, v4.7 CPU vs ROCm.
python .\benchmarks\windows_rocm\run_matrix.py --suite smoke

# Production timing: H64 + H128, 100 trees, all four backends.
python .\benchmarks\windows_rocm\run_matrix.py --suite production

# Full correctness matrix: 12 profiles, 20 trees, all four backends.
python .\benchmarks\windows_rocm\run_matrix.py --suite stress
```

See [`benchmarks/windows_rocm/README.md`](benchmarks/windows_rocm/README.md) for the exact profiles, correctness gates, and build wrapper.

## Source-only checkout and external build workspace

The working checkout is intentionally kept **source-only** because it may live in a synchronized Google Drive directory. Generated CMake files, compiler objects, binaries, benchmark datasets, predictions, models, and logs must stay outside the repository.

The default workspace is:

```text
C:\Temp\LightGBM-RDNA2\
├── build\
│   ├── cpu\
│   └── rocm\
└── benches\
    ├── bin\
    ├── data\
    └── artifacts\
```

The build wrapper enforces this layout by default:

```powershell
.\benchmarks\windows_rocm\build_and_benchmark.ps1 -Suite smoke
```

Use `-WorkRoot` to select another external location. Standalone Python benchmark commands use `C:\Temp\LightGBM-RDNA2\benches` by default and can be redirected with `LIGHTGBM_RDNA2_TEMP`; binary lookup can be redirected with `LIGHTGBM_RDNA2_BIN`.

## Toolchain currently used

- Windows 11
- AMD Radeon RX 6800 XT (`gfx1030`)
- AMD ROCm/HIP SDK 6.2
- Visual Studio 2022 / MSVC host environment
- AMD Clang for native HIP device compilation
- CMake + Ninja

The Windows ROCm configuration is currently single-GPU; Windows ROCm SDK does not provide the RCCL setup expected by upstream multi-GPU ROCm builds.

## Upstream

This project is a fork of the official [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM) repository. Upstream documentation, APIs, supported platforms, contribution guidance, and general LightGBM usage remain authoritative for functionality not explicitly changed here.

The intended remote layout is:

```text
origin   -> this LightGBM-RDNA2 fork
upstream -> https://github.com/lightgbm-org/LightGBM.git
```

## License

LightGBM-RDNA2 retains the upstream LightGBM MIT license. See [`LICENSE`](LICENSE).
