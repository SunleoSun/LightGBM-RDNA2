param(
    [string]$OldDll = 'C:\Drive\MyPrograms\SunTrader\Python\lightGBM\lightgbm.dll',
    [string]$RocmPath = 'C:\Program Files\AMD\ROCm\6.2',
    [int]$TrainRows = 40000,
    [int]$ValidRows = 50000,
    [int]$SmokeValidRows = 5000,
    [int]$Features = 3000,
    [int]$Iterations = 100,
    [ValidateSet('h64', 'h128', 'all')]
    [string]$Profile = 'all',
    [ValidateSet('production', 'smoke', 'stress', 'single')]
    [string]$Suite = 'production',
    [int]$MatrixIterations = 0,
    [double]$PredictionAtol = 1e-6,
    [double]$PredictionRtol = 1e-6,
    [string]$WorkRoot = 'C:\Temp\LightGBM-RDNA2'
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$BuildRoot = Join-Path $WorkRoot 'build'
$BenchRoot = Join-Path $WorkRoot 'benches'
$Bin = Join-Path $BenchRoot 'bin'
$CpuBuild = Join-Path $BuildRoot 'cpu'
$RocmBuild = Join-Path $BuildRoot 'rocm'
$VcVars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$Python = 'C:\Program Files\Python313\python.exe'

foreach ($path in @($OldDll, $RocmPath, $VcVars, $Python)) {
    if (-not (Test-Path $path)) { throw "Required path not found: $path" }
}
New-Item -ItemType Directory -Force -Path $Bin | Out-Null
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $BenchRoot | Out-Null
$env:LIGHTGBM_RDNA2_TEMP = $BenchRoot
$env:LIGHTGBM_RDNA2_BIN = $Bin
Copy-Item -Force $OldDll (Join-Path $Bin 'lightgbm_old.dll')

function Invoke-CmdChecked([string]$Command) {
    Write-Host "+ cmd /c $Command"
    cmd.exe /d /s /c $Command
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

Write-Host '=== Build LightGBM 4.7.0 CPU ==='
Remove-Item -Recurse -Force $CpuBuild -ErrorAction SilentlyContinue
$cpuOut = Join-Path $CpuBuild 'out'
$cpuCommand = "call `"$VcVars`" && cmake -S `"$Repo`" -B `"$CpuBuild`" -G Ninja -DUSE_ROCM=OFF -DUSE_GPU=OFF -DUSE_CUDA=OFF -DBUILD_CLI=ON -DBUILD_STATIC_LIB=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`"$cpuOut`" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`"$cpuOut`" -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`"$cpuOut`" && cmake --build `"$CpuBuild`" -j 8"
Invoke-CmdChecked $cpuCommand
$cpuDll = Join-Path $cpuOut 'lib_lightgbm.dll'
$cpuExe = Join-Path $cpuOut 'lightgbm.exe'
if (-not (Test-Path $cpuDll)) { throw "CPU build did not produce $cpuDll" }
Copy-Item -Force $cpuDll (Join-Path $Bin 'lightgbm_4.7.0_cpu.dll')
if (Test-Path $cpuExe) { Copy-Item -Force $cpuExe (Join-Path $Bin 'lightgbm_4.7.0_cpu.exe') }

Write-Host '=== Build LightGBM 4.7.0 native Windows ROCm/HIP gfx1030 ==='
Remove-Item -Recurse -Force $RocmBuild -ErrorAction SilentlyContinue
$clang = ($RocmPath -replace '\\','/') + '/bin/clang.exe'
$clangxx = ($RocmPath -replace '\\','/') + '/bin/clang++.exe'
$rocmCmake = $RocmPath -replace '\\','/'
$rocmOut = Join-Path $RocmBuild 'out'
$rocmCommand = "call `"$VcVars`" && set `"PATH=$RocmPath\bin;%PATH%`" && set `"HIP_PATH=$RocmPath`" && set `"HIP_PLATFORM=amd`" && cmake -S `"$Repo`" -B `"$RocmBuild`" -G Ninja -DUSE_ROCM=ON -DUSE_GPU=OFF -DUSE_CUDA=OFF -DBUILD_CLI=ON -DBUILD_STATIC_LIB=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=`"$clang`" -DCMAKE_CXX_COMPILER=`"$clangxx`" -DCMAKE_HIP_COMPILER=`"$clangxx`" -DCMAKE_HIP_COMPILER_ROCM_ROOT=`"$rocmCmake`" -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_PREFIX_PATH=`"$rocmCmake`" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=`"$rocmOut`" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=`"$rocmOut`" -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=`"$rocmOut`" && cmake --build `"$RocmBuild`" -j 8"
Invoke-CmdChecked $rocmCommand
$rocmDll = Join-Path $rocmOut '_lightgbm.dll'
$rocmExe = Join-Path $rocmOut 'lightgbm.exe'
if (-not (Test-Path $rocmDll)) { throw "ROCm build did not produce $rocmDll" }
if (-not (Test-Path $rocmExe)) { throw "ROCm build did not produce $rocmExe" }
Copy-Item -Force $rocmDll (Join-Path $Bin 'lightgbm_4.7.0_rocm.dll')
Copy-Item -Force $rocmExe (Join-Path $Bin 'lightgbm_4.7.0_rocm.exe')

Write-Host '=== Benchmark and correctness checks ==='
$env:ROCM_PATH = $RocmPath
if ($Suite -eq 'single') {
    $profiles = if ($Profile -eq 'all') { @('h64', 'h128') } else { @($Profile) }
    $benchmarkFailed = $false
    foreach ($benchmarkProfile in $profiles) {
        Write-Host "=== Profile $benchmarkProfile ==="
        & $Python (Join-Path $PSScriptRoot 'run_benchmarks.py') --profile $benchmarkProfile --train-rows $TrainRows --valid-rows $ValidRows --features $Features --iterations $Iterations --atol $PredictionAtol --rtol $PredictionRtol
        if ($LASTEXITCODE -ne 0) { $benchmarkFailed = $true }
    }
    if ($benchmarkFailed) { exit 2 }
    exit 0
}

$matrixValidRows = if ($Suite -eq 'smoke') { $SmokeValidRows } else { $ValidRows }
$matrixArgs = @((Join-Path $PSScriptRoot 'run_matrix.py'), '--suite', $Suite, '--train-rows', $TrainRows, '--valid-rows', $matrixValidRows, '--features', $Features)
if ($MatrixIterations -gt 0) { $matrixArgs += @('--iterations', $MatrixIterations) }
& $Python @matrixArgs
exit $LASTEXITCODE
