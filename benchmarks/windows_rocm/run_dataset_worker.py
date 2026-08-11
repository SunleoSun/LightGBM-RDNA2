from __future__ import annotations

import argparse
import ctypes as C
import json
from pathlib import Path
import time

import numpy as np

C_API_DTYPE_FLOAT32 = 0


class PROCESS_MEMORY_COUNTERS_EX(C.Structure):
    _fields_ = [
        ("cb", C.c_ulong),
        ("PageFaultCount", C.c_ulong),
        ("PeakWorkingSetSize", C.c_size_t),
        ("WorkingSetSize", C.c_size_t),
        ("QuotaPeakPagedPoolUsage", C.c_size_t),
        ("QuotaPagedPoolUsage", C.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", C.c_size_t),
        ("QuotaNonPagedPoolUsage", C.c_size_t),
        ("PagefileUsage", C.c_size_t),
        ("PeakPagefileUsage", C.c_size_t),
        ("PrivateUsage", C.c_size_t),
    ]


def memory_counters() -> tuple[int, int]:
    kernel32 = C.WinDLL("kernel32", use_last_error=True)
    psapi = C.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = C.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [C.c_void_p, C.POINTER(PROCESS_MEMORY_COUNTERS_EX), C.c_ulong]
    psapi.GetProcessMemoryInfo.restype = C.c_int
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = C.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), C.byref(counters), counters.cb)
    if not ok:
        raise OSError(C.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def bind(lib: C.CDLL) -> None:
    handle = C.c_void_p
    lib.LGBM_GetLastError.restype = C.c_char_p
    lib.LGBM_DatasetCreateFromMat.argtypes = [
        C.c_void_p, C.c_int, C.c_int32, C.c_int32, C.c_int, C.c_char_p, handle, C.POINTER(handle)
    ]
    lib.LGBM_DatasetSetField.argtypes = [handle, C.c_char_p, C.c_void_p, C.c_int, C.c_int]
    lib.LGBM_DatasetSaveBinary.argtypes = [handle, C.c_char_p]
    lib.LGBM_DatasetFree.argtypes = [handle]


def check(lib: C.CDLL, rc: int, where: str) -> None:
    if rc != 0:
        msg = lib.LGBM_GetLastError()
        text = msg.decode(errors="replace") if msg else "unknown LightGBM error"
        raise RuntimeError(f"{where}: {text}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dll", required=True)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--binary-out", required=True)
    p.add_argument("--result-out", required=True)
    p.add_argument("--max-bin", type=int, required=True)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--bin-construct-sample-cnt", type=int, default=200000)
    args = p.parse_args()

    wall0 = time.perf_counter()
    t0 = time.perf_counter()
    payload = np.load(args.input_npz)
    x_source = payload["X"]
    y_source = payload["y"]
    input_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = np.ascontiguousarray(x_source, dtype=np.float32)
    y = np.ascontiguousarray(y_source, dtype=np.float32)
    input_conversion_seconds = time.perf_counter() - t0
    baseline_ws, baseline_peak = memory_counters()

    lib = C.CDLL(str(Path(args.dll).resolve()))
    bind(lib)
    H = C.c_void_p
    dataset = H()
    params = (
        f"max_bin={args.max_bin} feature_pre_filter=false num_threads={args.num_threads} "
        f"bin_construct_sample_cnt={args.bin_construct_sample_cnt} verbosity=-1"
    ).encode()

    binary_out = Path(args.binary_out).resolve()
    binary_out.parent.mkdir(parents=True, exist_ok=True)
    if binary_out.exists():
        binary_out.unlink()

    try:
        t0 = time.perf_counter()
        check(
            lib,
            lib.LGBM_DatasetCreateFromMat(
                x.ctypes.data, C_API_DTYPE_FLOAT32, x.shape[0], x.shape[1], 1, params, None, C.byref(dataset)
            ),
            "DatasetCreateFromMat",
        )
        dataset_create_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        check(lib, lib.LGBM_DatasetSetField(dataset, b"label", y.ctypes.data, len(y), C_API_DTYPE_FLOAT32), "DatasetSetField(label)")
        set_label_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        check(lib, lib.LGBM_DatasetSaveBinary(dataset, str(binary_out).encode()), "DatasetSaveBinary")
        save_binary_seconds = time.perf_counter() - t0
        current_ws, peak_ws = memory_counters()
    finally:
        if dataset:
            lib.LGBM_DatasetFree(dataset)

    result = {
        "input_load_seconds": input_load_seconds,
        "input_conversion_seconds": input_conversion_seconds,
        "dataset_create_seconds": dataset_create_seconds,
        "set_label_seconds": set_label_seconds,
        "save_binary_seconds": save_binary_seconds,
        "pipeline_seconds": input_conversion_seconds + dataset_create_seconds + set_label_seconds,
        "worker_wall_seconds": time.perf_counter() - wall0,
        "baseline_working_set_mib": baseline_ws / (1024.0 * 1024.0),
        "working_set_mib": current_ws / (1024.0 * 1024.0),
        "peak_working_set_mib": peak_ws / (1024.0 * 1024.0),
        "peak_delta_mib": max(0, peak_ws - baseline_ws) / (1024.0 * 1024.0),
        "baseline_process_peak_mib": baseline_peak / (1024.0 * 1024.0),
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "max_bin": args.max_bin,
        "num_threads": args.num_threads,
        "bin_construct_sample_cnt": args.bin_construct_sample_cnt,
        "binary_bytes": binary_out.stat().st_size,
    }
    Path(args.result_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
