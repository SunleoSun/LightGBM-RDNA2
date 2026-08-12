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


def array_layout(array: np.ndarray) -> dict:
    return {
        "dtype": str(array.dtype),
        "shape": [int(v) for v in array.shape],
        "strides": [int(v) for v in array.strides],
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "owns_data": bool(array.flags.owndata),
        "nbytes": int(array.nbytes),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dll", required=True)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--binary-out", required=True)
    p.add_argument("--result-out", required=True)
    p.add_argument("--max-bin", type=int, required=True)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--bin-construct-sample-cnt", type=int, default=200000)
    p.add_argument("--device-type", default="cpu")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--train-start", type=int, required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--split-extraction", choices=["pipeline_fancy", "contiguous_slice"], required=True)
    args = p.parse_args()

    wall0 = time.perf_counter()
    t0 = time.perf_counter()
    payload = np.load(args.input_npz)
    x_loaded = payload["X"]
    y_loaded = payload["y"]
    input_load_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    phase_x = np.ascontiguousarray(x_loaded, dtype=np.float32)
    phase_y = np.ascontiguousarray(y_loaded, dtype=np.float32)
    phase_input_conversion_seconds = time.perf_counter() - t0
    phase_layout = array_layout(phase_x)
    baseline_ws, baseline_peak = memory_counters()

    if not (0 <= args.train_start < args.train_end <= phase_x.shape[0]):
        raise ValueError("invalid train range")

    t0 = time.perf_counter()
    split_index_bytes = 0
    if args.split_extraction == "pipeline_fancy":
        train_idx = np.arange(args.train_start, args.train_end, dtype=np.int64)
        split_index_bytes = int(train_idx.nbytes)
        x_split = phase_x[train_idx]
        y_split = phase_y[train_idx]
    else:
        train_slice = slice(args.train_start, args.train_end)
        x_split = phase_x[train_slice]
        y_split = phase_y[train_slice]
    split_materialization_seconds = time.perf_counter() - t0
    post_split_ws, _ = memory_counters()
    split_layout = array_layout(x_split)
    split_shares_phase_memory = bool(np.shares_memory(x_split, phase_x))

    t0 = time.perf_counter()
    x = np.ascontiguousarray(x_split, dtype=np.float32)
    y = np.ascontiguousarray(y_split, dtype=np.float32)
    model_input_conversion_seconds = time.perf_counter() - t0
    model_layout = array_layout(x)
    model_shares_phase_memory = bool(np.shares_memory(x, phase_x))

    lib = C.CDLL(str(Path(args.dll).resolve()))
    bind(lib)
    H = C.c_void_p
    dataset = H()
    params = (
        f"max_bin={args.max_bin} feature_pre_filter=false num_threads={args.num_threads} "
        f"bin_construct_sample_cnt={args.bin_construct_sample_cnt} device_type={args.device_type} "
        f"gpu_device_id={args.gpu_device_id} verbosity=-1"
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
        "phase_input_conversion_seconds": phase_input_conversion_seconds,
        "split_materialization_seconds": split_materialization_seconds,
        "model_input_conversion_seconds": model_input_conversion_seconds,
        "dataset_create_seconds": dataset_create_seconds,
        "set_label_seconds": set_label_seconds,
        "save_binary_seconds": save_binary_seconds,
        "pipeline_seconds": split_materialization_seconds + model_input_conversion_seconds + dataset_create_seconds + set_label_seconds,
        "worker_wall_seconds": time.perf_counter() - wall0,
        "baseline_working_set_mib": baseline_ws / (1024.0 * 1024.0),
        "post_split_working_set_mib": post_split_ws / (1024.0 * 1024.0),
        "split_working_set_delta_mib": max(0, post_split_ws - baseline_ws) / (1024.0 * 1024.0),
        "working_set_mib": current_ws / (1024.0 * 1024.0),
        "peak_working_set_mib": peak_ws / (1024.0 * 1024.0),
        "peak_delta_mib": max(0, peak_ws - baseline_ws) / (1024.0 * 1024.0),
        "baseline_process_peak_mib": baseline_peak / (1024.0 * 1024.0),
        "phase_layout": phase_layout,
        "split_layout": split_layout,
        "model_layout": model_layout,
        "split_shares_phase_memory": split_shares_phase_memory,
        "model_shares_phase_memory": model_shares_phase_memory,
        "split_index_bytes": split_index_bytes,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "max_bin": args.max_bin,
        "num_threads": args.num_threads,
        "bin_construct_sample_cnt": args.bin_construct_sample_cnt,
        "device_type": args.device_type,
        "gpu_device_id": args.gpu_device_id,
        "split_extraction": args.split_extraction,
        "binary_bytes": binary_out.stat().st_size,
    }
    Path(args.result_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
