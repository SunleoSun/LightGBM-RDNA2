from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time

import numpy as np

C_API_DTYPE_FLOAT32 = 0
C_API_PREDICT_NORMAL = 0
C_API_FEATURE_IMPORTANCE_SPLIT = 0


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(y_score):
        j = i + 1
        while j < len(y_score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError("AUC requires both classes")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bind(lib: C.CDLL) -> None:
    handle = C.c_void_p
    lib.LGBM_GetLastError.restype = C.c_char_p
    lib.LGBM_DatasetCreateFromFile.argtypes = [C.c_char_p, C.c_char_p, handle, C.POINTER(handle)]
    lib.LGBM_DatasetFree.argtypes = [handle]
    lib.LGBM_BoosterCreate.argtypes = [handle, C.c_char_p, C.POINTER(handle)]
    lib.LGBM_BoosterUpdateOneIter.argtypes = [handle, C.POINTER(C.c_int)]
    lib.LGBM_BoosterSaveModel.argtypes = [handle, C.c_int, C.c_int, C.c_int, C.c_char_p]
    lib.LGBM_BoosterPredictForMat.argtypes = [
        handle, C.c_void_p, C.c_int, C.c_int32, C.c_int32, C.c_int, C.c_int, C.c_int, C.c_int,
        C.c_char_p, C.POINTER(C.c_int64), C.POINTER(C.c_double)
    ]
    lib.LGBM_BoosterFree.argtypes = [handle]


def check(lib: C.CDLL, rc: int, where: str) -> None:
    if rc != 0:
        msg = lib.LGBM_GetLastError()
        text = msg.decode(errors="replace") if msg else "unknown LightGBM error"
        raise RuntimeError(f"{where}: {text}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dll", required=True)
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--valid-npz", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--pred-out", required=True)
    p.add_argument("--result-out", required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--params-json", required=True)
    args = p.parse_args()
    profile = json.loads(args.params_json)

    payload = np.load(args.valid_npz)
    x_valid = np.ascontiguousarray(payload["X"], dtype=np.float32)
    y_valid = np.asarray(payload["y"], dtype=np.int8)

    lib = C.CDLL(str(Path(args.dll).resolve()))
    bind(lib)
    H = C.c_void_p
    dataset = H()
    booster = H()
    if Path(args.train_file).suffix.lower() == ".bin":
        dataset_params = f"max_bin={int(profile['max_bin'])} feature_pre_filter=false".encode()
    else:
        dataset_params = f"header=false label_column=0 max_bin={int(profile['max_bin'])} feature_pre_filter=false".encode()
    common = " ".join([
        "objective=binary", "metric=auc",
        f"learning_rate={profile['learning_rate']}", f"num_leaves={profile['num_leaves']}",
        f"max_depth={profile['max_depth']}", f"min_data_in_leaf={profile['min_data_in_leaf']}",
        f"max_bin={profile['max_bin']}", f"feature_fraction={profile['feature_fraction']}",
        f"bagging_fraction={profile['bagging_fraction']}", f"bagging_freq={profile['bagging_freq']}",
        f"lambda_l1={profile['lambda_l1']}", f"lambda_l2={profile['lambda_l2']}",
        f"min_gain_to_split={profile['min_gain_to_split']}", f"path_smooth={profile['path_smooth']}",
        "verbosity=-1", "seed=20260809", "feature_fraction_seed=20260809",
        "bagging_seed=20260809", "data_random_seed=20260809",
        "deterministic=true", "force_col_wise=true",
    ])
    if args.device == "cpu":
        device_params = "device_type=cpu"
    else:
        device_params = "device_type=gpu gpu_use_dp=false gpu_platform_id=-1 gpu_device_id=-1"
    params = f"{common} {device_params}".encode()

    try:
        t0 = time.perf_counter()
        check(lib, lib.LGBM_DatasetCreateFromFile(str(Path(args.train_file).resolve()).encode(), dataset_params, None, C.byref(dataset)), "DatasetCreateFromFile")
        dataset_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        check(lib, lib.LGBM_BoosterCreate(dataset, params, C.byref(booster)), "BoosterCreate")
        init_seconds = time.perf_counter() - t0

        finished = C.c_int()
        t0 = time.perf_counter()
        for i in range(args.iterations):
            check(lib, lib.LGBM_BoosterUpdateOneIter(booster, C.byref(finished)), f"BoosterUpdateOneIter[{i}]")
        train_seconds = time.perf_counter() - t0

        model_path = Path(args.model_out).resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        check(lib, lib.LGBM_BoosterSaveModel(booster, 0, args.iterations, C_API_FEATURE_IMPORTANCE_SPLIT, str(model_path).encode()), "BoosterSaveModel")

        out_len = C.c_int64()
        preds = np.empty(x_valid.shape[0], dtype=np.float64)
        t0 = time.perf_counter()
        check(
            lib,
            lib.LGBM_BoosterPredictForMat(
                booster, x_valid.ctypes.data, C_API_DTYPE_FLOAT32, x_valid.shape[0], x_valid.shape[1], 1,
                C_API_PREDICT_NORMAL, 0, args.iterations, b"", C.byref(out_len), preds.ctypes.data_as(C.POINTER(C.c_double))
            ),
            "BoosterPredictForMat",
        )
        predict_seconds = time.perf_counter() - t0
        if out_len.value != len(preds):
            raise RuntimeError(f"prediction length mismatch: {out_len.value} != {len(preds)}")
        np.savetxt(args.pred_out, preds, fmt="%.17g")
        result = {
            "backend": "capi",
            "device": args.device,
            "dataset_seconds": dataset_seconds,
            "init_seconds": init_seconds,
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "iteration_ms": train_seconds * 1000.0 / args.iterations,
            "auc": auc_score(y_valid, preds),
            "prediction_min": float(preds.min()),
            "prediction_max": float(preds.max()),
            "prediction_std": float(preds.std()),
        }
        Path(args.result_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0
    finally:
        if booster:
            lib.LGBM_BoosterFree(booster)
        if dataset:
            lib.LGBM_DatasetFree(dataset)


if __name__ == "__main__":
    raise SystemExit(main())
