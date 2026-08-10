from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMP_ROOT = Path(os.environ.get("LIGHTGBM_RDNA2_TEMP", r"C:\Temp\LightGBM-RDNA2\benches"))
BIN = Path(os.environ.get("LIGHTGBM_RDNA2_BIN", str(TEMP_ROOT / "bin")))
DATA = TEMP_ROOT / "data"
ARTIFACTS = TEMP_ROOT / "artifacts"
SEED = 20260809

H64_BASE = {
    "description": "Stage 1-like H64 workload",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.075,
    "num_leaves": 40,
    "max_depth": 6,
    "min_data_in_leaf": 50,
    "max_bin": 63,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "min_gain_to_split": 0.0,
    "path_smooth": 0.0,
    "scale_pos_weight": 1.0,
}

H128_BASE = {
    "description": "Representative Stage 2 H128 workload",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 40,
    "max_depth": 6,
    "min_data_in_leaf": 50,
    "max_bin": 127,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "min_gain_to_split": 0.0,
    "path_smooth": 0.0,
    "scale_pos_weight": 1.0,
}

PROFILE_CONFIGS = {
    "h64": dict(H64_BASE),
    "h128": dict(H128_BASE),
    "h64_subsample1": {**H64_BASE, "description": "H64 without bagging", "bagging_fraction": 1.0},
    "h64_feature50": {**H64_BASE, "description": "H64 with 50% feature sampling", "feature_fraction": 0.5},
    "h64_scale16": {**H64_BASE, "description": "H64 with strong positive-class weighting", "scale_pos_weight": 16.0},
    "h64_strong_reg": {
        **H64_BASE, "description": "H64 strong regularization", "min_data_in_leaf": 100,
        "lambda_l1": 1.0, "lambda_l2": 1.0, "min_gain_to_split": 0.005,
        "path_smooth": 1.0, "bagging_fraction": 1.0,
    },
    "h64_regression": {
        **H64_BASE, "description": "H64 L2 regression correctness probe", "objective": "regression_l2",
        "metric": "l2", "scale_pos_weight": 1.0, "feature_fraction": 1.0, "bagging_fraction": 1.0,
    },
    "h128_subsample1": {**H128_BASE, "description": "H128 without bagging", "bagging_fraction": 1.0},
    "h128_feature50": {**H128_BASE, "description": "H128 with 50% feature sampling", "feature_fraction": 0.5},
    "h128_scale16": {**H128_BASE, "description": "H128 with strong positive-class weighting", "scale_pos_weight": 16.0},
    "h128_strong_reg": {
        **H128_BASE, "description": "H128 strong regularization", "min_data_in_leaf": 100,
        "lambda_l1": 1.0, "lambda_l2": 1.0, "min_gain_to_split": 0.005,
        "path_smooth": 1.0, "bagging_fraction": 1.0,
    },
    "h128_regression": {
        **H128_BASE, "description": "H128 L2 regression correctness probe", "objective": "regression_l2",
        "metric": "l2", "scale_pos_weight": 1.0, "feature_fraction": 1.0, "bagging_fraction": 1.0,
    },
}

SMOKE_PROFILES = ["h64", "h128", "h64_scale16", "h128_regression"]
STRESS_PROFILES = list(PROFILE_CONFIGS)


def profile_param_string(profile: dict) -> str:
    parts = [
        f"objective={profile['objective']}",
        f"metric={profile['metric']}",
        f"learning_rate={profile['learning_rate']}",
        f"num_leaves={profile['num_leaves']}",
        f"max_depth={profile['max_depth']}",
        f"min_data_in_leaf={profile['min_data_in_leaf']}",
        f"max_bin={profile['max_bin']}",
        f"feature_fraction={profile['feature_fraction']}",
        f"bagging_fraction={profile['bagging_fraction']}",
        f"bagging_freq={profile['bagging_freq']}",
        f"lambda_l1={profile['lambda_l1']}",
        f"lambda_l2={profile['lambda_l2']}",
        f"min_gain_to_split={profile['min_gain_to_split']}",
        f"path_smooth={profile['path_smooth']}",
        "verbosity=-1",
        f"seed={SEED}",
        f"feature_fraction_seed={SEED}",
        f"bagging_seed={SEED}",
        f"data_random_seed={SEED}",
        "deterministic=true",
        "force_col_wise=true",
    ]
    if profile["objective"] == "binary":
        parts.append(f"scale_pos_weight={profile['scale_pos_weight']}")
    return " ".join(parts)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


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
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def generate_dataset(train_rows: int, valid_rows: int, features: int) -> tuple[Path, Path, Path, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    train_file = DATA / f"train_{train_rows}x{features}.tsv"
    train_binary = DATA / f"train_{train_rows}x{features}.bin"
    valid_features_file = DATA / f"valid_features_{valid_rows}x{features}.tsv"
    valid_npz = DATA / f"valid_{valid_rows}x{features}.npz"
    if train_file.exists() and valid_features_file.exists() and valid_npz.exists():
        return train_file, train_binary, valid_features_file, valid_npz

    rng = np.random.default_rng(SEED)
    total = train_rows + valid_rows
    x = rng.standard_normal((total, features), dtype=np.float32)
    # A deliberately strong but nonlinear signal keeps split choices stable across CPU/OpenCL/HIP.
    score = (
        2.8 * x[:, 0] - 2.1 * x[:, 1] + 1.6 * x[:, 2] * x[:, 3]
        + 1.2 * (x[:, 4] > 0.25).astype(np.float32) - 0.8 * (x[:, 5] < -0.5).astype(np.float32)
        + 0.15 * rng.standard_normal(total).astype(np.float32)
    )
    y = (score > 0.0).astype(np.int8)
    x_train = x[:train_rows]
    y_train = y[:train_rows]
    x_valid = x[train_rows:]
    y_valid = y[train_rows:]

    with train_file.open("w", encoding="utf-8", newline="\n") as f:
        for label, row in zip(y_train, x_train):
            f.write(str(int(label)))
            f.write("\t")
            f.write("\t".join(format(float(v), ".8g") for v in row))
            f.write("\n")
    np.savetxt(valid_features_file, x_valid, delimiter="\t", fmt="%.8g")
    np.savez_compressed(valid_npz, X=x_valid, y=y_valid)
    return train_file, train_binary, valid_features_file, valid_npz


def generate_regression_dataset(train_rows: int, valid_rows: int, features: int) -> tuple[Path, Path, Path, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    train_file = DATA / f"train_regression_{train_rows}x{features}.tsv"
    train_binary = DATA / f"train_regression_{train_rows}x{features}.bin"
    valid_features_file = DATA / f"valid_regression_features_{valid_rows}x{features}.tsv"
    valid_npz = DATA / f"valid_regression_{valid_rows}x{features}.npz"
    if train_file.exists() and valid_features_file.exists() and valid_npz.exists():
        return train_file, train_binary, valid_features_file, valid_npz

    rng = np.random.default_rng(SEED)
    total = train_rows + valid_rows
    x = rng.standard_normal((total, features), dtype=np.float32)
    target = (
        2.8 * x[:, 0] - 2.1 * x[:, 1] + 1.6 * x[:, 2] * x[:, 3]
        + 1.2 * np.tanh(x[:, 4]) - 0.8 * (x[:, 5] < -0.5).astype(np.float32)
        + 0.15 * rng.standard_normal(total).astype(np.float32)
    ).astype(np.float32)
    x_train = x[:train_rows]
    y_train = target[:train_rows]
    x_valid = x[train_rows:]
    y_valid = target[train_rows:]

    with train_file.open("w", encoding="utf-8", newline="\n") as f:
        for label, row in zip(y_train, x_train):
            f.write(format(float(label), ".8g"))
            f.write("\t")
            f.write("\t".join(format(float(v), ".8g") for v in row))
            f.write("\n")
    np.savetxt(valid_features_file, x_valid, delimiter="\t", fmt="%.8g")
    np.savez_compressed(valid_npz, X=x_valid, y=y_valid)
    return train_file, train_binary, valid_features_file, valid_npz


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {subprocess.list2cmdline(cmd)}")
    return p


def ensure_binary_dataset(old_dll: Path, train_text: Path, train_binary: Path, max_bin: int) -> None:
    if train_binary.exists():
        return
    lib = __import__("ctypes").CDLL(str(old_dll.resolve()))
    C = __import__("ctypes")
    H = C.c_void_p
    lib.LGBM_GetLastError.restype = C.c_char_p
    lib.LGBM_DatasetCreateFromFile.argtypes = [C.c_char_p, C.c_char_p, H, C.POINTER(H)]
    lib.LGBM_DatasetSaveBinary.argtypes = [H, C.c_char_p]
    lib.LGBM_DatasetFree.argtypes = [H]
    ds = H()
    def chk(rc: int, where: str) -> None:
        if rc != 0:
            msg = lib.LGBM_GetLastError()
            raise RuntimeError(f"{where}: {msg.decode(errors='replace') if msg else 'unknown error'}")
    try:
        dataset_params = f"header=false label_column=0 max_bin={max_bin} feature_pre_filter=false".encode()
        chk(lib.LGBM_DatasetCreateFromFile(str(train_text.resolve()).encode(), dataset_params, None, C.byref(ds)), "DatasetCreateFromFile")
        chk(lib.LGBM_DatasetSaveBinary(ds, str(train_binary.resolve()).encode()), "DatasetSaveBinary")
    finally:
        if ds:
            lib.LGBM_DatasetFree(ds)


def run_capi_variant(name: str, dll: Path, device: str, train_file: Path, valid_npz: Path, iterations: int, profile: dict) -> dict:
    model = ARTIFACTS / f"{name}.model.txt"
    pred = ARTIFACTS / f"{name}.predictions.txt"
    result = ARTIFACTS / f"{name}.json"
    cmd = [
        sys.executable, str(HERE / "run_worker.py"), "--dll", str(dll), "--device", device,
        "--train-file", str(train_file), "--valid-npz", str(valid_npz),
        "--model-out", str(model), "--pred-out", str(pred), "--result-out", str(result),
        "--iterations", str(iterations), "--params-json", json.dumps(profile),
    ]
    run_checked(cmd)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload.update({"name": name, "model": str(model), "predictions": str(pred)})
    return payload


def run_rocm_cli(name: str, exe: Path, train_file: Path, valid_features_file: Path, valid_npz: Path, iterations: int, profile: dict) -> dict:
    model = ARTIFACTS / f"{name}.model.txt"
    pred = ARTIFACTS / f"{name}.predictions.txt"
    result = ARTIFACTS / f"{name}.json"
    env = os.environ.copy()
    rocm_bin = Path(os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\6.2")) / "bin"
    env["PATH"] = str(rocm_bin) + os.pathsep + env.get("PATH", "")
    common = [
        "task=train", f"data={train_file}", *profile_param_string(profile).split(),
        "verbosity=1", "device_type=cuda", "num_gpu=1", "gpu_device_id=0",
        f"num_iterations={iterations}", f"output_model={model}",
    ]
    wall0 = time.perf_counter()
    proc = run_checked([str(exe), *common], env=env)
    wall_seconds = time.perf_counter() - wall0
    match = re.findall(r"([0-9.]+) seconds elapsed, finished iteration (\d+)", proc.stdout)
    train_seconds = wall_seconds
    if match:
        final_time, final_iter = match[-1]
        if int(final_iter) == iterations:
            train_seconds = float(final_time)

    predict_cmd = [
        str(exe), "task=predict", f"data={valid_features_file}", f"input_model={model}",
        f"output_result={pred}", "header=false", "verbosity=-1",
    ]
    t0 = time.perf_counter()
    run_checked(predict_cmd, env=env)
    predict_seconds = time.perf_counter() - t0
    preds = np.loadtxt(pred, dtype=np.float64)
    y_valid = np.load(valid_npz)["y"]
    payload = {
        "name": name, "backend": "cli", "device": "cuda", "dataset_seconds": None,
        "init_seconds": None, "train_seconds": train_seconds, "train_wall_seconds": wall_seconds,
        "predict_seconds": predict_seconds, "iteration_ms": train_seconds * 1000.0 / iterations,
        "objective": profile["objective"],
        "auc": auc_score(y_valid, preds) if profile["objective"] == "binary" else None,
        "rmse": rmse_score(y_valid, preds) if profile["objective"] != "binary" else None,
        "prediction_min": float(preds.min()),
        "prediction_max": float(preds.max()), "prediction_std": float(preds.std()),
        "model": str(model), "predictions": str(pred),
    }
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def tree_signature(path: Path) -> tuple[int, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Tree=")]
    if not starts:
        raise RuntimeError(f"no trees found in {path}")
    end = next((i for i in range(starts[0], len(lines)) if lines[i].strip() == "end of trees"), len(lines))
    tree_text = "\n".join(lines[starts[0]:end]) + "\n"
    exact = hashlib.sha256(tree_text.encode()).hexdigest()
    # A relaxed structural signature ignores threshold/gain values but keeps topology, split features and leaf counts.
    structural_lines = []
    keep_prefixes = ("Tree=", "num_leaves=", "split_feature=", "decision_type=", "left_child=", "right_child=")
    for line in lines[starts[0]:end]:
        if line.startswith(keep_prefixes):
            structural_lines.append(line)
    structural = hashlib.sha256(("\n".join(structural_lines) + "\n").encode()).hexdigest()
    return len(starts), exact, structural


def confusion_matrix(y_true: np.ndarray, hard_pred: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(hard_pred, dtype=np.int8)
    return {
        "tn": int(np.sum((y == 0) & (p == 0))),
        "fp": int(np.sum((y == 0) & (p == 1))),
        "fn": int(np.sum((y == 1) & (p == 0))),
        "tp": int(np.sum((y == 1) & (p == 1))),
    }


def compare(
    results: list[dict], iterations: int, y_valid: np.ndarray, objective: str, atol: float, rtol: float,
    auc_tol: float, regression_metric_tol: float, correlation_min: float, threshold: float,
    min_class_fraction: float, min_confident_fraction: float,
) -> dict:
    reference = next(item for item in results if item["name"] == "v470_cpu")
    gated_names = {"v470_rocm_gpu"}
    ref_pred = np.loadtxt(reference["predictions"], dtype=np.float64)
    ref_trees, ref_exact, ref_struct = tree_signature(Path(reference["model"]))
    if ref_trees != iterations:
        raise RuntimeError(f"reference model has {ref_trees} trees, expected {iterations}")

    binary = objective == "binary"
    ref_auc = auc_score(y_valid, ref_pred) if binary else None
    ref_rmse = rmse_score(y_valid, ref_pred) if not binary else None
    ref_hard = (ref_pred >= threshold).astype(np.int8) if binary else None
    ref_confusion = confusion_matrix(y_valid, ref_hard) if binary else None

    comparisons = []
    failures = []
    for item in results:
        pred = np.loadtxt(item["predictions"], dtype=np.float64)
        if pred.shape != ref_pred.shape:
            raise RuntimeError(f"prediction shape mismatch for {item['name']}: {pred.shape} vs {ref_pred.shape}")
        finite = bool(np.isfinite(pred).all())
        nonconstant = bool(np.std(pred) > 0.0)
        allclose = bool(np.allclose(pred, ref_pred, atol=atol, rtol=rtol))
        max_abs = float(np.max(np.abs(pred - ref_pred)))
        mean_abs = float(np.mean(np.abs(pred - ref_pred)))
        correlation = float(np.corrcoef(ref_pred, pred)[0, 1])
        trees, exact, structural = tree_signature(Path(item["model"]))

        record = {
            "name": item["name"],
            "gated_against_v470_cpu": item["name"] in gated_names,
            "finite_predictions": finite,
            "nonconstant_predictions": nonconstant,
            "predictions_match_reference": allclose,
            "prediction_max_abs_diff": max_abs,
            "prediction_mean_abs_diff": mean_abs,
            "pearson_correlation": correlation,
            "tree_count": trees,
            "tree_text_exact_match": exact == ref_exact,
            "tree_structure_match": structural == ref_struct,
        }

        if binary:
            probabilities = bool(np.all((pred >= 0.0) & (pred <= 1.0)))
            hard = (pred >= threshold).astype(np.int8)
            hard_match = bool(np.array_equal(hard, ref_hard))
            predicted_positive = int(hard.sum())
            predicted_negative = int(len(hard) - predicted_positive)
            smallest_class_fraction = min(predicted_positive, predicted_negative) / len(hard)
            confident_low = int(np.sum(pred < 0.1))
            confident_high = int(np.sum(pred > 0.9))
            confident_fraction = min(confident_low, confident_high) / len(pred)
            confusion = confusion_matrix(y_valid, hard)
            confusion_match = confusion == ref_confusion
            auc = auc_score(y_valid, pred)
            auc_diff = abs(auc - ref_auc)
            record.update({
                "probabilities_in_unit_interval": probabilities,
                "auc": auc, "auc_abs_diff": auc_diff,
                "classification_threshold": threshold,
                "hard_labels_match_reference": hard_match,
                "predicted_positive": predicted_positive,
                "predicted_negative": predicted_negative,
                "smallest_predicted_class_fraction": smallest_class_fraction,
                "confident_below_0_1": confident_low,
                "confident_above_0_9": confident_high,
                "smallest_confident_tail_fraction": confident_fraction,
                "confusion_matrix": confusion,
                "confusion_matrix_matches_reference": confusion_match,
            })
            failed = (
                not finite or not probabilities or not nonconstant or trees != iterations
                or not allclose or structural != ref_struct or not hard_match or not confusion_match
                or auc_diff > auc_tol or correlation < correlation_min
                or smallest_class_fraction < min_class_fraction
                or confident_fraction < min_confident_fraction
            )
        else:
            rmse = rmse_score(y_valid, pred)
            rmse_diff = abs(rmse - ref_rmse)
            record.update({"rmse": rmse, "rmse_abs_diff": rmse_diff})
            failed = (
                not finite or not nonconstant or trees != iterations or not allclose
                or structural != ref_struct or correlation < correlation_min
                or rmse_diff > regression_metric_tol
            )

        comparisons.append(record)
        if item["name"] in gated_names and failed:
            failures.append(record)

    return {
        "reference": reference["name"], "gated_names": sorted(gated_names),
        "objective": objective, "atol": atol, "rtol": rtol,
        "auc_tol": auc_tol, "regression_metric_tol": regression_metric_tol,
        "correlation_min": correlation_min, "classification_threshold": threshold,
        "min_class_fraction": min_class_fraction, "min_confident_fraction": min_confident_fraction,
        "reference_confusion_matrix": ref_confusion, "reference_auc": ref_auc, "reference_rmse": ref_rmse,
        "all_checks_passed": not failures, "comparisons": comparisons, "failures": failures,
    }


def main() -> int:
    global ARTIFACTS
    p = argparse.ArgumentParser()
    p.add_argument("--train-rows", type=int, default=40000)
    p.add_argument("--valid-rows", type=int, default=50000)
    p.add_argument("--features", type=int, default=3000)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="h64")
    p.add_argument("--modes", choices=["all", "v470"], default="all")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--auc-tol", type=float, default=5e-8)
    p.add_argument("--regression-metric-tol", type=float, default=1e-8)
    p.add_argument("--correlation-min", type=float, default=0.99999999)
    p.add_argument("--classification-threshold", type=float, default=0.5)
    p.add_argument("--min-class-fraction", type=float, default=0.05)
    p.add_argument("--min-confident-fraction", type=float, default=0.05)
    args = p.parse_args()

    profile = PROFILE_CONFIGS[args.profile]
    ARTIFACTS = TEMP_ROOT / "artifacts" / args.profile
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dataset_factory = generate_dataset if profile["objective"] == "binary" else generate_regression_dataset
    train_text, _legacy_train_binary, valid_features_file, valid_npz = dataset_factory(
        args.train_rows, args.valid_rows, args.features
    )
    objective_tag = "binary" if profile["objective"] == "binary" else "regression"
    train_binary = DATA / f"train_{objective_tag}_{args.train_rows}x{args.features}_maxbin{profile['max_bin']}.bin"
    required = {
        "cpu": BIN / "lightgbm_4.7.0_cpu.dll",
        "rocm_exe": BIN / "lightgbm_4.7.0_rocm.exe",
    }
    if args.modes == "all":
        required["old"] = BIN / "lightgbm_old.dll"
    missing = [str(v) for v in required.values() if not v.exists()]
    if missing:
        raise RuntimeError("missing benchmark binaries; run build_and_benchmark.ps1 first:\n" + "\n".join(missing))

    # v4.7 CPU is the canonical dataset producer and correctness reference.
    ensure_binary_dataset(required["cpu"], train_text, train_binary, int(profile["max_bin"]))
    results = []
    if args.modes == "all":
        results.extend([
            run_capi_variant("old_cpu", required["old"], "cpu", train_binary, valid_npz, args.iterations, profile),
            run_capi_variant("old_opencl_gpu", required["old"], "gpu", train_binary, valid_npz, args.iterations, profile),
        ])
    results.extend([
        run_capi_variant("v470_cpu", required["cpu"], "cpu", train_binary, valid_npz, args.iterations, profile),
        run_rocm_cli("v470_rocm_gpu", required["rocm_exe"], train_binary, valid_features_file, valid_npz, args.iterations, profile),
    ])
    y_valid = np.load(valid_npz)["y"]
    comparison = compare(
        results, args.iterations, y_valid, profile["objective"], args.atol, args.rtol, args.auc_tol,
        args.regression_metric_tol, args.correlation_min, args.classification_threshold,
        args.min_class_fraction, args.min_confident_fraction,
    )
    report = {
        "profile": args.profile,
        "config": {
            **profile,
            "n_estimators": args.iterations, "train_rows": args.train_rows,
            "valid_rows": args.valid_rows, "features": args.features, "seed": SEED,
        },
        "modes": args.modes, "temp_root": str(TEMP_ROOT),
        "results": results, "comparison": comparison,
    }
    report_path = ARTIFACTS / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== BENCHMARK SUMMARY [{args.profile}: max_bin={profile['max_bin']}, objective={profile['objective']}] ===")
    metric_name = "auc" if profile["objective"] == "binary" else "rmse"
    print(f"{'mode':<20} {'train_s':>10} {'iter_ms':>10} {metric_name:>10} {'pred_diff':>12} {'trees':>7} {'struct':>8}")
    cmp_by_name = {x["name"]: x for x in comparison["comparisons"]}
    for r in results:
        c = cmp_by_name[r["name"]]
        metric_value = r[metric_name]
        print(f"{r['name']:<20} {r['train_seconds']:10.4f} {r['iteration_ms']:10.3f} {metric_value:10.6f} {c['prediction_max_abs_diff']:12.3g} {c['tree_count']:7d} {str(c['tree_structure_match']):>8}")
    print(f"\nArtifacts: {ARTIFACTS}")
    print(f"Correctness: {'PASS' if comparison['all_checks_passed'] else 'FAIL'}")
    return 0 if comparison["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
