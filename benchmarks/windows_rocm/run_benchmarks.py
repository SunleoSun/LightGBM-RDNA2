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
BIN = HERE / "bin"
DATA = HERE / "data"
ARTIFACTS = HERE / "artifacts"
SEED = 20260809


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

def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {subprocess.list2cmdline(cmd)}")
    return p


def ensure_binary_dataset(old_dll: Path, train_text: Path, train_binary: Path) -> None:
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
        chk(lib.LGBM_DatasetCreateFromFile(str(train_text.resolve()).encode(), b"header=false label_column=0 max_bin=255 feature_pre_filter=false", None, C.byref(ds)), "DatasetCreateFromFile")
        chk(lib.LGBM_DatasetSaveBinary(ds, str(train_binary.resolve()).encode()), "DatasetSaveBinary")
    finally:
        if ds:
            lib.LGBM_DatasetFree(ds)


def run_capi_variant(name: str, dll: Path, device: str, train_file: Path, valid_npz: Path, iterations: int) -> dict:
    model = ARTIFACTS / f"{name}.model.txt"
    pred = ARTIFACTS / f"{name}.predictions.txt"
    result = ARTIFACTS / f"{name}.json"
    cmd = [
        sys.executable, str(HERE / "run_worker.py"), "--dll", str(dll), "--device", device,
        "--train-file", str(train_file), "--valid-npz", str(valid_npz),
        "--model-out", str(model), "--pred-out", str(pred), "--result-out", str(result),
        "--iterations", str(iterations),
    ]
    run_checked(cmd)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload.update({"name": name, "model": str(model), "predictions": str(pred)})
    return payload


def run_rocm_cli(name: str, exe: Path, train_file: Path, valid_features_file: Path, valid_npz: Path, iterations: int) -> dict:
    model = ARTIFACTS / f"{name}.model.txt"
    pred = ARTIFACTS / f"{name}.predictions.txt"
    result = ARTIFACTS / f"{name}.json"
    env = os.environ.copy()
    rocm_bin = Path(os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\6.2")) / "bin"
    env["PATH"] = str(rocm_bin) + os.pathsep + env.get("PATH", "")
    common = [
        "task=train", f"data={train_file}", "objective=binary", "metric=auc",
        "learning_rate=0.05", "num_leaves=4", "max_depth=8", "min_data_in_leaf=20", "max_bin=255",
        "feature_fraction=1.0", "bagging_fraction=1.0", "bagging_freq=0", "seed=20260809",
        "feature_fraction_seed=20260809", "bagging_seed=20260809", "data_random_seed=20260809",
        "deterministic=true", "force_col_wise=true", "device_type=cuda", "num_gpu=1", "gpu_device_id=0", "verbosity=1",
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
        "auc": auc_score(y_valid, preds), "prediction_min": float(preds.min()),
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
    results: list[dict], iterations: int, y_valid: np.ndarray, atol: float, rtol: float,
    auc_tol: float, correlation_min: float, threshold: float, min_class_fraction: float,
    min_confident_fraction: float,
) -> dict:
    reference = results[0]
    ref_pred = np.loadtxt(reference["predictions"], dtype=np.float64)
    ref_hard = (ref_pred >= threshold).astype(np.int8)
    ref_confusion = confusion_matrix(y_valid, ref_hard)
    ref_auc = auc_score(y_valid, ref_pred)
    ref_trees, ref_exact, ref_struct = tree_signature(Path(reference["model"]))
    if ref_trees != iterations:
        raise RuntimeError(f"reference model has {ref_trees} trees, expected {iterations}")
    comparisons = []
    failures = []
    for item in results:
        pred = np.loadtxt(item["predictions"], dtype=np.float64)
        if pred.shape != ref_pred.shape:
            raise RuntimeError(f"prediction shape mismatch for {item['name']}: {pred.shape} vs {ref_pred.shape}")
        finite = bool(np.isfinite(pred).all())
        probabilities = bool(np.all((pred >= 0.0) & (pred <= 1.0)))
        nonconstant = bool(np.std(pred) > 0.0)
        allclose = bool(np.allclose(pred, ref_pred, atol=atol, rtol=rtol))
        max_abs = float(np.max(np.abs(pred - ref_pred)))
        mean_abs = float(np.mean(np.abs(pred - ref_pred)))
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
        correlation = float(np.corrcoef(ref_pred, pred)[0, 1])
        trees, exact, structural = tree_signature(Path(item["model"]))
        record = {
            "name": item["name"],
            "finite_predictions": finite,
            "probabilities_in_unit_interval": probabilities,
            "nonconstant_predictions": nonconstant,
            "predictions_match_reference": allclose,
            "prediction_max_abs_diff": max_abs,
            "prediction_mean_abs_diff": mean_abs,
            "pearson_correlation": correlation,
            "auc": auc,
            "auc_abs_diff": auc_diff,
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
            "tree_count": trees,
            "tree_text_exact_match": exact == ref_exact,
            "tree_structure_match": structural == ref_struct,
        }
        comparisons.append(record)
        failed = (
            not finite or not probabilities or not nonconstant or trees != iterations
            or not allclose or structural != ref_struct or not hard_match or not confusion_match
            or auc_diff > auc_tol or correlation < correlation_min
            or smallest_class_fraction < min_class_fraction
            or confident_fraction < min_confident_fraction
        )
        if failed:
            failures.append(record)
    return {
        "reference": reference["name"], "atol": atol, "rtol": rtol,
        "auc_tol": auc_tol, "correlation_min": correlation_min,
        "classification_threshold": threshold, "min_class_fraction": min_class_fraction,
        "min_confident_fraction": min_confident_fraction, "reference_confusion_matrix": ref_confusion,
        "all_checks_passed": not failures, "comparisons": comparisons, "failures": failures,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-rows", type=int, default=40000)
    p.add_argument("--valid-rows", type=int, default=50000)
    p.add_argument("--features", type=int, default=3000)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--auc-tol", type=float, default=5e-8)
    p.add_argument("--correlation-min", type=float, default=0.99999999)
    p.add_argument("--classification-threshold", type=float, default=0.5)
    p.add_argument("--min-class-fraction", type=float, default=0.05)
    p.add_argument("--min-confident-fraction", type=float, default=0.05)
    args = p.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train_text, train_binary, valid_features_file, valid_npz = generate_dataset(args.train_rows, args.valid_rows, args.features)
    required = {
        "old": BIN / "lightgbm_old.dll",
        "cpu": BIN / "lightgbm_4.7.0_cpu.dll",
        "rocm_dll": BIN / "lightgbm_4.7.0_rocm.dll",
        "rocm_exe": BIN / "lightgbm_4.7.0_rocm.exe",
    }
    missing = [str(v) for v in required.values() if not v.exists()]
    if missing:
        raise RuntimeError("missing benchmark binaries; run build_and_benchmark.ps1 first:\n" + "\n".join(missing))

    ensure_binary_dataset(required["old"], train_text, train_binary)
    results = [
        run_capi_variant("old_cpu", required["old"], "cpu", train_binary, valid_npz, args.iterations),
        run_capi_variant("old_opencl_gpu", required["old"], "gpu", train_binary, valid_npz, args.iterations),
        run_capi_variant("v470_cpu", required["cpu"], "cpu", train_binary, valid_npz, args.iterations),
        run_rocm_cli("v470_rocm_gpu", required["rocm_exe"], train_binary, valid_features_file, valid_npz, args.iterations),
    ]
    y_valid = np.load(valid_npz)["y"]
    comparison = compare(
        results, args.iterations, y_valid, args.atol, args.rtol, args.auc_tol,
        args.correlation_min, args.classification_threshold, args.min_class_fraction,
        args.min_confident_fraction,
    )
    report = {
        "config": {
            "objective": "binary", "metric": "auc", "n_estimators": args.iterations,
            "max_depth": 8, "num_leaves": 4, "train_rows": args.train_rows,
            "valid_rows": args.valid_rows, "features": args.features, "seed": SEED,
        },
        "results": results, "comparison": comparison,
    }
    report_path = ARTIFACTS / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'mode':<20} {'train_s':>10} {'iter_ms':>10} {'auc':>10} {'pred_diff':>12} {'trees':>7} {'struct':>8}")
    cmp_by_name = {x["name"]: x for x in comparison["comparisons"]}
    for r in results:
        c = cmp_by_name[r["name"]]
        print(f"{r['name']:<20} {r['train_seconds']:10.4f} {r['iteration_ms']:10.3f} {r['auc']:10.6f} {c['prediction_max_abs_diff']:12.3g} {c['tree_count']:7d} {str(c['tree_structure_match']):>8}")
    print(f"\nArtifacts: {ARTIFACTS}")
    print(f"Correctness: {'PASS' if comparison['all_checks_passed'] else 'FAIL'}")
    return 0 if comparison["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
