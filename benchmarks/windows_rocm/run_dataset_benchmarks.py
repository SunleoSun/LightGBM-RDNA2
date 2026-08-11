from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from run_benchmarks import (
    NATIVE_470_COMMIT,
    PROFILE_CONFIGS,
    SEED,
    compare,
    profile_param_string,
    run_capi_variant,
    run_checked,
    run_rocm_cli,
)

HERE = Path(__file__).resolve().parent
TEMP_ROOT = Path(os.environ.get("LIGHTGBM_RDNA2_TEMP", r"C:\Temp\LightGBM-RDNA2\benches"))
BIN = Path(os.environ.get("LIGHTGBM_RDNA2_BIN", str(TEMP_ROOT / "bin")))
DATA = TEMP_ROOT / "data"
ARTIFACTS = TEMP_ROOT / "artifacts" / "dataset_pipeline"
DEFAULT_ROCM_DLL = Path(os.environ.get("LIGHTGBM_RDNA2_ROCM_DLL", r"C:\Temp\LightGBM-RDNA2\build\rocm\out\_lightgbm.dll"))


def ensure_split_npz(train_rows: int, valid_rows: int, features: int, split_offset: int, objective: str) -> tuple[Path, Path]:
    DATA.mkdir(parents=True, exist_ok=True)
    tag = "binary" if objective == "binary" else "regression"
    stem = f"dataset_pipeline_{tag}_{train_rows}x{features}_valid{valid_rows}_offset{split_offset}"
    train_npz = DATA / f"{stem}_train.npz"
    valid_npz = DATA / f"{stem}_valid.npz"
    if train_npz.exists() and valid_npz.exists():
        return train_npz, valid_npz

    total = split_offset + train_rows + valid_rows
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((total, features), dtype=np.float32)
    if objective == "binary":
        score = (
            2.8 * x[:, 0] - 2.1 * x[:, 1] + 1.6 * x[:, 2] * x[:, 3]
            + 1.2 * (x[:, 4] > 0.25).astype(np.float32) - 0.8 * (x[:, 5] < -0.5).astype(np.float32)
            + 0.15 * rng.standard_normal(total).astype(np.float32)
        )
        y = (score > 0.0).astype(np.int8)
    else:
        y = (
            2.8 * x[:, 0] - 2.1 * x[:, 1] + 1.6 * x[:, 2] * x[:, 3]
            + 1.2 * np.tanh(x[:, 4]) - 0.8 * (x[:, 5] < -0.5).astype(np.float32)
            + 0.15 * rng.standard_normal(total).astype(np.float32)
        ).astype(np.float32)
    train_slice = slice(split_offset, split_offset + train_rows)
    valid_slice = slice(split_offset + train_rows, total)
    np.savez(train_npz, X=np.ascontiguousarray(x[train_slice]), y=np.ascontiguousarray(y[train_slice]))
    np.savez(valid_npz, X=np.ascontiguousarray(x[valid_slice]), y=np.ascontiguousarray(y[valid_slice]))
    return train_npz, valid_npz


def ensure_valid_text(valid_npz: Path) -> Path:
    valid_text = valid_npz.with_suffix(".features.tsv")
    if valid_text.exists():
        return valid_text
    payload = np.load(valid_npz)
    x_valid = np.ascontiguousarray(payload["X"], dtype=np.float32)
    np.savetxt(valid_text, x_valid, delimiter="\t", fmt="%.8g")
    return valid_text


def run_dataset_worker(name: str, dll: Path, input_npz: Path, binary_out: Path, max_bin: int, num_threads: int, sample_cnt: int) -> dict:
    result_out = ARTIFACTS / f"{name}.dataset.json"
    cmd = [
        sys.executable, str(HERE / "run_dataset_worker.py"),
        "--dll", str(dll), "--input-npz", str(input_npz), "--binary-out", str(binary_out),
        "--result-out", str(result_out), "--max-bin", str(max_bin), "--num-threads", str(num_threads),
        "--bin-construct-sample-cnt", str(sample_cnt),
    ]
    run_checked(cmd)
    payload = json.loads(result_out.read_text(encoding="utf-8"))
    payload.update({"name": name, "binary": str(binary_out)})
    return payload


def parse_rdna2_init(stdout: str) -> dict:
    match = re.search(
        r"RDNA2 packed dataset: .*?size=([0-9.]+) MiB .*?pack=([0-9.]+) ms alloc=([0-9.]+) ms H2D=([0-9.]+) ms",
        stdout,
    )
    if not match:
        return {"packed_mib": None, "pack_ms": None, "alloc_ms": None, "h2d_ms": None}
    packed_mib, pack_ms, alloc_ms, h2d_ms = map(float, match.groups())
    return {"packed_mib": packed_mib, "pack_ms": pack_ms, "alloc_ms": alloc_ms, "h2d_ms": h2d_ms}


def run_rdna2_init(exe: Path, train_binary: Path, profile: dict) -> dict:
    env = os.environ.copy()
    rocm_bin = Path(os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\6.2")) / "bin"
    env["PATH"] = str(rocm_bin) + os.pathsep + env.get("PATH", "")
    model = ARTIFACTS / "rdna2_init.model.txt"
    profile_args = [arg for arg in profile_param_string(profile).split() if not arg.startswith("verbosity=")]
    cmd = [
        str(exe), "task=train", f"data={train_binary}", *profile_args,
        "verbosity=1", "device_type=rdna2", "num_gpu=1", "gpu_device_id=0", "num_iterations=1",
        f"output_model={model}",
    ]
    t0 = time.perf_counter()
    proc = run_checked(cmd, env=env)
    wall_seconds = time.perf_counter() - t0
    result = {"probe_wall_seconds_includes_one_iteration": wall_seconds, **parse_rdna2_init(proc.stdout)}
    return result


def cpu_dataset_equivalence(cpu_dll: Path, canonical_binary: Path, candidate_binary: Path, valid_npz: Path, iterations: int, profile: dict, args: argparse.Namespace) -> dict:
    canonical = run_capi_variant("dataset_oracle_train", cpu_dll, "cpu", canonical_binary, valid_npz, iterations, profile)
    candidate = run_capi_variant("dataset_candidate_on_cpu", cpu_dll, "cpu", candidate_binary, valid_npz, iterations, profile)
    y_valid = np.load(valid_npz)["y"]
    original_names = (canonical["name"], candidate["name"])
    canonical["name"] = "v470_cpu"
    candidate["name"] = "v470_rdna2"
    result = compare(
        [canonical, candidate], iterations, y_valid, profile["objective"], args.atol, args.rtol, args.auc_tol,
        args.regression_metric_tol, args.correlation_min, args.classification_threshold,
        args.min_class_fraction, args.min_confident_fraction,
    )
    canonical["name"], candidate["name"] = original_names
    return result


def main() -> int:
    global ARTIFACTS
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["dataset_create", "dataset_to_rdna2", "end_to_train", "end_to_end"], default="dataset_create")
    p.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="h64")
    p.add_argument("--train-rows", type=int, default=40000)
    p.add_argument("--valid-rows", type=int, default=5000)
    p.add_argument("--features", type=int, default=3000)
    p.add_argument("--split-offset", type=int, default=0)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--bin-construct-sample-cnt", type=int, default=200000)
    p.add_argument("--candidate-dll", type=Path, default=DEFAULT_ROCM_DLL)
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
    ARTIFACTS = TEMP_ROOT / "artifacts" / "dataset_pipeline" / args.profile / f"offset_{args.split_offset}"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cpu_dll = BIN / "lightgbm_4.7.0_native_cpu.dll"
    provenance = BIN / "lightgbm_4.7.0_native_cpu.source.txt"
    rocm_exe = BIN / "lightgbm_4.7.0_rocm.exe"
    required = [cpu_dll, provenance, args.candidate_dll]
    if args.stage in {"dataset_to_rdna2", "end_to_train", "end_to_end"}:
        required.append(rocm_exe)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing benchmark binaries:\n" + "\n".join(missing))
    provenance_text = provenance.read_text(encoding="utf-8").strip()
    if NATIVE_470_COMMIT not in provenance_text:
        raise RuntimeError(f"CPU oracle provenance mismatch: {provenance_text}")

    train_npz, valid_npz = ensure_split_npz(args.train_rows, args.valid_rows, args.features, args.split_offset, profile["objective"])
    valid_text = ensure_valid_text(valid_npz) if args.stage == "end_to_end" else None
    canonical_binary = ARTIFACTS / "canonical_v470.bin"
    candidate_binary = ARTIFACTS / "candidate.bin"
    canonical_dataset = run_dataset_worker(
        "canonical_v470", cpu_dll, train_npz, canonical_binary, int(profile["max_bin"]), args.num_threads, args.bin_construct_sample_cnt
    )
    candidate_dataset = run_dataset_worker(
        "candidate", args.candidate_dll, train_npz, candidate_binary, int(profile["max_bin"]), args.num_threads, args.bin_construct_sample_cnt
    )

    equivalence = cpu_dataset_equivalence(
        cpu_dll, canonical_binary, candidate_binary, valid_npz, args.iterations, profile, args
    )
    report = {
        "stage": args.stage, "profile": args.profile, "split_offset": args.split_offset,
        "train_rows": args.train_rows, "valid_rows": args.valid_rows, "features": args.features,
        "canonical_dataset": canonical_dataset, "candidate_dataset": candidate_dataset,
        "dataset_equivalence_on_cpu_oracle": equivalence,
    }

    if args.stage in {"dataset_to_rdna2", "end_to_train", "end_to_end"}:
        report["rdna2_init"] = run_rdna2_init(rocm_exe, candidate_binary, profile)
        report["end_to_train_seconds"] = (
            candidate_dataset["input_conversion_seconds"]
            + candidate_dataset["dataset_create_seconds"]
            + candidate_dataset["set_label_seconds"]
            + sum(
                value for value in (
                    report["rdna2_init"]["pack_ms"],
                    report["rdna2_init"]["alloc_ms"],
                    report["rdna2_init"]["h2d_ms"],
                ) if value is not None
            ) / 1000.0
        )

    if args.stage == "end_to_end":
        cpu_result = run_capi_variant("v470_cpu", cpu_dll, "cpu", canonical_binary, valid_npz, args.iterations, profile)
        rdna2_result = run_rocm_cli("v470_rdna2", rocm_exe, "rdna2", candidate_binary, valid_text, valid_npz, args.iterations, profile)
        y_valid = np.load(valid_npz)["y"]
        end_to_end_comparison = compare(
            [cpu_result, rdna2_result], args.iterations, y_valid, profile["objective"], args.atol, args.rtol,
            args.auc_tol, args.regression_metric_tol, args.correlation_min, args.classification_threshold,
            args.min_class_fraction, args.min_confident_fraction,
        )
        report["end_to_end_results"] = [cpu_result, rdna2_result]
        report["end_to_end_comparison"] = end_to_end_comparison
        report["end_to_end_seconds"] = report["end_to_train_seconds"] + rdna2_result["train_seconds"]

    report_path = ARTIFACTS / f"{args.stage}.summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== DATASET PIPELINE BENCHMARK ===")
    print(f"stage={args.stage} profile={args.profile} offset={args.split_offset}")
    print(f"canonical dataset: {canonical_dataset['dataset_create_seconds']:.4f}s, peak +{canonical_dataset['peak_delta_mib']:.1f} MiB")
    print(f"candidate dataset: {candidate_dataset['dataset_create_seconds']:.4f}s, peak +{candidate_dataset['peak_delta_mib']:.1f} MiB")
    speedup = canonical_dataset["dataset_create_seconds"] / candidate_dataset["dataset_create_seconds"]
    print(f"dataset speedup vs pristine v4.7 CPU: {speedup:.3f}x")
    print(f"dataset semantic gate: {'PASS' if equivalence['all_checks_passed'] else 'FAIL'}")
    if "rdna2_init" in report:
        init = report["rdna2_init"]
        print(f"RDNA2 init components: pack={init['pack_ms']}ms alloc={init['alloc_ms']}ms H2D={init['h2d_ms']}ms (probe wall incl. 1 iter={init['probe_wall_seconds_includes_one_iteration']:.4f}s)")
    if "end_to_train_seconds" in report:
        print(f"end-to-train logical total (no binary serialization): {report['end_to_train_seconds']:.4f}s")
    if "end_to_end_comparison" in report:
        print(f"end-to-end logical total (features -> final train iter): {report['end_to_end_seconds']:.4f}s")
        print(f"end-to-end correctness: {'PASS' if report['end_to_end_comparison']['all_checks_passed'] else 'FAIL'}")
    print(f"Artifacts: {ARTIFACTS}")
    passed = equivalence["all_checks_passed"] and report.get("end_to_end_comparison", {"all_checks_passed": True})["all_checks_passed"]
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
