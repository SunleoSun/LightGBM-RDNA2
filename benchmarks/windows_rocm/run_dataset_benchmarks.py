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
EXTRACTION_MODES = ("pipeline_fancy", "contiguous_slice")


def ensure_phase_npz(train_rows: int, gap_rows: int, valid_rows: int, features: int, split_offset: int, objective: str) -> tuple[Path, Path, int, int]:
    DATA.mkdir(parents=True, exist_ok=True)
    tag = "binary" if objective == "binary" else "regression"
    total = split_offset + train_rows + gap_rows + valid_rows
    stem = f"dataset_phase_{tag}_{total}x{features}_train{train_rows}_gap{gap_rows}_offset{split_offset}"
    phase_npz = DATA / f"{stem}.npz"
    valid_npz = DATA / f"{stem}_valid.npz"
    train_start = split_offset
    train_end = split_offset + train_rows
    valid_start = train_end + gap_rows
    valid_end = valid_start + valid_rows
    if phase_npz.exists() and valid_npz.exists():
        return phase_npz, valid_npz, train_start, train_end

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
    np.savez(phase_npz, X=np.ascontiguousarray(x), y=np.ascontiguousarray(y))
    np.savez(valid_npz, X=np.ascontiguousarray(x[valid_start:valid_end]), y=np.ascontiguousarray(y[valid_start:valid_end]))
    return phase_npz, valid_npz, train_start, train_end


def ensure_valid_text(valid_npz: Path) -> Path:
    valid_text = valid_npz.with_suffix(".features.tsv")
    if valid_text.exists():
        return valid_text
    payload = np.load(valid_npz)
    x_valid = np.ascontiguousarray(payload["X"], dtype=np.float32)
    np.savetxt(valid_text, x_valid, delimiter="\t", fmt="%.8g")
    return valid_text


def run_dataset_worker(name: str, dll: Path, input_npz: Path, binary_out: Path, max_bin: int, num_threads: int, sample_cnt: int, train_start: int, train_end: int, extraction: str, device_type: str = "cpu", gpu_device_id: int = 0) -> dict:
    result_out = ARTIFACTS / f"{name}.dataset.json"
    cmd = [
        sys.executable, str(HERE / "run_dataset_worker.py"),
        "--dll", str(dll), "--input-npz", str(input_npz), "--binary-out", str(binary_out),
        "--result-out", str(result_out), "--max-bin", str(max_bin), "--num-threads", str(num_threads),
        "--bin-construct-sample-cnt", str(sample_cnt), "--device-type", device_type,
        "--gpu-device-id", str(gpu_device_id), "--train-start", str(train_start), "--train-end", str(train_end),
        "--split-extraction", extraction,
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


def run_rdna2_init(exe: Path, train_binary: Path, profile: dict, suffix: str) -> dict:
    env = os.environ.copy()
    rocm_bin = Path(os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\6.2")) / "bin"
    env["PATH"] = str(rocm_bin) + os.pathsep + env.get("PATH", "")
    model = ARTIFACTS / f"rdna2_init_{suffix}.model.txt"
    profile_args = [arg for arg in profile_param_string(profile).split() if not arg.startswith("verbosity=")]
    cmd = [
        str(exe), "task=train", f"data={train_binary}", *profile_args,
        "verbosity=1", "device_type=rdna2", "num_gpu=1", "gpu_device_id=0", "num_iterations=1",
        f"output_model={model}",
    ]
    t0 = time.perf_counter()
    proc = run_checked(cmd, env=env)
    return {"probe_wall_seconds_includes_one_iteration": time.perf_counter() - t0, **parse_rdna2_init(proc.stdout)}


def cpu_dataset_equivalence(cpu_dll: Path, canonical_binary: Path, candidate_binary: Path, valid_npz: Path, iterations: int, profile: dict, args: argparse.Namespace, suffix: str) -> dict:
    canonical = run_capi_variant(f"dataset_oracle_{suffix}", cpu_dll, "cpu", canonical_binary, valid_npz, iterations, profile)
    candidate = run_capi_variant(f"dataset_candidate_{suffix}", cpu_dll, "cpu", candidate_binary, valid_npz, iterations, profile)
    y_valid = np.load(valid_npz)["y"]
    original_names = (canonical["name"], candidate["name"])
    canonical["name"] = "v470_cpu"
    candidate["name"] = "v470_rdna2"
    result = compare(
        [canonical, candidate], iterations, y_valid, profile["objective"], args.atol, args.rtol, args.auc_tol,
        args.regression_metric_tol, args.correlation_min, args.classification_threshold,
        args.min_class_fraction, args.min_confident_fraction, profile.get("alpha"),
    )
    canonical["name"], candidate["name"] = original_names
    return result


def compare_canonical_split_representations(cpu_dll: Path, binaries: dict[str, Path], valid_npz: Path, iterations: int, profile: dict, args: argparse.Namespace) -> dict:
    fancy = run_capi_variant("canonical_pipeline_fancy", cpu_dll, "cpu", binaries["pipeline_fancy"], valid_npz, iterations, profile)
    sliced = run_capi_variant("canonical_contiguous_slice", cpu_dll, "cpu", binaries["contiguous_slice"], valid_npz, iterations, profile)
    y_valid = np.load(valid_npz)["y"]
    original_names = (fancy["name"], sliced["name"])
    fancy["name"] = "v470_cpu"
    sliced["name"] = "v470_rdna2"
    result = compare(
        [fancy, sliced], iterations, y_valid, profile["objective"], args.atol, args.rtol, args.auc_tol,
        args.regression_metric_tol, args.correlation_min, args.classification_threshold,
        args.min_class_fraction, args.min_confident_fraction, profile.get("alpha"),
    )
    fancy["name"], sliced["name"] = original_names
    return result


def main() -> int:
    global ARTIFACTS
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["dataset_create", "dataset_to_rdna2", "end_to_train", "end_to_end"], default="dataset_create")
    p.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="h64")
    p.add_argument("--train-rows", type=int, default=40000)
    p.add_argument("--gap-rows", type=int, default=0)
    p.add_argument("--valid-rows", type=int, default=5000)
    p.add_argument("--features", type=int, default=3000)
    p.add_argument("--split-offset", type=int, default=0)
    p.add_argument("--split-extraction", choices=["pipeline_fancy", "contiguous_slice", "both"], default="both")
    p.add_argument("--phase1-side-count", type=int, default=2, help="Projection only: Phase 1 currently constructs the same fold feature dataset independently for long and short.")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--bin-construct-sample-cnt", type=int, default=200000)
    p.add_argument("--candidate-dll", type=Path, default=DEFAULT_ROCM_DLL)
    p.add_argument("--candidate-dataset-device", choices=["cpu", "rdna2"], default="cpu")
    p.add_argument("--gpu-device-id", type=int, default=0)
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
    ARTIFACTS = TEMP_ROOT / "artifacts" / "dataset_pipeline" / args.profile / f"offset_{args.split_offset}_train_{args.train_rows}"
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

    phase_npz, valid_npz, train_start, train_end = ensure_phase_npz(
        args.train_rows, args.gap_rows, args.valid_rows, args.features, args.split_offset, profile["objective"]
    )
    valid_text = ensure_valid_text(valid_npz) if args.stage == "end_to_end" else None
    modes = list(EXTRACTION_MODES if args.split_extraction == "both" else (args.split_extraction,))
    report = {
        "stage": args.stage, "profile": args.profile, "split_offset": args.split_offset, "gap_rows": args.gap_rows,
        "train_rows": args.train_rows, "valid_rows": args.valid_rows, "features": args.features,
        "train_start": train_start, "train_end": train_end, "extraction_modes": modes, "modes": {},
    }
    canonical_binaries: dict[str, Path] = {}
    all_passed = True

    for extraction in modes:
        canonical_binary = ARTIFACTS / f"canonical_v470_{extraction}.bin"
        candidate_binary = ARTIFACTS / f"candidate_{extraction}.bin"
        canonical_binaries[extraction] = canonical_binary
        canonical_dataset = run_dataset_worker(
            f"canonical_v470_{extraction}", cpu_dll, phase_npz, canonical_binary, int(profile["max_bin"]),
            args.num_threads, args.bin_construct_sample_cnt, train_start, train_end, extraction, "cpu", args.gpu_device_id,
        )
        candidate_dataset = run_dataset_worker(
            f"candidate_{extraction}", args.candidate_dll, phase_npz, candidate_binary, int(profile["max_bin"]),
            args.num_threads, args.bin_construct_sample_cnt, train_start, train_end, extraction,
            args.candidate_dataset_device, args.gpu_device_id,
        )
        equivalence = cpu_dataset_equivalence(
            cpu_dll, canonical_binary, candidate_binary, valid_npz, args.iterations, profile, args, extraction
        )
        mode_report = {
            "canonical_dataset": canonical_dataset, "candidate_dataset": candidate_dataset,
            "dataset_equivalence_on_cpu_oracle": equivalence,
            "dataset_speedup_vs_pristine_v470": canonical_dataset["dataset_create_seconds"] / candidate_dataset["dataset_create_seconds"],
            "phase1_current_two_side_dataset_seconds_projection": candidate_dataset["pipeline_seconds"] * args.phase1_side_count,
        }
        all_passed = all_passed and equivalence["all_checks_passed"]

        if args.stage in {"dataset_to_rdna2", "end_to_train", "end_to_end"}:
            init = run_rdna2_init(rocm_exe, candidate_binary, profile, extraction)
            mode_report["rdna2_init"] = init
            mode_report["end_to_train_seconds"] = (
                candidate_dataset["split_materialization_seconds"]
                + candidate_dataset["model_input_conversion_seconds"]
                + candidate_dataset["dataset_create_seconds"]
                + candidate_dataset["set_label_seconds"]
                + sum(value for value in (init["pack_ms"], init["alloc_ms"], init["h2d_ms"]) if value is not None) / 1000.0
            )

        if args.stage == "end_to_end":
            cpu_result = run_capi_variant(f"v470_cpu_{extraction}", cpu_dll, "cpu", canonical_binary, valid_npz, args.iterations, profile)
            rdna2_result = run_rocm_cli(f"v470_rdna2_{extraction}", rocm_exe, "rdna2", candidate_binary, valid_text, valid_npz, args.iterations, profile)
            y_valid = np.load(valid_npz)["y"]
            original_names = (cpu_result["name"], rdna2_result["name"])
            cpu_result["name"] = "v470_cpu"
            rdna2_result["name"] = "v470_rdna2"
            end_to_end_comparison = compare(
                [cpu_result, rdna2_result], args.iterations, y_valid, profile["objective"], args.atol, args.rtol,
                args.auc_tol, args.regression_metric_tol, args.correlation_min, args.classification_threshold,
                args.min_class_fraction, args.min_confident_fraction, profile.get("alpha"),
            )
            cpu_result["name"], rdna2_result["name"] = original_names
            mode_report["end_to_end_results"] = [cpu_result, rdna2_result]
            mode_report["end_to_end_comparison"] = end_to_end_comparison
            mode_report["end_to_end_seconds"] = mode_report["end_to_train_seconds"] + rdna2_result["train_seconds"]
            all_passed = all_passed and end_to_end_comparison["all_checks_passed"]

        report["modes"][extraction] = mode_report

    if set(modes) == set(EXTRACTION_MODES):
        split_equivalence = compare_canonical_split_representations(
            cpu_dll, canonical_binaries, valid_npz, args.iterations, profile, args
        )
        report["split_representation_equivalence_on_cpu_oracle"] = split_equivalence
        all_passed = all_passed and split_equivalence["all_checks_passed"]
        fancy = report["modes"]["pipeline_fancy"]["candidate_dataset"]
        sliced = report["modes"]["contiguous_slice"]["candidate_dataset"]
        report["split_extraction_comparison"] = {
            "fancy_seconds": fancy["split_materialization_seconds"],
            "slice_seconds": sliced["split_materialization_seconds"],
            "fancy_working_set_delta_mib": fancy["split_working_set_delta_mib"],
            "slice_working_set_delta_mib": sliced["split_working_set_delta_mib"],
            "fancy_shares_phase_memory": fancy["split_shares_phase_memory"],
            "slice_shares_phase_memory": sliced["split_shares_phase_memory"],
        }

    report_path = ARTIFACTS / f"{args.stage}.summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== DATASET PIPELINE BENCHMARK ===")
    print(f"stage={args.stage} profile={args.profile} train=[{train_start},{train_end}) gap={args.gap_rows}")
    for extraction in modes:
        mode = report["modes"][extraction]
        canonical = mode["canonical_dataset"]
        candidate = mode["candidate_dataset"]
        print(
            f"{extraction}: split={candidate['split_materialization_seconds']:.4f}s "
            f"(+{candidate['split_working_set_delta_mib']:.1f} MiB, shares={candidate['split_shares_phase_memory']}) "
            f"dataset={candidate['dataset_create_seconds']:.4f}s vs v4.7={canonical['dataset_create_seconds']:.4f}s "
            f"peak=+{candidate['peak_delta_mib']:.1f} MiB gate={'PASS' if mode['dataset_equivalence_on_cpu_oracle']['all_checks_passed'] else 'FAIL'}"
        )
        if "end_to_train_seconds" in mode:
            init = mode["rdna2_init"]
            print(
                f"  end_to_train={mode['end_to_train_seconds']:.4f}s; RDNA2 pack={init['pack_ms']}ms "
                f"alloc={init['alloc_ms']}ms H2D={init['h2d_ms']}ms"
            )
        if "end_to_end_comparison" in mode:
            print(
                f"  end_to_end={mode['end_to_end_seconds']:.4f}s correctness="
                f"{'PASS' if mode['end_to_end_comparison']['all_checks_passed'] else 'FAIL'}"
            )
    if "split_representation_equivalence_on_cpu_oracle" in report:
        print(
            "fancy vs contiguous-slice canonical semantic gate: "
            + ("PASS" if report["split_representation_equivalence_on_cpu_oracle"]["all_checks_passed"] else "FAIL")
        )
    print(f"Artifacts: {ARTIFACTS}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
