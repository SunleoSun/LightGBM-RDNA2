from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_benchmarks import (
    FEATURE_FRACTION_PROFILES,
    OPTUNA_COMPAT_PROFILES,
    OPTUNA_LONG_ITERATIONS,
    OPTUNA_PROFILES,
    PROFILE_CONFIGS,
    QUANTILE_PROFILES,
    SMOKE_PROFILES,
    STRESS_PROFILES,
)

HERE = Path(__file__).resolve().parent
TEMP_ROOT = Path(os.environ.get("LIGHTGBM_RDNA2_TEMP", r"C:\Temp\LightGBM-RDNA2\benches"))


def run_profile(profile: str, iterations: int, train_rows: int, valid_rows: int, features: int, modes: str) -> dict:
    cmd = [
        sys.executable, str(HERE / "run_benchmarks.py"),
        "--profile", profile,
        "--iterations", str(iterations),
        "--train-rows", str(train_rows),
        "--valid-rows", str(valid_rows),
        "--features", str(features),
        "--modes", modes,
    ]
    print("+", subprocess.list2cmdline(cmd), flush=True)
    proc = subprocess.run(cmd, text=True)
    summary_path = TEMP_ROOT / "artifacts" / profile / "summary.json"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "profile": profile,
        "iterations": iterations,
        "exit_code": proc.returncode,
        "summary": summary,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite",
        choices=["smoke", "production", "stress", "quantile", "feature_fraction", "optuna", "optuna_long", "optuna_compat"],
        default="smoke",
    )
    p.add_argument("--train-rows", type=int, default=40000)
    p.add_argument("--valid-rows", type=int)
    p.add_argument("--features", type=int, default=3000)
    p.add_argument("--iterations", type=int)
    args = p.parse_args()

    if args.suite == "production":
        profiles = ["h64", "h128"]
        default_iterations = 100
        modes = "all"
    elif args.suite == "stress":
        profiles = STRESS_PROFILES
        default_iterations = 20
        modes = "all"
    elif args.suite == "quantile":
        profiles = QUANTILE_PROFILES
        default_iterations = 20
        modes = "v470"
    elif args.suite == "feature_fraction":
        profiles = FEATURE_FRACTION_PROFILES
        default_iterations = 20
        modes = "v470"
    elif args.suite == "optuna":
        profiles = OPTUNA_PROFILES
        default_iterations = 20
        modes = "v470"
    elif args.suite == "optuna_long":
        profiles = list(OPTUNA_LONG_ITERATIONS)
        default_iterations = None
        modes = "v470"
    elif args.suite == "optuna_compat":
        profiles = OPTUNA_COMPAT_PROFILES
        default_iterations = 20
        modes = "all"
    else:
        profiles = SMOKE_PROFILES
        default_iterations = 6
        modes = "v470"

    iterations = args.iterations if args.iterations is not None else default_iterations
    valid_rows = (
        args.valid_rows
        if args.valid_rows is not None
        else (5000 if args.suite in {"smoke", "quantile", "feature_fraction", "optuna", "optuna_compat"} else 50000)
    )
    unknown = [profile for profile in profiles if profile not in PROFILE_CONFIGS]
    if unknown:
        raise RuntimeError(f"unknown benchmark profiles: {unknown}")

    results = []
    for profile in profiles:
        if args.iterations is not None:
            profile_iterations = args.iterations
        elif args.suite == "optuna_long":
            profile_iterations = OPTUNA_LONG_ITERATIONS[profile]
        else:
            profile_iterations = iterations
        print(f"\n=== {args.suite.upper()} PROFILE {profile} ({profile_iterations} trees) ===", flush=True)
        results.append(run_profile(profile, profile_iterations, args.train_rows, valid_rows, args.features, modes))

    matrix = {
        "suite": args.suite,
        "iterations": (
            {profile: OPTUNA_LONG_ITERATIONS[profile] for profile in profiles}
            if args.suite == "optuna_long" and args.iterations is None
            else iterations
        ),
        "valid_rows": valid_rows,
        "modes": modes,
        "profiles": profiles,
        "all_checks_passed": all(
            item["exit_code"] == 0
            and item["summary"] is not None
            and item["summary"]["comparison"]["all_checks_passed"]
            for item in results
        ),
        "results": [
            {
                "profile": item["profile"],
                "iterations": item["iterations"],
                "exit_code": item["exit_code"],
                "correctness": (
                    item["summary"]["comparison"]["all_checks_passed"]
                    if item["summary"] is not None else False
                ),
                "config": item["summary"]["config"] if item["summary"] is not None else None,
                "cpu": next(
                    (r for r in item["summary"]["results"] if r["name"] == "v470_cpu"), None
                ) if item["summary"] is not None else None,
                "rdna2": next(
                    (r for r in item["summary"]["results"] if r["name"] == "v470_rdna2"), None
                ) if item["summary"] is not None else None,
                "rdna2_comparison": next(
                    (r for r in item["summary"]["comparison"]["comparisons"] if r["name"] == "v470_rdna2"), None
                ) if item["summary"] is not None else None,
                "opencl": next(
                    (r for r in item["summary"]["results"] if r["name"] == "old_opencl_gpu"), None
                ) if item["summary"] is not None else None,
                "opencl_comparison": next(
                    (r for r in item["summary"]["comparison"]["comparisons"] if r["name"] == "old_opencl_gpu"), None
                ) if item["summary"] is not None else None,
            }
            for item in results
        ],
    }
    output = TEMP_ROOT / "artifacts" / f"matrix_{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    print("\n=== MATRIX SUMMARY ===")
    for item in matrix["results"]:
        rdna2 = item["rdna2"] or {}
        cmp = item["rdna2_comparison"] or {}
        metric = (
            rdna2.get("auc") if rdna2.get("auc") is not None
            else rdna2.get("quantile_loss") if rdna2.get("quantile_loss") is not None
            else rdna2.get("rmse")
        )
        opencl_cmp = item.get("opencl_comparison") or {}
        opencl_suffix = (
            f" opencl_diff={opencl_cmp.get('prediction_max_abs_diff', float('nan')):.3g}"
            f" opencl_struct={opencl_cmp.get('tree_structure_match')}"
            if opencl_cmp else ""
        )
        print(
            f"{item['profile']:<20} correctness={str(item['correctness']):<5} "
            f"iter_ms={rdna2.get('iteration_ms', float('nan')):8.3f} "
            f"metric={metric if metric is not None else float('nan'):.8f} "
            f"pred_diff={cmp.get('prediction_max_abs_diff', float('nan')):.3g} "
            f"struct={cmp.get('tree_structure_match')}"
            f"{opencl_suffix}"
        )
    print(f"Matrix artifacts: {output}")
    print(f"Correctness: {'PASS' if matrix['all_checks_passed'] else 'FAIL'}")
    return 0 if matrix["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
