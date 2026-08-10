from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_benchmarks import PROFILE_CONFIGS, SMOKE_PROFILES, STRESS_PROFILES

HERE = Path(__file__).resolve().parent


def run_profile(profile: str, iterations: int, train_rows: int, valid_rows: int, features: int) -> dict:
    cmd = [
        sys.executable, str(HERE / "run_benchmarks.py"),
        "--profile", profile,
        "--iterations", str(iterations),
        "--train-rows", str(train_rows),
        "--valid-rows", str(valid_rows),
        "--features", str(features),
    ]
    print("+", subprocess.list2cmdline(cmd), flush=True)
    proc = subprocess.run(cmd, text=True)
    summary_path = HERE / "artifacts" / profile / "summary.json"
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
    p.add_argument("--suite", choices=["smoke", "production", "stress"], default="smoke")
    p.add_argument("--train-rows", type=int, default=40000)
    p.add_argument("--valid-rows", type=int, default=50000)
    p.add_argument("--features", type=int, default=3000)
    p.add_argument("--iterations", type=int)
    args = p.parse_args()

    if args.suite == "production":
        profiles = ["h64", "h128"]
        default_iterations = 100
    elif args.suite == "stress":
        profiles = STRESS_PROFILES
        default_iterations = 20
    else:
        profiles = SMOKE_PROFILES
        default_iterations = 20

    iterations = args.iterations if args.iterations is not None else default_iterations
    unknown = [profile for profile in profiles if profile not in PROFILE_CONFIGS]
    if unknown:
        raise RuntimeError(f"unknown benchmark profiles: {unknown}")

    results = []
    for profile in profiles:
        print(f"\n=== {args.suite.upper()} PROFILE {profile} ({iterations} trees) ===", flush=True)
        results.append(run_profile(profile, iterations, args.train_rows, args.valid_rows, args.features))

    matrix = {
        "suite": args.suite,
        "iterations": iterations,
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
                "exit_code": item["exit_code"],
                "correctness": (
                    item["summary"]["comparison"]["all_checks_passed"]
                    if item["summary"] is not None else False
                ),
                "config": item["summary"]["config"] if item["summary"] is not None else None,
                "rocm": next(
                    (r for r in item["summary"]["results"] if r["name"] == "v470_rocm_gpu"), None
                ) if item["summary"] is not None else None,
                "rocm_comparison": next(
                    (r for r in item["summary"]["comparison"]["comparisons"] if r["name"] == "v470_rocm_gpu"), None
                ) if item["summary"] is not None else None,
            }
            for item in results
        ],
    }
    output = HERE / "artifacts" / f"matrix_{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    print("\n=== MATRIX SUMMARY ===")
    for item in matrix["results"]:
        rocm = item["rocm"] or {}
        cmp = item["rocm_comparison"] or {}
        metric = rocm.get("auc") if rocm.get("auc") is not None else rocm.get("rmse")
        print(
            f"{item['profile']:<20} correctness={str(item['correctness']):<5} "
            f"iter_ms={rocm.get('iteration_ms', float('nan')):8.3f} "
            f"metric={metric if metric is not None else float('nan'):.8f} "
            f"pred_diff={cmp.get('prediction_max_abs_diff', float('nan')):.3g} "
            f"struct={cmp.get('tree_structure_match')}"
        )
    print(f"Matrix artifacts: {output}")
    print(f"Correctness: {'PASS' if matrix['all_checks_passed'] else 'FAIL'}")
    return 0 if matrix["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
