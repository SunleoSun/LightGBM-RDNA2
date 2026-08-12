from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "production_2026-08-12.json"
DEFAULT_OUTPUT = HERE / "images"

BACKENDS = [
    ("cpu_v470", "CPU v4.7"),
    ("opencl_legacy", "OpenCL legacy"),
    ("rdna2", "RDNA2 / ROCm"),
]
PROFILES = [("h64", "H64 / max_bin 63"), ("h128", "H128 / max_bin 127")]


def grouped_bar(payload: dict, metric: str, ylabel: str, title: str, output: Path) -> None:
    production = payload["production"]
    x = np.arange(len(PROFILES), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)

    for backend_index, (backend_key, backend_label) in enumerate(BACKENDS):
        values = [float(production[profile_key][backend_key][metric]) for profile_key, _ in PROFILES]
        positions = x + (backend_index - 1) * width
        bars = ax.bar(positions, values, width, label=backend_label)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x, [label for _, label in PROFILES])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    grouped_bar(
        payload,
        "end_to_end_seconds",
        "End-to-end training wall time (s)",
        "LightGBM 40k x 3000, 100 trees: end-to-end time",
        args.output_dir / "production_end_to_end.png",
    )
    grouped_bar(
        payload,
        "training_peak_working_set_mib",
        "Peak host process working set (MiB)",
        "LightGBM 40k x 3000, 100 trees: peak host memory",
        args.output_dir / "production_peak_host_memory.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
